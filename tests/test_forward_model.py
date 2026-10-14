"""Core KLIP-FM engine: the Pueyo (2016) Appendix E perturbation and the
single-frame forward model, checked against brute-force KLIP."""
import numpy as np
import pytest

from jighcontrast import jPCA, jFM, nan_rot

NY = NX = 41
CENTER = ((NX - 1) / 2.0, (NY - 1) / 2.0)


def gaussian_psf(sigma=1.6, half=7):
    yy, xx = np.indices((2 * half + 1, 2 * half + 1))
    return np.exp(-((xx - half) ** 2 + (yy - half) ** 2) / (2 * sigma ** 2))


PSF = gaussian_psf()


def annulus_mask(r_in, r_out):
    yy, xx = np.indices((NY, NX))
    rr = np.hypot(xx - CENTER[0], yy - CENTER[1])
    return (rr >= r_in) & (rr <= r_out)


def make_library(n=30, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((NY, NX))
    basis = np.array([np.cos(fx * xx + fy * yy + ph) for fx, fy, ph in
                      rng.uniform([0.1, 0.1, 0], [0.5, 0.5, 6], (6, 3))])
    frames = np.einsum('nk,kyx->nyx', rng.normal(size=(n, 6)), basis)
    return frames + 0.02 * rng.normal(size=frames.shape)


def test_zero_source_gives_zero_perturbation():
    frames = make_library()
    mask = annulus_mask(6, 13)
    pca = jPCA.PCA(frames, mask, method='svd')
    A0 = np.zeros((frames.shape[0], int(mask.sum())))
    dZ = jFM.perturbed_kl_modes(pca, A0, np.ones(frames.shape[0]), npca=5)
    assert np.allclose(dZ, 0)


def test_perturbation_converges_linearly():
    """Eq. 51 must match a finite difference of the KL images with an error that
    falls linearly with the source flux (the defining first-order property)."""
    rng = np.random.default_rng(3)
    n, npix = 24, 60
    S = rng.normal(size=(n, npix))
    base = np.zeros(npix)
    base[20:28] = np.hanning(8)
    A = rng.uniform(0.3, 1.0, n)[:, None] * base[None, :]

    class FakePCA:
        pass

    def svd_modes(M):
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        return U, s, Vt.T

    U, s, Z = svd_modes(S)
    p = FakePCA()
    p.left_vectors, p.singular_values, p.eigenvalues, p.eigenvectors = U, s, s ** 2, Z
    errs = []
    for eps in [1e-1, 1e-2, 1e-3]:
        dZ = jFM.perturbed_kl_modes(p, eps * A, np.ones(n), npca=6)
        _, _, Z2 = svd_modes(S + eps * A)
        num = den = 0.0
        for k in range(6):
            sgn = np.sign(Z2[:, k] @ Z[:, k]) or 1.0
            fd = sgn * Z2[:, k] - Z[:, k]
            num += np.linalg.norm(dZ[:, k] - fd) ** 2
            den += np.linalg.norm(fd) ** 2
        errs.append((num / den) ** 0.5)
    assert errs[1] < 0.25 * errs[0] and errs[2] < 0.25 * errs[1]
    assert errs[-1] < 2e-3


def _brute_vs_fm(centering, eps=0.01):
    frames = make_library(n=70)
    mask = annulus_mask(8, 15)
    n = frames.shape[0]
    qs = np.linspace(-70, 70, n)
    src = np.array([eps * jFM.place_psf((NY, NX), PSF,
                                        *jFM.companion_frame_position(11.0, 40.0, -q, CENTER))
                    for q in qs])
    refs, refs_src = frames[1:], src[1:]
    pca = jPCA.PCA(refs + refs_src, mask, method='svd', centering=centering)
    fm = jFM.forward_model_psf(pca, refs_src[:, mask], np.ones(n - 1), src[0][mask],
                               (frames[0] + src[0])[mask], 5)

    def klip(lib, tgt):
        return jPCA.PCA(lib, mask, method='svd', centering=centering)(tgt.copy(), 5)
    brute = klip(refs + refs_src, frames[0] + src[0]) - klip(refs, frames[0])
    return fm, brute, refs, refs_src, mask, pca


def test_forward_model_exact_with_frame_centering():
    """With Pueyo's per-frame centering the forward model has no neglected term:
    it must equal brute-force inject-and-KLIP to first order."""
    fm, brute, *_ = _brute_vs_fm('frame')
    assert np.linalg.norm(brute - fm) / np.linalg.norm(brute) < 2e-3


def test_forward_model_median_centering_differs_only_by_mean_shift():
    """With median centering the only discrepancy is the (documented, neglected)
    shift of the library median caused by the source."""
    fm, brute, refs, refs_src, mask, pca = _brute_vs_fm('median')
    dmed = np.median((refs + refs_src)[:, mask], axis=0) - np.median(refs[:, mask], axis=0)
    Z = pca.eigenvectors[:, :5]
    median_term = -(dmed - Z @ (Z.T @ dmed))
    r = brute - fm
    coef = (r @ median_term) / (median_term @ median_term)
    assert np.corrcoef(r, median_term)[0, 1] > 0.99
    assert np.linalg.norm(r - coef * median_term) / np.linalg.norm(brute) < 0.1


def test_delta_lambda_scales_linearly():
    frames = make_library()
    mask = annulus_mask(6, 13)
    pca = jPCA.PCA(frames, mask, method='svd')
    A = np.abs(np.random.default_rng(1).normal(size=(frames.shape[0], int(mask.sum()))))
    _, r1 = jFM.perturbed_kl_modes(pca, A, np.ones(frames.shape[0]), 5, return_delta_lambda=True)
    _, r2 = jFM.perturbed_kl_modes(pca, 3 * A, np.ones(frames.shape[0]), 5, return_delta_lambda=True)
    assert np.allclose(r2, 3 * r1)


@pytest.mark.parametrize("q", [0.0, 33.0, -71.0])
def test_geometry_closed_loop_subpixel(q):
    """A companion placed in a raw frame lands on its sky position after
    derotation about the star (pivot=True), to sub-pixel accuracy."""
    from scipy.ndimage import center_of_mass
    sep, pa = 10.0, 217.0
    raw = jFM.place_psf((NY, NX), PSF, *jFM.companion_frame_position(sep, pa, -q, CENTER))
    derot = np.nan_to_num(nan_rot.nan_rot(raw, -q, CENTER, out_size=(NY, NX), pivot=True))
    cy, cx = center_of_mass(derot)
    sky = (CENTER[0] - sep * np.sin(np.deg2rad(pa)), CENTER[1] + sep * np.cos(np.deg2rad(pa)))
    assert np.hypot(cx - sky[0], cy - sky[1]) < 0.02


def test_mixing_flags_near_degenerate_truncation():
    """A source that couples modes across the npca cut is flagged in proportion
    to 1/(eigenvalue gap), even when delta-Lambda/Lambda is tiny."""
    rng = np.random.default_rng(7)
    n, npix = 30, 80

    class P:
        pass

    U = np.linalg.qr(rng.normal(size=(n, n)))[0][:, :10]
    Z = np.linalg.qr(rng.normal(size=(npix, npix)))[0][:, :10]
    A = rng.normal(size=(n, npix)) * 1e-3

    def mixing(lam):
        p = P()
        p.left_vectors, p.eigenvectors = U, Z
        p.eigenvalues, p.singular_values = lam, np.sqrt(lam)
        _, rel, mix = jFM.perturbed_kl_modes(p, A, np.ones(n), 5, return_delta_lambda=True,
                                             return_mixing=True)
        return np.max(np.abs(rel)), mix

    lam_gap = np.array([10, 9, 8, 7, 6, 3, 2, 1.5, 1.2, 1.0])
    lam_deg = lam_gap.copy()
    lam_deg[5] = 6 - 1e-4                       # nearly degenerate with mode 4 (kept)
    r1, m1 = mixing(lam_gap)
    r2, m2 = mixing(lam_deg)
    assert r1 < 1e-2 and r2 < 1e-2
    assert m2 > 100 * m1
