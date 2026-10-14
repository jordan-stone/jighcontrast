import numpy as np
import pytest

from jighcontrast import jFM, jPCA, jPhot, nan_rot


def test_nan_rot_pivot_matches_get_unrot_coords():
    from scipy.ndimage import map_coordinates
    rng = np.random.default_rng(0)
    img = rng.normal(size=(48, 48))
    center = (23.5, 23.5)
    yy, xx = np.indices(img.shape).astype(float)
    for ang in (12.0, -57.0):
        a = nan_rot.nan_rot(img, ang, center, out_size=img.shape, pivot=True)
        ux, uy = jPhot.get_unrot_coords(xx, yy, ang, center)
        b = map_coordinates(img, [uy, ux], order=3, cval=np.nan)
        ok = np.isfinite(a) & np.isfinite(b)
        assert ok.sum() > 1000 and np.allclose(a[ok], b[ok])


def test_inject_source_lands_at_sky_position():
    from scipy.ndimage import center_of_mass
    ny = nx = 41
    c = (20.0, 20.0)
    stamp = np.exp(-np.hypot(*np.indices((15, 15)) - 7.0) ** 2 / 4)[None]
    pa = np.array([0.0, 40.0])
    cube = np.zeros((2, 1, ny, nx))
    out = jFM.inject_source(cube, stamp, pa, c, 10.0, 90.0, [2.0])
    # PA 90 (east, -x) at parallactic angle 0 -> 10 px to the left of the star
    cy, cx = center_of_mass(out[0, 0])
    assert np.hypot(cx - 10.0, cy - 20.0) < 0.05
    assert np.isclose(out[0, 0].sum(), 2 * stamp.sum(), rtol=1e-3)


def test_adi_sdi_displacement():
    d = jPCA.adi_sdi_displacement(20.0, q=[0, 10], wv=[3.0, 3.0], ref_q=0.0, ref_wv=3.0)
    assert np.isclose(d[0], 0) and np.isclose(d[1], 20 * np.deg2rad(10))
    d = jPCA.adi_sdi_displacement(20.0, q=0, wv=4.0, ref_q=0.0, ref_wv=3.0)
    assert np.isclose(d, 20 * (3 / 4 - 1) * -1)


def test_pca_centering_modes():
    rng = np.random.default_rng(2)
    lib = rng.normal(size=(30, 10, 10)) + 5.0
    mask = np.ones((10, 10), bool)
    for cen in ('median', 'frame'):
        p = jPCA.PCA(lib, mask, method='svd', centering=cen)
        U, s = p.left_vectors, p.singular_values
        assert np.allclose(p.library, (U * s) @ p.eigenvectors.T)
    p = jPCA.PCA(lib, mask, centering='frame')
    assert np.allclose(p.library.mean(axis=1), 0)
    with pytest.raises(ValueError):
        jPCA.PCA(lib, mask, centering='bogus')


def test_svd_and_eigh_agree():
    rng = np.random.default_rng(4)
    lib = rng.normal(size=(20, 12, 12))
    mask = np.ones((12, 12), bool)
    a = jPCA.PCA(lib, mask, method='svd')
    b = jPCA.PCA(lib, mask, method='eigh')
    k = 8
    assert np.allclose(a.eigenvalues[:k], b.eigenvalues[:k])
    assert np.allclose(np.abs(np.sum(a.eigenvectors[:, :k] * b.eigenvectors[:, :k], 0)), 1)


@pytest.mark.parametrize("n, out", [(108, None), (108, (108, 108)), (45, (45, 45)),
                                    (45, None), (45, (60, 60)), (44, (61, 61))])
@pytest.mark.parametrize("axis_frac", [(0.5, 0.5), (0.37, 0.28)])
def test_nan_rot_recenters_axis_on_pixel_center(n, out, axis_frac):
    """pivot=False must put the axis on the output center (N_out - 1)/2 in the
    map_coordinates pixel-center convention -- for even AND odd sizes."""
    from scipy.ndimage import center_of_mass
    axis = (axis_frac[0] * (n - 1), axis_frac[1] * (n - 1))
    yy, xx = np.indices((n, n))
    im = np.exp(-((xx - axis[0]) ** 2 + (yy - axis[1]) ** 2) / 4.0)
    rot = np.nan_to_num(nan_rot.nan_rot(im, 30.0, axis, out_size=out))
    cy, cx = center_of_mass(rot)
    oy, ox = rot.shape
    assert abs(cx - (ox - 1) / 2) < 0.02 and abs(cy - (oy - 1) / 2) < 0.02


def test_nan_rot_pivot_keeps_axis_in_place():
    from scipy.ndimage import center_of_mass
    n, axis = 50, (31.3, 17.8)
    yy, xx = np.indices((n, n))
    im = np.exp(-((xx - axis[0]) ** 2 + (yy - axis[1]) ** 2) / 4.0)
    rot = np.nan_to_num(nan_rot.nan_rot(im, -48.0, axis, out_size=(n, n), pivot=True))
    cy, cx = center_of_mass(rot)
    assert abs(cx - axis[0]) < 0.02 and abs(cy - axis[1]) < 0.02


def test_measure_and_register_star_centers():
    ny = nx = 40
    true = np.array([[20.3, 19.6], [21.1, 20.4], [19.2, 18.9]])
    yy, xx = np.indices((ny, nx))
    cube = np.array([[np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.6 ** 2)) for x, y in true]] * 2)
    got = jPhot.measure_star_centers(cube, (20, 20))
    assert np.allclose(got, true, atol=0.02)
    reg = jPhot.register_channels(cube, got, (19.5, 19.5))
    assert np.allclose(jPhot.measure_star_centers(reg, (20, 20)), 19.5, atol=0.03)


@pytest.mark.parametrize("center", [(20.0, 20.0), (20.5, 19.5), (19.3, 20.7)])
def test_offaxis_psf_stamps_centered(center):
    """The star must land on the stamp's central pixel even for fractional
    (e.g. half-integer, after registration) star positions."""
    from scipy.ndimage import center_of_mass
    yy, xx = np.indices((41, 41))
    psf = np.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2 * 1.7 ** 2))
    stamps = jFM.offaxis_psf_stamps(psf[None, None], center, half=7)
    cy, cx = center_of_mass(stamps[0])
    assert abs(cx - 7) < 0.02 and abs(cy - 7) < 0.02


def test_angle_differences_wrap_through_180():
    """Parallactic angles wrap through +/-180 for targets transiting near zenith;
    179 and -179 deg are 2 deg apart, not 358."""
    assert np.isclose(jPCA.wrap_degrees(179.0 - (-179.0)), -2.0)
    d = jPCA.adi_sdi_displacement(20.0, q=[-179.0, 170.0], wv=[3.0, 3.0], ref_q=179.0, ref_wv=3.0)
    assert np.allclose(d, 20 * np.deg2rad([2.0, 9.0]))


def test_make_adi_library_wraps_and_leaves_input_alone():
    frames = [np.full((3, 3), i, float) for i in range(5)]
    pa = np.array([-1.0, 0.5, 359.0, 30.0, -40.0])
    pa_copy = pa.copy()
    good, qs = jPCA.make_adi_library(frames, pa, rad_in_pix=20.0, ref_q=0.0, buffer_radius=4)
    # -1, 0.5 and 359 deg are within 1 deg of the reference (0.35 px at 20 px): rejected
    assert [int(f[0, 0]) for f in good] == [3, 4]
    assert np.allclose(qs, [30.0, 320.0])
    assert np.array_equal(pa, pa_copy)


def test_eigh_hint_silent_in_worker_processes(monkeypatch, capsys):
    import multiprocessing
    monkeypatch.setattr(jPCA, "_EIGH_HINT_SHOWN", False)
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: object())
    jPCA._maybe_suggest_eigh(40, 800)
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: None)
    jPCA._maybe_suggest_eigh(40, 800)
    assert "strongly rectangular" in capsys.readouterr().out
