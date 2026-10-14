'''KLIP forward modeling (Pueyo 2016, ApJ 824, 117).

This module implements the first-order matrix-perturbation forward model of
Appendix E of Pueyo (2016) and applies it to spectral extraction / detection
limits (Appendix F). The physical idea: a faint off-axis source that leaks into
the KLIP reference library perturbs the Karhunen-Loeve (KL) basis by a small
amount. To first order that perturbation is *linear* in the source's
brightness/spectrum, so the way KLIP attenuates the source (over-subtraction +
self-subtraction) can be computed analytically -- once, per geometry -- instead
of re-running KLIP for every injected fake source.

Main entry points:

* ``extract_spectrum``  -- a companion's spectrum (Pueyo Eq. SimpleMat) with an
  empirically measured, wavelength-correlated covariance; refines bright sources
  by negative injection.
* ``contrast_curve``    -- forward-modeled throughput + matched-filter noise per
  channel, at a fixed false-alarm probability; combine channels with
  ``combine_contrast`` (independent) or ``combine_contrast_cov`` (correlated).
* ``klip_fm_reduce``    -- the reduction + forward model behind the extraction.
* ``perturbed_kl_modes`` / ``forward_model_psf`` -- the single-frame engine
  (Pueyo Eqs. 51 and the App. F processed-source model).
* ``inject_source``, ``offaxis_psf_stamps``, ``load_model_spectrum`` -- helpers.

Conventions: pixel coordinates follow ``scipy.ndimage.map_coordinates`` (integer
values are pixel centers); position angles are degrees east of north (CCW from
+y, east toward -x); exposure t is derotated by ``derot_sign * pa[t]`` (default
-PA) with ``nan_rot(..., pivot=True)``; spectra are companion/star flux ratios.

Notation (matching the companion study of the paper's appendices):

* ``S``            signal-free reference frames, ``(n_refs, n_pixels)`` -- the
                   centered image library (``jPCA.PCA.library``).
* ``V_k``          frame-space KL eigenvectors, ``(n_refs,)`` -- left singular
                   vectors of ``S`` (``jPCA.PCA.left_vectors``).
* ``Lambda_k``     KL eigenvalues (``jPCA.PCA.eigenvalues``); ``sqrt(Lambda_k)``
                   are the singular values (``jPCA.PCA.singular_values``).
* ``Z_k``          pixel-space KL images, ``(n_pixels,)`` -- right singular
                   vectors of ``S`` (``jPCA.PCA.eigenvectors``).
* ``A_delta``      source-image matrix, ``(n_refs, n_pixels)`` -- row m is the
                   off-axis instrument PSF at the source's apparent position in
                   reference frame m.
* ``a`` / ``f``    the source's normalized spectrum ``a`` (shape only) and the
                   physical spectrum ``f = epsilon * a``; per reference frame the
                   relevant scalar is the spectrum value at that frame's
                   wavelength (Pueyo's ``Sel_lambda`` routing, here passed in as
                   ``a_frames``).
'''

import warnings

import numpy as np

from .jPCA import PCA, resolution_element_pix, adi_sdi_displacement
from .jPhot import get_unrot_coords, get_noise
from .nan_rot import nan_rot


def companion_frame_position(sep_pix, pa_deg, derot_angle_deg, center):
    '''Pixel position of a fixed-sky companion in a raw (un-derotated) exposure.

    The reduction derotates each exposure with ``nan_rot(resid, derot_angle_deg,
    center, pivot=True)`` so that a sky-fixed companion lands at a common place; for the
    SCALES ADI example that angle is ``-PA`` (see the example reduction). Given
    the companion's sky position -- separation ``sep_pix`` and position angle
    ``pa_deg`` (east of north: CCW from +y, with east toward -x) -- this returns
    where it sits in the raw frame, i.e. the inverse of that derotation, using
    ``jPhot.get_unrot_coords``.

    sep_pix : float
        Companion separation, pixels.
    pa_deg : float
        Companion position angle, degrees east of north.
    derot_angle_deg : float
        The angle passed to ``nan_rot`` to derotate this exposure (e.g. ``-PA``).
    center : (x, y)
        Star pixel position.

    Returns the (x, y) pixel coordinate of the companion in the raw frame.'''
    pa = np.deg2rad(pa_deg)
    sky_x = center[0] - sep_pix * np.sin(pa)   # east (+PA from north) is -x
    sky_y = center[1] + sep_pix * np.cos(pa)   # north (PA=0) is +y
    return get_unrot_coords(sky_x, sky_y, derot_angle_deg, center)


def place_psf(shape, psf_stamp, x, y):
    '''Drop a centered PSF stamp into a zero image at sub-pixel position (x, y).

    shape : (ydim, xdim)
        Output image shape.
    psf_stamp : ndarray (h, w)
        PSF stamp whose peak is on its central pixel ((h-1)/2, (w-1)/2).
    x, y : float
        Sub-pixel pixel coordinates for the stamp center in the output image.

    Returns a ``shape`` image, zero except for the placed (bilinearly
    interpolated) PSF. Out-of-frame parts are simply clipped.'''
    from scipy.ndimage import shift as ndshift
    out = np.zeros(shape, dtype=float)
    h, w = psf_stamp.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    x_int, y_int = int(np.floor(x)), int(np.floor(y))
    dx, dy = x - x_int, y - y_int
    # sub-pixel-shift the stamp, then paste at the integer location
    stamp = ndshift(psf_stamp, (dy, dx), order=1, mode='constant', cval=0.0)
    y0, x0 = y_int - int(round(cy)), x_int - int(round(cx))
    ys0, xs0 = max(0, -y0), max(0, -x0)
    ye, xe = min(h, shape[0] - y0), min(w, shape[1] - x0)
    if ye <= ys0 or xe <= xs0:
        return out
    out[y0 + ys0:y0 + ye, x0 + xs0:x0 + xe] = stamp[ys0:ye, xs0:xe]
    return out


def build_source_model(ref_meta, psf_of_wave, sep_pix, pa_deg, opt_mask, center,
                       spectrum_of_wave, derot_sign=-1.0):
    '''Build the source-image matrix ``A_delta`` for a set of reference frames.

    For each reference frame (with its parallactic angle and wavelength), place
    the wavelength-appropriate off-axis instrument PSF at the companion's
    apparent position in that raw frame, scale it by the (normalized) spectrum at
    that wavelength, and read out the optimization-region pixels. This is the
    ASDI-aware realization of Pueyo's ``A_delta``: parallactic angle sets the
    azimuth of each row and wavelength sets both the PSF (chromatic width) and the
    per-frame brightness ``a`` (the ``Sel_lambda`` routing of the spectrum).

    ref_meta : sequence of (q_deg, wavelength_um)
        Parallactic angle and wavelength for each reference frame, in order.
    psf_of_wave : callable wavelength_um -> ndarray (h, w)
        Returns the centered off-axis PSF stamp at a given wavelength.
    sep_pix, pa_deg : float
        Companion sky separation (pixels) and position angle (deg E of N).
    opt_mask : boolean ndarray (ydim, xdim)
        The optimization region (same mask used to build the PCA).
    center : (x, y)
        Star pixel position.
    spectrum_of_wave : callable wavelength_um -> float
        Normalized spectrum value (the shape ``a``) at a given wavelength.
    derot_sign : float
        Sign convention linking parallactic angle to the derotation angle; the
        raw-frame azimuth uses ``derot_angle = derot_sign * q`` (default -1, i.e.
        derotate by ``-PA`` as in the example reduction).

    Returns ``A_delta`` with shape ``(n_refs, n_pixels_in_mask)``.'''
    ny, nx = opt_mask.shape
    n_pix = int(opt_mask.sum())
    A = np.empty((len(ref_meta), n_pix), dtype=float)
    for m, (q, wv) in enumerate(ref_meta):
        xr, yr = companion_frame_position(sep_pix, pa_deg, derot_sign * q, center)
        stamp = psf_of_wave(wv) * spectrum_of_wave(wv)
        A[m] = place_psf((ny, nx), stamp, xr, yr)[opt_mask]
    return A


def perturbed_kl_modes(pca, A_delta, a_frames, npca, return_delta_lambda=False,
                       return_mixing=False):
    '''First-order perturbation of the pixel-space KL images (Pueyo Eq. 51).

    Returns ``Delta_Z`` with shape ``(n_pixels, npca)`` -- column k is the
    perturbation ``Delta Z_k(x)`` of the k-th KL image caused by the presence of
    a source ``a_frames * A_delta`` in the reference library. The three physical
    terms of Eq. 51 are all included:

    * normalization / indirect self-subtraction  ``-(delta Lambda_k / 2 Lambda_k) Z_k``
    * mode-mixing / indirect self-subtraction     ``sum_j sqrt(Lambda_j/Lambda_k) (V_j^T C V_k)/(Lambda_k-Lambda_j) Z_j``
    * direct self-subtraction                     ``(1/sqrt(Lambda_k)) V_k^T a A_delta``

    where ``C = C_{A_delta S} = (a A_delta) S^T + S (a A_delta)^T`` is the
    source-speckle cross-covariance. ``C`` is never formed explicitly (it is
    ``n_refs x n_refs``): every quantity factors through ``V^T (a A_delta)`` and
    ``S^T V = Z sqrt(Lambda)``, which are small.

    pca : jPCA.PCA
        A fitted PCA carrying ``left_vectors`` (V), ``eigenvectors`` (Z),
        ``eigenvalues`` (Lambda), ``singular_values`` (sqrt(Lambda)) and
        ``library`` (S). Build it with ``method='svd'`` for well-conditioned
        eigenvalue gaps.
    A_delta : ndarray (n_refs, n_pixels)
        Source-image matrix from ``build_source_model``.
    a_frames : ndarray (n_refs,)
        Per-frame spectrum value (the diagonal of Pueyo's bold ``a``).
    npca : int
        Number of KL modes carried into the reduction (columns returned).
    return_delta_lambda : bool
        If True, also return ``delta_Lambda_k / Lambda_k`` for the first ``npca``
        modes -- the eigenvalue perturbation relative to the eigenvalue. Pueyo
        (2016, App. E.3) uses its size to judge whether the first-order expansion
        is valid (it should be << 1).
    return_mixing : bool
        If True, also return the largest mixing ACROSS THE TRUNCATION,
        ``max |V_j^T C V_k| / |Lambda_k - Lambda_j|`` over kept modes k < npca and
        discarded modes j >= npca. Mixing among kept modes leaves the KLIP
        projection unchanged; mixing across the cut changes which speckle modes
        are removed. When two eigenvalues straddling the cut are nearly
        degenerate this is large even for a faint source, and first-order theory
        fails although ``delta_Lambda/Lambda`` is small (a "mode crossing").
        Returned after the delta-Lambda array when both are requested.'''
    V = pca.left_vectors                 # (n_refs, K)
    Z = pca.eigenvectors                 # (n_pixels, K)
    Lam = np.asarray(pca.eigenvalues, dtype=float)      # (K,)
    sqrtLam = np.asarray(pca.singular_values, dtype=float)  # (K,)
    K = Z.shape[1]
    npca = min(npca, K)

    aA = a_frames[:, None] * A_delta      # (n_refs, n_pixels)  = bold_a A_delta
    Da = V.T @ aA                         # (K, n_pixels)       = V_k^T a A_delta
    # W[i,j] = V_i^T (a A_delta) S^T V_j ; using S^T V = Z * sqrt(Lambda)
    W = Da @ (Z * sqrtLam[None, :])       # (K, K)
    M = W + W.T                           # (K, K) = V_i^T C_{A_delta S} V_j
    delta_Lam = np.diag(M).copy()         # (K,)   = delta Lambda_k

    # Guard the eigenvalue-gap divisions (degenerate / null-space modes).
    tol = np.finfo(float).eps * K * (Lam[0] if Lam.size else 1.0)
    safe_lam = Lam > tol

    dZ = np.zeros((Z.shape[0], npca), dtype=float)
    for k in range(npca):
        if not safe_lam[k]:
            continue
        # normalization term
        dZ[:, k] += -(delta_Lam[k] / (2.0 * Lam[k])) * Z[:, k]
        # mode-mixing term (sum over all modes j != k)
        gap = Lam[k] - Lam
        coeff = np.zeros(K, dtype=float)
        good = safe_lam & (np.abs(gap) > tol)
        good[k] = False
        coeff[good] = np.sqrt(Lam[good] / Lam[k]) * M[good, k] / gap[good]
        dZ[:, k] += Z @ coeff
        # direct self-subtraction term
        dZ[:, k] += Da[k, :] / sqrtLam[k]
    out = (dZ,)
    if return_delta_lambda:
        rel = np.zeros(npca)
        rel[safe_lam[:npca]] = delta_Lam[:npca][safe_lam[:npca]] / Lam[:npca][safe_lam[:npca]]
        out += (rel,)
    if return_mixing:
        mix = 0.0
        if npca < K:
            kept = np.arange(npca)[safe_lam[:npca]]
            disc = np.arange(npca, K)[safe_lam[npca:]]
            if kept.size and disc.size:
                gap = np.abs(Lam[kept][None, :] - Lam[disc][:, None])
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.abs(M[np.ix_(disc, kept)]) / gap
                mix = float(np.nanmax(np.where(gap > tol, ratio, np.inf)))
        out += (mix,)
    return out[0] if len(out) == 1 else out


def forward_model_psf(pca, A_delta, a_frames, model_target, target_pixels, npca,
                      return_delta_lambda=False, return_mixing=False):
    '''Forward-model how KLIP processes a source in one target frame (Pueyo App. F).

    Returns the post-KLIP image of the modeled source over the optimization
    region -- the source as it survives over-subtraction and self-subtraction.
    This is the throughput kernel: compare its aperture flux to that of
    ``model_target`` to get the KLIP throughput, and use it as the matched-filter
    template for the noise apertures.

    The processed source is (first order in the source flux):

        FM = model_target
             - sum_k <model_target, Z_k> Z_k              (over-subtraction)
             - sum_k <T, Z_k> Delta Z_k                    (self-subtraction)
             - sum_k <T, Delta Z_k> Z_k                    (self-subtraction)

    with sums over the first ``npca`` modes and inner products taken over the
    optimization region. ``Delta Z_k`` comes from ``perturbed_kl_modes``.

    pca : jPCA.PCA
        Fitted PCA (see ``perturbed_kl_modes``).
    A_delta : ndarray (n_refs, n_pixels)
        Source model in the reference frames (drives the basis perturbation).
    a_frames : ndarray (n_refs,)
        Per-frame spectrum value for the references.
    model_target : ndarray (n_pixels,)
        The source model in *this* target frame, over the region (the thing being
        over-subtracted). Not median-subtracted -- it is an additive source.
    target_pixels : ndarray (n_pixels,)
        The target data over the region. The PCA's ``median_image`` is removed
        internally to match how ``jPCA.PCA.__call__`` projects.
    npca : int
        Number of KL modes in the reduction.
    return_delta_lambda : bool
        Also return ``delta_Lambda_k / Lambda_k`` (see ``perturbed_kl_modes``).
    return_mixing : bool
        Also return the cross-truncation mixing (see ``perturbed_kl_modes``),
        after the delta-Lambda array.

    Returns the forward-modeled source over the region (``(n_pixels,)``).'''
    Z = pca.eigenvectors[:, :npca]                       # (n_pixels, npca)
    model_target = np.asarray(model_target, dtype=float)
    A_delta = np.asarray(A_delta, dtype=float)
    T = np.asarray(target_pixels, dtype=float)

    # Center the source model and the target consistently with how the PCA
    # centered its library (see jPCA.PCA ``centering``). For 'frame' this is
    # Pueyo's exact convention -- zero-mean A_delta (ms10.tex:890) and the
    # target/source over the zone -- and there is no neglected mean-shift term.
    # For 'median' the source model is left raw (the neglected median-image
    # leakage is the ~0.5% term validated in test_jfm.py).
    if getattr(pca, 'centering', 'median') == 'frame':
        A_delta = A_delta - A_delta.mean(axis=1, keepdims=True)
        model_target = model_target - model_target.mean()
        T = T - T.mean()
    else:
        T = T - pca.median_image

    dZ, rel, mix = perturbed_kl_modes(pca, A_delta, a_frames, npca, return_delta_lambda=True,
                                      return_mixing=True)
    over = Z @ (Z.T @ model_target)          # sum_k <model_target, Z_k> Z_k
    self_a = dZ @ (Z.T @ T)                  # sum_k <T, Z_k> Delta Z_k
    self_b = Z @ (dZ.T @ T)                  # sum_k <T, Delta Z_k> Z_k
    fm = model_target - over - self_a - self_b
    out = (fm,)
    if return_delta_lambda:
        out += (rel,)
    if return_mixing:
        out += (mix,)
    return out[0] if len(out) == 1 else out


# --------------------------------------------------------------------------- #
# Inputs for detection limits: model spectra and the off-axis PSF
# --------------------------------------------------------------------------- #
def load_model_spectrum(path, wavelengths_um, normalize='mean'):
    '''Load a two-column (micron, mJy) model spectrum and resample it to the data
    wavelength grid, returning the *normalized shape* used for KLIP forward
    modeling and for the wavelength combination.

    path : str
        Text file with columns ``wavelength_um  flux_mjy`` (``#`` comments ok).
    wavelengths_um : array
        Data wavelength grid to interpolate onto.
    normalize : {'mean', 'median', 'none'}
        The returned shape ``a`` is divided by this statistic of the resampled
        flux, so a flat spectrum -> ~1 everywhere and the absolute (mJy) scale
        drops out. ``'none'`` returns the interpolated flux unchanged.

    Returns the resampled (and normalized) spectrum, same length as
    ``wavelengths_um``.'''
    w, f = np.loadtxt(path, comments='#', unpack=True)
    order = np.argsort(w)
    fi = np.interp(np.asarray(wavelengths_um, float), w[order], f[order])
    if normalize == 'mean':
        return fi / fi.mean()
    if normalize == 'median':
        return fi / np.median(fi)
    return fi


def offaxis_psf_stamps(nocoro_cube, center, half=10):
    '''Per-wavelength off-axis instrument PSF stamps from unocculted exposures.

    The unocculted star is the off-axis PSF. The exposures (one snapshot is
    enough; e.g. one at the start and one at the end of a sequence) are
    median-combined and a stamp is cut at each wavelength, sub-pixel shifted so
    the star sits exactly on the stamp's central pixel -- the convention
    ``place_psf`` assumes. This is the chromatic ``A_delta`` template (its width
    grows with wavelength), and it defines unit contrast: a source equal to
    ``c * stamp`` has companion/star flux ratio ``c``.

    nocoro_cube : ndarray (n_exp, n_wave, ny, nx) or (n_wave, ny, nx)
        Unocculted exposure(s) (star at ``center``).
    center : (x, y)
        Star pixel position (pixel-center convention; may be fractional).
    half : int
        Half-width of the square stamp.

    Returns stamps of shape ``(n_wave, 2*half+1, 2*half+1)``.'''
    from scipy.ndimage import shift as ndshift
    cube = np.asarray(nocoro_cube, dtype=float)
    if cube.ndim == 4:
        cube = np.median(cube, axis=0)
    cx, cy = center
    ix, iy = int(np.floor(cx + 0.5)), int(np.floor(cy + 0.5))   # nearest pixel (no banker's rounding)
    pad = half + 3
    sub = cube[:, iy - pad:iy + pad + 1, ix - pad:ix + pad + 1]
    dx, dy = ix - cx, iy - cy                                    # move the star onto pixel (ix, iy)
    if dx or dy:
        sub = np.array([ndshift(s_, (dy, dx), order=3, mode='nearest') for s_ in sub])
    return sub[:, pad - half:pad + half + 1, pad - half:pad + half + 1]


def annulus_arc_masks(shape, center, r_in, r_out, n_arcs=4, phi0=0.0):
    '''Split an annulus into ``n_arcs`` equal azimuthal arcs (the KLIP subtraction
    zones). Returns a list of boolean masks, one per arc, and the arc-edge angles
    (radians) so a source's arc can be looked up. ``phi0`` rotates the arc
    boundaries.'''
    ny, nx = shape
    yy, xx = np.indices((ny, nx), dtype=float)
    rr = np.hypot(xx - center[0], yy - center[1])
    phi = np.mod(np.arctan2(yy - center[1], xx - center[0]) - phi0, 2 * np.pi)
    edges = np.linspace(0, 2 * np.pi, n_arcs + 1)
    in_ann = (rr >= r_in) & (rr <= r_out)
    masks = [in_ann & (phi >= edges[i]) & (phi < edges[i + 1]) for i in range(n_arcs)]
    return masks, edges


def _arc_of(x, y, center, edges, phi0=0.0):
    '''Which arc index a pixel position falls in (matches ``annulus_arc_masks``).'''
    phi = np.mod(np.arctan2(y - center[1], x - center[0]) - phi0, 2 * np.pi)
    return int(np.searchsorted(edges, phi, side='right') - 1) % (len(edges) - 1)


def _aperture_flux(image, x, y, radius):
    '''Sum of finite pixels within ``radius`` of (x, y).'''
    ny, nx = image.shape
    yy, xx = np.indices((ny, nx), dtype=float)
    m = (np.hypot(xx - x, yy - y) <= radius) & np.isfinite(image)
    return float(image[m].sum())


def matched_filter_ring(image, template, x, y, center, res_pix,
                        exclude_pos=None, exclude_radius=None, return_ring=False):
    '''Matched-filter (scale-factor) photometry around a ring of independent
    lambda/D apertures at the separation of ``(x, y)``.

    Rather than summing flux in an aperture, at each ring position we fit the
    amplitude of the template to the image patch,

        alpha_i = sum(T * P_i) / sum(T * T),

    i.e. the least-squares scale factor for ``P_i ~ alpha_i * T`` (the maximum-
    likelihood flux under white noise). The scatter of ``alpha_i`` around the
    ring is the matched-filter noise -- literally the variance in the scale
    factor, which is the correct thing to threshold on. When ``template`` is the
    KLIP forward-modeled PSF of a *unit-contrast* source (throughput already
    baked in), ``alpha`` and its scatter are in contrast (flux-ratio) units.

    image : ndarray (ny, nx)
        The reduced image to measure (may contain NaNs).
    template : ndarray (h, h)
        Centered, odd-sized template stamp (the forward-modeled PSF).
    x, y : float
        Source position (the first, on-source, aperture).
    center : (x, y)
        Star position (center of the ring).
    res_pix : float
        Resolution element (lambda/D) in pixels; sets the ring spacing.
    exclude_pos, exclude_radius : see ``jPhot.get_noise``.
    return_ring : bool
        Also return the amplitude at every ring position (fixed order, NaN where
        excluded or off the image), so that rings with the same ``res_pix`` line
        up position by position across channels.

    Returns (alpha_on_source, std_alpha_ring, n_apertures[, ring_amplitudes]).'''
    ny, nx = image.shape
    h = template.shape[0] // 2
    cx, cy = center
    r = np.hypot(x - cx, y - cy)
    dtheta = res_pix / r
    n_ring = int(np.rint(2 * np.pi / dtheta))
    theta0 = np.arctan2(y - cy, x - cx)
    thetas = theta0 + dtheta * np.arange(n_ring)
    xs = cx + r * np.cos(thetas)
    ys = cy + r * np.sin(thetas)
    tw = np.isfinite(template)
    tt = float(np.sum(template[tw] ** 2))

    def scale(xi, yi):
        xir, yir = int(round(xi)), int(round(yi))
        if yir - h < 0 or yir + h + 1 > ny or xir - h < 0 or xir + h + 1 > nx or tt <= 0:
            return np.nan
        patch = image[yir - h:yir + h + 1, xir - h:xir + h + 1]
        m = tw & np.isfinite(patch)
        return float(np.sum(template[m] * patch[m]) / tt)

    alpha_src = scale(xs[0], ys[0])
    er = exclude_radius if exclude_radius is not None else res_pix
    ring_all = np.full(n_ring - 1, np.nan)
    for i, (xi, yi) in enumerate(zip(xs[1:], ys[1:])):
        if exclude_pos is not None and np.hypot(xi - exclude_pos[0], yi - exclude_pos[1]) <= er:
            continue
        ring_all[i] = scale(xi, yi)
    ring = ring_all[np.isfinite(ring_all)]
    out = (alpha_src, (float(np.std(ring)) if ring.size else np.nan), ring.size)
    return out + (ring_all,) if return_ring else out


# --------------------------------------------------------------------------- #
# Contrast curve via KLIP forward modeling (Pueyo 2016, Appendix F detection limits)
# --------------------------------------------------------------------------- #
def contrast_curve(cube, nocoro, pa, waves, center, radii, D, pixscale,
                   npca=20, n_arcs=4, dr=3.5, exclusion_fwhm=0.75,
                   sigma_threshold=5.0, source_pa=0.0, n_probe=6, aperture_frac=0.5,
                   small_sample=True, combine='median', derot_sign=-1.0,
                   known_companion=None, spectral_refs=False, return_resid=False,
                   resid_exclude_pix=None, comp_exclude_fwhm=2.5, centering='frame',
                   return_images=False, ring_wave=None, psf_stamps=None, n_jobs=1,
                   verbose=True):
    '''KLIP-FM contrast curve for an IFS ADI coronagraph sequence.

    At each radius in ``radii`` the field annulus is split into ``n_arcs`` arcs
    (the KLIP subtraction zones). Every (exposure, channel) frame is reduced with
    references whose ADI(+SDI) displacement exceeds ``exclusion_fwhm`` lambda/D
    (``jPCA.adi_sdi_displacement``), derotated and combined into a per-channel
    image. In the same pass a unit-contrast probe is forward-modeled through the
    reduction (``forward_model_psf``) and derotated identically, giving the
    KLIP-attenuated template ``T_lambda`` at each probe position. The per-channel
    noise is the matched-filter scale-factor scatter (``matched_filter_ring``):
    the template is fit at a ring of independent lambda/D positions on the
    reduced image and the spread of the fitted amplitude is the 1-sigma contrast
    noise (throughput folded in). ``combine_contrast`` / ``combine_contrast_cov``
    then combine channels for an assumed spectrum.

    Detection limits are quoted at a *constant false-alarm probability* -- the
    one a Gaussian ``sigma_threshold`` implies (FAP = norm.sf(sigma_threshold),
    2.9e-7 for 5 sigma). With ``small_sample`` the per-radius threshold factor is
    the Student-t value at that FAP with ``n_aper - 1`` degrees of freedom, times
    the Mawet+2014 ``sqrt(1 + 1/n_aper)`` penalty, so the FAP is held fixed at
    every separation even though a naive ``sigma_threshold * noise`` would not.

    cube : ndarray (n_exp, n_wave, ny, nx)
        Coronagraph sequence to measure.
    nocoro : ndarray (n_snap, n_wave, ny, nx) or None
        Unocculted exposure(s) of the star -- e.g. one snapshot at the start and
        one at the end of the sequence -- giving the off-axis PSF that defines
        unit contrast (median over the exposures). Ignored if ``psf_stamps`` is
        given.
    pa : array (n_exp,)
        Parallactic angle per exposure (degrees).
    waves : array (n_wave,)
        Wavelength per slice (microns).
    center : (x, y)
        Star pixel position (pixel-center convention).
    radii : array
        Separations (pixels) at which to evaluate the contrast.
    D, pixscale : float
        Telescope diameter (m) and plate scale (mas/pixel).
    npca : int
        KL modes used in the reduction / forward model.
    n_arcs : int
        Number of azimuthal subtraction zones per annulus.
    dr : float
        Annulus half-width (pixels).
    exclusion_fwhm : float
        Reference exclusion in resolution elements (lambda/D).
    sigma_threshold : float
        Detection confidence as a Gaussian-equivalent sigma; defines the fixed
        false-alarm probability the whole curve is quoted at.
    source_pa : float
        Reference sky position angle (deg) for the throughput/template probes.
    n_probe : int
        Number of probe azimuths, evenly spaced from ``source_pa``, whose
        throughput and matched-filter noise are measured and averaged (median) to
        smooth azimuthal/coverage variation in the estimate.
    aperture_frac : float
        Template stamp half-size as a fraction of lambda/D (also the throughput
        aperture radius).
    small_sample : bool
        Apply the Mawet+2014 small-sample (Student-t) statistics correction.
    combine : {'median', 'mean'}
        Combination over derotated exposures. The template is the mean of the
        per-exposure forward models (exact for 'mean'; for 'median' it assumes the
        source's imprint is similar in every exposure).
    derot_sign : float
        Sign linking parallactic angle to derotation (see ``build_source_model``).
    known_companion : (pa_deg, sep_pix) or None
        Sky position of a real companion; ring apertures within
        ``comp_exclude_fwhm`` lambda/D of it are excluded from the noise
        estimate so it does not bias the curve.
    spectral_refs : bool
        If True, the reference library for each target frame is drawn from all
        wavelengths (combined ADI+SDI selection). If False (default), references
        are restricted to the *same wavelength* as the target -- i.e. ADI per
        spectral channel. Without spatial rescaling of the channels to a common
        wavelength, the cross-wavelength references are chromatically mismatched
        speckle fields, so ADI per channel is the sound and far cheaper choice;
        the wavelength information still enters through the spectral combination.
    return_resid : bool
        If True, also return ``resid_pix`` -- a list (one per radius) of the
        reduced annulus pixels, shape ``(n_wave, n_pixels_in_annulus)``, for
        estimating the channel-to-channel noise correlation (Greco & Brandt
        2016; see ``spectral_correlation`` / ``fit_gb16_correlation``).
    resid_exclude_pix : float or None
        Radius (pixels) around ``known_companion`` excluded from ``resid_pix``.
    comp_exclude_fwhm : float
        Noise-ring exclusion around ``known_companion``, lambda/D. (A bright
        companion's KLIP arcs extend over the whole sky rotation at its
        separation; subtract it first -- ``inject_source`` with minus its
        ``extract_spectrum`` result -- and keep this exclusion for the residual.)
    centering : {'frame', 'median'}
        Library centering (see ``jPCA.PCA``). 'frame' (default) makes the forward
        model -- hence the throughput and the template -- exact to first order.
        'median' removes less of a planet's own flux in small inner zones and on
        the SCALES validation data reaches ~25% deeper inside ~200 mas, but its
        forward model neglects a median-shift term worth a few percent of the
        throughput (validation/, V1 and V3).
    psf_stamps : ndarray (n_wave, h, h) or None
        Off-axis PSF per channel, centered, in the same flux units as ``cube`` (scale
        it for any exposure-time or neutral-density difference). If None it is
        built from ``nocoro`` with ``offaxis_psf_stamps``.
    ring_wave : float or None
        Wavelength (microns) whose lambda/D spaces the COMMON ring of noise
        positions shared by all channels (default: the longest wavelength), at
        which ``ring_amps`` are recorded for ``combine_contrast_empirical``.
    return_images : bool
        Also return ``reduced`` -- (n_radii, n_wave, ny, nx) reduced, derotated,
        combined images (each radius's annulus) -- and ``templates`` -- per radius
        and channel, the (n_probe, h, h) matched-filter template stamps (NaN where
        unavailable) centered on each probe's sky position.
    n_jobs : int
        With ``spectral_refs=False`` the channels are independent; ``n_jobs > 1``
        measures them in that many worker processes (one BLAS thread each with
        ``threadpoolctl``; see ``klip_fm_reduce``).
    verbose : bool
        Print progress.

    Returns a dict with per-radius, per-wavelength arrays ``mf_noise`` (matched-
    filter 1-sigma contrast noise), ``mf_signal`` (the on-source matched-filter
    amplitude at the probe position -- in contrast units, throughput cancels --
    only physically meaningful when a real source sits at ``source_pa``, e.g. set
    ``source_pa`` to a known companion's PA and ``n_probe=1`` to read its
    per-channel contrast; ``mf_signal / mf_noise`` is then the per-channel S/N),
    ``tau`` (throughput, diagnostic) and ``fstar``; per-radius ``n_aper`` and
    ``nsig_eff`` (the constant-FAP threshold factor); ``fap``; the wavelength
    grid; and ``ring_amps`` -- per radius, an (n_wave, n_ring) array of the first
    probe's matched-filter amplitude at the common ring positions (NaN where
    excluded). Combine into curves with ``combine_contrast_empirical``
    (recommended), ``combine_contrast`` or ``combine_contrast_cov``.'''
    if psf_stamps is None:
        half = int(np.ceil(2 * resolution_element_pix(np.max(waves), D, pixscale)))
        psf_stamps = offaxis_psf_stamps(nocoro, center, half=half)
    psf_stamps = np.asarray(psf_stamps, dtype=float)
    if n_jobs > 1 and not spectral_refs and np.size(waves) > 1:
        return _contrast_curve_parallel(
            cube, psf_stamps, pa, waves, center, radii, D, pixscale, n_jobs, small_sample,
            dict(npca=npca, n_arcs=n_arcs, dr=dr, exclusion_fwhm=exclusion_fwhm,
                 sigma_threshold=sigma_threshold, source_pa=source_pa, n_probe=n_probe,
                 aperture_frac=aperture_frac, small_sample=small_sample, combine=combine,
                 derot_sign=derot_sign, known_companion=known_companion,
                 return_resid=return_resid, resid_exclude_pix=resid_exclude_pix,
                 comp_exclude_fwhm=comp_exclude_fwhm, centering=centering,
                 return_images=return_images,
                 ring_wave=float(np.max(waves)) if ring_wave is None else ring_wave), verbose)

    cube = np.asarray(cube, dtype=float)
    n_exp, n_wave, ny, nx = cube.shape
    waves = np.asarray(waves, dtype=float)
    pa = np.asarray(pa, dtype=float)
    radii = np.asarray(radii, dtype=float)

    if source_pa is None:
        source_pa = 45.0
    combiner = np.nanmedian if combine == 'median' else np.nanmean

    # off-axis PSF stamps (chromatic) and stellar aperture flux per wavelength
    stamps = psf_stamps                                        # (n_wave, H, W)
    def psf_of_index(wi):
        return stamps[wi]
    ap_rad = np.array([aperture_frac * resolution_element_pix(w, D, pixscale) for w in waves])
    scx = (stamps.shape[2] - 1) / 2.0
    fstar = np.array([_aperture_flux(stamps[wi], scx, scx, ap_rad[wi])
                      for wi in range(n_wave)])

    # per (radius, arc) geometry
    arc_info = []
    for r in radii:
        masks, edges = annulus_arc_masks((ny, nx), center, r - dr, r + dr, n_arcs)
        arc_info.append((masks, edges))

    mf_noise = np.full((len(radii), n_wave), np.nan)   # matched-filter contrast noise
    mf_signal = np.full((len(radii), n_wave), np.nan)  # on-source matched-filter amplitude
    tau = np.full((len(radii), n_wave), np.nan)         # throughput (diagnostic)
    n_aper = np.zeros(len(radii), dtype=int)

    # known-companion sky pixel position (for noise-ring exclusion)
    comp_xy = None
    if known_companion is not None:
        cpa, csep = known_companion
        comp_xy = (center[0] - csep * np.sin(np.deg2rad(cpa)),
                   center[1] + csep * np.cos(np.deg2rad(cpa)))

    # frame flat-index bookkeeping (t, w) -> row
    T_idx = np.repeat(np.arange(n_exp), n_wave)
    W_idx = np.tile(np.arange(n_wave), n_exp)
    Q = pa[T_idx]
    WV = waves[W_idx]
    n_all = T_idx.size

    # throughput/template probe azimuths (averaged to smooth azimuthal variation)
    probe_pas = source_pa + np.linspace(0.0, 360.0, n_probe, endpoint=False)

    _yy, _xx = np.indices((ny, nx))
    RR = np.hypot(_xx - center[0], _yy - center[1])
    resid_pix = []
    res_ring = resolution_element_pix(waves.max() if ring_wave is None else ring_wave, D, pixscale)
    ring_amps = []
    reduced_all = np.full((len(radii), n_wave, ny, nx), np.nan) if return_images else None
    templates = [[None] * n_wave for _ in radii]

    for ri, r in enumerate(radii):
        masks, edges = arc_info[ri]
        # pre-extract region pixels for every arc and every frame, once
        cube_flat = cube.reshape(n_all, ny, nx)
        region_pix = [cube_flat[:, m] for m in masks]           # list of (n_all, n_pix)
        onemask = [np.ones(int(m.sum()), bool) for m in masks]
        all_meta = list(zip(Q, WV))
        psf_fn = psf_of_index_wrapper(psf_of_index, waves)
        # pre-build the unit-contrast probe source model on each arc, for every
        # frame, for each probe azimuth -> src_all[probe][arc] is (n_all, n_pix)
        src_all = [[build_source_model(all_meta, psf_fn, r, ppa, m, center,
                                       lambda wv: 1.0, derot_sign=derot_sign)
                    for m in masks] for ppa in probe_pas]
        # sky position of each probe (derotated frame)
        probe_xy = [(center[0] - r * np.sin(np.deg2rad(ppa)),
                     center[1] + r * np.cos(np.deg2rad(ppa))) for ppa in probe_pas]

        reduced_lambda = np.full((n_wave, ny, nx), np.nan)
        n_common = int(np.rint(2 * np.pi * r / res_ring)) - 1
        ring_amps.append(np.full((n_wave, n_common), np.nan))
        for w0 in range(n_wave):
            derot_res = []
            fm_sum = [np.zeros((ny, nx)) for _ in probe_pas]
            inj_sum = [np.zeros((ny, nx)) for _ in probe_pas]
            fm_cnt = [np.zeros((ny, nx)) for _ in probe_pas]
            inj_cnt = [np.zeros((ny, nx)) for _ in probe_pas]
            thr = exclusion_fwhm * np.array([resolution_element_pix(wv, D, pixscale) for wv in WV])
            wave_ok = np.ones(n_all, bool) if spectral_refs else (W_idx == w0)
            for t in range(n_exp):
                tgt = t * n_wave + w0
                disp = adi_sdi_displacement(r, Q, WV, pa[t], waves[w0])
                refs = np.where((disp > thr) & (np.arange(n_all) != tgt) & wave_ok)[0]
                if refs.size < npca + 1:
                    continue
                a_frames = np.ones(refs.size)   # flat spectrum for the probe

                full_resid = np.full((ny, nx), np.nan)
                fm_raw = [None] * n_probe
                inj_raw = [None] * n_probe
                for arc in range(n_arcs):
                    pca = PCA(region_pix[arc][refs], onemask[arc], method='svd',
                              centering=centering)
                    full_resid[masks[arc]] = pca(region_pix[arc][tgt].copy(), npca)
                    # Forward-model each probe in EVERY arc it affects: arcs its
                    # PSF touches in the target (over-subtraction, and flux that
                    # spills across a zone boundary) AND arcs where it appears only
                    # in the references (self-subtraction lobes, which for a probe
                    # near a boundary land right beside it after derotation).
                    for pi in range(n_probe):
                        mt = src_all[pi][arc][tgt]
                        if not (mt.any() or src_all[pi][arc][refs].any()):
                            continue
                        fmpix = forward_model_psf(pca, src_all[pi][arc][refs],
                                                  a_frames, mt,
                                                  region_pix[arc][tgt], npca)
                        if fm_raw[pi] is None:
                            fm_raw[pi] = np.zeros((ny, nx))
                            inj_raw[pi] = np.zeros((ny, nx))
                        fm_raw[pi][masks[arc]] = fmpix
                        inj_raw[pi][masks[arc]] = mt
                # Derotate the residual AND the per-frame models identically, so
                # the template lives in the same (sky) frame as the data. pivot=True
                # rotates about ``center`` exactly (the default re-centers the
                # axis on shape/2, a half-pixel shift for even-sized frames).
                ang = derot_sign * pa[t]
                dres = nan_rot(full_resid, ang, center, out_size=(ny, nx), pivot=True)
                derot_res.append(dres)
                valid = np.isfinite(dres)
                for pi in range(n_probe):
                    if fm_raw[pi] is None:
                        continue
                    fm_sum[pi] += valid * np.nan_to_num(
                        nan_rot(fm_raw[pi], ang, center, out_size=(ny, nx), pivot=True))
                    inj_sum[pi] += valid * np.nan_to_num(
                        nan_rot(inj_raw[pi], ang, center, out_size=(ny, nx), pivot=True))
                    fm_cnt[pi] += valid
                    inj_cnt[pi] += valid
            if not derot_res:
                continue
            with warnings.catch_warnings():               # pixels outside the annulus
                warnings.simplefilter("ignore", RuntimeWarning)
                reduced_lambda[w0] = combiner(np.array(derot_res), axis=0)

            # measure throughput and matched-filter noise at each probe, average.
            res_w = resolution_element_pix(waves[w0], D, pixscale)
            hs = max(2, int(np.ceil(aperture_frac * res_w)))
            taus, sigs, naps, amps = [], [], [], []
            tmpls = np.full((n_probe, 2 * hs + 1, 2 * hs + 1), np.nan)
            for pi in range(n_probe):
                sky_x, sky_y = probe_xy[pi]
                fm_img = np.where(fm_cnt[pi] > 0, fm_sum[pi] / np.maximum(fm_cnt[pi], 1), 0.0)
                inj_img = np.where(inj_cnt[pi] > 0, inj_sum[pi] / np.maximum(inj_cnt[pi], 1), 0.0)
                fm_flux = _aperture_flux(fm_img, sky_x, sky_y, ap_rad[w0])
                inj_flux = _aperture_flux(inj_img, sky_x, sky_y, ap_rad[w0])
                t_thru = fm_flux / inj_flux if inj_flux > 0 else np.nan
                if not (np.isfinite(t_thru) and t_thru > 0):
                    continue
                iy, ix = int(round(sky_y)), int(round(sky_x))
                tmpl = fm_img[iy - hs:iy + hs + 1, ix - hs:ix + hs + 1]
                if tmpl.shape != (2 * hs + 1, 2 * hs + 1) or np.sum(tmpl ** 2) <= 0:
                    continue
                tmpls[pi] = tmpl
                a_src, sig_a, nap = matched_filter_ring(
                    reduced_lambda[w0], tmpl, sky_x, sky_y, center, res_w,
                    exclude_pos=comp_xy,
                    exclude_radius=(comp_exclude_fwhm * res_w if comp_xy is not None else None))
                if pi == 0:
                    ring_amps[ri][w0] = matched_filter_ring(
                        reduced_lambda[w0], tmpl, sky_x, sky_y, center, res_ring,
                        exclude_pos=comp_xy,
                        exclude_radius=(comp_exclude_fwhm * res_ring if comp_xy is not None
                                        else None), return_ring=True)[3]
                if np.isfinite(sig_a) and sig_a > 0:
                    taus.append(t_thru); sigs.append(sig_a); naps.append(nap)
                    amps.append(a_src)
            templates[ri][w0] = tmpls
            if sigs:
                tau[ri, w0] = np.median(taus)
                mf_noise[ri, w0] = np.median(sigs)
                mf_signal[ri, w0] = np.nanmedian(amps)
                n_aper[ri] = max(n_aper[ri], int(np.median(naps)))
        if return_images:
            reduced_all[ri] = reduced_lambda
        if return_resid:
            # wavelength-independent annulus (so channels share pixels for the
            # covariance). Exclude the known companion with a FIXED-pixel radius
            # (a bright point source is perfectly correlated across channels and
            # would otherwise inflate the covariance at its own separation).
            ann = (RR >= r - dr) & (RR <= r + dr)
            if comp_xy is not None and resid_exclude_pix:
                ann &= (np.hypot(_xx - comp_xy[0], _yy - comp_xy[1]) > resid_exclude_pix)
            resid_pix.append(reduced_lambda[:, ann])       # (n_wave, n_pix_annulus)
        if verbose:
            print("  radius %.1f px done (median tau=%.3f, n_aper=%d)"
                  % (r, np.nanmedian(tau[ri]), n_aper[ri]), flush=True)

    fap, nsig_eff = _constant_fap_threshold(n_aper, sigma_threshold, small_sample)

    out = dict(radii=radii, waves=waves, mf_noise=mf_noise, mf_signal=mf_signal,
               tau=tau, fstar=fstar, n_aper=n_aper, nsig_eff=nsig_eff, fap=fap,
               sigma_threshold=sigma_threshold, pixscale=pixscale, ring_amps=ring_amps,
               small_sample=small_sample)
    if return_resid:
        out['resid_pix'] = resid_pix
    if return_images:
        out['reduced'] = reduced_all
        out['templates'] = templates
    return out


def _constant_fap_threshold(n_aper, sigma_threshold, small_sample):
    '''Constant false-alarm-probability threshold factor per radius. FAP is the
    one a Gaussian ``sigma_threshold`` implies; with ``small_sample`` the factor
    is the Student-t value at that FAP (dof = n_aper - 1) times the Mawet+2014
    sqrt(1 + 1/n_aper) penalty, holding the FAP fixed across separation.'''
    from scipy.stats import norm, t as student_t
    fap = float(norm.sf(sigma_threshold))
    if small_sample:
        nsig_eff = np.array([student_t.isf(fap, max(n - 1, 1)) * np.sqrt(1 + 1.0 / max(n, 2))
                             if n > 2 else sigma_threshold for n in n_aper])
    else:
        nsig_eff = np.full(len(n_aper), sigma_threshold)
    return fap, nsig_eff


def _single_blas_thread():
    '''Context manager limiting BLAS/OpenMP to one thread in a worker process,
    so n_jobs workers do not each start a thread per core. Uses threadpoolctl if
    installed (``pip install jighcontrast[parallel]``); otherwise a no-op, and
    OMP_NUM_THREADS=1 (etc.) should be set before numpy is imported.'''
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        import contextlib
        return contextlib.nullcontext()
    return threadpool_limits(limits=1)


def _contrast_one_channel(args):
    cube_w, psf_w, pa, wave, center, radii, D, pixscale, kw = args
    with _single_blas_thread():
        return contrast_curve(cube_w, None, pa, wave, center, radii, D, pixscale,
                              psf_stamps=psf_w, spectral_refs=False, n_jobs=1, verbose=False,
                              **kw)


def _contrast_curve_parallel(cube, psf_stamps, pa, waves, center, radii, D, pixscale, n_jobs,
                             small_sample, kw, verbose):
    '''Channel-parallel ``contrast_curve`` for ADI per channel.'''
    from concurrent.futures import ProcessPoolExecutor
    waves = np.asarray(waves, float)
    n_wave = waves.size
    jobs = ((np.asarray(cube[:, w:w + 1], dtype=float), psf_stamps[w:w + 1],
             pa, waves[w:w + 1], center, radii, D, pixscale, kw) for w in range(n_wave))
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        parts = list(ex.map(_contrast_one_channel, jobs))
    cat = lambda k: np.concatenate([p_[k] for p_ in parts], axis=1)
    n_aper = np.max(np.array([p_['n_aper'] for p_ in parts]), axis=0)
    fap, nsig_eff = _constant_fap_threshold(n_aper, kw['sigma_threshold'], small_sample)
    out = dict(radii=np.asarray(radii, float), waves=waves, mf_noise=cat('mf_noise'),
               mf_signal=cat('mf_signal'), tau=cat('tau'),
               fstar=np.concatenate([p_['fstar'] for p_ in parts]), n_aper=n_aper,
               nsig_eff=nsig_eff, fap=fap, sigma_threshold=kw['sigma_threshold'],
               pixscale=pixscale, small_sample=small_sample,
               ring_amps=[np.concatenate([p_['ring_amps'][ri] for p_ in parts], axis=0)
                          for ri in range(len(radii))])
    if kw.get('return_resid'):
        out['resid_pix'] = [np.concatenate([p_['resid_pix'][ri] for p_ in parts], axis=0)
                            for ri in range(len(radii))]
    if kw.get('return_images'):
        out['reduced'] = np.concatenate([p_['reduced'] for p_ in parts], axis=1)
        out['templates'] = [[p_['templates'][ri][0] for p_ in parts]
                            for ri in range(len(radii))]
    if verbose:
        print("  contrast curve: %d channels on %d workers" % (n_wave, n_jobs), flush=True)
    return out


def psf_of_index_wrapper(psf_of_index, waves):
    '''Return a callable wavelength_um -> PSF stamp by nearest wavelength index.'''
    waves = np.asarray(waves, float)
    def f(wv):
        return psf_of_index(int(np.argmin(np.abs(waves - wv))))
    return f


def combine_contrast(result, spectrum_shape=None):
    '''Combine the per-wavelength matched-filter noise into a contrast curve for
    an assumed spectral shape, via the matched-filter-across-wavelength weighting
    (Pueyo 2016 Appendix F detection limits).

    Each wavelength gives a 1-sigma contrast noise ``mf_noise_lambda(r)`` (from
    ``matched_filter_ring``; throughput already folded into the template). For a
    planet whose contrast at wavelength lambda is ``c * s_lambda`` the maximum-
    likelihood combined estimate of ``c`` has variance ``1 / sum_lambda
    (s_lambda / mf_noise_lambda)^2``, so the detection limit is

        contrast(r) = nsig_eff(r) / sqrt( sum_lambda ( s_lambda / mf_noise_lambda(r) )^2 ).

    A uniform ``spectrum_shape`` (or None) is the flat / inverse-variance case.

    result : dict
        Output of ``contrast_curve``.
    spectrum_shape : array (n_wave,) or None
        Normalized planet/star flux-ratio shape; None -> flat (ones).

    Returns (radii, contrast) with contrast as a flux ratio, quoted at the fixed
    false-alarm probability ``result['fap']``.'''
    mf = result['mf_noise']
    nsig = result['nsig_eff']
    s = np.ones_like(result['waves']) if spectrum_shape is None else np.asarray(spectrum_shape, float)
    w = s[None, :] / mf                                     # (n_radii, n_wave)
    denom = np.sqrt(np.nansum(w ** 2, axis=1))
    contrast = np.where(denom > 0, nsig / np.where(denom > 0, denom, 1.0), np.nan)
    return result['radii'], contrast


def spectral_correlation(resid, shrink=0.1):
    '''Empirical wavelength-to-wavelength correlation matrix (Greco & Brandt 2016).

    resid : ndarray (n_pixels, n_wave)
        Reduced annulus residuals; each column is one spectral channel. Rows
        (pixels) with any non-finite value are dropped so all channels share the
        same samples.
    shrink : float
        Shrinkage of the empirical correlation toward the identity (Ledoit-Wolf
        style) for a well-conditioned, invertible matrix.

    Returns an (n_wave, n_wave) correlation matrix.'''
    R = np.asarray(resid, float)
    good = np.all(np.isfinite(R), axis=1)
    R = R[good]
    n_wave = R.shape[1]
    if R.shape[0] < 3:
        return np.eye(n_wave)
    psi = np.nan_to_num(np.corrcoef(R, rowvar=False), nan=0.0)
    return (1.0 - shrink) * psi + shrink * np.eye(n_wave)


def combine_contrast_empirical(result, spectrum_shape=None, weights='ivar', shrink=0.3):
    '''Detection limit for a spectral shape with an EMPIRICALLY measured noise.

    At each radius the per-channel matched-filter amplitudes at the common ring
    positions (``result['ring_amps']``) are combined with weights normalized to
    ``w^T s = 1`` -- inverse-variance ``w ~ s / sigma^2`` (default) or GLS
    ``w ~ C^-1 s`` with ``C`` estimated from the same amplitudes and shrunk toward
    its diagonal by ``shrink`` -- and the scatter of the combined amplitude
    around the ring is the 1-sigma noise. GLS weights fitted to the very ring
    samples whose scatter sets the noise understate it (see
    ``broadband_contrast``), hence the inverse-variance default. The
    limit is ``nsig_eff * sigma``, with the Student-t constant-FAP factor
    recomputed for the number of common ring positions. Because the noise is
    measured, not propagated through ``C^-1``, it is correct even when speckles
    are almost perfectly correlated across wavelength (near the star), where
    ``combine_contrast_cov`` is overconfident.

    Returns (radii, contrast, n_ring).'''
    radii = result['radii']
    waves = result['waves']
    s_full = np.ones_like(waves) if spectrum_shape is None else np.asarray(spectrum_shape, float)
    contrast = np.full(len(radii), np.nan)
    n_ring = np.zeros(len(radii), int)
    for ri in range(len(radii)):
        amps = np.asarray(result['ring_amps'][ri], float)        # (n_wave, n_ring)
        ch_ok = np.sum(np.isfinite(amps), axis=1) > 2
        amps = amps[ch_ok]
        pos_ok = np.all(np.isfinite(amps), axis=0)
        X = amps[:, pos_ok]                                       # (n_ch, n_pos)
        n = X.shape[1]
        if n < 4:
            continue
        s = s_full[ch_ok]
        Cemp = np.cov(X)
        if weights == 'gls':
            C = (1 - shrink) * Cemp + shrink * np.diag(np.diag(Cemp))
            try:
                w = np.linalg.solve(C, s)
            except np.linalg.LinAlgError:
                w = np.linalg.pinv(C) @ s
        elif weights == 'ivar':
            w = s / np.diag(Cemp)
        else:
            raise ValueError("weights must be 'ivar' or 'gls'")
        w = w / float(w @ s)
        comb = w @ X
        sigma = float(np.std(comb, ddof=1))
        _, nsig = _constant_fap_threshold([n], result['sigma_threshold'],
                                          result.get('small_sample', True))
        contrast[ri] = nsig[0] * sigma
        n_ring[ri] = n
    return radii, contrast, n_ring


def gb16_correlation(waves, params):
    '''Greco & Brandt (2016) parametric wavelength-correlation model,

        psi_ij = A_rho exp(-(1/2) ((l_i - l_j) / (sigma_rho l_ij))^2)
               + A_lam exp(-(1/2) ((l_i - l_j) / sigma_lam)^2)
               + A_delta delta_ij,                A_delta = 1 - A_rho - A_lam,

    with ``l_ij`` the mean of the two wavelengths. The first term is the
    speckle correlation (a speckle at separation rho moves by ~rho dl/l, so its
    correlation length is a fixed FRACTION of the wavelength); the second a
    wavelength-independent smooth component; the third uncorrelated noise. A sum
    of Gaussian kernels plus a non-negative diagonal is positive semi-definite,
    and with A_delta > 0 positive definite -- robust to invert, unlike a noisy
    empirical estimate.

    params : (A_rho, sigma_rho, A_lam, sigma_lam)
    Returns the (n_wave, n_wave) correlation matrix.'''
    A_rho, s_rho, A_lam, s_lam = params
    lam = np.asarray(waves, float)
    dl = lam[:, None] - lam[None, :]
    lm = 0.5 * (lam[:, None] + lam[None, :])
    psi = (A_rho * np.exp(-0.5 * (dl / (s_rho * lm)) ** 2)
           + A_lam * np.exp(-0.5 * (dl / s_lam) ** 2))
    psi[np.diag_indices_from(psi)] = 1.0
    return psi


def fit_gb16_correlation(empirical, waves):
    '''Least-squares fit of ``gb16_correlation`` to an empirical correlation
    matrix (off-diagonal elements). Returns (psi_model, params).'''
    from scipy.optimize import least_squares
    lam = np.asarray(waves, float)
    emp = np.asarray(empirical, float)
    iu = np.triu_indices(lam.size, 1)
    good = np.isfinite(emp[iu])
    span = lam.max() - lam.min()

    def resid(x):
        return (gb16_correlation(lam, x)[iu] - emp[iu])[good]

    best = None
    for s0 in (0.01, 0.05, 0.2):                    # a few starts: the fit is not convex
        x0 = (0.5, s0, 0.2, 0.5 * span)
        r = least_squares(resid, x0, bounds=([0, 1e-4, 0, 1e-3 * span], [1, 2.0, 1, 10 * span]))
        if r.x[0] + r.x[2] > 0.999:                 # keep a positive white-noise term
            r.x[0], r.x[2] = r.x[0] * 0.999 / (r.x[0] + r.x[2]), r.x[2] * 0.999 / (r.x[0] + r.x[2])
        if best is None or r.cost < best.cost:
            best = r
    return gb16_correlation(lam, best.x), tuple(best.x)


def combine_contrast_cov(result, corr, spectrum_shape=None):
    '''Correlation-aware spectral combination (matched filter with a covariance):

        contrast(r) = nsig_eff(r) / sqrt( s^T C(r)^-1 s ),   C = D psi D

    with ``D = diag(mf_noise_lambda)`` and ``psi`` the wavelength correlation
    matrix ``corr[r]`` (from ``spectral_correlation``). This is the rigorous
    generalization of ``combine_contrast`` (which is the ``psi = I`` / independent
    special case) and accounts for correlated speckle noise between channels.

    corr : sequence of (n_wave, n_wave) arrays, one per radius.

    Returns (radii, contrast, n_eff), where ``n_eff`` is the effective number of
    independent channels, ``(s^T C^-1 s)/(s^T D^-2 s) * n_channels``.'''
    mf, nsig, waves = result['mf_noise'], result['nsig_eff'], result['waves']
    radii = result['radii']
    s_full = np.ones_like(waves) if spectrum_shape is None else np.asarray(spectrum_shape, float)
    contrast = np.full(len(radii), np.nan)
    n_eff = np.full(len(radii), np.nan)
    for ri in range(len(radii)):
        sig = mf[ri]
        ok = np.isfinite(sig) & (sig > 0)
        if ok.sum() < 2:
            continue
        s = s_full[ok]; d = sig[ok]
        psi = np.asarray(corr[ri])[np.ix_(ok, ok)]
        C = psi * np.outer(d, d)
        try:
            Cinv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            Cinv = np.linalg.pinv(C)
        I = float(s @ Cinv @ s)               # correlated information
        I_ind = float(np.sum((s / d) ** 2))   # independent-channel information
        if I > 0:
            contrast[ri] = nsig[ri] / np.sqrt(I)
            n_eff[ri] = (I / I_ind) * ok.sum()
    return radii, contrast, n_eff


# --------------------------------------------------------------------------- #
# Spectral extraction (Pueyo 2016, Appendix F, Eqs. SimpleMat)
# --------------------------------------------------------------------------- #
def inject_source(cube, psf_stamps, pa, center, sep_pix, pa_deg, contrast,
                  derot_sign=-1.0):
    '''Return a copy of ``cube`` with a point source added to every frame.

    The source sits at sky separation ``sep_pix`` and position angle ``pa_deg``;
    in exposure ``t`` it is placed at its raw-frame position (see
    ``companion_frame_position``) as the off-axis PSF ``psf_stamps[w]`` times
    ``contrast[w]``. Because the stamps are the unocculted star, ``contrast`` is
    the companion/star flux ratio per channel. Used for injection-recovery tests.

    cube : ndarray (n_exp, n_wave, ny, nx)
    psf_stamps : ndarray (n_wave, h, h)
        Off-axis PSF per channel (``offaxis_psf_stamps``).
    pa : array (n_exp,)
        Parallactic angles (degrees).
    contrast : float or array (n_wave,)
        Flux ratio per channel.

    Returns the injected cube (float, same shape).'''
    out = np.array(cube, dtype=float, copy=True)
    n_exp, n_wave, ny, nx = out.shape
    c = np.broadcast_to(np.asarray(contrast, float), (n_wave,))
    for t in range(n_exp):
        xr, yr = companion_frame_position(sep_pix, pa_deg, derot_sign * pa[t], center)
        for w in range(n_wave):
            if c[w] != 0:
                out[t, w] += c[w] * place_psf((ny, nx), psf_stamps[w], xr, yr)
    return out


def _sample_rotated(stamp, x0, y0, xs, ys, dtheta_rad, center, order=3):
    '''Values of a stamp image (origin ``(x0, y0)`` in full-image pixels) after the
    whole image is rotated about ``center`` by ``dtheta_rad`` (CCW on the sky, i.e.
    increasing position angle), sampled at full-image pixels ``(xs, ys)``.'''
    from scipy.ndimage import map_coordinates
    c, s_ = np.cos(dtheta_rad), np.sin(dtheta_rad)
    dx, dy = xs - center[0], ys - center[1]
    # inverse rotation: where did (xs, ys) come from before the rotation?
    xb = center[0] + c * dx + s_ * dy
    yb = center[1] - s_ * dx + c * dy
    return map_coordinates(stamp, [yb - y0, xb - x0], order=order, mode='constant', cval=0.0)


def klip_fm_reduce(cube, pa, waves, center, sep_pix, pa_deg, psf_stamps, D, pixscale,
                   npca=20, n_arcs=4, dr=3.5, exclusion_fwhm=0.75, spectral_refs=False,
                   centering='frame', derot_sign=-1.0, stamp_half=None,
                   forward_model=True, n_jobs=1, verbose=False):
    '''KLIP reduction of an IFS ADI sequence plus the KLIP-FM model of a point
    source at a given sky position (the ingredients of Pueyo 2016 Eq. SimpleMat).

    The annulus at ``sep_pix`` (half-width ``dr``) is split into ``n_arcs``
    subtraction zones. Following Pueyo's prescription the zones are anchored on
    the source: in every target frame zone 0 is centered on the source's raw-frame
    azimuth, so the source never straddles a zone boundary, and after derotation
    the zone boundaries are fixed on the sky. Each (exposure, channel) frame is
    reduced with ``npca`` KL modes from references whose ADI(+SDI) displacement
    exceeds ``exclusion_fwhm`` lambda/D; the residuals are derotated and
    MEAN-combined (a linear combine, so the forward model is exactly linear in
    the source spectrum -- a median would not be).

    In the same pass, for every target frame at channel ``p`` and every source
    channel ``q`` present in its reference library, ``forward_model_psf`` gives
    the processed image of a unit-contrast source emitting only at ``q`` -- the
    ``q``-th row of Pueyo's ``F_{lambda_p}`` matrix (Eq. DefLambdaMatrix): over-
    subtraction of the target's own copy (only when ``q == p``) plus the direct
    and indirect self-subtraction from the copies in the references. These are
    derotated and mean-combined exactly like the data. With ``spectral_refs=False``
    (ADI per channel, the default) only ``q == p`` is non-zero, so ``F`` is
    diagonal and the spectral fit decouples channel by channel.

    cube : ndarray (n_exp, n_wave, ny, nx)
        Coronagraphic sequence (may be a memmap).
    pa : array (n_exp,)
        Parallactic angles, degrees.
    waves : array (n_wave,)
        Channel wavelengths, microns.
    center : (x, y)
        Star position.
    sep_pix, pa_deg : float
        Sky position of the source (separation in pixels, PA in deg E of N).
    psf_stamps : ndarray (n_wave, h, h)
        Off-axis PSF per channel; it defines the unit of the spectrum (contrast).
    D, pixscale : float
        Telescope diameter (m) and plate scale (mas/pixel).
    npca, n_arcs, dr, exclusion_fwhm, spectral_refs, centering, derot_sign :
        Reduction parameters (see ``contrast_curve``).
    stamp_half : int or None
        Half-size of the stored model stamps around the source; default
        3 lambda/D at the longest wavelength.
    forward_model : bool
        If False skip the model (just reduce the data, e.g. for brute-force tests).
    n_jobs : int
        With ``spectral_refs=False`` the channels are independent; ``n_jobs > 1``
        reduces them in that many worker processes. Each worker is limited to one
        BLAS thread if ``threadpoolctl`` is installed (otherwise set
        OMP_NUM_THREADS=1 before importing numpy). Scripts using it on macOS or
        Windows need an ``if __name__ == "__main__":`` guard (spawned workers).

    Returns a dict with
      ``P``        (n_wave, ny, nx) reduced, derotated, mean-combined data;
      ``F``        (n_wave, n_wave, 2h+1, 2h+1) model stamps, ``F[p, q]`` = image
                   in channel p of a unit-contrast source emitting at channel q;
      ``stamp_origin`` (x0, y0) full-image pixel of ``F[..., 0, 0]``;
      ``src_xy``   sky (derotated) pixel position of the source;
      ``dlam``     (n_wave,) max over frames and modes of |delta Lambda / Lambda|
                   for a UNIT-contrast source emitting at each channel (multiply
                   by the fitted contrast for the first-order validity check);
      ``mixing_frames`` (n_wave, n_exp) cross-truncation mode mixing in each
                   target frame for a unit-contrast source (``perturbed_kl_modes``;
                   max over zones; computed for the target's own channel);
      ``n_frames`` (n_wave,) frames that entered each channel's combine.'''
    pa = np.asarray(pa, float)
    waves = np.asarray(waves, float)
    n_exp, n_wave, ny, nx = cube.shape
    cx, cy = center
    if stamp_half is None:
        stamp_half = int(np.ceil(3 * resolution_element_pix(waves.max(), D, pixscale)))
    if n_jobs > 1 and not spectral_refs and n_wave > 1:
        return _klip_fm_reduce_parallel(
            cube, pa, waves, center, sep_pix, pa_deg, psf_stamps, D, pixscale, n_jobs,
            dict(npca=npca, n_arcs=n_arcs, dr=dr, exclusion_fwhm=exclusion_fwhm,
                 centering=centering, derot_sign=derot_sign, stamp_half=stamp_half,
                 forward_model=forward_model), verbose)
    h = stamp_half
    src_xy = (cx - sep_pix * np.sin(np.deg2rad(pa_deg)),
              cy + sep_pix * np.cos(np.deg2rad(pa_deg)))
    x0, y0 = int(round(src_xy[0])) - h, int(round(src_xy[1])) - h
    sy, sx = np.mgrid[y0:y0 + 2 * h + 1, x0:x0 + 2 * h + 1]

    yy, xx = np.indices((ny, nx), dtype=float)
    rr = np.hypot(xx - cx, yy - cy)
    phi_all = np.arctan2(yy - cy, xx - cx)
    in_ann = (rr >= sep_pix - dr) & (rr <= sep_pix + dr)

    # frame bookkeeping
    T_idx = np.repeat(np.arange(n_exp), n_wave)
    W_idx = np.tile(np.arange(n_wave), n_exp)
    Q, WV = pa[T_idx], waves[W_idx]
    n_all = T_idx.size
    thr = exclusion_fwhm * np.array([resolution_element_pix(w, D, pixscale) for w in WV])
    # raw-frame source position per exposure
    raw_xy = np.array([companion_frame_position(sep_pix, pa_deg, derot_sign * q, center)
                       for q in pa])

    if spectral_refs:
        data = np.asarray(cube, dtype=float).reshape(n_all, ny, nx)

    # unit-contrast source image per (exposure, channel), built once per channel
    _src_cache = {}

    def src_images(w):
        if w not in _src_cache:
            _src_cache[w] = np.array([place_psf((ny, nx), psf_stamps[w], xr, yr)
                                      for xr, yr in raw_xy])
        return _src_cache[w]

    P_sum = np.zeros((n_wave, ny, nx))
    P_cnt = np.zeros((n_wave, ny, nx))
    F_sum = np.zeros((n_wave, n_wave, 2 * h + 1, 2 * h + 1))
    F_cnt = np.zeros((n_wave, 2 * h + 1, 2 * h + 1))
    dlam = np.zeros(n_wave)
    mixing_frames = np.zeros((n_wave, n_exp))
    n_frames = np.zeros(n_wave, int)
    from scipy.ndimage import map_coordinates

    for p in range(n_wave):
        if not spectral_refs:
            chan = np.asarray(cube[:, p], dtype=float)          # (n_exp, ny, nx)
            _src_cache.clear()                                   # only channel p needed
        for t in range(n_exp):
            tgt = t * n_wave + p
            disp = adi_sdi_displacement(sep_pix, Q, WV, pa[t], waves[p])
            ok = (disp > thr) & (np.arange(n_all) != tgt)
            if not spectral_refs:
                ok &= (W_idx == p)
            refs = np.where(ok)[0]
            if refs.size < npca + 1:
                continue
            ref_frames = data[refs] if spectral_refs else chan[T_idx[refs]]
            tgt_frame = data[tgt] if spectral_refs else chan[t]

            # source-anchored zones for this target frame
            phi_s = np.arctan2(raw_xy[t, 1] - cy, raw_xy[t, 0] - cx)
            zone_id = (np.floor(np.mod(phi_all - phi_s + np.pi / n_arcs, 2 * np.pi)
                                / (2 * np.pi / n_arcs)).astype(int)) % n_arcs
            resid = np.full((ny, nx), np.nan)
            fm_img = np.zeros((n_wave, ny, nx)) if forward_model else None
            for z in range(n_arcs):
                m = in_ann & (zone_id == z)
                npx = int(m.sum())
                if npx < npca + 1:
                    continue
                lib = ref_frames[:, m]
                pca = PCA(lib, np.ones(npx, bool), method='svd', centering=centering)
                tpix = tgt_frame[m]
                resid[m] = pca(tpix.copy(), npca)
                if not forward_model:
                    continue
                # unit-contrast source model in the references and the target
                A_refs = np.empty((refs.size, npx))
                for w in np.unique(W_idx[refs]):
                    sel = W_idx[refs] == w
                    A_refs[sel] = src_images(w)[T_idx[refs[sel]]][:, m]
                A_tgt = src_images(p)[t][m]
                if not (A_refs.any() or A_tgt.any()):
                    continue
                w_refs = W_idx[refs]
                for q in np.unique(w_refs):
                    a_frames = (w_refs == q).astype(float)
                    if not (A_refs[a_frames > 0].any() or (q == p and A_tgt.any())):
                        continue
                    mt = A_tgt if q == p else np.zeros(npx)
                    fm, rel, mx = forward_model_psf(pca, A_refs, a_frames, mt, tpix, npca,
                                                    return_delta_lambda=True, return_mixing=True)
                    fm_img[q][m] = fm
                    dlam[q] = max(dlam[q], float(np.max(np.abs(rel))) if rel.size else 0.0)
                    if q == p:
                        mixing_frames[q, t] = max(mixing_frames[q, t], mx)
            ang = derot_sign * pa[t]
            dres = nan_rot(resid, ang, center, out_size=(ny, nx), pivot=True)
            good = np.isfinite(dres)
            P_sum[p][good] += dres[good]
            P_cnt[p][good] += 1
            n_frames[p] += 1
            if forward_model:
                # derotate the model only over the stamp: sample the raw-frame
                # image where each derotated stamp pixel came from
                ux, uy = get_unrot_coords(sx, sy, ang, center)
                gstamp = good[y0:y0 + 2 * h + 1, x0:x0 + 2 * h + 1] if (
                    y0 >= 0 and x0 >= 0 and y0 + 2 * h + 1 <= ny and x0 + 2 * h + 1 <= nx) \
                    else np.ones_like(sx, bool)
                F_cnt[p] += gstamp
                for q in range(n_wave):
                    if fm_img[q].any():
                        F_sum[p, q] += gstamp * map_coordinates(
                            fm_img[q], [uy, ux], order=3, mode='constant', cval=0.0)
        if verbose:
            print("  channel %d/%d reduced (%d frames)" % (p + 1, n_wave, n_frames[p]),
                  flush=True)

    with np.errstate(invalid='ignore', divide='ignore'):
        P = np.where(P_cnt > 0, P_sum / np.maximum(P_cnt, 1), np.nan)
        F = np.where(F_cnt[:, None] > 0, F_sum / np.maximum(F_cnt[:, None], 1), 0.0)
    return dict(P=P, F=F, stamp_origin=(x0, y0), src_xy=src_xy, dlam=dlam,
                mixing_frames=mixing_frames,
                n_frames=n_frames, waves=waves, sep_pix=sep_pix, pa_deg=pa_deg,
                center=center, npca=npca, centering=centering)


def _reduce_one_channel(args):
    cube_w, pa, wave, center, sep_pix, pa_deg, psf_w, D, pixscale, kw = args
    with _single_blas_thread():
        return klip_fm_reduce(cube_w, pa, wave, center, sep_pix, pa_deg, psf_w, D, pixscale,
                              spectral_refs=False, n_jobs=1, **kw)


def _klip_fm_reduce_parallel(cube, pa, waves, center, sep_pix, pa_deg, psf_stamps, D,
                             pixscale, n_jobs, kw, verbose):
    '''Channel-parallel ``klip_fm_reduce`` for ADI per channel (F is diagonal).'''
    from concurrent.futures import ProcessPoolExecutor
    n_wave = waves.size
    jobs = ((np.asarray(cube[:, w:w + 1], dtype=float), pa, waves[w:w + 1], center,
             sep_pix, pa_deg, psf_stamps[w:w + 1], D, pixscale, kw) for w in range(n_wave))
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        parts = list(ex.map(_reduce_one_channel, jobs))
    first = parts[0]
    hh = first['F'].shape[-1]
    F = np.zeros((n_wave, n_wave, hh, hh))
    for w, part in enumerate(parts):
        F[w, w] = part['F'][0, 0]
    out = dict(first)
    out.update(P=np.concatenate([p_['P'] for p_ in parts]), F=F,
               dlam=np.array([p_['dlam'][0] for p_ in parts]),
               mixing_frames=np.concatenate([p_['mixing_frames'] for p_ in parts]),
               n_frames=np.array([p_['n_frames'][0] for p_ in parts]), waves=waves)
    if verbose:
        print("  reduced %d channels on %d workers" % (n_wave, n_jobs), flush=True)
    return out


def _fit_at(red, fit_mask_of, dtheta=0.0, order=3):
    '''Least-squares spectrum at the source position rotated by ``dtheta``
    (radians, CCW on sky) about the star: minimize
    sum_p || P_p - sum_q f_q F[p, q] ||^2 over each channel's fit region.'''
    P, F = red['P'], red['F']
    x0, y0 = red['stamp_origin']
    n_wave = P.shape[0]
    rows_X, rows_y = [], []
    for p in range(n_wave):
        ys, xs = fit_mask_of(p, dtheta)
        if ys.size == 0:
            continue
        yv = P[p][ys, xs]
        good = np.isfinite(yv)
        if not good.any():
            continue
        cols = np.zeros((good.sum(), n_wave))
        for q in range(n_wave):
            if F[p, q].any():
                cols[:, q] = _sample_rotated(F[p, q], x0, y0, xs[good], ys[good],
                                             dtheta, red['center'], order=order)
        rows_X.append(cols)
        rows_y.append(yv[good])
    X = np.vstack(rows_X)
    y = np.concatenate(rows_y)
    used = np.any(X != 0, axis=0)
    f = np.full(n_wave, np.nan)
    sol, *_ = np.linalg.lstsq(X[:, used], y, rcond=None)
    f[used] = sol
    return f, X, y


def extract_spectrum(cube, nocoro, pa, waves, center, sep_pix, pa_deg, D, pixscale,
                     npca=20, n_arcs=4, dr=3.5, exclusion_fwhm=0.75, spectral_refs=False,
                     centering='frame', derot_sign=-1.0, fit_radius_fwhm=1.0,
                     psf_stamps=None, ring_exclude_fwhm=2.5, exclude=None,
                     corr_method='ring', shrink=0.3, small_sample=True,
                     refine=3, refine_tol=0.05, refine_dlam=0.05, reduction=None, n_jobs=1,
                     verbose=True):
    '''Extract a point source's spectrum with KLIP forward modeling (Pueyo 2016,
    Appendix F) and estimate its full covariance from the data.

    The spectrum is the least-squares solution of Eq. SimpleMat,

        f_hat = argmin_f  sum_p || P_p - sum_q f_q F_p[q] ||^2   over the fit region,

    where ``P_p`` is the reduced image in channel p and ``F_p[q]`` the KLIP-
    processed image of a unit-contrast source emitting at channel q (from
    ``klip_fm_reduce``). Because ``F`` carries the over- AND self-subtraction the
    estimate is corrected for the KLIP flux loss; because the PSF stamps are the
    unocculted star, ``f_hat`` is the companion/star flux ratio per channel.

    Uncertainties. Pueyo leaves confidence intervals to future work; here they are
    measured empirically with the SAME linear estimator: the model is rotated to
    empty positions around the ring at the source separation and fit to the data
    there (pure noise, since the estimator is linear). Positions spaced by one
    lambda/D (at the longest wavelength) give independent samples whose scatter
    is the per-channel 1-sigma ``sigma``; with ``small_sample`` it is inflated by
    the Mawet+2014 ``sqrt(1 + 1/N)`` factor. The channel-to-channel correlation
    (speckles are correlated across wavelength -- Greco & Brandt 2016) is taken
    from the same estimator evaluated densely around the ring
    (``corr_method='ring'``), shrunk toward the identity by ``shrink``. The full
    covariance is ``cov = D psi D`` with ``D = diag(sigma)``.

    Bright sources: the forward model is first order in the source flux, so a
    source that noticeably perturbs the KL basis (``dlam_fit`` approaching 1;
    for the SCALES example, contrasts >~1e-4 at 0.2-0.5") is under-corrected by
    one pass. With ``refine > 0`` the estimate is iterated by negative
    injection: the current spectrum is subtracted from the cube, the cube is
    re-reduced, and the first-order fit is applied to what remains -- which is
    faint, so the linear model is accurate for it -- until the update is below
    ``refine_tol`` of the spectrum (or ``refine`` passes). Each pass costs one
    reduction. The uncertainties come from the final pass.

    Centering: ``'frame'`` (Pueyo's convention, the default here) makes the
    forward model exact to first order; ``'median'`` leaves a neglected
    median-shift term (a few-percent flux bias on the SCALES data).

    Parameters are as in ``klip_fm_reduce`` / ``contrast_curve``, plus

    nocoro : ndarray (n_snap, n_wave, ny, nx)
        Unocculted exposure(s), e.g. snapshots at the start/end of the sequence;
        supplies the off-axis PSF (median over exposures) unless ``psf_stamps``
        is given. The PSF defines unit contrast, so it must be in the same flux
        units as ``cube``.
    fit_radius_fwhm : float
        Fit-region radius in lambda/D (per channel).
    ring_exclude_fwhm : float
        Noise-ring positions closer than this (lambda/D) to the source -- or to any
        position in ``exclude`` -- are skipped (keeps out the source and its
        self-subtraction lobes).
    exclude : list of (sep_pix, pa_deg) or None
        Other known sources to keep out of the noise ring.
    refine : int
        Maximum number of negative-injection refinement passes (0 = single pass).
        Needs ``cube``; ignored when only a ``reduction`` is supplied.
    refine_tol : float
        Stop when the median over channels of |update| / sigma falls below this.
        (A few channels can keep moving by ~1 sigma from pass to pass for a
        bright source -- tiny changes in the subtracted flux flip near-degenerate
        KL modes in a few exposures; they are reported in ``unstable_channels``
        and do not hold up convergence.)
    refine_dlam : float
        Only refine when the first pass's ``dlam_fit`` exceeds this; below it the
        single pass is already accurate (on the SCALES data, dlam ~0.03 -> 0.1%,
        ~0.1 -> <1% flux error) and a refinement pass would only cost time.
    reduction : dict or None
        A ``klip_fm_reduce`` result to reuse (skips the first reduction).
    n_jobs : int
        Worker processes for the reduction (see ``klip_fm_reduce``).

    Returns a dict with ``waves``; ``spectrum`` (contrast per channel); ``sigma``;
    ``cov`` and ``corr`` (n_wave x n_wave); ``n_ring`` (independent noise
    samples); ``throughput`` (per channel, model aperture flux / injected);
    ``dlam_fit`` (|delta Lambda/Lambda| at the fitted flux: should be << 1 for
    the first-order model to hold); ``broadband`` (dict: ``contrast``, ``sigma``,
    ``snr``: generalized-least-squares mean over channels); ``ring_spectra``;
    ``n_iter``, ``history`` (spectrum after each pass) and
    ``unstable_channels`` (still moving by > 1 sigma in the last pass); and the final
    ``reduction`` (data, model) for plotting. ``dlam_fit`` refers to the first
    pass (the full source in the data); ``dlam_residual`` to the last.
    ``mixing_fit`` (per channel) is the median over exposures of the cross-
    truncation mode mixing at the fitted flux -- a second first-order validity
    measure that, like ``dlam_fit``, tracks the forward-model error on the
    SCALES data (~0.03 -> <0.3%, ~0.1 -> ~1%, ~0.3 -> 3-5%). ``crossing_frac``
    is the fraction of exposures in which it exceeds 1: there a near-degenerate
    eigenvalue pair at the ``npca`` cut may swap, adding a speckle-sized error
    from that one frame (noise-like, not a bias; matters for faint sources near
    the star).'''
    waves = np.asarray(waves, float)
    n_wave = waves.size
    if psf_stamps is None:
        half = int(np.ceil(2 * resolution_element_pix(waves.max(), D, pixscale)))
        psf_stamps = offaxis_psf_stamps(nocoro, center, half=half)
    if reduction is None:
        reduction = klip_fm_reduce(cube, pa, waves, center, sep_pix, pa_deg, psf_stamps,
                                   D, pixscale, npca=npca, n_arcs=n_arcs, dr=dr,
                                   exclusion_fwhm=exclusion_fwhm,
                                   spectral_refs=spectral_refs, centering=centering,
                                   derot_sign=derot_sign, n_jobs=n_jobs, verbose=verbose)
    else:
        centering = reduction.get('centering', centering)
    red0 = red = reduction
    rkw = dict(npca=npca, n_arcs=n_arcs, dr=dr, exclusion_fwhm=exclusion_fwhm,
               spectral_refs=spectral_refs, centering=centering, derot_sign=derot_sign,
               n_jobs=n_jobs)
    cx, cy = center
    sx0, sy0 = red['src_xy']
    res = np.array([resolution_element_pix(w, D, pixscale) for w in waves])
    rfit = fit_radius_fwhm * res
    ny, nx = red['P'].shape[1:]

    def fit_mask_of(p, dtheta):
        c, s_ = np.cos(dtheta), np.sin(dtheta)
        xc = cx + c * (sx0 - cx) - s_ * (sy0 - cy)
        yc = cy + s_ * (sx0 - cx) + c * (sy0 - cy)
        r = rfit[p]
        ys, xs = np.mgrid[int(np.floor(yc - r)):int(np.ceil(yc + r)) + 1,
                          int(np.floor(xc - r)):int(np.ceil(xc + r)) + 1]
        keep = ((np.hypot(xs - xc, ys - yc) <= r) & (xs >= 0) & (ys >= 0)
                & (xs < nx) & (ys < ny))
        return ys[keep], xs[keep]

    # ---- empirical noise from the same estimator around the ring -------------
    excl = [(sep_pix, pa_deg)] + list(exclude or [])
    excl_xy = [(cx - s * np.sin(np.deg2rad(a)), cy + s * np.cos(np.deg2rad(a))) for s, a in excl]
    res_max = res.max()

    def ring_angles(step_pix):
        n = int(np.floor(2 * np.pi * sep_pix / step_pix))
        angs = 2 * np.pi * np.arange(1, n) / n
        keep = []
        for a in angs:
            c, s_ = np.cos(a), np.sin(a)
            xc = cx + c * (sx0 - cx) - s_ * (sy0 - cy)
            yc = cy + s_ * (sx0 - cx) + c * (sy0 - cy)
            if all(np.hypot(xc - ex, yc - ey) > ring_exclude_fwhm * res_max for ex, ey in excl_xy):
                keep.append(a)
        return np.array(keep)

    indep = ring_angles(res_max)

    def ring_sigma(reduction):
        ring = np.array([_fit_at(reduction, fit_mask_of, a)[0] for a in indep])
        return ring, np.nanstd(ring, axis=0, ddof=1)

    f_hat, X, y = _fit_at(red, fit_mask_of, 0.0)
    history = [f_hat.copy()]
    df = f_hat
    n_iter = 0
    unstable = np.zeros(n_wave, bool)
    needs_refine = np.nanmax(np.abs(np.nan_to_num(f_hat)) * red0['dlam']) > refine_dlam
    if refine and cube is not None and needs_refine:
        for n_iter in range(1, refine + 1):
            resid_cube = inject_source(cube, psf_stamps, pa, center, sep_pix, pa_deg,
                                       -np.nan_to_num(f_hat), derot_sign=derot_sign)
            red = klip_fm_reduce(resid_cube, pa, waves, center, sep_pix, pa_deg,
                                 psf_stamps, D, pixscale, verbose=False, **rkw)
            df, _, _ = _fit_at(red, fit_mask_of, 0.0)
            f_hat = f_hat + np.nan_to_num(df)
            history.append(f_hat.copy())
            _, sig_pass = ring_sigma(red)
            step = np.abs(df) / sig_pass
            unstable = step > 1.0
            if verbose:
                print("  refinement pass %d: median |update|/sigma = %.3f (%d channels > 1 sigma)"
                      % (n_iter, np.nanmedian(step), unstable.sum()), flush=True)
            if np.nanmedian(step) < refine_tol:
                break

    indep = ring_angles(res_max)
    ring_indep, sigma = ring_sigma(red)
    n_ring = ring_indep.shape[0]
    if small_sample and n_ring > 1:
        sigma = sigma * np.sqrt(1.0 + 1.0 / n_ring)
    corr_params = None
    if corr_method in ('ring', 'gb16'):
        dense = ring_angles(1.0)
        ring_dense = np.array([_fit_at(red, fit_mask_of, a)[0] for a in dense])
        ok = np.all(np.isfinite(ring_dense), axis=0)
        corr = np.eye(n_wave)
        if ok.sum() > 1:
            if corr_method == 'ring':
                sub = spectral_correlation(ring_dense[:, ok], shrink=shrink)
            else:
                emp = spectral_correlation(ring_dense[:, ok], shrink=0.0)
                sub, corr_params = fit_gb16_correlation(emp, waves[ok])
            corr[np.ix_(ok, ok)] = sub
    elif corr_method == 'none':
        corr, ring_dense = np.eye(n_wave), None
    else:
        raise ValueError("corr_method must be 'gb16', 'ring' or 'none'")
    cov = corr * np.outer(sigma, sigma)

    # ---- diagnostics ---------------------------------------------------------
    x0, y0 = red['stamp_origin']
    thru = np.full(n_wave, np.nan)
    for p in range(n_wave):
        st = psf_stamps[p]
        h0 = (st.shape[0] - 1) / 2.0
        inj = _aperture_flux(st, h0, h0, 0.5 * res[p])
        fm = _aperture_flux(red['F'][p, p], sx0 - x0, sy0 - y0, 0.5 * res[p])
        thru[p] = fm / inj if inj > 0 else np.nan
    dlam_fit = np.abs(np.nan_to_num(f_hat)) * red0['dlam']
    dlam_residual = np.abs(np.nan_to_num(df)) * red['dlam']
    mix_frames = np.abs(np.nan_to_num(f_hat))[:, None] * red0['mixing_frames']
    mixing_fit = np.median(mix_frames, axis=1)
    crossing_frac = np.mean(mix_frames > 1.0, axis=1)

    out = dict(waves=waves, spectrum=f_hat, sigma=sigma, cov=cov, corr=corr,
               n_ring=n_ring, ring_spectra=ring_indep, throughput=thru,
               dlam_fit=dlam_fit, dlam_residual=dlam_residual, mixing_fit=mixing_fit,
               crossing_frac=crossing_frac, unstable_channels=np.where(unstable)[0],
               n_iter=n_iter,
               corr_params=corr_params, ring_dense=ring_dense,
               history=np.array(history), reduction=red, fit_radius_pix=rfit)
    out['broadband'] = broadband_contrast(out)
    return out


def broadband_contrast(result, spectrum_shape=None, weights='ivar', small_sample=True):
    '''Amplitude of a spectral shape fit to an extracted spectrum, with an
    empirically calibrated uncertainty and significance.

    The amplitude is a weighted least-squares fit of shape ``s`` (default flat) to
    the spectrum ``f``: ``A = w^T f`` with ``w^T s = 1`` and

        weights='ivar':  w ~ s / sigma^2           (inverse-variance; default)
        weights='gls':   w ~ C^-1 s                 (generalized least squares)

    Its uncertainty is measured, not propagated: the same weights are applied to
    the noise-only spectra at the independent ring positions
    (``result['ring_spectra']``) and their scatter is the 1-sigma error (times
    the Mawet+2014 ``sqrt(1 + 1/N)`` with ``small_sample``). This captures the
    strong correlation of speckle noise across wavelength, which makes the
    independent-channel error ``(sum s^2/sigma^2)^-1/2`` several times too small.
    Inverse-variance weights are the default because GLS weights built from a
    covariance estimated on those same few ring positions are tuned to them (an
    in-sample overfit): their ring scatter then understates the noise at a new
    position, badly so near the star where there are only ~10-15 positions for
    ~50 channels (validation/, V2b). ``sigma_cov`` is the covariance-propagated
    error for comparison.

    ``chi2`` = (f - A s)^T C^-1 (f - A s) on n-1 degrees of freedom compares the
    shape with the data using the full covariance ``result['cov']`` (calibrated
    on average on the SCALES validation data for the default shrinkage).

    Returns dict(contrast, sigma, snr, sigma_cov, chi2, dof, n_ring).'''
    f = np.asarray(result['spectrum'], float)
    C = np.asarray(result['cov'], float)
    s = np.ones_like(f) if spectrum_shape is None else np.asarray(spectrum_shape, float)
    ok = np.isfinite(f) & np.isfinite(np.diag(C)) & (np.diag(C) > 0)
    f, s, C = f[ok], s[ok], C[np.ix_(ok, ok)]
    try:
        Ci = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        Ci = np.linalg.pinv(C)
    if weights == 'gls':
        w = Ci @ s
    elif weights == 'ivar':
        w = s / np.diag(C)
    else:
        raise ValueError("weights must be 'ivar' or 'gls'")
    w = w / float(w @ s)
    A = float(w @ f)
    A_gls = float(s @ Ci @ f) / float(s @ Ci @ s)
    r = f - A_gls * s
    chi2 = float(r @ Ci @ r)
    sigma_cov = float(np.sqrt(w @ C @ w))
    ring = result.get('ring_spectra')
    n_ring = 0
    sigma = sigma_cov
    if ring is not None and len(ring) > 2:
        ring = np.asarray(ring, float)[:, ok]
        good = np.all(np.isfinite(ring), axis=1)
        amps = ring[good] @ w
        n_ring = amps.size
        if n_ring > 2:
            sigma = float(np.std(amps, ddof=1))
            if small_sample:
                sigma *= np.sqrt(1.0 + 1.0 / n_ring)
    return dict(contrast=A, sigma=sigma, snr=A / sigma, sigma_cov=sigma_cov, chi2=chi2,
                dof=int(ok.sum()) - 1, n_ring=n_ring)
