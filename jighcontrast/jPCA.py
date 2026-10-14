import numpy as np
from astropy.io import fits

_EIGH_HINT_SHOWN = False

def _maybe_suggest_eigh(n_images, n_pixels):
    '''Print a one-time hint when a rectangular region makes ``method='eigh'``
    worth trying.

    The reliable speed-up from ``eigh`` is dimensional: it eigendecomposes the
    smaller ``min(M, N) x min(M, N)`` covariance and never forms the larger
    dimension's singular vectors, so it wins when one of M (images) and N
    (optimization pixels) is well above the other. We fire at an aspect ratio of
    at least 5:1 (with the larger dimension non-trivial), where the advantage is
    dimensional and dependable. Near-square regions -- where any gain is only a
    small, implementation-dependent constant we have no data for -- stay silent.
    It is only a hint: ``eigh`` forms a covariance (Gram) matrix, which squares
    its condition number and can cost a few digits of precision, so the message
    says to keep ``svd`` when stability matters.'''
    global _EIGH_HINT_SHOWN
    if _EIGH_HINT_SHOWN:
        return
    import multiprocessing
    if multiprocessing.parent_process() is not None:
        return          # a worker process (e.g. jFM n_jobs > 1): leave hints to the main one
    lo, hi = min(n_images, n_pixels), max(n_images, n_pixels)
    if lo >= 1 and hi >= 5 * lo and hi >= 100:
        print("jPCA.PCA: strongly rectangular optimization region (%d images x "
              "%d pixels, %.0f:1). method='eigh' decomposes the smaller %dx%d "
              "covariance and skips the larger dimension, so it is usually "
              "faster here; over the many PCA builds in an ADI/SDI reduction "
              "(and repeated injection-recovery runs) that saving compounds. "
              "Trade-off: eigh forms a covariance (Gram) matrix, squaring its "
              "condition number and possibly costing a few digits of precision "
              "-- keep the default 'svd' if you need maximum numerical stability "
              "or push npca deep into the small-eigenvalue tail. (this hint "
              "prints once per session)"
              % (n_images, n_pixels, hi / lo, lo, lo))
        _EIGH_HINT_SHOWN = True


class PCA:
    '''Principal-component (KLIP-style) PSF subtraction.

    Build the principal components of a library of reference images (over the
    pixels selected by ``opt_mask``), then project a target image onto the
    first ``npca`` components and subtract that reconstruction to reveal faint
    off-axis sources.'''

    def __init__(self, library_cube, opt_mask, method='svd', centering='median'):
        '''Compute the principal components of a reference-image library.

        library_cube : ndarray (n_images, ydim, xdim)
            Stack of reference PSF images.
        opt_mask : boolean ndarray (ydim, xdim)
            Pixels to include in the PCA (the optimization region).
        method : {'svd', 'eigh'}
            How to compute the principal components of the image library (the
            centered reference images, arranged as an n_images x n_pixels
            matrix). Both options give the same components (up to a sign); they
            trade speed against numerical conditioning:

            'svd'  -- thin singular-value decomposition of the image library
                      (``full_matrices=False``). Returns exactly the
                      ``min(n_images, n_pixels)`` meaningful components and is the
                      most numerically robust choice.
            'eigh' -- eigendecomposition of the smaller covariance matrix: the
                      image-to-image covariance (n_images x n_images) when there
                      are fewer images than pixels, otherwise the pixel-to-pixel
                      covariance (n_pixels x n_pixels), mapped back into pixel
                      space. Fastest when one dimension greatly exceeds the other,
                      at the cost of a few digits of precision (it squares the
                      matrix condition number).
        centering : {'median', 'frame'}
            How the library is centered before the KL transform. Both remove a
            reference level so the components describe fluctuations, but they
            differ in what they remove -- and this choice interacts with KLIP-FM
            (see ``jFM.forward_model_psf``):

            'median' -- subtract the per-pixel *median over frames* (a shared
                        spatial image). This is a pooled ensemble statistic, so
                        an injected source leaks into it; the forward model
                        neglects that leakage (valid for a median combine of an
                        azimuthally transient source). It also does not remove a
                        per-frame DC/background level, so with a variable
                        background the rows keep non-zero, frame-varying means and
                        the KL basis spends a mode on that variation.
            'frame'  -- subtract each frame's *own mean over the zone* (a scalar
                        per row), the convention of Pueyo (2016) PCA step 1
                        (ms10.tex:579): T and every R_k are zero-mean over the
                        subtraction zone. This is a per-image linear functional
                        (not pooled across frames), so there is no source-leakage
                        term to neglect and it removes a spatially-flat variable
                        background exactly. It is the convention the Appendix E/F
                        forward model is derived for.'''
        self.opt_mask = opt_mask
        self.method = method
        self.centering = centering
        # Pull the optimization-region pixels out of every reference image and
        # center the library (one row per image, one column per optimization-
        # region pixel). See the ``centering`` docstring for the two conventions.
        library = library_cube[:,opt_mask]
        if centering == 'frame':
            # Pueyo step 1: zero-mean each frame over the zone (a scalar per row).
            # There is no shared image to subtract from the target later, so the
            # stored ``median_image`` is zero and __call__ removes the target's
            # own zone-mean instead.
            self.row_mean = library.mean(axis=1, keepdims=True)   # (n_images, 1)
            library = library - self.row_mean
            self.median_image = np.zeros(library.shape[1], dtype=float)
        elif centering == 'median':
            self.median_image = np.median(library, axis=0)
            library = library - self.median_image[None,:]
        else:
            raise ValueError("centering must be 'median' or 'frame', got %r" % (centering,))
        self.library = library  # centered reference frames S (n_images, n_pixels)
        self.A = library.T  # kept for backwards compatibility (images in columns)
        # Frame-space quantities used by KLIP forward modeling (jFM). The SVD path
        # populates these directly; the eigh path fills what it can (see below).
        # In Pueyo (2016) notation: left_vectors are the frame-space eigenvectors
        # V_k, eigenvalues are Lambda_k, eigenvectors are the pixel-space KL images
        # Z_k, and library is the signal-free reference matrix S.
        self.left_vectors = None      # V_k, shape (n_images, n_components)
        self.singular_values = None   # sqrt(Lambda_k), shape (n_components,)
        #////////////////////////////////////////////////////////
        # Find the principal components, i.e. the eigen-images of the image
        # library's covariance. (G. Strang on using the SVD for this:
        # https://www.youtube.com/watch?v=mBcLRGuAFUk )
        #////////////////////////////////////////////////////////
        if method == 'svd':
            # Thin SVD: the left singular vectors and the null-space padding are
            # never used, so full_matrices=False is identical for every component
            # that __call__ touches and far cheaper for large regions.
            _maybe_suggest_eigh(*library.shape)  # one-time speed hint for large PCA builds
            eigen_images, sqrt_eigvals, eigen_pixels_T = np.linalg.svd(library, full_matrices=False)
            self.eigenvalues = sqrt_eigvals**2
            self.eigenvectors = eigen_pixels_T.T  # already sorted and normalized
            # SVD of S = V Sigma Z^T gives the frame-space vectors V_k (left) and
            # the singular values sqrt(Lambda_k) for free -- forward modeling needs
            # them and this is the numerically robust way to get them (no squared
            # covariance, so the Lambda_k - Lambda_j gaps in Eq. 50 stay well
            # conditioned). This is why jFM prefers method='svd'.
            self.left_vectors = eigen_images
            self.singular_values = sqrt_eigvals
        elif method == 'eigh':
            self.eigenvalues, self.eigenvectors = _eigh_components(library)
            # Recover the frame-space vectors from the bridge S^T V_k = sqrt(Lambda_k) Z_k
            # -> V_k = S Z_k / sqrt(Lambda_k), so jFM can still forward-model from an
            # eigh-built basis (with the precision caveat noted for 'eigh').
            self.singular_values = np.sqrt(np.clip(self.eigenvalues, 0.0, None))
            safe = self.singular_values > 0
            self.left_vectors = np.zeros((library.shape[0], self.eigenvectors.shape[1]),
                                         dtype=library.dtype)
            self.left_vectors[:, safe] = (library @ self.eigenvectors[:, safe]) / self.singular_values[safe]
        else:
            raise ValueError("method must be 'svd' or 'eigh', got %r" % (method,))

    def __call__(self, image, npca):
        '''Subtract the PCA model PSF from ``image`` and return the residual.

        image : ndarray (ydim, xdim)
            Target image; modified in place over ``opt_mask``.
        npca : int
            Number of principal components to project out.

        Returns the residual pixels within ``opt_mask`` (1-D ndarray).'''
        if self.centering == 'frame':
            # Match Pueyo step 1: zero-mean the target over the zone.
            image[self.opt_mask] -= image[self.opt_mask].mean()
        else:
            image[self.opt_mask] -= self.median_image
        for k in range(npca):
            coefficient = np.dot(self.eigenvectors[:,k],image[self.opt_mask])
            image[self.opt_mask] -= (self.eigenvectors[:,k]) * coefficient
        return image[self.opt_mask]


def _eigh_components(library):
    '''Principal components via the smaller covariance-matrix eigendecomposition.

    library : ndarray (n_images, n_pixels)
        The median-subtracted image library.

    Returns (eigenvalues, eigenvectors) sorted by descending eigenvalue.
    ``eigenvectors`` has shape (n_pixels, n_components) with unit-norm columns --
    the same pixel-space components the SVD path returns (up to a sign), with
    n_components = min(n_images, n_pixels).'''
    n_images, n_pixels = library.shape
    if n_images <= n_pixels:
        # Fewer images than pixels: eigendecompose the small image-to-image
        # covariance. If library = U S V.T then library @ library.T = U S^2 U.T,
        # so its eigenvectors are the left vectors U and its eigenvalues are the
        # squared singular values. Map to pixel space via V = library.T U / s.
        covariance = library @ library.T                      # (n_images, n_images)
        eigenvalues, left_vectors = np.linalg.eigh(covariance)  # ascending
        eigenvalues = eigenvalues[::-1]                       # -> descending
        left_vectors = left_vectors[:, ::-1]
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        sing = np.sqrt(eigenvalues)
        tol = np.finfo(library.dtype).eps * max(library.shape) * (sing[0] if sing.size else 0.0)
        safe = sing > tol                                     # skip null-space modes
        components = np.zeros((n_pixels, n_images), dtype=library.dtype)
        components[:, safe] = (library.T @ left_vectors[:, safe]) / sing[safe]
    else:
        # More images than pixels: the small matrix is the pixel-to-pixel
        # covariance library.T @ library = V S^2 V.T, whose eigenvectors ARE the
        # pixel-space components V directly (no division needed).
        covariance = library.T @ library                     # (n_pixels, n_pixels)
        eigenvalues, components = np.linalg.eigh(covariance)
        eigenvalues = np.clip(eigenvalues[::-1], 0.0, None)
        components = components[:, ::-1]
    return eigenvalues, components


def wrap_degrees(angle_deg):
    '''Wrap an angle difference into [-180, 180) degrees, so that e.g. 179 and
    -179 are 2 degrees apart (parallactic angles wrap through +/-180 for targets
    transiting near the zenith).'''
    return np.mod(np.asarray(angle_deg, dtype=float) + 180.0, 360.0) - 180.0


def adi_sdi_displacement(rad_in_pix, q, wv, ref_q, ref_wv):
    '''Apparent displacement (pixels) of a source between a reference frame and a
    candidate frame, from the combined ADI + SDI geometry.

    A source at separation ``rad_in_pix`` in the reference frame (parallactic
    angle ``ref_q``, wavelength ``ref_wv``) moves in a candidate frame by two
    effects:

    * SDI radial term -- the speckle field (and the modeled source position)
      scales radially with wavelength, so the source sits at ``rad_in_pix *
      ref_wv / wv``; the *displacement* is the difference from ``rad_in_pix``,
      i.e. ``rad_in_pix * (ref_wv / wv - 1)``.
    * ADI azimuthal term -- field rotation moves the source along the arc by
      ``rad_in_pix * (q - ref_q)`` (arc length, radians), with the angle
      difference wrapped into [-180, 180) degrees (``wrap_degrees``).

    The two are combined in quadrature. ``q``/``wv`` may be arrays.

    Returns the displacement in pixels (same shape as ``q``/``wv``).'''
    q = np.asarray(q, dtype=float)
    wv = np.asarray(wv, dtype=float)
    delta_r = rad_in_pix * (ref_wv / wv - 1.0)               # SDI radial motion
    delta_phi = rad_in_pix * np.deg2rad(wrap_degrees(q - ref_q))   # ADI azimuthal motion
    return np.hypot(delta_r, delta_phi)


def resolution_element_pix(wavelength_um, D_m, pixscale_mas):
    '''Size of one resolution element (lambda/D) in pixels.'''
    lam_over_d_rad = (wavelength_um * 1e-6) / D_m
    lam_over_d_mas = lam_over_d_rad * 206264806.0
    return lam_over_d_mas / pixscale_mas


def make_adi_sdi_library(cubes_list, paras_arr, waves_arr,
                 rad_in_pix, ref_q, ref_wv, buffer_radius=2,
                 exclusion_fwhm=None, D=None, pixscale=None):
    '''Select combined ADI+SDI reference frames for a target at a given radius.

    For each frame, estimate how far a source at radius ``rad_in_pix`` would move
    relative to the reference frame -- from the wavelength scaling (SDI) and the
    parallactic-angle change (ADI), combined in quadrature by
    ``adi_sdi_displacement`` -- and keep only frames whose displacement is large
    enough that a real companion is not self-subtracted.

    The acceptance threshold can be set two ways:

    * ``exclusion_fwhm`` (preferred, physical) -- keep frames whose displacement
      exceeds ``exclusion_fwhm`` resolution elements (lambda/D). Requires ``D``
      (meters) and ``pixscale`` (mas/pixel); the resolution element is evaluated
      per candidate frame at its own wavelength. This makes the exclusion zone a
      fixed fraction of the PSF width for both the ADI and SDI motions.
    * ``buffer_radius`` (pixels) -- used when ``exclusion_fwhm`` is None; keep
      frames whose displacement exceeds this many pixels.

    cubes_list : sequence of ndarrays (n_frames, ydim, xdim)
        One spectral cube per exposure.
    paras_arr : sequence of float
        Parallactic angle (degrees) for each cube.
    waves_arr : sequence of float
        Wavelength for each frame within a cube.
    rad_in_pix : float
        Separation of the region of interest, in pixels.
    ref_q, ref_wv : float
        Parallactic angle and wavelength of the reference frame.
    buffer_radius : float
        Minimum displacement (pixels) to accept a frame (used only when
        ``exclusion_fwhm`` is None).
    exclusion_fwhm : float or None
        Minimum displacement in resolution elements (lambda/D). If given, ``D``
        and ``pixscale`` are required and ``buffer_radius`` is ignored.
    D : float or None
        Telescope diameter (meters), for ``exclusion_fwhm``.
    pixscale : float or None
        Plate scale (mas/pixel), for ``exclusion_fwhm``.

    Returns (frames, parallactic_angles, wavelengths) as ndarrays.'''
    if exclusion_fwhm is not None and (D is None or pixscale is None):
        raise ValueError("exclusion_fwhm requires both D and pixscale")
    good_frames = []
    good_qs = []
    good_waves = []
    for cube, q in zip(cubes_list, paras_arr):
        for frame, wv in zip(cube, waves_arr):
            dpix = adi_sdi_displacement(rad_in_pix, q, wv, ref_q, ref_wv)
            if exclusion_fwhm is not None:
                threshold = exclusion_fwhm * resolution_element_pix(wv, D, pixscale)
            else:
                threshold = buffer_radius
            if dpix > threshold:
                good_frames.append(frame)
                good_qs.append(q)
                good_waves.append(wv)
    return np.array(good_frames), np.array(good_qs), np.array(good_waves)

def make_adi_library(im_list, paras_arr,
                     rad_in_pix, ref_q, buffer_radius=4):
    '''Select angular-differential-imaging (ADI) reference frames.

    Keep only frames for which a source at radius ``rad_in_pix`` has rotated
    more than ``buffer_radius`` pixels away from its position in the reference
    frame (parallactic angle ``ref_q``), so a real companion is preserved
    rather than self-subtracted.

    im_list : sequence of ndarrays (ydim, xdim)
        Individual images.
    paras_arr : ndarray of float
        Parallactic angle (degrees) for each image (not modified; the returned
        angles are wrapped into [0, 360)).
    rad_in_pix : float
        Separation of the region of interest, in pixels.
    ref_q : float
        Parallactic angle (degrees) of the reference frame.
    buffer_radius : float
        Minimum arc displacement (pixels) required to accept a reference frame.

    Returns (frames, parallactic_angles) as ndarrays.'''
    paras_arr = np.mod(np.asarray(paras_arr, dtype=float), 360.0)
    good_frames = []
    good_qs = []
    for im, q in zip(im_list, paras_arr):
        # arc displacement from the reference angle, using the wrapped difference so
        # that a frame rotated by -1 deg is 1 deg away (not 359)
        dpix = rad_in_pix * np.deg2rad(wrap_degrees(q - ref_q))
        if np.abs(dpix) > buffer_radius:
            good_frames.append(im)
            good_qs.append(q)
    return np.array(good_frames), np.array(good_qs)

def get_sector_inds(sec_center, dr, dphi, arr_shape, arr_center):
    '''Return a boolean mask selecting one annular sector of an image.

    Selects the pixels within +/- ``dr`` in radius and +/- ``dphi`` in azimuth
    of the sector center, measured about ``arr_center`` -- i.e. one wedge of an
    annulus, as used to define per-region PCA subtraction zones.

    sec_center : (x, y)
        Center of the sector.
    dr : float
        Half-width of the annulus, in pixels.
    dphi : float
        Half-width of the angular extent of the sector, in radians.
    arr_shape : (ydim, xdim)
        Shape of the image the mask is built for.
    arr_center : (x, y)
        Pixel coordinate where the primary star is centered.

    Returns a boolean ndarray of shape ``arr_shape``.'''
    sec_center = [int(np.rint(coord)) for coord in sec_center]
    yy, xx = np.indices(arr_shape, dtype=float)
    yy -= arr_center[1]
    xx -= arr_center[0]
    zz = yy+1j*xx
    rr = np.abs(zz)
    phi = np.angle(zz)
    rr0 = rr[sec_center[1], sec_center[0]]
    phi0 = phi[sec_center[1], sec_center[0]]

    r_dist = np.abs(rr-rr0)
    phi_dist = np.abs(phi-phi0)

    return np.logical_and(r_dist <= dr, phi_dist <= dphi)

