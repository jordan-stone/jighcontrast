import numpy as np
from astropy.io import fits

class PCA:
    '''Principal-component (KLIP-style) PSF subtraction.

    Build the principal components of a library of reference images (over the
    pixels selected by ``opt_mask``), then project a target image onto the
    first ``npca`` components and subtract that reconstruction to reveal faint
    off-axis sources.'''

    def __init__(self, library_cube, opt_mask, method='svd'):
        '''Compute the principal components of a reference-image library.

        library_cube : ndarray (n_images, ydim, xdim)
            Stack of reference PSF images.
        opt_mask : boolean ndarray (ydim, xdim)
            Pixels to include in the PCA (the optimization region).
        method : {'svd', 'eigh'}
            How to compute the principal components of the image library (the
            median-subtracted reference images, arranged as an
            n_images x n_pixels matrix). Both options give the same components
            (up to a sign); they trade speed against numerical conditioning:

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
                      matrix condition number).'''
        self.opt_mask = opt_mask
        self.method = method
        # Pull the optimization-region pixels out of every reference image and
        # subtract the per-pixel median, giving the mean-removed image library
        # (one row per image, one column per optimization-region pixel).
        library = library_cube[:,opt_mask]
        self.median_image = np.median(library, axis=0)
        library = library - self.median_image[None,:]
        self.A = library.T  # kept for backwards compatibility (images in columns)
        #////////////////////////////////////////////////////////
        # Find the principal components, i.e. the eigen-images of the image
        # library's covariance. (G. Strang on using the SVD for this:
        # https://www.youtube.com/watch?v=mBcLRGuAFUk )
        #////////////////////////////////////////////////////////
        if method == 'svd':
            # Thin SVD: the left singular vectors and the null-space padding are
            # never used, so full_matrices=False is identical for every component
            # that __call__ touches and far cheaper for large regions.
            eigen_images, sqrt_eigvals, eigen_pixels_T = np.linalg.svd(library, full_matrices=False)
            self.eigenvalues = sqrt_eigvals**2
            self.eigenvectors = eigen_pixels_T.T  # already sorted and normalized
        elif method == 'eigh':
            self.eigenvalues, self.eigenvectors = _eigh_components(library)
        else:
            raise ValueError("method must be 'svd' or 'eigh', got %r" % (method,))

    def __call__(self, image, npca):
        '''Subtract the PCA model PSF from ``image`` and return the residual.

        image : ndarray (ydim, xdim)
            Target image; modified in place over ``opt_mask``.
        npca : int
            Number of principal components to project out.

        Returns the residual pixels within ``opt_mask`` (1-D ndarray).'''
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


def make_adi_sdi_library(cubes_list, paras_arr, waves_arr,
                 rad_in_pix, ref_q, ref_wv, buffer_radius=2):
    '''Select combined ADI+SDI reference frames for a target at a given radius.

    For each frame, estimate how far a source at radius ``rad_in_pix`` would
    move relative to the reference frame -- from the wavelength scaling
    (SDI, ``ref_wv/wv``) and the parallactic-angle change (ADI, ``q-ref_q``) --
    and keep only frames whose total displacement exceeds ``buffer_radius``
    pixels, so a real companion is not self-subtracted.

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
        Minimum displacement (pixels) required to accept a reference frame.

    Returns (frames, parallactic_angles, wavelengths) as ndarrays.'''
    good_frames = []
    good_qs = []
    good_waves = []
    for cube, q in zip(cubes_list, paras_arr):
        for frame, wv in zip(cube, waves_arr):
            dr_pix = rad_in_pix * (ref_wv/wv)
            dphi_pix = rad_in_pix*(ref_wv/wv) * np.deg2rad(q-ref_q)
            dpix = np.sqrt(dr_pix**2 + dphi_pix**2)
            if dpix > buffer_radius:
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
        Parallactic angle (degrees) for each image; wrapped into [0, 360).
    rad_in_pix : float
        Separation of the region of interest, in pixels.
    ref_q : float
        Parallactic angle (degrees) of the reference frame.
    buffer_radius : float
        Minimum arc displacement (pixels) required to accept a reference frame.

    Returns (frames, parallactic_angles) as ndarrays.'''
    paras_arr+=360
    paras_arr%=360
    good_frames = []
    good_qs = []
    for im, q in zip(im_list, paras_arr):
        dpix = rad_in_pix * np.mod(np.deg2rad(q-ref_q), 2*np.pi)
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

