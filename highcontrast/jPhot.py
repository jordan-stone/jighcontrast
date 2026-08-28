import numpy as np
from scipy.interpolate import interp2d
from scipy.ndimage import shift, map_coordinates, gaussian_filter
from scipy.signal import convolve2d, fftconvolve
from .gaussfit import gaussfit, twodgaussian
from .nan_rot import nan_rot
import matplotlib.pyplot as plt
import jfits

class PSFAdder:
    '''Inject a scaled copy of a measured PSF into an image at sub-pixel positions.

    The reference PSF is modeled as a circular Gaussian core plus the fit
    residuals. Calling the instance adds that model, scaled in flux and shifted
    to an arbitrary (x, y), into a target image -- useful for injecting fake
    companions to measure throughput / contrast.'''

    def __init__(self, psf_array, half_size=10, gauss_sig=2):
        '''Fit the reference PSF stamp used for later injection.

        psf_array : ndarray (ydim, xdim)
            Image containing an isolated PSF; the brightest smoothed pixel is
            taken as its center.
        half_size : int
            Half-width (pixels) of the square stamp extracted around the peak.
        gauss_sig : float
            Initial guess for the Gaussian sigma of the PSF core.'''
        sm_im = gaussian_filter(psf_array, 3)
        mx = sm_im.max()
        y, x = np.transpose((sm_im == mx).nonzero())[0]

        self.half_size = half_size
        im_stamp = psf_array[y-self.half_size:y+self.half_size+1, 
                             x-self.half_size:x+self.half_size+1].astype(float)

        gauss_amp = im_stamp[self.half_size,self.half_size]#flux at peak position
        
        gauss_part0, model = gaussfit(im_stamp.astype(float), 
                                    params=[gauss_amp, 
                                            float(self.half_size), 
                                            float(self.half_size), 
                                            gauss_sig],
                                     circle=True,
                                     vheight=False,
                                     returnfitimage=True)
        self.gauss_part0 = gauss_part0
        self.gauss_amp = gauss_part0[1]
        self.gauss_sig = gauss_part0[4]
        #now center residuals
        shx = self.half_size - gauss_part0[2]
        shy = self.half_size - gauss_part0[3] 
        resids0 = im_stamp - model
        self.resids = shift(resids0, (shy, shx))

    def __call__(self, image, x, y, scale_factor):
        '''Return a copy of ``image`` with the model PSF added at (x, y).

        image : ndarray (ydim, xdim)
            Target image to inject into; not modified (a copy is returned).
        x, y : float
            Sub-pixel pixel coordinates at which to place the PSF center.
        scale_factor : float
            Multiplicative flux scaling applied to the injected PSF.'''
        x_int = int(np.floor(x))
        y_int = int(np.floor(y))
        x_remainder = x % 1
        y_remainder = y % 1

        resids = scale_factor*shift(self.resids, (y_remainder, x_remainder))

        these_gauss_params = [scale_factor * self.gauss_amp, 
                              self.half_size + x_remainder, 
                              self.half_size + y_remainder, 
                              self.gauss_sig]

        psf_out = resids + twodgaussian(these_gauss_params,
                                        circle=True,  
                                        vheight=False, 
                                        shape=resids.shape)

        im_out = image.copy()
        #smaller by 1 pixel along each edge to make up for shift
        im_out[y_int-self.half_size+1:y_int+self.half_size,
               x_int-self.half_size+1:x_int+self.half_size] += psf_out[1:-1,1:-1]
        return im_out



def get_unrot_coords(finalx, finaly, q, axis):
    '''Map a coordinate in a derotated image back to its raw (unrotated) position.

    Given a pixel in an image that has been rotated by ``q`` degrees about
    ``axis``, return where that pixel came from in the original frame -- useful
    for reading raw-frame values at a location identified in the derotated image.

    finalx, finaly : float
        Pixel coordinates in the derotated image.
    q : float
        Rotation angle applied to make the derotated image, degrees.
    axis : (x, y)
        Pixel coordinate of the rotation center.

    Returns the (x, y) coordinates in the original, unrotated frame.'''
    final_pos = (finalx-axis[0])+1j*(finaly-axis[1])
    original_pos = np.exp(1j*np.deg2rad(q))*final_pos
    return original_pos.real+axis[0], original_pos.imag+axis[1]


def _disk_kernel(radius):
    '''Boolean top-hat (disk) kernel of the given radius, with unit weights.'''
    half = int(np.ceil(radius))
    yy, xx = np.indices((2 * half + 1, 2 * half + 1))
    return (np.hypot(xx - half, yy - half) <= radius).astype(float)


def _gaussian_kernel(sigma):
    '''Circular Gaussian kernel, truncated at 3 sigma.'''
    half = int(np.ceil(3 * sigma))
    yy, xx = np.indices((2 * half + 1, 2 * half + 1))
    rr = np.hypot(xx - half, yy - half)
    kernel = np.exp(-(rr ** 2) / (2 * sigma ** 2))
    kernel[rr > 3 * sigma] = 0.0
    return kernel


def get_noise(image, x, y, center_pix, wavelength, D, pixscale,
              filter='tophat', psf_template=None, include=False, show=False):
    '''Measure the signal and noise at a source position in a reduced image.

    Puts the source at pixel ``(x, y)`` and lays down a ring of non-overlapping
    resolution-element apertures at the same separation from ``center_pix``. The
    photometry in the aperture on the source is the signal; the standard
    deviation of the photometry in the other apertures is the noise; their ratio
    is the signal-to-noise. This is the standard "apertures around a ring"
    high-contrast estimator.

    image : ndarray (ydim, xdim)
        The reduced image to measure. Non-finite pixels are treated as zero.
    x, y : float
        Pixel coordinates of the source.
    center_pix : (x, y)
        Pixel coordinate of the primary star (centre of the ring).
    wavelength : float, microns.  D : float, meters.  pixscale : float, mas/pixel.
        Set the resolution-element size lambda/D used for the aperture radius and
        spacing.
    filter : {'tophat', 'gaussian', 'matched'}
        Photometric weighting applied within each aperture:
        'tophat'   -- uniform weights inside a radius lambda/2D aperture.
        'gaussian' -- Gaussian weights (sigma ~ lambda/2D).
        'matched'  -- weight by ``psf_template``; optimal when the template
                      matches the source's shape in the reduced image (see
                      ``pa_averaged_stamp`` for an ADI-appropriate template).
    psf_template : ndarray or None
        Centred PSF stamp (odd side length, same pixel scale as ``image``),
        required for ``filter='matched'``. Its normalisation cancels in the S/N.
    include : bool
        If True, include the source aperture among the noise samples.
    show : bool
        If True, overlay the aperture positions on the filtered image.

    Returns (signal, noise, n_apertures).'''
    center_x, center_y = center_pix
    r_pix = np.hypot(x - center_x, y - center_y)

    res_pix = (206264806 * (wavelength * 1e-6) / D) / pixscale   # lambda/D in pixels
    aperture_radius = res_pix / 2.0

    # Ring of independent resolution elements at the source separation, with the
    # first aperture centred on the source itself.
    dtheta = res_pix / r_pix
    n_ring = int(np.rint((2 * np.pi) / dtheta))
    theta0 = np.arctan2(y - center_y, x - center_x)
    thetas = theta0 + dtheta * np.arange(n_ring)
    xs = center_x + r_pix * np.cos(thetas)
    ys = center_y + r_pix * np.sin(thetas)

    # Choose the photometric kernel.
    if filter == 'tophat':
        kernel = _disk_kernel(aperture_radius)
    elif filter == 'gaussian':
        kernel = _gaussian_kernel(aperture_radius)
    elif filter == 'matched':
        if psf_template is None:
            raise ValueError("filter='matched' requires a psf_template stamp")
        kernel = np.nan_to_num(np.asarray(psf_template, dtype=float))
    else:
        raise ValueError("filter must be 'tophat', 'gaussian' or 'matched', got %r" % (filter,))

    # Correlate the (NaN-filled) image with the kernel so every pixel of
    # `filtered` holds the weighted aperture flux there. Flip the kernel so this
    # is a correlation, not a convolution (matters for an asymmetric matched
    # kernel; harmless for the symmetric tophat/gaussian).
    filled = np.where(np.isfinite(image), image, 0.0)
    filtered = fftconvolve(filled, kernel[::-1, ::-1], mode='same')

    # Sample at the ring positions. map_coordinates indexes [row (y), col (x)].
    samples = map_coordinates(filtered, [ys, xs], order=1)

    if show:
        disp = jfits.InteractiveDisplay(filtered)
        disp.ax.plot(xs[1:], ys[1:], 'ro')
        disp.ax.plot(xs[:1], ys[:1], 'go')
        plt.show()

    signal = float(samples[0])
    noise_samples = samples if include else samples[1:]
    return signal, float(np.std(noise_samples)), int(noise_samples.size)


def pa_averaged_stamp(psf_stamp, derot_angles, combine=np.median):
    '''Position-angle-average a PSF stamp to model a source in a derotated image.

    In angular differential imaging the instrument PSF is fixed to the pupil
    while each exposure is derotated by its own angle before the exposures are
    combined. A real source therefore appears in the final image as the
    instrument PSF rotated about its own centre by each derotation angle and then
    combined. This builds that template from a clean PSF stamp using the same
    NaN-aware rotation (``nan_rot``) the reduction applies to the data, so the
    two are self-consistent.

    psf_stamp : ndarray (h, w)
        Centred instrument-PSF stamp (peak on the central pixel).
    derot_angles : sequence of float
        The derotation angles applied to the data (degrees) -- the *same* values,
        with the same sign, that were passed to ``nan_rot`` in the reduction.
    combine : callable
        Reduction applied over the rotated stack (default ``np.median`` to match
        a median-combined reduction; use ``np.mean`` for a mean combine).

    Returns the position-angle-averaged stamp (same shape as ``psf_stamp``).

    Note: this captures only the geometric field-rotation averaging of the PSF.
    It does NOT include the KLIP self-subtraction response (negative lobes /
    throughput); for that, forward-model a source through the full reduction.'''
    height, width = psf_stamp.shape
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    # nan_rot fills rotated-out corners with NaN; for a PSF stamp those are just
    # zero flux, so replace them before combining.
    rotated = [np.nan_to_num(nan_rot(psf_stamp, angle, center, pivot=True,
                                     out_size=(height, width)))
               for angle in derot_angles]
    return combine(rotated, axis=0)

def get_noise_psf(image, xpos, ypos, center_pix, wavelength,
                  D=8.25, pixscale=34.5, include=False, returnPhot=False, show=False):
    '''Estimate the noise at a source position using PSF-matched photometry.

    Unlike the ``get_noise*`` variants, the source is specified directly by its
    pixel coordinates. At each independent resolution element on the ring at the
    same radius, the (self-referenced) PSF is shifted into place, normalized, and
    used as a matched filter to weight the flux. Returns (noise, N), or
    (noise, N, phot) if ``returnPhot`` is True, where phot is the matched-filter
    flux at the source position.

    image : ndarray (ydim, xdim)
        The reduced image to measure.
    xpos, ypos : float
        Pixel coordinates of the source.
    center_pix : (x, y)
        Pixel coordinate of the primary star.
    wavelength : float, microns.  D : float, meters.  pixscale : float, mas/pixel.
    include : bool
        If True, include the source position itself in the noise samples.
    returnPhot : bool
        If True, also return the matched-filter flux at the source position.
    show : bool
        If True, overlay the sample positions on the image.'''

    r_pix = ( (xpos - center_pix[0])**2 + (ypos - center_pix[1])**2 )**0.5
    imgcpy = image.copy()
    imgcpy[~np.isfinite(imgcpy)] = 0

    res = 206264806 * (wavelength*10**-6)/D
    res_pix = res / pixscale

    dtheta = res_pix / r_pix

    N = int(np.rint((2*np.pi)/dtheta))

    thetas = (dtheta*np.arange(N)) + np.arctan2(ypos-center_pix[1], xpos-center_pix[0])

    zs0 = r_pix*np.exp(1j*thetas)
    zs = zs0 + (center_pix[0]+1j*center_pix[1])

    xs = zs.real
    ys = zs.imag
    if show:
        disp = jfits.InteractiveDisplay(image)
        disp.ax.plot(xs[1:],ys[1:],'ro')
        disp.ax.plot(xs[:1],ys[:1],'go')
        plt.show()
        input('press Enter')

    yy, xx = np.indices(image.shape)
    outs = []
    if include:
        for p in zip(xs, ys):
            zz = (xx-p[0]) + 1j*(yy-p[1])
            psf = shift(imgcpy, [p[1] - center_pix[1], p[0] - center_pix[0]])
            psf[np.abs(zz) > res_pix] = 0
            psf /= psf.sum()
            
            outs.append(np.nansum((psf*image)[np.abs(zz)<=res_pix]))
    else:
        for p in zip(xs[1:], ys[1:]):
            zz = (xx-p[0]) + 1j*(yy-p[1])
            psf = shift(imgcpy, [p[1] - center_pix[1], p[0] - center_pix[0]])
            psf[np.abs(zz) > res_pix] = 0
            psf /= psf.sum()
            
            outs.append(np.nansum((psf*image)[np.abs(zz)<=res_pix]))
        N-=1
    if returnPhot:
        zz = (xx-xs[0]) + 1j*(yy-ys[0])

        psf = shift(imgcpy, [ys[0] - center_pix[1], xs[0] - center_pix[0]])
        psf[np.abs(zz) > res_pix] = 0
        psf /= psf.sum()

        return np.std(outs), N, np.nansum((psf*image)[np.abs(zz)<=res_pix/2])
    else:
        return np.std(outs), N

def make_SN_map(image, center_pix, wavelength, D=8.25, pixscale=34.5):
    '''Build a per-pixel signal-to-noise map by PSF-matched photometry.

    For every pixel, measure the PSF-weighted flux at that location (the
    signal) and divide it by the standard deviation of the same measurement
    made at the ring of independent resolution elements at the same radius from
    ``center_pix`` (the noise), following the approach in ``get_noise_psf``.

    image : ndarray (ydim, xdim)
        The reduced image (may contain NaNs).
    center_pix : (x, y)
        Pixel coordinate of the primary star.
    wavelength : float
        Observing wavelength, microns.
    D : float
        Telescope diameter, meters.
    pixscale : float
        Plate scale, mas/pixel.

    Returns an ndarray the shape of ``image`` holding the S/N at each pixel.
    Pixels too close to the center to define a noise ring are set to NaN.'''
    yy, xx = np.indices(image.shape)

    res = 206264806 * (wavelength*10**-6)/D
    res_pix = res / pixscale

    imgcpy = image.copy()
    imgcpy[~np.isfinite(imgcpy)] = 0

    out_im = np.full(image.shape, np.nan, dtype=float)

    for ypix in range(image.shape[0]):
        for xpix in range(image.shape[1]):
            r_pix = ((xpix - center_pix[0])**2 + (ypix - center_pix[1])**2)**0.5
            if r_pix < 1:
                continue
            dtheta = res_pix / r_pix
            N = int(np.rint((2*np.pi)/dtheta))
            if N < 2:
                # not enough independent resolution elements to estimate noise
                continue

            thetas = (dtheta*np.arange(N)) + np.arctan2(ypix-center_pix[1],
                                                        xpix-center_pix[0])
            zs = r_pix*np.exp(1j*thetas) + (center_pix[0]+1j*center_pix[1])
            xs = zs.real
            ys = zs.imag

            # noise: PSF-matched photometry at the other resolution elements
            noise_samples = []
            for p in zip(xs[1:], ys[1:]):
                zz = (xx-p[0]) + 1j*(yy-p[1])
                psf = shift(imgcpy, [p[1] - center_pix[1], p[0] - center_pix[0]])
                psf[np.abs(zz) > res_pix] = 0
                psf /= psf.sum()
                noise_samples.append(np.nansum((psf*image)[np.abs(zz)<=res_pix]))

            # signal: PSF-matched photometry at this pixel
            zz = (xx-xs[0]) + 1j*(yy-ys[0])
            psf = shift(imgcpy, [ys[0] - center_pix[1], xs[0] - center_pix[0]])
            psf[np.abs(zz) > res_pix] = 0
            psf /= psf.sum()
            phot = np.nansum((psf*image)[np.abs(zz)<=res_pix/2])

            noise = np.std(noise_samples)
            if noise > 0:
                out_im[ypix, xpix] = phot/noise

    return out_im
