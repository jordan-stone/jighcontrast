import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize
import jfits

def nan_rot(im, angle, axis, order=3, pivot=False, out_size=None):
    '''Rotate an image clockwise about an axis, propagating NaNs.

    If ``pivot`` is True the image pivots about ``axis``; otherwise ``axis`` is
    placed at the center of the output image. NaNs are preserved rather than
    smeared by the interpolation (see ``nan_map_coordinates_2d``).

    im : ndarray (ydim, xdim)
        Image to rotate (may contain NaNs).
    angle : float
        Rotation angle, degrees clockwise.
    axis : (x, y)
        Pixel coordinate of the rotation center.
    order : int
        Spline interpolation order passed to ``map_coordinates`` (default 3).
    pivot : bool
        If True, pivot about ``axis``; if False, center ``axis`` in the output.
    out_size : (ydim, xdim) or None
        Shape of the output image. If None, a square of side ``2*max(im.shape)``
        is used.

    Returns the rotated image as an ndarray.'''
    angle*=np.pi/180.#convert to radians
    if out_size is None:
        y, x = np.indices((2*max(im.shape),2*max(im.shape)))
        y-=(int(max(im.shape)/2))
        x-=(int(max(im.shape)/2))
    else:
        y, x = np.indices(out_size)
        y-=int((out_size[0]-im.shape[0])/2)
        x-=int((out_size[1]-im.shape[1])/2)
    
    #calculate how the axis moves when pivoting from bottom left corner
    theta_axis = np.arctan2(axis[1],axis[0])
    r_axis = np.abs(axis[0]+1j*axis[1])
    yoffset = r_axis*np.sin(theta_axis) - r_axis*np.sin(theta_axis-angle)
    xoffset = r_axis*np.cos(theta_axis) - r_axis*np.cos(theta_axis-angle)

    #put the axis in the middle? 
    ycenter_offset = (1-pivot) * ((im.shape[0]/2.)-axis[1])#pivot is a bool (i.e. 0 or 1)
    xcenter_offset = (1-pivot) * ((im.shape[1]/2.)-axis[0])
    yoffset += ycenter_offset
    xoffset += xcenter_offset

    #make rotation matrix elements
    ct = np.cos(angle)
    st = np.sin(angle)

    #do the rotation 
    new_x = (ct*(x-xoffset) - st*(y-yoffset)) 
    new_y = (st*(x-xoffset) + ct*(y-yoffset))
    return nan_map_coordinates_2d(im, new_y, new_x, order=order, cval=np.nan)

def nan_map_coordinates_2d(img, yp, xp, **mc_kwargs):
    """
    slightly modified from pyKLIP...

    scipy.ndimage.map_coordinates() that handles nans for 2-D transformations. Only works in 2-D!
    
    Do NaN detection by defining any pixel in the new coordiante system (xp, yp) as a nan
    If any one of the neighboring pixels in the original image is a nan (e.g. (xp, yp) = 
    (120.1, 200.1) is nan if either (120, 200), (121, 200), (120, 201), (121, 201) is a nan)

    Args:
        img (np.array): 2-D image that is looking to be transformed
        yp (np.array): 2-D array of y-coordinates that the image is evaluated out
        xp (np.array): 2-D array of x-coordinates that the image is evaluated out
        mc_kwargs (dict): other parameters to pass into the map_coordinates function.

    Returns:
        transformed_img (np.array): 2-D transformed image. Each pixel is evaluated at the (yp, xp) specified by xp and yp. 
    """
    if mc_kwargs is None:
        mc_kwargs = {}
    # if nothing specified, we will pad transformations with np.nan
    if "cval" not in mc_kwargs:
        mc_kwargs["cval"] = np.nan

    # check all four pixels around each pixel and see whether they are nans
    xp_floor = np.clip(np.floor(xp).astype(int), 0, img.shape[1]-1)
    xp_ceil = np.clip(np.ceil(xp).astype(int), 0, img.shape[1]-1)
    yp_floor = np.clip(np.floor(yp).astype(int), 0, img.shape[0]-1)
    yp_ceil = np.clip(np.ceil(yp).astype(int), 0, img.shape[0]-1)
    rotnans = np.where(np.isnan(img[yp_floor.ravel(), xp_floor.ravel()]) | 
                       np.isnan(img[yp_floor.ravel(), xp_ceil.ravel()]) |
                       np.isnan(img[yp_ceil.ravel(), xp_floor.ravel()]) |
                       np.isnan(img[yp_ceil.ravel(), xp_ceil.ravel()]))

    # resample image based on new coordinates, set nan values as median
    nanpix = np.where(np.isnan(img))
    medval = np.nanmedian(img)
    img_copy = np.copy(img)
    img_copy[nanpix] = medval
    transformed_img = map_coordinates(img_copy, [yp, xp], **mc_kwargs)

    # mask nans
    img_shape = transformed_img.shape
    transformed_img.shape = [img_shape[0] * img_shape[1]]
    transformed_img[rotnans] = np.nan
    transformed_img.shape = img_shape

    return transformed_img

def get_rotation_center(image, ax_guess, opt_box_size=10, Nsteps=21):
    '''Find the rotation center of an image by 180-degree self-subtraction.

    Scans a grid of candidate centers around ``ax_guess`` and, for each, rotates
    the image by 0 and 180 degrees about that point and measures the squared
    difference within a central box. The center that minimizes this residual
    (a point-symmetric PSF subtracts to ~zero) is returned.

    image : ndarray (ydim, xdim)
        The image whose rotation center is sought.
    ax_guess : (x, y)
        Initial guess for the center; the search spans +/-1 pixel around it.
    opt_box_size : int
        Side length (pixels) of the central box used to score each candidate.
    Nsteps : int
        Number of grid points per axis in the +/-1 pixel search.

    Returns the best-fit (x, y) center.'''
    opt_slices = (slice(image.shape[0]//2 - opt_box_size//2,
                        image.shape[0]//2 + opt_box_size//2),
                  slice(image.shape[1]//2 - opt_box_size//2, 
                        image.shape[1]//2 + opt_box_size//2))
    def fit_func(axis):
        rot0 = nan_rot(image, 0, axis, out_size=image.shape )
        rot180 = nan_rot(image, 180, axis, out_size=image.shape)
        chi = np.sum((rot0 - rot180)[opt_slices]**2)
        print(axis, chi)
        return chi

    #fitout = minimize(fit_func, ax_guess)

    fit_dict = {}
    for d0 in np.linspace(ax_guess[0]-1, ax_guess[0]+1, Nsteps):
        for d1 in np.linspace(ax_guess[1]-1, ax_guess[1]+1, Nsteps):
            fit_dict[(d0, d1)] = fit_func((d0, d1))


    skeys = sorted(fit_dict.keys(), key=fit_dict.get)

    #rot0 = nan_rot(image, 0, fitout.x, out_size=image.shape )
    #rot180 = nan_rot(image, 180, fitout.x, out_size=image.shape)
    #jfits.InteractiveDisplay((rot0 - rot180)[opt_slices])

    print(skeys[0], fit_func(skeys[0]))
    rot0 = nan_rot(image, 0, skeys[0], out_size=image.shape )
    rot180 = nan_rot(image, 180, skeys[0], out_size=image.shape)
    jfits.InteractiveDisplay((rot0 - rot180)[opt_slices])


    return skeys[0]

