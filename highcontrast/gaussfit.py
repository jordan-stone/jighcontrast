'''Self-contained 2-D Gaussian model and fitter.

Drop-in replacements for the two functions ``highcontrast`` previously imported
from the external ``gaussfitter`` package (``twodgaussian`` and ``gaussfit``),
implemented on top of numpy and ``scipy.optimize.leastsq`` so the package has no
dependency outside the standard scientific stack.

Parameter convention (blocks present only when the corresponding flag is on)::

    [height (if vheight)], amplitude, center_x, center_y,
    width (if circle) | width_x, width_y, [rotation_deg (if rotate)]

``gaussfit`` always returns a vector that *starts* with a height/background term
(0.0 when ``vheight`` is False) so that downstream indices are stable, i.e.
``out[1]`` is the amplitude, ``out[2]``/``out[3]`` the x/y centers and ``out[4]``
the (first) width.
'''

import numpy as np
from scipy.optimize import leastsq


def twodgaussian(inpars, circle=True, rotate=False, vheight=False, shape=None):
    '''Build a 2-D Gaussian from ``inpars``.

    inpars : sequence of float
        Parameters in the order described in the module docstring.
    circle : bool
        If True a single ``width`` is used for both axes; otherwise
        ``width_x, width_y``.
    rotate : bool
        If True the final parameter is a rotation angle in degrees.
    vheight : bool
        If True the first parameter is an additive background level.
    shape : (ydim, xdim) or None
        If given, evaluate the Gaussian on a grid of this shape and return the
        image. If None, return a callable ``f(y, x)``.

    Returns an ndarray (when ``shape`` is given) or a function of ``(y, x)``.'''
    inpars = list(inpars)
    height = float(inpars.pop(0)) if vheight else 0.0
    amplitude = float(inpars.pop(0))
    center_x = float(inpars.pop(0))
    center_y = float(inpars.pop(0))
    if circle:
        width_x = width_y = float(inpars.pop(0))
    else:
        width_x = float(inpars.pop(0))
        width_y = float(inpars.pop(0))
    rota = np.deg2rad(float(inpars.pop(0))) if rotate else 0.0
    if inpars:
        raise ValueError('twodgaussian: %d unused parameter(s): %r'
                         % (len(inpars), inpars))

    cosr, sinr = np.cos(rota), np.sin(rota)

    def rotgauss(y, x):
        xr = (x - center_x) * cosr - (y - center_y) * sinr
        yr = (x - center_x) * sinr + (y - center_y) * cosr
        return height + amplitude * np.exp(-((xr / width_x) ** 2
                                             + (yr / width_y) ** 2) / 2.0)

    if shape is not None:
        yy, xx = np.indices(shape, dtype=float)
        return rotgauss(yy, xx)
    return rotgauss


def _moments(data, circle=True, rotate=False, vheight=False):
    '''Crude parameter estimate from image moments, used when no initial
    ``params`` are supplied to :func:`gaussfit`.'''
    data = np.asarray(data, dtype=float)
    total = data.sum()
    Y, X = np.indices(data.shape)
    center_x = (X * data).sum() / total
    center_y = (Y * data).sum() / total
    col = data[int(np.clip(center_y, 0, data.shape[0] - 1)), :]
    row = data[:, int(np.clip(center_x, 0, data.shape[1] - 1))]
    width_x = np.sqrt(np.abs((np.arange(col.size) - center_x) ** 2 * col).sum()
                      / np.abs(col).sum())
    width_y = np.sqrt(np.abs((np.arange(row.size) - center_y) ** 2 * row).sum()
                      / np.abs(row).sum())
    amplitude = data.max()

    params = [amplitude, center_x, center_y]
    if circle:
        params.append((width_x + width_y) / 2.0)
    else:
        params.extend([width_x, width_y])
    if rotate:
        params.append(0.0)
    if vheight:
        params.insert(0, np.median(data))
    return params


def gaussfit(data, params=None, circle=True, rotate=False, vheight=False,
             returnfitimage=False):
    '''Least-squares fit of a 2-D Gaussian to ``data``.

    data : ndarray (ydim, xdim)
        Image to fit.
    params : sequence of float or None
        Initial guess in the convention of :func:`twodgaussian` (without a
        height term when ``vheight`` is False). If None, estimated from moments.
    circle, rotate, vheight : bool
        Model options, passed through to :func:`twodgaussian`.
    returnfitimage : bool
        If True, also return the best-fit model image.

    Returns the best-fit parameter vector (always length >= 5, with a height
    slot at index 0 that is 0.0 when ``vheight`` is False), or
    ``(params, fitimage)`` when ``returnfitimage`` is True.'''
    data = np.asarray(data, dtype=float)
    if params is None:
        params = _moments(data, circle=circle, rotate=rotate, vheight=vheight)
    params = list(params)

    def err(p):
        model = twodgaussian(p, circle=circle, rotate=rotate, vheight=vheight,
                             shape=data.shape)
        return (model - data).ravel()

    fit, _ = leastsq(err, params)
    fit = list(fit)

    # Rebuild the model image from the raw fit before normalising the vector.
    fitimage = twodgaussian(fit, circle=circle, rotate=rotate, vheight=vheight,
                            shape=data.shape) if returnfitimage else None

    # Ensure a stable output layout: a height slot at index 0 and positive
    # widths (the Gaussian is even in width, so the sign is degenerate).
    if not vheight:
        fit = [0.0] + fit
    fit[4] = abs(fit[4])
    if not circle:
        fit[5] = abs(fit[5])
    outpars = np.array(fit, dtype=float)

    if returnfitimage:
        return outpars, fitimage
    return outpars
