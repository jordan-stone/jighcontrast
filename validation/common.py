"""Shared set-up for the validation scripts: the synthetic SCALES HIP 21152 data
set (3 h ADI sequence, 90 exposures x 54 channels, 2.9-4.1 um) and its constants.

Data location: set JIGHCONTRAST_DATA to the directory holding the two FITS cubes
and parallactic_angle_degrees.csv (default: ../examples).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker process

import numpy as np
from astropy.io import fits

from jighcontrast import jFM

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("JIGHCONTRAST_DATA", os.path.join(HERE, "..", "examples"))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

CORO = "3hr_LowResL-2p9-4p15_IFScube_seq_adi_coronagraph_pLmag14p66_sLmag5p3.fits"
NOCORO = "3hr_LowResL-2p9-4p15_IFScube_seq_star_no_coronagraph_pLmag14p66_sLmag5p3.fits"
CENTER = (53.5, 53.5)
D_M, PIX_MAS = 10.0, 20.0
WAVES = np.linspace(2.91157407, 4.13842593, 54)
COMP_SEP, COMP_PA = 19.5, 217.0       # the companion injected by the simulation
N_JOBS = min(16, os.cpu_count() or 1)


CLEAN = os.path.join(OUT, "clean_registered.npz")


def load(raw=False):
    """Return (coro, psf_stamps, pa): the coronagraphic cube, the off-axis PSF per
    channel from the unocculted cube, and the parallactic angles.

    By default the cube is the registered, companion-subtracted one written by
    make_clean_cube.py (channels shifted so the star is at CENTER; HIP 21152 B
    removed). ``raw=True`` returns the simulation as delivered (memmap)."""
    pa = np.loadtxt(os.path.join(DATA, "parallactic_angle_degrees.csv"), delimiter=",")
    if not raw:
        d = np.load(CLEAN)
        return (d['coro'].astype(float), jFM.offaxis_psf_stamps(d['noco'], CENTER, half=9), pa)
    coro = fits.open(os.path.join(DATA, CORO), memmap=True)[0].data
    noco = fits.open(os.path.join(DATA, NOCORO), memmap=True)[0].data
    stamps = jFM.offaxis_psf_stamps(noco, CENTER, half=9)
    return coro, stamps, pa


def load_unocculted():
    """Registered unocculted cube (for contrast_curve)."""
    return np.load(CLEAN)['noco'].astype(float)


def aperture(img, x, y, r):
    yy, xx = np.indices(img.shape)
    m = (np.hypot(xx - x, yy - y) <= r) & np.isfinite(img)
    return float(img[m].sum())


def sky_xy(sep, pa_deg):
    return (CENTER[0] - sep * np.sin(np.deg2rad(pa_deg)),
            CENTER[1] + sep * np.cos(np.deg2rad(pa_deg)))
