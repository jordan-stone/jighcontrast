"""Small synthetic ADI + IFS data sets for the test suite.

Speckles are fixed in the detector frame (they do not rotate with the sky),
evolve slowly in time, and scale with wavelength; a companion with a known
spectrum rotates with the parallactic angle. Everything is small enough that a
full KLIP-FM reduction runs in about a second.
"""
import numpy as np

from jighcontrast import jFM

NY = NX = 45
CENTER = ((NX - 1) / 2.0, (NY - 1) / 2.0)
D, PIXSCALE = 10.0, 20.0          # m, mas/pix -> lambda/D ~ 3.6-4.1 px


def gaussian_stamp(fwhm_pix, half=9):
    yy, xx = np.indices((2 * half + 1, 2 * half + 1))
    sig = fwhm_pix / 2.3548
    g = np.exp(-((xx - half) ** 2 + (yy - half) ** 2) / (2 * sig ** 2))
    return g


def make_dataset(n_exp=24, waves=(3.0, 3.3, 3.6, 3.9), rotation=90.0, n_modes=8,
                 speckle_amp=1.0, white=0.0, drift=0.15, seed=0, star_peak=1.0e4):
    """Return dict(cube, nocoro_stamps, pa, waves) of a synthetic sequence.

    Speckle field = sum of ``n_modes`` smooth random patterns whose spatial
    frequencies scale as 1/lambda (so they stretch radially with wavelength),
    with time coefficients = static + ``drift`` * random walk. ``white`` adds
    Gaussian pixel noise (in units of the speckle rms)."""
    rng = np.random.default_rng(seed)
    waves = np.asarray(waves, float)
    pa = np.linspace(-rotation / 2, rotation / 2, n_exp)
    yy, xx = np.indices((NY, NX), dtype=float)
    dx, dy = xx - CENTER[0], yy - CENTER[1]
    rr = np.hypot(dx, dy)
    envelope = np.exp(-rr / 18.0)
    k = rng.uniform(0.25, 0.9, (n_modes, 2)) * rng.choice([-1, 1], (n_modes, 2))
    ph = rng.uniform(0, 2 * np.pi, n_modes)
    static = rng.normal(size=n_modes)
    walk = np.cumsum(rng.normal(size=(n_exp, n_modes)), axis=0) / np.sqrt(n_exp)
    coeff = static[None, :] + drift * walk
    cube = np.zeros((n_exp, waves.size, NY, NX))
    for w, lam in enumerate(waves):
        scale = waves[0] / lam
        basis = np.array([np.cos(scale * (k[j, 0] * dx + k[j, 1] * dy) + ph[j])
                          for j in range(n_modes)]) * envelope
        cube[:, w] = speckle_amp * np.einsum('tj,jyx->tyx', coeff, basis)
    if white:
        cube += white * speckle_amp * rng.normal(size=cube.shape)
    res = np.array([jFM.resolution_element_pix(l, D, PIXSCALE) for l in waves])
    stamps = np.array([star_peak * gaussian_stamp(r) for r in res])
    return dict(cube=cube, psf_stamps=stamps, pa=pa, waves=waves)
