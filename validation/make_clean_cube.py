#!/usr/bin/env python
"""Prepare the cube used by the injection-recovery validations.

1. Register every channel of both sequences on the star (it drifts ~1 pixel
   across the band; jPhot.measure_star_centers / register_channels).
2. Remove the simulation's own companion, HIP 21152 B, by negative injection of
   its KLIP-FM-extracted spectrum (from validate_companion.py, registered run).
   Left in, its KLIP self-subtraction arcs -- spanning the +/-108 deg sky rotation
   at 390 mas -- sit under injected test sources and in their noise rings.
Writes results/clean_registered.npz (float32 cubes + star centers). The
subtraction leaves a small LOCAL residual (light the PSF model cannot describe,
e.g. the simulator's band-edge cross-talk), so the validations still exclude the
companion's position (2.5 lambda/D) from their noise rings; its KLIP arcs, which
spread over the whole annulus, are gone.
"""
import os
import numpy as np
from astropy.io import fits

import common as C
from jighcontrast import jFM, jPhot


def main():
    coro = np.asarray(fits.open(os.path.join(C.DATA, C.CORO), memmap=True)[0].data, float)
    noco = np.asarray(fits.open(os.path.join(C.DATA, C.NOCORO), memmap=True)[0].data, float)
    pa = np.loadtxt(os.path.join(C.DATA, "parallactic_angle_degrees.csv"), delimiter=",")
    centers = jPhot.measure_star_centers(noco, (54, 54))
    coro_r = jPhot.register_channels(coro, centers, C.CENTER)
    noco_r = jPhot.register_channels(noco, centers, C.CENTER)
    stamps = jFM.offaxis_psf_stamps(noco_r, C.CENTER, half=9)
    comp = np.load(C.OUT + "/companion.npz")
    f = comp["registered per channel|spectrum"]
    clean = jFM.inject_source(coro_r, stamps, pa, C.CENTER, C.COMP_SEP, C.COMP_PA, -f)
    np.savez(C.OUT + "/clean_registered.npz", coro=clean.astype(np.float32),
             noco=noco_r.astype(np.float32), centers=centers, companion=f)
    print("wrote results/clean_registered.npz")


if __name__ == "__main__":
    main()
