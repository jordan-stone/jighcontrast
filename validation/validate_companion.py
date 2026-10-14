#!/usr/bin/env python
"""V4. End-to-end: the spectrum of the simulation's own companion vs what was injected.

The SCALES sequence contains HIP 21152 B as injected by scalesETC (a Sonora
Bobcat model; see build_truth_spectrum.py, which reconstructs the injected channel
spectrum by replaying the simulator's projection/rectification). This is the one
test that does not use jFM's own source model to make the data.

The simulated star drifts by ~1 pixel across the band (a chromatic offset fixed in
time, as from residual atmospheric dispersion); the companion rotates about the
star's per-channel position. We extract the companion
  (a) assuming a single star position (53.5, 53.5) for all channels, and
  (b) after registering each channel on the star (``jPhot.measure_star_centers``
      / ``register_channels``),
and compare each with the candidate injected spectra using the full covariance.
Writes results/companion.npz and results/companion.png.
"""
import os
import numpy as np
from astropy.io import fits

import common as C
from jighcontrast import jFM, jPhot

TRUTH = os.path.join(C.HERE, "data", "truth_candidates.npz")


def score(f, cov, truth):
    """chi2/n of f vs truth (absolute), and of f vs the best-scaled truth (shape)."""
    Ci = np.linalg.inv(cov)
    d = f - truth
    chi_abs = float(d @ Ci @ d) / f.size
    A = float(truth @ Ci @ f) / float(truth @ Ci @ truth)
    d2 = f - A * truth
    return chi_abs, A, float(d2 @ Ci @ d2) / (f.size - 1)


def main():
    coro = np.asarray(fits.open(os.path.join(C.DATA, C.CORO), memmap=True)[0].data, float)
    noco = np.asarray(fits.open(os.path.join(C.DATA, C.NOCORO), memmap=True)[0].data, float)
    pa = np.loadtxt(os.path.join(C.DATA, "parallactic_angle_degrees.csv"), delimiter=",")
    centers = jPhot.measure_star_centers(noco, (54, 54))
    print("star position per channel: ch0 (%.2f, %.2f) ... ch53 (%.2f, %.2f)"
          % (*centers[0], *centers[-1]))
    truth = dict(np.load(TRUTH))
    cands = [k for k in truth if k.endswith("filter x sky")]

    runs = {}
    for label, (cube, nocoro) in {
            "single center (53.5, 53.5)": (coro, noco),
            "registered per channel": (jPhot.register_channels(coro, centers, C.CENTER),
                                       jPhot.register_channels(noco, centers, C.CENTER))}.items():
        stamps = jFM.offaxis_psf_stamps(nocoro, C.CENTER, half=9)
        r = jFM.extract_spectrum(cube, None, pa, C.WAVES, C.CENTER, C.COMP_SEP, C.COMP_PA,
                                 C.D_M, C.PIX_MAS, psf_stamps=stamps, npca=20,
                                 corr_method='gb16', n_jobs=C.N_JOBS, verbose=False)
        runs[label] = r
        bb = r['broadband']
        print("\n%s: broadband S/N %.1f, passes %d, max dlam %.2f"
              % (label, bb['snr'], r['n_iter'], r['dlam_fit'].max()))
        print("   %-28s %10s %10s %12s" % ("injected model", "chi2/n", "best scale", "shape chi2/n"))
        for k in cands:
            ca, A, cs = score(r['spectrum'], r['cov'], truth[k])
            print("   %-28s %10.2f %10.3f %12.2f" % (k.split("|")[0], ca, A, cs))

    np.savez(C.OUT + "/companion.npz", centers=centers,
             **{"%s|%s" % (lab, k): r[k] for lab, r in runs.items()
                for k in ("spectrum", "sigma", "cov")},
             **{"truth|" + k: truth[k] for k in cands})
    plot(runs, truth, cands)


def plot(runs, truth, cands):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (lab, r), col, off in zip(runs.items(), ["#999999", "k"], [-0.004, 0.004]):
        ax.errorbar(C.WAVES + off, r['spectrum'], r['sigma'], fmt="o", ms=3, color=col, label=lab)
    for k, col in zip(cands[:3], ["#d95f02", "#1b9e77", "#7570b3"]):
        ax.plot(C.WAVES, truth[k], "-", color=col, lw=1.2, label="injected: " + k.split("|")[0])
    ax.set(xlabel="wavelength (um)", ylabel="companion / star flux ratio",
           title="HIP 21152 B: extracted vs injected (scalesETC)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(C.OUT + "/companion.png", dpi=150)


if __name__ == "__main__":
    main()
