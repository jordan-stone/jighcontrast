#!/usr/bin/env python
"""V2b. Which channel-correlation estimate gives calibrated uncertainties?

The per-channel sigma from the noise ring is calibrated (V2: pull rms ~1), but
the 54 x 54 correlation matrix, estimated from a few dozen correlated ring
samples and then inverted, can be noisy. For faint injections (first-order
regime, one reduction each) this script re-derives the covariance with several
estimators from the SAME reduction and scores each by
  * chi2/n of (f_hat - f_true) with the full covariance (should be ~1, and
    distributed as chi2 with n dof), and
  * the pull of the GLS broadband amplitude, (A - A_true)/sigma_A, over the
    injections (rms should be ~1) -- this sets detection significance -- with
    sigma_A both measured at the ring (as broadband_contrast does) and
    propagated through the covariance. GLS weights from a correlation matrix are
    compared with inverse-variance weights (the 'independent' row, which
    jFM.broadband_contrast now uses by default).
Writes results/covariance.npz.
"""
import time
import numpy as np

import common as C
from jighcontrast import jFM
from validate_extraction import contrast_shape, SPECTRA

SEPS = [10.0, 12.0, 16.0, 20.0, 24.0, 28.0]
PAS = [7.0, 52.0, 97.0, 142.0, 262.0, 307.0, 352.0]   # 42 independent noise realizations
LEVELS = [3e-5]                                     # faint: first-order regime
METHODS = [("ring, shrink 0.1", dict(corr_method='ring', shrink=0.1)),
           ("ring, shrink 0.3", dict(corr_method='ring', shrink=0.3)),
           ("ring, shrink 0.6", dict(corr_method='ring', shrink=0.6)),
           ("GB16 parametric", dict(corr_method='gb16')),
           ("independent", dict(corr_method='none'))]   # = inverse-variance weights (the default)


def main():
    coro, stamps, pa = C.load()
    cube = np.asarray(coro, float)
    shape = contrast_shape(SPECTRA["L 2000 K"])
    chi2 = {m: [] for m, _ in METHODS}
    bbpull = {m: [] for m, _ in METHODS}          # empirical (ring) sigma
    bbpull_cov = {m: [] for m, _ in METHODS}      # covariance-formula sigma
    params = []
    t0 = time.time()
    for lev in LEVELS:
        for sep in SEPS:
            for pa_deg in PAS:
                truth = lev * shape
                inj = jFM.inject_source(cube, stamps, pa, C.CENTER, sep, pa_deg, truth)
                red = jFM.klip_fm_reduce(inj, pa, C.WAVES, C.CENTER, sep, pa_deg, stamps,
                                         C.D_M, C.PIX_MAS, npca=20, n_jobs=C.N_JOBS)
                line = []
                for name, kw in METHODS:
                    r = jFM.extract_spectrum(None, None, pa, C.WAVES, C.CENTER, sep, pa_deg,
                                             C.D_M, C.PIX_MAS, psf_stamps=stamps, reduction=red,
                                             exclude=[(C.COMP_SEP, C.COMP_PA)],
                                             verbose=False, **kw)
                    d = r['spectrum'] - truth
                    x2 = float(d @ np.linalg.solve(r['cov'], d)) / d.size
                    bb = jFM.broadband_contrast(r, spectrum_shape=shape, weights='gls')
                    chi2[name].append(x2)
                    bbpull[name].append((bb['contrast'] - lev) / bb['sigma'])
                    bbpull_cov[name].append((bb['contrast'] - lev) / bb['sigma_cov'])
                    if kw.get('corr_method') == 'gb16':
                        params.append((sep, *r['corr_params']))
                    line.append("%s %.2f/%.2f" % (name.split(',')[0][:4], x2, bbpull[name][-1]))
                print("[%5.0fs] c=%.0e sep=%4.1f pa=%5.1f  chi2/n / bb-pull:  %s"
                      % (time.time() - t0, lev, sep, pa_deg, "  ".join(line)), flush=True)
    seps = np.repeat(SEPS, len(PAS))
    print("\nSUMMARY (n=%d injections)" % len(chi2[METHODS[0][0]]))
    print("%-18s %12s %12s %14s %14s" % ("method", "mean chi2/n", "median", "bb pull mean", "bb pull rms"))
    for name, _ in METHODS:
        x, b = np.array(chi2[name]), np.array(bbpull[name])
        bc = np.array(bbpull_cov[name])
        print("%-18s   covariance-formula sigma: bb pull rms %.2f   by sep: %s"
              % (name, np.sqrt(np.mean(bc ** 2)),
                 " ".join("%.0f:%.2f" % (s_, np.sqrt(np.mean(bc[seps == s_] ** 2))) for s_ in SEPS)))
        print("%-18s %12.2f %12.2f %14.2f %14.2f   bb-pull rms by sep: %s"
              % (name, x.mean(), np.median(x), b.mean(), np.sqrt(np.mean(b ** 2)),
                 " ".join("%.0f:%.2f" % (s_, np.sqrt(np.mean(b[seps == s_] ** 2))) for s_ in SEPS)))
    np.savez(C.OUT + "/covariance.npz", seps=seps, methods=np.array([m for m, _ in METHODS]),
             chi2=np.array([chi2[m] for m, _ in METHODS]),
             bbpull=np.array([bbpull[m] for m, _ in METHODS]),
             bbpull_cov=np.array([bbpull_cov[m] for m, _ in METHODS]), gb16_params=np.array(params))


if __name__ == "__main__":
    main()
