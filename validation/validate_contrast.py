#!/usr/bin/env python
"""V3. Contrast-curve calibration by injection at the reported limit.

``jFM.contrast_curve`` + ``combine_contrast_empirical`` quote, per separation,
the contrast a flat-spectrum source needs for detection at a fixed false-alarm
probability: c_lim(r) = nsig_eff(r) * sigma(r), with sigma the measured scatter
of the channel-combined matched-filter amplitude around the ring. If that is
right, a source injected AT c_lim and measured independently -- with
``extract_spectrum`` + ``broadband_contrast`` (forward-modeled spectral fit,
mean combine, source-anchored zones, its own ring) -- comes back with S/N ~
nsig_eff(r). S/N systematically below nsig_eff means the curve is optimistic
(too deep); above, conservative.

Variants: the configuration of the earlier proposal figures re-run with the
corrected code (median centering, median combine) and the new candidate
defaults (frame centering with median or mean combine). For each, three channel
combinations are reported: independent channels (``combine_contrast``), the
covariance formula (``combine_contrast_cov``) and the empirical one
(``combine_contrast_empirical``). Runs on the registered, companion-subtracted
cube (make_clean_cube.py); the companion's position is still kept out of the noise
rings, because its subtraction leaves a small local residual (left in, it
raises the 400 mas limit by 2.5x). Writes results/contrast.npz and
results/contrast.png.
"""
import time
import numpy as np

import common as C
from jighcontrast import jFM

RADII = np.array([8.0, 12.0, 16.0, 20.0, 24.0, 28.0])     # = 160..560 mas
INJ_PAS = [7.0, 97.0, 187.0, 277.0]
VARIANTS = {"median/median (old config)": dict(centering='median', combine='median'),
            "frame/median": dict(centering='frame', combine='median'),
            "frame/mean": dict(centering='frame', combine='mean')}


def main():
    coro, stamps, pa = C.load()
    noco = C.load_unocculted()
    curves, t0 = {}, time.time()
    for name, kw in VARIANTS.items():
        res = jFM.contrast_curve(coro, noco, pa, C.WAVES, C.CENTER, RADII, C.D_M, C.PIX_MAS,
                                 npca=20, n_arcs=4, dr=3.5, exclusion_fwhm=0.75, n_probe=8,
                                 known_companion=(C.COMP_PA, C.COMP_SEP), comp_exclude_fwhm=2.5,
                                 return_resid=True, n_jobs=C.N_JOBS, verbose=False, **kw)
        _, c_ind = jFM.combine_contrast(res)
        corr = [jFM.spectral_correlation(r.T, shrink=0.3) for r in res['resid_pix']]
        _, c_cov, n_eff = jFM.combine_contrast_cov(res, corr)
        _, c_emp, n_ring = jFM.combine_contrast_empirical(res)
        curves[name] = dict(c_ind=c_ind, c_cov=c_cov, c_emp=c_emp, n_eff=n_eff, n_ring=n_ring,
                            nsig=res['nsig_eff'], tau=np.nanmedian(res['tau'], axis=1))
        print("[%4.0fs] %-28s 5-sigma contrast: independent / covariance / EMPIRICAL"
              % (time.time() - t0, name))
        for i, r in enumerate(RADII):
            print("        %4.0f mas: %.2e / %.2e / %.2e   (N_ring %d, tau %.2f)"
                  % (r * C.PIX_MAS, c_ind[i], c_cov[i], c_emp[i], n_ring[i],
                     curves[name]['tau'][i]), flush=True)

    # ---- injection at the (empirical) limit ------------------------------------
    snr = {}
    from scipy.stats import norm, t as student_t
    for name in ("frame/median", "frame/mean"):
        cl = curves[name]['c_emp']
        snr[name] = np.full((len(RADII), len(INJ_PAS)), np.nan)
        for i, r in enumerate(RADII):
            for j, pa_deg in enumerate(INJ_PAS):
                inj = jFM.inject_source(coro, stamps, pa, C.CENTER, r, pa_deg, cl[i])
                e = jFM.extract_spectrum(inj, None, pa, C.WAVES, C.CENTER, r, pa_deg, C.D_M,
                                         C.PIX_MAS, npca=20, psf_stamps=stamps,
                                         exclude=[(C.COMP_SEP, C.COMP_PA)], n_jobs=C.N_JOBS, verbose=False)
                snr[name][i, j] = e['broadband']['snr']
            # expected S/N of a source AT the limit = threshold for that ring size
            n = curves[name]['n_ring'][i]
            expect = student_t.isf(norm.sf(5.0), n - 1) * np.sqrt(1 + 1.0 / n)
            print("[%4.0fs] %-12s r=%4.0f mas injected at c_lim=%.2e: S/N %s  (threshold %.2f)"
                  % (time.time() - t0, name, r * C.PIX_MAS, cl[i], np.round(snr[name][i], 1),
                     expect), flush=True)
    np.savez(C.OUT + "/contrast.npz", radii=RADII,
             **{"%s|%s" % (n, k): v for n, d in curves.items() for k, v in d.items()},
             **{"snr|%s" % n: v for n, v in snr.items()})
    plot(curves, snr)


def plot(curves, snr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm, t as student_t
    sep = RADII * C.PIX_MAS
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axs[0]
    for (name, d), col in zip(curves.items(), ["#7570b3", "#1b9e77", "#d95f02"]):
        ax.plot(sep, d['c_emp'], "-o", color=col, label=name + " (empirical)")
        ax.plot(sep, d['c_ind'], ":", color=col, label=name + " (indep. channels)")
    ax.set(yscale="log", xlabel="separation (mas)", ylabel="5-sigma-equivalent contrast",
           title="Contrast curve (flat spectrum)")
    ax.legend(fontsize=7)
    ax = axs[1]
    for (name, s), col in zip(snr.items(), ["#1b9e77", "#d95f02"]):
        n = curves[name]['n_ring']
        expect = np.array([student_t.isf(norm.sf(5.0), k - 1) * np.sqrt(1 + 1.0 / k) for k in n])
        ax.plot(np.repeat(sep, s.shape[1]) + (5 if 'mean' in name else -5),
                (s / expect[:, None]).ravel(), "o", color=col, alpha=.7, label=name)
    ax.axhline(1, color="k", lw=.8)
    ax.set(xlabel="separation (mas)", ylabel="recovered S/N / threshold",
           title="Injected at the limit: recovered significance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.OUT + "/contrast.png", dpi=150)


if __name__ == "__main__":
    main()
