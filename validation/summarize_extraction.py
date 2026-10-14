#!/usr/bin/env python
"""Summarize V2 (results/extraction.npz): tables to stdout, figures to results/."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

d = np.load(C.OUT + "/extraction.npz")
f, f1, truth, sig, naive = d['f'], d['f1'], d['truth'], d['sigma'], d['naive']
cov, lev, sep, cen, name = d['cov'], d['level'], d['sep'], d['centering'], d['name']
waves = d['waves']
pulls = (f - truth) / sig


def amp(x, t):
    """Least-squares amplitude of the true shape in x, relative to truth (band-mean ratio)."""
    return np.sum(x * t, axis=-1) / np.sum(t * t, axis=-1)


def bb_pull(i):
    """Pull of the band-averaged contrast as jFM.broadband_contrast reports it:
    inverse-variance weights, error measured at the ring positions (stored as
    bb_contrast / bb_sigma for a flat shape); the true value is the same
    weighted mean of the injected spectrum."""
    w = 1.0 / sig[i] ** 2
    w = w / w.sum()
    return (d['bb_contrast'][i] - w @ truth[i]) / d['bb_sigma'][i]


rows = []
print("\nSpectral extraction injection-recovery (frame centering unless noted)")
print("%-9s %-7s %5s %4s | %9s %9s %9s | %6s %6s | %6s %7s | %7s %5s"
      % ("spectrum", "level", "sep", "n", "FM/true", "1pass", "naive", "pull", "pull", "chi2/n",
         "bb pull", "bbS/N", "dlam"))
print("%-9s %-7s %5s %4s | %9s %9s %9s | %6s %6s | %6s %7s | %7s %5s"
      % ("", "", "(px)", "", "(median)", "", "", "mean", "rms", "med", "rms", "med", "max"))
for cent in ("frame", "median"):
    for nm in np.unique(name):
        for lv in np.unique(lev):
            for sp in np.unique(sep):
                m = (cen == cent) & (name == nm) & (lev == lv) & (sep == sp)
                if not m.any():
                    continue
                idx = np.where(m)[0]
                bbp = np.array([bb_pull(i) for i in idx])
                r = dict(cen=cent, name=nm, lev=lv, sep=sp, n=m.sum(),
                         fm=np.median(amp(f[m], truth[m])), one=np.median(amp(f1[m], truth[m])),
                         naive=np.median(amp(naive[m], truth[m])),
                         pmean=pulls[m].mean(), prms=np.sqrt(np.mean(pulls[m] ** 2)),
                         chi2=np.median(d['chi2'][m]), bbrms=np.sqrt(np.mean(bbp ** 2)),
                         snr=np.median(d['bb_snr'][m]), dlam=np.max(d['dlam'][m]))
                rows.append(r)
                print("%-9s %.0e %5.0f %4d | %9.3f %9.3f %9.3f | %6.2f %6.2f | %6.2f %7.2f | %7.1f %5.2f%s"
                      % (nm, lv, sp, r['n'], r['fm'], r['one'], r['naive'], r['pmean'], r['prms'],
                         r['chi2'], r['bbrms'], r['snr'], r['dlam'],
                         "  (median centering)" if cent == "median" else ""))

# ---- pooled statistics (frame) ------------------------------------------------
fr = cen == "frame"
print("\nPooled (frame, all cases, all channels): pull mean %.3f  rms %.3f  (N=%d)"
      % (pulls[fr].mean(), np.sqrt(np.mean(pulls[fr] ** 2)), pulls[fr].size))
for sp in np.unique(sep):
    m = fr & (sep == sp)
    bp = np.array([bb_pull(i) for i in np.where(m)[0]])
    print("  sep %4.0f px: per-channel pull rms %.2f | broadband pull mean %.2f rms %.2f"
          % (sp, np.sqrt(np.mean(pulls[m] ** 2)), bp.mean(), np.sqrt(np.mean(bp ** 2))))
for lv in np.unique(lev):
    m = fr & (lev == lv)
    print("  level %.0e: band-mean FM/true = %.3f +/- %.3f (median, robust sd of median)   "
          "1-pass %.3f   naive %.3f"
          % (lv, np.median(amp(f[m], truth[m])),
             1.2533 * np.std(amp(f[m], truth[m])) / np.sqrt(m.sum()),
             np.median(amp(f1[m], truth[m])), np.median(amp(naive[m], truth[m]))))

# ---- figures ------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 8.5))
ax = axs[0, 0]
cols = dict(zip(np.unique(lev), ["#a6cee3", "#1f78b4", "#33a02c", "#e31a1c"]))
for lv in np.unique(lev):
    m = fr & (lev == lv)
    ratio = f[m] / truth[m]
    se = 1.2533 * np.std(ratio, axis=0) / np.sqrt(m.sum())
    ax.errorbar(waves, np.median(ratio, axis=0), se, fmt="o-", ms=2.5, lw=1, color=cols[lv],
                label="KLIP-FM, c=%.0e" % lv)
    ax.plot(waves, np.median(naive[m] / truth[m], axis=0), ":", color=cols[lv], lw=1)
ax.axhline(1, color="k", lw=0.8)
ax.set(xlabel="wavelength (um)", ylabel="recovered / injected (median over positions)",
       ylim=(0, 1.6), title="Per-channel bias (dotted: raw-PSF fit, no FM)")
ax.legend(fontsize=7)

ax = axs[0, 1]
p = pulls[fr].ravel()
ax.hist(np.clip(p, -5, 5), bins=60, density=True, color="0.6", label="pulls (all channels)")
x = np.linspace(-5, 5, 300)
ax.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), "k", label="N(0, 1)")
ax.set(xlabel="(f_hat - f_true) / sigma", title="Per-channel error calibration")
ax.legend(fontsize=8)

ax = axs[1, 0]
for lv in np.unique(lev):
    m = fr & (lev == lv)
    ax.semilogx(d['dlam'][m].max(axis=1), amp(f1[m], truth[m]), "o", mfc="none", color=cols[lv],
                ms=4)
    ax.semilogx(d['dlam'][m].max(axis=1), amp(f[m], truth[m]), "o", color=cols[lv], ms=4,
                label="c=%.0e" % lv)
ax.axhline(1, color="k", lw=0.8)
ax.set(xlabel="max dlam at injected flux", ylabel="band-mean recovered / injected",
       title="Refinement (filled) vs single pass (open)")
ax.legend(fontsize=7)

ax = axs[1, 1]
for nm, col in zip(np.unique(name), ["#1b9e77", "#d95f02"]):
    m = np.where(fr & (name == nm) & (lev == 3e-5) & (sep == 20.0))[0]
    if m.size:
        i = m[0]
        ax.plot(waves, truth[i], "-", color=col, lw=1.5, label="%s injected" % nm)
        ax.errorbar(waves, f[i], sig[i], fmt="o", ms=2.5, color=col, alpha=.8,
                    label="%s recovered (PA %.0f)" % (nm, d['pa'][i]))
ax.set(xlabel="wavelength (um)", ylabel="contrast", title="Example: c=3e-5 at 400 mas")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(C.OUT + "/extraction.png", dpi=140)
print("\nwrote", C.OUT + "/extraction.png")
