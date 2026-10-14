#!/usr/bin/env python
"""V1. Forward model vs brute-force injection on the real SCALES sequence.

For each (separation, contrast, centering) a flat-spectrum source is injected
into the coronagraphic cube, the cube is KLIP-reduced, and the source-free
reduction is subtracted: what remains is exactly how KLIP processed the source
(speckle noise cancels). That brute-force image is compared with the KLIP-FM
prediction ``c * F`` from ``klip_fm_reduce``:

  * flux ratio   = aperture(brute) / aperture(FM)        (1 = throughput right)
  * shape error  = ||brute - FM|| / ||brute||  in the 1 lambda/D fit region
  * dlam         = max |delta Lambda / Lambda| at the injected flux

as a function of contrast (first-order validity), separation and centering.
Writes results/fm_vs_brute.npz and results/fm_vs_brute.png.
"""
import time
import numpy as np

import common as C
from jighcontrast import jFM

SEPS = [10.0, 15.0, 20.0, 25.0]
PA_INJ = 37.0                                  # opposite the real companion (217)
CONTRASTS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
CHANNELS = np.arange(0, 54, 6)                 # 9 channels spanning the band
NPCA = 20


def main():
    coro, stamps, pa = C.load()
    cube = np.asarray(coro[:, CHANNELS], float)
    waves, st = C.WAVES[CHANNELS], stamps[CHANNELS]
    kw = dict(npca=NPCA, n_arcs=4, dr=3.5, exclusion_fwhm=0.75, n_jobs=C.N_JOBS)
    res = {}
    t0 = time.time()
    for cen in ("median", "frame"):
        for sep in SEPS:
            base = jFM.klip_fm_reduce(cube, pa, waves, C.CENTER, sep, PA_INJ, st, C.D_M,
                                      C.PIX_MAS, centering=cen, forward_model=False, **kw)
            for c in CONTRASTS:
                inj = jFM.inject_source(cube, st, pa, C.CENTER, sep, PA_INJ, c)
                red = jFM.klip_fm_reduce(inj, pa, waves, C.CENTER, sep, PA_INJ, st, C.D_M,
                                         C.PIX_MAS, centering=cen, **kw)
                x0, y0 = red['stamp_origin']
                sx, sy = red['src_xy']
                h = red['F'].shape[-1]
                ratio, shape_err = [], []
                for p in range(len(waves)):
                    brute = (red['P'][p] - base['P'][p])[y0:y0 + h, x0:x0 + h]
                    fm = c * red['F'][p, p]
                    rfit = jFM.resolution_element_pix(waves[p], C.D_M, C.PIX_MAS)
                    yy, xx = np.indices(fm.shape)
                    m = (np.hypot(xx - (sx - x0), yy - (sy - y0)) <= rfit) & np.isfinite(brute)
                    ratio.append(brute[m].sum() / fm[m].sum())
                    shape_err.append(np.linalg.norm(brute[m] - fm[m]) / np.linalg.norm(brute[m]))
                ap = [C.aperture(c * red['F'][p, p], sx - x0, sy - y0, 0.5 *
                                 jFM.resolution_element_pix(waves[p], C.D_M, C.PIX_MAS)) /
                      C.aperture(c * st[p], 9, 9, 0.5 * jFM.resolution_element_pix(
                          waves[p], C.D_M, C.PIX_MAS)) for p in range(len(waves))]
                res[(cen, sep, c)] = dict(ratio=np.array(ratio), shape_err=np.array(shape_err),
                                          dlam=c * red['dlam'], tau=np.array(ap))
                print("%-6s sep %4.1f c %.0e : brute/FM %.3f (min %.3f max %.3f)  shape err %.3f  "
                      "dlam %.2g  tau %.2f   [%.0fs]"
                      % (cen, sep, c, np.median(ratio), np.min(ratio), np.max(ratio),
                         np.median(shape_err), np.max(c * red['dlam']), np.median(ap),
                         time.time() - t0), flush=True)
    keys = list(res)
    np.savez(C.OUT + "/fm_vs_brute.npz",
             keys=np.array([str(k) for k in keys]),
             ratio=np.array([res[k]['ratio'] for k in keys]),
             shape_err=np.array([res[k]['shape_err'] for k in keys]),
             dlam=np.array([res[k]['dlam'] for k in keys]),
             tau=np.array([res[k]['tau'] for k in keys]), channels=CHANNELS)
    plot(res)


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    cols = dict(zip(SEPS, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]))
    for (cen, sep, c), r in res.items():
        ls = "-" if cen == "frame" else "--"
        mk = "o" if cen == "frame" else "s"
        axs[0].plot(c, np.median(r['ratio']), mk, color=cols[sep], mfc="none" if cen == "median" else cols[sep])
        axs[2].plot(r['dlam'], r['shape_err'], mk, color=cols[sep], ms=3, alpha=.6,
                    mfc="none" if cen == "median" else cols[sep])
    for cen, ls in (("frame", "-"), ("median", "--")):
        for sep in SEPS:
            cs = CONTRASTS
            axs[0].plot(cs, [np.median(res[(cen, sep, c)]['ratio']) for c in cs], ls, color=cols[sep],
                        label="%s, %.0f px" % (cen, sep))
            axs[1].plot(cs, [np.median(res[(cen, sep, c)]['tau']) for c in cs], ls, color=cols[sep])
    axs[0].axhline(1, color="k", lw=.8)
    axs[0].set(xscale="log", xlabel="injected contrast", ylabel="brute-force / KLIP-FM flux",
               title="Forward-model photometric accuracy")
    axs[0].legend(fontsize=7, ncol=2)
    axs[1].set(xscale="log", xlabel="injected contrast", ylabel="KLIP throughput (FM)",
               title="Throughput (0.5 lambda/D aperture)")
    axs[2].set(xscale="log", yscale="log", xlabel="|delta Lambda / Lambda| at injected flux",
               ylabel="||brute - FM|| / ||brute||", title="Validity diagnostic vs model error")
    fig.tight_layout()
    fig.savefig(C.OUT + "/fm_vs_brute.png", dpi=150)


if __name__ == "__main__":
    main()
