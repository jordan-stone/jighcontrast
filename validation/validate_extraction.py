#!/usr/bin/env python
"""V2. Spectral-extraction injection-recovery on the real SCALES sequence.

Sources with known contrast spectra (Sonora Bobcat L-type 2000 K and methane
T-type 700 K models over the 8000 K star, data/injection_shapes.csv) are
injected at several brightnesses,
separations and position angles into the registered, companion-subtracted
cube (make_clean_cube.py; the companion's position is kept out of the noise
ring) and
extracted with ``jFM.extract_spectrum`` over all 54 channels.

Reported:
  * bias            mean over positions of f_hat / f_true, per channel
  * pulls           (f_hat - f_true) / sigma  -- should be ~ N(0, 1)
  * chi2            (f_hat - f_true)^T cov^-1 (f_hat - f_true) / n  -- ~1 if the
                    full covariance (incl. channel correlations) is right
  * naive           a KLIP spectrum fit with the raw PSF (no forward model) --
                    what you get without correcting for over/self-subtraction
  * broadband S/N   GLS-combined significance vs the injected contrast
Writes results/extraction.npz and results/extraction_*.png.
"""
import os
import sys
import time
import numpy as np

import common as C
from jighcontrast import jFM

SEPS = [12.0, 20.0, 28.0]
PAS = [37.0, 107.0, 287.0, 337.0]               # avoid the companion at 217
LEVELS = [1e-5, 3e-5, 1e-4, 3e-4]                # band-mean contrast
NPCA = 20
SHAPES_FILE = os.path.join(C.HERE, "data", "injection_shapes.csv")
SPECTRA = {"L 2000 K": 1, "T 700 K": 2}          # columns of SHAPES_FILE


def contrast_shape(column):
    """Test contrast spectrum (mean 1): a Sonora Bobcat model over an 8000 K
    PHOENIX star, through the simulated instrument's response
    (build_truth_spectrum.py writes data/injection_shapes.csv)."""
    return np.loadtxt(SHAPES_FILE, delimiter=",")[:, column]


def naive_spectrum(red, stamps, waves):
    """Fit the reduced data with the RAW off-axis PSF (no KLIP forward model)."""
    sx, sy = red['src_xy']
    out = np.zeros(len(waves))
    for p in range(len(waves)):
        r = jFM.resolution_element_pix(waves[p], C.D_M, C.PIX_MAS)
        ny, nx = red['P'][p].shape
        psf = jFM.place_psf((ny, nx), stamps[p], sx, sy)
        yy, xx = np.indices((ny, nx))
        m = (np.hypot(xx - sx, yy - sy) <= r) & np.isfinite(red['P'][p])
        out[p] = np.sum(psf[m] * red['P'][p][m]) / np.sum(psf[m] ** 2)
    return out


def run_one(cube, stamps, pa, sep, pa_deg, truth, centering):
    inj = jFM.inject_source(cube, stamps, pa, C.CENTER, sep, pa_deg, truth)
    red0 = jFM.klip_fm_reduce(inj, pa, C.WAVES, C.CENTER, sep, pa_deg, stamps, C.D_M,
                              C.PIX_MAS, npca=NPCA, centering=centering, n_jobs=C.N_JOBS)
    naive = naive_spectrum(red0, stamps, C.WAVES)       # first pass: source present
    r = jFM.extract_spectrum(inj, None, pa, C.WAVES, C.CENTER, sep, pa_deg, C.D_M, C.PIX_MAS,
                             npca=NPCA, psf_stamps=stamps, centering=centering,
                             exclude=[(C.COMP_SEP, C.COMP_PA)], n_jobs=C.N_JOBS,
                             reduction=red0, verbose=False)
    return dict(f=r['spectrum'], f1=r['history'][0], n_iter=r['n_iter'], sigma=r['sigma'],
                cov=r['cov'], bb=r['broadband'], dlam=r['dlam_fit'], naive=naive,
                tau=r['throughput'], n_ring=r['n_ring'])


def main():
    coro, stamps, pa = C.load()
    cube = np.asarray(coro, float)
    shapes = {k: contrast_shape(v) for k, v in SPECTRA.items()}
    cases = []
    for name, shp in shapes.items():
        for lev in LEVELS:
            for sep in SEPS:
                for pa_deg in PAS:
                    cases.append((name, lev, sep, pa_deg, "frame"))
    # (median-image centering is characterized by V1, validate_forward_model.py)
    if len(sys.argv) > 1:                       # quick test: first N cases
        cases = cases[:int(sys.argv[1])]
    case_dir = os.path.join(C.OUT, "extraction_cases")       # one file per case:
    os.makedirs(case_dir, exist_ok=True)                      # resumable after a stop
    t0 = time.time()
    for i, (name, lev, sep, pa_deg, cen) in enumerate(cases):
        path = os.path.join(case_dir, "case_%03d.npz" % i)
        if os.path.exists(path):
            continue
        truth = lev * shapes[name]
        o = run_one(cube, stamps, pa, sep, pa_deg, truth, cen)
        d = o['f'] - truth
        chi2 = float(d @ np.linalg.solve(o['cov'], d)) / d.size
        np.savez(path, f=o['f'], f1=o['f1'], n_iter=o['n_iter'], sigma=o['sigma'],
                 cov=o['cov'], truth=truth, naive=o['naive'], dlam=o['dlam'], tau=o['tau'],
                 name=name, level=lev, sep=sep, pa=pa_deg, centering=cen, chi2=chi2,
                 bb_snr=o['bb']['snr'], bb_contrast=o['bb']['contrast'],
                 bb_sigma=o['bb']['sigma'], n_ring=o['n_ring'])
        print("[%3d/%d %5.0fs] %-8s c=%.0e sep=%4.1f pa=%5.1f %-6s  rec/true(median) %.3f  "
              "1-pass %.3f  naive/true %.3f  pull rms %.2f  chi2/n %.2f  bbSNR %.1f  dlam %.2g"
              % (i + 1, len(cases), time.time() - t0, name, lev, sep, pa_deg, cen,
                 np.median(o['f'] / truth), np.median(o['f1'] / truth),
                 np.median(o['naive'] / truth),
                 np.sqrt(np.mean((d / o['sigma']) ** 2)), chi2, o['bb']['snr'],
                 np.max(o['dlam'])), flush=True)
    assemble(case_dir, len(cases))


def assemble(case_dir, n):
    rows = [dict(np.load(os.path.join(case_dir, "case_%03d.npz" % i))) for i in range(n)]
    keys = ['f', 'f1', 'n_iter', 'sigma', 'truth', 'naive', 'dlam', 'tau', 'cov', 'name',
            'level', 'sep', 'pa', 'centering', 'chi2', 'bb_snr', 'bb_contrast', 'bb_sigma',
            'n_ring']
    np.savez(C.OUT + "/extraction.npz", waves=C.WAVES,
             **{k: np.array([r[k] for r in rows]) for k in keys})
    print("assembled %d cases -> %s" % (n, C.OUT + "/extraction.npz"))


if __name__ == "__main__":
    main()
