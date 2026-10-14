#!/usr/bin/env python
"""Reconstruct the contrast spectrum injected into the SCALES HIP 21152 simulation.

The example data were made with scalesETC (https://github.com/scalessim/scalesETC).
Its chain for a point source (targs.py, io.py, focal_plane.py) is:
  1. the planet is a Sonora Bobcat cloudless model ``sp_t{Teff}g{g}nc_m0.0`` and the
     star a PHOENIX-ACES model ``lte{Teff}-4.50-0.0``, each scaled so that its MEAN
     F_lambda over |lambda - 3.4 um| < 0.45 um matches its L magnitude (L = 14.66
     and 5.3 here);
  2. F_lambda is rebinned (flux-conserving, ``spectres``) onto the projection grid,
     converted to photons (x lambda) and multiplied by the throughput;
  3. a projection matrix disperses each spaxel's spectrum onto the detector and a
     rectification matrix extracts the 54-channel cube.
Steps 2-3 are linear, so the channel spectrum of a point source is K @ (T * F * lambda),
with K (54 x n_lambda) the response of the companion's spaxels to each projection
wavelength. This script builds K from the ETC's own matrices (averaged along the
companion's track across the detector), applies it to the star and to candidate
companion models, and writes the channel contrast spectra.

The simulation used: planet 1400 K, g = 316 m/s^2 (log g 4.5), L = 14.66; star
10000 K, log g 4.5, [Fe/H] 0.0, L = 5.35 (the file names round this to 5p3); the
L magnitudes set the normalization (distance and radii do not matter).

Requires the scalesETC data directory (projmats/, rectmats/, sonora_2018/,
ifs_filters/, skytrans/, PHOENIX_HiRes/). Not needed to run the examples:
the result is stored in data/hip21152b_injected_contrast.csv.

Usage: build_truth_spectrum.py <etc_data dir> <scalesETC repo dir>
"""
import os
import sys
import numpy as np
from astropy.io import fits
from scipy.sparse import load_npz

from jighcontrast import jFM

HERE = os.path.dirname(os.path.abspath(__file__))
STAR_L, COMP_L = 5.35, 14.66
STAR_FILE = "lte10000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
CENTER, SEP, PA = (54.0, 54.0), 19.5, 217.0   # scalesETC places the star on pixel 54
CANDIDATES = [(1400, 316), (1300, 316), (1200, 316), (1400, 1000), (1300, 1000),
              (1400, 178), (1300, 178)]
INJECTION_SHAPES = {"L 2000 K": (2000, 316), "T 700 K": (700, 316)}   # for V2


def band_scaled(wav, flam, mag):
    """scalesETC normalization: mean F_lambda (over the native samples) in the L band."""
    m = np.abs(wav - 3.4) < 0.45
    return flam * 10 ** (-0.4 * mag) / np.mean(flam[m])


def main(etc, etc_repo):
    sys.path.insert(0, etc_repo)
    from scalesETC.io import spectres

    plams = np.loadtxt(os.path.join(etc, "projmats", "Low-Res_L_lams.txt"))
    rlams = np.loadtxt(os.path.join(etc, "rectmats", "Low-Res_L_lams.txt"))
    P = load_npz(os.path.join(etc, "projmats", "Low-Res_L.npz")).tocsc()
    R = load_npz(os.path.join(etc, "rectmats", "Low-Res_L_ones.npz")).tocsr()
    nl, ny, nx = plams.size, 108, 108

    # channel response of the companion's spaxels, averaged along its track
    pa_track = np.loadtxt(os.path.join(HERE, "..", "examples", "parallactic_angle_degrees.csv"),
                          delimiter=",")
    K = np.zeros((rlams.size, nl))
    spaxels = sorted({tuple(np.round(jFM.companion_frame_position(SEP, PA, -q, CENTER)).astype(int))
                      for q in pa_track[::10]})
    for x, y in spaxels:
        cols = [l * ny * nx + y * nx + x for l in range(nl)]
        out = (R @ P[:, cols]).toarray().reshape(rlams.size, ny, nx, nl)
        K += out[:, y - 1:y + 2, x - 1:x + 2].sum(axis=(1, 2))
    K /= len(spaxels)

    # throughput common to star and planet (filter x Mauna Kea sky, vapor 1, airmass 1)
    filt = np.loadtxt(os.path.join(etc, "ifs_filters", "asahi_2.9_4.15.txt"))
    sky = np.loadtxt(os.path.join(etc, "skytrans", "mktrans_zm_10_10.dat"))
    thr = {"none": np.ones(nl),
           "filter x sky": (np.interp(plams, filt[:, 0], filt[:, 1])
                            * spectres(plams, sky[:, 0], sky[:, 1]))}

    def channel_counts(wav, flam, T):
        return K @ (T * spectres(plams, wav, flam) * plams)

    wav_phx = fits.getdata(os.path.join(etc, "PHOENIX_HiRes",
                                        "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")) / 1e4
    m = (wav_phx > 2.5) & (wav_phx < 4.6)
    star = band_scaled(wav_phx, fits.getdata(os.path.join(etc, "PHOENIX_HiRes", STAR_FILE)).astype(float),
                       STAR_L)

    out = {"wavelength_um": rlams}
    for teff, g in CANDIDATES:
        a = np.loadtxt(os.path.join(etc, "sonora_2018", "spectra",
                                    "sp_t%dg%dnc_m0.0.gz" % (teff, g)), skiprows=2)[::-1]
        wav_p = a[:, 0]
        flam_p = a[:, 1] * 2.998e10 / (wav_p * 1e-4) ** 2          # F_nu -> F_lambda shape
        sel = (wav_p > 0.8) & (wav_p < 5.6)
        planet = band_scaled(wav_p[sel], flam_p[sel], COMP_L)
        for tname, T in thr.items():
            c = (channel_counts(wav_p[sel], planet, T)
                 / channel_counts(wav_phx[m], star[m], T))
            out["T%d_g%d|%s" % (teff, g, tname)] = c
        print("Teff %d g %4d: band-mean contrast %.3e  (filter x sky vs none: max diff %.1f%%)"
              % (teff, g, out["T%d_g%d|none" % (teff, g)].mean(),
                 100 * np.max(np.abs(out["T%d_g%d|filter x sky" % (teff, g)]
                                     / out["T%d_g%d|none" % (teff, g)] - 1))))
    np.savez(os.path.join(HERE, "data", "truth_candidates.npz"), **out)
    np.savetxt(os.path.join(HERE, "data", "hip21152b_injected_contrast.csv"),
               np.column_stack([rlams, out["T1400_g316|filter x sky"]]), delimiter=",",
               fmt=["%.6f", "%.6e"],
               header="HIP 21152 B as injected into the scalesETC SCALES L-band simulation:\n"
                      "companion/star flux ratio per IFS channel. Companion: Sonora Bobcat\n"
                      "cloudless, Teff 1400 K, g 316 m/s^2 (log g 4.5), L = 14.66. Star:\n"
                      "PHOENIX-ACES 10000 K, log g 4.5, L = 5.35. Through the simulator's\n"
                      "projection and rectification at the companion's spaxels, with filter and\n"
                      "Mauna Kea sky transmission (build_truth_spectrum.py).\n"
                      "columns: wavelength_um, contrast")

    # contrast SHAPES (mean 1) used as test spectra in validate_extraction.py -- made
    # once (with an 8000 K star, before the simulation's parameters were known) and
    # kept so V2 stays reproducible; the star's shape barely matters in L (<0.2%).
    if os.path.exists(os.path.join(HERE, "data", "injection_shapes.csv")):
        print("wrote data/truth_candidates.npz, data/hip21152b_injected_contrast.csv")
        return
    cols, names = [rlams], ["wavelength_um"]
    for name, (teff, g) in INJECTION_SHAPES.items():
        a = np.loadtxt(os.path.join(etc, "sonora_2018", "spectra",
                                    "sp_t%dg%dnc_m0.0.gz" % (teff, g)), skiprows=2)[::-1]
        sel = (a[:, 0] > 0.8) & (a[:, 0] < 5.6)
        flam = a[sel, 1] * 2.998e10 / (a[sel, 0] * 1e-4) ** 2
        c = (channel_counts(a[sel, 0], flam, thr["filter x sky"])
             / channel_counts(wav_phx[m], star[m], thr["filter x sky"]))
        cols.append(c / c.mean()); names.append(name.replace(" ", "_"))
    np.savetxt(os.path.join(HERE, "data", "injection_shapes.csv"), np.column_stack(cols),
               delimiter=",", fmt="%.6e",
               header="Sonora Bobcat (cloudless, g=316) / PHOENIX 8000 K contrast SHAPES (mean 1)\n"
                      "through the scalesETC SCALES L low-res response; test spectra for V2.\n"
                      + ",".join(names))
    print("wrote data/truth_candidates.npz (%d spaxels along the track)" % len(spaxels))


if __name__ == "__main__":
    main(*sys.argv[1:3])
