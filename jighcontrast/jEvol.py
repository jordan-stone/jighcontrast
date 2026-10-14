'''Turn a contrast curve into a companion-mass sensitivity using substellar
evolutionary models and synthetic photometry.

The chain is: a detection limit is a flux ratio (companion/star) in the observing
band. For a companion of a given mass and age an evolutionary model gives its
effective temperature, gravity and radius; a grid of model atmosphere spectra
gives the band-averaged *surface* flux; multiplying by radius^2 gives the
in-band flux. The host star's in-band flux comes the same way. So

    model_contrast(mass, age) = [Fnu_comp(Teff,logg) * R_comp^2]
                                / [Fnu_star(Teff_star) * R_star^2]

is a pure flux ratio (the photometric zeropoint and distance cancel), and the
detectable mass at a given separation is the smallest mass whose model_contrast
reaches the measured contrast limit there.

Data formats assumed (all easily swapped):

* Sonora Diamondback (Morley 2024) evolution tables: whitespace columns
  ``age(Gyr)  M/Msun  logL/Lsun  Teff(K)  logg  R/Rsun`` with per-age header
  lines that are simply skipped.
* SCALES-resolution model spectra ``# microns, mJy`` (surface flux density Fnu),
  named ``..._t{Teff}g{glabel}{cloud}_m{feh}_co{co}.dat`` where the gravity label
  is ``g = 10^(logg-2)`` (so g316 -> logg 4.5) and ``cloud`` is ``nc`` or
  ``f{n}``.
* PHOENIX-ACES HiRes stellar spectra (F_lambda, erg/s/cm^2/cm) with wavelengths
  in a companion ``WAVE_...`` file (Angstrom).
'''

import glob
import os
import re

import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

R_SUN_CM = 6.957e10
C_CGS = 2.99792458e10
MJUP_MSUN = 9.5458e-4     # 1 Mjup in Msun
RJUP_RSUN = 0.100449      # 1 Rjup in Rsun (the Sonora tables' R column is in Rjup
                          # despite the "R/Rsun" header -- verified via L=4pi R^2 sigma T^4)


def band_average(wl_um, fnu, band):
    '''Tophat band-average of a flux density sampled at ``wl_um``.'''
    lo, hi = band
    m = (wl_um >= lo) & (wl_um <= hi)
    if m.sum() < 2:
        return np.nan
    return np.trapezoid(fnu[m], wl_um[m]) / (wl_um[m].max() - wl_um[m].min())


# --------------------------------------------------------------------------- #
# Evolutionary model
# --------------------------------------------------------------------------- #
def load_sonora_evolution(path):
    '''Read a Sonora evolution table into flat arrays.

    Returns a dict with 1-D arrays ``age`` (Gyr), ``mass`` (Msun), ``logL``,
    ``teff`` (K), ``logg``, ``r`` (Rsun), one entry per (age, mass) node. Per-age
    header lines (fewer than six numeric columns) are skipped automatically.'''
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 6:
                try:
                    rows.append([float(x) for x in p[:6]])
                except ValueError:
                    continue
    a = np.array(rows)
    return dict(age=a[:, 0], mass=a[:, 1], logL=a[:, 2],
                teff=a[:, 3], logg=a[:, 4], r=a[:, 5])


class Isochrones:
    '''Interpolator over an evolution table: (age, mass) -> Teff, logg, radius.

    Uses linear interpolation on (log10 age, mass); ``logg`` and ``log10 R`` are
    interpolated so the outputs stay smooth and positive.'''

    def __init__(self, evol):
        pts = np.column_stack([np.log10(evol['age']), evol['mass']])
        self._teff = LinearNDInterpolator(pts, evol['teff'])
        self._logg = LinearNDInterpolator(pts, evol['logg'])
        self._logr = LinearNDInterpolator(pts, np.log10(evol['r']))
        self.mass_range = (evol['mass'].min(), evol['mass'].max())

    def at(self, age_gyr, mass_msun):
        '''Return (Teff, logg, R) at one age for an array of masses, R in Rsun.

        (The table's radius column is in Rjup despite its "R/Rsun" header; it is
        converted to Rsun here so radii are consistent with the stellar hosts.)'''
        la = np.full_like(np.asarray(mass_msun, float), np.log10(age_gyr))
        teff = self._teff(la, mass_msun)
        logg = self._logg(la, mass_msun)
        r = 10.0 ** self._logr(la, mass_msun) * RJUP_RSUN
        return teff, logg, r


# --------------------------------------------------------------------------- #
# Companion synthetic photometry (Sonora Diamondback grid, hybrid clouds)
# --------------------------------------------------------------------------- #
_FNAME = re.compile(r'_t(\d+)g(\d+)(nc|f\d+)_m([+\-0-9.]+)_co')


def _glabel_to_logg(glabel):
    return np.log10(float(glabel)) + 2.0     # g316 -> logg 4.5


def diamondback_flux_grid(spectra_dir, band, feh='+0.5',
                          hybrid=(1300.0, 1000.0), warm_cloud='f2'):
    '''Band-averaged surface flux density Fnu(Teff, logg) for the Diamondback grid.

    Builds separate (Teff, logg) tables for the cloudy (``warm_cloud``, default
    f2) and cloudless (``nc``) spectra at metallicity ``feh``, then blends them
    into a *hybrid* sequence: cloudy above ``hybrid[0]`` K, cloudless below
    ``hybrid[1]`` K, linear in Teff between. Returns a callable
    ``fnu(Teff, logg)`` (RegularGridInterpolator).

    Outside the grid -- in particular companions colder than the coolest
    Diamondback model (900 K) -- ``fnu`` returns NaN unless ``extrapolate`` is
    passed. It must NOT clip to the edge: a 500 K planet assigned the 900 K flux
    is ~10x too bright in L, which makes low masses look detectable when they are
    not. Downstream, ``detectable_mass`` treats NaN as "no model" and reports NaN
    where the limit is fainter than the coldest model (see ``grid_floor_mass``).'''
    files = glob.glob(os.path.join(spectra_dir, '*_m%s_co*.dat' % feh))
    warm, cold = {}, {}
    for fn in files:
        m = _FNAME.search(os.path.basename(fn))
        if not m:
            continue
        teff = float(m.group(1)); logg = _glabel_to_logg(m.group(2)); cloud = m.group(3)
        if cloud not in (warm_cloud, 'nc'):
            continue
        d = np.loadtxt(fn)
        val = band_average(d[:, 0], d[:, 1], band)
        (warm if cloud == warm_cloud else cold)[(teff, logg)] = val

    teffs = np.array(sorted({k[0] for k in warm}))
    loggs = np.array(sorted({k[1] for k in warm}))
    t_hi, t_lo = hybrid
    grid = np.full((teffs.size, loggs.size), np.nan)
    for i, T in enumerate(teffs):
        w = np.clip((T - t_lo) / (t_hi - t_lo), 0.0, 1.0)   # 1=cloudy, 0=clear
        for j, g in enumerate(loggs):
            fw, fc = warm.get((T, g), np.nan), cold.get((T, g), np.nan)
            grid[i, j] = w * fw + (1.0 - w) * fc

    interp = RegularGridInterpolator((teffs, loggs), grid, bounds_error=False,
                                     fill_value=None)   # extrapolates; fnu() masks off-grid to NaN

    def fnu(teff, logg, extrapolate=False):
        teff = np.asarray(teff, float)
        logg = np.broadcast_to(np.asarray(logg, float), teff.shape)
        out = interp(np.column_stack([teff.ravel(), logg.ravel()])).reshape(teff.shape)
        if not extrapolate:
            off = ((teff < teffs.min()) | (teff > teffs.max())
                   | (logg < loggs.min()) | (logg > loggs.max()))
            out = np.where(off, np.nan, out)
        return out

    fnu.teff_grid, fnu.logg_grid = teffs, loggs
    return fnu


# --------------------------------------------------------------------------- #
# Host-star synthetic photometry (PHOENIX F_lambda -> surface Fnu)
# --------------------------------------------------------------------------- #
def phoenix_flux(phoenix_dir, band):
    '''Band-averaged surface flux density Fnu(Teff) from a PHOENIX HiRes grid.

    Reads every ``lteTTTTT-*.fits`` (F_lambda, erg/s/cm^2/cm) with the shared
    ``WAVE_*.fits`` (Angstrom), converts to Fnu (mJy) and band-averages. Returns
    a callable ``fnu(Teff)`` (log-flux interpolated in Teff; NaN outside the grid).'''
    wave = glob.glob(os.path.join(phoenix_dir, 'WAVE_*.fits'))[0]
    from astropy.io import fits
    wl = fits.open(wave)[0].data.astype(float)          # Angstrom
    wl_um = wl * 1e-4
    lam_cm = wl * 1e-8
    teffs, vals = [], []
    for fn in sorted(glob.glob(os.path.join(phoenix_dir, 'lte*.fits'))):
        T = float(os.path.basename(fn)[3:8])
        fl = fits.open(fn)[0].data.astype(float)        # F_lambda erg/s/cm2/cm
        fnu_mjy = fl * (lam_cm ** 2) / C_CGS * 1e26      # -> Fnu (mJy), surface
        teffs.append(T); vals.append(band_average(wl_um, fnu_mjy, band))
    teffs = np.array(teffs); vals = np.array(vals)
    order = np.argsort(teffs); teffs, vals = teffs[order], vals[order]
    logv = np.log10(vals)

    def fnu(teff):
        teff = np.asarray(teff, float)
        out = 10.0 ** np.interp(teff, teffs, logv)
        return np.where((teff < teffs.min()) | (teff > teffs.max()), np.nan, out)

    fnu.teff_grid = teffs
    return fnu


# --------------------------------------------------------------------------- #
# Contrast -> mass
# --------------------------------------------------------------------------- #
def model_contrast_vs_mass(age_gyr, host_teff, host_r_rsun, iso, comp_fnu,
                           star_fnu, mass_jup):
    '''Model flux ratio (companion/star) vs companion mass at one age.

    mass_jup : array of companion masses (Mjup).
    Returns (mass_jup, model_contrast) with model_contrast increasing with mass.
    Masses outside the evolution grid give NaN.'''
    mass_msun = np.asarray(mass_jup, float) * MJUP_MSUN
    teff_c, logg_c, r_c = iso.at(age_gyr, mass_msun)
    L_c = comp_fnu(teff_c, logg_c) * r_c ** 2
    L_star = star_fnu(host_teff) * host_r_rsun ** 2
    return np.asarray(mass_jup, float), L_c / L_star


def grid_floor_mass(age_gyr, iso, comp_fnu, mass_jup=None):
    '''Lowest companion mass (Mjup) at ``age_gyr`` that the atmosphere grid covers.

    Below this mass the companion is colder (or lower gravity) than any grid
    model, so ``model_contrast_vs_mass`` returns NaN and no mass limit can be
    quoted. A contrast limit fainter than the floor mass's contrast therefore
    only says "masses >= floor are detectable"; it does not reach lower masses.'''
    if mass_jup is None:
        mass_jup = np.geomspace(0.1, 80.0, 800)
    mass_jup = np.asarray(mass_jup, float)
    teff, logg, _ = iso.at(age_gyr, mass_jup * MJUP_MSUN)
    ok = np.isfinite(comp_fnu(teff, logg))
    return float(mass_jup[ok].min()) if ok.any() else np.nan


def detectable_mass(sep, contrast_limit, age_gyr, host_teff, host_r_rsun, iso,
                    comp_fnu, star_fnu, mass_jup=None):
    '''Smallest detectable companion mass (Mjup) at each separation.

    sep, contrast_limit : arrays of separation and 5-sigma contrast (flux ratio).
    Returns an array of masses (Mjup); NaN where the limit is beyond the model
    grid -- either brighter than the most massive model, or fainter than the
    coldest atmosphere model (then use ``grid_floor_mass`` as an upper bound: all
    masses above the floor are detectable, lower masses are unconstrained).'''
    if mass_jup is None:
        mass_jup = np.geomspace(0.6, 80.0, 400)
    m, mc = model_contrast_vs_mass(age_gyr, host_teff, host_r_rsun, iso, comp_fnu,
                                   star_fnu, mass_jup)
    good = np.isfinite(mc)
    m, mc = m[good], mc[good]
    order = np.argsort(mc)                       # monotonic in mass -> in mc
    mc_s, m_s = mc[order], m[order]
    out = np.interp(contrast_limit, mc_s, m_s, left=np.nan, right=np.nan)
    return out
