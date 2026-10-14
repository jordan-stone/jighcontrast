"""jEvol: companion photometry must not extrapolate off the atmosphere grid."""
import numpy as np

from jighcontrast import jEvol


def _write_grid(d):
    wl = np.linspace(2.8, 4.3, 200)
    for teff in (900, 1000, 1100):
        for g in (100, 316):
            for cloud in ('nc', 'f2'):
                fnu = (teff / 1000.0) ** 4 * np.ones_like(wl)
                np.savetxt(d / f"SCALES_L_t{teff}g{g}{cloud}_m+0.5_co1.0.dat",
                           np.column_stack([wl, fnu]), header="microns mJy")


def test_off_grid_is_nan_not_clipped(tmp_path):
    _write_grid(tmp_path)
    fnu = jEvol.diamondback_flux_grid(str(tmp_path), (2.91, 4.14))
    on = fnu(np.array([950.0]), np.array([4.25]))
    assert np.isfinite(on).all()
    # colder than the 900 K floor, or off the gravity grid -> NaN (was: edge value)
    assert np.isnan(fnu(np.array([600.0]), np.array([4.25]))).all()
    assert np.isnan(fnu(np.array([950.0]), np.array([3.0]))).all()
    assert np.isfinite(fnu(np.array([600.0]), np.array([4.25]), extrapolate=True)).all()


def test_detectable_mass_nan_below_floor():
    class Iso:
        def at(self, age, mass_msun):
            m = np.asarray(mass_msun) / jEvol.MJUP_MSUN
            return 100.0 * m, np.full_like(m, 4.0), np.full_like(m, 0.1)
    comp = lambda teff, logg: np.where(teff >= 900, (teff / 1000.0) ** 4, np.nan)
    star = lambda teff: 1.0
    floor = jEvol.grid_floor_mass(0.1, Iso(), comp)
    assert 8.9 < floor < 9.1
    _, mc = jEvol.model_contrast_vs_mass(0.1, 5772, 1.0, Iso(), comp, star, [9.0])
    m = jEvol.detectable_mass([1, 2], np.array([mc[0] * 10, mc[0] * 0.01]), 0.1, 5772, 1.0,
                              Iso(), comp, star)
    assert np.isfinite(m[0]) and np.isnan(m[1])     # fainter than floor -> no claim
