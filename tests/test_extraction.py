"""Spectral extraction (Pueyo 2016 Appendix F) on synthetic ADI + IFS data."""
import numpy as np
import pytest

from jighcontrast import jFM
from synthetic import make_dataset, CENTER, D, PIXSCALE

SHAPE = np.array([2.0, 3.0, 2.5, 1.5])
SEP, PA0 = 12.0, 40.0


@pytest.fixture(scope="module")
def ds():
    return make_dataset()


def _speckle_free(ds, truth, centering, spectral_refs=False, npca=5):
    """Extract from (injected reduction - source-free reduction): isolates the
    forward-model accuracy from speckle noise."""
    base = jFM.klip_fm_reduce(ds['cube'], ds['pa'], ds['waves'], CENTER, SEP, PA0,
                              ds['psf_stamps'], D, PIXSCALE, npca=npca, centering=centering,
                              spectral_refs=spectral_refs, forward_model=False)
    cube = jFM.inject_source(ds['cube'], ds['psf_stamps'], ds['pa'], CENTER, SEP, PA0, truth)
    red = jFM.klip_fm_reduce(cube, ds['pa'], ds['waves'], CENTER, SEP, PA0, ds['psf_stamps'],
                             D, PIXSCALE, npca=npca, centering=centering,
                             spectral_refs=spectral_refs)
    red['P'] = red['P'] - base['P']
    r = jFM.extract_spectrum(None, None, ds['pa'], ds['waves'], CENTER, SEP, PA0, D, PIXSCALE,
                             psf_stamps=ds['psf_stamps'], reduction=red, verbose=False)
    return r['spectrum'], red


def test_frame_centering_recovers_spectrum_exactly_in_linear_regime(ds):
    for scale, tol in [(1e-7, 3e-3), (1e-8, 1e-3)]:
        f, _ = _speckle_free(ds, SHAPE * scale, 'frame')
        assert np.max(np.abs(f / (SHAPE * scale) - 1)) < tol


def test_median_centering_bias_is_small_and_flux_independent(ds):
    f7, _ = _speckle_free(ds, SHAPE * 1e-7, 'median')
    f8, _ = _speckle_free(ds, SHAPE * 1e-8, 'median')
    r7, r8 = f7 / (SHAPE * 1e-7), f8 / (SHAPE * 1e-8)
    assert np.max(np.abs(r8 - 1)) < 0.03          # the neglected median term
    assert np.max(np.abs(r7 - r8)) < 0.01         # ... is first order: constant ratio


def test_validity_diagnostic_tracks_nonlinearity(ds):
    """dlam (|delta Lambda/Lambda| at the true flux) small -> accurate; large -> biased."""
    f_lo, red_lo = _speckle_free(ds, SHAPE * 1e-8, 'frame')
    f_hi, red_hi = _speckle_free(ds, SHAPE * 1e-5, 'frame')
    assert np.max(red_lo['dlam'] * SHAPE * 1e-8) < 0.01
    assert np.max(red_hi['dlam'] * SHAPE * 1e-5) > 0.5
    assert np.max(np.abs(f_hi / (SHAPE * 1e-5) - 1)) > 0.05


def test_adi_per_channel_model_is_diagonal(ds):
    _, red = _speckle_free(ds, SHAPE * 1e-8, 'frame')
    F = red['F']
    off = [np.abs(F[p, q]).max() for p in range(4) for q in range(4) if p != q]
    assert max(off) == 0.0
    assert all(np.abs(F[p, p]).max() > 0 for p in range(4))


def test_spectral_refs_coupled_solve(ds):
    """With cross-wavelength references F couples channels; the joint solve must
    still recover the injected spectrum."""
    f, red = _speckle_free(ds, SHAPE * 1e-8, 'frame', spectral_refs=True)
    assert np.abs(red['F'][0, 1]).max() > 0
    assert np.max(np.abs(f / (SHAPE * 1e-8) - 1)) < 5e-3


def test_extraction_is_linear_in_the_data(ds):
    """The estimator is linear: doubling the residual image doubles f."""
    cube = jFM.inject_source(ds['cube'], ds['psf_stamps'], ds['pa'], CENTER, SEP, PA0, SHAPE * 1e-6)
    r = jFM.extract_spectrum(cube, None, ds['pa'], ds['waves'], CENTER, SEP, PA0, D, PIXSCALE,
                             npca=5, psf_stamps=ds['psf_stamps'], refine=0, verbose=False)
    red = dict(r['reduction'])
    red['P'] = 2 * red['P']
    r2 = jFM.extract_spectrum(None, None, ds['pa'], ds['waves'], CENTER, SEP, PA0, D, PIXSCALE,
                              psf_stamps=ds['psf_stamps'], reduction=red, verbose=False)
    assert np.allclose(r2['spectrum'], 2 * r['spectrum'])
    assert np.allclose(r2['sigma'], 2 * r['sigma'])


def test_error_bars_are_calibrated():
    """Over independent noise realizations the pulls (f - truth)/sigma should be
    ~N(0, 1): mean ~ 0, rms ~ 1 (generous tolerance for the sample size)."""
    pulls = []
    truth = SHAPE * 2e-6
    for seed in range(12):
        d = make_dataset(seed=100 + seed, white=0.3, drift=0.4)
        cube = jFM.inject_source(d['cube'], d['psf_stamps'], d['pa'], CENTER, SEP, PA0, truth)
        r = jFM.extract_spectrum(cube, None, d['pa'], d['waves'], CENTER, SEP, PA0, D, PIXSCALE,
                                 npca=5, psf_stamps=d['psf_stamps'], centering='frame',
                                 verbose=False)
        pulls.append((r['spectrum'] - truth) / r['sigma'])
    pulls = np.concatenate(pulls)
    assert abs(np.mean(pulls)) < 0.5
    assert 0.6 < np.std(pulls) < 1.6


def test_broadband_gls():
    f = np.array([1.0, 1.2, 0.8, 1.0])
    res = dict(spectrum=f, cov=np.diag([0.01, 0.01, 0.04, 0.04]))
    bb = jFM.broadband_contrast(res)
    w = 1 / np.array([0.01, 0.01, 0.04, 0.04])
    assert np.isclose(bb['contrast'], np.sum(w * f) / np.sum(w))
    assert np.isclose(bb['sigma'], np.sum(w) ** -0.5)
    assert bb['dof'] == 3


def test_refinement_removes_bright_source_bias(ds):
    """A bright source (first-order model under-corrects by ~20%) is recovered
    by negative-injection refinement."""
    truth = SHAPE * 1e-5
    cube = jFM.inject_source(ds['cube'], ds['psf_stamps'], ds['pa'], CENTER, SEP, PA0, truth)
    kw = dict(npca=5, psf_stamps=ds['psf_stamps'], verbose=False)
    one = jFM.extract_spectrum(cube, None, ds['pa'], ds['waves'], CENTER, SEP, PA0, D,
                               PIXSCALE, refine=0, **kw)
    ref = jFM.extract_spectrum(cube, None, ds['pa'], ds['waves'], CENTER, SEP, PA0, D,
                               PIXSCALE, refine=6, **kw)
    # single pass: biased by several sigma; refined: within the noise. (Use the
    # refined sigma for both: the single-pass ring still contains the bright
    # source's residual arcs, which inflate its noise estimate.)
    assert np.max(np.abs(one['spectrum'] - truth) / ref['sigma']) > 3
    assert np.max(np.abs(ref['spectrum'] - truth) / ref['sigma']) < 2
    assert ref['n_iter'] >= 1 and ref['history'].shape[0] == ref['n_iter'] + 1


def test_broadband_error_is_calibrated_empirically():
    """The broadband amplitude's error comes from the ring, so its pulls are
    ~N(0,1) even though the synthetic speckle noise is strongly correlated
    across channels."""
    pulls = []
    truth = SHAPE * 2e-6
    for seed in range(30):
        d = make_dataset(seed=500 + seed, white=0.3, drift=0.4)
        cube = jFM.inject_source(d['cube'], d['psf_stamps'], d['pa'], CENTER, SEP, PA0, truth)
        r = jFM.extract_spectrum(cube, None, d['pa'], d['waves'], CENTER, SEP, PA0, D, PIXSCALE,
                                 npca=5, psf_stamps=d['psf_stamps'], verbose=False)
        bb = jFM.broadband_contrast(r, spectrum_shape=SHAPE)
        pulls.append((bb['contrast'] - 2e-6) / bb['sigma'])
    pulls = np.array(pulls)
    assert abs(pulls.mean()) < 0.6
    assert 0.6 < pulls.std() < 1.5
