"""contrast_curve: the matched-filter template must be registered with, and
normalized like, the derotated data."""
import numpy as np
import pytest

from jighcontrast import jFM
from synthetic import make_dataset, CENTER, D, PIXSCALE

SEP, PA0 = 12.0, 40.0


def _nocoro(ds):
    n_exp, n_wave, ny, nx = ds['cube'].shape
    noco = np.zeros_like(ds['cube'])
    for w in range(n_wave):
        noco[:, w] = jFM.place_psf((ny, nx), ds['psf_stamps'][w], *CENTER)[None]
    return noco


@pytest.fixture(scope="module")
def ds():
    d = make_dataset(white=0.05, drift=0.05)
    d['nocoro'] = _nocoro(d)
    return d


@pytest.mark.parametrize("n_arcs", [4, 1])
def test_matched_filter_template_returns_injected_contrast(ds, n_arcs):
    """Fitting the forward-modeled template to what KLIP leaves of an injected
    source returns its contrast (throughput cancels). Fails if the template is
    not derotated/registered/normalized like the data. The source's imprint is
    isolated as (reduction with source) - (reduction without), and ONE template
    is applied to it: differencing two separately-templated amplitudes instead
    would multiply the template's own O(c) change by the large speckle residual."""
    c = 3e-9                                      # deep in the linear regime
    cube = jFM.inject_source(ds['cube'], ds['psf_stamps'], ds['pa'], CENTER, SEP, PA0, c)
    kw = dict(npca=5, n_arcs=n_arcs, source_pa=PA0, n_probe=1, combine='mean',
              centering='frame', return_images=True, verbose=False)
    r1 = jFM.contrast_curve(cube, ds['nocoro'], ds['pa'], ds['waves'], CENTER, [SEP], D,
                            PIXSCALE, **kw)
    r0 = jFM.contrast_curve(ds['cube'], ds['nocoro'], ds['pa'], ds['waves'], CENTER, [SEP], D,
                            PIXSCALE, **kw)
    sx = CENTER[0] - SEP * np.sin(np.deg2rad(PA0))
    sy = CENTER[1] + SEP * np.cos(np.deg2rad(PA0))
    ix, iy = int(round(sx)), int(round(sy))
    for w in range(len(ds['waves'])):
        T = r1['templates'][0][w][0]
        h = T.shape[0] // 2
        diff = (r1['reduced'][0, w] - r0['reduced'][0, w])[iy - h:iy + h + 1, ix - h:ix + h + 1]
        m = np.isfinite(T) & np.isfinite(diff)        # as matched_filter_ring does
        assert np.isclose(np.sum(T[m] * diff[m]) / np.sum(T[m] ** 2), c, rtol=0.01)
    assert np.all((r1['tau'][0] > 0.2) & (r1['tau'][0] < 1.05))


def test_parallel_matches_serial(ds):
    kw = dict(npca=5, n_probe=2, centering='frame', return_resid=True, verbose=False)
    a = jFM.contrast_curve(ds['cube'], ds['nocoro'], ds['pa'], ds['waves'], CENTER,
                           [10.0, 14.0], D, PIXSCALE, **kw)
    b = jFM.contrast_curve(ds['cube'], ds['nocoro'], ds['pa'], ds['waves'], CENTER,
                           [10.0, 14.0], D, PIXSCALE, n_jobs=2, **kw)
    for k in ('mf_noise', 'tau', 'fstar', 'n_aper', 'nsig_eff'):
        assert np.allclose(a[k], b[k], equal_nan=True), k
    assert all(np.allclose(x, y, equal_nan=True) for x, y in zip(a['resid_pix'], b['resid_pix']))


def test_empirical_combination_sees_channel_correlation():
    """Perfectly correlated channels give no sqrt(N) gain: the empirical combined
    noise equals the single-channel noise, while the independent-channel formula
    is overconfident by sqrt(N)."""
    rng = np.random.default_rng(0)
    n_wave, n_ring = 8, 40
    common = rng.normal(size=n_ring)
    amps = common[None, :] + 1e-3 * rng.normal(size=(n_wave, n_ring))
    res = dict(radii=np.array([10.0]), waves=np.linspace(3, 4, n_wave), ring_amps=[amps],
               mf_noise=np.full((1, n_wave), 1.0), nsig_eff=np.array([5.0]),
               sigma_threshold=5.0, small_sample=False)
    _, c_emp, n = jFM.combine_contrast_empirical(res)
    _, c_ind = jFM.combine_contrast(res)
    assert n[0] == n_ring
    assert np.isclose(c_emp[0], 5.0 * np.std(common, ddof=1), rtol=0.05)
    assert np.isclose(c_ind[0], 5.0 / np.sqrt(n_wave))
    # independent channels: empirical recovers the sqrt(N) gain
    res['ring_amps'] = [rng.normal(size=(n_wave, 400))]
    _, c_emp, _ = jFM.combine_contrast_empirical(res)
    assert np.isclose(c_emp[0], 5.0 / np.sqrt(n_wave), rtol=0.15)


def test_worker_blas_limit_is_safe_without_and_with_threadpoolctl():
    with jFM._single_blas_thread():
        assert np.linalg.svd(np.eye(3))[1].sum() == 3
