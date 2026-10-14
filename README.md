# jighcontrast

Tools for high-contrast imaging with integral-field spectrographs: KLIP/PCA PSF
subtraction for ADI/SDI sequences, **KLIP forward modeling** (spectral
extraction and calibrated detection limits, after
[Pueyo 2016](https://ui.adsabs.harvard.edu/abs/2016ApJ...824..117P)), aperture
and matched-filter photometry, NaN-aware image rotation, and conversion of
contrast limits to companion masses with evolutionary models.

## Modules

| module | contents |
|---|---|
| `jighcontrast.jPCA` | `PCA` (KLIP) with SVD or eigen-decomposition and two library-centering conventions; ADI/SDI reference selection. |
| `jighcontrast.jFM` | KLIP forward modeling: the first-order perturbation of the KL basis (Pueyo 2016, App. E), `extract_spectrum` (App. F) with an empirically measured, wavelength-correlated covariance, and `contrast_curve` (forward-modeled throughput, matched-filter noise, fixed false-alarm probability). |
| `jighcontrast.jPhot` | PSF injection, aperture / Gaussian / matched-filter S/N with the Mawet et al. (2014) small-sample statistics. |
| `jighcontrast.nan_rot` | NaN-preserving image rotation (`nan_rot`) and rotation-center fitting. |
| `jighcontrast.jEvol` | Contrast -> detectable mass with Sonora Diamondback evolution + atmosphere grids and PHOENIX host spectra (data files not included). |

## Installation

```bash
pip install git+https://github.com/jordan-stone/jighcontrast.git
```

or, to develop,

```bash
git clone https://github.com/jordan-stone/jighcontrast.git
cd jighcontrast
pip install -e ".[test,parallel]"
pytest
```

Requires Python >= 3.8, numpy, scipy, matplotlib, astropy and
[jfits](https://pypi.org/project/jfits/) (all installed automatically).

## Quick start

### PCA PSF subtraction

```python
from jighcontrast import jPCA, nan_rot

pca = jPCA.PCA(reference_cube, opt_mask)        # (n_refs, ny, nx), boolean (ny, nx)
residual_pixels = pca(target_image, npca=10)    # residual within opt_mask (1-D)
derotated = nan_rot.nan_rot(image, -parallactic_angle, star_xy, pivot=True)
```

### Spectrum of a companion (KLIP-FM)

```python
import numpy as np
from jighcontrast import jFM, jPhot

# cube: (n_exp, n_wave, ny, nx) coronagraphic sequence
# unocc: (n_snap, n_wave, ny, nx) unocculted snapshot(s), e.g. before and after the run
# pa: parallactic angles (deg); waves: microns
# star_xy: the star's position behind the coronagraph -- the coronagraph's
#          calibrated position on the detector (fixed for the whole sequence)

# 1. PSF calibration: center each snapshot on its own peak, then combine
snaps = [jPhot.register_channels(s[None], jPhot.measure_star_centers(s, star_xy), star_xy)[0]
         for s in unocc]
psf = jFM.offaxis_psf_stamps(np.array(snaps), star_xy, half=9)   # unit-contrast planet model

# 2. extract (KLIP-FM, refined for bright sources, errors measured on the data)
result = jFM.extract_spectrum(cube, None, pa, waves, star_xy, sep_pix, pa_deg,
                              D=10.0, pixscale=20.0, psf_stamps=psf, npca=20,
                              n_jobs=8)
result["spectrum"], result["sigma"]   # companion/star flux ratio per channel, 1-sigma
result["cov"]                         # full covariance (channels are correlated)
result["broadband"]                   # band-averaged contrast, empirical sigma, S/N
result["dlam_fit"], result["mixing_fit"]   # first-order validity diagnostics (<< 1)
```

### Detection limits

```python
curve = jFM.contrast_curve(cube, unocc, pa, waves, star_xy, radii_pix,
                           D=10.0, pixscale=20.0, npca=20, centering="frame",
                           known_companion=(pa_deg, sep_pix), n_jobs=8)
radii, c_limit, n_ring = jFM.combine_contrast_empirical(curve)   # flat spectrum
radii, c_naive = jFM.combine_contrast(curve)   # channels assumed independent (optimistic)
```

`n_jobs > 1` farms spectral channels to worker processes. Install with
`pip install "jighcontrast[parallel]"` (adds `threadpoolctl`) and each worker is
limited to one BLAS thread automatically; without it, set `OMP_NUM_THREADS=1`
before importing numpy so the workers do not oversubscribe the cores. On
macOS/Windows workers are *spawned*: run from a notebook or from a script whose
top level is guarded by `if __name__ == "__main__":`.

## How the forward modeling works

A faint companion in the KLIP reference images perturbs the library covariance
by a term linear in its flux; to first order the principal components, and so
the KLIP-processed companion, are linear in its spectrum (Pueyo 2016, Eqs. 51,
SimpleMat). `jFM` builds, in one reduction, the processed image of a
unit-contrast source in each channel, and

* **extracts spectra** by linear least squares against the reduced data in a
  1 lambda/D region (Appendix F). Errors come from applying the same estimator
  at empty positions around the ring at the companion's separation: per-channel
  sigma from lambda/D-spaced positions with the Mawet et al. (2014) small-sample
  factor; the inter-channel correlation (speckles are strongly correlated across
  wavelength; Greco & Brandt 2016) from densely spaced positions. Combinations
  of channels -- a band-averaged contrast, a template amplitude -- get their
  errors the same way, by evaluating the combination at the ring positions,
  rather than by propagating through the inverse covariance (which, near the
  star, understates them severalfold);
* **refines bright companions** whose flux perturbs the basis beyond first order
  (diagnostics `dlam_fit`, the relative eigenvalue perturbation of App. E.3, and
  `mixing_fit`, the mode mixing across the truncation): the current estimate is
  subtracted from the data (negative injection), the data re-reduced, and the
  faint remainder fit again;
* **calibrates contrast curves** with the forward-modeled template as a matched
  filter, at a fixed false-alarm probability (Student-t thresholds), combining
  channels with their noise measured at common ring positions.

Conventions: pixel coordinates follow `scipy.ndimage.map_coordinates` (integer
values are pixel centers, so the center of an N-pixel axis is (N-1)/2); position
angles are degrees east of north; exposures are derotated by -PA; spectra are
companion/star flux ratios, with the unocculted PSF defining unit contrast.

**Star position and PSF calibration.** The star's position in the coronagraphic
frames is an input (`center`), not something the code measures: without a
reference such as satellite spots, use the coronagraph's calibrated position on the
detector, fixed for the sequence. Drifts of the star during the sequence -- tip/tilt
residuals, atmospheric dispersion changing with airmass (~0.1 pixel across
2.9-4.1 um for a target like HIP 21152 from Mauna Kea) -- are not modeled. The
unocculted snapshots provide the PSF model and flux scale; center each on its own
peak before combining them.

## Validation

`tests/` holds fast unit tests on synthetic data (`pytest`, ~20 s): the
perturbation theory against finite differences, the forward model against
brute-force KLIP, sub-pixel registration, spectral recovery, error-bar
calibration and refinement. [`validation/`](validation) holds injection-recovery
studies on a simulated SCALES L-band IFS sequence; see
[validation/README.md](validation/README.md) for the methods and full results.
In summary:

* **Forward model vs brute-force KLIP** (V1): with per-frame (Pueyo) centering
  the predicted flux matches inject-and-reduce to 0.0-0.3% for contrasts
  <~3e-5 and the error grows with the first-order diagnostic; median-image
  centering carries a 3-5% bias.
* **Spectral extraction** (V2, 96 injections of L- and T-type spectra,
  1e-5 to 3e-4, 240-560 mas): unbiased (band mean 1.010 +/- 0.007 at 3e-4
  after refinement; a single pass over-recovers by 5% there), against a 25-50%
  flux loss without forward modeling; per-channel errors calibrated (pull rms
  1.10, mean 0.01).
* **Band-averaged contrast errors** (V2b, 42 independent injections):
  calibrated from 240 mas outward (pull rms 0.75-1.2); about 2x too small at
  200 mas. Assuming independent channels instead understates them 3x.
* **Contrast curve** (V3): sources injected at the 5-sigma-equivalent limit are
  recovered at the detection threshold from 160 to 320 mas, at 0.66-0.86 of it
  at 400-560 mas; independent-channel limits would be 3-5x too deep.
* **End to end** (V4): the companion injected by an independent simulator
  (scalesETC; Sonora Bobcat 1400 K over a 10000 K star) is recovered with its
  injected spectrum: amplitude 0.995 of the reconstructed truth, 6% rms per
  channel. A PSF from two unocculted snapshots does as well as one from a full
  unocculted sequence.

## Examples

* [`examples/fm_spectrum.ipynb`](examples/fm_spectrum.ipynb) -- recipe: extract a
  companion's spectrum (with covariance) from an IFS ADI sequence, using one or
  two unocculted snapshots as the PSF calibration. Edit one configuration cell.
* [`examples/fm_contrast_curve.ipynb`](examples/fm_contrast_curve.ipynb) -- recipe:
  forward-modeled contrast curve, after removing any known companion.
* [`examples/example_klipfm.ipynb`](examples/example_klipfm.ipynb) -- tutorial:
  how KLIP distorts a companion and what the forward model corrects.
* [`examples/example_adi_reduction.ipynb`](examples/example_adi_reduction.ipynb) --
  ADI + KLIP reduction of a SCALES IFS sequence and detection of an injected planet.

Both download their data from the
[`example-data` release](https://github.com/jordan-stone/jighcontrast/releases/tag/example-data)
on first run.

## Changes in 0.2.0

* New `jFM` (KLIP forward modeling) and `jEvol` (contrast -> mass) modules.
* `jPCA.PCA`: `centering='median'|'frame'`; stores the frame-space vectors
  needed for forward modeling. `make_adi_sdi_library`: fixed the SDI radial
  displacement (was `r*ref_wv/wv`, now `r*(ref_wv/wv - 1)`); optional lambda/D
  exclusion.
* Parallactic-angle differences now wrap through +/-180 deg (`jPCA.wrap_degrees`),
  in `adi_sdi_displacement` and `make_adi_library` -- needed for targets that
  transit near the zenith. `make_adi_library` previously counted a frame
  rotated by -1 deg as 359 deg (accepting it as a reference) and modified the
  caller's angle array in place.
* The jPCA `method='eigh'` speed hint is printed by the main process only, not
  by every parallel worker.
* `jPhot.get_noise`: `exclude_pos` / `exclude_radius` to keep a known source out
  of the noise ring. New `jPhot.measure_star_centers` / `register_channels`
  (centering PSF calibration snapshots; removing a static chromatic offset).
* `jEvol` atmosphere photometry no longer clips to the model grid: off-grid
  (e.g. colder than the coolest model) returns NaN, and `grid_floor_mass` gives
  the lowest mass the grid covers. Clipping assigned the coolest model's flux to
  colder companions and made low masses look detectable.
* **Behavior change -- `nan_rot`:** with the default `pivot=False` the rotation
  axis is now placed at the output center `(N-1)/2` (pixel-center convention,
  as `map_coordinates` uses). It was placed at `N/2`, shifting every rotated
  image by half a pixel in x and y. `pivot=True` is unchanged. Results that
  depend on absolute positions in rotated images shift by (-0.5, -0.5) pixel.

## References

* Pueyo, L. 2016, ApJ, 824, 117 -- KLIP forward modeling.
* Soummer, R., Pueyo, L. & Larkin, J. 2012, ApJL, 755, L28 -- KLIP.
* Mawet, D. et al. 2014, ApJ, 792, 97 -- small-sample statistics.
* Greco, J. P. & Brandt, T. D. 2016, ApJ, 833, 134 -- correlated spectral noise.
* Morley, C. V. et al. 2024, ApJ, 975, 59 -- Sonora Diamondback.

## License

MIT -- see [LICENSE](LICENSE).
