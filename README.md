# jighcontrast

Tools for high-contrast imaging: ADI/SDI PCA point-spread-function subtraction,
PSF photometry / contrast estimation, and NaN-aware image rotation.

## Modules

- `jighcontrast.jPCA` — PCA (KLIP-style) PSF subtraction and ADI/SDI reference-library construction.
- `jighcontrast.jPhot` — PSF injection, aperture/PSF photometry, and radial noise / S-N estimation.
- `jighcontrast.nan_rot` — NaN-aware image rotation and rotation-center fitting.

## Installation

All dependencies (including [`jfits`](https://pypi.org/project/jfits/)) are on
PyPI and install automatically.

Clone and install in editable mode:

```bash
git clone https://github.com/jordan-stone/jighcontrast.git
cd jighcontrast
pip install -e .
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/jordan-stone/jighcontrast.git
```

## Usage

```python
from jighcontrast import jPCA, jPhot, nan_rot

# NaN-aware rotation about a given axis (x, y)
rotated = nan_rot.nan_rot(image, angle=37.0, axis=(128, 128))

# PCA PSF subtraction.
#   library_cube : (n_images, ydim, xdim) stack of reference PSF images
#   opt_mask     : (ydim, xdim) boolean mask; True marks the pixels the PCA is
#                  optimized over (the optimization region)
pca = jPCA.PCA(library_cube, opt_mask)   # method='svd' (default) or 'eigh'
residual = pca(target_image, npca=10)    # target_image (ydim, xdim); returns the
                                         # residual pixels within opt_mask (1-D)
```

## Example

[`examples/example_adi_reduction.ipynb`](examples/example_adi_reduction.ipynb) is
a worked, end-to-end ADI + KLIP reduction of a synthetic SCALES coronagraphic
integral-field data set, recovering an injected planet at ~10σ. The data cube
(~430 MB) is attached to the [`example-data`
release](https://github.com/jordan-stone/jighcontrast/releases/tag/example-data)
and is downloaded automatically the first time the notebook runs.

## Requirements

- Python >= 3.8
- numpy, scipy, matplotlib, astropy
- [jfits](https://pypi.org/project/jfits/)

## License

MIT — see [LICENSE](LICENSE).
