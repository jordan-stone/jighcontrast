#!/usr/bin/env python
"""Build the compact example data set attached to the ``example-data`` release.

Input: the synthetic SCALES L-band (2.9-4.1 um, R~50) IFS sequences of HIP 21152
made with scalesETC (https://github.com/scalessim/scalesETC) -- a 3 h ADI
sequence of 90 exposures x 54 channels, behind the coronagraph and unocculted.
The star is a 10000 K PHOENIX model (L = 5.35); the companion, a Sonora Bobcat
model with Teff/logg from Kuzuhara et al. (2022) / Franson et al. (2023), is
injected at 390 mas, PA 217 deg, L = 14.66 (delta-L = 9.36).

Output: float32 cubes cropped to 72 x 72 pixels about the star (+/-720 mas,
covering every separation used in the examples), each a single self-contained
FITS file:
    HDU 0  CUBE         (90, 54, 72, 72) counts
    HDU 1  WAV_MICRONS  channel wavelengths
    HDU 2  PARANG       parallactic angle per exposure (deg)
                        across the band (a fixed chromatic offset, as from residual
                        atmospheric dispersion); register channels on it before ADI.
Usage: make_example_data.py <dir with the original cubes + PA csv> <output dir>
"""
import os
import sys
import numpy as np
from astropy.io import fits


HALF = 36
SRC = {"coronagraph": "3hr_LowResL-2p9-4p15_IFScube_seq_adi_coronagraph_pLmag14p66_sLmag5p3.fits",
       "unocculted": "3hr_LowResL-2p9-4p15_IFScube_seq_star_no_coronagraph_pLmag14p66_sLmag5p3.fits"}
STAR = (54.0, 54.0)   # where scalesETC places the star in its 108 x 108 frames


def main(src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pa = np.loadtxt(os.path.join(src_dir, "parallactic_angle_degrees.csv"), delimiter=",")
    x0 = int(STAR[0]) - HALF                   # 54 - 36 = 18 -> star at pixel 36
    y0 = int(STAR[1]) - HALF
    for kind, fname in SRC.items():
        with fits.open(os.path.join(src_dir, fname), memmap=True) as h:
            data = np.asarray(h[0].data[:, :, y0:y0 + 2 * HALF, x0:x0 + 2 * HALF], np.float32)
            wav = np.asarray(h[1].data['wav_microns'], float)
        hdr = fits.Header()
        hdr['OBJECT'] = ('HIP 21152 (synthetic)', 'scalesETC simulation')
        hdr['SEQUENCE'] = (kind, 'coronagraph or unocculted')
        hdr['STARX'] = (STAR[0] - x0, 'star x in simulation (0-based)')
        hdr['STARY'] = (STAR[1] - y0, 'star y in simulation (0-based)')
        hdr['PIXSCALE'] = (20.0, 'mas / pixel')
        hdr['TELDIAM'] = (10.0, 'telescope diameter (m)')
        hdr['EXPTOTAL'] = (3.0, 'total sequence duration (h)')
        hdr['STARMAG'] = (5.35, 'L, host (10000 K PHOENIX)')
        hdr['COMPMAG'] = (14.66, 'L, injected companion (Sonora Bobcat)')
        hdr['COMPSEP'] = (390.0, 'companion separation (mas)')
        hdr['COMPPA'] = (217.0, 'companion position angle (deg E of N)')
        hdr['CROPX0'] = (x0, 'crop origin in the original 108x108 frame')
        hdr['CROPY0'] = (y0, 'crop origin in the original 108x108 frame')
        hdr['COMMENT'] = 'Derotate exposure t by -PARANG[t] (nan_rot) to put north up.'
        hdr['COMMENT'] = 'Pixel coordinates: 0-based, integer = pixel center.'
        hdus = fits.HDUList([
            fits.PrimaryHDU(data, header=hdr),
            fits.BinTableHDU.from_columns([fits.Column('wav_microns', 'D', array=wav)],
                                          name='WAV_MICRONS'),
            fits.BinTableHDU.from_columns([fits.Column('parang_deg', 'D', array=pa)],
                                          name='PARANG')])
        out = os.path.join(out_dir, "HIP21152_SCALES_L_3h_%s.fits" % kind)
        hdus.writeto(out, overwrite=True)
        print("wrote %s  %s  %.0f MB" % (out, data.shape, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
