"""Load a ZTF deep reference image into an `Image`: linear counts with NaN marking unusable pixels, the header
calibration (zero point, FWHM, limiting magnitude), and the gain for a Poisson noise term (5.8 e/ADU per coadded
frame). IRSA's per-pixel uncertainty map (`refunc`) is not used: on field 345 it masked no pixel beyond the image's
own NaNs and changed aperture errors by ~1%, so it was not worth the 3.4 TB download."""
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

ZTF_GAIN_PER_FRAME = 5.8  # e/ADU, Masci et al. 2019; the reference image is a sum of NFRAMES exposures


@dataclass(frozen=True)
class Image:
    """A calibrated image in linear counts with its calibration; NaN marks unusable pixels."""
    band: str
    name: str
    data: np.ndarray                   # float32, NaN where bad
    wcs: WCS
    zero_point: float                  # AB mag of one count
    fwhm_px: float
    pixel_scale: float                 # arcsec per pixel
    gain: float                        # e/ADU for a Poisson noise term
    mag_limit: Optional[float]         # the header's limiting magnitude, if any
    header: fits.Header

    @property
    def bad(self) -> np.ndarray:
        return np.isnan(self.data)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape


def load_refimg(path: str, band: Optional[str] = None) -> Image:
    """A ZTF deep reference image; the coadd's own NaNs mark the unimaged edges."""
    with fits.open(path) as hdul:
        header = hdul[0].header.copy()
        data = np.array(hdul[0].data, dtype=np.float32)
    if 'RADECSYS' in header:
        if 'RADESYS' not in header:
            header['RADESYS'] = header['RADECSYS']
        del header['RADECSYS']
    wcs = WCS(header)
    if band is None:
        band = os.path.basename(path).split('_z')[1][0]
    fwhm = float(header['MEDFWHM']) if 'MEDFWHM' in header else float(header['SEEING'])
    return Image(
        band=band, name=os.path.basename(path).replace('_refimg.fits', ''),
        data=data, wcs=wcs, zero_point=float(header['MAGZP']), fwhm_px=fwhm,
        pixel_scale=float(np.mean(proj_plane_pixel_scales(wcs)) * 3600.0),
        gain=ZTF_GAIN_PER_FRAME * float(header['NFRAMES']),
        mag_limit=float(header['MAGLIM']) if 'MAGLIM' in header else None, header=header,
    )
