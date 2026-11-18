"""Load a ZTF deep reference image into an `Image`: linear counts with NaN marking unusable pixels, the header
calibration (zero point, FWHM, limiting magnitude), and either the survey's variance map (`refunc`) or the gain for
a Poisson noise term (5.8 e/ADU per coadded frame)."""
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
    variance: Optional[np.ndarray]     # per-pixel variance from the survey's uncertainty map, else None
    gain: Optional[float]              # e/ADU for a Poisson noise term when there is no variance map
    mag_limit: Optional[float]         # the header's limiting magnitude, if any
    header: fits.Header

    @property
    def bad(self) -> np.ndarray:
        return np.isnan(self.data)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape


def load_refimg(path: str, unc_path: Optional[str] = None, cov_path: Optional[str] = None,
                    band: Optional[str] = None, min_coverage: float = 0.5) -> Image:
    """A ZTF deep reference image.

    With `unc_path` (IRSA's `refunc`, 1-sigma per pixel in DN) the variance map is its square and
    the gain is not needed. With `cov_path` (`refcov`, input frames per pixel) pixels below
    `min_coverage` times the median coverage are marked bad; those thin edges are noisy.
    """
    with fits.open(path) as hdul:
        header = hdul[0].header.copy()
        data = np.array(hdul[0].data, dtype=np.float32)
    variance = None
    if unc_path is not None:
        with fits.open(unc_path) as hdul:
            unc = np.array(hdul[0].data, dtype=np.float32)
        variance = unc ** 2
        data[np.isnan(unc) | ~(unc > 0)] = np.nan
    if cov_path is not None:
        with fits.open(cov_path) as hdul:
            cov = np.array(hdul[0].data, dtype=np.float32)
        data[~(cov >= min_coverage * np.nanmedian(cov))] = np.nan
    if variance is not None:
        variance[np.isnan(data)] = np.nan
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
        variance=variance, gain=None if variance is not None else ZTF_GAIN_PER_FRAME * float(header['NFRAMES']),
        mag_limit=float(header['MAGLIM']) if 'MAGLIM' in header else None, header=header,
    )
