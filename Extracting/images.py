"""Load survey images into one common record so every later stage is survey-agnostic.

`load_ztf_refimg` reproduces the conventions of the ZTF extraction (header zero point and FWHM, NaN
bad pixels, gain 5.8 e/ADU per coadded frame). `load_ps1_skycell` undoes the PS1 stack's asinh
compression, turns the mask bits into NaNs, and measures the FWHM from stars because the stack
header carries none. Both return an `Image`.
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import sep
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils.psf import fit_fwhm

# PS1 mask bits (PS1 Pixel flags page). Treated as unusable pixels: no data, saturated, cosmic ray,
# diffraction spike, ghost, streak, bright-star core, bad after convolution. Undocumented bit 0x1,
# FLAT/DARK/CTE, LOW, SUSPECT and CONV.POOR are left in.
PS1_MASK_BITS = {
    'FLAT': 0x0002, 'DARK': 0x0004, 'BLANK': 0x0008, 'CTE': 0x0010, 'SAT': 0x0020, 'LOW': 0x0040,
    'SUSPECT': 0x0080, 'CR': 0x0100, 'SPIKE': 0x0200, 'GHOST': 0x0400, 'STREAK': 0x0800,
    'STARCORE': 0x1000, 'CONV_BAD': 0x2000, 'CONV_POOR': 0x4000,
}
PS1_BAD_BITS = (PS1_MASK_BITS['BLANK'] | PS1_MASK_BITS['SAT'] | PS1_MASK_BITS['CR'] | PS1_MASK_BITS['SPIKE']
                | PS1_MASK_BITS['GHOST'] | PS1_MASK_BITS['STREAK'] | PS1_MASK_BITS['STARCORE'] | PS1_MASK_BITS['CONV_BAD'])
ZTF_GAIN_PER_FRAME = 5.8  # e/ADU, Masci et al. 2019; the reference image is a sum of NFRAMES exposures


@dataclass(frozen=True)
class Image:
    """A calibrated image in linear counts with its calibration; NaN marks unusable pixels."""
    survey: str
    band: str
    name: str
    data: np.ndarray                   # float32, NaN where bad
    wcs: WCS
    zero_point: float                  # AB mag of one count
    fwhm_px: float
    pixel_scale: float                 # arcsec per pixel
    variance: Optional[np.ndarray]     # per-pixel variance where the survey provides it (PS1), else None
    gain: Optional[float]              # e/ADU for a Poisson noise term where there is no variance map (ZTF)
    saturation: Optional[float]        # counts above which pixels are non-linear, if known
    mag_limit: Optional[float]         # survey's quoted limiting magnitude, if any
    header: fits.Header

    @property
    def bad(self) -> np.ndarray:
        return np.isnan(self.data)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape


def asinh_decode(v: np.ndarray, bsoften: float, boffset: float) -> np.ndarray:
    """Undo PS1's asinh pixel compression: flux = BOFFSET + BSOFTEN * (10^(0.4 v) - 10^(-0.4 v))."""
    v = np.asarray(v, dtype=np.float64)
    return (boffset + bsoften * (10.0 ** (0.4 * v) - 10.0 ** (-0.4 * v))).astype(np.float32)


def _read_ps1_layer(path: str) -> Tuple[np.ndarray, fits.Header]:
    """PS1 skycell files are tile-compressed with the image in HDU 1; astropy applies BZERO/BSCALE."""
    with fits.open(path) as hdul:
        hdu = hdul[1] if len(hdul) > 1 and hdul[1].data is not None else hdul[0]
        data = np.array(hdu.data)
        header = hdu.header.copy()
    if 'BSOFTEN' in header and 'BOFFSET' in header:
        data = asinh_decode(data, float(header['BSOFTEN']), float(header['BOFFSET']))
    return data.astype(np.float32), header


def measure_fwhm(data: np.ndarray, saturation: Optional[float] = None, n_stars: int = 300,
                 max_side: int = 2500, fwhm_guess: float = 4.0) -> float:
    """FWHM in pixels: median circular-Gaussian fit to the brightest round, unsaturated detections.

    Only used to size cutouts and apertures, so a central window of `max_side` pixels is enough.
    (Isophotal moments were tried first and overestimate undersampled PSFs by ~2x.)
    """
    ny, nx = data.shape
    y0, x0 = max(0, (ny - max_side) // 2), max(0, (nx - max_side) // 2)
    sub = np.ascontiguousarray(data[y0:y0 + max_side, x0:x0 + max_side], dtype=np.float32)
    bad = np.isnan(sub)
    filled = np.where(bad, 0.0, sub).astype(np.float32)
    bkg = sep.Background(filled, mask=bad)
    img = filled - bkg.back()
    src = sep.extract(img, thresh=5.0, err=bkg.globalrms, minarea=5, mask=bad, deblend_cont=0.005)
    ok = (src['flag'] == 0) & (src['b'] / src['a'] > 0.8) & (src['peak'] > 50.0 * bkg.globalrms)
    if saturation is not None:
        ok &= src['peak'] < 0.3 * saturation
    src = src[ok][np.argsort(-src['peak'][ok])][:n_stars]
    if len(src) < 20:
        raise ValueError(f'only {len(src)} usable stars for the FWHM estimate')
    fit_shape = int(2 * round(1.5 * fwhm_guess) + 1)
    fwhm = fit_fwhm(img, xypos=np.c_[src['x'], src['y']], fwhm=fwhm_guess, fit_shape=fit_shape, mask=bad)
    return float(np.nanmedian(fwhm))


def load_ps1_skycell(stack_path: str, mask_path: Optional[str] = None, wt_path: Optional[str] = None,
                     bad_bits: int = PS1_BAD_BITS, band: Optional[str] = None) -> Image:
    """A PS1 DR2 skycell stack as linear counts, with its mask applied as NaN and its variance map.

    Zero point is 25 + 2.5 log10(EXPTIME) (FPA.ZP convention). The header's PC001001-style WCS is
    understood by astropy; RADESYS is missing and set to FK5 as the archive advises.
    """
    data, header = _read_ps1_layer(stack_path)
    if mask_path is not None:
        with fits.open(mask_path) as hdul:
            mask = np.array(hdul[1].data if len(hdul) > 1 else hdul[0].data).astype(np.int32)
        data[(mask & bad_bits) != 0] = np.nan
    variance = None
    if wt_path is not None:
        variance, _ = _read_ps1_layer(wt_path)
        variance[np.isnan(data) | ~(variance > 0)] = np.nan
    header.setdefault('RADESYS', 'FK5')
    wcs = WCS(header)
    pixel_scale = float(np.mean(proj_plane_pixel_scales(wcs)) * 3600.0)
    saturation = float(header['CELL.SATURATION']) if 'CELL.SATURATION' in header else None
    if band is None:
        band = str(header.get('FPA.FILTERID', header.get('FPA.FILTER', ''))).split('.')[0] or os.path.basename(stack_path).split('.stk.')[1][0]
    return Image(
        survey='ps1', band=band, name=os.path.basename(stack_path).split('.stk.')[0],
        data=data, wcs=wcs,
        zero_point=float(header['FPA.ZP']) + 2.5 * np.log10(float(header['EXPTIME'])),
        fwhm_px=measure_fwhm(data, saturation=saturation), pixel_scale=pixel_scale,
        variance=variance, gain=None, saturation=saturation, mag_limit=None, header=header,
    )


def load_ztf_refimg(path: str, band: Optional[str] = None) -> Image:
    """A ZTF deep reference image, with the conventions of the original extraction (`Source_Extractor`)."""
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
        survey='ztf', band=band, name=os.path.basename(path).replace('_refimg.fits', ''),
        data=data, wcs=wcs, zero_point=float(header['MAGZP']), fwhm_px=fwhm,
        pixel_scale=float(np.mean(proj_plane_pixel_scales(wcs)) * 3600.0),
        variance=None, gain=ZTF_GAIN_PER_FRAME * float(header['NFRAMES']),
        saturation=float(header['SATURATE']) if 'SATURATE' in header else None,
        mag_limit=float(header['MAGLIM']) if 'MAGLIM' in header else None, header=header,
    )
