"""EPSF construction and PSF photometry on an `Image`, as pure functions.

Ports `Source_Extractor.set_sources_for_psf`, `_build_epsf` and `get_psf_mags` with every size in
units of the image FWHM, so the same settings serve ZTF and PS1 pixels. Each call makes its own
fitter: photutils' default `TRFLSQFitter()` is a single shared instance, which swapped fit results
between threads in the original pipeline (see bugs.md).
"""
from typing import Optional, Tuple

import numpy as np
import sep
from astropy.coordinates import SkyCoord
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata import NDData
from astropy.table import Table
from photutils.psf import EPSFBuilder, EPSFFitter, EPSFStars, ImagePSF, PSFPhotometry, extract_stars

from .images import Image
from .utils import img_ab_mag_to_flux, img_flux_to_ab_mag, true_nearby

CUTOUT_FWHM = 14.0        # EPSF star cutout side, in FWHM
FIT_BOX_FWHM = 2.0        # EPSF fitter box, in FWHM
FIT_SHAPE_FWHM = 2.9      # PSF-photometry fit box, in FWHM (5 px on ZTF's 1.7 px FWHM)
ISOLATION_ARCSEC = 10.0   # nearest catalog neighbour farther than this for EPSF stars
MAX_NAN_FRACTION = 0.05   # of the star cutout
OVERSAMPLING = 4


def _odd(x: float) -> int:
    return int(np.ceil(x) // 2 * 2 + 1)


def psf_star_positions(image: Image, ra: np.ndarray, dec: np.ndarray) -> SkyCoord:
    """Catalog stars usable for the EPSF: isolated, inside the image, with a clean cutout.

    The caller supplies point sources (e.g. PS1 PSF-Kron < 0.05, SNR > 10, PSF mag < 17, or Gaia).
    """
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    x, y = image.wcs.world_to_pixel_values(ra, dec)
    x, y = np.asarray(x, float), np.asarray(y, float)
    half = _odd(CUTOUT_FWHM * image.fwhm_px) // 2
    iso_px = ISOLATION_ARCSEC / image.pixel_scale
    keep = np.zeros(len(ra), bool)
    ny, nx = image.shape
    for k in range(len(ra)):
        if not (half <= x[k] < nx - half and half <= y[k] < ny - half):
            continue
        d2 = (x - x[k]) ** 2 + (y - y[k]) ** 2
        d2[k] = np.inf
        if d2.min() < iso_px ** 2:
            continue
        xi, yi = int(x[k]), int(y[k])
        cut = image.bad[yi - half:yi + half, xi - half:xi + half]
        keep[k] = cut.size > 0 and cut.mean() < MAX_NAN_FRACTION
    return SkyCoord(ra[keep], dec[keep], unit='deg')


def build_epsf(image: Image, image_sub: np.ndarray, stars: SkyCoord, oversampling: int = OVERSAMPLING,
               maxiters: int = 10) -> Tuple[ImagePSF, int]:
    """EPSF from star cutouts of the background-subtracted image. Returns (epsf, n_stars_used)."""
    size = _odd(CUTOUT_FWHM * image.fwhm_px)
    data = NDData(data=np.where(image.bad, np.nan, image_sub), wcs=image.wcs)
    cut = extract_stars(data, Table([stars], names=['skycoord']), size=size)
    cut = EPSFStars([s for s in cut if not np.any(np.isnan(s.data))])
    if len(cut) == 0:
        raise ValueError('no clean star cutouts for the EPSF')
    fitter = EPSFFitter(fit_boxsize=_odd(FIT_BOX_FWHM * image.fwhm_px))
    fitter.fit_boxsize = tuple(int(v) for v in fitter.fit_boxsize)  # newer astropy needs a tuple here
    epsf, _ = EPSFBuilder(fitter=fitter, oversampling=oversampling, maxiters=maxiters)(cut)
    return epsf, len(cut)


def psf_photometry(image: Image, image_sub: np.ndarray, bkg: sep.Background, epsf: ImagePSF,
                   sources: Table, fit_shape: Optional[int] = None) -> Table:
    """PSF-fit magnitudes at every source, added as `{band}PSFMag/Err/Flags`, `qfit`, `cfit`.

    Initial fluxes come from the Kron magnitudes; sources with a bad pixel inside the fit box,
    a truncated aperture, or no Kron magnitude get NaN. Positions are free, as in the original.
    """
    b = image.band
    out = sources.copy()
    n = len(out)
    fit_shape = fit_shape or _odd(FIT_SHAPE_FWHM * image.fwhm_px)
    kron = np.asarray(out[f'{b}KronMag'], float)
    kflag = np.asarray(out[f'{b}KronFlag'], int)
    x, y = np.asarray(out['x'], float), np.asarray(out['y'], float)

    nan_near = np.array([true_nearby(int(yy), int(xx), radius=fit_shape // 2, mask=image.bad) for xx, yy in zip(x, y)])
    truncated = ((kflag & sep.APER_TRUNC) != 0) | ((kflag & sep.OBJ_TRUNC) != 0)
    bad = nan_near | truncated | ~np.isfinite(kron)
    good = ~bad

    init = Table({'x': x[good], 'y': y[good], 'flux_init': img_ab_mag_to_flux(kron[good], image.zero_point)})
    init['flux_init'][kron[good] == -999.0] = 0.0
    error = np.sqrt(np.where(np.isnan(image.variance), np.nanmedian(image.variance), image.variance)) if image.variance is not None else bkg.rms()
    phot = PSFPhotometry(epsf, fit_shape=fit_shape, fitter=TRFLSQFitter())
    res = phot(np.where(image.bad, 0.0, image_sub), error=error, init_params=init, mask=image.bad)
    mag, magerr = img_flux_to_ab_mag(np.asarray(res['flux_fit'], float), image.zero_point, fluxerr=np.asarray(res['flux_err'], float))

    for col, vals in ((f'{b}PSFMag', mag), (f'{b}PSFMagErr', magerr), (f'{b}PSFFlags', np.asarray(res['flags'], float)),
                      ('qfit', np.asarray(res['qfit'], float)), ('cfit', np.asarray(res['cfit'], float)),
                      ('x_psf', np.asarray(res['x_fit'], float)), ('y_psf', np.asarray(res['y_fit'], float))):
        full = np.full(n, np.nan)
        full[good] = vals
        out[col] = full
    return out
