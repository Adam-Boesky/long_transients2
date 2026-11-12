"""Source detection and Kron photometry on an `Image`, as pure functions.

Ports the sep stages of `Source_Extractor` with sizes expressed in arcseconds or FWHM so the same
settings apply to ZTF (1"/px) and PS1 (0.25"/px) pixels. Noise comes from the image's variance map
when it has one (PS1) and from the global background rms plus gain otherwise (ZTF), which is what
the original pipeline did.
"""
from typing import Optional, Tuple

import numpy as np
import sep
from astropy.table import Table

from .images import Image, ZTF_GAIN_PER_FRAME
from .utils import img_flux_to_ab_mag

sep.set_extract_pixstack(10_000_000)
sep.set_sub_object_limit(4096)

BKG_BOX_ARCSEC = 64.0          # Source_Extractor used bw=bh=64 px on ZTF
DEBLEND_CONT = 0.00001
DEBLEND_NTHRESH = 32
THRESH_SIGMA = 1.0
MINAREA_PX = 5
KRON_FACTOR = 2.5              # flux within KRON_FACTOR * kron radius
KRON_RADIUS_SCALE = 6.0        # sep.kron_radius integration radius in units of a, b
KRON_MIN_ARCSEC = 1.5          # below this Kron radius a circular aperture of CIRCLE_FWHM * FWHM is used
CIRCLE_FWHM = 2.5


def _filled(image: Image) -> Tuple[np.ndarray, np.ndarray]:
    """Image with NaNs zeroed, and the boolean bad mask, as sep wants them."""
    bad = image.bad
    return np.where(bad, 0.0, image.data).astype(np.float32), bad


def background(image: Image, box_arcsec: float = BKG_BOX_ARCSEC, filter_size: int = 3) -> sep.Background:
    filled, bad = _filled(image)
    box = int(round(box_arcsec / image.pixel_scale))
    return sep.Background(filled, mask=bad, bw=box, bh=box, fw=filter_size, fh=filter_size)


def noise(image: Image, bkg: sep.Background, per_pixel: bool = True) -> Tuple[np.ndarray | float, Optional[float]]:
    """(err, gain) for sep.

    With `per_pixel` and a survey variance map: the per-pixel rms and no gain. Otherwise the global
    background rms plus the gain for a Poisson term, which is what the original pipeline used.
    """
    if per_pixel and image.variance is not None:
        err = np.sqrt(np.where(np.isnan(image.variance), np.nanmedian(image.variance), image.variance)).astype(np.float32)
        return err, None
    gain = image.gain
    if gain is None and image.survey == 'ztf':
        gain = ZTF_GAIN_PER_FRAME * float(image.header['NFRAMES'])
    return float(bkg.globalrms), gain


def detect(
        image: Image,
        bkg: Optional[sep.Background] = None,
        thresh: float = THRESH_SIGMA,
        minarea: int = MINAREA_PX,
        deblend_cont: float = DEBLEND_CONT,
        deblend_nthresh: int = DEBLEND_NTHRESH,
        per_pixel_threshold: bool = False,
    ) -> Tuple[Table, np.ndarray, np.ndarray, sep.Background]:
    """sep detections on the background-subtracted image.

    The threshold is `thresh` times the global background rms by default, as in the original
    pipeline; `per_pixel_threshold` uses the variance map instead (SExtractor weight-map style),
    which re-segments blended sources because the Poisson term raises the threshold on galaxies.

    Returns (sources, segmap, image_sub, bkg). `sources` has sep's columns, the flag renamed to
    `sepExtractionFlag`, a 1-based `segid` matching `segmap`, and `ra`, `dec`.
    """
    if bkg is None:
        bkg = background(image)
    filled, bad = _filled(image)
    image_sub = filled - bkg.back()
    image_sub[bad] = np.nan
    err, gain = noise(image, bkg, per_pixel=per_pixel_threshold)
    raw, segmap = sep.extract(
        np.where(bad, 0.0, image_sub).astype(np.float32), thresh=thresh, err=err, mask=bad, minarea=minarea,
        deblend_cont=deblend_cont, deblend_nthresh=deblend_nthresh, gain=gain, segmentation_map=True,
    )
    sources = Table(raw)
    sources.rename_column('flag', 'sepExtractionFlag')
    sources['segid'] = np.arange(1, len(sources) + 1, dtype=np.int32)
    ra, dec = image.wcs.pixel_to_world_values(sources['x'], sources['y'])
    sources['ra'], sources['dec'] = np.asarray(ra, float), np.asarray(dec, float)
    return sources, segmap, image_sub, bkg


def kron_photometry(image: Image, sources: Table, segmap: np.ndarray, image_sub: np.ndarray,
                    bkg: sep.Background, neighbour_masked_radius: bool = False) -> Table:
    """Kron magnitudes for `sources`, added as `{band}KronMag/Err/CircleFlag/Flag` and `KronRad`.

    Same recipe as `Source_Extractor.get_kron_mags`: elliptical aperture of KRON_FACTOR times the
    Kron radius, or a circle of CIRCLE_FWHM * FWHM when the Kron radius is under KRON_MIN_ARCSEC.
    Negative fluxes get magnitude -999. The flux sum always excludes other sources' segments; the
    Kron radius itself does only with `neighbour_masked_radius` (the original did not, and the
    choice moves blended sources' Kron magnitudes by up to ~0.7 mag).
    """
    b = image.band
    out = sources.copy()
    filled, bad = _filled(image)
    sub = np.where(bad, 0.0, image_sub).astype(np.float32)
    err, gain = noise(image, bkg)
    theta = np.asarray(out['theta'], float)
    theta[theta > np.pi / 2] -= np.pi
    theta[theta < -np.pi / 2] += np.pi / 2
    out['theta'] = theta
    x, y, a, bb = (np.asarray(out[c], float) for c in ('x', 'y', 'a', 'b'))
    segid = np.asarray(out['segid'], np.int32)

    seg_kw = dict(segmap=segmap, seg_id=segid) if neighbour_masked_radius else {}
    kronrad, kr_flag = sep.kron_radius(sub, x, y, a, bb, theta, KRON_RADIUS_SCALE, mask=bad, **seg_kw)
    out['KronRad'] = kronrad
    use_circle = kronrad * np.sqrt(a * bb) < KRON_MIN_ARCSEC / image.pixel_scale

    flux = np.full(len(out), np.nan)
    fluxerr = np.full(len(out), np.nan)
    flag = np.zeros(len(out), dtype=int)
    if (~use_circle).any():
        e = ~use_circle
        f, fe, fl = sep.sum_ellipse(sub, x[e], y[e], a[e], bb[e], theta[e], KRON_FACTOR * kronrad[e],
                                    err=err, gain=gain, mask=bad, segmap=segmap, seg_id=segid[e])
        flux[e], fluxerr[e], flag[e] = f, fe, fl
    if use_circle.any():
        c = use_circle
        f, fe, fl = sep.sum_circle(sub, x[c], y[c], CIRCLE_FWHM * image.fwhm_px, subpix=1,
                                   err=err, gain=gain, mask=bad, segmap=segmap, seg_id=segid[c])
        flux[c], fluxerr[c], flag[c] = f, fe, fl
    flag |= kr_flag.astype(int)

    mag, magerr = img_flux_to_ab_mag(flux, image.zero_point, fluxerr=fluxerr)
    mag = np.where(flux < 0, -999.0, mag)
    out[f'{b}KronMag'] = mag
    out[f'{b}KronMagErr'] = magerr
    out[f'{b}KronCircleFlag'] = use_circle.astype(int)
    out[f'{b}KronFlag'] = flag
    return out
