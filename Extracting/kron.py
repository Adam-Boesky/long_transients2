"""Kron photometry on an `Image` at known source positions, with PS1's radius estimator measured on our own pixels,
so the ZTF Kron magnitudes are an independent measurement that can be compared with PS1's afterwards.

Per image: background and per-pixel noise, a copy smoothed by 0.9 sigma_PSF (used only for the radius), and a
segmentation seeded at the known positions (marker watershed on the smoothed image above threshold), so deblending
follows the catalog and every position owns a segment. The floor on the radius is the median plain first moment,
inside a 4 sigma_PSF window, of every source with S/N >= 2 PSF_SN_LIM. Per source, twice:
  window W   = the sky radius (log-spaced annuli, robust median profile, stop where the slope flattens below the sky
               model's sigma or the profile reaches the sky) on the first pass, then
               min(6 R1, sqrt(F_Kron / (pi * KRON_SB_MIN_FACTOR * sigma_sky))); never below 4 sigma_PSF;
  radius R1  = sum sign(f1) sqrt(|f1 f2|) r / sum (f1 + f2)/2 over unmasked mirror-pixel pairs inside W on the
               smoothed image; undefined (no Kron) if either sum is not positive; floored at the stars' R1 and, for
               faint peaks, capped at W;
  flux       = whole-pixel sum inside 2.5 R1 on the unsmoothed image, other sources' segments and bad pixels
               masked, error sqrt(sum variance).
Everything is in pixels inside and arcseconds at the interface.

References
  Kron 1980, ApJS 43, 305: the first-moment radius and the 2.5 R1 aperture.
  Magnier et al. 2020, ApJS 251, 5 (arXiv:1612.05244), section IV.4.3 (eq. 18, the radial moment; R_min from the PSF
    stars) and section IV.6.4 (the refined Kron measurement on the stacks).
  PS1 IPP source, github.com/ps-ipp/ipp-trunk-20250502 (referred to below as IPP:<path>):
    psphot/src/psphotKronIterate.c      psphotKronIterateReadout (windows, iterations, smoothing, floor and cap),
                                        psphotKronRadiusMeasure (the symmetrised moment), psphotKronFluxMeasure (the sum);
    psphot/src/psphotRadialProfileWings.c  the sky radius;
    psphot/src/psphotSourceStats.c and psModules/src/objects/pmSourceMoments.c  pmSourceMinKronRadius (the floor);
    ippconfig/recipes/psphot.config     the recipe values quoted at the constants below.
  Beucher & Meyer 1993, in Mathematical Morphology in Image Processing, ch. 12: marker-controlled watershed, as
    implemented in skimage.segmentation.watershed and used by photutils.segmentation.deblend_sources.
  Bertin & Arnouts 1996, A&AS 117, 393: the filtered-image threshold convention for the segmentation mask.
Differences from psphot that remain: neighbours are masked by segment rather than removed by subtracting their fitted
models, and the surface-brightness window, faint cap and floor are evaluated in this image's own units.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import sep
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed

from .images import Image, ZTF_GAIN_PER_FRAME
from .utils import img_flux_to_ab_mag

BKG_BOX_ARCSEC = 64.0          # sep background mesh (Source_Extractor used bw=bh=64 px on ZTF)


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


# Recipe values. "psphot.config" means the PS1 value in IPP:ippconfig/recipes/psphot.config; "default" means the
# fallback compiled into psphotKronIterate.c when the recipe does not set it.
KRON_FACTOR = 2.5           # aperture radius in units of R1 (Kron 1980; psphotKronFluxMeasure: radKron = 2.5 Mrf)
KRON_ITERATIONS = 2         # psphot.config KRON_ITERATIONS
KRON_WINDOW_FACTOR = 6.0    # second-pass window in units of the first-pass R1 (psphotKronIterateReadout: 6.0 * Mrf)
KRON_SB_MIN_FACTOR = 2.0    # psphot.config KRON_SB_MIN_FACTOR: window limited where mean surface brightness < factor * sigma_sky
MIN_WINDOW_SIGMA = 4.0      # window never below this many PSF sigma (psphotSourceStats.c: PSF_MOMENTS_RADIUS = 4 Sigma)
SMOOTH_SIGMA_PSF = 0.9      # smoothing sigma in units of sigma_PSF (psphot.config KRON_SMOOTH_SIGMA 1.7 px on a ~1.9 px PSF)
SMOOTH_TRUNCATE = 2.0       # kernel truncated at this many sigma (psphot.config KRON_SMOOTH_NSIGMA)
FAINT_PEAK_SNR = 10.0       # below this peak S/N, R1 is capped at the window (psphotKronRadiusMeasure: sqrt(detValue) < 10)
FLOOR_SNR = 40.0            # sources with S/N above this set the floor (psphotSourceStats.c: 2 * PSF_SN_LIM, PSF_SN_LIM = 20)
SEG_THRESH_SIGMA = 1.0      # pixels above this many sigma (smoothed image) belong to a segment (our choice; cf. DETECT_THRESH)
SKY_NSTEP = 25              # sky-radius profile: log-spaced annuli from 4 sigma_PSF to MAX_RADIUS (psphotRadialProfileWings.c NSTEP)
SKY_MIN_DR_ARCSEC = 0.5     # minimum annulus width (psphotRadialProfileWings.c MIN_DR = 2 PS1 px)
SKY_NSIGMA = 1.0            # profile has reached the sky when its annular median is within this many sigma of zero (NSIGMA = 1.0)
MAX_RADIUS_ARCSEC = 15.0    # cap on windows and apertures (psphot.config stack EXT_FIT_MAX_RADIUS: 50 px = 12.5")
FLAG_NO_RADIUS, FLAG_AT_FLOOR, FLAG_FAINT_CAP, FLAG_TRUNCATED, FLAG_NEGATIVE = 1, 2, 4, 8, 16   # NO_RADIUS: moment undefined, no Kron


@dataclass
class KronFrame:
    image: Image
    sub: np.ndarray          # background-subtracted, padded, bad -> 0
    smooth: np.ndarray       # sub smoothed, padded
    var: np.ndarray          # per-pixel variance, padded
    bad: np.ndarray          # bad pixels, padded
    segmap: np.ndarray       # 0 = sky, k = source k-1 (into the positions given to `prepare`), padded
    pad: int
    x: np.ndarray            # padded pixel coordinates of the sources
    y: np.ndarray
    inside: np.ndarray
    sigma_psf: float         # px
    sigma_sky: float         # per-pixel rms of the background-subtracted image (psphot MSKY_DEV)
    sigma_smooth: float      # per-pixel rms of the smoothed image
    sky_slope_min: float     # rms of the background model (psphot.config SKY_SLOPE_MIN: "the sigma of the sky model")
    r_min: float = np.nan    # floor on R1 (px)


def prepare(image: Image, ra: np.ndarray, dec: np.ndarray) -> KronFrame:
    """Background, noise, smoothed copy (psphotKronIterateReadout: psImageSmoothNoMask) and a catalog-seeded segmentation."""
    bkg = background(image)
    bad = image.bad
    back = bkg.back()
    sub = np.where(bad, 0.0, image.data - back).astype(np.float32)
    err, _ = noise(image, bkg)                       # per-pixel rms from the survey's variance map where it has one
    var = (np.broadcast_to(np.asarray(err, np.float32), image.shape) ** 2).astype(np.float32)
    sigma_psf = image.fwhm_px / 2.3548
    smooth = gaussian_filter(sub, SMOOTH_SIGMA_PSF * sigma_psf, truncate=SMOOTH_TRUNCATE).astype(np.float32)
    sigma_sky = float(bkg.globalrms)
    rstd = lambda a: float(1.4826 * np.nanmedian(np.abs(a - np.nanmedian(a))))
    sigma_smooth = rstd(smooth[~bad])
    sky_slope_min = rstd(back[~bad])

    # pad every array by the largest possible cutout so edge sources get centred, in-bounds cutouts; padding is masked
    pad = int(np.ceil(KRON_FACTOR * MAX_RADIUS_ARCSEC / image.pixel_scale)) + 2
    P = lambda a, fill: np.pad(a, pad, constant_values=fill)
    x, y = (np.asarray(v, float) + pad for v in image.wcs.world_to_pixel_values(np.asarray(ra, float), np.asarray(dec, float)))
    ny, nx = image.shape
    inside = np.isfinite(x) & np.isfinite(y) & (x >= pad) & (x < nx + pad) & (y >= pad) & (y < ny + pad)

    # Marker-controlled watershed (Beucher & Meyer 1993; skimage), as photutils' deblend_sources does but with the
    # catalog positions as markers: every above-threshold pixel drains to the known position whose basin it is in.
    smooth_p, bad_p = P(smooth, 0.0), P(bad, True)
    markers = np.zeros(smooth_p.shape, np.int32)
    xi, yi = np.round(x[inside]).astype(int), np.round(y[inside]).astype(int)
    ids = np.flatnonzero(inside) + 1
    markers[yi, xi] = np.where(markers[yi, xi] == 0, ids, markers[yi, xi])     # first source wins a shared pixel
    above = (smooth_p > SEG_THRESH_SIGMA * sigma_smooth) & ~bad_p
    above[yi, xi] = True
    segmap = watershed(-smooth_p, markers, mask=above).astype(np.int32)

    return KronFrame(image=image, sub=P(sub, 0.0), smooth=smooth_p, var=P(var, np.nan), bad=bad_p, segmap=segmap, pad=pad,
                     x=x, y=y, inside=inside, sigma_psf=sigma_psf, sigma_sky=sigma_sky, sigma_smooth=sigma_smooth,
                     sky_slope_min=sky_slope_min)


def _cutout(frame: KronFrame, k: int, half: int):
    """Arrays around source k with the source's own pixel offsets; other sources' basins count as masked.

    psphot instead subtracts the other sources' fitted models and adds the target back (psphotKronIterate_Threaded:
    pmSourceAdd / pmSourceSub); masking by segment is our stand-in.
    """
    xc, yc = frame.x[k], frame.y[k]
    x0, y0 = int(round(xc)), int(round(yc))
    sl = (slice(y0 - half, y0 + half + 1), slice(x0 - half, x0 + half + 1))
    seg = frame.segmap[sl]
    masked = frame.bad[sl] | ((seg != 0) & (seg != k + 1))
    yy, xx = np.mgrid[y0 - half:y0 + half + 1, x0 - half:x0 + half + 1]
    return frame.sub[sl], frame.smooth[sl], frame.var[sl], masked, xx - xc, yy - yc


def sky_radius(frame: KronFrame, k: int, r_max: float) -> float:
    """psphot's `skyRadius` (IPP:psphot/src/psphotRadialProfileWings.c), px: log-spaced annuli from 4 sigma_PSF outward,
    robust median per annulus; stop where the profile slope flattens below the sky model's sigma (interpolating to
    zero flux: lastRadius + lastFlux / SKY_SLOPE_MIN), reaches the sky within NSIGMA, or rises."""
    r_min = MIN_WINDOW_SIGMA * frame.sigma_psf
    if r_min >= r_max:
        return r_max
    half = int(np.ceil(r_max)) + 1
    sub, _, var, masked, dx, dy = _cutout(frame, k, half)
    r = np.hypot(dx, dy)
    alpha = (r_max / r_min) ** (1.0 / SKY_NSTEP) - 1.0
    min_dr = SKY_MIN_DR_ARCSEC / frame.image.pixel_scale
    last_med = last_rmean = np.nan
    r_in = r_min
    while r_in < r_max:
        dr = max(r_in * alpha, min_dr)
        r_out = r_in + dr
        r_mean = (2.0 / 3.0) * (r_out ** 3 - r_in ** 3) / (r_out ** 2 - r_in ** 2)     # flux-weighted mean radius of the annulus
        ann = (r >= r_in) & (r < r_out) & ~masked
        if ann.sum() >= 4:
            med = float(np.median(sub[ann]))
            err = float(np.sqrt(np.nanmean(var[ann]) / ann.sum()))
            if np.isfinite(last_med):
                slope = (med - last_med) / (r_mean - last_rmean)
                if abs(slope) < frame.sky_slope_min:
                    return float(min(max(last_rmean + last_med / frame.sky_slope_min, r_min), r_max))
                if med > last_med:
                    return float(min(r_out, r_max))
            if med - SKY_NSIGMA * err <= 0.0:
                return float(min(r_out, r_max))
            last_med, last_rmean = med, r_mean
        r_in = r_out
    return float(r_max)


def plain_moment(frame: KronFrame, k: int, window: float) -> float:
    """Ordinary first radial moment sum f r / sum f inside `window` on the unsmoothed image (Magnier et al. 2020 eq. 18;
    psphot's source-statistics moment, used for the floor). NaN if the flux is not positive."""
    half = int(np.ceil(window)) + 1
    sub, _, _, masked, dx, dy = _cutout(frame, k, half)
    r = np.hypot(dx, dy)
    ok = (r <= window) & ~masked
    den = sub[ok].sum()
    return float((sub[ok] * r[ok]).sum() / den) if den > 0 else np.nan


def kron_radius(frame: KronFrame, k: int, window: float):
    """psphotKronRadiusMeasure: symmetrised first radial moment inside `window` on the smoothed image.

    Each pixel is paired with its mirror through the centre (both must be unmasked); the pair contributes
    sign(f1) sqrt(|f1 f2|) r to the numerator and (f1 + f2)/2 to the denominator. Returns (R1 px, peak S/N);
    R1 is NaN when either sum is not positive, which psphot treats as no measurement.
    """
    half = int(np.ceil(window)) + 1
    _, sm, _, masked, dx, dy = _cutout(frame, k, half)
    r = np.hypot(dx, dy)
    xc_i, yc_i = frame.x[k] - (int(round(frame.x[k])) - half), frame.y[k] - (int(round(frame.y[k])) - half)
    rows, cols = np.indices(sm.shape)
    mr, mc = (yc_i - (rows - yc_i)).astype(int), (xc_i - (cols - xc_i)).astype(int)      # mirror pixel, as psphot indexes it
    ok = (r <= window) & ~masked & (mr >= 0) & (mr < sm.shape[0]) & (mc >= 0) & (mc < sm.shape[1])
    mr, mc = np.clip(mr, 0, sm.shape[0] - 1), np.clip(mc, 0, sm.shape[1] - 1)
    ok &= ~masked[mr, mc]
    f1, f2 = sm[ok], sm[mr[ok], mc[ok]]
    num = np.sum(np.sign(f1) * np.sqrt(np.abs(f1 * f2)) * r[ok])
    den = np.sum(0.5 * (f1 + f2))
    peak_snr = float(sm[int(round(yc_i)), int(round(xc_i))] / frame.sigma_smooth)
    if not (num > 0 and den > 0):
        return np.nan, peak_snr
    return float(num / den), peak_snr


def aperture_sum(frame: KronFrame, k: int, radius: float):
    """psphotKronFluxMeasure: whole-pixel sum (pixel centre inside `radius`, px) on the unsmoothed image, no sub-pixel
    sampling and no aperture correction; error sqrt(sum variance). Returns (flux, fluxerr, masked_frac)."""
    half = int(np.ceil(radius)) + 1
    sub, _, var, masked, dx, dy = _cutout(frame, k, half)
    inside = np.hypot(dx, dy) < radius
    use = inside & ~masked
    return float(sub[use].sum()), float(np.sqrt(np.nansum(var[use]))), float(1.0 - use.sum() / max(inside.sum(), 1))


def moment_floor(frame: KronFrame, snr: np.ndarray, blended: Optional[np.ndarray] = None) -> float:
    """psphot's MOMENTS_MIN_KRON (pmSourceMinKronRadius, called from psphotSourceStats.c): the median plain first moment,
    inside the 4 sigma_PSF window, of every non-blended source with S/N >= FLOOR_SNR. Sets and returns frame.r_min (px)."""
    w = MIN_WINDOW_SIGMA * frame.sigma_psf
    sel = frame.inside & (np.asarray(snr, float) >= FLOOR_SNR)
    if blended is not None:
        sel &= ~np.asarray(blended, bool)
    r1 = np.array([plain_moment(frame, k, w) for k in np.flatnonzero(sel)])
    frame.r_min = float(np.nanmedian(r1))
    return frame.r_min


def measure(frame: KronFrame, k: int) -> dict:
    """psphotKronIterate_Threaded for one source: the two-pass window, radius, floor, cap and flux."""
    im = frame.image
    r_max = MAX_RADIUS_ARCSEC / im.pixel_scale
    w_min = MIN_WINDOW_SIGMA * frame.sigma_psf
    out = dict(sky_radius=np.nan, window=np.nan, r1=np.nan, r_ap=np.nan, flux=np.nan, fluxerr=np.nan, masked_frac=np.nan, peak_snr=np.nan, flag=0)
    if not frame.inside[k]:
        return out
    out['sky_radius'] = sky_radius(frame, k, r_max)
    window = min(max(out['sky_radius'], w_min), r_max)       # pass 1: skyRadius, never below PSF_MOMENTS_RADIUS
    r1 = flux = np.nan
    for it in range(KRON_ITERATIONS):
        if it > 0:
            # pass 2: min(6 Mrf, Rmax) with Rmax = sqrt(KronFlux) / sqrt(pi * KRON_SB_MIN_FACTOR * SKY_STDEV)
            window = KRON_WINDOW_FACTOR * r1
            if flux > 0:
                window = min(window, np.sqrt(flux) / np.sqrt(np.pi * KRON_SB_MIN_FACTOR * frame.sigma_sky))
            window = min(max(window, w_min), r_max)
        r1_try, peak_snr = kron_radius(frame, k, window)
        out['peak_snr'] = peak_snr
        if not np.isfinite(r1_try):
            out['flag'] |= FLAG_NO_RADIUS          # psphot: Mrf = NaN, no Kron for this source
            out['window'] = window
            return out
        r1 = max(r1_try, frame.r_min)              # Mrf = MAX(minKronRadius, MrfTry)
        if r1 > r1_try:
            out['flag'] |= FLAG_AT_FLOOR
        if peak_snr < FAINT_PEAK_SNR and r1 > window:   # "saturate the 1st radial moment" for faint peaks
            r1 = window
            out['flag'] |= FLAG_FAINT_CAP
        r_ap = KRON_FACTOR * r1
        if r_ap > r_max:
            r_ap = r_max
            out['flag'] |= FLAG_TRUNCATED
        flux, fluxerr, masked_frac = aperture_sum(frame, k, r_ap)
    out.update(window=window, r1=r1, r_ap=r_ap, flux=flux, fluxerr=fluxerr, masked_frac=masked_frac)
    if not flux > 0:
        out['flag'] |= FLAG_NEGATIVE
    return out


def kron_photometry(frame: KronFrame, which: Optional[np.ndarray] = None) -> pd.DataFrame:
    """`measure` for every source (or those in the boolean `which`), with magnitudes. Radii and windows in arcsec."""
    if not np.isfinite(frame.r_min):
        raise ValueError('call moment_floor(frame, snr) first')
    n = len(frame.x)
    idx = np.flatnonzero(which) if which is not None else np.arange(n)
    rows = [measure(frame, k) for k in idx]
    out = pd.DataFrame(rows, index=idx).reindex(np.arange(n))
    out['flag'] = out['flag'].fillna(-1).astype(int)
    for c in ('sky_radius', 'window', 'r1', 'r_ap'):
        out[c] *= frame.image.pixel_scale
    mag, magerr = img_flux_to_ab_mag(out['flux'].values, frame.image.zero_point, fluxerr=out['fluxerr'].values)
    out['mag'] = np.where(out['flux'].values > 0, mag, np.nan)
    out['magerr'] = np.where(out['flux'].values > 0, magerr, np.nan)
    return out
