"""Filter one field for long-duration transients: ZTF deep-reference objects against the PS1 DR2 stack catalog.

Runs on the field tables from `Extracting.field_tables` (ZTF objects merged across g, r, i with our Kron photometry,
matched to the deduplicated PS1 catalog at 1"). Every object is a row (a ZTF object with its PS1 match, if any, or an
unmatched PS1 object), and every band of it is judged in the same order, each step tagged with its purpose (GROUPS;
`Waterfall.groups`, boxed in `Waterfall.flowchart`):

  1. quality: each survey's side of the band passes its quality checks (`ztf_checks`, `ps1_checks`) or the band is
     dropped. ZTF: the image covers the position, DAOPhot flags = 0, NAN_RADIUS_PX
     clear of masked pixels (at the detection, or at the PS1 position for a non-detection) and, for extended objects,
     a ZTF Kron aperture at most KRON_MAX_MASKED masked. PS1, per band: optionally the residual sky at most `ps1_sky_max` (light the background
     model left behind, from bright-star halos, large galaxies and stack background steps), the position not in one
     of MAST's mis-registered polar skycells (`ps1_skycells`; with or without a PS1 object) and at least
     `ps1_min_nframes` stack input frames (PS1 runs its mixture-model outlier rejection only with more than 6 inputs,
     Waters et al. 2020; few-frame stacks keep single-exposure artefacts), the defect / spike /
     star-core / burntool / saturation
     flags (incl. ON_SPIKE), a measurement in the band (forced photometry counts) and psfQfPerfect > PSF_QF_MIN. An untrustworthy side is never read as a
     non-detection;
  2. confident: are we confident the object is real? (`real_checks`, object-wide) Outside every bright star's mask:
     the circle (`bright_star_radius`) and the four diffraction-spike boxes (`spike_half_length`; optional); optionally a ZTF
     sharpness (`ztf_sharpness`) within `sharp_max` for point sources (`sharp_point_only`; a point-source statistic,
     applied to point sources: in field 860 the extended objects it removed were galaxies, two of them known AGN);
     for in_pstarr objects nStackDetections >=
     PS1_ONLY_MIN_STACK_DETECTIONS; for in_ztf and in_pstarr objects, optionally (`annotate_neighbours`), no other
     object of either survey (ZTF or PS1, matched or not, whether or not it survives any cut) within
     `isolation_arcsec`, nor inside a brighter object's Kron aperture (`neighbour_kron`): a one-sided object that
     close to another object is most often a piece of the same light that the other survey, or the match, assigned to
     that object (a galaxy's outskirts, a source one survey merged with a neighbour), not a source the other survey
     missed;
  3. detection and comparison: on the usable sides (`usable_sides`, step 1: else X) each side's status
     (`detection_status`, step 2) is D (S/N >= SNR_MIN), L (below it: a ZTF detection under SNR_MIN, or a PS1
     measurement, forced fluxes <= 0 included) or U (nothing measured: no ZTF detection, no PS1 object; the limit
     stands in). `COMPARISON` (step 3) maps the pair to
     what the band compares, `{b}_cat`: 0 ZTF against PS1, 1 ZTF
     against PS1's forced measurement where there is one below SNR_MIN and brighter than the PS1 limit, else against
     the PS1 limit, 2 PS1 against the ZTF limit, -1 nothing. Each survey's limit is the quadrant's magnitude at
     which its recovery of the other survey's detections falls to LIMIT_COMPLETENESS (`completeness_limits`: the
     reference is the other survey's detections that pass their own trust and real checks in >= 2 bands, the
     population the one-sided candidates come from; in PSF and in Kron magnitude, a band uses the kind it compares),
     never deeper than the survey's nominal
     depth: ZTF's MAGLIM, or `ps1_stack.stack_depth` of the stack's nFrames at the position (scaled from Chambers et
     al. 2016's single-visit and stack 5-sigma depths and capped at the latter).
     An object's catalog (`object_catalog`) is the kind of comparison its bands carry: at least two two-sided bands
     -> `in_both` (its one-sided bands are dropped); else at least two one-sided bands, all against the PS1 limit ->
     `in_ztf`, all against the ZTF limit -> `in_pstarr`; anything else (fewer than two comparable bands, one
     two-sided band, mixed one-sided kinds) is not comparable (-1, `catalog_reason`) and leaves at the shared stage.
     So the catalog label is only ever given to an object that will be judged, and limits enter only in in_ztf
     and in_pstarr, where there is nothing else. A sub-threshold side (L) is a limit on either survey: its own measurement where that is brighter than the
     survey's limit (PS1's forced photometry; a ZTF detection under SNR_MIN), else the limit.
  4. undesired population: no Gaia proper motion or parallax at 5 sigma;
  5. interesting: |sigma| > SIGMA_BOUNDARY in at least two bands, at least two of them of the same sign;
  6. best (`snr_keep`, optional and off by default, last): a band is kept only if each side measured in its
     comparison (not a limit) reaches that S/N, and the |sigma| requirement is re-checked on the bands that remain.
     The envelope is built before this step, from the S/N >= SNR_MIN population. Off, every candidate carries its
     per-band S/N (`{b}_ztf_snr`, `{b}_ps1_snr`) and the website applies the threshold.

A dropped band stays dropped, and an object leaves as soon as fewer than two bands remain.

Morphology is one decision per object: per band, the excess of PSF - Kron over PSF_KRON_CUT in units of each catalog's
error, combined by inverse variance over the two catalogs; extended when that exceeds PSF_KRON_NSIGMA in >= 2 measured
bands. dmag = ZTF - PS1 uses PSF magnitudes for point sources and Kron magnitudes for extended ones. Why not PSF for
galaxies: a PSF fit recovers only the part of a galaxy that looks like the PSF, C_psf/C_tot = 1 / (1 + (alpha_g /
alpha_psf)^2 / 2) for Gaussian galaxy and PSF widths (Slater, Ivezic & Lupton 2020, AJ, arXiv:1912.08210, sec. III.3
eq. 35), so the same galaxy is brighter in ZTF's ~2" PSF than in PS1's ~1.1" one; Kron apertures scale with the object,
not the seeing. One KDE envelope per band is built from the two-sided bands that reach the |sigma| step, optionally clipped of
outliers first (`clip_envelope_sample`), and every compared band is judged
against it;
fainter than ENVELOPE_MONOTONE_FROM each +-1, 2, 3 sigma edge of an envelope row is the most extreme value of that
edge in it or any brighter row (`widen_faint_rows`).

The optional cuts are a `Cuts` (defaults: the full pipeline; `NO_CUTS`: none of them), passed through `filter_field`,
`filter_prepared`, `assign_catalogs`, `cut_bands` and `run_branch`. Everything is a function of a pandas frame; the
only state is the `Waterfall` record each catalog returns, from which the flowchart and the dropped tables are
derived.
"""
import os
import warnings
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple, Iterable, Union

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
from scipy.spatial import cKDTree
from scipy.stats import norm
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from Extracting.field_tables import BANDS, field_dir
from Extracting.utils import get_data_path
from source_db.constants import SIGMA_CAP
from Extracting import ps1_stack, ps1_skycells

CATALOG = {0: 'in_both', 1: 'in_ztf', 2: 'in_pstarr'}
NOT_COMPARABLE = 'not comparable'   # objects with no catalog (-1): fewer than two comparable bands, or of mixed kinds
UNDETECTED = NOT_COMPARABLE         # the old name
REASONS = ('fewer than two measured bands', 'no comparable band', 'one comparable band', 'one two-sided band', 'mixed one-sided kinds')
SNR_MIN = 5.0                   # sorts each side into detected (D) or not (L): sets catalogs and comparisons
SIGMA_BOUNDARY = 3.0
PSF_KRON_CUT = 0.05             # literature point/extended cut on PSF - Kron (Chambers et al. 2016)
PSF_KRON_NSIGMA = 3.0           # extended if the combined excess over the cut exceeds this many sigma
KRON_MAX_MASKED = 0.10          # ZTF Kron aperture fraction masked (neighbours' basins, bad pixels) above which Kron is contaminated
NAN_RADIUS_PX = 5               # a ZTF measurement or limit this close to a masked ZTF pixel is untrustworthy
GAIA_MATCH_ARCSEC = 5.0
GAIA_OBS_EPOCH = 2021.0
BAD_STACK, SUSPECT_STACK = 128, 64                        # qualityFlag
DEFECT, PEAK_ON_SPIKE, SATSTAR = 2048, 1 << 29, 128        # infoFlag
ON_SPIKE, ON_STARCORE, ON_BURNTOOL, SATSTAR_PROFILE = 8, 16, 32, 4096   # infoFlag2
PSF_QF_MIN = 0.85               # PS1 psfQfPerfect: PSF-weighted unmasked fraction (Magnier et al. 2020 'generous' threshold)
PS1_ONLY_MIN_STACK_DETECTIONS = 2   # in_pstarr objects need PS1 nStackDetections >= this (no ZTF detection confirms them)
BRIGHT_STAR_G_MAX = 16.0        # Gaia stars brighter than this get a mask circle and spike boxes
SELF_ARCSEC = 1.0               # a mask star this close to an object is the object itself
KRON_REACH_ARCSEC = 40.0        # Kron containment is searched out to this radius; real apertures reach ~30"
SPIKE_PA = (0.0, 90.0, 30.0, 150.0)   # ZTF diffraction spikes: position angles (deg, N through E) of the four boxes
SPIKE_WIDTH_ARCSEC = 12.0       # full width of a spike box
SPIKE_GROWTH, SPIKE_ANCHOR_G = 1.15, 16.0   # spike half-length = R * SPIKE_GROWTH ** (SPIKE_ANCHOR_G - G): R at G = 16, ~4 R at G = 6
ENVELOPE_CLIP_BIN = 0.5         # mag, the bins of the envelope-sample clipping
LIMIT_COMPLETENESS = 0.9        # a survey's limit is where its recovery of the other survey's detections drops to this
LIMIT_BIN = 0.25                # mag
ENVELOPE_MONOTONE_FROM = 20.0   # fainter than this, no envelope edge comes in from where a brighter row had it
ERR_FLOOR_MAG = 0.01            # calibration floor per survey (PS1 ubercal, zubercal: ~1%), added in quadrature


@dataclass(frozen=True)
class Cuts:
    """The optional cuts of the pipeline. The defaults are the full pipeline; None (or False) turns a cut off."""
    bright_star_mask: bool = True             # drop objects inside a Gaia G < BRIGHT_STAR_G_MAX star's mask circle or spike boxes
    sharp_max: Optional[float] = 0.5          # drop objects whose S/N^2-weighted DAOPhot |sharp| exceeds this
    sharp_point_only: bool = True             # ... point sources only: extended objects are broad by nature
    ps1_sky_max: Optional[float] = 1e-7       # a PS1 band with residual sky above this (Jy/arcsec^2) is untrustworthy
    ps1_min_nframes: Optional[int] = 4        # a PS1 band from a stack of fewer input frames is untrustworthy
    isolation_arcsec: Optional[float] = 3.0   # drop in_ztf / in_pstarr objects with any other object this close
    neighbour_kron: bool = True               # ... or inside a brighter object's Kron aperture (either survey)
    snr_keep: Optional[float] = None          # optional last step: a candidate's band needs this S/N on every side it measures (off: the site filters on S/N)
    envelope_clip_mad: Optional[float] = 5.0  # envelope sample clipped at this many MAD-sigma per ENVELOPE_CLIP_BIN

    def replace(self, **changes) -> 'Cuts':
        return replace(self, **changes)


NO_CUTS = Cuts(bright_star_mask=False, sharp_max=None, sharp_point_only=False, ps1_sky_max=None, ps1_min_nframes=None, isolation_arcsec=None,
               neighbour_kron=False, snr_keep=None, envelope_clip_mad=None)


# ----------------------------------------------------------------------------------------------- loading

BAND_COLS = ['ra', 'dec', 'ccd', 'qid', 'mag', 'magerr', 'snr', 'chi', 'sharp', 'flags', 'nan_dist_px', 'maglim', 'zkron', 'zkron_err', 'zkron_masked', 'zkron_rap']
OBJ_COLS = ['ra', 'dec', 'ps1_idx'] + [f'{b}_{c}' for b in BANDS for c in ('idx', 'covered')]
PS1_COLS = (['PanSTARR_ID', 'ra', 'dec', 'matched', 'ccd', 'qid', 'on_ztf', 'nan_dist_px', 'qualityFlag', 'nStackDetections',
             'projectionID', 'skyCellID']
            + [f'{b}_{c}' for b in BANDS for c in ('detected', 'projcell', 'skycell')]
            + [f'{b}{c}' for b in BANDS for c in ('PSFMag', 'PSFMagErr', 'KronMag', 'KronMagErr', 'PSFFlux', 'PSFFluxErr',
                                                  'infoFlag', 'infoFlag2', 'psfQfPerfect', 'sky', 'nFrames', 'KronRad')]
            + [f'{b}_covered' for b in BANDS])


def load_field(field_id: int, data_dir: Optional[str] = None) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """The field tables, only the columns the filter reads."""
    d = field_dir(field_id, data_dir)
    bands = {b: pd.read_parquet(os.path.join(d, f'ztf_{b}.parquet'), columns=BAND_COLS) for b in BANDS}
    return (bands, pd.read_parquet(os.path.join(d, 'ztf_objects.parquet'), columns=OBJ_COLS),
            pd.read_parquet(os.path.join(d, 'ps1.parquet'), columns=PS1_COLS))


def _bits(values: np.ndarray, bit: int) -> np.ndarray:
    return (values & bit) != 0


# PS1 per-band flags that make a band's PS1 measurement untrustworthy, in flowchart order: (label, column, bit)
PS1_BAND_FLAGS = [
    ('PS1: not DEFECT', 'infoFlag', DEFECT),
    ('PS1: not PEAK_ON_SPIKE', 'infoFlag', PEAK_ON_SPIKE),
    ('PS1: not ON_SPIKE', 'infoFlag2', ON_SPIKE),
    ('PS1: not ON_STARCORE', 'infoFlag2', ON_STARCORE),
    ('PS1: not ON_BURNTOOL', 'infoFlag2', ON_BURNTOOL),
    ('PS1: not SATSTAR', 'infoFlag', SATSTAR),
    ('PS1: not SATSTAR_PROFILE', 'infoFlag2', SATSTAR_PROFILE),
]
# ----------------------------------------------------------------------------------------------- object table

def _attach(src: pd.DataFrame, cols: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Add many columns in one concat (column-by-column inserts fragment a frame of millions of rows)."""
    old = [c for c in cols if c in src.columns]
    if old:
        src = src.drop(columns=old)
    return pd.concat([src, pd.DataFrame(cols, index=src.index, copy=False)], axis=1, copy=False)


def floor_err(err_mag: np.ndarray) -> np.ndarray:
    """A magnitude error with ERR_FLOOR_MAG added in quadrature."""
    return np.hypot(err_mag, ERR_FLOOR_MAG)


def floor_flux_err(flux: np.ndarray, flux_err: np.ndarray) -> np.ndarray:
    """A flux error with the same floor, as a fraction of the flux, added in quadrature."""
    return np.hypot(flux_err, ERR_FLOOR_MAG / 1.0857 * np.abs(flux))


def floor_snr(snr: np.ndarray) -> np.ndarray:
    """S/N once the floor is in the error: 1 / sqrt(1/snr^2 + floor_frac^2), capped near 1.0857 / ERR_FLOOR_MAG."""
    return snr / np.sqrt(1.0 + (snr * ERR_FLOOR_MAG / 1.0857) ** 2)


NFRAMES_NEIGHBOURS, NFRAMES_MAX_ARCSEC = 5, 60.0   # where there is no PS1 object: median nFrames of its nearest PS1 objects


def nframes_at(ps1: pd.DataFrame, b: str, ra: np.ndarray, dec: np.ndarray, k: int = NFRAMES_NEIGHBOURS,
               max_arcsec: float = NFRAMES_MAX_ARCSEC) -> np.ndarray:
    """PS1 stack frames in band b at positions without a PS1 object: the median `{b}nFrames` of the k nearest PS1
    objects within max_arcsec (coverage changes on small scales, e.g. at chip gaps); NaN without any."""
    ok = ps1[f'{b}nFrames'].values >= 1
    cosd = np.cos(np.radians(np.median(ps1.dec.values)))
    tree = cKDTree(np.c_[ps1.ra.values[ok] * cosd, ps1.dec.values[ok]])
    dist, idx = tree.query(np.c_[np.asarray(ra) * cosd, np.asarray(dec)], k=k, distance_upper_bound=max_arcsec / 3600)
    vals = np.where(np.isfinite(dist), ps1[f'{b}nFrames'].values[ok][np.clip(idx, 0, ok.sum() - 1)], np.nan)
    with warnings.catch_warnings():                       # a position with no neighbour is an all-NaN row
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(vals, axis=1) if len(vals) else np.array([])


def completeness_limit(mag: np.ndarray, recovered: np.ndarray, quad: np.ndarray, cap: Dict[Tuple[int, int], float],
                       completeness: float = LIMIT_COMPLETENESS, step: float = LIMIT_BIN, min_per_bin: int = 20,
                       min_per_quad: int = 200) -> Dict[Tuple[int, int], float]:
    """One survey's detection limit per ZTF quadrant {(ccd, qid): mag}, from the other survey's detections as the
    reference: the magnitude at which the fraction of reference objects (`mag`, the reference survey's magnitude)
    that the survey also detects (`recovered`) falls to `completeness`, interpolated between `step` bins. The reference
    must be clean: spurious reference detections are never recovered and read as incompleteness (raw ZTF detections
    have a ~10% floor of them at every magnitude). A quadrant with fewer than `min_per_quad` reference objects, or
    without a crossing, takes the field's value; where the field's reference runs out (fewer than `min_per_bin` per
    bin) while still above `completeness`, the limit is the edge of the last populated bin, since completeness is
    unknown beyond it. No quadrant goes deeper than its `cap` (the survey's nominal depth: ZTF's MAGLIM, PS1's stack
    depth), which is ~1 mag deeper than where detection actually stops."""
    ok = np.isfinite(mag)
    mag, recovered, quad = mag[ok], recovered[ok], quad[ok]
    edges = np.arange(16, 25 + step, step); mid = edges[:-1] + step / 2
    def limit(sel, last_bin_if_no_crossing=False) -> float:
        n = np.histogram(mag[sel], edges)[0]; d = np.histogram(mag[sel & recovered], edges)[0]
        frac = np.where(n >= min_per_bin, d / np.maximum(n, 1), np.nan)
        below = np.flatnonzero(np.isfinite(frac) & (frac < completeness) & (mid > 19))
        if not len(below):
            populated = np.flatnonzero(np.isfinite(frac) & (mid > 19))
            return edges[populated[-1] + 1] if last_bin_if_no_crossing and len(populated) else np.nan
        i = below[0]
        if i == 0 or not (frac[i - 1] > completeness):       # no bin above the threshold before it (the 19 mag guard)
            return mid[i]
        return mid[i - 1] + (frac[i - 1] - completeness) / (frac[i - 1] - frac[i]) * step   # linear between the two bins
    field_limit = limit(np.ones(len(mag), bool), last_bin_if_no_crossing=True)
    out = {}
    for q, c in cap.items():
        sel = quad == (q[0] * 10 + q[1])
        v = limit(sel) if sel.sum() >= min_per_quad else np.nan
        out[q] = float(np.nanmin([v if np.isfinite(v) else field_limit, c]))
    return out


def object_table(bands: Dict[str, pd.DataFrame], obj: pd.DataFrame, ps1: pd.DataFrame) -> pd.DataFrame:
    """One row per object: every ZTF object (with its PS1 match, if any) followed by every unmatched PS1 object.

    Object-level: `ra`, `dec` (the ZTF position for ZTF objects), `ps1_ra`, `ps1_dec` (the PS1 object's own position),
    `ztf_idx`, `ps1_idx` (-1 without one), `ps1_objid` (PS1's objID, -1 without a PS1 object), `ccd`, `qid`,
    `ps1_qualityFlag`, `ps1_nStackDetections`,
    `ps1_clear` (the PS1 position
    on valid ZTF pixels, NAN_RADIUS_PX from masked ones). Per band b: `{b}_ztf_det`, `{b}_ztf_psf/_err` (DAOPhot),
    `{b}_ztf_kron/_err` (our Kron), `{b}_ztf_kron_masked`, `{b}_ztf_kron_rap` (our Kron aperture radius, "), `{b}_ztf_snr`, `{b}_ztf_chi`, `{b}_ztf_sharp`, `{b}_ztf_flags`,
    `{b}_ztf_nan_dist`, `{b}_ztf_covered`, `{b}_ztf_maglim` (the reference image's); `{b}_ps1_detected` (a PS1
    detection, not forced photometry), `{b}_ps1_psf/_err`, `{b}_ps1_kron/_err`,
    `{b}_ps1_flux/_err`, `{b}_ps1_kron_rad` (PS1's Kron aperture radius, "), `{b}_ps1_snr` (flux / error), `{b}_ps1_meas` (a flux with a positive error, forced included),
    `{b}_ps1_infoFlag`, `{b}_ps1_infoFlag2` (0 without a PS1 object), `{b}_ps1_psfQfPerfect`, `{b}_ps1_sky` (residual
    sky, Jy/arcsec^2), `{b}_ps1_nFrames` (the object's own stack input frames, NaN without a PS1 object),
    `{b}_ps1_limit_nframes` (its own if at least one, else `nframes_at`) and `{b}_ps1_depth` (its `ps1_stack.stack_depth`,
    the nominal PS1 depth).
    `completeness_limits` adds the limits, `assign_catalogs` picks `{b}_ztf_limit` / `{b}_ps1_limit` from them by the
    magnitude a band compares.

    Every error and S/N here carries the ERR_FLOOR_MAG calibration floor (`floor_err`, `floor_flux_err`,
    `floor_snr`); the field tables keep the catalogs' own values.
    """
    ps1_only = np.flatnonzero(~ps1.matched.values)
    n_z, n_p = len(obj), len(ps1_only)
    ps1_idx = np.concatenate([obj.ps1_idx.values, ps1_only])
    has_ps1 = ps1_idx >= 0
    pi = np.where(has_ps1, ps1_idx, 0)
    from_ps1 = lambda values, fill: np.where(has_ps1, values[pi], fill)
    # the quadrant: from the first band with a detection for ZTF objects, from the PS1 query for PS1-only rows
    ccd = np.full(n_z + n_p, -1, np.int16); qid = np.full(n_z + n_p, -1, np.int8)
    for b in BANDS:
        zi = obj[f'{b}_idx'].values; has = (zi >= 0) & (ccd[:n_z] < 0)
        ccd[:n_z][has] = bands[b].ccd.values[zi[has]]; qid[:n_z][has] = bands[b].qid.values[zi[has]]
    ccd[n_z:] = ps1.ccd.values[ps1_only]; qid[n_z:] = ps1.qid.values[ps1_only]
    cols = {
        'ra': np.concatenate([obj.ra.values, ps1.ra.values[ps1_only]]),
        'dec': np.concatenate([obj.dec.values, ps1.dec.values[ps1_only]]),
        'ztf_idx': np.concatenate([np.arange(n_z), np.full(n_p, -1)]),
        'ps1_idx': ps1_idx,
        'ps1_objid': from_ps1(ps1.PanSTARR_ID.values.astype(np.int64), -1),
        'ps1_qualityFlag': from_ps1(ps1.qualityFlag.fillna(0).values.astype(np.int64), 0),
        'ps1_nStackDetections': from_ps1(ps1.nStackDetections.fillna(0).values.astype(np.int16), 0),
        'ps1_projcell': from_ps1(ps1.projectionID.fillna(-1).values.astype(np.int32), -1),      # the skycell PS1's row came from
        'ps1_skycell': from_ps1(ps1.skyCellID.fillna(-1).values.astype(np.int16), -1),
        'ps1_clear': from_ps1(ps1.on_ztf.values & (ps1.nan_dist_px.values > NAN_RADIUS_PX), False),
        'ps1_ra': from_ps1(ps1.ra.values, np.nan), 'ps1_dec': from_ps1(ps1.dec.values, np.nan),
        'ccd': ccd, 'qid': qid,
    }
    key = np.where(ccd >= 0, ccd.astype(np.int32) * 10 + qid, 0)
    nan_p = np.full(n_p, np.nan)
    for b in BANDS:
        cat = bands[b]; zi = obj[f'{b}_idx'].values; det = zi >= 0
        take = lambda col: np.concatenate([np.where(det, cat[col].values[np.clip(zi, 0, None)], np.nan), nan_p])
        lut = np.full(200, np.nan)                                            # quadrant key ccd*10+qid -> MAGLIM
        ml = cat.groupby(['ccd', 'qid']).maglim.first()
        lut[np.array([c * 10 + q for c, q in ml.index])] = ml.values
        cols.update({
            f'{b}_ztf_det': np.concatenate([det, np.zeros(n_p, bool)]),
            f'{b}_ztf_psf': take('mag'), f'{b}_ztf_psf_err': floor_err(take('magerr')),
            f'{b}_ztf_kron': take('zkron'), f'{b}_ztf_kron_err': floor_err(take('zkron_err')),
            f'{b}_ztf_kron_masked': take('zkron_masked').astype(np.float32),
            f'{b}_ztf_kron_rap': take('zkron_rap').astype(np.float32),
            f'{b}_ztf_snr': floor_snr(take('snr')), f'{b}_ztf_chi': take('chi').astype(np.float32), f'{b}_ztf_sharp': take('sharp').astype(np.float32),
            f'{b}_ztf_flags': np.nan_to_num(take('flags'), nan=0).astype(np.int16),
            f'{b}_ztf_nan_dist': take('nan_dist_px').astype(np.float32),
            f'{b}_ztf_covered': np.concatenate([obj[f'{b}_covered'].values, ps1[f'{b}_covered'].values[ps1_only]]),
            f'{b}_ztf_maglim': np.where(ccd >= 0, lut[key], np.nan),
        })
        cols[f'{b}_ps1_detected'] = from_ps1(ps1[f'{b}_detected'].values.astype(bool), False)
        cols[f'{b}_ps1_projcell'] = from_ps1(ps1[f'{b}_projcell'].fillna(-1).values.astype(np.int32), -1)   # the skycell the band's row came from
        cols[f'{b}_ps1_skycell'] = from_ps1(ps1[f'{b}_skycell'].fillna(-1).values.astype(np.int16), -1)
        for kind, col in (('psf', 'PSFMag'), ('psf_err', 'PSFMagErr'), ('kron', 'KronMag'), ('kron_err', 'KronMagErr'),
                          ('flux', 'PSFFlux'), ('flux_err', 'PSFFluxErr'), ('psfQfPerfect', 'psfQfPerfect'), ('sky', 'sky'),
                          ('kron_rad', 'KronRad')):
            cols[f'{b}_ps1_{kind}'] = from_ps1(ps1[f'{b}{col}'].values, np.nan)
        for col in ('infoFlag', 'infoFlag2'):
            cols[f'{b}_ps1_{col}'] = from_ps1(ps1[f'{b}{col}'].fillna(0).values.astype(np.int64), 0)
        for kind in ('psf_err', 'kron_err'):
            cols[f'{b}_ps1_{kind}'] = floor_err(cols[f'{b}_ps1_{kind}'])
        cols[f'{b}_ps1_flux_err'] = floor_flux_err(cols[f'{b}_ps1_flux'], cols[f'{b}_ps1_flux_err'])
        flux, err = cols[f'{b}_ps1_flux'], cols[f'{b}_ps1_flux_err']
        cols[f'{b}_ps1_meas'] = np.isfinite(flux) & (err > 0)
        cols[f'{b}_ps1_snr'] = np.where(cols[f'{b}_ps1_meas'], flux / np.where(err > 0, err, 1.0), np.nan)
        own = from_ps1(ps1[f'{b}nFrames'].values.astype(float), np.nan)
        cols[f'{b}_ps1_nFrames'] = own.astype(np.float32)
        at = own.copy()
        missing = ~(own >= 1)
        at[missing] = nframes_at(ps1, b, cols['ra'][missing], cols['dec'][missing])
        cols[f'{b}_ps1_limit_nframes'] = at.astype(np.float32)
        cols[f'{b}_ps1_depth'] = ps1_stack.stack_depth(at, b).astype(np.float32)   # the nominal depth, a cap on the limit
    return pd.DataFrame(cols, copy=False)

# ----------------------------------------------------------------------------------------------- morphology

def extendedness(src: pd.DataFrame, b: str) -> np.ndarray:
    """Inverse-variance combination over the two catalogs of (PSF - Kron - cut) / sigma; a side without a Kron drops out.
    -inf where neither side has one."""
    num = np.zeros(len(src)); den = np.zeros(len(src))
    for side in ('ps1', 'ztf'):
        excess = src[f'{b}_{side}_psf'].values - src[f'{b}_{side}_kron'].values - PSF_KRON_CUT
        var = src[f'{b}_{side}_psf_err'].values ** 2 + src[f'{b}_{side}_kron_err'].values ** 2
        ok = np.isfinite(excess) & (var > 0)
        num[ok] += excess[ok] / var[ok]; den[ok] += 1.0 / var[ok]
    return np.where(den > 0, num / np.sqrt(np.where(den > 0, den, 1.0)), -np.inf)


def point_source(src: pd.DataFrame) -> np.ndarray:
    """One morphology per object: extended when `extendedness` exceeds PSF_KRON_NSIGMA in >= 2 of the measured bands
    (in the one band if only one is measured)."""
    S = np.array([extendedness(src, b) for b in BANDS])
    n_ext, n_meas = (S > PSF_KRON_NSIGMA).sum(axis=0), np.isfinite(S).sum(axis=0)
    return ~(n_ext >= np.minimum(2, np.maximum(n_meas, 1)))



# ----------------------------------------------------------------------------------------------- trust, detection, catalogs

def ztf_checks(src: pd.DataFrame, b: str, point: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """ZTF's side of band b, (label, passes) in order. A detection is judged where it was measured; a non-detection
    stands for the reference image's limit at the PS1 position, so it needs the image and valid pixels there (without
    a PS1 object it has nothing to be compared with, and only coverage is checked)."""
    det, has_ps1 = src[f'{b}_ztf_det'].values, src.ps1_idx.values >= 0
    clear = np.where(det, src[f'{b}_ztf_nan_dist'].values > NAN_RADIUS_PX, ~has_ps1 | src['ps1_clear'].values)
    kron_ok = src[f'{b}_ztf_kron_masked'].values <= KRON_MAX_MASKED
    return [('ZTF image covers the position', det | src[f'{b}_ztf_covered'].values),
            ('ZTF: DAOPhot flags = 0', ~det | (src[f'{b}_ztf_flags'].values == 0)),
            (f'ZTF: > {NAN_RADIUS_PX} px from a masked pixel', clear),
            (f'extended: ZTF Kron aperture <= {100 * KRON_MAX_MASKED:.0f}% masked', ~det | point | kron_ok)]


def ps1_checks(src: pd.DataFrame, b: str, cuts: Cuts = Cuts()) -> List[Tuple[str, np.ndarray]]:
    """PS1's side of band b, (label, passes) in order; without a PS1 object only the skycell check applies (the PS1
    limit stands in). First the band's row not from one of MAST's mis-registered polar skycells (`annotate_skycells`;
    PS1 invalidated every magnitude there, and the detections and fluxes come from the same bad stacks, so a
    measurement from one is unusable and a missing PS1 object in one says nothing). With `cuts.ps1_sky_max`, then a
    residual sky at most that (a band without a sky value passes); with
    `cuts.ps1_min_nframes`, then a stack of at least that many input frames (nFrames 0, i.e. no stack input, fails)."""
    none = src.ps1_idx.values < 0
    sky_max, min_nframes = cuts.ps1_sky_max, cuts.ps1_min_nframes
    sky = ([(f'PS1: residual sky <= {sky_max:g}', none | ~(src[f'{b}_ps1_sky'].values > sky_max))]
           if sky_max is not None else [])
    frames = ([(f'PS1: stack nFrames >= {min_nframes}', none | (src[f'{b}_ps1_nFrames'].values >= min_nframes))]
              if min_nframes is not None else [])
    stack = [('PS1: not in a flagged polar skycell', ~src[f'{b}_ps1_skycell_bad'].values)]
    return (stack + sky + frames + [(label, none | ~_bits(src[f'{b}_ps1_{col}'].values, bit)) for label, col, bit in PS1_BAND_FLAGS]
            + [('PS1: measured in the band', none | src[f'{b}_ps1_meas'].values),
               (f'PS1: psfQfPerfect > {PSF_QF_MIN}', none | (src[f'{b}_ps1_psfQfPerfect'].values > PSF_QF_MIN))])


def _passes(checks: List[Tuple[str, np.ndarray]]) -> np.ndarray:
    return np.logical_and.reduce([ok for _, ok in checks])


def usable_sides(src: pd.DataFrame, b: str, point: np.ndarray, cuts: Cuts = Cuts()) -> Tuple[np.ndarray, np.ndarray]:
    """Step 1, quality: (ZTF usable, PS1 usable) for band b per object, each side passing all of its checks
    (`ztf_checks`, `ps1_checks`). A side that fails is X in `band_status`: nothing on it, measurement or limit, is
    compared. A bad skycell makes the PS1 side unusable even without a PS1 object."""
    return _passes(ztf_checks(src, b, point)), _passes(ps1_checks(src, b, cuts))


def detection_status(src: pd.DataFrame, b: str) -> Tuple[np.ndarray, np.ndarray]:
    """Step 2, detection: (ZTF, PS1) status of band b per object, what each survey has there: 'D' a measurement at
    S/N >= SNR_MIN, 'L' a measurement below it (a ZTF detection under SNR_MIN, a PS1 measurement, forced fluxes
    included), 'U' nothing measured (no ZTF detection, or no PS1 object): the survey's limit stands in."""
    z = np.where(~src[f'{b}_ztf_det'].values, 'U', np.where(src[f'{b}_ztf_snr'].values >= SNR_MIN, 'D', 'L'))
    p = np.where(src.ps1_idx.values < 0, 'U', np.where(src[f'{b}_ps1_snr'].values >= SNR_MIN, 'D', 'L'))
    return z, p


def band_status(src: pd.DataFrame, b: str, point: np.ndarray, cuts: Cuts = Cuts()) -> Tuple[np.ndarray, np.ndarray]:
    """(ZTF, PS1) status of band b per object: the detection status (`detection_status`) on the usable sides
    (`usable_sides`), 'X' on the others."""
    z_ok, p_ok = usable_sides(src, b, point, cuts)
    z, p = detection_status(src, b)
    return np.where(z_ok, z, 'X'), np.where(p_ok, p, 'X')


# (ZTF status, PS1 status) -> the band's comparison; every other pair compares nothing (-1)
COMPARISON = {('D', 'D'): 0,                       # ZTF against PS1
              ('D', 'L'): 1, ('D', 'U'): 1,        # ZTF against the PS1 limit
              ('L', 'D'): 2, ('U', 'D'): 2}        # PS1 against the ZTF limit


def comparison(z: np.ndarray, p: np.ndarray) -> np.ndarray:
    out = np.full(len(z), -1, np.int8)
    for (zs, ps), c in COMPARISON.items():
        out[(z == zs) & (p == ps)] = c
    return out


def object_catalog(cats: Dict[str, np.ndarray], n_measured: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(catalog, reason) from the kind of comparison each band carries (`comparison`): 0 in_both with at least two
    two-sided bands (its one-sided bands are dropped), 1 in_ztf / 2 in_pstarr with at least two one-sided bands all
    of that kind, else -1 (not comparable) with the reason (REASONS): fewer than two measured bands, no comparable
    band, one comparable band, one two-sided band, or one-sided bands of mixed kinds."""
    n_two = sum((cats[b] == 0).astype(int) for b in BANDS)
    n_z, n_p = sum((cats[b] == 1).astype(int) for b in BANDS), sum((cats[b] == 2).astype(int) for b in BANDS)
    catalog = np.select([n_two >= 2, (n_two == 0) & (n_z >= 2) & (n_p == 0), (n_two == 0) & (n_p >= 2) & (n_z == 0)], [0, 1, 2], -1).astype(np.int8)
    reason = np.select([catalog >= 0, n_measured < 2, n_two == 1, (n_z > 0) & (n_p > 0), n_z + n_p == 1],
                       ['', REASONS[0], REASONS[3], REASONS[4], REASONS[2]], REASONS[1])
    return catalog, reason


def completeness_references(src: pd.DataFrame, cuts: Cuts = Cuts()) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """The reference populations of `completeness_limits`, per band: (ZTF detections that PS1's limit is measured
    against, PS1 detections that ZTF's limit is measured against)."""
    point = point_source(src)
    mask = ~src.bright_star_masked.values if cuts.bright_star_mask else np.ones(len(src), bool)
    isolated = ~(src.nearest_other_arcsec.values <= cuts.isolation_arcsec) if cuts.isolation_arcsec is not None else np.ones(len(src), bool)
    clear_of_kron = ~src.inside_brighter_kron.values if cuts.neighbour_kron else np.ones(len(src), bool)
    population_ok = ~(src.gaia_pm_sig.values >= 5.0) & ~(src.gaia_parallax_over_error.values >= 5.0)
    position_ok = mask & isolated & clear_of_kron & population_ok & ~src.ps1_skycell_bad.values   # no PS1 data to recover with, or to be recovered
    z_sharp_ok = ~sharpness_fails(src, point, cuts)
    z_trust = {b: _passes(ztf_checks(src, b, point)) & src[f'{b}_ztf_det'].values & (src[f'{b}_ztf_snr'].values >= SNR_MIN) for b in BANDS}
    p_trust = {b: _passes(ps1_checks(src, b, cuts)) & src[f'{b}_ps1_detected'].values & (src[f'{b}_ps1_snr'].values >= SNR_MIN) for b in BANDS}
    n_z, n_p = sum(z_trust[b].astype(int) for b in BANDS), sum(p_trust[b].astype(int) for b in BANDS)
    stack_ok = (src.ps1_nStackDetections.values >= PS1_ONLY_MIN_STACK_DETECTIONS) & ~_bits(src.ps1_qualityFlag.values, BAD_STACK | SUSPECT_STACK)
    z_ref = {b: z_trust[b] & (n_z >= 2) & position_ok & z_sharp_ok for b in BANDS}                                   # PS1's limit: ZTF's detections
    p_ref = {b: p_trust[b] & (n_p >= 2) & position_ok & stack_ok & src.ps1_clear.values & src[f'{b}_ztf_covered'].values for b in BANDS}
    return z_ref, p_ref


def completeness_limits(src: pd.DataFrame, cuts: Cuts = Cuts()) -> pd.DataFrame:
    """`{b}_ztf_limit_psf/_kron` and `{b}_ps1_limit_psf/_kron`: each survey's `completeness_limit` per quadrant, in
    PSF and in Kron magnitude, against the other survey's detections that would reach the |sigma| step on their own
    side: passing the reference side's quality and confident checks (`ztf_checks` / `ps1_checks`, the `real_checks`
    that read only the reference survey, under `cuts`) at S/N >= SNR_MIN in at least two of its bands, and the Gaia
    population cuts. That is the population the one-sided candidates are drawn from. Checks that read the
    recovering survey (the PS1 checks for a ZTF reference, sharpness for a PS1 reference) would remove recovered objects
    only and are left out. Recovery is a measurement of the band by the other survey at the 1" match, i.e. what the
    pipeline would use in place of the limit: a ZTF DAOPhot detection at any S/N, a PS1 detection or forced flux > 0
    (a PS1 magnitude). Caps: ZTF's MAGLIM per quadrant, PS1's published stack depth per quadrant and `{b}_ps1_depth`
    per object."""
    key = np.where(src.ccd.values >= 0, src.ccd.values.astype(np.int32) * 10 + src.qid.values, 0)
    quads = sorted({(int(c), int(q)) for c, q in zip(src.ccd.values[src.ccd.values >= 0], src.qid.values[src.ccd.values >= 0])})
    z_refs, p_refs = completeness_references(src, cuts)
    cols = {}
    for b in BANDS:
        z_ref, p_ref = z_refs[b], p_refs[b]
        z_cap = {q: float(v) for q, v in src.groupby(['ccd', 'qid'])[f'{b}_ztf_maglim'].first().items() if q in quads}
        p_cap = {q: ps1_stack.STACK_5SIG[b] for q in quads}
        for kind in ('psf', 'kron'):
            for survey, ref, rec, mag, cap in (('ps1', z_ref, np.isfinite(src[f'{b}_ps1_psf'].values), src[f'{b}_ztf_{kind}'].values, p_cap),
                                               ('ztf', p_ref, src[f'{b}_ztf_det'].values, src[f'{b}_ps1_{kind}'].values, z_cap)):
                lut = np.full(200, np.nan)
                for (c, q), v in completeness_limit(np.where(ref, mag, np.nan), rec, key, cap).items():
                    lut[c * 10 + q] = v
                lim = np.where(src.ccd.values >= 0, lut[key], np.nan)
                cols[f'{b}_{survey}_limit_{kind}'] = (np.fmin(lim, src[f'{b}_ps1_depth'].values) if survey == 'ps1' else lim).astype(np.float32)
    return _attach(src, cols)


def assign_catalogs(src: pd.DataFrame, cuts: Cuts = Cuts()) -> pd.DataFrame:
    """`catalog` per object; per band `{b}_ztf_status`, `{b}_ps1_status`, `{b}_cat` (the comparison), `{b}_point`,
    `{b}_extendedness`, `{b}_kron_contaminated`, `{b}_use_kron` and the magnitudes that enter dmag, `{b}_ztf_use/_err`
    and `{b}_ps1_use/_err`; `catalog` and `catalog_reason` from `object_catalog`. An in_both object keeps only its
    two-sided bands (`{b}_cat` 0); its one-sided bands are dropped (-1), as is every band of an object that is not
    comparable. On the undetected side of a one-sided band, either survey uses its own sub-threshold measurement
    where it has one brighter than its limit (`{b}_ps1_use_forced`, `{b}_ztf_use_meas` True; PS1's forced photometry,
    a ZTF detection under SNR_MIN), else the limit (`{b}_ps1_limit`, `{b}_ztf_limit`: the completeness limit in the
    magnitude the band compares, Kron for extended objects, PSF otherwise). A measurement, however noisy, says more
    than a limit deeper than it.

    Extended sources (`point_source`) use Kron magnitudes, point sources PSF; PSF where a side that needs one lacks a Kron.
    `run_branch` drops objects left with fewer than two compared bands."""
    src = completeness_limits(src, cuts)
    point = point_source(src)
    status = {b: band_status(src, b, point, cuts) for b in BANDS}
    raw = {b: comparison(z, p) for b, (z, p) in status.items()}
    n_measured = sum((src[f'{b}_ztf_det'].values | src[f'{b}_ps1_meas'].values).astype(int) for b in BANDS)
    catalog, reason = object_catalog(raw, n_measured)
    cols = {'catalog': catalog, 'catalog_reason': reason}
    for b, (z, p) in status.items():
        cat = raw[b].copy()
        cat[(catalog == 0) & (cat > 0)] = -1                  # in_both: two-sided bands only
        cat[catalog < 0] = -1                                 # not comparable: nothing is compared
        has_kron = ((cat == 2) | np.isfinite(src[f'{b}_ztf_kron'].values)) & ((cat == 1) | np.isfinite(src[f'{b}_ps1_kron'].values))
        use_kron = (cat >= 0) & ~point & has_kron
        z_mag = np.where(use_kron, src[f'{b}_ztf_kron'].values, src[f'{b}_ztf_psf'].values)
        z_err = np.where(use_kron, src[f'{b}_ztf_kron_err'].values, src[f'{b}_ztf_psf_err'].values)
        p_mag = np.where(use_kron, src[f'{b}_ps1_kron'].values, src[f'{b}_ps1_psf'].values)
        p_err = np.where(use_kron, src[f'{b}_ps1_kron_err'].values, src[f'{b}_ps1_psf_err'].values)
        limit = np.where(use_kron, src[f'{b}_ps1_limit_kron'].values, src[f'{b}_ps1_limit_psf'].values)
        forced = (cat == 1) & (p == 'L') & np.isfinite(p_mag) & (p_mag < limit)
        z_limit = np.where(use_kron, src[f'{b}_ztf_limit_kron'].values, src[f'{b}_ztf_limit_psf'].values)
        z_meas = (cat == 2) & (z == 'L') & np.isfinite(z_mag) & (z_mag < z_limit)
        cols.update({
            f'{b}_ztf_status': z, f'{b}_ps1_status': p, f'{b}_cat': cat,
            f'{b}_extendedness': extendedness(src, b), f'{b}_point': point,
            f'{b}_kron_contaminated': src[f'{b}_ztf_det'].values & ~point & ~(src[f'{b}_ztf_kron_masked'].values <= KRON_MAX_MASKED),
            f'{b}_use_kron': use_kron, f'{b}_ztf_limit': z_limit, f'{b}_ps1_limit': limit,
            f'{b}_ztf_use': np.where(cat == 2, np.where(z_meas, z_mag, z_limit), z_mag),
            f'{b}_ztf_use_err': np.where(cat == 2, np.where(z_meas, z_err, 0.0), z_err),
            f'{b}_ztf_use_meas': z_meas,
            f'{b}_ps1_use': np.where(cat == 1, np.where(forced, p_mag, limit), p_mag),
            f'{b}_ps1_use_err': np.where(cat == 1, np.where(forced, p_err, 0.0), p_err),
            f'{b}_ps1_use_forced': forced,
        })
    return _attach(src, cols)

def envelope_tables(src: pd.DataFrame, use: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Table]:
    """The two-sided bands per band in the old column schema, magnitudes already chosen by morphology; with `use`
    ({band: rows}), only those rows (the pipeline passes the bands that reach the |sigma| step)."""
    tabs = {}
    for b in BANDS:
        rows = (src[f'{b}_cat'].values == 0) & np.isfinite(src[f'{b}_ztf_use'].values) & np.isfinite(src[f'{b}_ps1_use'].values)
        s = src[rows & (use[b] if use is not None else True)]
        tabs[b] = Table({f'ZTF_{b}PSFMag': s[f'{b}_ztf_use'].values, f'ZTF_{b}PSFMagErr': s[f'{b}_ztf_use_err'].values,
                         f'PSTARR_{b}PSFMag': s[f'{b}_ps1_use'].values, f'PSTARR_{b}PSFMagErr': s[f'{b}_ps1_use_err'].values})
    return tabs


def clip_envelope_sample(src: pd.DataFrame, use: Dict[str, np.ndarray], k_mad: float = 5.0,
                         bin_mag: float = ENVELOPE_CLIP_BIN) -> Dict[str, np.ndarray]:
    """`use` ({band: rows}) restricted to its two-sided bands, less those whose dmag lies more than k_mad MAD-sigma
    (1.4826 MAD) from the median dmag of their `bin_mag` bin of mean magnitude, repeated until none do. For a Gaussian,
    5 MAD-sigma removes ~6e-7 of the distribution, so the clip removes the stack-artefact, fragment and mismatch
    outliers (10-60 MAD-sigma out) that otherwise set the envelope's +-3 sigma edges, not the population's tails."""
    out = {}
    for b in BANDS:
        z, p = src[f'{b}_ztf_use'].values.astype(float), src[f'{b}_ps1_use'].values.astype(float)
        keep = use[b] & (src[f'{b}_cat'].values == 0) & np.isfinite(z) & np.isfinite(p)
        bins, d = np.floor((z + p) / 2 / bin_mag), z - p
        while True:
            idx = np.flatnonzero(keep)
            s = pd.DataFrame({'bin': bins[idx], 'd': d[idx]})
            med = s.groupby('bin').d.transform('median').values
            mad = 1.4826 * (s.d - med).abs().groupby(s.bin).transform('median').values
            bad = np.abs(s.d.values - med) > k_mad * mad
            if not bad.any():
                break
            keep[idx[bad]] = False
        out[b] = keep
    return out


def _row_edges(row: np.ndarray, dmag_grid: np.ndarray, levels: Tuple[float, ...]) -> Tuple[np.ndarray, np.ndarray, float]:
    """(lower edges at -level sigma, upper edges at +level sigma, median) of one envelope row; NaNs for an empty row."""
    if not np.isfinite(row).all() or row.sum() <= 0:
        return np.full(len(levels), np.nan), np.full(len(levels), np.nan), np.nan
    cdf = np.cumsum(row) / row.sum()
    q = lambda p: np.interp(p, cdf, dmag_grid)
    return np.array([q(norm.cdf(-n)) for n in levels]), np.array([q(norm.cdf(n)) for n in levels]), q(0.5)


def _move_edges(row: np.ndarray, dmag_grid: np.ndarray, old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """The row with its quantiles at the knots `old` moved to `new` (piecewise linear between knots, shifted beyond
    the outermost), returned as per-grid-step probabilities whose running sum is the new CDF."""
    cdf = np.cumsum(row) / row.sum()
    pad = dmag_grid[-1] - dmag_grid[0]
    back = np.interp(dmag_grid, np.r_[new[0] - pad, new, new[-1] + pad], np.r_[old[0] - pad, old, old[-1] + pad])
    return np.diff(np.interp(back, dmag_grid, cdf, left=0.0, right=1.0), prepend=0.0)


def widen_faint_rows(kde_ys: np.ndarray, mag_grid: np.ndarray, dmag_grid: np.ndarray,
                     from_mag: float = ENVELOPE_MONOTONE_FROM, levels: Tuple[float, ...] = (3.0, 2.0, 1.0)) -> np.ndarray:
    """Fainter than `from_mag`, every +-1, 2, 3 sigma edge of a row is moved out to the most extreme value that edge
    takes in this or any brighter row (the upper edges to the running maximum, the lower to the running minimum); the
    median is left where it is. The scatter cannot shrink with magnitude; where it does, the faint windows hold too
    few objects to sample the tails."""
    out = kde_ys.copy()
    lowest, highest = np.full(len(levels), np.inf), np.full(len(levels), -np.inf)
    for i in range(len(mag_grid)):
        lo, hi, med = _row_edges(out[i], dmag_grid, levels)
        if not np.isfinite(med):
            continue
        lowest, highest = np.minimum(lowest, lo), np.maximum(highest, hi)
        if mag_grid[i] > from_mag and ((lo > lowest).any() or (hi < highest).any()):
            out[i] = _move_edges(out[i], dmag_grid, np.r_[lo, med, hi[::-1]], np.r_[lowest, med, highest[::-1]])
    return out


class Envelope_KDE:
    """Rolling KDE envelope over the (mag, Δmag) plane for one band: `get_percentile` for plotting, `sigma` for judging."""

    def __init__(self, mag_grid: np.ndarray, dmag_grid: np.ndarray, kde_ys: np.ndarray):
        self.mag_grid = mag_grid
        self.dmag_grid = dmag_grid
        self.kde_ys = kde_ys  # shape (N_mag, N_dmag)

    def get_percentile(
        self,
        mag_grid: np.ndarray,
        percentile: Union[float, Iterable[float]],
    ) -> np.ndarray:
        """Return the Δmag value at *percentile* for each entry of *mag_grid*.

        Returns a 1-D array if *percentile* is scalar, otherwise a 2-D array
        of shape (len(percentiles), len(mag_grid)).
        """
        scalar = isinstance(percentile, (int, float))
        percentiles = [percentile] if scalar else list(percentile)

        cdf = np.cumsum(self.kde_ys, axis=1) / np.sum(self.kde_ys, axis=1, keepdims=True)

        results = np.array([
            [np.interp(p, cdf[i], self.dmag_grid) for i in range(len(self.mag_grid))]
            for p in percentiles
        ])
        results = np.array([
            np.interp(mag_grid, self.mag_grid, results[i])
            for i in range(len(percentiles))
        ])

        return results[0] if scalar else results

    def sigma(self, mag: np.ndarray, dmag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Signed sigma of each (mag, Δmag): Φ^-1 of the Δmag CDF at mag, bilinear on the grid; NaN where undefined.

        Returns (sigma, capped). Negative = brighter in ZTF than PS1. |sigma| beyond what the grid resolves is
        reported as ±SIGMA_CAP with capped=True.
        """
        mag, dmag = np.asarray(mag, float), np.asarray(dmag, float)
        ys = np.nan_to_num(self.kde_ys, nan=0.0)
        tot = ys.sum(axis=1, keepdims=True)
        cdf = np.divide(np.cumsum(ys, axis=1), tot, out=np.full_like(ys, np.nan), where=tot > 0)
        mg, dg = self.mag_grid, self.dmag_grid
        fi = np.clip((mag - mg[0]) / (mg[1] - mg[0]), 0, len(mg) - 1 - 1e-9)
        fj = np.clip((dmag - dg[0]) / (dg[1] - dg[0]), 0, len(dg) - 1 - 1e-9)
        bad = ~np.isfinite(mag) | ~np.isfinite(dmag)
        fi, fj = np.nan_to_num(fi), np.nan_to_num(fj)
        i0, j0 = fi.astype(int), fj.astype(int)
        w, u = fi - i0, fj - j0
        c = ((1 - w) * ((1 - u) * cdf[i0, j0] + u * cdf[i0, j0 + 1]) +
             w * ((1 - u) * cdf[i0 + 1, j0] + u * cdf[i0 + 1, j0 + 1]))
        p_cap = norm.cdf(-SIGMA_CAP)
        low = (c < p_cap) | (dmag < dg[0])
        high = (c > 1 - p_cap) | (dmag > dg[-1])
        capped = (low | high) & ~bad & np.isfinite(c)
        z = norm.ppf(np.clip(c, p_cap, 1 - p_cap))
        # Off the Δmag grid the CDF above is the grid edge's, not the cap; report the cap.
        z[capped & low] = -SIGMA_CAP
        z[capped & high] = SIGMA_CAP
        z[bad | ~np.isfinite(c)] = np.nan
        return z, capped


def kde_envelopes(tabs: Dict[str, Table], mag_grid: np.ndarray = np.arange(16, 24, 0.25),
                  dmag_grid: np.ndarray = np.linspace(-5, 5, 500), half_bin_width: float = 0.5,
                  min_dmagerr: float = 0.01, n_sigma: float = 10.0, chunk: int = 200_000,
                  monotone_from: Optional[float] = ENVELOPE_MONOTONE_FROM) -> Dict[str, Envelope_KDE]:
    """Rolling KDE envelopes (a window of +-half_bin_width around each grid
    magnitude, a Gaussian per source with bandwidth = combined error, floored at min_dmagerr), computed fast: each
    source's Gaussian is evaluated once, only within +-n_sigma (beyond 10 sigma the weight is ~1e-23), and accumulated into sub-bins of the grid step; a
    window is the sum of the sub-bins it covers. Rows are normalised later (Envelope_KDE builds the CDF per row).
    Fainter than `monotone_from` (None: off), rows are widened by `widen_faint_rows`. The grid starts at 16 mag: the
    bright-star mask removes every Gaia G < 16 star, so brighter rows would sample galaxies, not the typical object;
    brighter sources are judged against the 16 mag row (`Envelope_KDE.sigma` clamps to the grid)."""
    step = mag_grid[1] - mag_grid[0]
    per_window = int(round(2 * half_bin_width / step))                    # sub-bins per window (4 for 0.5 / 0.25)
    edge0 = mag_grid[0] - half_bin_width
    n_sub = len(mag_grid) + per_window - 1
    dg = dmag_grid[1] - dmag_grid[0]
    envelopes = {}
    for band, tab in tabs.items():
        zm, ze = np.asarray(tab[f'ZTF_{band}PSFMag'], float), np.asarray(tab[f'ZTF_{band}PSFMagErr'], float)
        pm, pe = np.asarray(tab[f'PSTARR_{band}PSFMag'], float), np.asarray(tab[f'PSTARR_{band}PSFMagErr'], float)
        mag, d = (zm + pm) / 2, zm - pm
        sig = np.maximum(np.sqrt(pe ** 2 + ze ** 2), min_dmagerr)
        ok = np.isfinite(mag) & np.isfinite(d) & np.isfinite(sig) & (sig < 100.0)
        sub = np.floor((mag - edge0) / step).astype(int)
        ok &= (sub >= 0) & (sub < n_sub)
        mag, d, sig, sub = mag[ok], d[ok], sig[ok], sub[ok]
        acc = np.zeros(n_sub * len(dmag_grid))
        for c0 in range(0, len(d), chunk):
            dd, ss, bb = d[c0:c0 + chunk], sig[c0:c0 + chunk], sub[c0:c0 + chunk]
            lo = np.clip(np.ceil((dd - n_sigma * ss - dmag_grid[0]) / dg).astype(int), 0, len(dmag_grid))
            hi = np.clip(np.floor((dd + n_sigma * ss - dmag_grid[0]) / dg).astype(int) + 1, 0, len(dmag_grid))
            n = np.maximum(hi - lo, 0)
            src_i = np.repeat(np.arange(len(dd)), n)
            gi = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + lo[src_i]
            z = (dmag_grid[gi] - dd[src_i]) / ss[src_i]
            w = np.exp(-0.5 * z * z) / ss[src_i]
            acc += np.bincount(bb[src_i] * len(dmag_grid) + gi, weights=w, minlength=acc.size)
        acc = acc.reshape(n_sub, len(dmag_grid))
        counts = np.bincount(sub, minlength=n_sub)
        kde_ys = np.array([acc[i:i + per_window].sum(axis=0) for i in range(len(mag_grid))])
        n_win = np.array([counts[i:i + per_window].sum() for i in range(len(mag_grid))])
        kde_ys[n_win <= 1] = np.nan
        if monotone_from is not None:
            kde_ys = widen_faint_rows(kde_ys, mag_grid, dmag_grid, monotone_from)
        envelopes[band] = Envelope_KDE(mag_grid, dmag_grid, kde_ys)
    return envelopes


def annotate_sigma(src: pd.DataFrame, envelopes: Dict[str, Envelope_KDE]) -> pd.DataFrame:
    """`{b}_dmag`, `{b}_sigma` (ZTF brighter = negative) and `{b}_capped` for every row with a usable band. A limit
    bounds one direction only: against the PS1 limit (PS1 at least that faint) a band can only have brightened, so
    sigma is clipped at 0 from above, and against the ZTF limit only faded, clipped from below; a sub-threshold
    measurement standing in for a limit is a measurement and stays two-sided."""
    cols = {}
    for b in BANDS:
        z, p = src[f'{b}_ztf_use'].values.astype(float), src[f'{b}_ps1_use'].values.astype(float)
        dmag = z - p
        sigma, capped = envelopes[b].sigma((z + p) / 2, dmag)
        cat = src[f'{b}_cat'].values
        against_ps1_limit = (cat == 1) & ~src[f'{b}_ps1_use_forced'].values
        against_ztf_limit = (cat == 2) & ~src[f'{b}_ztf_use_meas'].values
        sigma = np.where(against_ps1_limit, np.fmin(sigma, 0.0), np.where(against_ztf_limit, np.fmax(sigma, 0.0), sigma))
        sigma[(cat < 0) | ~(p > 14.0)] = np.nan
        cols.update({f'{b}_dmag': dmag, f'{b}_sigma': sigma, f'{b}_capped': capped & np.isfinite(sigma)})
    return _attach(src, cols)


# ----------------------------------------------------------------------------------------------- Gaia

def _gaia_login():
    """Log in to the Gaia archive with ~/vault credentials when available (anonymous queries have small result caps)."""
    from astroquery.gaia import Gaia
    try:
        from Extracting.utils import get_credentials
        user, password = get_credentials('gaia_login.txt')
        if getattr(Gaia, '_TapPlus__user', None) is None:
            Gaia.login(user=str(user), password=str(password))
    except Exception as e:                                   # no credentials: fall back to anonymous, tiles stay small
        print(f'Gaia login skipped ({e})')
    return Gaia


def _gaia_tile(Gaia, ra0, ra1, dec0, dec1, cap: int, depth: int = 0) -> pd.DataFrame:
    """One box; split into four if the result hits the row cap (so a truncated answer is never kept)."""
    q = f'''SELECT TOP {cap} ra, dec, ref_epoch, parallax_over_error, pm, pmra, pmdec, pmra_error, pmdec_error, astrometric_params_solved
            FROM gaiadr3.gaia_source WHERE ra >= {ra0} AND ra < {ra1} AND dec >= {dec0} AND dec < {dec1}'''
    for attempt in range(3):
        try:
            tab = Gaia.launch_job_async(q).get_results().to_pandas(); break
        except Exception as e:
            if attempt == 2: raise
            print(f'Gaia tile retry ({e})')
    if len(tab) >= cap:
        if depth > 6: raise RuntimeError('Gaia tile still at the row cap after 6 splits')
        rm, dm = (ra0 + ra1) / 2, (dec0 + dec1) / 2
        return pd.concat([_gaia_tile(Gaia, a0, a1, d0, d1, cap, depth + 1) for a0, a1 in ((ra0, rm), (rm, ra1)) for d0, d1 in ((dec0, dm), (dm, dec1))], ignore_index=True)
    return tab


def gaia_table(src: pd.DataFrame, name: str, data_dir: Optional[str] = None, padding_deg: float = 0.05,
               tile_deg: float = 1.0, cap: int = 500_000, field_id: Optional[int] = None) -> pd.DataFrame:
    """Gaia DR3 astrometry over the field. Uses the original pipeline's per-field cache `gaia_cache/<field>.ecsv`
    (`filter_fields.query_gaia_for_field`: same columns, the field's box padded by 0.1 deg) when it exists; otherwise
    queries in tiles of `tile_deg` (a whole-field query is truncated or refused by the archive), each split further if
    it reaches `cap` rows, cached as parquet keyed by the sky box."""
    cache = os.path.join(data_dir or get_data_path(), 'gaia_cache')
    os.makedirs(cache, exist_ok=True)
    if field_id is not None:
        ecsv, pq = os.path.join(cache, f'{int(field_id):06d}.ecsv'), os.path.join(cache, f'{int(field_id):06d}.parquet')
        if os.path.exists(pq):
            return pd.read_parquet(pq)
        if os.path.exists(ecsv):
            t = Table.read(ecsv, format='ascii.ecsv')
            tab = pd.DataFrame({c: np.asarray(t[c].filled(np.nan) if hasattr(t[c], 'filled') and t[c].dtype.kind == 'f' else t[c]) for c in t.colnames})
            tab.to_parquet(pq)                               # a parquet copy: the ECSV takes ~1 min to parse
            return tab
    ra0, ra1 = src.ra.min() - padding_deg, src.ra.max() + padding_deg
    dec0, dec1 = src.dec.min() - padding_deg, src.dec.max() + padding_deg
    path = os.path.join(cache, f'{name}_{ra0:.2f}_{ra1:.2f}_{dec0:.2f}_{dec1:.2f}_tiled.parquet')
    if os.path.exists(path):
        return pd.read_parquet(path)
    Gaia = _gaia_login()
    ra_edges = np.linspace(ra0, ra1, int(np.ceil((ra1 - ra0) / tile_deg)) + 1)
    dec_edges = np.linspace(dec0, dec1, int(np.ceil((dec1 - dec0) / tile_deg)) + 1)
    tiles = [_gaia_tile(Gaia, a0, a1, d0, d1, cap) for a0, a1 in zip(ra_edges[:-1], ra_edges[1:]) for d0, d1 in zip(dec_edges[:-1], dec_edges[1:])]
    tab = pd.concat(tiles, ignore_index=True)
    print(f'Gaia: {len(tab):,} sources in {len(tiles)} tiles')
    tab.to_parquet(path)
    return tab


def annotate_gaia(src: pd.DataFrame, gaia: pd.DataFrame, max_arcsec: float = GAIA_MATCH_ARCSEC) -> pd.DataFrame:
    """Nearest Gaia source within `max_arcsec`, positions propagated to GAIA_OBS_EPOCH: `gaia_pm_sig`,
    `gaia_parallax_over_error` (NaN without a match or without that quantity), as the old pipeline's annotate_gaia."""
    coords = SkyCoord(gaia.ra.values * u.deg, gaia.dec.values * u.deg,
                      pm_ra_cosdec=np.nan_to_num(gaia.pmra.values) * u.mas / u.yr, pm_dec=np.nan_to_num(gaia.pmdec.values) * u.mas / u.yr,
                      obstime=Time(gaia.ref_epoch.values, format='jyear')).apply_space_motion(new_obstime=Time(GAIA_OBS_EPOCH, format='jyear'))
    idx, sep, _ = match_coordinates_sky(SkyCoord(src.ra.values * u.deg, src.dec.values * u.deg), coords)
    ok = sep.arcsec <= max_arcsec
    pm_sig = (gaia.pm / (gaia.pmra_error + gaia.pmdec_error)).values[idx]
    return _attach(src, {'gaia_sep_arcsec': np.where(ok, sep.arcsec, np.nan), 'gaia_pm_sig': np.where(ok, pm_sig, np.nan),
                         'gaia_parallax_over_error': np.where(ok, gaia.parallax_over_error.values[idx], np.nan)})


# ----------------------------------------------------------------------------------------------- bright stars

def bright_star_radius(g: np.ndarray) -> np.ndarray:
    """Mask radius in arcsec around a star of Gaia G magnitude `g`: the Legacy Surveys DR9 MEDIUM radius
    (legacypipe `mask_radius_for_mag`), 59" at G = 10, 15" at G = 14, 8" at G = 16."""
    return 1630.0 * 1.396 ** (-np.asarray(g, float))


def fetch_bright_stars(ra0: float, ra1: float, dec0: float, dec1: float) -> pd.DataFrame:
    """Gaia DR2 stars with G < BRIGHT_STAR_G_MAX in a sky box (`source_id`, `ra`, `dec`, `phot_g_mean_mag`) from the
    Gaia archive; a box across RA = 0 (ra0 < 0 or ra1 > 360) is fetched in two pieces."""
    Gaia = _gaia_login()
    pieces = [(ra0 % 360, 360.0), (0.0, ra1 % 360)] if ra0 < 0 or ra1 > 360 else [(ra0, ra1)]
    tabs = []
    for a0, a1 in pieces:
        q = f"""SELECT source_id, ra, dec, phot_g_mean_mag FROM gaiadr2.gaia_source
                WHERE phot_g_mean_mag < {BRIGHT_STAR_G_MAX} AND ra >= {a0} AND ra <= {a1} AND dec >= {dec0} AND dec <= {dec1}"""
        tabs.append(Gaia.launch_job_async(q).get_results().to_pandas())
    return pd.concat(tabs, ignore_index=True)


def bright_stars(field_id: int, data_dir: Optional[str] = None, box: Optional[Tuple[float, float, float, float]] = None) -> pd.DataFrame:
    """Gaia stars with G < BRIGHT_STAR_G_MAX over the field (`ra`, `dec`, `phot_g_mean_mag`), from the per-field cache
    `gaia_cache/<field>_bright_G16_dr2.parquet`; without one, fetched over `box` = (ra0, ra1, dec0, dec1) and cached."""
    path = os.path.join(data_dir or get_data_path(), 'gaia_cache', f'{int(field_id):06d}_bright_G16_dr2.parquet')
    if not os.path.exists(path):
        if box is None:
            raise FileNotFoundError(f'no bright-star cache for field {field_id}: {path}')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fetch_bright_stars(*box).to_parquet(path)
    stars = pd.read_parquet(path, columns=['ra', 'dec', 'phot_g_mean_mag'])
    return stars[stars.phot_g_mean_mag < BRIGHT_STAR_G_MAX].reset_index(drop=True)


def annotate_skycells(src: pd.DataFrame) -> pd.DataFrame:
    """`{b}_ps1_skycell_bad` per band and `ps1_skycell_bad` (any band): the PS1 data come from one of MAST's
    mis-registered polar skycells (`ps1_skycells.bad_cells`), where nothing PS1 measured is usable. With a PS1
    object, each band is judged by the skycell its own row came from (`{b}_ps1_projcell`, `{b}_ps1_skycell`; an
    object in a skycell overlap can draw its bands from different skycells, and a measurement from a good one stays
    good). Without a PS1 object, every band is bad if any skycell covering the position is (`ps1_skycells.is_bad`):
    PS1 may have seen nothing there because its stack is bad."""
    bad_codes = np.array([100 * p + c for p, c in ps1_skycells.bad_cells()], np.int64)
    has = src.ps1_idx.values >= 0
    empty_bad = np.zeros(len(src), bool)
    empty_bad[~has] = ps1_skycells.is_bad(src.ra.values[~has], src.dec.values[~has])
    cols = {}
    for b in BANDS:
        code = 100 * np.nan_to_num(src[f'{b}_ps1_projcell'].values, nan=-1).astype(np.int64) + np.nan_to_num(src[f'{b}_ps1_skycell'].values, nan=-1).astype(np.int64)
        cols[f'{b}_ps1_skycell_bad'] = np.where(has, np.isin(code, bad_codes), empty_bad)
    cols['ps1_skycell_bad'] = np.logical_or.reduce([cols[f'{b}_ps1_skycell_bad'] for b in BANDS])
    return _attach(src, cols)


def spike_half_length(mag: np.ndarray) -> np.ndarray:
    """Half-length in arcsec of a bright star's diffraction-spike boxes: the mask radius at G = SPIKE_ANCHOR_G, growing
    by SPIKE_GROWTH per magnitude brighter (tuned by eye on field 860's reference images, Notebooks/star_mask_860)."""
    mag = np.asarray(mag, float)
    return bright_star_radius(mag) * SPIKE_GROWTH ** (SPIKE_ANCHOR_G - mag)


def annotate_bright_stars(src: pd.DataFrame, stars: pd.DataFrame, out_to: float = 3.0) -> pd.DataFrame:
    """`bright_star_d_over_R`: distance to the nearest bright star in units of that star's mask radius
    (`bright_star_radius`); inf beyond `out_to` radii of every star. `in_star_spike`: inside one of a star's four
    diffraction-spike boxes (SPIKE_PA, SPIKE_WIDTH_ARCSEC, `spike_half_length`; ZTF is an equatorial Schmidt, so the
    spikes sit at fixed position angles on the sky). `bright_star_masked`: either. A star within SELF_ARCSEC of the
    object is the object itself and does not count."""
    cosd = np.cos(np.radians(np.median(src.dec.values)))
    tree = cKDTree(np.c_[src.ra.values * cosd, src.dec.values])
    R = bright_star_radius(stars.phot_g_mean_mag.values); L = spike_half_length(stars.phot_g_mean_mag.values)
    centres = np.c_[stars.ra.values * cosd, stars.dec.values]
    best = np.full(len(src), np.inf); spike = np.zeros(len(src), bool)
    reach = np.maximum(1.1 * out_to * R, 1.05 * L) / 3600
    for (ra0, dec0), r, l, hits in zip(stars[['ra', 'dec']].values, R, L, tree.query_ball_point(centres, reach)):
        if not hits:
            continue
        hits = np.asarray(hits)
        dx = (src.ra.values[hits] - ra0) * np.cos(np.radians(dec0)) * 3600           # east, arcsec
        dy = (src.dec.values[hits] - dec0) * 3600                                    # north
        sep = np.hypot(dx, dy)
        own = sep < SELF_ARCSEC
        d = np.where(own, np.inf, sep / r)
        np.minimum.at(best, hits, np.where(d <= out_to, d, np.inf))
        for pa in SPIKE_PA:
            th = np.radians(pa)
            along = dx * np.sin(th) + dy * np.cos(th); across = dx * np.cos(th) - dy * np.sin(th)
            spike[hits[~own & (np.abs(along) <= l) & (np.abs(across) <= SPIKE_WIDTH_ARCSEC / 2)]] = True
    return _attach(src, {'bright_star_d_over_R': best, 'in_star_spike': spike, 'bright_star_masked': (best < 1) | spike})


# ----------------------------------------------------------------------------------------------- branches

COMBOS = ('gri', 'gr', 'gi', 'ri')                  # band combinations an object can be counted under
GROUPS = {                                          # what each waterfall step decides, in pipeline order
    'structure': 'what is compared',
    'quality': 'is the data usable?',
    'comparison': 'does the band compare anything?',
    'confident': 'are we confident the object is real?',
    'population': 'is it from a population we want?',
    'interesting': 'is it interesting?',
    'best': 'is it among the best?',
}


@dataclass
class Waterfall:
    """One branch's record: the per-band keep-masks after each step, counted as objects split by the exact combination
    of bands still in (as in the old pipeline's flowcharts)."""
    catalog: str
    steps: List[str] = field(default_factory=list)
    groups: List[str] = field(default_factory=list)                       # step -> its GROUPS key
    packed: List[Dict[str, np.ndarray]] = field(default_factory=list)     # step -> {band: rows still in, bit-packed}
    counts: List[Dict[str, int]] = field(default_factory=list)
    n_rows: int = 0

    def add(self, label: str, band_masks: Dict[str, np.ndarray], group: str = 'structure'):
        assert group in GROUPS, group
        self.n_rows = len(next(iter(band_masks.values())))
        self.steps.append(label)
        self.groups.append(group)
        self.packed.append({b: np.packbits(np.asarray(m, bool)) for b, m in band_masks.items()})
        m = {b: np.asarray(band_masks[b], bool) for b in BANDS}
        exact = lambda combo: np.logical_and.reduce([m[b] if b in combo else ~m[b] for b in BANDS])
        c = {k: int(np.count_nonzero(exact(k))) for k in COMBOS}
        self.counts.append({'objects': sum(c.values()), **c})

    def mask(self, i: int) -> Dict[str, np.ndarray]:
        """The per-band keep-masks after step i (an index into `steps`)."""
        return {b: np.unpackbits(p, count=self.n_rows).astype(bool) for b, p in self.packed[i].items()}

    @property
    def masks(self) -> List[Dict[str, np.ndarray]]:
        return [self.mask(i) for i in range(len(self.steps))]

    @property
    def stats(self) -> pd.DataFrame:
        """Objects left after each step, and their band combinations."""
        return pd.DataFrame([{'group': g, 'step': s, **c} for g, s, c in zip(self.groups, self.steps, self.counts)],
                            columns=['group', 'step', 'objects', *COMBOS])

    def dropped_by(self, label: str) -> Dict[str, np.ndarray]:
        i = self.steps.index(label)
        before, after = self.mask(i - 1), self.mask(i)
        return {b: before[b] & ~after[b] for b in BANDS}

    def flowchart(self, title: Optional[str] = None, boxes: bool = True):
        """Flowchart in the old pipeline's style: objects and their band combinations after each step, boxes sized to
        their text, the cut on each arrow; with `boxes`, a dashed frame around each run of steps of one group
        (GROUPS), named at its top right."""
        import schemdraw
        from schemdraw import flow
        from schemdraw.elements import Line, Rect, Label
        text = lambda c: f"objects: {c['objects']:,}\n" + '\n'.join(f'{k}: {c[k]:,}' for k in COMBOS)
        size = lambda t, pad: (_box_width_for_text(t, padding=pad), 0.45 * (t.count('\n') + 1) + 0.4)
        label_w = max(_box_width_for_text(l, padding=0.0) for l in self.steps[1:]) if len(self.steps) > 1 else 0.0
        runs = []                                                             # (group, first step, last step) of each run
        for i, g in enumerate(self.groups):
            if i and runs and runs[-1][0] == g and i > 0:
                runs[-1][2] = i
            elif i:
                runs.append([g, i, i])
        with schemdraw.Drawing(show=False) as d:
            t = f'{title or self.catalog}\n' + text(self.counts[0])
            w, h = size(t, 1.0)
            prev = d.add(flow.Start(w=w, h=h).label(t))
            elems = [prev]
            for label, c in zip(self.steps[1:], self.counts[1:]):
                d += flow.Arrow().down(d.unit / 2).at(prev.S).label(label)
                t = text(c); w, h = size(t, 0.5)
                prev = d.add(flow.Box(w=w, h=h).anchor('N').label(t))
                elems.append(prev)
            left, right = -label_w - 3.6, max(e.E[0] for e in elems) + 0.4
            if boxes:
                for g, i, j in runs:
                    top = elems[i - 1].S[1] - 0.15                               # from just under the previous box ...
                    bottom = elems[j].S[1] - 0.2                                 # ... to just under the run's last box
                    d += Rect(corner1=(left, bottom), corner2=(right, top)).at((0, 0)).theta(0).linestyle('--').color('gray')
                    d += Label().at((right - 0.15, top - 0.05)).label(g, halign='right', valign='top', color='gray')
            d += Line().at((left - 0.4, 0)).to((left - 0.39, 0)).color('none')   # room for the arrow labels
        return d


def ztf_sharpness(src: pd.DataFrame) -> np.ndarray:
    """One DAOPhot sharpness per object: the inverse-variance mean over its ZTF detections, weighting each band by
    S/N^2 (DAOPhot gives no sharpness error; it scales as 1/S/N). NaN without a ZTF detection."""
    num = np.zeros(len(src)); den = np.zeros(len(src))
    for b in BANDS:
        sharp, snr = src[f'{b}_ztf_sharp'].values, src[f'{b}_ztf_snr'].values
        ok = src[f'{b}_ztf_det'].values & np.isfinite(sharp) & (snr > 0)
        num[ok] += sharp[ok] * snr[ok] ** 2; den[ok] += snr[ok] ** 2
    return np.divide(num, den, out=np.full(len(src), np.nan), where=den > 0)


def sharpness_fails(src: pd.DataFrame, point: np.ndarray, cuts: Cuts = Cuts()) -> np.ndarray:
    """Objects whose S/N^2-weighted |sharp| exceeds `cuts.sharp_max`; with `cuts.sharp_point_only`, point sources only."""
    if cuts.sharp_max is None:
        return np.zeros(len(src), bool)
    fails = np.abs(ztf_sharpness(src)) > cuts.sharp_max
    return fails & point if cuts.sharp_point_only else fails


def real_checks(src: pd.DataFrame, catalog: int, cuts: Cuts = Cuts()) -> List[Tuple[str, np.ndarray]]:
    """Object-wide tests that a `catalog` object is real, (label, passes) in order. PS1's BAD_STACK / SUSPECT_STACK
    (qualityFlag: at most one good measurement among its stack bands) count where the PS1 side is a measurement
    (in_both, in_pstarr); for in_ztf the PS1 side is a limit and the flag only says the object is faint there."""
    checks = []
    sharp_max, isolation_arcsec = cuts.sharp_max, cuts.isolation_arcsec
    if cuts.bright_star_mask:
        checks.append(('outside bright-star masks', ~src.bright_star_masked.values))
    if isolation_arcsec is not None and catalog in (1, 2):                 # neighbour cuts before sharpness: what they
        checks.append((f'no other object within {isolation_arcsec:g}"',    # catch (fragments, pieces of other objects)
                       ~(src.nearest_other_arcsec.values <= isolation_arcsec)))   # is attributed to them in the waterfalls
    if cuts.neighbour_kron and catalog in (1, 2):
        checks.append(("outside every brighter object's Kron aperture", ~src.inside_brighter_kron.values))
    if sharp_max is not None:
        label = f'ZTF: S/N²-weighted |sharp| <= {sharp_max}' + (' (point sources)' if cuts.sharp_point_only else '')
        checks.append((label, ~sharpness_fails(src, src.g_point.values, cuts)))
    if catalog == 2:
        checks.append((f'PS1: nStackDetections >= {PS1_ONLY_MIN_STACK_DETECTIONS}',
                       src.ps1_nStackDetections.values >= PS1_ONLY_MIN_STACK_DETECTIONS))
    if catalog in (0, 2):                                                  # PS1's own verdict on its detections of the object:
        checks.append(('PS1: not BAD_STACK', ~_bits(src.ps1_qualityFlag.values, BAD_STACK)))          # <= 1 good or suspect measurement
        checks.append(('PS1: not SUSPECT_STACK', ~_bits(src.ps1_qualityFlag.values, SUSPECT_STACK)))  # <= 1 good measurement
    return checks


def measured_sides_snr_ok(src: pd.DataFrame, b: str, snr_min: Optional[float] = None) -> np.ndarray:
    """Band b is compared (`{b}_cat` >= 0) and every side measured in that comparison, not a limit, has S/N >=
    `snr_min` (None: no S/N requirement beyond the comparison)."""
    cat = src[f'{b}_cat'].values
    if snr_min is None:
        return cat >= 0
    ztf_ok = ~np.isin(cat, (0, 1)) | (src[f'{b}_ztf_snr'].values >= snr_min)
    ps1_ok = ~np.isin(cat, (0, 2)) | (src[f'{b}_ps1_snr'].values >= snr_min)
    return (cat >= 0) & ztf_ok & ps1_ok


COMPARABLE = '>= 2 comparable bands of one kind'      # the step that ends the shared stage and starts the catalogs


def quality_steps(src: pd.DataFrame, point: np.ndarray, cuts: Cuts = Cuts()) -> List[Tuple[str, str, Dict[str, np.ndarray]]]:
    """The shared per-band steps every object goes through, (group, label, {band: passes}) in order: what each band
    starts from (measured by either survey), the quality checks of both sides (step 1), then whether the band is
    compared (`{b}_cat` >= 0: a comparison kind from the detection statuses, step 2 and 3, of the object's catalog's
    kind, step 4). Detection itself drops nothing, so it is not a step here."""
    def per_band(group, checks_of):             # band -> [(label, passes)], the same labels in every band
        checks = {b: checks_of(b) for b in BANDS}
        return [(group, label, {b: checks[b][i][1] for b in BANDS}) for i, (label, _) in enumerate(checks[BANDS[0]])]
    start = per_band('structure', lambda b: [('measured by ZTF or PS1', src[f'{b}_ztf_det'].values | src[f'{b}_ps1_meas'].values)])
    quality = per_band('quality', lambda b: ztf_checks(src, b, point) + ps1_checks(src, b, cuts))
    comparable = per_band('comparison', lambda b: [(COMPARABLE, src[f'{b}_cat'].values >= 0)])
    return start + quality + comparable


def quality_waterfall(src: pd.DataFrame, cuts: Cuts = Cuts()) -> Waterfall:
    """The shared stage as a waterfall over every object: measured, the quality steps, then comparable (the objects
    the catalogs start from). A per-band cut removes the band for every later step, and an object leaves as soon as
    fewer than two bands remain."""
    wf = Waterfall('all')
    m = {b: np.ones(len(src), bool) for b in BANDS}
    for i, (group, label, passes) in enumerate(quality_steps(src, src['g_point'].values, cuts)):
        for b in BANDS:
            m[b] &= passes[b]
        alive = sum(m[b].astype(int) for b in BANDS) >= 2
        for b in BANDS:
            m[b] &= alive
        wf.add(f'>= 2 bands: {label}' if i == 0 else label, m, group)
    for b in BANDS:
        assert np.array_equal(m[b], (src.catalog.values >= 0) & (src[f'{b}_cat'].values >= 0)), f'{b}: the shared stage does not end at the comparable bands'
    return wf


def cut_bands(src: pd.DataFrame, catalog: int, cuts: Cuts = Cuts()) -> Tuple[Waterfall, Dict[str, np.ndarray]]:
    """One catalog's chain from its comparable objects up to the |sigma| cut, every cut its own waterfall step:
    the confident checks, then the population checks. Returns (waterfall, {band: rows still in}), the bands that
    reach the |sigma| step. An object leaves as soon as fewer than two bands remain."""
    name = CATALOG.get(catalog, NOT_COMPARABLE)
    wf = Waterfall(name)
    rows = src['catalog'].values == catalog
    m = {b: rows & (src[f'{b}_cat'].values >= 0) for b in BANDS}
    wf.add(COMPARABLE, m)
    alive = sum(m[b].astype(int) for b in BANDS) >= 2
    for label, ok in real_checks(src, catalog, cuts):
        for b in BANDS:
            m[b] &= ok
        alive = sum(m[b].astype(int) for b in BANDS) >= 2
        for b in BANDS:
            m[b] &= alive
        wf.add(label, m, 'confident')
    keep = alive.copy()
    keep &= ~(src['gaia_pm_sig'].values >= 5.0)
    wf.add('Gaia: no proper motion at 5σ', {b: keep & m[b] for b in BANDS}, 'population')
    keep &= ~(src['gaia_parallax_over_error'].values >= 5.0)
    wf.add('Gaia: no parallax at 5σ', {b: keep & m[b] for b in BANDS}, 'population')
    return wf, {b: keep & m[b] for b in BANDS}


def judge_bands(src: pd.DataFrame, wf: Waterfall, reach: Dict[str, np.ndarray], cuts: Cuts = Cuts()) -> pd.DataFrame:
    """The |sigma| steps on the bands that reach them (`cut_bands`), then the `snr_keep` step on what passed, added to
    `wf`. Returns the candidates."""
    def sigma_steps(bands, keep, group, suffix=''):
        sig = {b: np.where(bands[b], src[f'{b}_sigma'].values, np.nan) for b in BANDS}
        big = {b: np.abs(sig[b]) > SIGMA_BOUNDARY for b in BANDS}
        keep = keep & (sum(big[b].astype(int) for b in BANDS) >= 2)
        wf.add(f'|σ| > {SIGMA_BOUNDARY:.0f} in >= 2 bands{suffix}', {b: keep & big[b] for b in BANDS}, group)
        pos = sum((sig[b] > SIGMA_BOUNDARY).astype(int) for b in BANDS)
        neg = sum((sig[b] < -SIGMA_BOUNDARY).astype(int) for b in BANDS)
        keep = keep & (np.maximum(pos, neg) >= 2)
        wf.add(f'... >= 2 of them the same sign{suffix}', {b: keep & big[b] for b in BANDS}, group)
        return keep, big
    keep, big = sigma_steps(reach, np.logical_or.reduce(list(reach.values())), 'interesting')
    if cuts.snr_keep is not None:               # the best of the candidates: every measured side at this S/N
        strong = {b: keep & big[b] & measured_sides_snr_ok(src, b, cuts.snr_keep) for b in BANDS}
        wf.add(f'measured sides S/N >= {cuts.snr_keep:g}', strong, 'best')
        keep, big = sigma_steps(strong, keep, 'best', ' at that S/N')
    cand = src[keep].copy()
    cand['catalog'] = wf.catalog
    cand['n_big_dmag_bands'] = sum(big[b][keep].astype(int) for b in BANDS)
    return cand


def run_branch(src: pd.DataFrame, catalog: int, cuts: Cuts = Cuts()) -> Tuple[pd.DataFrame, Waterfall]:
    """One catalog's whole chain on a table that already has `{b}_sigma`. Returns (candidates, waterfall)."""
    wf, reach = cut_bands(src, catalog, cuts)
    return judge_bands(src, wf, reach, cuts), wf

# ----------------------------------------------------------------------------------------------- driver

def annotate_neighbours(src: pd.DataFrame) -> pd.DataFrame:
    """Every object's relation to the other objects of either survey -- every ZTF object at its position and every PS1
    object at its own position, matched or not, whatever it does in the pipeline; the object's own ZTF and PS1
    entries are never counted:

      `nearest_other_arcsec`   distance to the nearest other object (NaN beyond 60");
      `inside_brighter_kron`   inside the Kron aperture of a brighter object (PS1's KronRad, or our ZTF aperture): the
                               light there belongs to that object (a galaxy's outskirts).

    An object's magnitude is its brightest PSF magnitude in any band of either survey; a Kron radius is the largest
    finite one over the bands."""
    cosd = np.cos(np.radians(np.median(src.dec.values)))
    pos = np.c_[src.ra.values * cosd, src.dec.values]
    ztf_rows, ps1_rows = np.flatnonzero(src.ztf_idx.values >= 0), np.flatnonzero(src.ps1_idx.values >= 0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)                                   # all-NaN rows
        z_mag = np.nanmin(np.c_[[src[f'{b}_ztf_psf'].values for b in BANDS]].T, axis=1)
        p_mag = np.nanmin(np.c_[[src[f'{b}_ps1_psf'].values for b in BANDS]].T, axis=1)
        z_rad = np.nanmax(np.c_[[src[f'{b}_ztf_kron_rap'].values for b in BANDS]].T, axis=1)
        p_rad = np.nanmax(np.c_[[src[f'{b}_ps1_kron_rad'].values for b in BANDS]].T, axis=1)
        own_mag = np.nanmin(np.c_[z_mag, p_mag], axis=1)
    owner = np.r_[ztf_rows, ps1_rows]                                                   # the object each entry belongs to
    entries = np.r_[pos[ztf_rows], np.c_[src.ps1_ra.values[ps1_rows] * cosd, src.ps1_dec.values[ps1_rows]]]
    e_mag, e_rad = np.r_[z_mag[ztf_rows], p_mag[ps1_rows]], np.r_[z_rad[ztf_rows], p_rad[ps1_rows]]
    tree = cKDTree(entries)
    dist, k = tree.query(pos, k=3, distance_upper_bound=60 / 3600)                       # up to two entries are the object's own
    own = owner[np.clip(k, 0, len(owner) - 1)] == np.arange(len(src))[:, None]
    nearest = np.where(own | ~np.isfinite(dist), np.inf, dist).min(axis=1)
    # inside a brighter entry's Kron aperture: every object within each aperture's radius (searched out to
    # KRON_REACH_ARCSEC: a larger PS1 KronRad is a deblending accident), as one flat pair list
    in_kron = np.zeros(len(src), bool)
    with_rad = np.flatnonzero(np.isfinite(e_rad))
    hits = cKDTree(pos).query_ball_point(entries[with_rad], np.minimum(e_rad[with_rad], KRON_REACH_ARCSEC) / 3600, return_sorted=False)
    n_hits = np.fromiter((len(h) for h in hits), int, len(hits))
    e = np.repeat(with_rad, n_hits)                                                     # the entry of each pair
    i = np.fromiter((j for h in hits for j in h), int, int(n_hits.sum()))               # the object inside it
    inside = (owner[e] != i) & (e_mag[e] < own_mag[i])
    in_kron[i[inside]] = True
    return _attach(src, {'nearest_other_arcsec': np.where(np.isfinite(nearest), nearest * 3600, np.nan).astype(np.float32),
                         'inside_brighter_kron': in_kron})


def field_box(src: pd.DataFrame, pad_deg: float = 0.4) -> Tuple[float, float, float, float]:
    """(ra0, ra1, dec0, dec1) around the objects, padded for the brightest stars' masks (3 R ~ 0.35 deg at G = 4); a
    field across RA = 0 gives ra0 < 0."""
    ra, dec = src.ra.values, src.dec.values
    if ra.max() - ra.min() > 180:                           # across RA = 0: shift the western half below zero
        ra = np.where(ra > 180, ra - 360, ra)
    return float(ra.min() - pad_deg), float(ra.max() + pad_deg), float(dec.min() - pad_deg), float(dec.max() + pad_deg)


def prepare_field(field_id: int, data_dir: Optional[str] = None) -> pd.DataFrame:
    """The parts that do not depend on the photometry choices: the object table with its Gaia match, its PS1 skycell
    (and whether that is one of MAST's bad polar skycells), and its distance to the nearest bright star and to its
    nearest other-survey neighbours."""
    data_dir = data_dir or get_data_path()
    bands, obj, ps1 = load_field(field_id, data_dir)
    base = object_table(bands, obj, ps1)
    stars = bright_stars(field_id, data_dir, box=field_box(base))
    del bands, obj, ps1
    base = annotate_gaia(base, gaia_table(base, f'field{int(field_id):06d}', data_dir, field_id=field_id))
    return annotate_neighbours(annotate_bright_stars(annotate_skycells(base), stars))


KEPT = 'kept'


def annotate_outcomes(src: pd.DataFrame, waterfalls: Dict[str, 'Waterfall'], candidates: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The record of what happened to every object: `{b}_dropped_by`, the first step that dropped the band (KEPT if
    none: the shared stage's steps, then its catalog branch's), and `outcome`: 'candidate', 'not comparable: <reason>'
    for objects that never entered a branch, else 'dropped: <step>' at which fewer than two bands remained."""
    n = len(src)
    first = {b: np.full(n, -1, np.int32) for b in BANDS}             # step index of the first drop, per band (-1: kept)
    labels = []                                                        # index -> label, shared steps then branch steps
    def walk(wf: 'Waterfall', rows: np.ndarray, offset: int):
        alive = {b: rows & (first[b] < 0) for b in BANDS}              # a band dropped earlier stays dropped
        died_at = np.full(n, -1, np.int32)
        for i in range(len(wf.steps)):
            m = wf.mask(i)
            for b in BANDS:
                newly = alive[b] & ~m[b]
                first[b][newly] = offset + i
                alive[b] &= m[b]
            n_alive = sum(alive[b].astype(int) for b in BANDS)
            died_at[(died_at < 0) & rows & (n_alive < 2)] = offset + i
        return died_at
    all_rows = np.ones(n, bool)
    labels += waterfalls['all'].steps
    died_shared = walk(waterfalls['all'], all_rows, 0)
    died_branch = np.full(n, -1, np.int32)
    for name, wf in waterfalls.items():
        if name == 'all':
            continue
        rows = src.catalog.values == {v: k for k, v in CATALOG.items()}[wf.catalog]
        offset = len(labels); labels += wf.steps
        d = walk(wf, rows, offset)
        died_branch[rows] = d[rows]
    is_cand = np.zeros(n, bool)
    for cand in candidates.values():
        is_cand[src.index.get_indexer(cand.index)] = True
    label_of = lambda idx: np.where(idx < 0, KEPT, np.array(labels + [KEPT], dtype=object)[np.where(idx < 0, len(labels), idx)])
    died = np.where(died_branch >= 0, died_branch, died_shared)
    reason = np.char.add('not comparable: ', src.catalog_reason.astype(str).values.astype(str))
    dropped = np.char.add('dropped: ', label_of(died).astype(str))
    outcome = np.where(is_cand, 'candidate', np.where(src.catalog.values < 0, reason, dropped))
    cols = {f'{b}_dropped_by': pd.Categorical(label_of(first[b])) for b in BANDS}
    cols['outcome'] = pd.Categorical(outcome)
    return _attach(src, cols)


def filter_prepared(base: pd.DataFrame, cuts: Cuts = Cuts(), verbose: bool = True):
    """Catalogs, the shared quality stage, the three branches up to the |sigma| cut, the envelopes from the two-sided
    bands that reach it, then sigma, the |sigma| cut and the `snr_keep` step, on a `prepare_field` table (left
    unmodified). Returns (src, envelopes, candidates, waterfalls); `waterfalls` also holds 'all', the shared stage
    (`quality_waterfall`).

    The envelope sample is the S/N >= SNR_MIN population that reaches the |sigma| step (`snr_keep` comes after), clipped
    by `clip_envelope_sample` when `cuts.envelope_clip_mad` is set."""
    src = assign_catalogs(base, cuts)
    waterfalls = {'all': quality_waterfall(src, cuts)}
    cut = {c: cut_bands(src, c, cuts) for c in CATALOG}
    reach_any = {b: np.logical_or.reduce([reach[b] for _, reach in cut.values()]) for b in BANDS}
    if cuts.envelope_clip_mad is not None:
        reach_any = clip_envelope_sample(src, reach_any, cuts.envelope_clip_mad)
    envelopes = kde_envelopes(envelope_tables(src, reach_any))
    src = annotate_sigma(src, envelopes)
    candidates = {}
    for c, (wf, reach) in cut.items():
        if verbose: print(f'Filtering {wf.catalog}')
        candidates[wf.catalog] = judge_bands(src, wf, reach, cuts)
        waterfalls[wf.catalog] = wf
    src = annotate_outcomes(src, waterfalls, candidates)
    for name, cand in candidates.items():                  # the record rides with the candidates (what the site gets)
        for c in [f'{b}_dropped_by' for b in BANDS] + ['outcome']:
            cand[c] = src[c].values[src.index.get_indexer(cand.index)]
    return src, envelopes, candidates, waterfalls


def filter_field(field_id: int, data_dir: Optional[str] = None, cuts: Cuts = Cuts(), save: bool = True, verbose: bool = True):
    """Everything above for one field. Returns (src, envelopes, candidates, waterfalls), the last two keyed by catalog
    name; with `save`, writes them under `<data>/fields/<field>/filter/`."""
    data_dir = data_dir or get_data_path()
    src, envelopes, candidates, waterfalls = filter_prepared(prepare_field(field_id, data_dir), cuts, verbose=verbose)
    if save:
        out = os.path.join(field_dir(field_id, data_dir), 'filter')
        os.makedirs(out, exist_ok=True)
        src.to_parquet(os.path.join(out, 'objects.parquet'))
        for cname, cand in candidates.items():
            cand.to_parquet(os.path.join(out, f'{cname}_candidates.parquet'))
        for name, wf in waterfalls.items():
            wf.stats.to_csv(os.path.join(out, f'{name}_stats.csv'), index=False)
        for b, env in envelopes.items():
            np.savez(os.path.join(out, f'envelope_{b}.npz'), mag_grid=env.mag_grid, dmag_grid=env.dmag_grid, kde_ys=env.kde_ys)
    return src, envelopes, candidates, waterfalls


def _box_width_for_text(text: str, fontsize: float = 12.5, inches_per_unit: float = 0.5, padding: float = 0.5) -> float:
    """Return the minimum schemdraw Box w needed to contain `text` without clipping.

    Uses TextPath to measure glyph extents, then applies an empirical 1.13× correction
    to match matplotlib's actual rendered width (TextPath underestimates by ~13%).
    """
    fp = FontProperties(size=fontsize)
    max_w_pts = 0.0
    for line in text.split('\n'):
        if line.strip():
            tp = TextPath((0, 0), line, prop=fp)
            bb = tp.get_extents()
            max_w_pts = max(max_w_pts, bb.width)
    text_units = (max_w_pts / 72.0) / inches_per_unit * 1.13
    return max(3.0, text_units + padding)


EXPLAIN_COLS = (['ra', 'dec', 'ztf_idx', 'ps1_idx', 'catalog', 'catalog_reason', 'outcome', 'g_point']
                + [f'{b}_{c}' for b in BANDS for c in ('ztf_status', 'ps1_status', 'ztf_psf', 'ps1_psf', 'ztf_snr', 'ps1_snr', 'sigma', 'dropped_by')])


def explain(field_id: int, ra: float, dec: float, data_dir: Optional[str] = None, max_arcsec: float = 2.0) -> Optional[pd.Series]:
    """What the last filter run did with the object nearest (ra, dec): its statuses, magnitudes, catalog, the step that
    dropped each band and its outcome, read from the saved object table; None if nothing lies within `max_arcsec`."""
    path = os.path.join(field_dir(field_id, data_dir or get_data_path()), 'filter', 'objects.parquet')
    t = pd.read_parquet(path, columns=EXPLAIN_COLS)
    sep = SkyCoord(t.ra.values * u.deg, t.dec.values * u.deg).separation(SkyCoord(ra * u.deg, dec * u.deg)).arcsec
    i = int(np.argmin(sep))
    if sep[i] > max_arcsec:
        print(f'field {field_id}: no object within {max_arcsec}" of ({ra:.5f}, {dec:.5f}); nearest is {sep[i]:.1f}" away')
        return None
    r = t.iloc[i]
    cat = {**CATALOG, -1: NOT_COMPARABLE}[int(r.catalog)]
    fmt = lambda v: f'{v:.2f}' if np.isfinite(v) else '-'
    print(f'field {field_id}: object {sep[i]:.2f}" from ({ra:.5f}, {dec:.5f})  ztf_idx {r.ztf_idx}  ps1_idx {r.ps1_idx}  '
          f'{"point" if r.g_point else "extended"}  catalog: {cat} ({r.catalog_reason})')
    for b in BANDS:
        print(f'  {b}: ZTF {r[f"{b}_ztf_status"]} {fmt(r[f"{b}_ztf_psf"])} (S/N {fmt(r[f"{b}_ztf_snr"])}) | PS1 {r[f"{b}_ps1_status"]} {fmt(r[f"{b}_ps1_psf"])} '
              f'(S/N {fmt(r[f"{b}_ps1_snr"])}) | sigma {fmt(r[f"{b}_sigma"])} | {r[f"{b}_dropped_by"]}')
    print(f'  outcome: {r.outcome}')
    return r


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Filter a field, or explain what the last run did with one object.')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='filter one field; writes objects, candidates, stats and envelopes under <data>/fields/<field>/filter/')
    r.add_argument('field', type=int); r.add_argument('--data-dir', default=None)
    r.add_argument('--snr-keep', type=float, default=None, help='also apply the measured-sides S/N step at this value')
    e = sub.add_parser('explain', help='the object nearest a position: statuses, magnitudes, the step that dropped each band, outcome')
    e.add_argument('field', type=int); e.add_argument('ra', type=float); e.add_argument('dec', type=float)
    e.add_argument('--data-dir', default=None); e.add_argument('--max-arcsec', type=float, default=2.0)
    a = ap.parse_args()
    if a.cmd == 'run':
        _src, _env, _cands, _wfs = filter_field(a.field, a.data_dir, Cuts(snr_keep=a.snr_keep), save=True, verbose=True)
        print('candidates:', {k: len(v) for k, v in _cands.items()}, flush=True)
    else:
        explain(a.field, a.ra, a.dec, a.data_dir, a.max_arcsec)
