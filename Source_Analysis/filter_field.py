"""Filter one field for long-duration transients: ZTF deep-reference objects against the PS1 DR2 stack catalog.

Runs on the field tables from `Extracting.field_tables` (ZTF objects merged across g, r, i with our Kron photometry,
matched to the deduplicated PS1 catalog at 1"). Every object is a row (a ZTF object with its PS1 match, if any, or an
unmatched PS1 object), and every band of it is judged in the same order, each step tagged with its purpose (GROUPS;
`Waterfall.groups`; tables `quality_table`, `sort_table`, `catalog_table`; flowcharts `quality_flowchart`,
`catalog_flowchart`):

  1. quality (`usable_sides`): each survey's side of the band passes its quality checks or is X, unusable for a
     measurement or a limit. ZTF (`ztf_checks`): outside every Gaia G < 16 star's mask (the circle `bright_star_radius`
     and four diffraction-spike boxes `spike_half_length`, calibrated on ZTF's images), the image covers the
     position, DAOPhot flags = 0, DAOPhot chi <= `ztf_chi_max`, residual sky <= `ztf_sky_max`, NAN_RADIUS_PX clear
     of masked pixels (at the detection, or at the PS1 position for a non-detection) and, for extended objects, a ZTF
     Kron aperture at most KRON_MAX_MASKED masked. PS1 (`ps1_checks`): not one of MAST's mis-registered polar
     skycells (`ps1_skycells`; with or without a PS1 object); without a PS1 object, the DR2 catalog covering the
     position (MAST's missing-data boxes, its Dec -30 cut) and some PS1 object measured in the band within
     1' (its stack reaches the position; `field_tables.ps1_neighbourhood`); with one, per band: at least `ps1_min_nframes` stack input
     frames (PS1 runs its mixture-model outlier rejection only with more than 6 inputs, Waters et al. 2020), residual
     sky at most `ps1_sky_max`, none of the DEFECT / PEAK_ON_SPIKE / ON_SPIKE / ON_STARCORE / ON_BURNTOOL / SATSTAR /
     SATSTAR_PROFILE flags (PS1_BAND_FLAGS), a measurement in the band (forced photometry counts) and psfQfPerfect >
     PSF_QF_MIN;
  2. detection (`detection_status`): on a usable side, D a measurement at S/N >= SNR_MIN, L one below it (a ZTF
     detection under SNR_MIN, a PS1 measurement, forced fluxes <= 0 included), U nothing measured (no ZTF detection, no
     PS1 object): the survey's limit stands in;
  3. comparison and sorting: `COMPARISON` maps the (ZTF, PS1) pair to what the band compares, `{b}_cat`: 0 ZTF
     against PS1 (D, D); 1 ZTF against the PS1 limit (D with L or U; a forced PS1 measurement brighter than the limit
     is used instead); 2 PS1 against the ZTF limit (L or U with D); -1 nothing (an X, or neither side D). The object's
     catalog (`object_catalog`) is the kind its bands carry: at least two two-sided bands -> `in_both` (its one-sided
     bands are dropped); else at least two one-sided bands all of one kind -> `in_ztf` / `in_pstarr`; anything else is
     not comparable (-1, `catalog_reason`) and leaves here. Each survey's limit is the quadrant's magnitude at which
     its recovery of the other survey's detections falls to LIMIT_COMPLETENESS (`completeness_limits`: the reference
     is the other survey's detections passing their own quality and confident checks in >= 2 bands and the Gaia cuts,
     the population the one-sided candidates come from; each reference binned by the magnitude it compares,
     `reference_magnitude`), never deeper than the survey's nominal depth (ZTF's MAGLIM; `ps1_stack.stack_depth` of the stack's
     nFrames at the position);
  4. confident (`real_checks`, object-wide): optionally a ZTF sharpness (`ztf_sharpness`) within
     `sharp_max` for point sources; for in_ztf and in_pstarr, optionally no other object of either survey within
     `isolation_arcsec` (`nearest_other_arcsec`, from the builder) nor inside a brighter object's Kron aperture (`neighbour_kron`): a
     one-sided object that close to another is most often a piece of the same light; for in_both and in_pstarr, PS1's
     own verdict on its detections: nStackDetections >= PS1_MIN_STACK_DETECTIONS and neither BAD_STACK nor
     SUSPECT_STACK (for in_ztf the PS1 side is a limit and these only say the object is faint there);
  5. population: no Gaia proper motion or parallax at 5 sigma;
  6. calibrate, once, from what survives 4 and 5 in every catalog: the limits above and one KDE envelope per band from
     the two-sided bands (`kde_envelopes`, optionally clipped of outliers first, `clip_envelope_sample`; fainter than
     ENVELOPE_MONOTONE_FROM each +-1, 2, 3 sigma edge of a row is the most extreme value of that edge in it or any
     brighter row, `widen_faint_rows`);
  7. interesting: sigma of every compared band against the envelope (`annotate_sigma`; a limit band is clipped to the
     direction its limit supports), |sigma| > SIGMA_BOUNDARY in at least two bands, at least two of them of the same
     sign; optionally (`snr_keep`, off by default) each measured side of a kept band at that S/N, the rule re-checked.
     Off, every candidate carries its per-band S/N and the website applies the threshold.

A dropped band stays dropped, and an object leaves as soon as fewer than two bands remain. Every object's record of
what happened to it is `{b}_dropped_by` and `outcome` (`annotate_outcomes`; `explain` reads them back by position).

Morphology is one decision per object: per band, the excess of PSF - Kron over PSF_KRON_CUT in units of each catalog's
error, combined by inverse variance over the two catalogs; extended when that exceeds PSF_KRON_NSIGMA in >= 2 measured
bands. dmag = ZTF - PS1 uses PSF magnitudes for point sources and Kron magnitudes for extended ones. Why not PSF for
galaxies: a PSF fit recovers only the part of a galaxy that looks like the PSF, C_psf/C_tot = 1 / (1 + (alpha_g /
alpha_psf)^2 / 2) for Gaussian galaxy and PSF widths (Slater, Ivezic & Lupton 2020, AJ, arXiv:1912.08210, sec. III.3
eq. 35), so the same galaxy is brighter in ZTF's ~2" PSF than in PS1's ~1.1" one; Kron apertures scale with the object,
not the seeing.

The optional cuts are a `Cuts` (defaults: the full pipeline; `NO_CUTS`: none of them), passed through `filter_field`,
`filter_prepared`, `assign_catalogs`, `cut_bands` and `run_branch`. Everything is a function of a pandas frame; the
only state is the `Waterfall` record each catalog returns, from which the outcomes are derived.
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
from Extracting.utils import chord, get_data_path, sky_xyz
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
GAIA_NSIGMA = 5.0               # proper motion or parallax at this significance: a Galactic star
GAIA_OBS_EPOCH = 2021.0
BAD_STACK, SUSPECT_STACK = 128, 64                        # qualityFlag
DEFECT, PEAK_ON_SPIKE, SATSTAR = 2048, 1 << 29, 128        # infoFlag
ON_SPIKE, ON_STARCORE, ON_BURNTOOL, SATSTAR_PROFILE = 8, 16, 32, 4096   # infoFlag2
PSF_QF_MIN = 0.85               # PS1 psfQfPerfect: PSF-weighted unmasked fraction (Magnier et al. 2020 'generous' threshold)
PS1_MIN_STACK_DETECTIONS = 2    # where PS1's side is a measurement (in_both, in_pstarr): PS1 detected the object in at least this many stacks
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
    bright_star_mask: bool = True             # a ZTF side inside a Gaia G < BRIGHT_STAR_G_MAX star's mask circle or spike boxes is X
    ztf_chi_max: Optional[float] = 6.0        # a ZTF detection whose DAOPhot chi exceeds this is X (ZTF recommends chi > 4 as bad for single epochs)
    ztf_sky_max: Optional[float] = 1e-7       # a ZTF detection on residual sky above this (Jy/arcsec^2, PS1's threshold) is X
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


NO_CUTS = Cuts(bright_star_mask=False, ztf_chi_max=None, ztf_sky_max=None, sharp_max=None, sharp_point_only=False, ps1_sky_max=None, ps1_min_nframes=None, isolation_arcsec=None,
               neighbour_kron=False, snr_keep=None, envelope_clip_mad=None)


# ----------------------------------------------------------------------------------------------- loading

def load_merged(field_id: int, data_dir: Optional[str] = None) -> pd.DataFrame:
    """The field's merged table (`field_tables.merge_tables`), with the ERR_FLOOR_MAG calibration floor added to every
    error and S/N, and `{b}_ps1_meas` (a finite flux with a positive error, forced included) and `{b}_ps1_snr`."""
    src = pd.read_parquet(os.path.join(field_dir(field_id, data_dir), 'merged.parquet'))
    cols = {}
    for b in BANDS:
        cols[f'{b}_ztf_psf_err'] = floor_err(src[f'{b}_ztf_psf_err'].values)
        cols[f'{b}_ztf_kron_err'] = floor_err(src[f'{b}_ztf_kron_err'].values)
        cols[f'{b}_ztf_snr'] = floor_snr(src[f'{b}_ztf_snr'].values)
        cols[f'{b}_ps1_psf_err'] = floor_err(src[f'{b}_ps1_psf_err'].values)
        cols[f'{b}_ps1_kron_err'] = floor_err(src[f'{b}_ps1_kron_err'].values)
        flux = src[f'{b}_ps1_flux'].values
        err = floor_flux_err(flux, src[f'{b}_ps1_flux_err'].values)
        cols[f'{b}_ps1_flux_err'] = err
        meas = np.isfinite(flux) & (err > 0)
        cols[f'{b}_ps1_meas'] = meas
        cols[f'{b}_ps1_snr'] = np.where(meas, flux / np.where(err > 0, err, 1.0), np.nan)
    return _attach(src, cols)


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


def ps1_nframes(src: pd.DataFrame, b: str) -> np.ndarray:
    """PS1 stack input frames at each object in band b: its own `{b}_ps1_nFrames` where it has at least one, else the
    local estimate `{b}_ps1_nframes_local`."""
    own = src[f'{b}_ps1_nFrames'].values
    return np.where(own >= 1, own, src[f'{b}_ps1_nframes_local'].values)


def ps1_depth(src: pd.DataFrame, b: str) -> np.ndarray:
    """PS1's nominal depth at each object in band b (`ps1_stack.stack_depth` of `ps1_nframes`): a cap on its limit."""
    return ps1_stack.stack_depth(ps1_nframes(src, b), b)


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

def ztf_detected(src: pd.DataFrame, b: str) -> np.ndarray:
    """ZTF's DAOPhot catalog has the object in band b: it has a ZTF magnitude there."""
    return np.isfinite(src[f'{b}_ztf_psf'].values)


def bright_star_masked(src: pd.DataFrame) -> np.ndarray:
    """Inside a Gaia G < BRIGHT_STAR_G_MAX star's mask: its circle (`bright_star_d_over_R` < 1) or one of its spike
    boxes (`in_star_spike`), as calibrated on ZTF's images (`annotate_bright_stars`)."""
    return (src.bright_star_d_over_R.values < 1) | src.in_star_spike.values


def ztf_checks(src: pd.DataFrame, b: str, point: np.ndarray, cuts: Cuts = Cuts()) -> List[Tuple[str, np.ndarray]]:
    """ZTF's side of band b, (label, passes) in order. A detection is judged where it was measured; a non-detection
    stands for the reference image's limit at the PS1 position, so it needs the image and valid pixels there (without
    a PS1 object it has nothing to be compared with, and only coverage is checked). With `cuts.bright_star_mask`,
    first the position outside every bright star's mask: ZTF's data there (detections, limits) belong to the star's
    halo or spikes. PS1 judges its own side near stars with its spike and star-core flags (`ps1_checks`). With
    `cuts.ztf_chi_max`, a detection's DAOPhot chi (PSF-fit residual RMS over the expected) at most that, for point
    sources and extended objects alike: above it the PSF fit failed (galaxy shreds, blends, bleed columns). With
    `cuts.ztf_sky_max`, a detection's residual sky (`field_tables.residual_sky`) at most that: above it the detection
    sits on large-scale excess light (a halo, a bright star's wing, scattered light), as PS1's `sky` cut judges PS1's side."""
    det = ztf_detected(src, b)
    clear = src[f'{b}_ztf_nan_dist'].values > NAN_RADIUS_PX                           # the detection's image, or the limit's
    kron_ok = src[f'{b}_ztf_kron_masked'].values <= KRON_MAX_MASKED
    mask = [('ZTF: outside bright-star masks', ~bright_star_masked(src))] if cuts.bright_star_mask else []
    return mask + [('ZTF image covers the position', det | src[f'{b}_ztf_covered'].values),
            ('ZTF: DAOPhot flags = 0', ~det | (src[f'{b}_ztf_flags'].values == 0)),
            *([(f'ZTF: DAOPhot chi <= {cuts.ztf_chi_max:g}', ~det | ~(src[f'{b}_ztf_chi'].values > cuts.ztf_chi_max))]
              if cuts.ztf_chi_max is not None else []),
            *([(f'ZTF: residual sky <= {cuts.ztf_sky_max:g}', ~det | ~(src[f'{b}_ztf_sky'].values > cuts.ztf_sky_max))]
              if cuts.ztf_sky_max is not None else []),
            (f'ZTF: > {NAN_RADIUS_PX} px from a masked pixel', clear),
            (f'extended: ZTF Kron aperture <= {100 * KRON_MAX_MASKED:.0f}% masked', ~det | point | kron_ok)]


def ps1_checks(src: pd.DataFrame, b: str, cuts: Cuts = Cuts()) -> List[Tuple[str, np.ndarray]]:
    """PS1's side of band b, (label, passes) in order; without a PS1 object only the three coverage checks apply
    (skycell, catalog, objects nearby) and the PS1 limit stands in. First the band's row not from one of MAST's mis-registered polar skycells (`annotate_skycells`;
    PS1 invalidated every magnitude there, and the detections and fluxes come from the same bad stacks, so a
    measurement from one is unusable and a missing PS1 object in one says nothing), and the position not in one of
    DR2's missing-catalog regions when there is no PS1 object (`annotate_skycells`), and, again without a PS1 object,
    the band's stack reaching the position (`{b}_ps1_covered`: a PS1 object measured in the band within
    1', `field_tables.ps1_neighbourhood`; otherwise its limit cannot stand in). With `cuts.ps1_sky_max`, then a
    residual sky at most that (a band without a sky value passes); with
    `cuts.ps1_min_nframes`, then a stack of at least that many input frames (nFrames 0, i.e. no stack input, fails)."""
    none = src.ps1_idx.values < 0
    sky_max, min_nframes = cuts.ps1_sky_max, cuts.ps1_min_nframes
    sky = ([(f'PS1: residual sky <= {sky_max:g}', none | ~(src[f'{b}_ps1_sky'].values > sky_max))]
           if sky_max is not None else [])
    frames = ([(f'PS1: stack nFrames >= {min_nframes}', none | (src[f'{b}_ps1_nFrames'].values >= min_nframes))]
              if min_nframes is not None else [])
    stack = [('PS1: not in a flagged polar skycell', ~ps1_skycell_bad(src, b)),
             ('PS1: catalog covers the position', ~src.ps1_catalog_missing.values),
             ("PS1: objects within 1' in the band", (~none) | src[f'{b}_ps1_covered'].values)]
    return (stack + sky + frames + [(label, none | ~_bits(src[f'{b}_ps1_{col}'].values, bit)) for label, col, bit in PS1_BAND_FLAGS]
            + [('PS1: measured in the band', none | src[f'{b}_ps1_meas'].values),
               (f'PS1: psfQfPerfect > {PSF_QF_MIN}', none | (src[f'{b}_ps1_psfQfPerfect'].values > PSF_QF_MIN))])


def _passes(checks: List[Tuple[str, np.ndarray]]) -> np.ndarray:
    return np.logical_and.reduce([ok for _, ok in checks])


def usable_sides(src: pd.DataFrame, b: str, point: np.ndarray, cuts: Cuts = Cuts()) -> Tuple[np.ndarray, np.ndarray]:
    """Step 1, quality: (ZTF usable, PS1 usable) for band b per object, each side passing all of its checks
    (`ztf_checks`, `ps1_checks`). A side that fails is X in `band_status`: nothing on it, measurement or limit, is
    compared. A bad skycell makes the PS1 side unusable even without a PS1 object."""
    return _passes(ztf_checks(src, b, point, cuts)), _passes(ps1_checks(src, b, cuts))


def detection_status(src: pd.DataFrame, b: str) -> Tuple[np.ndarray, np.ndarray]:
    """Step 2, detection: (ZTF, PS1) status of band b per object, what each survey has there: 'D' a measurement at
    S/N >= SNR_MIN, 'L' a measurement below it (a ZTF detection under SNR_MIN, a PS1 measurement, forced fluxes
    included), 'U' nothing measured (no ZTF detection, or no PS1 object): the survey's limit stands in."""
    z = np.where(~ztf_detected(src, b), 'U', np.where(src[f'{b}_ztf_snr'].values >= SNR_MIN, 'D', 'L'))
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
    mask = ~bright_star_masked(src) if cuts.bright_star_mask else np.ones(len(src), bool)   # ZTF's limit is unreliable there too
    isolated = ~(src.nearest_other_arcsec.values <= cuts.isolation_arcsec) if cuts.isolation_arcsec is not None else np.ones(len(src), bool)
    clear_of_kron = ~src.inside_brighter_kron.values if cuts.neighbour_kron else np.ones(len(src), bool)
    pm_ok, plx_ok = gaia_population_ok(src)
    population_ok = pm_ok & plx_ok
    position_ok = mask & isolated & clear_of_kron & population_ok & ~ps1_skycell_bad(src)   # no PS1 data to recover with, or to be recovered
    z_sharp_ok = ~sharpness_fails(src, point, cuts)
    z_trust = {b: _passes(ztf_checks(src, b, point, cuts)) & ztf_detected(src, b) & (src[f'{b}_ztf_snr'].values >= SNR_MIN) for b in BANDS}
    p_trust = {b: _passes(ps1_checks(src, b, cuts)) & src[f'{b}_ps1_detected'].values & (src[f'{b}_ps1_snr'].values >= SNR_MIN) for b in BANDS}
    n_z, n_p = sum(z_trust[b].astype(int) for b in BANDS), sum(p_trust[b].astype(int) for b in BANDS)
    stack_ok = (src.ps1_nStackDetections.values >= PS1_MIN_STACK_DETECTIONS) & ~_bits(src.ps1_qualityFlag.values, BAD_STACK | SUSPECT_STACK)
    z_ref = {b: z_trust[b] & (n_z >= 2) & position_ok & z_sharp_ok for b in BANDS}                                   # PS1's limit: ZTF's detections
    p_ref = {b: p_trust[b] & (n_p >= 2) & position_ok & stack_ok & src[f'{b}_ztf_covered'].values & (src[f'{b}_ztf_nan_dist'].values > NAN_RADIUS_PX) for b in BANDS}
    return z_ref, p_ref


def completeness_limits(src: pd.DataFrame, cuts: Cuts = Cuts()) -> pd.DataFrame:
    """`{b}_ztf_limit` and `{b}_ps1_limit`: each survey's `completeness_limit` per quadrant of the band
    (`{b}_ccd`/`{b}_qid`) against the other survey's detections that would reach the |sigma| step on their own side:
    passing the reference side's quality and confident checks (`ztf_checks` / `ps1_checks`, the `real_checks` that
    read only the reference survey, under `cuts`) at S/N >= SNR_MIN in at least two of its bands, and the Gaia
    population cuts. That is the population the one-sided candidates are drawn from. Each reference object is binned
    by the magnitude the pipeline would compare for it (`reference_magnitude`: Kron for extended objects, PSF
    otherwise), so the limit is in the same mixed magnitude system as the objects judged against it. Checks that read
    the recovering survey would remove recovered objects only and are left out. Recovery is a measurement of the band
    by the other survey at the 1" match, i.e. what the pipeline would use in place of the limit: a ZTF DAOPhot
    detection at any S/N, a PS1 detection or forced flux > 0 (a PS1 magnitude). Caps: ZTF's MAGLIM per quadrant,
    PS1's published stack depth per quadrant and `ps1_depth` per object."""
    z_refs, p_refs = completeness_references(src, cuts)
    point = point_source(src)
    cols = {}
    for b in BANDS:
        ccd, qid = src[f'{b}_ccd'].values, src[f'{b}_qid'].values                     # the image the band is judged in
        has = ccd >= 0
        key = np.where(has, ccd.astype(np.int32) * 10 + qid, 0)
        quads = sorted({(int(c), int(q)) for c, q in zip(ccd[has], qid[has])})
        z_cap = {(k // 10, k % 10): float(v) for k, v in pd.Series(src[f'{b}_ztf_maglim'].values[has]).groupby(key[has]).first().items()}
        p_cap = {q: ps1_stack.STACK_5SIG[b] for q in quads}
        for survey, ref, rec, mag, cap in (('ps1', z_refs[b], np.isfinite(src[f'{b}_ps1_psf'].values), reference_magnitude(src, b, 'ztf', point), p_cap),
                                           ('ztf', p_refs[b], ztf_detected(src, b), reference_magnitude(src, b, 'ps1', point), z_cap)):
            lut = np.full(200, np.nan)
            for (c, q), v in completeness_limit(np.where(ref, mag, np.nan), rec, key, cap).items():
                lut[c * 10 + q] = v
            lim = np.where(has, lut[key], np.nan)
            cols[f'{b}_{survey}_limit'] = (np.fmin(lim, ps1_depth(src, b)) if survey == 'ps1' else lim).astype(np.float32)
    return _attach(src, cols)


def reference_magnitude(src: pd.DataFrame, b: str, survey: str, point: np.ndarray) -> np.ndarray:
    """A survey's magnitude of each object in band b as the pipeline would compare it: Kron for an extended object
    where the survey has one, PSF otherwise."""
    kron, psf = src[f'{b}_{survey}_kron'].values, src[f'{b}_{survey}_psf'].values
    return np.where(~point & np.isfinite(kron), kron, psf)


def compared_magnitudes(src: pd.DataFrame, b: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(ZTF magnitude, its error, PS1 magnitude, its error) that band b compares, from the stored magnitudes and the
    band's flags: PSF or Kron by `{b}_use_psf`; on a limit side the survey's own sub-threshold measurement where it
    is brighter than the limit (`{b}_ztf_use_meas`, `{b}_ps1_use_forced`), else the limit with zero error."""
    psf, cat = src[f'{b}_use_psf'].values, src[f'{b}_cat'].values
    pick = lambda side, kind: np.where(psf, src[f'{b}_{side}_psf{kind}'].values, src[f'{b}_{side}_kron{kind}'].values)
    z, z_err, p, p_err = pick('ztf', ''), pick('ztf', '_err'), pick('ps1', ''), pick('ps1', '_err')
    z_lim, p_lim = src[f'{b}_ztf_limit'].values, src[f'{b}_ps1_limit'].values
    z_meas, forced = src[f'{b}_ztf_use_meas'].values, src[f'{b}_ps1_use_forced'].values
    z_out = np.where(cat == 2, np.where(z_meas, z, z_lim), z)
    z_err_out = np.where(cat == 2, np.where(z_meas, z_err, 0.0), z_err)
    p_out = np.where(cat == 1, np.where(forced, p, p_lim), p)
    p_err_out = np.where(cat == 1, np.where(forced, p_err, 0.0), p_err)
    return z_out, z_err_out, p_out, p_err_out


def assign_catalogs(src: pd.DataFrame, cuts: Cuts = Cuts()) -> pd.DataFrame:
    """`catalog` per object; per band `{b}_ztf_status`, `{b}_ps1_status`, `{b}_cat` (the comparison), `point_source`
    (one morphology per object), `{b}_extendedness`, `{b}_use_psf` (the band compares PSF magnitudes, else Kron) and
    `{b}_ztf_use_meas` / `{b}_ps1_use_forced` (a limit side uses its own sub-threshold measurement); the magnitudes a
    band compares follow from these (`compared_magnitudes`); `catalog` and `catalog_reason` from `object_catalog`. An in_both object keeps only its
    two-sided bands (`{b}_cat` 0); its one-sided bands are dropped (-1), as is every band of an object that is not
    comparable. On the undetected side of a one-sided band, either survey uses its own sub-threshold measurement
    where it has one brighter than its limit (`{b}_ps1_use_forced`, `{b}_ztf_use_meas` True; PS1's forced photometry,
    a ZTF detection under SNR_MIN), else the limit (`{b}_ps1_limit`, `{b}_ztf_limit`). A measurement, however noisy, says more
    than a limit deeper than it.

    Extended sources (`point_source`) use Kron magnitudes, point sources PSF; PSF where a side that needs one lacks a Kron.
    `run_branch` drops objects left with fewer than two compared bands."""
    src = completeness_limits(src, cuts)
    point = point_source(src)
    status = {b: band_status(src, b, point, cuts) for b in BANDS}
    raw = {b: comparison(z, p) for b, (z, p) in status.items()}
    n_measured = sum((ztf_detected(src, b) | src[f'{b}_ps1_meas'].values).astype(int) for b in BANDS)
    catalog, reason = object_catalog(raw, n_measured)
    cols = {'catalog': catalog, 'catalog_reason': reason, 'point_source': point}
    for b, (z, p) in status.items():
        cat = raw[b].copy()
        cat[(catalog == 0) & (cat > 0)] = -1                  # in_both: two-sided bands only
        cat[catalog < 0] = -1                                 # not comparable: nothing is compared
        has_kron = ((cat == 2) | np.isfinite(src[f'{b}_ztf_kron'].values)) & ((cat == 1) | np.isfinite(src[f'{b}_ps1_kron'].values))
        use_kron = (cat >= 0) & ~point & has_kron
        z_mag = np.where(use_kron, src[f'{b}_ztf_kron'].values, src[f'{b}_ztf_psf'].values)
        p_mag = np.where(use_kron, src[f'{b}_ps1_kron'].values, src[f'{b}_ps1_psf'].values)
        forced = (cat == 1) & (p == 'L') & np.isfinite(p_mag) & (p_mag < src[f'{b}_ps1_limit'].values)
        z_meas = (cat == 2) & (z == 'L') & np.isfinite(z_mag) & (z_mag < src[f'{b}_ztf_limit'].values)
        cols.update({
            f'{b}_ztf_status': z, f'{b}_ps1_status': p, f'{b}_cat': cat,
            f'{b}_extendedness': extendedness(src, b), f'{b}_use_psf': ~use_kron,
            f'{b}_ztf_use_meas': z_meas, f'{b}_ps1_use_forced': forced,
        })
    return _attach(src, cols)

def envelope_tables(src: pd.DataFrame, use: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Table]:
    """The two-sided bands per band in the old column schema, magnitudes already chosen by morphology; with `use`
    ({band: rows}), only those rows (the pipeline passes the bands that reach the |sigma| step)."""
    tabs = {}
    for b in BANDS:
        z, z_err, p, p_err = compared_magnitudes(src, b)
        rows = (src[f'{b}_cat'].values == 0) & np.isfinite(z) & np.isfinite(p) & (use[b] if use is not None else True)
        tabs[b] = Table({f'ZTF_{b}PSFMag': z[rows], f'ZTF_{b}PSFMagErr': z_err[rows], f'PSTARR_{b}PSFMag': p[rows], f'PSTARR_{b}PSFMagErr': p_err[rows]})
    return tabs


def clip_envelope_sample(src: pd.DataFrame, use: Dict[str, np.ndarray], k_mad: float = 5.0,
                         bin_mag: float = ENVELOPE_CLIP_BIN) -> Dict[str, np.ndarray]:
    """`use` ({band: rows}) restricted to its two-sided bands, less those whose dmag lies more than k_mad MAD-sigma
    (1.4826 MAD) from the median dmag of their `bin_mag` bin of mean magnitude, repeated until none do. For a Gaussian,
    5 MAD-sigma removes ~6e-7 of the distribution, so the clip removes the stack-artefact, fragment and mismatch
    outliers (10-60 MAD-sigma out) that otherwise set the envelope's +-3 sigma edges, not the population's tails."""
    out = {}
    for b in BANDS:
        z, _, p, _ = compared_magnitudes(src, b)
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
    """`{b}_dmag` and `{b}_sigma` (ZTF brighter = negative; ±SIGMA_CAP beyond what the grid resolves) for every row
    with a usable band. A limit
    bounds one direction only: against the PS1 limit (PS1 at least that faint) a band can only have brightened, so
    sigma is clipped at 0 from above, and against the ZTF limit only faded, clipped from below; a sub-threshold
    measurement standing in for a limit is a measurement and stays two-sided."""
    cols = {}
    for b in BANDS:
        z, _, p, _ = compared_magnitudes(src, b)
        dmag = z - p
        sigma, capped = envelopes[b].sigma((z + p) / 2, dmag)
        cat = src[f'{b}_cat'].values
        against_ps1_limit = (cat == 1) & ~src[f'{b}_ps1_use_forced'].values
        against_ztf_limit = (cat == 2) & ~src[f'{b}_ztf_use_meas'].values
        sigma = np.where(against_ps1_limit, np.fmin(sigma, 0.0), np.where(against_ztf_limit, np.fmax(sigma, 0.0), sigma))
        sigma[(cat < 0) | ~(p > 14.0)] = np.nan
        cols.update({f'{b}_dmag': dmag, f'{b}_sigma': sigma})
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
    """Nearest Gaia DR3 source within `max_arcsec`, positions propagated to GAIA_OBS_EPOCH: `gaia_sep_arcsec`,
    `gaia_pm`, `gaia_pmra`, `gaia_pmdec`, `gaia_pmra_err`, `gaia_pmdec_err` (mas/yr) and `gaia_parallax_over_error` (the
    Gaia cache holds the ratio, not the parallax); NaN without a match or without that quantity. The significances
    the population cut uses are computed from these (`gaia_pm_significance`)."""
    coords = SkyCoord(gaia.ra.values * u.deg, gaia.dec.values * u.deg,
                      pm_ra_cosdec=np.nan_to_num(gaia.pmra.values) * u.mas / u.yr, pm_dec=np.nan_to_num(gaia.pmdec.values) * u.mas / u.yr,
                      obstime=Time(gaia.ref_epoch.values, format='jyear')).apply_space_motion(new_obstime=Time(GAIA_OBS_EPOCH, format='jyear'))
    idx, sep, _ = match_coordinates_sky(SkyCoord(src.ra.values * u.deg, src.dec.values * u.deg), coords)
    ok = sep.arcsec <= max_arcsec
    take = lambda col: np.where(ok, gaia[col].values[idx], np.nan).astype(np.float32)
    return _attach(src, {'gaia_sep_arcsec': np.where(ok, sep.arcsec, np.nan).astype(np.float32),
                         'gaia_pm': take('pm'), 'gaia_pmra': take('pmra'), 'gaia_pmdec': take('pmdec'),
                         'gaia_pmra_err': take('pmra_error'), 'gaia_pmdec_err': take('pmdec_error'),
                         'gaia_parallax_over_error': take('parallax_over_error')})


def gaia_pm_significance(src: pd.DataFrame) -> np.ndarray:
    """Total proper motion over its error, the errors of the two components propagated in quadrature:
    sigma_pm = sqrt((pmra sigma_pmra)^2 + (pmdec sigma_pmdec)^2) / pm. NaN without a Gaia match."""
    pmra, pmdec = src.gaia_pmra.values.astype(float), src.gaia_pmdec.values.astype(float)
    err = np.hypot(pmra * src.gaia_pmra_err.values, pmdec * src.gaia_pmdec_err.values) / np.hypot(pmra, pmdec)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.hypot(pmra, pmdec) / err


def gaia_population_ok(src: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """(no proper motion at GAIA_NSIGMA, no parallax at GAIA_NSIGMA); True without a Gaia match."""
    return ~(gaia_pm_significance(src) >= GAIA_NSIGMA), ~(src.gaia_parallax_over_error.values >= GAIA_NSIGMA)


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
    """`ps1_catalog_missing`: no PS1 object at a position
    where DR2 has no catalog at all (`ps1_skycells.catalog_missing`: MAST's missing-data boxes, or south of its
    declination cut), so the absence says nothing."""
    has = src.ps1_idx.values >= 0
    return _attach(src, {'ps1_catalog_missing': ~has & ps1_skycells.catalog_missing(src.ra.values, src.dec.values)})


def ps1_skycell_bad(src: pd.DataFrame, b: Optional[str] = None) -> np.ndarray:
    """PS1's data for band b (any band if None) come from one of MAST's mis-registered polar skycells
    (`ps1_skycells.bad_cells`), where nothing PS1 measured is usable. With a PS1 object, each band is judged by the
    skycell its own row came from (`{b}_ps1_projcell`, `{b}_ps1_skycell`; an object in a skycell overlap can draw its
    bands from different skycells, and a measurement from a good one stays good). Without a PS1 object, every band is
    bad if any skycell covering the position is (`ps1_skycells.is_bad`): PS1 may have seen nothing there because its
    stack is bad."""
    has = src.ps1_idx.values >= 0
    out = np.zeros(len(src), bool)
    out[~has] = ps1_skycells.is_bad(src.ra.values[~has], src.dec.values[~has])
    codes = ps1_skycells.bad_codes()
    for band in ([b] if b else BANDS):
        code = 100 * src[f'{band}_ps1_projcell'].values.astype(np.int64) + src[f'{band}_ps1_skycell'].values.astype(np.int64)
        out[has] |= np.isin(code[has], codes)
    return out


def spike_half_length(mag: np.ndarray) -> np.ndarray:
    """Half-length in arcsec of a bright star's diffraction-spike boxes: the mask radius at G = SPIKE_ANCHOR_G, growing
    by SPIKE_GROWTH per magnitude brighter (tuned by eye on field 860's reference images, Notebooks/star_mask_860)."""
    mag = np.asarray(mag, float)
    return bright_star_radius(mag) * SPIKE_GROWTH ** (SPIKE_ANCHOR_G - mag)


def annotate_bright_stars(src: pd.DataFrame, stars: pd.DataFrame, out_to: float = 3.0) -> pd.DataFrame:
    """`bright_star_d_over_R`: distance to the nearest bright star in units of that star's mask radius
    (`bright_star_radius`); inf beyond `out_to` radii of every star. `in_star_spike`: inside one of a star's four
    diffraction-spike boxes (SPIKE_PA, SPIKE_WIDTH_ARCSEC, `spike_half_length`; ZTF is an equatorial Schmidt, so the
    spikes sit at fixed position angles on the sky); `bright_star_masked` combines them. A star within SELF_ARCSEC of the
    object is the object itself and does not count."""
    from astropy.coordinates import angular_separation, position_angle
    tree = cKDTree(sky_xyz(src.ra.values, src.dec.values))
    R = bright_star_radius(stars.phot_g_mean_mag.values); L = spike_half_length(stars.phot_g_mean_mag.values)
    best = np.full(len(src), np.inf); spike = np.zeros(len(src), bool)
    reach = chord(np.maximum(1.1 * out_to * R, 1.05 * L))
    ra_r, dec_r = np.radians(src.ra.values), np.radians(src.dec.values)
    for (ra0, dec0), r, l, hits in zip(stars[['ra', 'dec']].values, R, L,
                                       tree.query_ball_point(sky_xyz(stars.ra.values, stars.dec.values), reach)):
        if not hits:
            continue
        hits = np.asarray(hits)
        lon0, lat0 = np.radians(ra0), np.radians(dec0)                               # exact separation and position angle
        sep = np.degrees(angular_separation(lon0, lat0, ra_r[hits], dec_r[hits])) * 3600
        pa_obj = np.asarray(position_angle(lon0, lat0, ra_r[hits], dec_r[hits]), float)
        dx, dy = sep * np.sin(pa_obj), sep * np.cos(pa_obj)                          # east, north (arcsec)
        own = sep < SELF_ARCSEC
        d = np.where(own, np.inf, sep / r)
        np.minimum.at(best, hits, np.where(d <= out_to, d, np.inf))
        for pa in SPIKE_PA:
            th = np.radians(pa)
            along = dx * np.sin(th) + dy * np.cos(th); across = dx * np.cos(th) - dy * np.sin(th)
            spike[hits[~own & (np.abs(along) <= l) & (np.abs(across) <= SPIKE_WIDTH_ARCSEC / 2)]] = True
    return _attach(src, {'bright_star_d_over_R': best, 'in_star_spike': spike})


# ----------------------------------------------------------------------------------------------- branches

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
    """One branch's record: the per-band keep-masks after each step (what `annotate_outcomes` reads), counted as objects
    still in and bands still in. The tables and flowcharts are `quality_table`, `catalog_table` and their flowcharts."""
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
        self.counts.append({'objects': int(np.count_nonzero(np.logical_or.reduce(list(m.values())))),
                            **{b: int(np.count_nonzero(m[b])) for b in BANDS}})

    def mask(self, i: int) -> Dict[str, np.ndarray]:
        """The per-band keep-masks after step i (an index into `steps`)."""
        return {b: np.unpackbits(p, count=self.n_rows).astype(bool) for b, p in self.packed[i].items()}

    @property
    def masks(self) -> List[Dict[str, np.ndarray]]:
        return [self.mask(i) for i in range(len(self.steps))]

    @property
    def stats(self) -> pd.DataFrame:
        """Objects left after each step, and the bands left in each."""
        return pd.DataFrame([{'group': g, 'step': s, **c} for g, s, c in zip(self.groups, self.steps, self.counts)],
                            columns=['group', 'step', 'objects', *BANDS])

    def dropped_by(self, label: str) -> Dict[str, np.ndarray]:
        i = self.steps.index(label)
        before, after = self.mask(i - 1), self.mask(i)
        return {b: before[b] & ~after[b] for b in BANDS}


def ztf_sharpness(src: pd.DataFrame) -> np.ndarray:
    """One DAOPhot sharpness per object: the inverse-variance mean over its ZTF detections, weighting each band by
    S/N^2 (DAOPhot gives no sharpness error; it scales as 1/S/N). NaN without a ZTF detection."""
    num = np.zeros(len(src)); den = np.zeros(len(src))
    for b in BANDS:
        sharp, snr = src[f'{b}_ztf_sharp'].values, src[f'{b}_ztf_snr'].values
        ok = ztf_detected(src, b) & np.isfinite(sharp) & (snr > 0)
        num[ok] += sharp[ok] * snr[ok] ** 2; den[ok] += snr[ok] ** 2
    return np.divide(num, den, out=np.full(len(src), np.nan), where=den > 0)


def sharpness_fails(src: pd.DataFrame, point: np.ndarray, cuts: Cuts = Cuts()) -> np.ndarray:
    """Objects whose S/N^2-weighted |sharp| exceeds `cuts.sharp_max`; with `cuts.sharp_point_only`, point sources only."""
    if cuts.sharp_max is None:
        return np.zeros(len(src), bool)
    fails = np.abs(ztf_sharpness(src)) > cuts.sharp_max
    return fails & point if cuts.sharp_point_only else fails


def real_checks(src: pd.DataFrame, catalog: int, cuts: Cuts = Cuts()) -> List[Tuple[str, np.ndarray]]:
    """Object-wide tests that a `catalog` object is real, (label, passes) in order. Where the PS1 side is a measurement
    (in_both, in_pstarr) PS1's own verdict on its detections counts: nStackDetections >= PS1_MIN_STACK_DETECTIONS (our
    S/N can call a forced measurement D where PS1's detection pass found nothing: a blend or an underestimated error)
    and neither BAD_STACK nor SUSPECT_STACK (qualityFlag: at most one good measurement). For in_ztf the PS1 side is a
    limit and all three only say the object is faint there."""
    checks = []
    sharp_max, isolation_arcsec = cuts.sharp_max, cuts.isolation_arcsec
    if isolation_arcsec is not None and catalog in (1, 2):                 # neighbour cuts before sharpness: what they
        checks.append((f'no other object within {isolation_arcsec:g}"',    # catch (fragments, pieces of other objects)
                       ~(src.nearest_other_arcsec.values <= isolation_arcsec)))   # is attributed to them in the waterfalls
    if cuts.neighbour_kron and catalog in (1, 2):
        checks.append(("outside every brighter object's Kron aperture", ~src.inside_brighter_kron.values))
    if sharp_max is not None:
        label = f'ZTF: S/N²-weighted |sharp| <= {sharp_max}' + (' (point sources)' if cuts.sharp_point_only else '')
        checks.append((label, ~sharpness_fails(src, src.point_source.values, cuts)))
    if catalog in (0, 2):                                                  # PS1's own verdict on its detections of the object,
        checks.append((f'PS1: nStackDetections >= {PS1_MIN_STACK_DETECTIONS}',   # wherever its side is a measurement
                       src.ps1_nStackDetections.values >= PS1_MIN_STACK_DETECTIONS))
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
    start = per_band('structure', lambda b: [('measured by ZTF or PS1', ztf_detected(src, b) | src[f'{b}_ps1_meas'].values)])
    quality = per_band('quality', lambda b: ztf_checks(src, b, point, cuts) + ps1_checks(src, b, cuts))
    comparable = per_band('comparison', lambda b: [(COMPARABLE, src[f'{b}_cat'].values >= 0)])
    return start + quality + comparable


def quality_waterfall(src: pd.DataFrame, cuts: Cuts = Cuts()) -> Waterfall:
    """The shared stage as a waterfall over every object: measured, the quality steps, then comparable (the objects
    the catalogs start from). A per-band cut removes the band for every later step, and an object leaves as soon as
    fewer than two bands remain."""
    wf = Waterfall('all')
    m = {b: np.ones(len(src), bool) for b in BANDS}
    for i, (group, label, passes) in enumerate(quality_steps(src, src.point_source.values, cuts)):
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
    for label, ok in population_checks(src):
        keep &= ok
        wf.add(label, {b: keep & m[b] for b in BANDS}, 'population')
    return wf, {b: keep & m[b] for b in BANDS}


def population_checks(src: pd.DataFrame) -> List[Tuple[str, np.ndarray]]:
    """Object-wide Gaia checks, (label, passes) in order (`gaia_population_ok`)."""
    pm_ok, plx_ok = gaia_population_ok(src)
    return [(f'Gaia: no proper motion at {GAIA_NSIGMA:.0f}σ', pm_ok), (f'Gaia: no parallax at {GAIA_NSIGMA:.0f}σ', plx_ok)]


SIGMA_STEP = f'|σ| > {SIGMA_BOUNDARY:.0f} in >= 2 bands'
SIGN_STEP = '... >= 2 of them the same sign'


def interesting_checks(src: pd.DataFrame) -> List[Tuple[str, np.ndarray]]:
    """The |sigma| checks over each object's compared bands (`{b}_cat` >= 0), (label, passes) in order; what
    `judge_bands` applies to the bands that reach it."""
    sig = {b: np.where(src[f'{b}_cat'].values >= 0, src[f'{b}_sigma'].values, np.nan) for b in BANDS}
    n_big = sum((np.abs(sig[b]) > SIGMA_BOUNDARY).astype(int) for b in BANDS)
    pos = sum((sig[b] > SIGMA_BOUNDARY).astype(int) for b in BANDS)
    neg = sum((sig[b] < -SIGMA_BOUNDARY).astype(int) for b in BANDS)
    return [(SIGMA_STEP, n_big >= 2), (SIGN_STEP, np.maximum(pos, neg) >= 2)]


def judge_bands(src: pd.DataFrame, wf: Waterfall, reach: Dict[str, np.ndarray], cuts: Cuts = Cuts()) -> pd.DataFrame:
    """The |sigma| steps on the bands that reach them (`cut_bands`), then the `snr_keep` step on what passed, added to
    `wf`. Returns the candidates."""
    def sigma_steps(bands, keep, group, suffix=''):
        sig = {b: np.where(bands[b], src[f'{b}_sigma'].values, np.nan) for b in BANDS}
        big = {b: np.abs(sig[b]) > SIGMA_BOUNDARY for b in BANDS}
        keep = keep & (sum(big[b].astype(int) for b in BANDS) >= 2)
        wf.add(SIGMA_STEP + suffix, {b: keep & big[b] for b in BANDS}, group)
        pos = sum((sig[b] > SIGMA_BOUNDARY).astype(int) for b in BANDS)
        neg = sum((sig[b] < -SIGMA_BOUNDARY).astype(int) for b in BANDS)
        keep = keep & (np.maximum(pos, neg) >= 2)
        wf.add(SIGN_STEP + suffix, {b: keep & big[b] for b in BANDS}, group)
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

# ----------------------------------------------------------------------------------------------- summaries

SURVEY_CHECKS = {'ztf': lambda src, b, cuts: ztf_checks(src, b, src.point_source.values, cuts),
                 'ps1': lambda src, b, cuts: ps1_checks(src, b, cuts)}
SURVEY_NAME = {'ztf': 'ZTF', 'ps1': 'PS1'}


def check_table(checks: List[Tuple[str, np.ndarray]], start: np.ndarray) -> pd.DataFrame:
    """Per check over the `start` rows, in order: `after` (passing it and every earlier check), `total fail` (failing
    it), `unique` (failing it and no other check), each with its % of the start."""
    n = max(int(start.sum()), 1)
    fails = [start & ~np.asarray(ok, bool) for _, ok in checks]
    n_fail = sum(f.astype(np.int8) for f in fails) if checks else None
    rows, alive = [], start.copy()
    for (label, ok), f in zip(checks, fails):
        alive &= ok
        tot, uniq = int(f.sum()), int((f & (n_fail == 1)).sum())
        rows.append({'check': label, 'after': int(alive.sum()), 'total fail': tot, 'total fail %': 100 * tot / n,
                     'unique': uniq, 'unique %': 100 * uniq / n})
    return pd.DataFrame(rows, columns=['check', 'after', 'total fail', 'total fail %', 'unique', 'unique %']).set_index('check')


def quality_table(src: pd.DataFrame, survey: str, cuts: Cuts = Cuts()) -> pd.DataFrame:
    """`check_table` of one survey's quality checks (`ztf_checks` or `ps1_checks`) per band over every object, each band
    on its own; indexed by (band, check)."""
    start = np.ones(len(src), bool)
    return pd.concat({b: check_table(SURVEY_CHECKS[survey](src, b, cuts), start) for b in BANDS}, names=['band', 'check'])


def catalog_checks(src: pd.DataFrame, catalog: int, cuts: Cuts = Cuts()) -> List[Tuple[str, np.ndarray]]:
    """A catalog's object-wide checks after the sort, in order: `real_checks`, `population_checks`, then
    `interesting_checks` once `{b}_sigma` is annotated."""
    sigma = interesting_checks(src) if all(f'{b}_sigma' in src for b in BANDS) else []
    return real_checks(src, catalog, cuts) + population_checks(src) + sigma


def catalog_table(src: pd.DataFrame, cuts: Cuts = Cuts()) -> pd.DataFrame:
    """`check_table` of `catalog_checks` per catalog over the objects sorted into it; indexed by (catalog, check). The
    optional `snr_keep` step is not included."""
    return pd.concat({name: check_table(catalog_checks(src, k, cuts), src.catalog.values == k) for k, name in CATALOG.items()},
                     names=['catalog', 'check'])


def sort_table(src: pd.DataFrame) -> pd.DataFrame:
    """Objects by the bands with a comparison of any kind (`comparison` of the band's statuses, before an in_both object
    drops its one-sided bands) against the catalog they are sorted into."""
    combo = np.full(len(src), '', dtype=object)
    for b in BANDS:
        c = comparison(src[f'{b}_ztf_status'].values.astype(str), src[f'{b}_ps1_status'].values.astype(str))
        combo = combo + np.where(c >= 0, b, '')
    combo = np.where(combo == '', 'none', combo)
    cat = pd.Series(src.catalog.values).map({**CATALOG, -1: NOT_COMPARABLE})
    t = pd.crosstab(pd.Series(combo, name='compared bands'), cat.rename('catalog'))
    order = ['gri', 'gr', 'gi', 'ri', 'g', 'r', 'i', 'none']
    t = t.reindex(index=[o for o in order if o in t.index], columns=[*CATALOG.values(), NOT_COMPARABLE], fill_value=0)
    t['total'] = t.sum(axis=1)
    return pd.concat([t, t.sum().rename('total').to_frame().T])


def significant_bands_table(candidates: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Candidates by their compared bands with |sigma| > SIGMA_BOUNDARY, per catalog."""
    cols = {}
    for name, cand in candidates.items():
        combo = np.full(len(cand), '', dtype=object)
        for b in BANDS:
            big = (cand[f'{b}_cat'].values >= 0) & (np.abs(cand[f'{b}_sigma'].values) > SIGMA_BOUNDARY)
            combo = combo + np.where(big, b, '')
        cols[name] = pd.Series(combo, dtype=object).value_counts()
    t = pd.DataFrame(cols).fillna(0).astype(int)
    first = [o for o in ('gri', 'gr', 'gi', 'ri') if o in t.index]
    return t.reindex(first + [o for o in t.index if o not in first])


def _fraction(v: float) -> str:
    return f'{v:.3f}' if v >= 0.001 or v == 0 else f'{v:.1e}'


def _rows_text(rows: List[Tuple[str, Optional[int], Optional[float]]]) -> str:
    """Box text: a header, then one line per (name, pass, fraction of the start); None prints a dash."""
    w = max(len(r[0]) for r in rows)
    head = f"{'':<{w}}  {'Pass':>11}  {'Fraction':>8}"
    lines = [f"{name:<{w}}  {('—' if n is None else f'{n:,}'):>11}  {('—' if f is None else _fraction(f)):>8}" for name, n, f in rows]
    return '\n'.join([head] + lines)


def _flowchart(start: str, steps: List[Tuple[str, str]], end: Optional[Tuple[str, str]] = None):
    """A vertical chain of boxes: `start`, then each (arrow label, box text), then the optional `end` the same way;
    monospace box text so the columns line up."""
    import schemdraw
    from schemdraw import flow
    from schemdraw.elements import Line
    font, fontsize = 'monospace', 14                           # schemdraw's default size; a monospace glyph is ~0.6 em
    size = lambda t, pad: (max(len(l) for l in t.split('\n')) * 0.6 * fontsize / 72 / 0.5 + 2 * pad, 0.45 * (t.count('\n') + 1) + 0.4)
    chain = steps + ([end] if end else [])
    label_w = max((_box_width_for_text(l, padding=0.0) for l, _ in chain), default=0.0)
    with schemdraw.Drawing(show=False) as d:
        w, h = size(start, 1.5)
        prev = d.add(flow.Start(w=w, h=h).label(start, font=font))
        for label, text in chain:
            d += flow.Arrow().down(d.unit / 2).at(prev.S).label(label)
            w, h = size(text, 1.0)
            prev = d.add(flow.Box(w=w, h=h).anchor('N').label(text, font=font))
        d += Line().at((-label_w - 4.0, 0)).to((-label_w - 3.99, 0)).color('none')   # room for the arrow labels
    return d


def quality_flowchart(src: pd.DataFrame, survey: str, cuts: Cuts = Cuts(), table: Optional[pd.DataFrame] = None):
    """One survey's quality checks as a flowchart: every object, then each check with g, r, i on their own (Pass:
    passing it and every earlier check; Fraction: of every object), then the usable sides (Total) split into Detected
    (D, S/N >= SNR_MIN) and Undetected (L or U)."""
    table = quality_table(src, survey, cuts) if table is None else table
    n = len(src)
    checks = list(table.loc[BANDS[0]].index)
    steps = [(c, _rows_text([(b, int(table.loc[(b, c), 'after']), table.loc[(b, c), 'after'] / n) for b in BANDS])) for c in checks]
    w = 10
    det = [f"{'':<2}  " + '  '.join(f'{h:>{w}}' for h in ('Total', 'Detected', 'Undetected'))]
    for b in BANDS:
        st = pd.Series(src[f'{b}_{survey}_status'].values.astype(str)).value_counts()
        counts = (n - st.get('X', 0), st.get('D', 0), st.get('L', 0) + st.get('U', 0))
        det.append(f'{b:<2}  ' + '  '.join(f'{int(c):>{w},}' for c in counts))
    return _flowchart(f'{SURVEY_NAME[survey]} quality\nobjects {n:,}', steps, (f'SNR >= {SNR_MIN:g}', '\n'.join(det)))


def catalog_flowchart(src: pd.DataFrame, cuts: Cuts = Cuts(), table: Optional[pd.DataFrame] = None):
    """The catalogs' object-wide checks after the sort as one flowchart: the objects sorted into each catalog, then
    each check with in_both, in_ztf, in_pstarr (Pass: passing it and every earlier check; Fraction: of the
    objects sorted into the catalog; a dash where the catalog has no such check)."""
    table = catalog_table(src, cuts) if table is None else table
    names = list(CATALOG.values())
    start = {name: int((src.catalog.values == k).sum()) for k, name in CATALOG.items()}
    order: List[str] = []                                   # every catalog's checks, in chain order
    for name in names:
        own = list(table.loc[name].index)
        for i, c in enumerate(own):
            if c not in order:
                prev = [x for x in own[:i] if x in order]
                order.insert(order.index(prev[-1]) + 1 if prev else 0, c)
    def row(name, c):
        if c not in table.loc[name].index:
            return (name, None, None)
        after = int(table.loc[(name, c), 'after'])
        return (name, after, after / max(start[name], 1))
    steps = [(c, _rows_text([row(name, c) for name in names])) for c in order]
    return _flowchart('sorted into catalogs\n' + _rows_text([(name, start[name], 1.0) for name in names]), steps)


# ----------------------------------------------------------------------------------------------- driver

def annotate_neighbours(src: pd.DataFrame) -> pd.DataFrame:
    """`inside_brighter_kron`: inside the Kron aperture of a brighter object (PS1's KronRad, or our ZTF aperture) of
    either survey -- every ZTF object at its position and every PS1 object at its own position, matched or not,
    whatever it does in the pipeline; the object's own entries never count. The light there belongs to that object (a
    galaxy's outskirts). The distance to the nearest other object, `nearest_other_arcsec`, comes from the builder
    (`field_tables.nearest_other`). An object's magnitude is its brightest PSF magnitude in any band of either survey;
    a Kron radius is the largest finite one over the bands. Distances are exact angles (`sky_xyz` trees)."""
    pos = sky_xyz(src.ra.values, src.dec.values)
    ztf_rows, ps1_rows = np.flatnonzero(src.ztf_idx.values >= 0), np.flatnonzero(src.ps1_idx.values >= 0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)                                   # all-NaN rows
        z_mag = np.nanmin(np.c_[[src[f'{b}_ztf_psf'].values for b in BANDS]].T, axis=1)
        p_mag = np.nanmin(np.c_[[src[f'{b}_ps1_psf'].values for b in BANDS]].T, axis=1)
        z_rad = np.nanmax(np.c_[[src[f'{b}_ztf_kron_rap'].values for b in BANDS]].T, axis=1)
        p_rad = np.nanmax(np.c_[[src[f'{b}_ps1_kron_rad'].values for b in BANDS]].T, axis=1)
        own_mag = np.nanmin(np.c_[z_mag, p_mag], axis=1)
    owner = np.r_[ztf_rows, ps1_rows]                                                   # the object each entry belongs to
    entries = np.r_[pos[ztf_rows], sky_xyz(src.ps1_ra.values[ps1_rows], src.ps1_dec.values[ps1_rows])]
    e_mag, e_rad = np.r_[z_mag[ztf_rows], p_mag[ps1_rows]], np.r_[z_rad[ztf_rows], p_rad[ps1_rows]]
    # inside a brighter entry's Kron aperture: every object within each aperture's radius (searched out to
    # KRON_REACH_ARCSEC: a larger PS1 KronRad is a deblending accident), as one flat pair list
    in_kron = np.zeros(len(src), bool)
    with_rad = np.flatnonzero(np.isfinite(e_rad))
    hits = cKDTree(pos).query_ball_point(entries[with_rad], chord(np.minimum(e_rad[with_rad], KRON_REACH_ARCSEC)), return_sorted=False)
    n_hits = np.fromiter((len(h) for h in hits), int, len(hits))
    e = np.repeat(with_rad, n_hits)                                                     # the entry of each pair
    i = np.fromiter((j for h in hits for j in h), int, int(n_hits.sum()))               # the object inside it
    inside = (owner[e] != i) & (e_mag[e] < own_mag[i])
    in_kron[i[inside]] = True
    return _attach(src, {'inside_brighter_kron': in_kron})


def field_box(src: pd.DataFrame, pad_deg: float = 0.4) -> Tuple[float, float, float, float]:
    """(ra0, ra1, dec0, dec1) around the objects, padded for the brightest stars' masks (3 R ~ 0.35 deg at G = 4); a
    field across RA = 0 gives ra0 < 0."""
    ra, dec = src.ra.values, src.dec.values
    if ra.max() - ra.min() > 180:                           # across RA = 0: shift the western half below zero
        ra = np.where(ra > 180, ra - 360, ra)
    return float(ra.min() - pad_deg), float(ra.max() + pad_deg), float(dec.min() - pad_deg), float(dec.max() + pad_deg)


def prepare_field(field_id: int, data_dir: Optional[str] = None) -> pd.DataFrame:
    """The parts that do not depend on the photometry choices: the merged table (`load_merged`) with its Gaia match, its PS1 skycell
    (and whether that is one of MAST's bad polar skycells), its distance to the nearest bright star, and whether it
    sits inside a brighter object's Kron aperture. `nearest_other_arcsec` comes with the merged table."""
    data_dir = data_dir or get_data_path()
    base = load_merged(field_id, data_dir)
    stars = bright_stars(field_id, data_dir, box=field_box(base))
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
    labels += waterfalls['all'].steps
    walk(waterfalls['all'], np.ones(n, bool), 0)                       # per-band drops only: objects leave the shared stage as 'not comparable'
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
    reason = np.char.add('not comparable: ', src.catalog_reason.astype(str).values.astype(str))
    dropped = np.char.add('dropped: ', label_of(died_branch).astype(str))
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
        for old in [f'{name}_stats.csv' for name in ['all', *CATALOG.values()]]:   # the previous layout's tables
            if os.path.exists(os.path.join(out, old)):
                os.remove(os.path.join(out, old))
        for survey in SURVEY_CHECKS:
            quality_table(src, survey, cuts).to_csv(os.path.join(out, f'quality_{survey}.csv'))
        sort_table(src).to_csv(os.path.join(out, 'sort.csv'))
        catalog_table(src, cuts).to_csv(os.path.join(out, 'catalogs.csv'))
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


EXPLAIN_COLS = (['ra', 'dec', 'ztf_idx', 'ps1_idx', 'catalog', 'catalog_reason', 'outcome', 'point_source']
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
          f'{"point" if r.point_source else "extended"}  catalog: {cat} ({r.catalog_reason})')
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
