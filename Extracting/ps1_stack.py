"""The PS1 DR2 stack catalog, from our mirror of MAST's HATS `stack_object` table, reduced to one row per object.

The mirror (panstarrs/stack_object on the lab share; see its README) keeps every stack measurement: an object in a
skycell overlap, or measured on several stack images, has one row per measurement, and a band can be forced
(`{b}infoFlag2 & 4`) in one row and detected in another. Each partition is cleaned on its own -- all rows of an objID
share its position and so its partition -- in three steps:

  1. drop repeated `uniquePspsSTid` (one stack row MAST joined to two ObjectThin records with the same objID);
  2. object-level columns from the object's best row: a row from the object's primary skycell first, then the most
     detected bands, then the highest S/N in any band;
  3. each band from its best row: the row PS1 itself used for the object's stack photometry in that band first, then a
     detection before a forced or missing measurement, then a primary-skycell row, then the highest S/N.

"Primary" and "used for the stack photometry" come from PS1's per-band `{b}infoFlag3` bits STACK_PRIMARY (0x00010000,
"This stack measurement is in the primary skycell") and STACK_PHOT_SRC (0x00020000, "This measurement supplied the
stack photometry"), as defined in the IPP DR2 flag table, Detection Flags 3
(https://svn.pan-starrs.ifa.hawaii.edu/trac/ipp/wiki/DR2Flags). The `primaryDetection` column is not used:
Flewelling et al. 2020 (ApJS 251, 7, "The Pan-STARRS1 Database and Data Products", arXiv:1612.05243, in the
StackObjectThin table description and its schema) state that "the fields StackObjectThin.primaryDetection and
StackObjectThin.bestDetection are incorrectly set for DR1 and DR2. A coding error resulted in these bits being set
based on the wrong input fields." Selecting stack rows by these infoFlag3 bits instead follows e.g. arXiv:2510.15836
(both bits required) and Cañameras et al. 2020 (HOLISMOKES II, arXiv:2004.13048). In a field-860 partition every object has a
STACK_PHOT_SRC row in each band, and the previous primaryDetection-based choice differed from it for 0.3-0.5% of
objects per band.

PS1's -999 (no measurement) becomes NaN. Forced measurements are kept, with `{b}_detected` False. An object's
position (`ra`, `dec`) is its stack position, which matches ZTF better than the mean position fainter than r ~ 20.5
(the mean rests on a few noisy single-epoch detections there); the mean position stands in where there is no stack
position, and both are kept as columns.
"""
import glob
import os
from functools import lru_cache
from typing import List, Optional, Sequence

import hpgeom
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .utils import get_data_path

BANDS = ('g', 'r', 'i')
OBJECT_COLS = ['objID', 'raMean', 'decMean', 'raMeanErr', 'decMeanErr', 'raStack', 'decStack', 'raStackErr',
               'decStackErr', 'epochMean', 'qualityFlag', 'objInfoFlag', 'primaryDetection', 'nStackObjectRows',
               'nStackDetections', 'nDetections', 'ng', 'nr', 'ni', 'projectionID', 'skyCellID']
BAND_COLS = ['PSFMag', 'PSFMagErr', 'PSFFlux', 'PSFFluxErr', 'KronMag', 'KronMagErr', 'KronFlux', 'KronFluxErr',
             'KronRad', 'infoFlag', 'infoFlag2', 'infoFlag3', 'psfQfPerfect', 'ExtNSigma', 'nFrames', 'psfChiSq', 'sky',
             'Epoch']
FORCED = 4                              # infoFlag2 bit: measured at a position detected in other bands
STACK_PRIMARY = 0x00010000              # infoFlag3 bit: this stack measurement is in the primary skycell
STACK_PHOT_SRC = 0x00020000             # infoFlag3 bit: this measurement supplied the stack photometry
MISSING = -999.0


# PS1 3pi depths, from Chambers et al. 2016, "The Pan-STARRS1 Surveys", arXiv:1612.05560 (v4):
#   - Table 11 ("3pi Steradian Survey Characteristics"): "5sigma single epoch" g 22.0, r 21.8, i 21.5 and "5sigma stack
#     limit" g 23.3, r 23.2, i 23.1 mag; the abstract calls the latter "the mean 5sigma point source limiting
#     sensitivities in the stacked 3pi Steradian Survey". A sky mean, kept as a reference line for the review notebooks;
#     the pipeline's limit is each stack's own injection-recovery (below).
STACK_5SIG = {'g': 23.3, 'r': 23.2, 'i': 23.1}


# ----------------------------------------------------------------------------------------------- injection recovery

# PS1's own injection-recovery test of every deep stack (Flewelling et al. 2020, ApJS 251, 7, StackDetEffMeta): nInjected
# fake PSF sources per bin at 13 offsets from magref (= StackMeta.detectionThreshold, about the 50% point), countsNN of
# them recovered by psphot's detection in that stack. One row per skycell and filter.
DETEFF_OFFSETS = [f'offset{i:02d}' for i in range(1, 14)]
DETEFF_COUNTS = [f'counts{i:02d}' for i in range(1, 14)]
DETEFF_QUERY = ("select m.stackImageID, m.projectionID, m.skyCellID, m.filterID, m.crval1, m.crval2, m.nP2Images, m.detectionThreshold, "
                "e.magref, e.nInjected, " + ', '.join(f'e.{o}, e.{c}' for o, c in zip(DETEFF_OFFSETS, DETEFF_COUNTS)) +
                " from StackMeta m join StackDetEffMeta e on e.stackImageID = m.stackImageID "
                "where m.stackTypeID = 1 and m.filterID <= 3 and m.projectionID >= {lo} and m.projectionID < {hi}")
DETEFF_PROJCELLS = (635, 2644)         # projection cells of the 3pi deep stacks, [lo, hi)
DETEFF_CHUNK = 100                     # projection cells per quick query (~30k rows, ~4 s; 300 exceeds the quick-job memory limit)
FILTER_BAND = {1: 'g', 2: 'r', 3: 'i'}


def deteff_path(data_dir: Optional[str] = None) -> str:
    return os.path.join(data_dir or get_data_path(), 'ps1', 'stack_deteff_gri.parquet')


def fetch_stack_deteff(path: Optional[str] = None, retries: int = 5) -> pd.DataFrame:
    """Download every 3pi deep gri stack's injection-recovery row (DETEFF_QUERY, ~600k rows) from MAST CasJobs to
    `deteff_path`, as quick queries over DETEFF_CHUNK projection cells at a time (the long-job queue can sit for
    hours; MAST's connection drops now and then, hence `retries` per chunk). Returns the table as `stack_deteff`
    reads it."""
    import time
    from Source_Analysis.casjobs import MASTCASJOBS as J
    path = path or deteff_path()
    parts = []
    for lo in range(DETEFF_PROJCELLS[0], DETEFF_PROJCELLS[1], DETEFF_CHUNK):
        for attempt in range(retries):
            try:
                parts.append(J.quick(DETEFF_QUERY.format(lo=lo, hi=lo + DETEFF_CHUNK), task_name='stack_deteff').to_pandas()); break
            except Exception as e:                         # a dropped connection or a busy server
                if attempt == retries - 1:
                    raise
                time.sleep(30)
    df = pd.concat(parts, ignore_index=True)
    df['band'] = df.filterID.astype(int).map(FILTER_BAND)
    for c in ('projectionID', 'skyCellID', 'filterID', 'nP2Images', 'nInjected'):
        df[c] = df[c].astype(int)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path + '.part'); os.replace(path + '.part', path)
    stack_deteff.cache_clear()
    return df


@lru_cache(maxsize=None)
def stack_deteff(path: Optional[str] = None) -> pd.DataFrame:
    """The downloaded injection-recovery table (`fetch_stack_deteff`), one row per deep stack."""
    return pd.read_parquet(path or deteff_path())


def recovery_mag(table: pd.DataFrame, completeness: float) -> np.ndarray:
    """Each stack's magnitude at recovered fraction `completeness`: magref plus the offset interpolated linearly
    between the first bin whose fraction falls below it and the bin before. NaN where the fraction is below it already
    in the brightest bin, or never falls below it."""
    o = table[DETEFF_OFFSETS].to_numpy(float)
    f = table[DETEFF_COUNTS].to_numpy(float) / table.nInjected.to_numpy(float)[:, None]
    below = f < completeness
    i = below.argmax(axis=1)
    ok = below.any(axis=1) & (i > 0)
    rows, i0, i1 = np.arange(len(table)), np.clip(i - 1, 0, None), i
    with np.errstate(divide='ignore', invalid='ignore'):
        off = o[rows, i0] + (f[rows, i0] - completeness) / (f[rows, i0] - f[rows, i1]) * (o[rows, i1] - o[rows, i0])
    return np.where(ok, table.magref.to_numpy(float) + off, np.nan)


@lru_cache(maxsize=None)
def _recovery_lookup(completeness: float, path: Optional[str]) -> pd.Series:
    t = stack_deteff(path)
    code = t.filterID.to_numpy(np.int64) * 10_000_000 + t.projectionID.to_numpy(np.int64) * 100 + t.skyCellID.to_numpy(np.int64)
    return pd.Series(recovery_mag(t, completeness), index=code).groupby(level=0).first()    # a stack rebuilt twice: one row


def stack_recovery_mag(b: str, projcell: np.ndarray, skycell: np.ndarray, completeness: float, path: Optional[str] = None) -> np.ndarray:
    """The band-b deep stack's `recovery_mag` at `completeness` for each (projection cell, skycell); NaN where no
    stack has an injection row, or the cell is -1."""
    filt = {v: k for k, v in FILTER_BAND.items()}[b]
    proj, cell = np.asarray(projcell, np.int64), np.asarray(skycell, np.int64)
    out = _recovery_lookup(completeness, path).reindex(filt * 10_000_000 + proj * 100 + cell).to_numpy(float)
    return np.where((proj >= 0) & (cell >= 0), out, np.nan)


def stack_dir(path: Optional[str] = None) -> str:
    """The mirror's dataset/ directory: `path`, else $PS1_STACK_DIR, else <data>/ps1_stack/dataset."""
    return path or os.environ.get('PS1_STACK_DIR') or os.path.join(get_data_path(), 'ps1_stack', 'dataset')


def partition_files(ra: np.ndarray, dec: np.ndarray, directory: Optional[str] = None) -> List[str]:
    """The mirror's partition files containing any of these positions."""
    directory = stack_dir(directory)
    files = glob.glob(os.path.join(directory, 'Norder=*', 'Dir=*', 'Npix=*.parquet'))
    have = {(int(f.split('Norder=')[1].split('/')[0]), int(f.split('Npix=')[1].split('.')[0])): f for f in files}
    wanted = set()
    for order in {o for o, _ in have}:
        wanted |= {(order, int(p)) for p in np.unique(hpgeom.angle_to_pixel(2 ** order, ra, dec, nest=True))}
    return sorted(have[k] for k in wanted & set(have))


def _missing_to_nan(rows: pd.DataFrame) -> pd.DataFrame:
    floats = [c for c in rows.columns if rows[c].dtype.kind == 'f']
    rows[floats] = rows[floats].mask(rows[floats] == MISSING)
    return rows


def _best(rows: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """One row per objID: the first after sorting on `keys` (ascending), ties broken by uniquePspsSTid."""
    return rows.sort_values(['objID', *keys, 'uniquePspsSTid'], kind='stable').drop_duplicates('objID').set_index('objID')


def clean(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per objID from a partition's stack measurements (see the module docstring for the rules).

    Adds `n_rows` and `n_primary_rows` (rows with STACK_PRIMARY in any band) per object and, per band, `{b}_detected`,
    `{b}_row` (the uniquePspsSTid the band came from) and `{b}_projcell` / `{b}_skycell` (that row's skycell: the
    bands of an object in a skycell overlap can come from different skycells). Objects keep PS1's names, with objID as `PanSTARR_ID`, and `ra`, `dec` are the stack position
    (the mean position where there is none)."""
    rows = _missing_to_nan(rows.drop_duplicates('uniquePspsSTid').reset_index(drop=True))
    for b in BANDS:
        rows[f'{b}_detected'] = ((rows[f'{b}infoFlag2'].fillna(0).astype('int64') & FORCED) == 0) \
                                & (rows[f'{b}PSFFluxErr'] > 0) & rows[f'{b}PSFFlux'].notna()
        rows[f'{b}_snr'] = (rows[f'{b}PSFFlux'] / rows[f'{b}PSFFluxErr']).where(rows[f'{b}_detected'])
    flag3 = {b: rows[f'{b}infoFlag3'].fillna(0).astype('int64') for b in BANDS}
    rows['_not_primary'] = ~np.logical_or.reduce([(flag3[b] & STACK_PRIMARY) != 0 for b in BANDS])
    rows['_neg_n_detected'] = -rows[[f'{b}_detected' for b in BANDS]].sum(axis=1)
    rows['_neg_snr'] = -rows[[f'{b}_snr' for b in BANDS]].max(axis=1).fillna(-np.inf)

    obj = _best(rows, ['_not_primary', '_neg_n_detected', '_neg_snr'])[OBJECT_COLS[1:]]
    counts = rows.assign(_primary=~rows._not_primary).groupby('objID').agg(n_rows=('_primary', 'size'),
                                                                          n_primary_rows=('_primary', 'sum'))
    parts = [obj, counts]
    for b in BANDS:
        rows['_not_phot_src'] = (flag3[b] & STACK_PHOT_SRC) == 0
        rows['_undetected'] = ~rows[f'{b}_detected']
        rows['_neg_band_snr'] = -rows[f'{b}_snr'].fillna(-np.inf)
        best = _best(rows, ['_not_phot_src', '_undetected', '_not_primary', '_neg_band_snr'])
        parts.append(best[[f'{b}{c}' for c in BAND_COLS] + [f'{b}_detected']]
                     .assign(**{f'{b}_row': best.uniquePspsSTid, f'{b}_projcell': best.projectionID, f'{b}_skycell': best.skyCellID}))
    out = pd.concat(parts, axis=1).reset_index().rename(columns={'objID': 'PanSTARR_ID'})
    out.insert(1, 'ra', out.raStack.fillna(out.raMean))
    out.insert(2, 'dec', out.decStack.fillna(out.decMean))
    return out


def objects(ra: np.ndarray, dec: np.ndarray, directory: Optional[str] = None) -> pd.DataFrame:
    """The cleaned objects of every mirror partition containing any of these positions (whole partitions, so an object
    is cleaned the same way whichever field asks for it). An objID in two partitions -- only possible through step 1's
    duplicate join, whose two ObjectThin positions can straddle a boundary -- is kept once."""
    files = partition_files(ra, dec, directory)
    if not files:
        raise FileNotFoundError(f'no PS1 stack partitions for these positions in {stack_dir(directory)}')
    cat = pd.concat([clean(pq.ParquetFile(f).read().to_pandas()) for f in files], ignore_index=True)
    return cat.drop_duplicates('PanSTARR_ID').reset_index(drop=True)
