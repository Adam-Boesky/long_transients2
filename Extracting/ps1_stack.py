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
#     sensitivities in the stacked 3pi Steradian Survey". Table 11 also lists the validated exposures per filter:
#     g 60,528, r 70,918, i 104,414.
#   - Sec. 3.2 ("The 3pi Steradian Survey"): "The total number of exposures per field per year is still 4"; sec. 6.2
#     ("Variation of 3pi Steradian Survey Depth"): "by the design of the survey each pixel on the sky notionally has 12
#     visits, in practice the coverage can be much more variable than this."
# Taking the stack limit to be the depth at NOMINAL_VISITS frames is our assumption: the paper gives the stack limit as
# a sky mean, not as a function of the number of frames.
SINGLE_VISIT_5SIG = {'g': 22.0, 'r': 21.8, 'i': 21.5}
STACK_5SIG = {'g': 23.3, 'r': 23.2, 'i': 23.1}
NOMINAL_VISITS = 12


def depth_scale(b: str) -> float:
    """c in m(N) = m1 - 2.5 c log10(1/sqrt(N)), solved so that m(1) = SINGLE_VISIT_5SIG and m(NOMINAL_VISITS) =
    STACK_5SIG: c = (m1 - m12) / (2.5 log10(1/sqrt(12))). c = 1 is pure sqrt(N) averaging of identical exposures; the
    values are g 0.964, r 1.038 and i 1.186, i's above 1 consistent with i having ~1.5x the exposures of r (Table 11),
    i.e. more than 12 frames behind its mean stack depth."""
    return (SINGLE_VISIT_5SIG[b] - STACK_5SIG[b]) / (2.5 * np.log10(1 / np.sqrt(NOMINAL_VISITS)))


def stack_depth(n_frames: np.ndarray, b: str) -> np.ndarray:
    """5-sigma point-source depth in band b of a PS1 stack of `n_frames` exposures (`{b}nFrames`): m1 - 2.5 c
    log10(1/sqrt(N)), m1 = SINGLE_VISIT_5SIG and c from `depth_scale`, capped at STACK_5SIG (Chambers et al. 2016,
    Table 11) so that the limit is never deeper than the published mean stack depth. Fewer than one frame, or none
    known, gives the single-visit depth."""
    n = np.asarray(n_frames, float)
    n = np.where(n >= 1, n, 1.0)
    depth = SINGLE_VISIT_5SIG[b] - 2.5 * depth_scale(b) * np.log10(1 / np.sqrt(n))
    return np.minimum(depth, STACK_5SIG[b])


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
