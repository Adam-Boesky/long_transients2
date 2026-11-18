"""PanSTARRS DR2 light curves, in bulk, from the MAST CasJobs Detection table.

Join on objID, not position
---------------------------
`Light_Curve` looks these up by PanSTARRS object id (`Sources.py` passes
`PSTARR_PanSTARR_ID` straight from the source row); the 0.1" cone in
`get_pstarr_lc_from_coord` is only a fallback for sources with no id. MAST says
the same thing outright: "Matching objects using IDs is much faster than
matching by RA and Dec."

We already hold the ids, so for those sources this is an exact join with no
match radius to justify. 158,892 of 169,052 have one.

The other 10,160 (ZTF-only rows, id absent -- see SENTINEL_NULLS in
source_db.catalog_tier) still need the coordinate path, and it is not a
formality: sampling 8 of them, 7 returned PanSTARRS detections within 0.1".
So "no objID" does not mean "nothing there" -- these positions do have
catalogued PS1 objects, which suggests the local PS1 parquet the crossmatch
runs against is less complete than the archive. Skipping them would drop real
data, and would disagree with the pages, which take this path too.

The sanctioned CasJobs pattern
------------------------------
From the PS1 sample queries, for a job this size:

    SELECT ... INTO mydb.[Result]
    FROM mydb.[Uploaded] o
    JOIN Detection d ON d.ObjID = o.ObjID

Three things that matter, all from MAST's own guidance:

* **Always give JOIN an ON clause.** "If you fail to include an ON condition
  in JOIN statements, CasJobs will attempt a cross product that can quickly
  fill up all MyDB disk space."
* **Write results INTO MyDB, then download.** Their worked example joins
  192,611 objects -> 8,560,632 detections (~1.3 GB) in 3:18, which "exceeded
  the standard limit in CasJobs" for a directly returned result. Ours is the
  same order.
* **submit()/monitor(), not quick().** `quick` is documented for queries
  "that execute in less than 1 minute".

Stored raw, reduced at render time
----------------------------------
`Source_Analysis.casjobs` drops `infoFlag & 2048` (defect) and `infoFlag2 & 4`
(forced-photometry artifact) before plotting, and converts psfFlux to AB mag
at zero point 8.9. The magnitude conversion is lossless so it happens here;
the flag cut is lossy, so the flags are stored and the cut is registered in
`_base.RENDER_TIME_REDUCTIONS` alongside ZTF's catflags and WISE's binning.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from source_db.lightcurves import _base  # noqa: E402
from source_db.paths import RAW_DIR  # noqa: E402

SURVEY = 'panstarrs'

# Matches Source.max_arcsec / enrich_combined_tabs.
MATCH_RADIUS_ARCSEC = 1.5

# From the PS1 `Filter` table, not guessed.
FILTERID_TO_BAND = {1: 'g', 2: 'r', 3: 'i', 4: 'z', 5: 'y'}

# psfFlux is in Janskys; 8.9 is the AB zero point used by _add_pstarr_mag_cols.
AB_ZERO_POINT = 8.9

# One CasJobs job per chunk: bounds a failure to one chunk and keeps each
# result table well inside MyDB limits.
CHUNK_SIZE = 40_000

DETECTION_COLUMNS = ['objID', 'obsTime', 'filterID', 'psfFlux', 'psfFluxErr',
                     'infoFlag', 'infoFlag2', 'zp']

# 1.5" to match Source.max_arcsec and every other catalog here. *Nearest*,
# not *nearby*: PS1 duplicates sources at skycell boundaries, so nearby
# returns several objIDs for one source and nearest returns exactly one.
COORD_RADIUS_ARCSEC = MATCH_RADIUS_ARCSEC


def _sources(limit: Optional[int] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split sources into the objID path and the coordinate path.

    Returns `(with_id, without_id)`: the first carries `source_uuid` + `objid`,
    the second `source_uuid` + `ra`/`dec`.
    """
    import pyarrow.parquet as pq
    from source_db.paths import DEFAULT_OUT_DIR
    df = pq.read_table(
        Path(DEFAULT_OUT_DIR) / 'sources.parquet',
        columns=['source_uuid', 'ra', 'dec', 'PSTARR_PanSTARR_ID']).to_pandas()
    if limit:
        df = df.head(limit)
    has = df['PSTARR_PanSTARR_ID'].notna()
    with_id = df[has].copy()
    with_id['objid'] = with_id['PSTARR_PanSTARR_ID'].astype('int64')
    with_id = with_id[['source_uuid', 'objid']].reset_index(drop=True)
    without = df[~has][['source_uuid', 'ra', 'dec']].reset_index(drop=True)
    return with_id, without


def _objids(limit: Optional[int] = None) -> pd.DataFrame:
    return _sources(limit)[0]


def _run(upload, sql_for, chunk_id: str, task: str) -> pd.DataFrame:
    """Upload a table, run the join INTO MyDB, download, drop both tables."""
    from Source_Analysis.casjobs import MASTCASJOBS

    up, res = f'lt_{task}_{chunk_id}', f'lt_det_{task}_{chunk_id}'
    for t in (up, res):
        try:
            MASTCASJOBS.drop_table_if_exists(t)
        except Exception:
            pass
    MASTCASJOBS.upload_table(up, upload)
    MASTCASJOBS.monitor(MASTCASJOBS.submit(sql_for(up, res),
                                           task_name=f'lt_ps1_{task}_{chunk_id}'))
    try:
        out = MASTCASJOBS.get_table(res, format='CSV').to_pandas()
    finally:
        for t in (up, res):
            try:
                MASTCASJOBS.drop_table_if_exists(t)
            except Exception:
                pass
    return out


def fetch_chunk(sub: pd.DataFrame, chunk_id: str) -> pd.DataFrame:
    """objID path: exact join of Detection against uploaded ids."""
    from astropy.table import Table
    cols = ', '.join(f'd.{c}' for c in DETECTION_COLUMNS)
    # Only the join key goes up; source_uuid is reattached locally, which
    # avoids sending a 36-char string column through CasJobs.
    out = _run(Table({'objid': sub.objid.to_numpy(np.int64)}),
               lambda up, res: (f'SELECT {cols} INTO mydb.[{res}] '
                                f'FROM mydb.[{up}] o '
                                f'JOIN Detection d ON d.objID = o.objid'),
               chunk_id, 'id')
    if out.empty:
        return out
    return out.merge(sub, left_on='objID', right_on='objid',
                     how='inner').drop(columns=['objid'])


def fetch_chunk_coord(sub: pd.DataFrame, chunk_id: str,
                      radius_arcsec: float = COORD_RADIUS_ARCSEC) -> pd.DataFrame:
    """Coordinate path, for sources with no objID.

    fGetNearestObjEq returns at most one object per position, which is what
    STScI's PS1 DR2 tutorial uses for single-object light curves and which
    avoids PS1's skycell-boundary duplicates. `idx` carries each row back to
    its source_uuid.

    Note the third argument is in *arcminutes*, as is the `distance` the
    function returns -- verified against the catalog, not assumed.
    """
    from astropy.table import Table
    cols = ', '.join(f'd.{c}' for c in DETECTION_COLUMNS)
    rad_arcmin = radius_arcsec / 60.0
    up_tab = Table({'idx': np.arange(len(sub), dtype=np.int64),
                    'ra': sub.ra.to_numpy(float),
                    'dec': sub.dec.to_numpy(float)})
    out = _run(up_tab,
               lambda up, res: (f'SELECT o.idx, {cols} INTO mydb.[{res}] '
                                f'FROM mydb.[{up}] o '
                                f'CROSS APPLY fGetNearestObjEq(o.ra, o.dec, {rad_arcmin}) nb '
                                f'JOIN Detection d ON d.objID = nb.objID'),
               chunk_id, 'coord')
    if out.empty:
        return out
    out['source_uuid'] = sub.source_uuid.to_numpy()[out['idx'].astype(int).to_numpy()]
    return out.drop(columns=['idx'])


def fetch(limit: Optional[int] = None, raw_dir: Path = RAW_DIR,
          chunk_size: int = CHUNK_SIZE, resume: bool = True) -> dict:
    """Both paths: exact objID join, then the 0.1" cone for sources without one."""
    with_id, without = _sources(limit)
    covered = _base.covered_sources(SURVEY, raw_dir) if resume else {}
    print(f'{SURVEY}: {len(with_id):,} by objID, {len(without):,} by coordinate',
          flush=True)

    # chunk ids are prefixed so the two paths never collide in raw/
    work = ([(f'id{c}', sub, fetch_chunk) for c, sub in _base.row_chunks(with_id, chunk_size)]
            + [(f'co{c}', sub, fetch_chunk_coord)
               for c, sub in _base.row_chunks(without, chunk_size)])
    todo = [w for w in work if not set(w[1].source_uuid) <= covered.get(w[0], set())]
    print(f'  {len(work)} chunks, {len(todo)} to do', flush=True)

    t0 = time.time()
    n = 0
    for cid, sub, fn in todo:
        t1 = time.time()
        df = fn(sub, cid)
        _base.write_raw(SURVEY, cid, df, raw_dir=raw_dir, n_sources=len(sub))
        _base.write_coverage(SURVEY, cid, sub.source_uuid, raw_dir=raw_dir)
        n += len(df)
        print(f'  chunk {cid}: {len(sub):,} sources -> {len(df):,} detections '
              f'[{(time.time() - t1) / 60:.1f} min]', flush=True)
    print(f'{SURVEY}: {n:,} detections in {(time.time() - t0) / 60:.1f} min',
          flush=True)
    return {'survey': SURVEY, 'n_rows': n, 'seconds': time.time() - t0}


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Detection rows -> the shared light-curve schema.

    obsTime is MJD at exposure midpoint (TAI, not UTC -- the ~37 s offset is
    far below anything that matters for these light curves).
    """
    if raw.empty:
        return _base.empty_lightcurves()
    flux = raw['psfFlux'].to_numpy(float)
    ferr = raw['psfFluxErr'].to_numpy(float)
    ok = np.isfinite(flux) & (flux > 0)
    sub = raw[ok]
    flux, ferr = flux[ok], ferr[ok]
    mag = AB_ZERO_POINT - 2.5 * np.log10(flux)
    out = pd.DataFrame({
        'source_uuid': sub['source_uuid'].astype('string'),
        'survey': pd.array([SURVEY] * len(sub), dtype='string'),
        # The two fetch paths disagree on filterID's type -- the objID join
        # returns int64, fGetNearestObjEq returns strings -- so concatenating
        # the chunks gives an object column. Coerce before mapping.
        'band': (pd.to_numeric(sub['filterID'], errors='coerce')
                 .astype('Int64').map(FILTERID_TO_BAND).astype('string')),
        'mjd': sub['obsTime'].astype('float64'),
        'mag': mag,
        'magerr': 2.5 / np.log(10) * ferr / flux,
        'snr': flux / ferr,
        'catalog_id': sub['objID'].astype('string'),
    })
    out['infoFlag'] = sub['infoFlag'].to_numpy()
    out['infoFlag2'] = sub['infoFlag2'].to_numpy()
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    ap.add_argument('--raw-dir', type=Path, default=RAW_DIR)
    a = ap.parse_args()
    fetch(limit=a.limit, raw_dir=a.raw_dir, chunk_size=a.chunk_size)
