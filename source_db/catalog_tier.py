"""The catalog tier: one flattened row per source.

Reads the enriched combined tables and writes `sources.parquet` -- the ~300
science columns the crossmatching already produced, plus the four keys
(`source_uuid`, `coord_slug`, `catalog_tier`, `row_hash`).

No network access: everything here is already on disk, resolved during
`enrich_combined_tabs.py` against the local catalogs.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table

from Extracting.utils import load_ecsv
from Source_Analysis.catalogs._base import to_pandas_array
from source_db.identity import (assign_uuids, coord_slugs, load_identity,
                                update_identity)
from source_db.paths import DEFAULT_OUT_DIR, REPO


# (filename in <combined_dir>, catalog_tier label). Order fixes row order in the
# output, so the smallest tables land first and a schema problem surfaces early.
# `2` holds the PanSTARRS-only detections. `1_wide_association` (ZTF-only detections with a PS1 source 1-3" away)
# is only in older runs and is no longer offered as a category; both are optional.
TIERS: list[tuple[str, str]] = [
    ('2_enriched.ecsv', '2'),
    ('1_wide_association_enriched.ecsv', '1_wide_association'),
    ('1_enriched.ecsv', '1'),
    ('0_enriched.ecsv', '0'),
]
OPTIONAL_TIERS = {'2', '1_wide_association'}


# Values that upstream writes as a real number but means as "absent", mapped to
# null at build time so the database never serves a sentinel as data.
#
# Deliberately minimal. `cross_match.py` fills an unmatched PanSTARRS ID with 0
# rather than masking it; 0 is not a valid PanSTARRS objID, and all 10,160
# occurrences are ZTF-only tier-1 rows (r_Catalog_Flag == 1 for 9,836 of them,
# and only 6 have any finite PanSTARRS photometry).
#
# Other sentinels found in the audit are intentionally left alone:
#   ZTF_{g,r,i}KronMag == -999   a real photometric fill, but kept as-is
#   ZTF_KronRad == 0             ambiguous; 3,575 of those rows have KronFlag 0
# Anything added here makes the database differ from its source tables, so each
# entry needs evidence, and the verification has to know about it.
SENTINEL_NULLS: dict[str, tuple] = {
    'PSTARR_PanSTARR_ID': (0,),
}


KEY_COLUMNS = ['source_uuid', 'coord_slug', 'catalog_tier', 'row_hash']


def science_columns(df: pd.DataFrame) -> list[str]:
    """Columns that carry science, i.e. everything this module did not add."""
    return [c for c in df.columns if c not in KEY_COLUMNS]


def row_hashes(df: pd.DataFrame, columns: Optional[list[str]] = None) -> np.ndarray:
    """Per-row hash of the science columns, as uint64.

    Used only to answer "did the data under this label change?". A false
    *positive* (reporting a change when the values are identical) is harmless
    and possible across pandas upgrades, since this delegates to
    `pd.util.hash_pandas_object`; a false negative is not possible, which is
    the direction that matters.
    """
    cols = columns if columns is not None else science_columns(df)
    return pd.util.hash_pandas_object(df[cols], index=False).to_numpy(dtype=np.uint64)


def read_combined(combined_dir: Path) -> pd.DataFrame:
    """Stack the enriched tables into one frame, tagged with `catalog_tier`.

    Columns go through `_base.to_pandas_array`, never `Table.to_pandas()` --
    the latter rounds masked integers above 2**53, which corrupts most
    `desi_targetid` values. See that function's docstring.
    """
    combined_dir = Path(combined_dir)
    frames = []
    for fname, tier in TIERS:
        path = combined_dir / fname
        if not path.exists() and tier in OPTIONAL_TIERS:
            print(f'  {fname}: not found, skipped')
            continue
        if not path.exists():
            raise FileNotFoundError(f'missing {path} -- run enrich_combined_tabs.py first')
        table = load_ecsv(str(path))
        frame = pd.DataFrame({c: to_pandas_array(table[c]) for c in table.colnames})
        frame.insert(0, 'catalog_tier', tier)
        frames.append(frame)
        print(f'  {fname}: {len(frame):,} rows, {len(frame.columns) - 1} columns')
        del table
    out = pd.concat(frames, ignore_index=True)
    del frames
    regroup(out)
    return out


GROUP_RADIUS_ARCSEC = 1.0


def add_group_ids(tables: Iterable[Table], radius_arcsec: float = GROUP_RADIUS_ARCSEC) -> None:
    """Link rows within radius_arcsec of each other, across fields and across tables (plan_filter_review.md #1).

    Overlapping fields find the same object more than once. Linked rows (chains included) share an integer
    `group_id` (1, 2, ..., numbered by the group's mean RA); a row with no partner gets -1. Rows are never
    dropped or merged.
    """
    tables = [t for t in tables if len(t) > 0]
    if not tables:
        return
    ra = np.concatenate([np.asarray(t['ra'], dtype=float) for t in tables])
    dec = np.concatenate([np.asarray(t['dec'], dtype=float) for t in tables])
    coords = SkyCoord(ra, dec, unit='deg')
    i, j, _, _ = search_around_sky(coords, coords, radius_arcsec * u.arcsec)

    # Union-find over the pairs, so chains A-B-C end up in one group
    parent = np.arange(len(ra))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(i[i < j], j[i < j]):
        ra_, rb_ = find(a), find(b)
        if ra_ != rb_:
            parent[max(ra_, rb_)] = min(ra_, rb_)
    roots = np.array([find(x) for x in range(len(ra))])

    group_ids = np.full(len(ra), -1, dtype=np.int64)
    uniq, inv, counts = np.unique(roots, return_inverse=True, return_counts=True)
    xyz = coords.cartesian.xyz.value.T  # average on the unit sphere, safe across RA=0
    sums = np.zeros((len(uniq), 3))
    np.add.at(sums, inv, xyz)
    multi = np.where(counts > 1)[0]
    if len(multi):
        # Number groups by mean RA (then Dec) so the ids don't depend on file order
        mean = SkyCoord(*sums[multi].T, representation_type='cartesian').spherical
        order = np.lexsort((mean.lat.deg, mean.lon.deg))
        number = np.zeros(len(uniq), dtype=np.int64)
        number[multi[order]] = np.arange(1, len(multi) + 1)
        in_group = counts[inv] > 1
        group_ids[in_group] = number[inv[in_group]]

    start = 0
    for t in tables:
        t['group_id'] = group_ids[start:start + len(t)]
        start += len(t)


def regroup(df: pd.DataFrame) -> None:
    """Renumber `group_id` over all tiers together, in place.

    Each combined table numbers its duplicate groups from 1, so stacking them
    would make unrelated sources in different tiers share an id; one pass over
    the stacked rows also links an object found in more than one tier. Same
    1" rule and numbering as `add_group_ids`.
    """
    if 'group_id' not in df.columns:
        return
    t = Table({'ra': df['ra'].to_numpy(float), 'dec': df['dec'].to_numpy(float)})
    add_group_ids([t])
    df['group_id'] = np.asarray(t['group_id']) if len(t) else df['group_id']
    print(f'  group_id: {int(np.sum(df["group_id"] > 0)):,} rows in '
          f'{df.loc[df["group_id"] > 0, "group_id"].nunique():,} groups across tiers')


def apply_sentinel_nulls(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Replace the declared sentinel values in `SENTINEL_NULLS` with null."""
    for col, sentinels in SENTINEL_NULLS.items():
        if col not in df.columns:
            continue
        hit = df[col].isin(sentinels)
        n = int(hit.sum())
        if n:
            df.loc[hit, col] = pd.NA if str(df[col].dtype) != 'float64' else np.nan
            if verbose:
                print(f'  {col}: {n:,} sentinel {sentinels} -> null')
    return df


def _git_sha() -> Optional[str]:
    """HEAD of the repo, to stamp the release. None only if git itself fails."""
    try:
        return subprocess.check_output(
            ['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        # Deliberately narrow: a bare `except` here once swallowed a NameError
        # and silently stamped every release with git_sha=null.
        return None


def next_release_id(out_dir: Path) -> int:
    path = Path(out_dir) / 'release.json'
    if not path.exists():
        return 1
    return int(json.loads(path.read_text()).get('release_id', 0)) + 1


def build(
    combined_dir: Path,
    out_dir: Path = DEFAULT_OUT_DIR,
    dry_run: bool = False,
    write_sqlite: bool = False,
) -> dict:
    """Build the source database from a `combined/` directory of enriched tables.

    Idempotent with respect to identity: re-running against the same inputs
    reuses every UUID and creates none.
    """
    combined_dir, out_dir = Path(combined_dir), Path(out_dir)
    release_id = next_release_id(out_dir)
    print(f'release {release_id}  <- {combined_dir}')

    df = read_combined(combined_dir)
    df = apply_sentinel_nulls(df)
    ra, dec = df['ra'].to_numpy(float), df['dec'].to_numpy(float)
    slugs = coord_slugs(ra, dec)

    identity = load_identity(out_dir)
    print(f'  identity: {len(identity):,} known sources')
    uuids, is_new = assign_uuids(ra, dec, identity)
    identity = update_identity(identity, uuids, is_new, ra, dec, slugs, release_id)
    n_retired = int((identity['retired_at'] == release_id).sum())
    print(f'  {len(df):,} rows -> {int(is_new.sum()):,} new UUIDs, '
          f'{int((~is_new).sum()):,} carried forward, {n_retired:,} retired')

    df.insert(0, 'row_hash', row_hashes(df))
    df.insert(0, 'coord_slug', slugs)
    df.insert(0, 'source_uuid', uuids)
    df = df[KEY_COLUMNS + [c for c in df.columns if c not in KEY_COLUMNS]]

    n_dupe_uuid = len(df) - df['source_uuid'].nunique()
    if n_dupe_uuid:
        raise ValueError(f'{n_dupe_uuid} duplicate source_uuid values -- refusing to write')

    summary = {
        'release_id': release_id,
        'built_at': datetime.now(timezone.utc).isoformat(),
        'git_sha': _git_sha(),
        'combined_dir': str(combined_dir),
        'inputs': [f for f, _ in TIERS],
        'n_sources': int(len(df)),
        'n_columns': int(len(df.columns)),
        'n_new_uuids': int(is_new.sum()),
        'n_carried_forward': int((~is_new).sum()),
        'n_retired': n_retired,
        'n_identity': int(len(identity)),
    }

    if dry_run:
        print('  dry run -- nothing written')
        return summary

    out_dir.mkdir(parents=True, exist_ok=True)
    # Identity is written last: if the parquet write fails, identity is not
    # advanced and the build can simply be re-run.
    df.to_parquet(out_dir / 'sources.parquet', compression='zstd', index=False)
    print(f'  wrote sources.parquet '
          f'({(out_dir / "sources.parquet").stat().st_size / 1e6:.0f} MB)')

    if write_sqlite:
        import sqlite3
        db = out_dir / 'sources.sqlite'
        db.unlink(missing_ok=True)
        with sqlite3.connect(db) as con:
            df.to_sql('sources', con, index=False, chunksize=20000)
            con.execute('CREATE UNIQUE INDEX idx_uuid ON sources(source_uuid)')
            con.execute('CREATE INDEX idx_slug ON sources(coord_slug)')
        print(f'  wrote sources.sqlite ({db.stat().st_size / 1e6:.0f} MB)')

    identity.to_parquet(out_dir / 'identity.parquet', compression='zstd', index=False)
    (out_dir / 'release.json').write_text(json.dumps(summary, indent=2))
    print(f'  wrote identity.parquet ({len(identity):,} rows) and release.json')
    return summary
