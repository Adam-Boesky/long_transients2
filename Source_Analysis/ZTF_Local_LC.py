import os
import re
import sys
import numpy as np
import pandas as pd
import ztffields

from typing import Dict, List, Optional, Tuple
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord

sys.path.append('/Users/adamboesky/Research/long_transients')
sys.path.append('/n/home04/aboesky/berger/long_transients')



def rcid_to_ccdid_qid(rcid: int) -> tuple:
    """Convert a ZTF readout-channel ID to (ccdid, qid).

    ZTF has 16 CCDs each with 4 quadrants, giving 64 readout channels.
    The mapping is: rcid = (ccdid - 1) * 4 + (qid - 1).
    """
    return rcid // 4 + 1, rcid % 4 + 1


# Source-level bits from ZTF Explanatory Supplement 10.3, plus bit 15 (32768):
# cloud- or moon-affected epochs (DR23 release notes 9b).
BAD_CATFLAG_MASK = 6141 | 32768

# Parquet filename pattern: ztf_{fieldid}_z{band}_c{ccdid}_q{qid}_dr{dr}.parquet
_FNAME_RE = re.compile(r'^ztf_(\d+)_z([gri])_c(\d+)_q(\d+)_dr(\d+)\.parquet$')


class ZTF_LC:
    """Light curves sourced from locally-stored ZTF parquet files.

    Each parquet file covers one field / CCD / quadrant / band combination and
    is named ``ztf_{fieldid}_z{band}_c{ccdid}_q{qid}_dr{dr}.parquet``.  Rows
    are one-per-object with array-valued time-series columns (``hmjd``, ``mag``,
    ``magerr``, ``clrcoeff``, ``catflags``).

    ``ztffields`` is used to map the target coordinate to the correct
    field / quadrant(s), so only the relevant files are loaded.  Coordinates
    near field boundaries may fall in multiple overlapping fields; all
    matching epochs are included.

    Parameters
    ----------
    data_dir:
        Directory containing ZTF parquet files.
    ra, dec:
        ICRS coordinates (degrees) of the target.  Either ``(ra, dec)`` or
        ``objectid`` must be provided.
    objectid:
        ZTF objectid.  If ``(ra, dec)`` is also given, files are found via
        field lookup and the objectid is used to select the exact row.
        If only ``objectid`` is given, all files in ``data_dir`` are scanned.
    query_rad_arcsec:
        Cone-search radius used for coordinate matching (default 1.5″).
    apply_catflag_mask:
        If True (default), epochs with ``catflags & BAD_CATFLAG_MASK != 0``
        are discarded.
    """

    def __init__(
        self,
        data_dir: str = '/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/ztf_lc_dir/',
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        objectid: Optional[int] = None,
        query_rad_arcsec: float = 1.5,
        apply_catflag_mask: bool = True,
    ):
        if ra is None and objectid is None:
            raise ValueError("Provide either (ra, dec) or objectid.")
        if ra is not None and dec is None:
            raise ValueError("dec is required when ra is given.")

        self.data_dir = data_dir
        self.ra = ra
        self.dec = dec
        self.objectid = objectid
        self.query_rad_arcsec = query_rad_arcsec
        self.apply_catflag_mask = apply_catflag_mask

        self._lc: Optional[Table] = None
        self._lc_by_band: Optional[Dict[str, Table]] = None

    # ------------------------------------------------------------------
    # Filename parsing  (mirrors padding logic in Catalogs.py)
    # ------------------------------------------------------------------

    @staticmethod
    def parse_filename(fname: str) -> Optional[Dict[str, str]]:
        """Parse ZTF parquet filename metadata. Returns None if not a match.

        Returns a dict with keys: fieldid, band, ccdid, qid, dr.
        """
        m = _FNAME_RE.match(os.path.basename(fname))
        if m is None:
            return None
        return {'fieldid': m.group(1), 'band': m.group(2),
                'ccdid': m.group(3), 'qid': m.group(4), 'dr': m.group(5)}

    @staticmethod
    def build_filename(fieldid: int, band: str, ccdid: int, qid: int, dr: int) -> str:
        """Construct the expected parquet filename for a given field/band/ccd/quadrant."""
        return f"ztf_{fieldid:06d}_z{band}_c{ccdid:02d}_q{qid}_dr{dr}.parquet"

    # ------------------------------------------------------------------
    # Field / quadrant lookup via ztffields
    # ------------------------------------------------------------------

    def _get_field_quadrants(self) -> List[Tuple[int, int, int]]:
        """Return list of (fieldid, ccdid, qid) covering the target coordinate.

        Uses ``ztffields.radec_to_fieldid`` at the quadrant level.  A single
        coordinate can fall in multiple overlapping fields.
        """
        fq_df = ztffields.radec_to_fieldid([[self.ra, self.dec]], level='quadrant')
        result = []
        for _, row in fq_df.iterrows():
            ccdid, qid = rcid_to_ccdid_qid(int(row['rcid']))
            result.append((int(row['fieldid']), ccdid, qid))
        return result

    def _find_file(self, fieldid: int, ccdid: int, qid: int, band: str) -> Optional[str]:
        """Find the parquet file for a given field/ccd/quadrant/band in ``data_dir``.

        Matches any data release version present on disk.
        """
        pattern = re.compile(
            rf'^ztf_{fieldid:06d}_z{band}_c{ccdid:02d}_q{qid}_dr\d+\.parquet$'
        )

        subdir_char = f"{fieldid:06d}"[2]  # '0' for fields 0–999, '1' for 1000–1999
        field_dir = os.path.join(self.data_dir, subdir_char, f"field{fieldid:06d}")
        if not os.path.isdir(field_dir):
            return None

        for fname in os.listdir(field_dir):
            if pattern.match(fname):
                return os.path.join(field_dir, fname)
        return None

    # ------------------------------------------------------------------
    # Parquet loading helpers
    # ------------------------------------------------------------------

    def _search_file(self, fpath: str) -> Optional[pd.Series]:
        """Return the matching row from *fpath*, or None if not found.

        Two-pass read: lightweight index columns first, full row only on match.
        """
        df_idx = pd.read_parquet(fpath, columns=['objectid', 'objra', 'objdec'])

        if self.objectid is not None:
            mask = df_idx['objectid'] == self.objectid
            if not mask.any():
                return None
            row_index = df_idx.index[mask][0]
        else:
            ras  = df_idx['objra'].to_numpy(dtype=float)
            decs = df_idx['objdec'].to_numpy(dtype=float)
            seps = SkyCoord(self.ra, self.dec, unit='deg').separation(
                SkyCoord(ras, decs, unit='deg')
            ).arcsec
            best = int(np.argmin(seps))
            if seps[best] >= self.query_rad_arcsec:
                return None
            row_index = df_idx.index[best]

        df_full = pd.read_parquet(fpath)
        return df_full.loc[row_index]

    def _row_to_table(self, row: pd.Series, band: str) -> Table:
        """Convert a matched parquet row to an astropy Table of LC epochs.

        Output columns mirror ``Light_Curve.get_catalog_lc('ztf')``:
        ``ztf_id``, ``ra``, ``dec``, ``hmjd``, ``filtercode``, ``mag``,
        ``magerr``, ``clrcoeff``, ``catflags``, and the NaN-filled band
        columns ``g_mag``, ``g_magerr``, ``r_mag``, ``r_magerr``,
        ``i_mag``, ``i_magerr``.

        Note: the time column is ``hmjd`` (heliocentric MJD, as stored in
        the parquet files) rather than geocentric ``mjd``.
        """
        hmjd     = np.array(row['hmjd'],     dtype=float)
        mag      = np.array(row['mag'],      dtype=float)
        magerr   = np.array(row['magerr'],   dtype=float)
        catflags = np.array(row['catflags'], dtype=int)
        clrcoeff = np.array(row['clrcoeff'], dtype=float)

        if self.apply_catflag_mask:
            good = (catflags & BAD_CATFLAG_MASK == 0) & np.isfinite(mag) & np.isfinite(magerr)
        else:
            good = np.isfinite(mag) & np.isfinite(magerr)

        n          = int(good.sum())
        filtercode = f'z{band}'
        nan_col    = np.full(n, np.nan)

        tab = Table({
            'ztf_id':     np.full(n, int(row['objectid']), dtype=np.int64),
            'ra':         np.full(n, float(row['objra'])),
            'dec':        np.full(n, float(row['objdec'])),
            'hmjd':       hmjd[good],
            'filtercode': np.full(n, filtercode),
            'mag':        mag[good],
            'magerr':     magerr[good],
            'clrcoeff':   clrcoeff[good],
            'catflags':   catflags[good],
            # NaN-filled band columns — only the matching band is populated
            'g_mag':      mag[good]    if band == 'g' else nan_col.copy(),
            'g_magerr':   magerr[good] if band == 'g' else nan_col.copy(),
            'r_mag':      mag[good]    if band == 'r' else nan_col.copy(),
            'r_magerr':   magerr[good] if band == 'r' else nan_col.copy(),
            'i_mag':      mag[good]    if band == 'i' else nan_col.copy(),
            'i_magerr':   magerr[good] if band == 'i' else nan_col.copy(),
        })
        return tab

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_lc(self) -> Table:
        """Load all matching parquet files and return the combined light curve.

        Returns
        -------
        Table
            All epochs across all bands, sorted by ``hmjd``.  Columns match
            ``Light_Curve.get_catalog_lc('ztf')``: ``ztf_id``, ``ra``,
            ``dec``, ``hmjd``, ``filtercode``, ``mag``, ``magerr``,
            ``clrcoeff``, ``catflags``, ``g_mag``, ``g_magerr``, ``r_mag``,
            ``r_magerr``, ``i_mag``, ``i_magerr``.
        """
        tabs: List[Table] = []

        if self.ra is not None:
            # Use ztffields to find only the relevant files
            for fieldid, ccdid, qid in self._get_field_quadrants():
                for band in 'gri':
                    fpath = self._find_file(fieldid, ccdid, qid, band)
                    if fpath is None:
                        continue
                    row = self._search_file(fpath)
                    if row is not None:
                        tabs.append(self._row_to_table(row, band))
        else:
            # objectid-only fallback: walk all subdirs
            for root, _dirs, files in os.walk(self.data_dir):
                for fname in sorted(files):
                    meta = self.parse_filename(fname)
                    if meta is None:
                        continue
                    row = self._search_file(os.path.join(root, fname))
                    if row is not None:
                        tabs.append(self._row_to_table(row, meta['band']))

        if tabs:
            lc = vstack(tabs)
            lc.sort('hmjd')
        else:
            lc = Table(
                names=['ztf_id', 'ra', 'dec', 'hmjd', 'filtercode', 'mag', 'magerr',
                        'clrcoeff', 'catflags', 'g_mag', 'g_magerr', 'r_mag', 'r_magerr',
                        'i_mag', 'i_magerr'],
                dtype=[np.int64, float, float, float, str, float, float,
                       float, int, float, float, float, float, float, float],
            )
        return lc

    @property
    def lc(self) -> Table:
        """Combined LC table (all bands), sorted by hmjd. Loaded lazily."""
        if self._lc is None:
            self._lc = self.get_lc()
        return self._lc

    @property
    def lc_by_band(self) -> Dict[str, Table]:
        """Per-band views of ``lc``, each sorted by hmjd."""
        if self._lc_by_band is None:
            self._lc_by_band = {
                band: self.lc[self.lc['filtercode'] == f'z{band}']
                for band in 'gri'
                if np.any(self.lc['filtercode'] == f'z{band}')
            }
        return self._lc_by_band
