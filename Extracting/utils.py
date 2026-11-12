import os
import pathlib
import subprocess
import time
import warnings
from contextlib import nullcontext

from astropy.io import ascii, fits
from astropy.table import Table, MaskedColumn
from functools import lru_cache
from mastcasjobs import MastCasJobs
from typing import Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd

MAST_CREDENTIAL_FNAME = 'mast_dino_login.txt'
print(f'CasJobs will use the credentials from {MAST_CREDENTIAL_FNAME}')


def fitscheck_valid(fpath: str) -> bool:
    """Return True if the FITS file passes fitscheck and is not truncated."""
    result = subprocess.run(
        ["fitscheck", "--ignore-missing", "--compliance", fpath],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return False
    # fitscheck doesn't catch partial downloads; opening with astropy fires a
    # warning when the file is shorter than the size declared in the header.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            with fits.open(fpath, memmap=True) as hdul:
                _ = hdul[0].header
        except Exception:
            return False
    return not any('truncated' in str(x.message).lower() for x in w)


def get_credentials(fname: str) -> Union[Tuple[str, str], str]:
    """Retrieves credentials from a specified file in my ~/vault/directory."""
    key_location = os.path.join(pathlib.Path.home(), f'vault/{fname}')
    return np.genfromtxt(key_location, dtype = 'str')


def img_flux_to_ab_mag(flux: np.ndarray, zero_point: float, fluxerr: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert image flux to AB magnitude.

    Parameters:
    flux (np.ndarray): The flux values of the image.
    zero_point (float): The zero point for the magnitude calculation.
    fluxerr (Optional[np.ndarray]): The flux error values of the image, by default None.

    Returns:
        1. The AB magnitudes corresponding to the input fluxes.
        2. If `fluxerr` is provided, the AB magnitude error.
    """
    mag = np.full_like(flux, np.nan, dtype=np.float64)
    positive_flux = flux > 0
    mag[positive_flux] = -2.5 * np.log10(flux[positive_flux]) + zero_point

    if fluxerr is not None:
        magerr = np.full_like(flux, np.nan, dtype=np.float64)
        magerr[positive_flux] = 2.5 * fluxerr[positive_flux] / (np.log(10) * flux[positive_flux])
        return mag, magerr

    return mag


def img_ab_mag_to_flux(mag: np.ndarray, zero_point: np.ndarray, magerr: Optional[np.ndarray] = None) -> np.ndarray:
    """Inverse of img_flux_to_ab_mag."""
    flux = 10 ** ((mag - zero_point) / -2.5)
    if magerr is not None:
        fluxerr = (magerr * (np.log(10) * flux)) / 2.5
        return flux, fluxerr
    return flux


def get_snr_from_mag(mag: np.ndarray, magerr: np.ndarray, zp: float) -> np.ndarray:
        flux, fluxerr = img_ab_mag_to_flux(mag, zero_point=zp, magerr=magerr)
        return flux / fluxerr


def on_cluster() -> bool:
    """Check if the code is running on the cluster."""
    return os.path.exists('/n/home04')


def get_data_path() -> str:
    """Get the path to the data directory."""
    if on_cluster():
        return '/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients2'
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')  # <repo>/Data


def true_nearby(row: int, column: int, radius: int, mask: np.ndarray) -> bool:

    # Adjust for negative indexing issues
    if radius > row:
        row = radius
    if radius > column:
        column = radius

    return np.any(
        mask[int(row - radius):int(row + radius + 1), int(column - radius):int(column + radius + 1)]
    )


def nan_nearby(row: int, column: int, radius: int, arr: np.ndarray) -> bool:
    """Check if there are any NaN values in the nearby pixels."""
    return true_nearby(row, column, radius, np.isnan(arr))


@lru_cache(maxsize=16)
def load_cached_table(table_path: str) -> Table:
    """Load a table from disk and cache it in memory."""
    return load_ecsv(table_path)


# Useful paths
RESULT_DIRPATH = os.path.join(get_data_path(), 'catalog_results')
MERGED_RESULTS_DIRPATH = os.path.join(RESULT_DIRPATH, 'field_results')

def get_n_quadrants_merged(field_name: str) -> Dict[str, int]:
    """Get the number of quadrants for a merged field. Returns a dictionary of the form {'band': number_of_quadrants}"""

    # Iterate through the extracted results and count the number of quadrants
    bands = ('g', 'r', 'i')
    quad_counts_per_band = {band: 0 for band in bands}
    for dirname in os.listdir(RESULT_DIRPATH):
    
        # If the field name is in the extracted directory
        if field_name == dirname[:6]:

            # Check if merged g, r, and i are in the directory
            for band in bands:
                if os.path.exists(os.path.join(RESULT_DIRPATH, dirname, f'{band}_associated.ecsv')):
                    quad_counts_per_band[band] += 1

    return quad_counts_per_band


def get_n_quadrants_merged_all_fields() -> Dict[str, Dict[str, int]]:
    """Get number of quadrants in all merged fields. Returns dict of form {'field': {'band': number_of_quadrants}}
    NOTE: ~0.73 sq deg / quadrant
    """
    quad_counts_per_band_all_fields = {}
    for merged_field_fname in os.listdir(MERGED_RESULTS_DIRPATH):
        field_name = merged_field_fname[:6]
        quad_counts_per_band_all_fields[field_name] = get_n_quadrants_merged(field_name)

    return quad_counts_per_band_all_fields


def metadata_from_field_dirname(field_dirname: str) -> Dict[str, str]:
    """Get the ZTF image metadata given the field's dirname in our saved data."""
    field, ccdid, qid = field_dirname.split('_')
    return {'fieldid': field, 'ccdid': ccdid, 'qid': qid}


def rcid_to_ccdid_qid(rcid: int) -> tuple:
    """Convert a ZTF readout-channel ID to (ccdid, qid).

    ZTF has 16 CCDs each with 4 quadrants, giving 64 readout channels.
    The mapping is: rcid = (ccdid - 1) * 4 + (qid - 1).
    """
    return rcid // 4 + 1, rcid % 4 + 1


# Columns that are logically integers but may arrive as nullable floats (None/NaN fill).
# These must be cast to masked int64 before writing to avoid float rounding of large IDs.
_INT64_COLUMNS = frozenset({'PSTARR_PanSTARR_ID', 'PanSTARR_ID', 'qualityFlag', 'primaryDetection'})


def prepare_table_for_write(table: Table) -> Table:
    """Return a copy of *table* with columns cast for safe serialization.

    Two fixes are applied:
    - Byte-string columns (dtype kind 'S' or object-of-bytes) are converted to str
      so ECSV writers don't choke on them.
    - Columns in ``_INT64_COLUMNS`` that arrived as nullable floats (None / NaN fill)
      are cast to masked int64, preserving integer precision for large IDs like
      PanSTARRS objIDs that would otherwise be rounded.
    """
    table = table.copy()
    for colname in table.colnames:
        col = table[colname]
        # Convert integer ID columns
        if colname in _INT64_COLUMNS:
            data = np.array(col)
            mask = np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in data])
            values = np.array(
                [v if not (v is None or (isinstance(v, float) and np.isnan(v))) else -1 for v in data],
                dtype=np.int64,
            )
            table[colname] = MaskedColumn(values, mask=mask)
        # Bytes or any remaining object dtype -> str (HDF5 has no object equivalent)
        elif col.dtype.kind == 'S' or col.dtype.kind == 'O':
            table[colname] = col.astype(str)
    return table


def load_ecsv(fpath: str, careful_load: bool = True) -> Table:
    """Load a ecsv file as an astropy table. If you want speed and are okay with roundoff, careful_load can be False."""
    # If the file is a hdf5 file, load it directly
    if fpath.endswith('.hdf5'):
        return Table.read(fpath, path='data')

    # Try getting the hdf5 version if it exists
    if fpath.endswith('.ecsv'):
        hdf5_version_fpath = f'{fpath[:-5]}.hdf5'
        if os.path.exists(hdf5_version_fpath):
            print(f'hdf5 version of {fpath.split("/")[-1]} found, loading instead of ecsv...')
            return Table.read(hdf5_version_fpath, path='data')

    if careful_load:
        return Table.read(fpath, format='ascii.ecsv')
    return Table.from_pandas(pd.read_csv(fpath, comment='#', delimiter=' '))


# Set up casjobs object
wsid, password = get_credentials(MAST_CREDENTIAL_FNAME)
MASTCASJOBS = MastCasJobs(context="PanSTARRS_DR2", userid=wsid, password=password, request_type='POST')

# Per-process CasJobs lock (set by init_worker_lock in parallel contexts).
# Only one process should talk to the CasJobs service at a time because the
# shared account cannot have multiple open DataReaders simultaneously.
_casjobs_lock = None


def init_worker_lock(lock) -> None:
    """Initializer for ProcessPoolExecutor workers — stores the shared lock."""
    global _casjobs_lock
    _casjobs_lock = lock


def casjobs_query_lock():
    """Return the active CasJobs lock, or a no-op context if none was set."""
    return _casjobs_lock if _casjobs_lock is not None else nullcontext()


# Per-process SDSS rate-limit lock (set by init_worker_sdss_lock in parallel contexts).
# SDSS SkyServer enforces a hard cap of 60 queries/minute (>=1 second between queries) and
# blocks with an HTTP 403 if that's exceeded:
# https://skyserver.sdss.org/dr14/en/help/docs/limits.aspx
SDSS_MIN_QUERY_INTERVAL_SEC = 1.1
_sdss_lock = None
_sdss_last_call_time = None


def init_worker_sdss_lock(lock, last_call_time) -> None:
    """Initializer for Pool/ProcessPoolExecutor workers — stores the shared SDSS rate-limit lock."""
    global _sdss_lock, _sdss_last_call_time
    _sdss_lock = lock
    _sdss_last_call_time = last_call_time


def throttled_sdss_query(query_fn, *args, **kwargs):
    """Call query_fn (an SDSS query), pausing as needed so calls across all worker processes
    stay under SDSS's rate limit."""
    if _sdss_lock is None:
        return query_fn(*args, **kwargs)
    with _sdss_lock:
        wait = SDSS_MIN_QUERY_INTERVAL_SEC - (time.time() - _sdss_last_call_time.value)
        if wait > 0:
            time.sleep(wait)
        try:
            return query_fn(*args, **kwargs)
        finally:
            _sdss_last_call_time.value = time.time()


def _add_pstarr_mag_cols(tab: Table) -> Table:
    mags, magerrs = img_flux_to_ab_mag(tab['psfFlux'], fluxerr=tab['psfFluxErr'], zero_point=8.9)  # 8.9 is used by 
    tab['mag'] = mags
    tab['magerr'] = magerrs

    return tab


def _drop_bad_pstarr_flags(tab: Table) -> Table:
    """Drop bad flags. For flag references see https://outerspace.stsci.edu/display/PANSTARRS/PS1+Detection+Flags."""
    tab = tab[tab['infoFlag'] & 2048 == 0]  # source is thought to be a defect
    tab = tab[tab['infoFlag2'] & 4 == 0]    # weird forced photometry thing

    return tab


def get_pstarr_lc_from_id(objid: int) -> Table:
    if not isinstance(objid, int) and not np.isnan(objid):
        objid = int(objid)

    # Construct the query
    query = f"""
SELECT d.objID, d.ra, d.dec, d.obsTime, d.filterID, d.psfFlux, d.psfFluxErr, d.infoFlag, d.infoFlag2, d.zp FROM Detection d WHERE d.objID = {objid}
"""

    # Get table and add mag cols
    tab = MASTCASJOBS.quick(query, task_name=f"PanSTARRS_lc")
    tab = _drop_bad_pstarr_flags(tab)
    tab = _add_pstarr_mag_cols(tab)

    return tab


def get_pstarr_lc_from_coord(ra: float, dec: float, rad_arcsec: float = 1.0) -> Table:
    # fGetNearestObjEq takes ARCMINUTES, so convert. This used to pass
    # rad_arcsec straight through, making the radius 60x too large.
    # Nearest, not nearby: PS1 duplicates sources at skycell boundaries.
    query = f"""
SELECT d.objID, d.ra, d.dec, d.obsTime, d.filterID, d.psfFlux, d.psfFluxErr, d.infoFlag, d.infoFlag2, d.zp
FROM fGetNearestObjEq({ra}, {dec}, {rad_arcsec / 60.0}) nb
INNER JOIN Detection d on d.objID = nb.objID
"""

    # Get table and add mag cols
    tab = MASTCASJOBS.quick(query, task_name=f"PanSTARRS_lc")
    tab = _drop_bad_pstarr_flags(tab)
    tab = _add_pstarr_mag_cols(tab)

    return tab


# PS1 query box padding beyond the union of a quadrant's band images; covers the 1" match and the 1-3" diversion
PS1_QUERY_PAD_DEG = 30.0 / 3600.0


def coord_range_from_wcs(wcs, shape: Tuple[int, int]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """RA/Dec range of an image's corners. An image straddling RA=0 is returned as (lo, hi + 360)."""
    ny, nx = shape
    corners = [wcs.pixel_to_world(0, 0), wcs.pixel_to_world(nx, 0), wcs.pixel_to_world(nx, ny), wcs.pixel_to_world(0, ny)]
    corner_ras = np.array([c.ra.deg for c in corners])
    corner_decs = np.array([c.dec.deg for c in corners])
    ra_range = (np.min(corner_ras), np.max(corner_ras))
    dec_range = (np.min(corner_decs), np.max(corner_decs))

    # Straddles RA=0: span the outer corners, as (lo, hi + 360)
    if ra_range[1] - ra_range[0] > 180.0:
        ra_range = (np.min(corner_ras[corner_ras > 180.0]), np.max(corner_ras[corner_ras < 180.0]) + 360.0)

    return ra_range, dec_range


def union_coord_range(
        ranges, pad_deg: float = PS1_QUERY_PAD_DEG,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Padded union of (ra_range, dec_range) boxes, e.g. one per band image of a quadrant.

    The band deep stacks are offset from each other, so a PS1 box from one band misses strips of the others.
    RA follows coord_range_from_wcs: a box straddling RA=0 is returned as (lo, hi + 360).
    """
    ras = [tuple(r) for r, _ in ranges]
    if any(hi > 360.0 for _, hi in ras) or max(hi for _, hi in ras) - min(lo for lo, _ in ras) > 180.0:  # same side of RA=0
        ras = [(lo + 360.0, hi + 360.0) if lo < 180.0 else (lo, hi) for lo, hi in ras]
    ra_lo, ra_hi = min(lo for lo, _ in ras), max(hi for _, hi in ras)
    dec_lo = max(-90.0, min(d[0] for _, d in ranges) - pad_deg)
    dec_hi = min(90.0, max(d[1] for _, d in ranges) + pad_deg)

    max_abs_dec = max(abs(dec_lo), abs(dec_hi))
    if max_abs_dec > 89.5:
        return (0.0, 360.0), (dec_lo, dec_hi)
    ra_pad = pad_deg / np.cos(np.radians(max_abs_dec))
    ra_lo, ra_hi = ra_lo - ra_pad, ra_hi + ra_pad
    if ra_hi - ra_lo >= 360.0:
        return (0.0, 360.0), (dec_lo, dec_hi)
    if ra_lo < 0.0:
        ra_lo, ra_hi = ra_lo + 360.0, ra_hi + 360.0
    return (ra_lo, ra_hi), (dec_lo, dec_hi)
