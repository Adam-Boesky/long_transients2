"""Small helpers shared by the extraction, the filter and the site: data paths, credentials, FITS checks, the
image flux <-> AB magnitude conversion and ECSV/HDF5 table loading."""
import os
import pathlib
import subprocess
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import MaskedColumn, Table


def sky_xyz(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Unit vectors (N, 3) of positions in degrees, as astropy's `SkyCoord.cartesian`: KD-trees on these give exact
    angular neighbours at any declination."""
    r, d = np.radians(ra), np.radians(dec)
    return np.c_[np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)]


def chord(arcsec):
    """The unit-vector (chord) distance of an angle in arcsec, for `sky_xyz` trees."""
    return 2.0 * np.sin(np.radians(np.asarray(arcsec, float) / 3600.0) / 2.0)


def chord_to_arcsec(c):
    """The angle in arcsec of a chord distance between `sky_xyz` unit vectors."""
    return np.degrees(2.0 * np.arcsin(np.clip(np.asarray(c, float), 0.0, 2.0) / 2.0)) * 3600.0


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

def on_cluster() -> bool:
    """Check if the code is running on the cluster."""
    return os.path.exists('/n/home04')

def get_data_path() -> str:
    """Get the path to the data directory."""
    if on_cluster():
        return '/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients2'
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')  # <repo>/Data

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
