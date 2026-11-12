"""Download the Pan-STARRS DR2 skycell stacks (rings.v3) that cover a sky footprint.

A skycell is a ~0.4 deg, 6250 px tile at 0.25"/px; a ZTF quadrant overlaps ~9 of them, usually
from two projection cells. Each skycell and band has a stack image (`stk.<b>.unconv.fits`), a
16-bit mask (`.unconv.mask.fits`), a variance map (`.unconv.wt.fits`) and a few others; see
PRODUCT_SUFFIX. Stacks are tile-compressed and asinh-scaled; that is undone at load time, not here.

Docs: https://outerspace.stsci.edu/spaces/PANSTARRS/pages/298812251/PS1+Image+Cutout+Service
"""
import io
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import requests
from astropy.table import Table, unique

try:
    from utils import fitscheck_valid
except ModuleNotFoundError:
    from .utils import fitscheck_valid

PS1_HOST = 'https://ps1images.stsci.edu'
FILENAME_SERVICE = f'{PS1_HOST}/cgi-bin/ps1filenames.py'
SKYCELL_DEG = 0.4  # nominal skycell size; footprint sampling must be finer than this
PRODUCT_SUFFIX = {
    'stack': 'unconv.fits',
    'mask': 'unconv.mask.fits',
    'wt': 'unconv.wt.fits',
    'num': 'unconv.num.fits',
    'exp': 'unconv.exp.fits',
    'psf': 'target.psf',  # IPP analytic PSF model (PS_MODEL_PS1_V1), not a PSF image
}
DEFAULT_PRODUCTS = ('stack', 'mask', 'wt')
MAX_THREADS = 4  # STScI asks for no more than 10 simultaneous connections
REQUEST_TIMEOUT = (10, 120)  # (connect, read) seconds


def skycell_name(projcell: int, subcell: int) -> str:
    return f'rings.v3.skycell.{int(projcell):04d}.{int(subcell):03d}'


def skycell_url(projcell: int, subcell: int, band: str, product: str = 'stack') -> str:
    """URL of one skycell product, e.g. .../rings.v3.skycell/1921/090/rings.v3.skycell.1921.090.stk.r.unconv.fits"""
    name = skycell_name(projcell, subcell)
    return f'{PS1_HOST}/rings.v3.skycell/{int(projcell):04d}/{int(subcell):03d}/{name}.stk.{band}.{PRODUCT_SUFFIX[product]}'


def skycell_path(data_dir: str, projcell: int, subcell: int, band: str, product: str = 'stack') -> str:
    """Local path of a skycell product; the archive's own file name, flat in `data_dir`."""
    return os.path.join(data_dir, f'{skycell_name(projcell, subcell)}.stk.{band}.{PRODUCT_SUFFIX[product]}')


def footprint_grid(
        ra_range: Tuple[float, float],
        dec_range: Tuple[float, float],
        step_deg: float = SKYCELL_DEG / 2,
    ) -> np.ndarray:
    """(N, 2) RA/Dec sample points covering a box, spaced finer than a skycell so none is missed.

    RA follows `coord_range_from_wcs`: a box straddling RA=0 is (lo, hi + 360). The RA step is
    widened by 1/cos(dec) so the spacing on the sky stays below `step_deg`.
    """
    ra_lo, ra_hi = ra_range
    dec_lo, dec_hi = dec_range
    max_cos = max(np.cos(np.radians(max(abs(dec_lo), abs(dec_hi)))), 1e-3)
    decs = np.unique(np.append(np.arange(dec_lo, dec_hi, step_deg), dec_hi))
    ras = np.unique(np.append(np.arange(ra_lo, ra_hi, step_deg / max_cos), ra_hi)) % 360.0
    grid = np.array([(ra, dec) for dec in decs for ra in ras])
    return grid


def query_filenames(positions: np.ndarray, bands: Sequence[str] = ('g', 'r', 'i'), product: str = 'stack') -> Table:
    """One row per (position, band) from the PS1 filename service, batched in a single POST.

    Columns: projcell, subcell, ra, dec, filter, mjd, type, filename, shortname, badflag.
    """
    type_name = 'stack' if product == 'stack' else f'stack.{product}'
    body = 'ra dec\n' + '\n'.join(f'{ra:.6f} {dec:.6f}' for ra, dec in positions) + '\n'
    resp = requests.post(
        FILENAME_SERVICE,
        data=dict(filters=''.join(bands), type=type_name),
        files=dict(file=io.StringIO(body)),
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return Table.read(resp.text, format='ascii')


def skycells_covering(
        ra_range: Tuple[float, float],
        dec_range: Tuple[float, float],
        bands: Sequence[str] = ('g', 'r', 'i'),
        step_deg: float = SKYCELL_DEG / 2,
    ) -> Table:
    """Unique (projcell, subcell, band) skycells whose stacks cover the box, with the archive filename.

    The service returns the single best skycell per position; the 60" overlaps between skycells
    mean sampling finer than a skycell covers the box with no gaps.
    """
    tab = query_filenames(footprint_grid(ra_range, dec_range, step_deg), bands=bands, product='stack')
    tab = tab[tab['badflag'] == 0]
    tab = unique(tab, keys=['projcell', 'subcell', 'filter'])
    tab.rename_column('filter', 'band')
    tab.sort(['projcell', 'subcell', 'band'])
    return tab['projcell', 'subcell', 'band', 'filename']


def download_file(url: str, dest: str, n_attempts: int = 5, overwrite: bool = False) -> Optional[str]:
    """Stream `url` to `dest` with backoff; None on 404 or after `n_attempts`. FITS files are validated."""
    is_fits = dest.endswith('.fits')
    if os.path.exists(dest) and not overwrite:
        if not is_fits or fitscheck_valid(dest):
            return dest
        os.remove(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + '.part'
    for attempt in range(n_attempts):
        try:
            resp = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 404:
                print(f'DOWNLOAD_FAIL HTTP 404: {url}')
                return None
            if resp.status_code != 200:
                print(f'DOWNLOAD_RETRY HTTP {resp.status_code} (attempt {attempt + 1}/{n_attempts}): {url}')
            else:
                with open(part, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                if not is_fits or fitscheck_valid(part):
                    os.replace(part, dest)
                    return dest
                print(f'DOWNLOAD_RETRY invalid file (attempt {attempt + 1}/{n_attempts}): {url}')
        except requests.exceptions.RequestException as e:
            print(f'DOWNLOAD_RETRY {type(e).__name__} (attempt {attempt + 1}/{n_attempts}): {url}')
        if os.path.exists(part):
            os.remove(part)
        if attempt < n_attempts - 1:
            time.sleep(min(15 * 2 ** attempt, 300) * random.uniform(0.8, 1.2))
    print(f'DOWNLOAD_FAIL gave up after {n_attempts} attempts: {url}')
    return None


def download_skycell(
        projcell: int,
        subcell: int,
        band: str,
        data_dir: str,
        products: Iterable[str] = DEFAULT_PRODUCTS,
        overwrite: bool = False,
    ) -> Dict[str, Optional[str]]:
    """Fetch the requested products of one skycell and band. Returns {product: local path or None}."""
    return {
        p: download_file(skycell_url(projcell, subcell, band, p), skycell_path(data_dir, projcell, subcell, band, p), overwrite=overwrite)
        for p in products
    }


def download_footprint(
        ra_range: Tuple[float, float],
        dec_range: Tuple[float, float],
        data_dir: str,
        bands: Sequence[str] = ('g', 'r', 'i'),
        products: Iterable[str] = DEFAULT_PRODUCTS,
        n_threads: int = MAX_THREADS,
        overwrite: bool = False,
    ) -> Table:
    """Download every skycell product covering a box, e.g. a ZTF quadrant's `coord_range_from_wcs`.

    Returns `skycells_covering` with one path column per product (None where the archive has no file).
    """
    cells = skycells_covering(ra_range, dec_range, bands=bands)
    products = tuple(products)
    jobs = [(int(r['projcell']), int(r['subcell']), str(r['band'])) for r in cells]
    print(f'{len(cells)} skycell-bands cover RA {ra_range}, Dec {dec_range}; fetching {products} with {n_threads} threads')
    with ThreadPoolExecutor(max_workers=min(n_threads, 10)) as pool:
        results = list(pool.map(lambda j: download_skycell(*j, data_dir=data_dir, products=products, overwrite=overwrite), jobs))
    for p in products:
        cells[f'{p}_path'] = [r[p] if r[p] is not None else '' for r in results]
    return cells
