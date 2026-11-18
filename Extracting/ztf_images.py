"""Download ZTF deep reference-image products from IRSA for one quadrant and band.

Per quadrant and band IRSA serves the reference image (`refimg`), its per-pixel 1-sigma
uncertainty in DN (`refunc`), the number of input frames per pixel (`refcov`), ZTF's own DAOPhot
PSF catalog (`refpsfcat`) and SExtractor catalog (`refsexcat`), plus logs. The original pipeline
fetched only `refimg`; the loader now uses `refunc` as the variance map and `refcov` to mask thin
edges, which puts ZTF on the same per-pixel noise model as PS1.
"""
import os
import random
import time
from typing import Dict, Iterable, Optional, Tuple

import requests
from astropy.io import fits

from .utils import fitscheck_valid, get_credentials

IBE_DEEP = 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/deep'
PRODUCT_SUFFIX = {
    'refimg': 'refimg.fits', 'refunc': 'refunc.fits', 'refcov': 'refcov.fits',
    'refpsfcat': 'refpsfcat.fits', 'refsexcat': 'refsexcat.fits', 'refimlog': 'refimlog.txt',
}
DEFAULT_PRODUCTS = ('refimg', 'refunc', 'refcov')
REQUEST_TIMEOUT = (10, 120)   # (connect, read) seconds
THROTTLE_CODES = (429, 503)   # IRSA answers these when a client asks faster than ~1 file per 2 s


def refimg_name(field: int, ccd: int, qid: int, band: str) -> str:
    return f'ztf_{int(field):06d}_z{band}_c{int(ccd):02d}_q{int(qid)}'


def refimg_url(field: int, ccd: int, qid: int, band: str, product: str = 'refimg') -> str:
    """IBE URL; the directory prefix is the first three digits of the zero-padded field."""
    pf = f'{int(field):06d}'
    return f'{IBE_DEEP}/{pf[:3]}/field{pf}/z{band}/ccd{int(ccd):02d}/q{int(qid)}/{refimg_name(field, ccd, qid, band)}_{PRODUCT_SUFFIX[product]}'


def product_dir(ztf_dir: str, field: int) -> str:
    """Products are stored one directory per field: `<ztf_dir>/field000643/`."""
    return os.path.join(ztf_dir, f'field{int(field):06d}')


def refimg_path(ztf_dir: str, field: int, ccd: int, qid: int, band: str, product: str = 'refimg') -> str:
    """`<ztf_dir>/field000643/ztf_000643_zr_c01_q1_refimg.fits`."""
    return os.path.join(product_dir(ztf_dir, field), f'{refimg_name(field, ccd, qid, band)}_{PRODUCT_SUFFIX[product]}')


def fits_table_valid(path: str) -> bool:
    """A complete FITS file whose data can be read (catches truncation): a table HDU for catalogs, else the image."""
    try:
        with fits.open(path, memmap=False) as hdul:
            if any(not h.is_image for h in hdul[1:]):
                return any(not h.is_image and h.data is not None for h in hdul[1:])
            return hdul[0].data is not None and hdul[0].data.size > 0
    except Exception:
        return False


def fetch_product(session: requests.Session, url: str, dest: str, max_attempts: int = 30,
                  timeout: Tuple[float, float] = (10.0, 120.0)) -> str:
    """One product over a keep-alive session; returns 'ok', 'cached', '404' or 'fail'.

    Reusing the connection is 1.5x faster than one request per file against IRSA's per-client
    throttle. Throttle responses back off from 3 s; other failures from 10 s. Files land under a
    `.part` name and are renamed only once the FITS table reads back.
    """
    if os.path.exists(dest):
        return 'cached'
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = f'{dest}.part{os.getpid()}'
    for attempt in range(max_attempts):
        wait = 10.0
        try:
            resp = session.get(url, stream=True, timeout=timeout)
            if resp.status_code == 404:
                return '404'
            if resp.status_code == 200:
                with open(part, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                if not dest.endswith('.fits') or fits_table_valid(part):
                    os.replace(part, dest)
                    return 'ok'
            elif resp.status_code in THROTTLE_CODES:
                wait = 3.0
            resp.close()
        except requests.exceptions.RequestException:
            pass
        if os.path.exists(part):
            os.remove(part)
        time.sleep(min(wait * 1.5 ** attempt, 60.0) * random.uniform(0.8, 1.2))
    return 'fail'


def download_file(url: str, dest: str, n_attempts: int = 5, overwrite: bool = False,
                  auth: Optional[Tuple[str, str]] = None) -> Optional[str]:
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
            resp = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT, auth=auth)
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


def irsa_auth() -> Tuple[str, str]:
    user, password = get_credentials('irsa_login.txt')
    return str(user), str(password)


def download_refimg(field: int, ccd: int, qid: int, band: str, data_dir: str,
                    products: Iterable[str] = DEFAULT_PRODUCTS, overwrite: bool = False,
                    auth: Optional[Tuple[str, str]] = None) -> Dict[str, Optional[str]]:
    """Fetch the requested products of one quadrant and band. Returns {product: local path or None}."""
    auth = auth or irsa_auth()
    return {
        p: download_file(refimg_url(field, ccd, qid, band, p), refimg_path(data_dir, field, ccd, qid, band, p),
                         overwrite=overwrite, auth=auth)
        for p in products
    }
