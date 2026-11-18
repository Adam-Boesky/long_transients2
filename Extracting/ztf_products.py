"""ZTF deep reference-image products on IRSA: names, URLs, the per-field storage layout and the fetcher.

Per quadrant and band IRSA serves the reference image (`refimg`), its per-pixel 1-sigma uncertainty in DN
(`refunc`), the number of input frames per pixel (`refcov`), ZTF's DAOPhot PSF catalog (`refpsfcat`) and
SExtractor catalog (`refsexcat`), plus logs. Products are stored one directory per field,
`<ztf_dir>/field000643/`; `tools/fetch_ztf_products.py` fills it with `fetch_product`.
"""
import os
import random
import time
from typing import Tuple

import requests
from astropy.io import fits


IBE_DEEP = 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/deep'
PRODUCT_SUFFIX = {
    'refimg': 'refimg.fits', 'refunc': 'refunc.fits', 'refcov': 'refcov.fits',
    'refpsfcat': 'refpsfcat.fits', 'refsexcat': 'refsexcat.fits', 'refimlog': 'refimlog.txt',
}
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


def fits_valid(path: str) -> bool:
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
                if not dest.endswith('.fits') or fits_valid(part):
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
