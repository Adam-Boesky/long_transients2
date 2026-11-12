"""Download ZTF deep reference-image products from IRSA for one quadrant and band.

Per quadrant and band IRSA serves the reference image (`refimg`), its per-pixel 1-sigma
uncertainty in DN (`refunc`), the number of input frames per pixel (`refcov`), ZTF's own DAOPhot
PSF catalog (`refpsfcat`) and SExtractor catalog (`refsexcat`), plus logs. The original pipeline
fetched only `refimg`; the loader now uses `refunc` as the variance map and `refcov` to mask thin
edges, which puts ZTF on the same per-pixel noise model as PS1.
"""
import os
from typing import Dict, Iterable, Optional, Tuple

from .ps1_images import download_file
from .utils import get_credentials

IBE_DEEP = 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/deep'
PRODUCT_SUFFIX = {
    'refimg': 'refimg.fits', 'refunc': 'refunc.fits', 'refcov': 'refcov.fits',
    'refpsfcat': 'refpsfcat.fits', 'refsexcat': 'refsexcat.fits', 'refimlog': 'refimlog.txt',
}
DEFAULT_PRODUCTS = ('refimg', 'refunc', 'refcov')


def refimg_name(field: int, ccd: int, qid: int, band: str) -> str:
    return f'ztf_{int(field):06d}_z{band}_c{int(ccd):02d}_q{int(qid)}'


def refimg_url(field: int, ccd: int, qid: int, band: str, product: str = 'refimg') -> str:
    """IBE URL; the directory prefix is the first three digits of the zero-padded field."""
    pf = f'{int(field):06d}'
    return f'{IBE_DEEP}/{pf[:3]}/field{pf}/z{band}/ccd{int(ccd):02d}/q{int(qid)}/{refimg_name(field, ccd, qid, band)}_{PRODUCT_SUFFIX[product]}'


def refimg_path(data_dir: str, field: int, ccd: int, qid: int, band: str, product: str = 'refimg') -> str:
    return os.path.join(data_dir, f'{refimg_name(field, ccd, qid, band)}_{PRODUCT_SUFFIX[product]}')


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
