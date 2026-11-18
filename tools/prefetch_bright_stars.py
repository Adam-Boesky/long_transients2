"""Cache the Gaia DR2 G < 16 stars of every field for the bright-star mask: `gaia_cache/<field>_bright_G16_dr2.parquet`.

The box is the union of the field's reference-image footprints (the r-band headers, any band where r is missing),
padded for the brightest stars' spikes, and the query is the one the filter uses when a field has no cache
(`filter_field.fetch_bright_stars`). Fields with a cache are skipped, so a rerun resumes.

    python tools/prefetch_bright_stars.py                 # the whole footprint, in manifest order
    python tools/prefetch_bright_stars.py --fields 345 370
"""
import argparse
import glob
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Extracting.utils import get_data_path  # noqa: E402
from Extracting.ztf_products import product_dir  # noqa: E402
from Source_Analysis.filter_field import BRIGHT_STAR_G_MAX, fetch_bright_stars, field_box  # noqa: E402

MANIFEST = Path(__file__).resolve().parents[1] / 'Extracting' / 'footprint' / 'irsa_deep_quadrants.csv'
PAD_DEG = 0.4                  # 3 R of a G = 4 star is ~0.35 deg; the mask reaches that far outside the images


def footprint_corners(field: int, ztf_dir: str) -> pd.DataFrame:
    """The sky corners of every reference image of the field (r band first; every band if r is sparse)."""
    paths = sorted(glob.glob(os.path.join(product_dir(ztf_dir, field), f'ztf_{field:06d}_zr_c??_q?_refimg.fits')))
    if len(paths) < 32:
        paths = sorted(glob.glob(os.path.join(product_dir(ztf_dir, field), f'ztf_{field:06d}_z?_c??_q?_refimg.fits')))
    corners = []
    for p in paths:
        h = fits.getheader(p)
        if 'RADECSYS' in h:
            h.setdefault('RADESYS', h['RADECSYS']); del h['RADECSYS']
        corners.append(WCS(h).calc_footprint(axes=(h['NAXIS1'], h['NAXIS2'])))
    if not corners:
        raise FileNotFoundError(f'field {field}: no reference images under {product_dir(ztf_dir, field)}')
    xy = np.concatenate(corners)
    return pd.DataFrame({'ra': xy[:, 0], 'dec': xy[:, 1]})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fields', nargs='*', type=int, help='restrict to these field IDs')
    ap.add_argument('--data-dir', default=None, help='data directory holding ztf_data/ and gaia_cache/')
    a = ap.parse_args()
    warnings.filterwarnings('ignore')
    data_dir = a.data_dir or get_data_path()
    cache = os.path.join(data_dir, 'gaia_cache'); os.makedirs(cache, exist_ok=True)
    fields = a.fields or sorted(pd.read_csv(MANIFEST)['field'].unique())
    for field in fields:
        out = os.path.join(cache, f'{field:06d}_bright_G16_dr2.parquet')
        if os.path.exists(out):
            continue
        box = field_box(footprint_corners(field, os.path.join(data_dir, 'ztf_data')), pad_deg=PAD_DEG)
        stars = fetch_bright_stars(*box)
        stars.to_parquet(out)
        print(f'BRIGHT_OK field {field}: box {tuple(round(x, 2) for x in box)} -> {len(stars):,} stars G < {BRIGHT_STAR_G_MAX:g}', flush=True)
    print('BRIGHT_DONE', flush=True)


if __name__ == '__main__':
    main()
