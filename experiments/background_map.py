"""A ZTF reference quadrant next to our background model of it (the 64″ sep mesh `kron.background`).

Left: the reference image with the cutouts' usual stretch. Right: the background model's excess over its median on
the unmasked pixels, in DN (blue below, red above), which is what `field_tables.residual_sky` reads at each source.
Both panels carry the contour where that excess reaches `--sky` Jy/arcsec² (default 1e-7, PS1's sky threshold) and
mark `--at` with a cross. Pixel coordinates; masked (NaN) pixels blank.

  python experiments/background_map.py 860 r --at 169.23414 75.28806
  python experiments/background_map.py 860 i --quadrant 10 2 --sky 3e-7
Writes background_<band>_c<cc>_q<q>.png to Data/fields/<field>/review/experiments/.
"""
import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.visualization import simple_norm
from matplotlib.colors import TwoSlopeNorm

from experiments.common import experiments_dir
from Extracting import kron
from Extracting.field_tables import field_dir
from Extracting.images import load_refimg
from Extracting.utils import get_data_path, sky_xyz
from Extracting.ztf_products import refimg_path
from source_db.cutouts.render import DEFAULT_PERCENT, DEFAULT_STRETCH
from scipy.spatial import cKDTree


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('field', type=int)
    p.add_argument('band', choices=list('gri'))
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument('--at', nargs=2, type=float, metavar=('RA', 'DEC'), help='the quadrant holding this position')
    where.add_argument('--quadrant', nargs=2, type=int, metavar=('CCD', 'QID'))
    p.add_argument('--sky', type=float, default=1e-7, help='contour level, Jy/arcsec^2')
    a = p.parse_args()
    data_dir = get_data_path()

    if a.at:                                              # the quadrant the band is judged in at that position (merged's {b}_ccd/qid)
        m = pd.read_parquet(os.path.join(field_dir(a.field, data_dir), 'merged.parquet'), columns=['ra', 'dec', f'{a.band}_ccd', f'{a.band}_qid'])
        _, i = cKDTree(sky_xyz(m.ra.values, m.dec.values)).query(sky_xyz([a.at[0]], [a.at[1]])[0])
        ccd, qid = int(m[f'{a.band}_ccd'].values[i]), int(m[f'{a.band}_qid'].values[i])
        if ccd < 0:
            sys.exit(f'no {a.band} reference image covers {a.at}')
    else:
        ccd, qid = a.quadrant
    ztf_dir = os.path.join(data_dir, 'ztf_data')
    im = load_refimg(refimg_path(ztf_dir, a.field, ccd, qid, a.band))
    back = kron.background(im).back()
    excess = back - np.median(back[~im.bad])
    level_dn = a.sky * im.pixel_scale ** 2 / (3631.0 * 10 ** (-0.4 * im.zero_point))      # Jy/arcsec^2 -> DN per pixel

    img = np.where(im.bad, np.nan, im.data)
    exc = np.where(im.bad, np.nan, excess)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10.5))
    axes[0].imshow(img, origin='lower', cmap='viridis', norm=simple_norm(img[np.isfinite(img)], DEFAULT_STRETCH, percent=DEFAULT_PERCENT))
    lim = max(np.nanpercentile(np.abs(exc), 99.5), 1.5 * level_dn)
    shown = axes[1].imshow(exc, origin='lower', cmap='RdBu_r', norm=TwoSlopeNorm(0.0, -lim, lim))
    cb = fig.colorbar(shown, ax=axes[1], fraction=0.046, pad=0.02)
    cb.set_label('background model − its median  [DN per pixel]', fontsize=12)
    for ax in axes:
        ax.contour(np.nan_to_num(excess, nan=0.0) >= level_dn, levels=[0.5], colors='k', linewidths=1.0, origin='lower')
        if a.at:
            x, y = im.wcs.world_to_pixel_values(a.at[0], a.at[1])
            ax.plot(x, y, 'x', color='magenta', ms=16, mew=2.5)
        ax.set_xlabel('x [px]', fontsize=12); ax.set_ylabel('y [px]', fontsize=12)
    frac = float(np.mean(excess[~im.bad] >= level_dn))
    axes[0].set_xlabel(f'x [px]   ZTF {a.band} reference, field {a.field} c{ccd:02d} q{qid}', fontsize=12)
    axes[1].set_xlabel(f'x [px]   black: residual sky ≥ {a.sky:g} Jy/arcsec² (= {level_dn:.2f} DN; {100 * frac:.1f}% of the image)', fontsize=12)
    plt.tight_layout()
    out = os.path.join(experiments_dir(a.field), f'background_{a.band}_c{ccd:02d}_q{qid}.png')
    fig.savefig(out, dpi=110)
    print(f'{a.sky:g} Jy/arcsec^2 = {level_dn:.2f} DN; above it: {100 * frac:.1f}% of the unmasked pixels; written {out}')


if __name__ == '__main__':
    main()
