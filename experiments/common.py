"""Shared pieces of the experiments: running the filter, approximate 90% recovery limits, and ZTF | PS1 stamps.

Everything takes a field id, reads the field's merged table through the filter, and writes nothing but what the
caller saves. Stamps follow the review notebook's convention: north up, red + the object, white ○ ZTF objects,
orange □ PS1 objects, the survey's magnitude top right; PS1 cutouts are cached under the field's review/ps1_stamps.
"""
import glob
import os
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D, NoOverlapError
from astropy.visualization import simple_norm
from astropy.wcs import WCS
import astropy.units as u
from scipy.spatial import cKDTree

from Extracting.images import load_refimg
from Extracting.utils import get_data_path, sky_xyz
from Extracting.ztf_products import refimg_path
from Source_Analysis import filter_field as ff
from source_db.cutouts.panstarrs import _post as ps1_filenames
from source_db.cutouts.render import DEFAULT_PERCENT, DEFAULT_STRETCH, _get_fits, ps1_cutout_url, ps1_linear_flux

B3 = 'gri'


def experiments_dir(field: int, data_dir: Optional[str] = None) -> str:
    """Data/fields/<field>/review/experiments/, created if missing (outside git)."""
    path = os.path.join(data_dir or get_data_path(), 'fields', f'{int(field):06d}', 'review', 'experiments')
    os.makedirs(path, exist_ok=True)
    return path


# ----------------------------------------------------------------------------------------------- filtering

def prepare(field: int, data_dir: Optional[str] = None) -> pd.DataFrame:
    """The field's prepared table (`ff.prepare_field`), the input every filter run shares."""
    return ff.prepare_field(field, data_dir or get_data_path())


def candidates(base: pd.DataFrame, cuts: ff.Cuts, m90: Optional['M90'] = None):
    """Run the filter on a prepared table. Returns (src, {catalog name: candidate row positions}). With `m90`, the
    |sigma| checks are redone with 90% recovery limits on every object passing its catalog's non-sigma checks."""
    src, env, cands, _ = ff.filter_prepared(base, cuts, verbose=False)
    if m90 is None:
        return src, {name: src.index.get_indexer(c.index) for name, c in cands.items()}
    out = {}
    for k, name in ff.CATALOG.items():
        checks = ff.real_checks(src, k, cuts) + ff.population_checks(src)
        rows = np.flatnonzero((src.catalog.values == k) & np.logical_and.reduce([np.asarray(ok, bool) for _, ok in checks]))
        new = m90.apply(src, src.iloc[rows], env)
        out[name] = rows[np.logical_and.reduce([np.asarray(ok, bool) for _, ok in ff.interesting_checks(new)])]
    return src, out


# ----------------------------------------------------------------------------------------------- 90% recovery limits

class M90:
    """Approximate 90% recovery limits (plan_completeness.md):
      ZTF: per quadrant and band, the refpsfcat S/N = 5 magnitude widened by eta = GPCTSIG / MEDPUNC:
           m90 = m5 - 2.5 log10(1 + 1.2816 eta / 5);
      PS1: per stack, the injection-recovery 90% point from a StackDetEffMeta table (columns projectionID, skyCellID,
           filterID, m90), the object's band skycell, else the nearest PS1 row's."""

    def __init__(self, field: int, ps1_deteff: str, data_dir: Optional[str] = None):
        data_dir = data_dir or get_data_path()
        self.ztf = {}
        for f in glob.glob(os.path.join(data_dir, 'ztf_data', f'field{int(field):06d}', '*_refpsfcat.fits')):
            parts = os.path.basename(f).split('_')
            h = fits.getheader(f.replace('refpsfcat', 'refimg')); d = fits.getdata(f, 1)
            m5 = np.median(d['mag'][(d['snr'] >= 4.5) & (d['snr'] <= 5.5)] + h['MAGZP'])
            self.ztf[(parts[2][1], int(parts[3][1:]), int(parts[4][1:]))] = m5 - 2.5 * np.log10(1 + 1.2816 * (h['GPCTSIG'] / h['MEDPUNC']) / 5)
        st = pd.read_parquet(ps1_deteff)
        st['band'] = st.filterID.astype(int).map({1: 'g', 2: 'r', 3: 'i'})
        self.ps1 = {(r.band, int(r.projectionID), int(r.skyCellID)): r.m90 for r in st.itertuples()}

    def apply(self, src: pd.DataFrame, sub: pd.DataFrame, envelopes) -> pd.DataFrame:
        """`sub` with both limits replaced, the sub-threshold-measurement flags redone (as `ff.assign_catalogs`) and
        sigma recomputed (`ff.annotate_sigma`)."""
        out = sub.copy()
        for b in B3:
            out[ff.limit_col('ztf', b)] = [self.ztf.get((b, int(c), int(q)), np.nan) if c >= 0 else np.nan
                                     for c, q in zip(sub[f'{b}_ccd'], sub[f'{b}_qid'])]
            has = src[f'{b}_ps1_projcell'].values > 0
            cells = src.loc[has, [f'{b}_ps1_projcell', f'{b}_ps1_skycell']].values
            _, j = cKDTree(sky_xyz(src.ps1_ra.values[has], src.ps1_dec.values[has])).query(sky_xyz(sub.ra.values, sub.dec.values))
            own = sub[f'{b}_ps1_projcell'].values > 0
            pc = np.where(own, sub[f'{b}_ps1_projcell'].values, cells[j, 0]); sc = np.where(own, sub[f'{b}_ps1_skycell'].values, cells[j, 1])
            out[ff.limit_col('ps1', b)] = [self.ps1.get((b, int(p), int(s)), np.nan) for p, s in zip(pc, sc)]
            cat = sub[f'{b}_cat'].values
            z_mag, p_mag = ff.band_photometry(sub, b, 'ztf')[0], ff.band_photometry(sub, b, 'ps1')[0]
            out[f'{b}_ps1_use_forced'] = (cat == 1) & (sub[f'{b}_ps1_status'].values == 'L') & np.isfinite(p_mag) & (p_mag < out[ff.limit_col('ps1', b)].values)
            out[f'{b}_ztf_use_meas'] = (cat == 2) & (sub[f'{b}_ztf_status'].values == 'L') & np.isfinite(z_mag) & (z_mag < out[ff.limit_col('ztf', b)].values)
        return ff.annotate_sigma(out, envelopes)


# ----------------------------------------------------------------------------------------------- stamps

class Stamps:
    """ZTF reference | PS1 stack cutouts of a field's objects."""

    def __init__(self, field: int, src: pd.DataFrame, data_dir: Optional[str] = None):
        self.field, self.src = int(field), src
        self.data_dir = data_dir or get_data_path()
        self.cache = os.path.join(self.data_dir, 'fields', f'{self.field:06d}', 'review', 'ps1_stamps')
        os.makedirs(self.cache, exist_ok=True)
        self._refs: Dict = OrderedDict()
        self.objs = {'ZTF': (src.ra.values[src.ztf_idx.values >= 0], src.dec.values[src.ztf_idx.values >= 0]),
                     'PS1': (src.ps1_ra.values[src.ps1_idx.values >= 0], src.ps1_dec.values[src.ps1_idx.values >= 0])}
        self.trees = {k: cKDTree(sky_xyz(*v)) for k, v in self.objs.items()}

    def ps1_files(self, rows: pd.DataFrame):
        """PS1 stack filenames for these rows (one query for all)."""
        return ps1_filenames(rows[['ra', 'dec']]).round({'ra': 6, 'dec': 6}).groupby(['ra', 'dec', 'filter']).filename.first()

    def _ref(self, ccd: int, qid: int, b: str):
        key = (ccd, qid, b)
        if key not in self._refs:
            if len(self._refs) >= 12:
                self._refs.popitem(last=False)
            self._refs[key] = load_refimg(refimg_path(os.path.join(self.data_dir, 'ztf_data'), self.field, *key))
        return self._refs[key]

    def _ps1(self, r, b, files, size):
        path = os.path.join(self.cache, f'{r.ra:.6f}_{r.dec:+.6f}_{b}_{size}.fits')
        if not os.path.exists(path):
            name = files.get((round(r.ra, 6), round(r.dec, 6), b))
            try:
                raw = _get_fits(ps1_cutout_url(r.ra, r.dec, name, int(size / 0.25))) if name else None
            except Exception as err:                          # a failed download leaves the panel blank, not the run
                print(f'PS1 cutout {r.ra:.5f} {r.dec:+.5f} {b}: {type(err).__name__}')
                raw = None
            if not raw:
                return None, None
            with open(path + '.part', 'wb') as f:
                f.write(raw)
            os.replace(path + '.part', path)
        hdu = fits.open(path)[0]
        return ps1_linear_flux(hdu.data.astype(np.float32), hdu.header), WCS(hdu.header)

    def _draw(self, ax, data, wcs, r, b, survey, size, star=False):
        ax.set_xticks([]); ax.set_yticks([])
        if data is None or data.size == 0 or not np.isfinite(data).any():
            ax.text(0.5, 0.5, 'no image', ha='center', va='center', transform=ax.transAxes)
            return
        flip = wcs.pixel_scale_matrix[0, 0] > 0
        data = data[::-1, ::-1] if flip else data

        def pix(ra, dec):
            x, y = wcs.world_to_pixel_values(np.asarray(ra), np.asarray(dec))
            return (data.shape[1] - 1 - x, data.shape[0] - 1 - y) if flip else (x, y)

        ax.imshow(data, origin='lower', cmap='viridis', norm=simple_norm(data, DEFAULT_STRETCH, percent=DEFAULT_PERCENT))
        centre = sky_xyz([r.ra], [r.dec])[0]
        reach = 2 * np.sin(np.radians(size * 0.75 / 3600) / 2)
        for (k, (ra, dec)), (col, mk, s) in zip(self.objs.items(), (('white', 'o', 90), ('orange', 's', 50))):
            hit = self.trees[k].query_ball_point(centre, reach)
            x, y = pix(ra[hit], dec[hit])
            ax.scatter(x, y, s=s, marker=mk, facecolors='none', edgecolors=col, lw=1.1)
        x, y = pix([r.ra], [r.dec])
        ax.plot(x, y, '+', color='red', ms=14, mew=1.6)
        ax.set_xlim(-0.5, data.shape[1] - 0.5); ax.set_ylim(-0.5, data.shape[0] - 0.5)
        m = r[f'{b}_ztf_psf'] if survey == 'ZTF' else r[f'{b}_ps1_psf']
        ax.text(0.97, 0.97, f'{survey} {b} ' + (f'{m:.1f}' if np.isfinite(m) else '–'), color='red', fontsize=11,
                ha='right', va='top', transform=ax.transAxes)
        if star:
            ax.text(0.97, 0.03, '★', color='red', fontsize=24, ha='right', va='bottom', transform=ax.transAxes)

    def pair(self, ax_z, ax_p, r, b: str, files, size: int = 30, star: bool = False):
        """The ZTF reference cutout in `ax_z` and the PS1 stack cutout in `ax_p`, band b, `size` arcsec; `star` puts a
        red ★ in the ZTF panel."""
        try:
            im = self._ref(int(r[f'{b}_ccd']), int(r[f'{b}_qid']), b)
            cut = Cutout2D(im.data, SkyCoord(r.ra * u.deg, r.dec * u.deg), size * u.arcsec, wcs=im.wcs, mode='partial')
            self._draw(ax_z, cut.data, cut.wcs, r, b, 'ZTF', size, star)
        except (NoOverlapError, FileNotFoundError, ValueError):
            self._draw(ax_z, None, None, r, b, 'ZTF', size, star)
        self._draw(ax_p, *self._ps1(r, b, files, size), r, b, 'PS1', size)
        ax_p.set_xlabel(f'{r.ra:.4f} {r.dec:+.4f}   ' + '/'.join(f"{r[f'{x}_ztf_status']}{r[f'{x}_ps1_status']}" for x in B3)
                        + f'   band {b}', fontsize=9.5)


# ----------------------------------------------------------------------------------------------- figures and tables

KEY_COLS = ['ra', 'dec', 'catalog', 'point_source', 'nearest_other_arcsec', 'inside_brighter_kron'] + \
           [f'{b}_{c}' for b in B3 for c in ('ztf_status', 'ps1_status', 'ztf_psf', 'ps1_psf', 'ztf_chi', 'ztf_sharp', 'sigma')]


def largest_sigma_band(r) -> str:
    """The compared band with the largest |sigma|, else r."""
    sig = {b: abs(r[f'{b}_sigma']) for b in B3 if r[f'{b}_cat'] >= 0 and np.isfinite(r[f'{b}_sigma'])}
    return max(sig, key=sig.get) if sig else 'r'


def stamp_figure(stamps: 'Stamps', src: pd.DataFrame, rows: np.ndarray, path: str, title: str, n: int = 10,
                 band_of: Optional[Dict[int, str]] = None, seed: int = 0, starred: Optional[np.ndarray] = None,
                 others: Optional[np.ndarray] = None, captions=None):
    """ZTF | PS1 stamps of up to `n` random `rows` (positions in `src`), five objects per figure row, saved to `path`.
    Each object is shown in `band_of[row]` if given, else its largest-|sigma| band; the ZTF caption gives chi and
    sigma per compared band, the nearest other entry, morphology and Kron containment. With `others`, up to `n` of
    those follow the `rows` ones; rows in `starred` get a red ★. `captions(row)`, if given, replaces the ZTF caption."""
    import matplotlib.pyplot as plt
    rows = np.asarray(rows, int)
    others = np.asarray(others if others is not None else [], int)
    rng = np.random.default_rng(seed)
    take = lambda r: np.sort(rng.choice(r, min(n, len(r)), replace=False)) if len(r) else np.array([], int)
    pick = np.r_[take(rows), take(others)].astype(int)
    if not len(pick):
        return
    starred = set(np.asarray(starred if starred is not None else [], int).tolist())
    sub = src.iloc[pick]
    files = stamps.ps1_files(sub)
    nrow = int(np.ceil(len(pick) / 5))
    fig, axes = plt.subplots(nrow, 10, figsize=(32, 4.2 * nrow), squeeze=False)
    for k, (i, (_, r)) in enumerate(zip(pick, sub.iterrows())):
        b = (band_of or {}).get(int(i)) or largest_sigma_band(r)
        row, col = divmod(k, 5)
        ax_z, ax_p = axes[row, 2 * col], axes[row, 2 * col + 1]
        stamps.pair(ax_z, ax_p, r, b, files, star=int(i) in starred)
        if captions is not None:
            ax_z.set_xlabel(captions(int(i)), fontsize=9)
        else:
            per_band = '  '.join(f"{x} χ{r[f'{x}_ztf_chi']:.1f} σ{r[f'{x}_sigma']:+.1f}" for x in B3 if r[f'{x}_cat'] >= 0)
            ax_z.set_xlabel(f'{per_band}\nnearest other {r.nearest_other_arcsec:.1f}″  ' + ('point' if r.point_source else 'extended')
                            + ('  Kron-contained' if r.inside_brighter_kron else ''), fontsize=9.5)
    for k in range(len(pick), nrow * 5):
        row, col = divmod(k, 5)
        axes[row, 2 * col].axis('off'); axes[row, 2 * col + 1].axis('off')
    shown = f'{len(rows)}; {len(pick)} shown' if not len(others) else f'★ {len(rows)}, others {len(others)}; {len(pick)} shown'
    axes[0, 0].set_ylabel(f'{title}\n({shown})\nZTF', fontsize=13)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
