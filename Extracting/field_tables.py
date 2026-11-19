"""Per-field source tables from the per-quadrant ZTF reference products and the PS1 catalog.

Order, per field:
  1. per band: each quadrant's DAOPhot catalog (`refpsfcat`) with our Kron photometry attached, concatenated over
     the quadrants, then deduplicated in the overlap strips (the 3200 px reference
     images are larger than the 3072 x 3080 px quadrants, so neighbouring images overlap by ~1-2');
  2. the bands merged into objects: every band pair matched by mutual nearest neighbour at MATCH_ARCSEC, the pairs
     linked into objects (no band order), each object at the S/N-weighted mean position of its detections;
  3. the PS1 stack catalog (`ps1_stack`: our mirror of MAST's stack table, cleaned to one row per object) for the
     partitions under the field, restricted to objects inside some reference image;
  4. objects matched to PS1 by mutual nearest neighbour at MATCH_ARCSEC.
Every step works on one homogeneous table; nothing downstream needs to know about quadrants.
"""
import argparse
import glob
import os
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from . import kron, ps1_stack
from .ps1_stack import BANDS
from .images import load_refimg
from .utils import get_data_path
from .ztf_products import product_dir, refimg_path

MATCH_ARCSEC = 1.0
REF_SIZE_PX = 3200                 # reference-image side (Explanatory Supplement section 7.2)
PSF_COLS = {'ra': 'ra', 'dec': 'dec', 'xpos': 'x', 'ypos': 'y', 'mag': 'mag', 'sigmag': 'magerr', 'snr': 'snr', 'chi': 'chi', 'sharp': 'sharp', 'flags': 'flags'}
KRON_COLS = {'mag': 'zkron', 'magerr': 'zkron_err', 'r1': 'zkron_r1', 'r_ap': 'zkron_rap', 'sky_radius': 'zkron_sky', 'flag': 'zkron_flag', 'masked_frac': 'zkron_masked'}
OBJECT_COLS = ('mag', 'magerr', 'snr', 'chi', 'sharp', 'flags', 'nan_dist_px', 'zkron', 'zkron_err', 'zkron_flag', 'zkron_masked')


def field_dir(field: int, data_dir: Optional[str] = None) -> str:
    return os.path.join(data_dir or get_data_path(), 'fields', f'{int(field):06d}')


def quadrant_name(ccd: int, qid: int) -> str:
    return f'c{int(ccd):02d}_q{int(qid)}'


def _optional(path: str) -> Optional[str]:
    """A product path, or None when the file was not downloaded (refunc/refcov are optional: without them the loader
    uses the gain model for the variance and the image's own NaNs for the mask)."""
    return path if os.path.exists(path) else None


def has_quadrant_band(field: int, ccd: int, qid: int, band: str, data_dir: Optional[str] = None) -> bool:
    """Whether the reference image and its DAOPhot catalog exist for this quadrant in this band."""
    ztf_dir = os.path.join(data_dir or get_data_path(), 'ztf_data')
    return all(os.path.exists(refimg_path(ztf_dir, field, ccd, qid, band, p)) for p in ('refimg', 'refpsfcat'))


def field_quadrants(field: int, data_dir: Optional[str] = None, band: str = 'r') -> list:
    """The (ccd, qid) of every quadrant with a reference image and DAOPhot catalog in `band` (r: the band whose
    footprint the PS1 step uses)."""
    ztf_dir = os.path.join(data_dir or get_data_path(), 'ztf_data')
    found = []
    for path in sorted(glob.glob(os.path.join(product_dir(ztf_dir, field), f'ztf_{int(field):06d}_z{band}_c??_q?_refimg.fits'))):
        name = os.path.basename(path)
        ccd, qid = int(name.split('_c')[1][:2]), int(name.split('_q')[1][0])
        if has_quadrant_band(field, ccd, qid, band, data_dir):
            found.append((ccd, qid))
    return found


# ----------------------------------------------------------------------------------------------- matching helpers

def _xyz(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    r, d = np.radians(ra), np.radians(dec)
    return np.c_[np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)]


def _chord(arcsec: float) -> float:
    return 2.0 * np.sin(np.radians(arcsec / 3600.0) / 2.0)


def mutual_nearest(ra1, dec1, ra2, dec2, radius_arcsec: float = MATCH_ARCSEC) -> Tuple[np.ndarray, np.ndarray]:
    """For each row of set 1, the index of its nearest row in set 2 within the radius if that row's nearest in set 1
    is it (else -1), and the separation in arcsec."""
    ra1, dec1, ra2, dec2 = (np.asarray(v, float) for v in (ra1, dec1, ra2, dec2))
    idx = np.full(len(ra1), -1); sep = np.full(len(ra1), np.nan)
    if len(ra1) == 0 or len(ra2) == 0:
        return idx, sep
    p1, p2 = _xyz(ra1, dec1), _xyz(ra2, dec2)
    d12, j12 = cKDTree(p2).query(p1, distance_upper_bound=_chord(radius_arcsec))
    d21, j21 = cKDTree(p1).query(p2, distance_upper_bound=_chord(radius_arcsec))
    ok = np.isfinite(d12)
    ok[ok] &= j21[j12[ok]] == np.flatnonzero(ok)
    idx[ok] = j12[ok]
    sep[ok] = np.degrees(2.0 * np.arcsin(d12[ok] / 2.0)) * 3600.0
    return idx, sep


# ----------------------------------------------------------------------------------------------- per band, per quadrant

def band_catalog(field: int, ccd: int, qid: int, band: str, data_dir: Optional[str] = None, with_kron: bool = True,
                 overwrite: bool = False, require: Sequence[str] = ()) -> pd.DataFrame:
    """One quadrant's DAOPhot catalog in one band, with our Kron photometry attached; cached under the field
    directory. `require` names optional products (refunc, refcov) that must exist."""
    data_dir = data_dir or get_data_path()
    out_dir = os.path.join(field_dir(field, data_dir), 'quadrants')
    path = os.path.join(out_dir, f'ztf_{quadrant_name(ccd, qid)}_{band}.parquet')
    if os.path.exists(path) and not overwrite:
        return pd.read_parquet(path)
    ztf_dir = os.path.join(data_dir, 'ztf_data')
    for product in require:
        if not os.path.exists(refimg_path(ztf_dir, field, ccd, qid, band, product)):
            raise FileNotFoundError(f'required product missing: {refimg_path(ztf_dir, field, ccd, qid, band, product)}')
    with fits.open(refimg_path(ztf_dir, field, ccd, qid, band, 'refpsfcat')) as hdul:
        raw = Table(hdul[1].data)
        magzp = float(hdul[0].header['MAGZP'])
        hdul_nmatches = hdul[0].header.get('NMATCHES')
    cat = pd.DataFrame({new: np.asarray(raw[old], float) for old, new in PSF_COLS.items()})
    cat['flags'] = cat['flags'].astype(int)
    cat['mag'] += magzp                                   # refpsfcat magnitudes are instrumental; MAGZP calibrates them (Supplement section 7.3)
    cat['magzp'] = magzp
    cat['ccd'], cat['qid'] = int(ccd), int(qid)
    cat['edge_px'] = np.minimum.reduce([cat.x.values, cat.y.values, REF_SIZE_PX - cat.x.values, REF_SIZE_PX - cat.y.values])
    im = load_refimg(refimg_path(ztf_dir, field, ccd, qid, band), unc_path=_optional(refimg_path(ztf_dir, field, ccd, qid, band, 'refunc')),
                         cov_path=_optional(refimg_path(ztf_dir, field, ccd, qid, band, 'refcov')))
    if abs(im.zero_point - magzp) > 1e-6 or im.header.get('NMATCHES') != hdul_nmatches:   # the deep references are rebuilt on a rolling basis
        raise RuntimeError(f'{refimg_path(ztf_dir, field, ccd, qid, band, "refpsfcat")}: catalog and image come from different reference builds '
                           f'(MAGZP {magzp} vs {im.zero_point}, NMATCHES {hdul_nmatches} vs {im.header.get("NMATCHES")})')
    xi, yi = np.clip(np.round(cat.x.values - 1).astype(int), 0, im.shape[1] - 1), np.clip(np.round(cat.y.values - 1).astype(int), 0, im.shape[0] - 1)
    cat['nan_dist_px'] = distance_transform_edt(~im.bad)[yi, xi]              # distance to the nearest masked pixel
    if with_kron:
        frame = kron.prepare(im, cat.ra.values, cat.dec.values)
        kron.moment_floor(frame, np.nan_to_num(cat.snr.values, nan=0.0))
        res = kron.kron_photometry(frame)
        for src, col in KRON_COLS.items():
            cat[col] = res[src].values
        cat['zkron_floor_arcsec'] = frame.r_min * im.pixel_scale
        cat['maglim'] = im.mag_limit
    os.makedirs(out_dir, exist_ok=True)
    cat.to_parquet(path)
    return cat


# ----------------------------------------------------------------------------------------------- per band, per field

def dedupe_band(cat: pd.DataFrame, radius_arcsec: float = MATCH_ARCSEC) -> pd.DataFrame:
    """Drop the duplicate of every source that appears in two quadrants' overlap strips: of a mutual-nearest pair from
    different quadrants within the radius, keep the row farther from its image edge. Repeats for corner triples."""
    cat = cat.reset_index(drop=True)
    keep = np.ones(len(cat), bool)
    for _ in range(3):
        rows = np.flatnonzero(keep)
        sub = cat.iloc[rows]
        p = _xyz(sub.ra.values, sub.dec.values)
        tree = cKDTree(p)
        pairs = tree.query_pairs(_chord(radius_arcsec), output_type='ndarray')
        if len(pairs) == 0:
            break
        q = (sub.ccd.values * 10 + sub.qid.values)
        pairs = pairs[q[pairs[:, 0]] != q[pairs[:, 1]]]
        if len(pairs) == 0:
            break
        # mutual nearest only: keep each pair whose members are each other's closest cross-quadrant neighbour
        d = np.linalg.norm(p[pairs[:, 0]] - p[pairs[:, 1]], axis=1)
        order = np.argsort(d)
        seen = np.zeros(len(sub), bool); drop = []
        for a, b in pairs[order]:
            if seen[a] or seen[b]:
                continue
            seen[a] = seen[b] = True
            drop.append(b if sub.edge_px.values[a] >= sub.edge_px.values[b] else a)
        keep[rows[np.array(drop)]] = False
    out = cat[keep].reset_index(drop=True)
    out['n_duplicates_dropped'] = int((~keep).sum())
    return out


def _band_catalog_job(args):
    field, ccd, qid, band, data_dir, kw = args
    return band_catalog(field, ccd, qid, band, data_dir, **kw)


def field_band(field: int, quadrants: Iterable[Tuple[int, int]], band: str, data_dir: Optional[str] = None, n_jobs: int = 1, **kw) -> pd.DataFrame:
    """The deduplicated per-band catalog of a field; the per-quadrant catalogs are built in `n_jobs` processes.
    Quadrants without this band's products (bands do not share a footprint) are skipped."""
    jobs = [(field, ccd, qid, band, data_dir, kw) for ccd, qid in quadrants if has_quadrant_band(field, ccd, qid, band, data_dir)]
    if not jobs:
        raise FileNotFoundError(f'field {field}: no quadrant has {band}-band products')
    if n_jobs > 1:
        from multiprocessing import get_context
        with get_context('fork').Pool(n_jobs) as pool:
            cats = pool.map(_band_catalog_job, jobs)
    else:
        cats = [_band_catalog_job(j) for j in jobs]
    return dedupe_band(pd.concat(cats, ignore_index=True))


# ----------------------------------------------------------------------------------------------- objects

def merge_bands(cats: Dict[str, pd.DataFrame], radius_arcsec: float = MATCH_ARCSEC) -> pd.DataFrame:
    """ZTF objects from the band tables, independent of any band order: every pair of bands is matched by mutual
    nearest neighbour within the radius, the pairs are the edges of a graph whose connected components are the objects,
    and a component that holds two rows of one band (possible in a blend, through a third band) loses its longest edge
    until every object has at most one row per band. An object's position is the S/N-weighted mean of its detections
    (PS1's convention for its objects). Columns: ra, dec, and per band `{b}_idx` (row in the band
    table, -1 if absent), `{b}_sep` (the row's separation from the object position, arcsec) and copies of the main
    photometry columns."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    present = [b for b in BANDS if b in cats]
    offset = {}; n = 0
    for b in present:
        offset[b] = n; n += len(cats[b])
    ra = np.concatenate([cats[b].ra.values for b in present]); dec = np.concatenate([cats[b].dec.values for b in present])
    snr = np.concatenate([cats[b].snr.values for b in present]); band = np.concatenate([np.full(len(cats[b]), k) for k, b in enumerate(present)])
    edges = []                                                             # (node, node, separation) over every band pair
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            j, sep = mutual_nearest(cats[a].ra, cats[a].dec, cats[b].ra, cats[b].dec, radius_arcsec)
            k = np.flatnonzero(j >= 0)
            edges.append(np.c_[k + offset[a], j[k] + offset[b], sep[k]])
    e = np.concatenate(edges) if edges else np.zeros((0, 3))
    while True:
        g = coo_matrix((np.ones(len(e)), (e[:, 0].astype(int), e[:, 1].astype(int))), shape=(n, n))
        n_obj, comp = connected_components(g, directed=False)
        two = pd.DataFrame({'comp': comp, 'band': band}).duplicated().values     # a second row of some band in its component
        bad = np.unique(comp[two])
        if len(bad) == 0:
            break
        ec = comp[e[:, 0].astype(int)]
        drop = pd.Series(e[:, 2]).groupby(ec).idxmax().loc[bad].values         # the longest edge of each bad component
        e = np.delete(e, drop, axis=0)
    # one row per object, members by band
    w = np.where(np.isfinite(snr) & (snr > 0), snr, 1.0)
    xyz = _xyz(ra, dec) * w[:, None]
    sums = np.zeros((n_obj, 3)); np.add.at(sums, comp, xyz)
    norm = np.linalg.norm(sums, axis=1); sums /= norm[:, None]
    obj_ra = (np.degrees(np.arctan2(sums[:, 1], sums[:, 0])) + 360) % 360
    obj_dec = np.degrees(np.arcsin(np.clip(sums[:, 2], -1, 1)))
    obj = pd.DataFrame({'ra': obj_ra, 'dec': obj_dec})
    for k, b in enumerate(present):
        rows = np.flatnonzero(band == k)
        idx = np.full(n_obj, -1); idx[comp[rows]] = rows - offset[b]
        obj[f'{b}_idx'] = idx
        has = idx >= 0
        member = cats[b].iloc[np.clip(idx, 0, None)]
        sep = np.degrees(2 * np.arcsin(np.linalg.norm(_xyz(member.ra.values, member.dec.values) - _xyz(obj_ra, obj_dec), axis=1) / 2)) * 3600
        obj[f'{b}_sep'] = np.where(has, sep, np.nan)
        for col in OBJECT_COLS:
            if col in cats[b].columns:
                obj[f'{b}_{col}'] = np.where(has, cats[b][col].values[np.clip(idx, 0, None)], np.nan)
    for b in BANDS:
        if b not in cats:
            obj[f'{b}_idx'], obj[f'{b}_sep'] = -1, np.nan
    return obj


def quadrant_mask(field: int, ccd: int, qid: int, band: str, data_dir: Optional[str] = None):
    """(bad-pixel mask, WCS) of one quadrant's reference image in one band, as the loader defines bad (NaN, invalid
    refunc, thin coverage). Cached bit-packed beside the field tables, so later builds do not reload the three FITS files."""
    from astropy.wcs import WCS
    data_dir = data_dir or get_data_path()
    path = os.path.join(field_dir(field, data_dir), 'quadrants', f'mask_{quadrant_name(ccd, qid)}_{band}.npz')
    ztf_dir = os.path.join(data_dir, 'ztf_data')
    if not has_quadrant_band(field, ccd, qid, band, data_dir):                 # band not imaged here: all bad, r-band WCS
        bad, wcs = quadrant_mask(field, ccd, qid, 'r', data_dir)
        return np.ones_like(bad), wcs
    if os.path.exists(path):
        z = np.load(path)
        shape = tuple(z['shape'])
        bad = np.unpackbits(z['bad'], count=shape[0] * shape[1]).reshape(shape).astype(bool)
        header = fits.getheader(refimg_path(ztf_dir, field, ccd, qid, band))
        if 'RADECSYS' in header:
            header.setdefault('RADESYS', header['RADECSYS']); del header['RADECSYS']
        return bad, WCS(header)
    im = load_refimg(refimg_path(ztf_dir, field, ccd, qid, band), unc_path=_optional(refimg_path(ztf_dir, field, ccd, qid, band, 'refunc')),
                         cov_path=_optional(refimg_path(ztf_dir, field, ccd, qid, band, 'refcov')))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, bad=np.packbits(im.bad), shape=np.array(im.shape))
    return im.bad, im.wcs


def _near_image(wcs, shape: Tuple[int, int], ra: np.ndarray, dec: np.ndarray, pad_deg: float = 0.02) -> np.ndarray:
    """Indices of the positions inside an image's sky box (a cheap prefilter before projecting)."""
    fp = wcs.calc_footprint(axes=(shape[1], shape[0]))                  # image corners, deg
    ra0, ra1, dec0, dec1 = fp[:, 0].min() - pad_deg, fp[:, 0].max() + pad_deg, fp[:, 1].min() - pad_deg, fp[:, 1].max() + pad_deg
    near = (dec >= dec0) & (dec <= dec1)
    if ra1 - ra0 <= 180:                                                 # a footprint across RA = 0 gets no RA prefilter
        near &= (ra >= ra0) & (ra <= ra1)
    return np.flatnonzero(near)


def _pixels(wcs, shape: Tuple[int, int], ra: np.ndarray, dec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(inside the image, nearest column, nearest row) for each position."""
    ny, nx = shape
    x, y = wcs.world_to_pixel_values(ra, dec)
    inside = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
    return inside, np.clip(np.round(x).astype(int), 0, nx - 1), np.clip(np.round(y).astype(int), 0, ny - 1)


def _covered_in_quadrant(args):
    """Indices into the object table of the objects on a valid pixel of one quadrant-band image."""
    field, ccd, qid, band, data_dir, ra, dec = args
    bad, wcs = quadrant_mask(field, ccd, qid, band, data_dir)
    idx = _near_image(wcs, bad.shape, ra, dec)
    inside, xi, yi = _pixels(wcs, bad.shape, ra[idx], dec[idx])
    return idx[inside & ~bad[yi, xi]]


def object_coverage(obj: pd.DataFrame, field: int, quadrants: Iterable[Tuple[int, int]], data_dir: Optional[str] = None,
                    n_jobs: int = 1) -> pd.DataFrame:
    """`{b}_covered`: whether each object sits on a valid pixel of some quadrant's reference image in band b. A band
    without a detection is a non-detection only where this is true; the band images have different footprints.
    Only objects inside each image's sky box are projected; quadrant-bands run in `n_jobs` processes."""
    ra, dec = obj.ra.values, obj.dec.values
    jobs = [(field, ccd, qid, b, data_dir, ra, dec) for b in BANDS for ccd, qid in quadrants]
    if n_jobs > 1:
        from multiprocessing import get_context
        with get_context('fork').Pool(n_jobs) as pool:
            hits = pool.map(_covered_in_quadrant, jobs)
    else:
        hits = [_covered_in_quadrant(j) for j in jobs]
    for b in BANDS:
        covered = np.zeros(len(obj), bool)
        for job, h in zip(jobs, hits):
            if job[3] == b:
                covered[h] = True
        obj[f'{b}_covered'] = covered
    return obj


# ----------------------------------------------------------------------------------------------- PS1

def _quadrant_sky_grid(field: int, ccd: int, qid: int, data_dir: Optional[str], step_px: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Positions every `step_px` pixels across one quadrant's reference image (r band; the bands share a footprint)."""
    bad, wcs = quadrant_mask(field, ccd, qid, 'r', data_dir)
    yy, xx = np.mgrid[0:bad.shape[0]:step_px, 0:bad.shape[1]:step_px]
    ra, dec = wcs.pixel_to_world_values(xx.ravel(), yy.ravel())
    return np.asarray(ra), np.asarray(dec)


def _ps1_in_quadrant(args):
    """For the PS1 objects near one quadrant: (indices, inside some band's image, {band: on a valid pixel}, distance to
    the nearest masked pixel over the bands that contain them, inf elsewhere)."""
    field, ccd, qid, data_dir, ra, dec = args
    masks = {b: quadrant_mask(field, ccd, qid, b, data_dir) for b in BANDS}
    idx = np.unique(np.concatenate([_near_image(wcs, bad.shape, ra, dec) for bad, wcs in masks.values()]))
    inside_any, dist, covered = np.zeros(len(idx), bool), np.full(len(idx), np.inf), {}
    for b, (bad, wcs) in masks.items():
        inside, xi, yi = _pixels(wcs, bad.shape, ra[idx], dec[idx])
        covered[b] = inside & ~bad[yi, xi]
        inside_any |= inside
        dist = np.where(inside, np.minimum(dist, distance_transform_edt(~bad)[yi, xi]), dist)
    return idx, inside_any, covered, dist


def ps1_field(field: int, quadrants: Iterable[Tuple[int, int]], data_dir: Optional[str] = None,
              stack: Optional[str] = None, n_jobs: int = 1) -> pd.DataFrame:
    """The field's PS1 objects: the cleaned stack catalog of every partition under its reference images
    (`ps1_stack.objects`), restricted to objects inside some image. Each object takes the quadrant where it lies farthest
    from a masked pixel (ties, e.g. on masked pixels in two overlapping images, go to the quadrant where it is on valid
    pixels in more bands), with that quadrant's `{b}_covered`, `on_ztf` (on a valid pixel in some band) and
    `nan_dist_px`. Objects on masked pixels (saturated stars) stay, so they can still serve as counterparts."""
    quadrants = list(quadrants)
    grid = [_quadrant_sky_grid(field, ccd, qid, data_dir) for ccd, qid in quadrants]
    cat = ps1_stack.objects(np.concatenate([g[0] for g in grid]), np.concatenate([g[1] for g in grid]), stack)
    ra, dec = cat.ra.to_numpy(), cat.dec.to_numpy()
    jobs = [(field, ccd, qid, data_dir, ra, dec) for ccd, qid in quadrants]
    if n_jobs > 1:
        from multiprocessing import get_context
        with get_context('fork').Pool(n_jobs) as pool:
            results = pool.map(_ps1_in_quadrant, jobs)
    else:
        results = [_ps1_in_quadrant(j) for j in jobs]
    best = np.full(len(cat), -1.0)                                       # -1: inside no reference image
    best_n_covered = np.full(len(cat), -1)
    ccd, qid = np.full(len(cat), -1, np.int16), np.full(len(cat), -1, np.int8)
    covered = {b: np.zeros(len(cat), bool) for b in BANDS}
    for (q_ccd, q_qid), (idx, inside_any, q_covered, dist) in zip(quadrants, results):
        n_covered = sum(q_covered[b].astype(int) for b in BANDS)
        better = inside_any & ((dist > best[idx]) | ((dist == best[idx]) & (n_covered > best_n_covered[idx])))
        take = idx[better]
        best[take], best_n_covered[take], ccd[take], qid[take] = dist[better], n_covered[better], q_ccd, q_qid
        for b in BANDS:
            covered[b][take] = q_covered[b][better]
    cat['ccd'], cat['qid'], cat['nan_dist_px'] = ccd, qid, best
    for b in BANDS:
        cat[f'{b}_covered'] = covered[b]
    cat['on_ztf'] = np.logical_or.reduce([covered[b] for b in BANDS])
    return cat[best >= 0].sort_values('PanSTARR_ID').reset_index(drop=True)


def match_ps1(obj: pd.DataFrame, ps1: pd.DataFrame, radius_arcsec: float = MATCH_ARCSEC) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """`ps1_idx`, `ps1_sep` and `matched` on the objects (-1 / NaN / False without a match); `ztf_idx` and `matched` on the PS1 table."""
    j, sep = mutual_nearest(obj.ra, obj.dec, ps1.ra, ps1.dec, radius_arcsec)
    obj = obj.copy(); ps1 = ps1.copy()
    obj['ps1_idx'], obj['ps1_sep'] = j, sep
    obj['matched'] = j >= 0
    ps1['ztf_idx'] = -1
    ps1.loc[j[j >= 0], 'ztf_idx'] = np.flatnonzero(j >= 0)
    ps1['matched'] = ps1['ztf_idx'] >= 0
    return obj, ps1


# ----------------------------------------------------------------------------------------------- driver

def build_field(field: int, quadrants: Sequence[Tuple[int, int]], data_dir: Optional[str] = None, overwrite: bool = False, n_jobs: int = 1,
                require: Sequence[str] = ()):
    """Steps 1 to 4 for one field. Writes ztf_<b>.parquet, ztf_objects.parquet and ps1.parquet under the field directory
    and returns (bands, objects, ps1)."""
    out = field_dir(field, data_dir)
    os.makedirs(out, exist_ok=True)
    bands = {b: field_band(field, quadrants, b, data_dir, n_jobs=n_jobs, overwrite=overwrite, require=require) for b in BANDS}
    for b, cat in bands.items():
        cat.to_parquet(os.path.join(out, f'ztf_{b}.parquet'))
    obj = merge_bands(bands)
    obj = object_coverage(obj, field, quadrants, data_dir, n_jobs=n_jobs)
    ps1 = ps1_field(field, quadrants, data_dir, n_jobs=n_jobs)
    obj, ps1 = match_ps1(obj, ps1)
    obj.to_parquet(os.path.join(out, 'ztf_objects.parquet'))
    ps1.to_parquet(os.path.join(out, 'ps1.parquet'))
    return bands, obj, ps1


if __name__ == '__main__':                                 # python -m Extracting.field_tables 860 --n-jobs 8
    ap = argparse.ArgumentParser(description='Build one field\'s source tables from its reference products and the PS1 mirror.')
    ap.add_argument('field', type=int)
    ap.add_argument('--n-jobs', type=int, default=1)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--require', nargs='*', default=[], help='optional products that must be present, e.g. refunc')
    a = ap.parse_args()
    quads = field_quadrants(a.field, a.data_dir)
    print(f'field {a.field}: {len(quads)} quadrants', flush=True)
    bands, obj, ps1 = build_field(a.field, quads, a.data_dir, overwrite=a.overwrite, n_jobs=a.n_jobs, require=a.require)
    print(f'done: ' + ', '.join(f'{b} {len(c):,}' for b, c in bands.items()) + f'; objects {len(obj):,}; PS1 {len(ps1):,}; matched {int(obj.matched.sum()):,}', flush=True)
