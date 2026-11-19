"""Per-field source tables from the per-quadrant ZTF reference products and the PS1 catalog.

Order, per field:
  1. per band: each quadrant's DAOPhot catalog (`refpsfcat`) with our Kron photometry attached, concatenated over
     the quadrants, then deduplicated in the overlap strips (the 3200 px reference
     images are larger than the 3072 x 3080 px quadrants, so neighbouring images overlap by ~1-2');
  2. the bands merged into objects: every band pair matched by mutual nearest neighbour at MATCH_ARCSEC, the pairs
     linked into objects (no band order), each object at the S/N-weighted mean position of its detections;
  3. the PS1 stack catalog (`ps1_stack`: our mirror of MAST's stack table, cleaned to one row per object) for the
     partitions under the field, restricted to objects inside some reference image; from the untrimmed partitions, PS1's
     coverage and stack frames around every ZTF and PS1 object (`ps1_neighbourhood`);
  4. objects matched to PS1 by mutual nearest neighbour at MATCH_ARCSEC, written as a match table of row indices.
Every step works on one homogeneous table; nothing downstream needs to know about quadrants. A table holds only its
own data and the row indices of the tables it was made from, never anything about tables made from it.
"""
import argparse
import glob
import os
import warnings
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
CLEAN_PX = 5                       # a position this far from a masked pixel counts as clean when choosing among covering images
PS1_COVERAGE_ARCMIN = 1.0          # PS1's band-b stack reaches a position if it measured some object in b this close (the old pipeline's 1' rule)
NFRAMES_NEIGHBOURS, NFRAMES_MAX_ARCSEC = 5, 60.0   # PS1 stack frames at a position: median nFrames of its nearest PS1 objects
PSF_COLS = {'ra': 'ra', 'dec': 'dec', 'xpos': 'x', 'ypos': 'y', 'mag': 'mag', 'sigmag': 'magerr', 'snr': 'snr', 'chi': 'chi', 'sharp': 'sharp', 'flags': 'flags'}
KRON_COLS = {'mag': 'zkron', 'magerr': 'zkron_err', 'r1': 'zkron_r1', 'r_ap': 'zkron_rap', 'sky_radius': 'zkron_sky', 'flag': 'zkron_flag', 'masked_frac': 'zkron_masked'}


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
    table, -1 if absent) and `{b}_sep` (the row's separation from the object position, arcsec); the photometry stays
    in the band tables."""
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


def _quadrant_maglim(field: int, ccd: int, qid: int, band: str, data_dir: Optional[str] = None) -> float:
    """The reference image's MAGLIM (its nominal 5-sigma depth) from the header; NaN where the band has no image."""
    data_dir = data_dir or get_data_path()
    if not has_quadrant_band(field, ccd, qid, band, data_dir):
        return np.nan
    h = fits.getheader(refimg_path(os.path.join(data_dir, 'ztf_data'), field, ccd, qid, band))
    return float(h['MAGLIM']) if 'MAGLIM' in h else np.nan


def _coverage_in_quadrant(args):
    """For the positions inside one quadrant-band image: (indices, on a valid pixel, distance to the nearest masked pixel
    in px, the image's MAGLIM)."""
    field, ccd, qid, band, data_dir, ra, dec = args
    bad, wcs = quadrant_mask(field, ccd, qid, band, data_dir)
    idx = _near_image(wcs, bad.shape, ra, dec)
    inside, xi, yi = _pixels(wcs, bad.shape, ra[idx], dec[idx])
    idx, xi, yi = idx[inside], xi[inside], yi[inside]
    dist = distance_transform_edt(~bad)[yi, xi] if len(idx) else np.zeros(0)
    return idx, ~bad[yi, xi], dist, _quadrant_maglim(field, ccd, qid, band, data_dir)


def ztf_coverage(ra: np.ndarray, dec: np.ndarray, field: int, quadrants: Iterable[Tuple[int, int]], data_dir: Optional[str] = None,
                 n_jobs: int = 1) -> pd.DataFrame:
    """What ZTF has at each position, per band, from the one covering image chosen in this order: the position on a
    valid pixel, then at least CLEAN_PX from a masked pixel, then the deepest MAGLIM. Columns `{b}_covered` (some
    image has it on a valid pixel), `{b}_ccd`, `{b}_qid` (that image; -1 if none covers it), `{b}_nan_dist_px` (the
    distance to the nearest masked pixel in it; 0 if none covers it) and `{b}_maglim` (its MAGLIM; NaN if none), plus
    `on_image`: inside some image of some band at all. Quadrant-bands run in `n_jobs` processes."""
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    quadrants = list(quadrants)
    jobs = [(field, ccd, qid, b, data_dir, ra, dec) for b in BANDS for ccd, qid in quadrants]
    if n_jobs > 1:
        from multiprocessing import get_context
        with get_context('fork').Pool(n_jobs) as pool:
            hits = pool.map(_coverage_in_quadrant, jobs)
    else:
        hits = [_coverage_in_quadrant(j) for j in jobs]
    n = len(ra)
    out = {'on_image': np.zeros(n, bool)}
    for b in BANDS:
        score = np.full(n, -np.inf)
        ccd, qid = np.full(n, -1, np.int16), np.full(n, -1, np.int8)
        covered, dist, maglim = np.zeros(n, bool), np.zeros(n), np.full(n, np.nan)
        for job, (idx, valid, d, ml) in zip(jobs, hits):
            if job[3] != b or len(idx) == 0:
                continue
            out['on_image'][idx] = True
            sc = valid * 1e4 + (valid & (d > CLEAN_PX)) * 1e2 + np.where(valid, np.nan_to_num(ml, nan=0.0), d)   # covered > clean > deeper
            better = sc > score[idx]
            take = idx[better]
            score[take], ccd[take], qid[take] = sc[better], job[1], job[2]
            covered[take], dist[take], maglim[take] = valid[better], d[better], ml
        out.update({f'{b}_covered': covered, f'{b}_ccd': np.where(covered, ccd, -1).astype(np.int16), f'{b}_qid': np.where(covered, qid, -1).astype(np.int8),
                    f'{b}_nan_dist_px': np.where(covered, dist, 0.0).astype(np.float32), f'{b}_maglim': np.where(covered, maglim, np.nan).astype(np.float32)})
    return pd.DataFrame(out)


def object_coverage(obj: pd.DataFrame, field: int, quadrants: Iterable[Tuple[int, int]], data_dir: Optional[str] = None,
                    n_jobs: int = 1) -> pd.DataFrame:
    """The ZTF objects with `ztf_coverage` at their positions: for a band without a detection this is where its limit
    comes from; a band without a detection is a non-detection only where `{b}_covered`."""
    cov = ztf_coverage(obj.ra.values, obj.dec.values, field, quadrants, data_dir, n_jobs)
    return pd.concat([obj.reset_index(drop=True), cov.drop(columns='on_image')], axis=1)


# ----------------------------------------------------------------------------------------------- PS1

def _quadrant_sky_grid(field: int, ccd: int, qid: int, data_dir: Optional[str], step_px: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Positions every `step_px` pixels across one quadrant's reference image (r band; the bands share a footprint)."""
    bad, wcs = quadrant_mask(field, ccd, qid, 'r', data_dir)
    yy, xx = np.mgrid[0:bad.shape[0]:step_px, 0:bad.shape[1]:step_px]
    ra, dec = wcs.pixel_to_world_values(xx.ravel(), yy.ravel())
    return np.asarray(ra), np.asarray(dec)


def ps1_neighbourhood(ra: np.ndarray, dec: np.ndarray, ps1: pd.DataFrame) -> pd.DataFrame:
    """What PS1's catalog says about its own coverage at each position, per band: `{b}_ps1_covered` (some PS1 object
    measured in b within PS1_COVERAGE_ARCMIN: the band's stack reaches the position) and `{b}_ps1_nframes_local` (the
    median `{b}nFrames` of the NFRAMES_NEIGHBOURS nearest PS1 objects with at least one frame, within
    NFRAMES_MAX_ARCSEC; NaN without any). `ps1` should be the untrimmed partition catalog, so positions near a field's
    edge see their neighbours outside it."""
    pos = _xyz(ra, dec)
    out = {}
    for b in BANDS:
        flux, err, frames = ps1[f'{b}PSFFlux'].values, ps1[f'{b}PSFFluxErr'].values, ps1[f'{b}nFrames'].values
        meas = np.isfinite(flux) & (err > 0)
        p = _xyz(ps1.ra.values, ps1.dec.values)
        out[f'{b}_ps1_covered'] = (np.isfinite(cKDTree(p[meas]).query(pos, k=1, distance_upper_bound=_chord(PS1_COVERAGE_ARCMIN * 60))[0])
                                   if meas.any() else np.zeros(len(pos), bool))
        ok = frames >= 1
        local = np.full(len(pos), np.nan)
        if ok.any():
            k = min(NFRAMES_NEIGHBOURS, int(ok.sum()))
            d, i = cKDTree(p[ok]).query(pos, k=k, distance_upper_bound=_chord(NFRAMES_MAX_ARCSEC))
            d, i = d.reshape(len(pos), -1), i.reshape(len(pos), -1)
            vals = np.where(np.isfinite(d), frames[ok][np.clip(i, 0, ok.sum() - 1)], np.nan)
            with warnings.catch_warnings():                                     # a position with no neighbour is an all-NaN row
                warnings.simplefilter('ignore', RuntimeWarning)
                local = np.nanmedian(vals, axis=1)
        out[f'{b}_ps1_nframes_local'] = local.astype(np.float32)
    return pd.DataFrame(out)


def ps1_field(field: int, quadrants: Iterable[Tuple[int, int]], data_dir: Optional[str] = None,
              stack: Optional[str] = None, n_jobs: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The field's PS1 objects: the cleaned stack catalog of every partition under its reference images
    (`ps1_stack.objects`), restricted to objects inside some image, with `ztf_coverage` at their positions (what ZTF
    has there, per band) and `{b}_nframes_local` (`ps1_neighbourhood`; their own `{b}nFrames` where they have one).
    Objects on masked pixels (saturated stars) stay, so they can still serve as counterparts. Returns (the field's
    objects, the untrimmed partition catalog, for the neighbourhood of the ZTF objects)."""
    quadrants = list(quadrants)
    grid = [_quadrant_sky_grid(field, ccd, qid, data_dir) for ccd, qid in quadrants]
    full = ps1_stack.objects(np.concatenate([g[0] for g in grid]), np.concatenate([g[1] for g in grid]), stack).reset_index(drop=True)
    cov = ztf_coverage(full.ra.to_numpy(), full.dec.to_numpy(), field, quadrants, data_dir, n_jobs)
    near = ps1_neighbourhood(full.ra.to_numpy(), full.dec.to_numpy(), full)
    near = near[[f'{b}_ps1_nframes_local' for b in BANDS]].rename(columns=lambda c: c.replace('_ps1_', '_'))
    cat = pd.concat([full, cov.drop(columns='on_image'), near], axis=1)
    return cat[cov.on_image.values].sort_values('PanSTARR_ID').reset_index(drop=True), full


def match_ps1(obj: pd.DataFrame, ps1: pd.DataFrame, radius_arcsec: float = MATCH_ARCSEC) -> pd.DataFrame:
    """The ZTF-PS1 match as its own table, one row per matched pair: `ztf_obj` (row in the ZTF objects table), `ps1`
    (row in the PS1 table), `sep_arcsec`. The two source tables are left as they are."""
    j, sep = mutual_nearest(obj.ra, obj.dec, ps1.ra, ps1.dec, radius_arcsec)
    k = np.flatnonzero(j >= 0)
    return pd.DataFrame({'ztf_obj': k, 'ps1': j[k], 'sep_arcsec': sep[k]})


# ----------------------------------------------------------------------------------------------- driver

def build_field(field: int, quadrants: Sequence[Tuple[int, int]], data_dir: Optional[str] = None, overwrite: bool = False, n_jobs: int = 1,
                require: Sequence[str] = ()):
    """Steps 1 to 4 for one field. Writes ztf_<b>.parquet, ztf_objects.parquet, ps1.parquet and matches.parquet under the
    field directory and returns (bands, objects, ps1, matches). Each table holds only its own data plus the row indices
    of the tables it was made from (`{b}_idx` on the objects, `ztf_obj` and `ps1` on the matches)."""
    out = field_dir(field, data_dir)
    os.makedirs(out, exist_ok=True)
    bands = {b: field_band(field, quadrants, b, data_dir, n_jobs=n_jobs, overwrite=overwrite, require=require) for b in BANDS}
    for b, cat in bands.items():
        cat.to_parquet(os.path.join(out, f'ztf_{b}.parquet'))
    obj = merge_bands(bands)
    obj = object_coverage(obj, field, quadrants, data_dir, n_jobs=n_jobs)
    ps1, ps1_full = ps1_field(field, quadrants, data_dir, n_jobs=n_jobs)
    obj = pd.concat([obj, ps1_neighbourhood(obj.ra.values, obj.dec.values, ps1_full)], axis=1)   # what PS1 has at each ZTF object
    matches = match_ps1(obj, ps1)
    obj.to_parquet(os.path.join(out, 'ztf_objects.parquet'))
    ps1.to_parquet(os.path.join(out, 'ps1.parquet'))
    matches.to_parquet(os.path.join(out, 'matches.parquet'))
    return bands, obj, ps1, matches


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
    bands, obj, ps1, matches = build_field(a.field, quads, a.data_dir, overwrite=a.overwrite, n_jobs=a.n_jobs, require=a.require)
    print(f'done: ' + ', '.join(f'{b} {len(c):,}' for b, c in bands.items()) + f'; objects {len(obj):,}; PS1 {len(ps1):,}; matched {len(matches):,}', flush=True)
