"""PS1's sky tessellation (RINGS.V3): which projection cell and skycell a position falls in, and MAST's list of skycells
near the north pole whose stacks are mis-registered.

Geometry, from MAST's "PS1 Sky tessellation patterns" page
(https://outerspace.stsci.edu/spaces/PANSTARRS/pages/298812317/PS1+Sky+tessellation+patterns) and its `ps1grid.fits`
(GRID below, one row per declination zone): 4 x 4 degree tangent-plane projection cells centred on declination rings 4
degrees apart, NBAND cells per ring at RA = n * 360 / NBAND, numbered consecutively from PROJCELL; each cell is a 10 x
10 grid of skycells of XCELL x YCELL pixels (0.25" per pixel) that overlap their neighbours by 240 pixels on each side,
named `skycell.<cell>.0yx` with 000 at the bottom left (x east to the left, y north). All skycells of a cell share its
TAN WCS; CRPIX1/2 is where the cell centre falls inside the central skycell (column and row 5), so in the full cell
image, whose skycell i starts at pixel i * (XCELL - 480), the centre is at 5 * (XCELL - 480) + CRPIX1. A position's
skycell (`skycell_of`) is the one it lies farthest from the edge of, i.e. the one whose non-overlapping core contains
it, in the projection cell nearest in RA. Projection cells are 4.15 degrees wide but their centres are closer than
that in RA at high declination, so neighbouring cells overlap by up to a skycell there and a position near a cell's
edge is covered by two skycells (`covering_skycells`); PS1's own choice of primary skycell for an object is its
`projectionID` / `skyCellID`, which this module reproduces for 99.97% of primary stack rows in a field-860 partition.

Bad skycells: PS1 DR2's stacks near the north pole have cross-registration errors of up to several pixels (MAST, "PS1
DR2 caveats", Questionable Images near the Pole). MAST set every magnitude of the affected skycells to -999 in the
main object and detection tables while the flux columns stayed populated, and lists the skycells (those with more than
5% of sources with an internal positional rms above 1.5 pixels in some band) in `badcells_near_north_pole_v6.dat`
(linked from https://outerspace.stsci.edu/display/PANSTARRS/Plot+of+bad+skycells+near+the+North+Pole; 461 skycells,
Dec 68.6-89.7), a copy of which sits next to this module. The detection and photometry there come from the same
mis-registered stacks, so nothing PS1 measured in them is usable; see `is_bad`.
"""
import os
from functools import lru_cache
from typing import Set, Tuple

import numpy as np
from astropy.wcs import WCS

PIXEL_ARCSEC = 0.25
OVERLAP_PX = 240                # each skycell extends this far into its neighbours, on every side
GRID = np.array([               # ZONE, PROJCELL, NBAND, DEC, DEC_MIN, DEC_MAX, XCELL, YCELL, CRPIX1, CRPIX2  (ps1grid.fits)
    (13, 487, 72, -38.0, -39.98569922229713, -35.98748871852416, 6305, 6279, 239.5, 238.0),
    (14, 559, 76, -34.0, -35.98748871852416, -31.988947843740302, 6265, 6274, 237.5, 237.5),
    (15, 635, 79, -30.0, -31.988947843740302, -27.990585819085737, 6274, 6269, 239.5, 241.5),
    (16, 714, 82, -26.0, -27.990585819085737, -23.991856317278298, 6255, 6265, 241.5, 240.5),
    (17, 796, 84, -22.0, -23.991856317278298, -19.99333106003115, 6279, 6261, 241.0, 242.0),
    (18, 880, 86, -18.0, -19.99333106003115, -15.994464100667857, 6274, 6258, 241.5, 238.0),
    (19, 966, 88, -14.0, -15.994464100667857, -11.995671545879146, 6242, 6254, 240.5, 239.5),
    (20, 1054, 89, -10.0, -11.995671545879146, -7.997014822849273, 6248, 6250, 240.0, 242.0),
    (21, 1143, 89, -6.0, -7.997014822849273, -3.998086931795508, 6291, 6247, 237.5, 240.0),
    (22, 1232, 90, -2.0, -3.998086931795508, 0.0, 6240, 6243, 240.0, 242.0),
    (23, 1322, 90, 2.0, 0.0, 3.998086931795508, 6240, 6243, 240.0, 242.0),
    (24, 1412, 89, 6.0, 3.998086931795508, 7.997014822849273, 6291, 6247, 237.5, 240.0),
    (25, 1501, 89, 10.0, 7.997014822849273, 11.995671545879146, 6248, 6250, 240.0, 242.0),
    (26, 1590, 88, 14.0, 11.995671545879146, 15.994464100667857, 6242, 6254, 240.5, 239.5),
    (27, 1678, 86, 18.0, 15.994464100667857, 19.99333106003115, 6274, 6258, 241.5, 238.0),
    (28, 1764, 84, 22.0, 19.99333106003115, 23.991856317278298, 6279, 6261, 241.0, 242.0),
    (29, 1848, 82, 26.0, 23.991856317278298, 27.990585819085737, 6255, 6265, 241.5, 240.5),
    (30, 1930, 79, 30.0, 27.990585819085737, 31.988947843740302, 6274, 6269, 239.5, 241.5),
    (31, 2009, 76, 34.0, 31.988947843740302, 35.98748871852416, 6265, 6274, 237.5, 237.5),
    (32, 2085, 72, 38.0, 35.98748871852416, 39.98569922229713, 6305, 6279, 239.5, 238.0),
    (33, 2157, 68, 42.0, 39.98569922229713, 43.98384988850235, 6320, 6284, 239.5, 240.0),
    (34, 2225, 64, 46.0, 43.98384988850235, 47.98179099058205, 6307, 6289, 238.0, 242.0),
    (35, 2289, 60, 50.0, 47.98179099058205, 51.97972846465295, 6261, 6295, 241.0, 239.5),
    (36, 2349, 55, 54.0, 51.97972846465295, 55.97676677254534, 6283, 6302, 239.0, 242.0),
    (37, 2404, 50, 58.0, 55.97676677254534, 59.97411834356914, 6278, 6311, 238.5, 237.5),
    (38, 2454, 45, 62.0, 59.97411834356914, 63.97005521318337, 6240, 6319, 240.0, 241.0),
    (39, 2499, 39, 66.0, 63.97005521318337, 67.96475284198297, 6307, 6333, 239.5, 240.0),
    (40, 2538, 33, 70.0, 67.96475284198297, 71.95817764373305, 6365, 6350, 238.5, 240.0),
    (41, 2571, 27, 74.0, 71.95817764373305, 75.94974140116577, 6413, 6371, 240.5, 242.0),
    (42, 2598, 21, 78.0, 75.94974140116577, 79.93968190571101, 6452, 6398, 240.0, 242.0),
    (43, 2619, 15, 82.0, 79.93968190571101, 83.93610604798963, 6481, 6429, 241.0, 242.0),
    (44, 2634, 9, 86.0, 83.93610604798963, 87.96940975734594, 6501, 6419, 239.0, 239.5),
    (45, 2643, 1, 90.0, 87.96940975734594, 90.0001, 6240, 6240, 239.5, 240.0),
])
ZONE, PROJCELL, NBAND, DEC, DEC_MIN, DEC_MAX, XCELL, YCELL, CRPIX1, CRPIX2 = range(10)
BAD_CELLS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'badcells_near_north_pole_v6.dat')


def zone_of(dec: np.ndarray) -> np.ndarray:
    """Row of GRID for each declination (the zone whose [DEC_MIN, DEC_MAX) holds it); -1 south of the table."""
    dec = np.asarray(dec, float)
    row = np.searchsorted(GRID[:, DEC_MIN], dec, side='right') - 1
    row[(row >= 0) & (dec >= GRID[np.clip(row, 0, None), DEC_MAX])] = -1          # only the top row can fail this
    return row


def cell_wcs(projcell: int) -> WCS:
    """The TAN WCS of a projection cell's full image (all its skycells share it), 1-based pixels."""
    row = GRID[(GRID[:, PROJCELL] <= projcell).nonzero()[0][-1]]
    n = projcell - int(row[PROJCELL])
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [n * 360.0 / row[NBAND], row[DEC]]
    w.wcs.crpix = [5 * (row[XCELL] - 2 * OVERLAP_PX) + row[CRPIX1], 5 * (row[YCELL] - 2 * OVERLAP_PX) + row[CRPIX2]]
    w.wcs.cdelt = [-PIXEL_ARCSEC / 3600, PIXEL_ARCSEC / 3600]
    return w


def skycell_of(ra: np.ndarray, dec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(projection cell, skycell 0-99) of each position: the cell of its declination zone nearest in RA, and within it
    the skycell whose non-overlapping core holds the position. -1 where the position is south of the table."""
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    row = zone_of(dec)
    proj = np.full(len(ra), -1, np.int32); cell = np.full(len(ra), -1, np.int16)
    ok = row >= 0
    g = GRID[np.clip(row, 0, None)]
    n = np.round(ra / (360.0 / g[:, NBAND])).astype(int) % g[:, NBAND].astype(int)
    proj[ok] = (g[:, PROJCELL].astype(int) + n)[ok]
    for p in np.unique(proj[ok]):
        sel = proj == p
        x, y = cell_wcs(int(p)).all_world2pix(ra[sel], dec[sel], 1)
        r = GRID[(GRID[:, PROJCELL] <= p).nonzero()[0][-1]]
        ix = np.clip(np.floor((x - 1 - OVERLAP_PX) / (r[XCELL] - 2 * OVERLAP_PX)), 0, 9)
        iy = np.clip(np.floor((y - 1 - OVERLAP_PX) / (r[YCELL] - 2 * OVERLAP_PX)), 0, 9)
        cell[sel] = (10 * iy + ix).astype(np.int16)
    return proj, cell


@lru_cache(maxsize=1)
def bad_cells() -> Set[Tuple[int, int]]:
    """MAST's bad skycells near the north pole, as (projection cell, skycell)."""
    out = set()
    for line in open(BAD_CELLS_FILE):
        if line.startswith('skycell.'):
            _, p, c = line.split()[0].split('.')
            out.add((int(p), int(c)))
    return out


def covering_skycells(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Every skycell whose image covers each position, as an (N, 3, 2) array of (projection cell, skycell) over the
    nearest projection cell in RA and its two RA neighbours (-1 where a cell does not cover the position). Within a
    cell the skycell is the one whose core holds the position, as in `skycell_of`; the cell's edge skycells reach into
    the neighbouring cell's area, which is how a position comes to lie in two skycells."""
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    row = zone_of(dec)
    out = np.full((len(ra), 3, 2), -1, np.int32)
    ok = row >= 0
    g = GRID[np.clip(row, 0, None)]
    m = g[:, NBAND].astype(int)
    n0 = np.round(ra / (360.0 / m)).astype(int)
    for k, dn in enumerate((-1, 0, 1)):
        proj = np.where(ok, g[:, PROJCELL].astype(int) + (n0 + dn) % m, -1)
        for p in np.unique(proj[ok]):
            sel = proj == p
            x, y = cell_wcs(int(p)).all_world2pix(ra[sel], dec[sel], 1)
            r = GRID[(GRID[:, PROJCELL] <= p).nonzero()[0][-1]]
            nx, ny = 10 * (r[XCELL] - 2 * OVERLAP_PX) + 2 * OVERLAP_PX, 10 * (r[YCELL] - 2 * OVERLAP_PX) + 2 * OVERLAP_PX
            inside = (x >= 1) & (x <= nx) & (y >= 1) & (y <= ny)
            ix = np.clip(np.floor((x - 1 - OVERLAP_PX) / (r[XCELL] - 2 * OVERLAP_PX)), 0, 9)
            iy = np.clip(np.floor((y - 1 - OVERLAP_PX) / (r[YCELL] - 2 * OVERLAP_PX)), 0, 9)
            idx = np.flatnonzero(sel)[inside]
            out[idx, k, 0] = p
            out[idx, k, 1] = (10 * iy + ix)[inside].astype(np.int32)
    return out


def is_bad(ra: np.ndarray, dec: np.ndarray, any_covering: bool = True) -> np.ndarray:
    """Whether a position lies in one of MAST's bad polar skycells: any skycell covering it (`covering_skycells`,
    the liberal reading, right for a position where PS1 has no object), or only its primary one (`skycell_of`)."""
    codes = np.array([100 * p + c for p, c in bad_cells()], np.int64)
    if any_covering:
        cells = covering_skycells(ra, dec)
        code = 100 * cells[:, :, 0].astype(np.int64) + cells[:, :, 1]
        return np.isin(code, codes).any(axis=1)                      # an absent cell is -1: never a bad code
    proj, cell = skycell_of(ra, dec)
    return np.isin(100 * proj.astype(np.int64) + cell, codes)
