"""field_tables.ps1_field: which PS1 objects a field keeps, and which quadrant each is assigned, on synthetic images."""
import numpy as np
import pandas as pd
import pytest
from astropy.wcs import WCS

from Extracting import field_tables as ft

SIZE = 100                                     # pixels per side, 1"/px


def _wcs(ra0):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [ra0, 0.0]
    w.wcs.crpix = [SIZE / 2 + 0.5, SIZE / 2 + 0.5]
    w.wcs.cdelt = [-1 / 3600, 1 / 3600]
    return w


# Two quadrants overlapping by 40": A centred at RA 10 deg, B 60" west of it (RA falls with x, so A's x > 60 is in B).
WCS_A, WCS_B = _wcs(10.0), _wcs(10.0 - 60 / 3600)


def _bad(*pixels):
    bad = np.zeros((SIZE, SIZE), bool)
    for x, y in pixels:
        bad[y, x] = True
    return bad


def _sky(wcs, x, y):
    ra, dec = wcs.pixel_to_world_values(x, y)
    return float(ra), float(dec)


@pytest.fixture
def field(monkeypatch):
    """Quadrant A has a bad pixel 3 px from the overlap object; B has one 15 px away, plus a pixel masked in g only. A is
    the deeper image (MAGLIM 22 against 21)."""
    near_x, near_y = 80, 50                                          # A-pixel of the overlap object
    ra_o, dec_o = _sky(WCS_A, near_x, near_y)
    bx, by = (int(round(float(v))) for v in WCS_B.world_to_pixel_values(ra_o, dec_o))
    tie_ax, tie_ay = 85, 20                                          # the tie object, masked in every band of A
    ra_t, dec_t = _sky(WCS_A, tie_ax, tie_ay)
    tbx, tby = (int(round(float(v))) for v in WCS_B.world_to_pixel_values(ra_t, dec_t))
    ra_c, dec_c = _sky(WCS_A, 70, 80)                                # a clean overlap object: far from every mask in both
    masks = {
        (1, 1): {b: (_bad((near_x + 3, near_y), (tie_ax, tie_ay)), WCS_A) for b in 'gri'},
        (1, 2): {'g': (_bad((bx - 15, by), (tbx, tby)), WCS_B), 'r': (_bad((bx - 15, by)), WCS_B),
                 'i': (_bad((bx - 15, by)), WCS_B)},
    }
    monkeypatch.setattr(ft, 'quadrant_mask', lambda field, ccd, qid, band, data_dir=None: masks[(ccd, qid)][band])
    monkeypatch.setattr(ft, '_quadrant_maglim', lambda field, ccd, qid, band, data_dir=None: {(1, 1): 22.0, (1, 2): 21.0}[(ccd, qid)])
    cat = pd.DataFrame({
        'PanSTARR_ID': [1, 2, 3, 4, 5],
        'ra': [_sky(WCS_A, 10, 50)[0], ra_o, ra_t, 11.0, ra_c],       # A only, overlap near a mask in A, tie, outside both, clean overlap
        'dec': [_sky(WCS_A, 10, 50)[1], dec_o, dec_t, 0.0, dec_c],
    })
    monkeypatch.setattr(ft.ps1_stack, 'objects', lambda ra, dec, directory=None: cat.copy())
    return ft.ps1_field(0, [(1, 1), (1, 2)]).set_index('PanSTARR_ID')


def test_objects_outside_every_image_are_dropped_and_the_rest_kept(field):
    assert sorted(field.index) == [1, 2, 3, 5]


def test_an_object_in_one_image_takes_that_quadrant_in_every_band(field):
    for b in 'gri':
        assert (field.loc[1, f'{b}_ccd'], field.loc[1, f'{b}_qid']) == (1, 1) and field.loc[1, f'{b}_covered']
        assert field.loc[1, f'{b}_maglim'] == 22.0


def test_a_clean_image_beats_a_deeper_one_where_the_position_is_near_a_mask(field):
    for b in 'gri':                                                  # A is deeper but its copy is 3 px from a mask
        assert (field.loc[2, f'{b}_ccd'], field.loc[2, f'{b}_qid']) == (1, 2)
        assert field.loc[2, f'{b}_nan_dist_px'] == pytest.approx(15) and field.loc[2, f'{b}_maglim'] == 21.0


def test_among_clean_images_the_deeper_one_wins(field):
    for b in 'gri':
        assert (field.loc[5, f'{b}_ccd'], field.loc[5, f'{b}_qid']) == (1, 1) and field.loc[5, f'{b}_maglim'] == 22.0


def test_coverage_is_decided_band_by_band(field):
    assert not field.loc[3, 'g_covered'] and field.loc[3, 'g_ccd'] == -1 and np.isnan(field.loc[3, 'g_maglim'])   # masked in g everywhere
    assert field.loc[3, 'r_covered'] and (field.loc[3, 'r_ccd'], field.loc[3, 'r_qid']) == (1, 2)                 # B's r is clean there


def _band(ra, dec, snr):
    return pd.DataFrame({'ra': ra, 'dec': dec, 'snr': snr, 'mag': 20.0, 'magerr': 0.1, 'chi': 1.0, 'sharp': 0.0, 'flags': 0})


def test_merge_bands_links_every_pair_without_a_band_order_and_averages_positions():
    s = 1 / 3600                                                              # one arcsecond, in degrees
    g = _band([10.0], [20.0], [50.0])                                         # the star in g
    r = _band([10.0 + 0.2 * s, 10.0 + 1.5 * s], [20.0, 20.0], [100.0, 5.0])    # the star 0.2" off, a faint neighbour 1.5" away
    i = _band([10.0, 10.0 + 0.4 * s], [20.0 + 0.2 * s, 20.0 + 3.0 * s], [20.0, 8.0])   # the star; an unrelated i source 3" away
    for order in ({'g': g, 'r': r, 'i': i}, {'i': i, 'r': r, 'g': g}):
        o = ft.merge_bands(order).sort_values(['ra', 'dec']).reset_index(drop=True)
        n = o[['g_idx', 'r_idx', 'i_idx']].ge(0).sum(axis=1)
        assert len(o) == 3 and sorted(n) == [1, 1, 3]
        star = o[n == 3].iloc[0]
        assert (star.g_idx, star.r_idx, star.i_idx) == (0, 0, 0)
        assert abs(star.ra - (10.0 + 0.2 * s * 100 / 170)) < 1e-7              # S/N-weighted mean: r (100) pulls hardest
        assert 0 < star.r_sep < 0.2 and star.g_sep > 0
        assert set(zip(o.g_idx, o.r_idx, o.i_idx)) == {(0, 0, 0), (-1, 1, -1), (-1, -1, 1)}


def test_merge_bands_splits_a_component_with_two_rows_of_one_band_at_its_longest_edge():
    s = 1 / 3600
    g = _band([10.0], [20.0], [10.0])
    r = _band([10.0 + 0.3 * s], [20.0], [10.0])                               # r1: 0.3" from g1
    i = _band([10.0 + 0.9 * s, 10.0 - 0.8 * s], [20.0, 20.0], [10.0, 10.0])  # i1: 0.6" from r1 (mutual); i2: 0.8" from g1 (mutual with g1)
    o = ft.merge_bands({'g': g, 'r': r, 'i': i})
    counts = o[['g_idx', 'r_idx', 'i_idx']].ge(0).sum(axis=1)
    assert len(o) == 2 and sorted(counts) == [1, 3]                           # the 0.8" g-i2 edge goes; i2 stands alone
    three = o[counts == 3].iloc[0]
    assert (three.g_idx, three.r_idx, three.i_idx) == (0, 0, 0)
