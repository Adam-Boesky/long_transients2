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
    """Quadrant A has a bad pixel 3 px from the overlap object; B has one 15 px away, plus a pixel masked in g only."""
    near_x, near_y = 80, 50                                          # A-pixel of the overlap object
    ra_o, dec_o = _sky(WCS_A, near_x, near_y)
    bx, by = (int(round(float(v))) for v in WCS_B.world_to_pixel_values(ra_o, dec_o))
    tie_ax, tie_ay = 85, 20                                          # the tie object, masked in every band of A
    ra_t, dec_t = _sky(WCS_A, tie_ax, tie_ay)
    tbx, tby = (int(round(float(v))) for v in WCS_B.world_to_pixel_values(ra_t, dec_t))
    masks = {
        (1, 1): {b: (_bad((near_x + 3, near_y), (tie_ax, tie_ay)), WCS_A) for b in 'gri'},
        (1, 2): {'g': (_bad((bx - 15, by), (tbx, tby)), WCS_B), 'r': (_bad((bx - 15, by)), WCS_B),
                 'i': (_bad((bx - 15, by)), WCS_B)},
    }
    monkeypatch.setattr(ft, 'quadrant_mask', lambda field, ccd, qid, band, data_dir=None: masks[(ccd, qid)][band])
    cat = pd.DataFrame({
        'PanSTARR_ID': [1, 2, 3, 4],
        'ra': [_sky(WCS_A, 10, 50)[0], ra_o, ra_t, 11.0],            # A only, overlap, tie, outside both
        'dec': [_sky(WCS_A, 10, 50)[1], dec_o, dec_t, 0.0],
    })
    monkeypatch.setattr(ft.ps1_stack, 'objects', lambda ra, dec, directory=None: cat.copy())
    return ft.ps1_field(0, [(1, 1), (1, 2)]).set_index('PanSTARR_ID')


def test_objects_outside_every_image_are_dropped_and_the_rest_kept(field):
    assert sorted(field.index) == [1, 2, 3]


def test_an_object_in_one_image_takes_that_quadrant(field):
    assert (field.loc[1, 'ccd'], field.loc[1, 'qid']) == (1, 1)
    assert field.loc[1, 'on_ztf'] and all(field.loc[1, f'{b}_covered'] for b in 'gri')


def test_an_overlap_object_takes_the_quadrant_where_it_is_farthest_from_a_masked_pixel(field):
    assert (field.loc[2, 'ccd'], field.loc[2, 'qid']) == (1, 2)
    assert field.loc[2, 'nan_dist_px'] == pytest.approx(15)


def test_a_tie_on_masked_pixels_goes_to_the_quadrant_with_more_valid_bands(field):
    assert field.loc[3, 'nan_dist_px'] == 0
    assert (field.loc[3, 'ccd'], field.loc[3, 'qid']) == (1, 2)
    assert not field.loc[3, 'g_covered'] and field.loc[3, 'r_covered'] and field.loc[3, 'on_ztf']
