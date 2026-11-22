"""field_tables: exact angular distances between catalogued sources (`nearest_other`)."""
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u

from Extracting import field_tables as ft
from Extracting.utils import chord, chord_to_arcsec, sky_xyz


def test_the_unit_vector_distance_matches_astropy_at_any_declination():
    rng = np.random.default_rng(0)
    ra, dec = rng.uniform(0, 360, 200), rng.uniform(-89, 89, 200)
    ra2, dec2 = ra + rng.normal(0, 3 / 3600, 200) / np.cos(np.radians(dec)), dec + rng.normal(0, 3 / 3600, 200)
    ours = chord_to_arcsec(np.linalg.norm(sky_xyz(ra, dec) - sky_xyz(ra2, dec2), axis=1))
    theirs = SkyCoord(ra * u.deg, dec * u.deg).separation(SkyCoord(ra2 * u.deg, dec2 * u.deg)).arcsec
    assert ours == pytest.approx(theirs, abs=1e-6)
    assert chord_to_arcsec(chord(3.0)) == pytest.approx(3.0)


def test_nearest_other_counts_both_surveys_matched_or_not_but_not_the_rows_own_entries():
    off = np.array([0, 2, 10, 1.5]) / 3600
    merged = pd.DataFrame({'ra': np.full(4, 10.0), 'dec': 20.0 + off,
                           'ztf_idx': [0, 1, -1, 2], 'ps1_idx': [-1, -1, 0, 1],               # the last is a matched pair
                           'ps1_ra': [np.nan, np.nan, 10.0, 10.0], 'ps1_dec': [np.nan, np.nan, 20.0 + 10 / 3600, 20.0 + 1.9 / 3600]})
    d = ft.nearest_other(merged)
    assert d == pytest.approx([1.5, 0.1, 8.0, 0.5], abs=1e-3)   # the pair's own PS1 entry (0.4" from its ZTF one) is skipped


def test_nearest_other_is_an_exact_angle_far_from_the_fields_median_declination():
    dec0 = 80.0
    east = 2.5 / 3600 / np.cos(np.radians(dec0))                 # 2.5" due east at Dec 80
    merged = pd.DataFrame({'ra': [150.0, 150.0 + east, 10.0, 20.0, 30.0], 'dec': [dec0, dec0, 70.0, 70.0, 70.0],
                           'ztf_idx': [0, 1, 2, 3, 4], 'ps1_idx': -1, 'ps1_ra': np.nan, 'ps1_dec': np.nan})
    d = ft.nearest_other(merged)
    assert d[:2] == pytest.approx([2.5, 2.5], abs=1e-3)
    assert np.isnan(d[2:]).all()                                  # nothing within 60" of the filler rows
