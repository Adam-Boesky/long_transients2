"""filter_field: each band's (ZTF, PS1) status, the comparison it makes, and the catalog an object lands in."""
import numpy as np
import pandas as pd
import pytest

from Source_Analysis import filter_field as ff

BANDS = ff.BANDS
OFF = ff.NO_CUTS                  # the pipeline with every optional cut off
LAST_CONFIDENT = lambda wf: wf.steps.index('Gaia: no proper motion at 5σ') - 1      # the step before the population cuts
# ZTF codes: D S/N >= 5, S S/N >= 5 with sharp 0.8, L S/N < 5, U no detection, X flagged, N not covered, - no ZTF object (covered)
# PS1 codes: D S/N >= 5, L S/N < 5, F forced flux <= 0, M no measurement, X DEFECT, S ON_SPIKE, Q psfQfPerfect 0.8,
# - no PS1 object
ZTF = {'D': dict(det=True, snr=10.0), 'S': dict(det=True, snr=10.0, sharp=0.8), 'L': dict(det=True, snr=4.0), 'U': dict(det=False), 'X': dict(det=True, snr=10.0, flags=1),
       'N': dict(det=False, covered=False), '-': dict(det=False)}
PS1 = {'D': (1e-5, 1e-6, 0, 0, 1.0), 'L': (3e-6, 1e-6, 0, 0, 1.0), 'F': (-1e-6, 1e-6, 0, 0, 1.0),
       'M': (np.nan, np.nan, 0, 0, np.nan), 'X': (1e-5, 1e-6, ff.DEFECT, 0, 1.0), 'S': (1e-5, 1e-6, 0, ff.ON_SPIKE, 1.0),
       'Q': (1e-5, 1e-6, 0, 0, 0.8), '-': (np.nan, np.nan, 0, 0, np.nan)}


def object_row(ztf: str, ps1: str, n_stack_detections: int = 3) -> dict:
    """One object from three-letter g, r, i status codes per survey, in the columns `object_table` writes."""
    r = {'ra': 10.0, 'dec': 20.0, 'ccd': 1, 'qid': 1, **{f'{b}_{c}': v for b in BANDS for c, v in (('ccd', 1), ('qid', 1))}, 'ps1_qualityFlag': 0, 'ps1_catalog_missing': False, **{f'{b}_ps1_covered': True for b in BANDS}, **{f'{b}_ps1_{c}': 0 for b in BANDS for c in ('projcell', 'skycell')},
         'ps1_nStackDetections': n_stack_detections,
         'ztf_idx': -1 if set(ztf) <= {'-', 'N'} else 0, 'ps1_idx': -1 if set(ps1) == {'-'} else 0,
         **{c: np.nan for c in ('gaia_pm', 'gaia_pmra', 'gaia_pmdec', 'gaia_pmra_err', 'gaia_pmdec_err')}, 'gaia_parallax_over_error': np.nan, 'bright_star_d_over_R': np.inf, 'in_star_spike': False,
         'nearest_other_arcsec': np.nan, 'inside_brighter_kron': False}
    for b, zc, pc in zip(BANDS, ztf, ps1):
        z = {'det': False, 'snr': np.nan, 'flags': 0, 'covered': True, **ZTF[zc]}
        flux, err, flag, flag2, qf = PS1[pc]
        meas = np.isfinite(flux)
        r.update({f'{b}_ztf_snr': z['snr'], f'{b}_ztf_flags': z['flags'], f'{b}_ztf_sharp': z.get('sharp', 0.0),
                  f'{b}_ztf_covered': z['covered'], f'{b}_ztf_nan_dist': 20.0, f'{b}_ztf_kron_masked': 0.0,
                  f'{b}_ztf_psf': 20.0 if z['det'] else np.nan, f'{b}_ztf_psf_err': 0.1, f'{b}_ztf_kron': np.nan,
                  f'{b}_ztf_kron_err': np.nan, f'{b}_ztf_kron_rap': np.nan, f'{b}_ps1_kron_rad': np.nan, f'{b}_ztf_maglim': 21.0,
                  f'{b}_ps1_flux': flux, f'{b}_ps1_flux_err': err, f'{b}_ps1_meas': meas,
                  f'{b}_ps1_snr': flux / err if meas else np.nan, f'{b}_ps1_infoFlag': flag, f'{b}_ps1_infoFlag2': flag2,
                  f'{b}_ps1_psfQfPerfect': qf, f'{b}_ps1_sky': 0.0, f'{b}_ps1_nframes_local': 10.0, f'{b}_ps1_detected': bool(meas and flux > 0 and flux / err >= 5), f'{b}_ps1_nFrames': 10.0,
                  f'{b}_ps1_psf': 20.0 if meas and flux > 0 else np.nan, f'{b}_ps1_psf_err': 0.1,
                  f'{b}_ps1_kron': np.nan, f'{b}_ps1_kron_err': np.nan, f'{b}_sigma': np.nan})
    return r


# (ZTF g r i, PS1 g r i, catalog, comparison per band, survives the >= 2 band rule); the numbers are the cases we discussed
CASES = {
    '1':  ('DDD', 'DDD', 0, (0, 0, 0), True),
    '2':  ('DDD', 'DLL', -1, (-1, -1, -1), False),
    '3 no PS1 measurement': ('DDD', 'DMM', -1, (-1, -1, -1), False),
    '3 forced flux <= 0':   ('DDD', 'DFF', -1, (-1, -1, -1), False),
    '4':  ('DDD', 'LLL', 1, (1, 1, 1), True),
    '4 mirrored': ('LLL', 'DDD', 2, (2, 2, 2), True),
    '5':  ('LDD', 'DDD', 0, (-1, 0, 0), True),
    '6':  ('UDD', 'DDD', 0, (-1, 0, 0), True),
    '7':  ('XDD', 'DDD', 0, (-1, 0, 0), True),
    '8':  ('DDD', 'DDX', 0, (0, 0, -1), True),
    '9':  ('LLD', 'LLD', -1, (-1, -1, -1), False),
    'in_both, one two-sided band': ('DUU', 'DDD', -1, (-1, -1, -1), False),
    'one-sided bands of one kind despite a PS1 detection': ('DDX', 'LLD', 1, (1, 1, -1), True),
    'mixed one-sided kinds': ('DUU', 'LDD', -1, (-1, -1, -1), False),
    '10': ('DDD', '---', 1, (1, 1, 1), True),
    '11': ('DDU', '---', 1, (1, 1, -1), True),
    '12': ('DLU', '---', -1, (-1, -1, -1), False),
    '13': ('---', 'DDD', 2, (2, 2, 2), True),
    '14': ('---', 'DDL', 2, (2, 2, -1), True),
    '15': ('---', 'DMM', -1, (-1, -1, -1), False),
    '16': ('---', 'DDX', 2, (2, 2, -1), True),
    'PS1 band on a spike': ('DDD', 'DDS', 0, (0, 0, -1), True),
    'PS1 band mostly masked': ('DDD', 'DQQ', -1, (-1, -1, -1), False),
    'not covered by ZTF': ('NNN', 'DDD', -1, (-1, -1, -1), False),
    'detected by neither': ('LLL', 'LLL', -1, (-1, -1, -1), False),
}


@pytest.fixture(scope='module')
def src():
    return ff.assign_catalogs(pd.DataFrame([object_row(z, p) for z, p, *_ in CASES.values()], index=list(CASES)), OFF)


@pytest.mark.parametrize('case', CASES)
def test_catalog_and_band_comparisons(src, case):
    _, _, catalog, comparisons, _ = CASES[case]
    assert src.loc[case, 'catalog'] == catalog
    assert tuple(src.loc[case, f'{b}_cat'] for b in BANDS) == comparisons


@pytest.mark.parametrize('case', CASES)
def test_the_chain_keeps_an_object_only_with_two_compared_bands(src, case):
    *_, survives = CASES[case]
    catalog = CASES[case][2]
    wf = ff.run_branch(src, catalog, OFF)[1]                                 # catalog -1: an empty branch
    still_in = sum(wf.mask(0)[b] for b in BANDS) > 0
    assert still_in[list(CASES).index(case)] == survives
    assert (src.loc[case, 'catalog_reason'] == '') == survives
    shared = ff.quality_waterfall(src, OFF)
    assert (sum(shared.mask(len(shared.steps) - 1)[b] for b in BANDS) > 0)[list(CASES).index(case)] == survives


def test_a_flagged_ps1_band_is_never_read_as_a_non_detection(src):
    z, p = ff.band_status(src.loc[['8']], 'i', np.array([True]))
    assert (z[0], p[0]) == ('D', 'X')
    assert ff.detection_status(src.loc[['8']], 'i')[1][0] == 'D' and not ff.usable_sides(src.loc[['8']], 'i', np.array([True]))[1][0]   # a measurement, on an unusable side
    assert ff.compared_magnitudes(src.loc[['8']], 'i')[2][0] != src.loc['8', 'i_ps1_limit']


def test_one_sided_bands_use_the_sub_threshold_measurement_where_brighter_than_the_limit(src):
    used = lambda case, b: ff.compared_magnitudes(src.loc[[case]], b)                                                 # (ztf, ztf_err, ps1, ps1_err)
    assert used('4', 'g')[2][0] == 20.0 and used('4', 'g')[3][0] > 0 and src.loc['4', 'g_ps1_use_forced']                   # PS1 'L' with a forced mag
    assert used('10', 'g')[2][0] == src.loc['10', 'g_ps1_limit'] and not src.loc['10', 'g_ps1_use_forced']                                       # no PS1 object: the limit
    assert used('4 mirrored', 'g')[0][0] == 20.0 and used('4 mirrored', 'g')[1][0] > 0 and src.loc['4 mirrored', 'g_ztf_use_meas']   # ZTF 'L', symmetric
    assert used('13', 'g')[0][0] == 21.0 and used('13', 'g')[1][0] == 0                                                  # no detection: the completeness limit, not MAGLIM
    s = ff.assign_catalogs(pd.DataFrame([object_row('DDD', 'LFF')], index=['forced <= 0']), OFF)                        # in_ztf, no magnitude in r, i: the limit
    z, ze, p, pe = ff.compared_magnitudes(s, 'r')
    assert p[0] == s.loc['forced <= 0', 'r_ps1_limit'] and pe[0] == 0 and s.loc['forced <= 0', 'r_cat'] == 1


def test_an_in_both_object_drops_its_one_sided_bands_and_the_limit_is_never_used_for_it(src):
    for case, band in (('5', 'g'), ('6', 'g')):
        assert src.loc[case, 'catalog'] == 0 and src.loc[case, f'{band}_cat'] == -1
    for case in ('2', '3 forced flux <= 0'):                                       # one two-sided band left: not comparable, no limit used
        assert src.loc[case, 'catalog'] == -1 and src.loc[case, 'catalog_reason'] == 'one two-sided band'
    z = ff.compared_magnitudes(src.loc[['6']], 'g')[0][0]
    assert z != src.loc['6', 'g_ztf_limit'] or np.isnan(z)


def test_an_object_whose_ps1_side_is_a_measurement_needs_two_stack_detections():
    rows = {'in_pstarr, 1 detection': object_row('---', 'DDD', 1), 'in_pstarr, 2 detections': object_row('---', 'DDD', 2),
            'in_both, 1 detection': object_row('DDD', 'DDD', 1), 'in_both, 2 detections': object_row('DDD', 'DDD', 2),
            'in_ztf, 1 detection': object_row('DDD', 'LLL', 1)}
    s = ff.assign_catalogs(pd.DataFrame(list(rows.values()), index=list(rows)), OFF)
    def survivors(catalog):
        wf = ff.run_branch(s, catalog, OFF)[1]
        return sum(wf.mask(LAST_CONFIDENT(wf))[b] for b in BANDS) > 0
    assert list(survivors(2)) == [False, True, False, False, False]
    assert list(survivors(0)) == [False, False, False, True, False]
    assert list(survivors(1)) == [False, False, False, False, True]                  # in_ztf: the PS1 side is a limit, no stack test


def test_a_bright_star_mask_makes_the_ztf_side_unusable_only_when_the_mask_is_on():
    rows = {'inside the circle': {**object_row('DDD', 'DDD'), 'bright_star_d_over_R': 0.5},
            'in a spike': {**object_row('UUU', 'DDD'), 'bright_star_d_over_R': 2.0, 'in_star_spike': True},
            'outside': {**object_row('DDD', 'DDD'), 'bright_star_d_over_R': 1.5}}
    frame = pd.DataFrame(list(rows.values()), index=list(rows))
    on = ff.assign_catalogs(frame, OFF.replace(bright_star_mask=True))
    assert list(on.g_ztf_status) == ['X', 'X', 'D'] and list(on.g_ps1_status) == ['D', 'D', 'D']   # PS1 judges its own side
    assert list(on.catalog) == [-1, -1, 0]                                                       # masked: not comparable
    off = ff.assign_catalogs(frame, OFF)
    assert list(off.catalog) == [0, 2, 0]


def test_bright_star_distance_is_in_units_of_the_star_mask_radius():
    R = float(ff.bright_star_radius(12.0))
    dec0 = 30.0
    offsets = np.array([0.5, 2.0, 10.0]) * R / 3600                        # along declination, in degrees
    src = pd.DataFrame({'ra': np.full(3, 150.0), 'dec': dec0 + offsets})
    stars = pd.DataFrame({'ra': [150.0], 'dec': [dec0], 'phot_g_mean_mag': [12.0]})
    d = ff.annotate_bright_stars(src, stars).bright_star_d_over_R.values
    assert d[:2] == pytest.approx([0.5, 2.0]) and np.isinf(d[2])


def test_the_sharpness_cut_is_object_wide_on_the_snr_weighted_mean():
    rows = {'broad in g only, faint': {**object_row('SDD', 'DDD'), 'g_ztf_snr': 6.0},      # mean 0.8*36/236 ~ 0.12
            'broad in g and r': object_row('SSD', 'DDD'),                                  # mean 0.53
            'no ZTF detection': object_row('---', 'DDD')}
    s = ff.assign_catalogs(pd.DataFrame(list(rows.values()), index=list(rows)), OFF)
    assert ff.ztf_sharpness(s) == pytest.approx([0.8 * 36 / 236, 1.6 / 3, np.nan], nan_ok=True)
    def survivors(catalog, sharp_max):
        wf = ff.run_branch(s, catalog, OFF.replace(sharp_max=sharp_max))[1]
        return list(sum(wf.mask(LAST_CONFIDENT(wf))[b] for b in BANDS) > 0)
    assert survivors(0, None) == [True, True, False]
    assert survivors(0, 0.5) == [True, False, False]                  # removed whole, not band by band
    assert survivors(2, 0.5) == [False, False, True]                  # nothing to judge without a ZTF detection


def test_the_stack_flags_are_confident_checks_of_in_both_and_in_pstarr_and_do_not_touch_in_ztf():
    rows = {'suspect': {**object_row('DDD', 'DDD'), 'ps1_qualityFlag': ff.SUSPECT_STACK}, 'bad': {**object_row('DDD', 'DDD'), 'ps1_qualityFlag': ff.BAD_STACK},
            'clean': object_row('DDD', 'DDD'), 'ztf only, bad': {**object_row('DDD', 'LLL'), 'ps1_qualityFlag': ff.BAD_STACK}}
    s = ff.assign_catalogs(pd.DataFrame(list(rows.values()), index=list(rows)), OFF)
    assert list(s.catalog) == [0, 0, 0, 1]                                                       # the flag no longer sets the status
    wf = ff.run_branch(s, 0, OFF)[1]
    after = lambda step: list(sum(wf.mask(wf.steps.index(step))[b] for b in BANDS) > 0)[:3]
    assert after('PS1: not BAD_STACK') == [True, False, True] and after('PS1: not SUSPECT_STACK') == [False, False, True]
    assert all(step not in ff.run_branch(s, 1, OFF)[1].steps for step in ('PS1: not BAD_STACK', 'PS1: not SUSPECT_STACK'))
    assert list(sum(ff.run_branch(s, 1, OFF)[1].mask(2)[b] for b in BANDS) > 0)[3:] == [True]    # in_ztf keeps it


def test_the_error_floor_adds_in_quadrature_and_caps_snr_consistently():
    assert ff.floor_err(np.array([0.0, 0.1])) == pytest.approx([0.01, np.hypot(0.1, 0.01)])
    snr = np.array([5.0, 50.0, 1e6])
    floored = ff.floor_snr(snr)
    assert floored[0] == pytest.approx(5.0, rel=2e-3)                       # faint detections barely move (5.0 -> 4.995)
    assert floored[2] == pytest.approx(1.0857 / ff.ERR_FLOOR_MAG, rel=1e-3) # bright ones cap at ~109
    flux, err = np.array([1.0, -0.5]), np.array([1e-6, 0.1])
    assert flux[0] / ff.floor_flux_err(flux, err)[0] == pytest.approx(1.0857 / ff.ERR_FLOOR_MAG, rel=1e-3)
    assert ff.floor_flux_err(flux, err)[1] == pytest.approx(np.hypot(0.1, 0.01 / 1.0857 * 0.5))


def test_the_snr_10_step_comes_last_and_re_checks_the_sigma_rule_on_the_strong_bands():
    rows = {'all strong': object_row('DDD', 'DDD'), 'one weak band': object_row('DDD', 'DDD'), 'one-sided weak limit side': object_row('DDD', 'LLL')}
    rows['one weak band'].update({'i_ps1_snr': 7.0})
    s = ff.assign_catalogs(pd.DataFrame(list(rows.values()), index=list(rows)), OFF)
    for b in BANDS:
        s[f'{b}_sigma'] = 4.0                                              # every compared band stands out
    s.loc['one weak band', 'g_sigma'] = 0.0                                # ... except g: only r, i carry it
    weak = ff.run_branch(s, 0, OFF.replace(snr_keep=10.0))[1]
    assert weak.steps[-3:] == ['measured sides S/N >= 10', '|σ| > 3 in >= 2 bands at that S/N', '... >= 2 of them the same sign at that S/N']
    assert list(ff.run_branch(s, 0, OFF)[0].index) == ['all strong', 'one weak band']               # off: both are candidates
    assert list(ff.run_branch(s, 0, OFF.replace(snr_keep=10.0))[0].index) == ['all strong']        # on: i goes, g never stood out
    assert list(ff.run_branch(s, 1, OFF.replace(snr_keep=10.0))[0].index) == ['one-sided weak limit side']   # a limit side has no S/N to test


def test_a_compared_band_needs_snr_10_on_each_measured_side_but_not_on_a_limit():
    rows = {'two-sided, PS1 at 7': object_row('DDD', 'DDD'), 'one-sided, ZTF at 7': object_row('DDD', 'LLL'),
            'one-sided, ZTF at 12': object_row('DDD', 'LLL')}
    rows['two-sided, PS1 at 7'].update({'g_ps1_snr': 7.0})
    rows['one-sided, ZTF at 7'].update({'g_ztf_snr': 7.0})
    rows['one-sided, ZTF at 12'].update({f'{b}_ztf_snr': 12.0 for b in BANDS})
    s = ff.assign_catalogs(pd.DataFrame(list(rows.values()), index=list(rows)), OFF)
    assert ff.measured_sides_snr_ok(s, 'g').all()                     # off by default
    ok = ff.measured_sides_snr_ok(s, 'g', 10.0)
    assert list(ok) == [False, False, True]                           # the PS1 limit side of a one-sided band has no S/N to test
    assert tuple(s.loc['two-sided, PS1 at 7', [f'{b}_cat' for b in BANDS]]) == (0, 0, 0)   # sorting is still at S/N 5


def test_faint_envelope_edges_are_the_running_extremes_and_the_median_stays():
    mag_grid, dmag_grid = np.arange(19, 23, 1.0), np.linspace(-5, 5, 4001)
    centre, width = np.array([0.0, 0.1, -0.2, 0.0]), np.array([0.2, 0.3, 0.15, 0.25])    # rows at 21 and 22 pull in
    ys = np.exp(-0.5 * ((dmag_grid[None] - centre[:, None]) / width[:, None]) ** 2)
    out = ff.widen_faint_rows(ys, mag_grid, dmag_grid, from_mag=20.0)
    before = np.array([ff._row_edges(r, dmag_grid, (3.0, 2.0, 1.0)) [:2] for r in ys])    # (row, lo/hi, level)
    after = np.array([ff._row_edges(r, dmag_grid, (3.0, 2.0, 1.0))[:2] for r in out])
    assert np.allclose(after[2:, 0], np.minimum.accumulate(before[:, 0])[2:], atol=5e-3)   # lower edges: running minimum
    assert np.allclose(after[2:, 1], np.maximum.accumulate(before[:, 1])[2:], atol=5e-3)   # upper edges: running maximum
    assert [ff._row_edges(r, dmag_grid, (1.0,))[2] for r in out] == pytest.approx(centre, abs=5e-3)
    assert np.array_equal(out[:2], ys[:2])


def test_the_ps1_sky_cut_is_off_by_default_and_marks_a_high_sky_ps1_band_untrustworthy():
    row = {**object_row('DDD', 'DDD'), 'g_ps1_sky': 3e-7}
    rows = pd.DataFrame([row], index=['high sky in g'])
    assert tuple(ff.assign_catalogs(rows, OFF).loc['high sky in g', [f'{b}_cat' for b in BANDS]]) == (0, 0, 0)
    s = ff.assign_catalogs(rows, OFF.replace(ps1_sky_max=1e-7))
    assert s.loc['high sky in g', 'g_ps1_status'] == 'X' and tuple(s.loc['high sky in g', [f'{b}_cat' for b in BANDS]]) == (-1, 0, 0)
    wf = ff.quality_waterfall(s, OFF.replace(ps1_sky_max=1e-7))                  # the shared stage reproduces it (asserted inside)
    assert 'PS1: residual sky <= 1e-07' in wf.steps


def test_the_nframes_cut_is_off_by_default_and_marks_a_few_frame_ps1_band_untrustworthy():
    rows = pd.DataFrame([{**object_row('DDD', 'DDD'), 'g_ps1_nFrames': 3.0, 'r_ps1_nFrames': 0.0}], index=['few frames'])
    assert tuple(ff.assign_catalogs(rows, OFF).loc['few frames', [f'{b}_cat' for b in BANDS]]) == (0, 0, 0)
    s = ff.assign_catalogs(rows, OFF.replace(ps1_min_nframes=4))
    assert tuple(s.loc['few frames', [f'{b}_ps1_status' for b in BANDS]]) == ('X', 'X', 'D')
    wf = ff.quality_waterfall(s, OFF.replace(ps1_min_nframes=4))                  # the chain reproduces it (asserted inside)
    assert 'PS1: stack nFrames >= 4' in wf.steps


def test_isolation_cut_removes_one_sided_objects_with_any_other_object_within_3_arcsec_only():
    rows = {'in_ztf, other at 2"': {**object_row('DDD', '---'), 'nearest_other_arcsec': 2.0},
            'in_ztf, other at 0.5"': {**object_row('DDD', '---'), 'nearest_other_arcsec': 0.5},
            'in_ztf, other at 5"': {**object_row('DDD', '---'), 'nearest_other_arcsec': 5.0},
            'in_pstarr, other at 2"': {**object_row('---', 'DDD'), 'nearest_other_arcsec': 2.0},
            'in_both, other at 2"': {**object_row('DDD', 'DDD'), 'nearest_other_arcsec': 2.0}}
    s = ff.assign_catalogs(pd.DataFrame(list(rows.values()), index=list(rows)), OFF)
    alive = lambda catalog, iso: list(sum(ff.cut_bands(s, catalog, OFF.replace(isolation_arcsec=iso))[0].mask(-1)[b] for b in BANDS) > 0)
    assert alive(1, 3.0) == [False, False, True, False, False] and alive(1, None) == [True, True, True, False, False]
    assert alive(2, 3.0) == [False] * 5 and alive(2, None) == [False, False, False, True, False]
    assert alive(0, 3.0) == [False, False, False, False, True]                # in_both is never cut


def test_nearest_other_object_counts_both_surveys_matched_or_not_but_not_the_objects_own_entries():
    off = np.array([0, 2, 10, 1.5]) / 3600
    src = pd.DataFrame({'ra': np.full(4, 10.0), 'dec': 20.0 + off,
                        'ztf_idx': [0, 1, -1, 2], 'ps1_idx': [-1, -1, 0, 1],               # the last is a matched pair
                        'ps1_ra': [np.nan, np.nan, 10.0, 10.0], 'ps1_dec': [np.nan, np.nan, 20.0 + 10 / 3600, 20.0 + 1.9 / 3600],
                        **{f'{b}_ps1_meas': [False, False, True, True] for b in BANDS}})
    for b in BANDS:
        src[f'{b}_ztf_psf'] = src[f'{b}_ps1_psf'] = src[f'{b}_ztf_kron_rap'] = src[f'{b}_ps1_kron_rad'] = np.nan
    d = ff.annotate_neighbours(src).nearest_other_arcsec.values
    assert d == pytest.approx([1.5, 0.1, 8.0, 0.5], abs=1e-3)   # the pair's own PS1 entry (0.4" from its ZTF one) is skipped


def test_the_completeness_limit_is_where_recovery_crosses_the_threshold_and_never_deeper_than_the_cap():
    rng = np.random.default_rng(0)
    mag = rng.uniform(18, 24, 20000)
    recovered = rng.random(len(mag)) < np.clip(1 - (mag - 21) / 2, 0, 1)       # recovery 1 at 21, 0.5 at 22, 0 at 23
    quad = np.full(len(mag), 11)
    expected = 21 + 2 * (1 - ff.LIMIT_COMPLETENESS)                            # where the synthetic curve crosses the threshold
    assert ff.completeness_limit(mag, recovered, quad, {(1, 1): 23.6})[(1, 1)] == pytest.approx(expected, abs=0.15)
    assert ff.completeness_limit(mag, recovered, quad, {(1, 1): 20.5})[(1, 1)] == 20.5
    assert ff.completeness_limit(mag, recovered, quad, {(1, 1): 23.6, (1, 2): 23.0})[(1, 2)] == pytest.approx(expected, abs=0.15)   # an empty quadrant takes the field's value
    bright = mag < 21.6                                                         # the reference runs out while still complete
    assert ff.completeness_limit(mag[bright], recovered[bright], quad[bright], {(1, 1): 23.6})[(1, 1)] == pytest.approx(21.5, abs=0.3)


def _reference_field(rng, n=6000, kron_offset=0.0):
    """ZTF objects over 18-24 mag, PS1 recovering a fraction of them that falls from 1 at 21 to 0 at 23, and the
    mirror image (PS1 objects that ZTF recovers); Kron `kron_offset` brighter than PSF (0: point sources); one quadrant."""
    mag = rng.uniform(18, 24, n)
    rec = rng.random(n) < np.clip(1 - (mag - 21) / 2, 0, 1)
    rows = [object_row('DDD', 'DDD' if r else '---') for r in rec] + [object_row('DDD' if r else 'UUU', 'DDD') for r in rec]
    s = pd.DataFrame(rows)
    for b in BANDS:
        s[f'{b}_ztf_psf'] = np.where(ff.ztf_detected(s, b), np.r_[mag, mag], np.nan); s[f'{b}_ztf_kron'] = s[f'{b}_ztf_psf'] - kron_offset; s[f'{b}_ztf_kron_err'] = 0.1
        s[f'{b}_ps1_psf'] = np.where(s[f'{b}_ps1_detected'], np.r_[mag, mag], np.nan); s[f'{b}_ps1_kron'] = s[f'{b}_ps1_psf'] - kron_offset; s[f'{b}_ps1_kron_err'] = 0.1
        s[f'{b}_ztf_maglim'] = 23.6; s[f'{b}_ps1_nFrames'] = 30.0
    return s, mag, rec


def test_completeness_limits_run_both_ways_in_the_magnitude_each_reference_object_compares(monkeypatch):
    s, mag, rec = _reference_field(np.random.default_rng(0))
    expected = 21 + 2 * (1 - ff.LIMIT_COMPLETENESS)
    out = ff.completeness_limits(s, OFF)                                       # point sources: PSF
    for survey in ('ztf', 'ps1'):
        assert out[f'g_{survey}_limit'].iloc[0] == pytest.approx(expected, abs=0.15)
    ext, _, _ = _reference_field(np.random.default_rng(0), kron_offset=0.5)    # every reference extended: binned by Kron, 0.5 brighter
    ext_out = ff.completeness_limits(ext, OFF)
    assert not ff.point_source(ext).any()
    for survey in ('ztf', 'ps1'):
        assert ext_out[f'g_{survey}_limit'].iloc[0] == pytest.approx(expected - 0.5, abs=0.15)
    monkeypatch.setattr(ff, 'ps1_depth', lambda src, b: np.full(len(src), 20.0))   # a nominal depth shallower than measured caps it
    assert (ff.completeness_limits(s, OFF).g_ps1_limit == 20.0).all()


def test_the_reference_excludes_detections_that_fail_their_own_checks_but_not_the_other_surveys():
    s, mag, rec = _reference_field(np.random.default_rng(0))
    expected = 21 + 2 * (1 - ff.LIMIT_COMPLETENESS)
    junk = pd.DataFrame([object_row('XXX', '---') for _ in range(len(mag) // 2)])     # flagged ZTF detections PS1 never recovers, 18-21 mag
    for b in BANDS:
        junk[f'{b}_ztf_psf'] = np.linspace(18, 21, len(junk)); junk[f'{b}_ztf_kron'] = junk[f'{b}_ztf_psf']; junk[f'{b}_ztf_kron_err'] = 0.1
        junk[f'{b}_ztf_maglim'] = 23.6; junk[f'{b}_ps1_nFrames'] = 30.0
    assert ff.completeness_limits(pd.concat([s, junk], ignore_index=True), OFF).g_ps1_limit.iloc[0] == pytest.approx(expected, abs=0.15)
    junk_rows = np.flatnonzero((mag < 21.5) & (np.arange(len(mag)) % 2 == 0))                      # half the bright ZTF objects: under a mask, unrecovered
    masked = s.copy(); masked.loc[masked.index[junk_rows], 'bright_star_d_over_R'] = 0.5
    masked.loc[masked.index[junk_rows], ['g_ps1_detected', 'ps1_idx']] = [False, -1]; masked.loc[masked.index[junk_rows], 'g_ps1_psf'] = np.nan
    assert ff.completeness_limits(masked, OFF).g_ps1_limit.iloc[0] < 20.0                      # mask off: they read as PS1 incompleteness
    assert ff.completeness_limits(masked, OFF.replace(bright_star_mask=True)).g_ps1_limit.iloc[0] == pytest.approx(expected, abs=0.3)
    bad = s.copy(); bad['ps1_qualityFlag'] = ff.BAD_STACK                        # a recovering-side flag does not touch the ZTF reference
    assert ff.completeness_limits(bad, OFF).g_ps1_limit.iloc[0] == pytest.approx(ff.completeness_limits(s, OFF).g_ps1_limit.iloc[0], abs=1e-6)


def test_a_band_records_whether_it_compares_psf_or_kron_and_the_magnitudes_follow():
    s, mag, rec = _reference_field(np.random.default_rng(0))
    point, extended = object_row('DDD', '---'), object_row('DDD', '---')
    for r in (point, extended):
        r.update({f'{b}_ztf_kron_err': 0.1 for b in BANDS}); r.update({f'{b}_ztf_maglim': 23.6 for b in BANDS}); r.update({f'{b}_ps1_nFrames': 30.0 for b in BANDS})
    point.update({f'{b}_ztf_kron': 20.0 for b in BANDS}); extended.update({f'{b}_ztf_kron': 19.0 for b in BANDS})
    table = pd.concat([s, pd.DataFrame([point, extended], index=['point', 'extended'])])
    out = ff.assign_catalogs(table, OFF)
    assert list(out.loc[['point', 'extended'], 'g_use_psf']) == [True, False]
    z, _, p, _ = ff.compared_magnitudes(out.loc[['point', 'extended']], 'g')
    assert list(z) == [20.0, 19.0] and list(p) == list(out.loc[['point', 'extended'], 'g_ps1_limit'])    # in_ztf: the PS1 side is the limit


def test_annotate_neighbours_kron_containment_is_against_brighter_objects_only():
    src = pd.DataFrame({'ra': [10.0, 10.0, 10.0], 'dec': [20.0, 20.0 + 2.0 / 3600, 20.0 + 2.5 / 3600],
                        'ztf_idx': [0, 1, 2], 'ps1_idx': [-1, -1, -1], 'ps1_ra': np.nan, 'ps1_dec': np.nan, **{f'{b}_ps1_meas': False for b in BANDS}})
    for b in BANDS:
        src[f'{b}_ztf_psf'] = [17.0, 21.0, 16.0]; src[f'{b}_ps1_psf'] = np.nan
        src[f'{b}_ztf_kron_rap'] = [3.0, 3.0, np.nan]; src[f'{b}_ps1_kron_rad'] = np.nan
    out = ff.annotate_neighbours(src)
    assert list(out.inside_brighter_kron) == [False, True, False]       # only the faint object sits in a brighter object's aperture


def test_an_object_is_not_masked_by_its_own_star_entry():
    src = pd.DataFrame({'ra': [150.0, 150.0], 'dec': [30.0, 30.0 + 4.0 / 3600]})
    stars = pd.DataFrame({'ra': [150.0], 'dec': [30.0], 'phot_g_mean_mag': [18.0]})      # the first object itself, as a PS1 point source
    d = ff.annotate_bright_stars(src, stars).bright_star_d_over_R.values
    assert np.isinf(d[0]) and d[1] == pytest.approx(1.0, abs=0.01)                      # 4" away in units of the 18 mag radius (4")


def test_a_limit_only_supports_a_change_in_the_direction_it_constrains():
    class Env:                                                                  # a stand-in envelope: sigma = dmag / 0.1
        def sigma(self, mag, dmag):
            return np.asarray(dmag) / 0.1, np.zeros(len(dmag), bool)
    rows = {'in_ztf brighter than the PS1 limit': object_row('DDD', '---'), 'in_ztf fainter than the PS1 limit': object_row('DDD', '---'),
            'in_pstarr brighter than the ZTF limit': object_row('UUU', 'DDD'), 'in_pstarr fainter than the ZTF limit': object_row('UUU', 'DDD'),
            'in_ztf against a forced measurement': object_row('DDD', 'LLL')}
    rows['in_ztf fainter than the PS1 limit'].update({f'{b}_ztf_psf': 23.5 for b in BANDS})         # PS1 depth ~23.2 (10 frames)
    rows['in_pstarr fainter than the ZTF limit'].update({f'{b}_ps1_psf': 21.5 for b in BANDS})      # ZTF MAGLIM 21.0
    rows['in_ztf against a forced measurement'].update({f'{b}_ps1_psf': 20.5 for b in BANDS})       # ZTF 20: fainter in PS1, two-sided
    s = ff.assign_catalogs(pd.DataFrame(list(rows.values()), index=list(rows)), OFF)
    out = ff.annotate_sigma(s, {b: Env() for b in BANDS})
    lim = s.loc['in_ztf brighter than the PS1 limit', 'g_ps1_limit']
    assert out.g_sigma.round(1).tolist() == [round((20.0 - lim) / 0.1, 1), 0.0, 10.0, 0.0, -5.0]


def test_every_step_carries_its_group_in_pipeline_order():
    s = ff.assign_catalogs(pd.DataFrame([object_row('DDD', 'DDD'), object_row('DDD', '---')]), OFF)
    for b in BANDS:
        s[f'{b}_sigma'] = 4.0
    wf = ff.run_branch(s, 0, OFF.replace(bright_star_mask=True, snr_keep=10.0))[1]
    order = list(ff.GROUPS)
    assert wf.groups[0] == 'structure' and [order.index(g) for g in wf.groups] == sorted(order.index(g) for g in wf.groups)
    assert set(wf.groups) == {'structure', 'confident', 'population', 'interesting', 'best'}
    shared = ff.quality_waterfall(s, OFF)
    assert shared.groups[0] == 'structure' and set(shared.groups) == {'structure', 'quality', 'comparison'} and shared.steps[-1] == ff.COMPARABLE
    assert 'group' in wf.stats.columns and wf.stats.group.iloc[-1] == 'best'


def test_a_position_in_a_bad_polar_skycell_has_no_usable_ps1_side_with_or_without_a_ps1_object():
    pole = {'ra': 157.9, 'dec': 73.85}                                          # inside bad skycell 2583.046; 2583.056 is good
    bad = {**pole, **{f'{b}_ps1_{c}': v for b in BANDS for c, v in (('projcell', 2583), ('skycell', 46))}}
    rows = {'in_both, bad cell': {**object_row('DDD', 'DDD'), **bad}, 'in_ztf, bad cell': {**object_row('DDD', '---'), **bad},
            'in_pstarr, bad cell': {**object_row('UUU', 'DDD'), **bad}, 'in_ztf, good cell': object_row('DDD', '---'),
            'in_both, g from a bad cell': {**object_row('DDD', 'DDD'), **pole, **{f'{b}_ps1_{c}': v for b in BANDS for c, v in (('projcell', 2583), ('skycell', 56))},
                                           'g_ps1_skycell': 46}}
    s = ff.assign_catalogs(pd.DataFrame(list(rows.values()), index=list(rows)), OFF)
    assert list(s.g_ps1_status) == ['X', 'X', 'X', 'U', 'X'] and list(s.g_cat) == [-1, -1, -1, 1, -1]
    assert list(s.r_ps1_status) == ['X', 'X', 'X', 'U', 'D'] and list(s.r_cat) == [-1, -1, -1, 1, 0]      # only the bad band goes
    assert list(s.catalog) == [-1, -1, -1, 1, 0]                                 # nothing comparable in a bad skycell
    assert 'PS1: not in a flagged polar skycell' in [l for l, _ in ff.ps1_checks(s, 'g', OFF)]


def test_skycell_lookup_matches_mast_and_the_bad_list():
    from Extracting import ps1_skycells as sc
    proj, cell = sc.skycell_of([157.9, 10.0], [73.85, 20.0])                    # MAST's ps1filenames.py: 2583 / 046 (badflag 1)
    assert (proj[0], cell[0]) == (2583, 46) and sc.is_bad([157.9], [73.85])[0] and not sc.is_bad([10.0], [20.0])[0]
    assert len(sc.bad_cells()) == 461 and (2583, 46) in sc.bad_cells()
    centres = [(l.split()[0], float(l.split()[1]), float(l.split()[2])) for l in open(sc.BAD_CELLS_FILE) if l.startswith('skycell.')]
    names = [tuple(int(v) for v in n.split('.')[1:]) for n, _, _ in centres]
    cover = sc.covering_skycells([r for _, r, _ in centres], [d for _, _, d in centres])
    below_cap = np.array([d < 84 for _, _, d in centres])                      # the tessellation is special-cased at the pole
    hit = np.array([n in {(int(p), int(c)) for p, c in row} for n, row in zip(names, cover)])
    assert hit[below_cap].mean() > 0.99                                         # every listed centre lies in its own skycell
    assert sc.is_bad([r for _, r, _ in centres][:50], [d for _, _, d in centres][:50]).all()


def test_the_skycell_check_judges_each_band_by_its_own_row_and_is_liberal_without_a_ps1_object():
    s = pd.DataFrame({'ra': [157.9] * 3, 'dec': [73.85] * 3, 'ps1_idx': [0, 1, -1],
                      'g_ps1_projcell': [2583, 2583, -1], 'g_ps1_skycell': [46, 56, -1],                  # 046 is bad, 056 is not
                      'r_ps1_projcell': [2583, 2583, -1], 'r_ps1_skycell': [56, 46, -1],
                      'i_ps1_projcell': [2583, 2583, -1], 'i_ps1_skycell': [56, 56, -1]})
    assert list(ff.ps1_skycell_bad(s, 'g')) == [True, False, True] and list(ff.ps1_skycell_bad(s, 'r')) == [False, True, True]
    assert list(ff.ps1_skycell_bad(s, 'i')) == [False, False, True] and list(ff.ps1_skycell_bad(s)) == [True, True, True]


def test_the_sharpness_cut_can_be_limited_to_point_sources():
    rows = {'broad point source': object_row('SSS', 'DDD'), 'broad extended': object_row('SSS', 'DDD')}
    for r in rows.values():
        r.update({f'{b}_ztf_kron_err': 0.1 for b in BANDS}); r.update({f'{b}_ps1_kron_err': 0.1 for b in BANDS})
    rows['broad point source'].update({f'{b}_ztf_kron': 20.0 for b in BANDS}); rows['broad extended'].update({f'{b}_ztf_kron': 19.0 for b in BANDS})
    s = ff.assign_catalogs(pd.DataFrame(list(rows.values()), index=list(rows)), OFF)
    assert list(s.point_source) == [True, False]
    both = ff.run_branch(s, 0, OFF.replace(sharp_max=0.5))[1]; point_only = ff.run_branch(s, 0, OFF.replace(sharp_max=0.5, sharp_point_only=True))[1]
    survive = lambda wf: list(sum(wf.mask(LAST_CONFIDENT(wf))[b] for b in BANDS) > 0)
    assert survive(both) == [False, False] and survive(point_only) == [False, True]
    assert 'ZTF: S/N²-weighted |sharp| <= 0.5 (point sources)' in point_only.steps


def test_spike_boxes_mask_along_the_four_position_angles_out_to_the_half_length_and_not_across():
    G = 10.0; R = ff.bright_star_radius(G); L = float(ff.spike_half_length(G)); W = ff.SPIKE_WIDTH_ARCSEC
    assert L == pytest.approx(R * ff.SPIKE_GROWTH ** 6) and L > 2 * R
    dec0, cosd = 30.0, np.cos(np.radians(30.0))
    def at(d_arcsec, pa, across=0.0):                                               # a position d along PA with an offset across
        th = np.radians(pa); dx = d_arcsec * np.sin(th) + across * np.cos(th); dy = d_arcsec * np.cos(th) - across * np.sin(th)
        return 150.0 + dx / 3600 / cosd, dec0 + dy / 3600
    pts = [at(0.8 * L, 0), at(0.8 * L, 90), at(0.8 * L, 30), at(0.8 * L, 150),              # along each spike, beyond the circle: masked
           at(1.2 * L, 0), at(0.8 * L, 60), at(0.8 * L, 0, across=W)]                      # past the tip, between spikes, off to the side: not
    src = pd.DataFrame({'ra': [p[0] for p in pts], 'dec': [p[1] for p in pts]})
    stars = pd.DataFrame({'ra': [150.0], 'dec': [dec0], 'phot_g_mean_mag': [G]})
    out = ff.annotate_bright_stars(src, stars)
    assert list(out.in_star_spike) == [True, True, True, True, False, False, False]
    assert list(ff.bright_star_masked(out)) == [True, True, True, True, False, False, False]
    assert out.bright_star_d_over_R.iloc[0] == pytest.approx(0.8 * L / R, rel=1e-3) and 0.8 * L / R > 1   # along the spike but outside the circle: the box did it


def test_outcomes_record_the_first_step_that_dropped_each_band_and_where_the_object_left():
    src = pd.DataFrame({'ra': [1.0, 2.0, 3.0], 'dec': [0.0, 0.0, 0.0], 'catalog': [0, -1, 0], 'catalog_reason': ['', 'no comparable band', '']})
    shared = ff.Waterfall('all')
    shared.add('>= 2 bands: start', {'g': np.array([1, 1, 1], bool), 'r': np.array([1, 1, 1], bool), 'i': np.array([1, 0, 1], bool)})
    shared.add('quality A', {'g': np.array([1, 0, 1], bool), 'r': np.array([1, 0, 1], bool), 'i': np.array([0, 0, 1], bool)}, 'quality')
    branch = ff.Waterfall('in_both')
    branch.add(ff.COMPARABLE, {'g': np.array([1, 0, 1], bool), 'r': np.array([1, 0, 1], bool), 'i': np.array([0, 0, 1], bool)})
    branch.add('confident B', {'g': np.array([1, 0, 0], bool), 'r': np.array([1, 0, 1], bool), 'i': np.array([0, 0, 1], bool)}, 'confident')
    branch.add('|σ| step', {'g': np.array([1, 0, 0], bool), 'r': np.array([1, 0, 0], bool), 'i': np.array([0, 0, 0], bool)}, 'interesting')
    out = ff.annotate_outcomes(src, {'all': shared, 'in_both': branch}, {'in_both': src.iloc[[0]]})
    assert list(out.outcome) == ['candidate', 'not comparable: no comparable band', 'dropped: |σ| step']
    assert list(out.g_dropped_by) == [ff.KEPT, 'quality A', 'confident B']
    assert list(out.i_dropped_by) == ['quality A', '>= 2 bands: start', '|σ| step']
    assert list(out.r_dropped_by) == [ff.KEPT, 'quality A', '|σ| step']


def test_a_position_where_dr2_has_no_catalog_gives_the_ps1_side_nothing_to_say():
    from Extracting import ps1_skycells
    assert list(ps1_skycells.catalog_missing([200.0, 200.0, 10.0, 215.0, 76.5], [35.48, 35.6, -31.0, -33.0, -27.0])) == [True, False, True, True, True]
    rows = {'ztf only, in a box': {**object_row('DDD', '---'), 'ps1_catalog_missing': True}, 'ztf only, covered': object_row('DDD', '---')}
    s = ff.assign_catalogs(pd.DataFrame(list(rows.values()), index=list(rows)), OFF)
    assert list(s.catalog) == [-1, 1] and 'PS1: catalog covers the position' in [l for l, _ in ff.ps1_checks(s, 'g', OFF)]


def test_without_a_ps1_object_a_band_needs_ps1_objects_nearby_for_its_limit_to_stand_in():
    rows = {'ztf only, PS1 reaches it': object_row('DDD', '---'), 'ztf only, no PS1 objects near in g': {**object_row('DDD', '---'), 'g_ps1_covered': False},
            'matched, flag irrelevant': {**object_row('DDD', 'DDD'), 'g_ps1_covered': False}}
    s = ff.assign_catalogs(pd.DataFrame(list(rows.values()), index=list(rows)), OFF)
    assert list(s.g_ps1_status) == ['U', 'X', 'D'] and list(s.catalog) == [1, 1, 0]            # r and i still carry the second object
    assert "PS1: objects within 1' in the band" in [l for l, _ in ff.ps1_checks(s, 'g', OFF)]


def test_the_proper_motion_cut_uses_the_components_errors_in_quadrature():
    s = pd.DataFrame({'gaia_pmra': [3.0, 3.0, np.nan], 'gaia_pmdec': [4.0, 4.0, np.nan], 'gaia_pmra_err': [0.5, 1.5, np.nan],
                      'gaia_pmdec_err': [0.5, 1.5, np.nan], 'gaia_parallax_over_error': [1.0, 1.0, np.nan]})
    assert ff.gaia_pm_significance(s)[:2] == pytest.approx([10.0, 10 / 3])       # pm 5, sigma 0.5 and 1.5
    pm_ok, plx_ok = ff.gaia_population_ok(s)
    assert list(pm_ok) == [False, True, True] and list(plx_ok) == [True, True, True]   # no Gaia match passes


def test_the_ps1_depth_uses_the_objects_own_frames_and_the_local_estimate_only_without_them():
    s = pd.DataFrame({'g_ps1_nFrames': [12.0, np.nan, 0.0], 'g_ps1_nframes_local': [3.0, 3.0, 3.0]})
    assert list(ff.ps1_nframes(s, 'g')) == [12.0, 3.0, 3.0]
