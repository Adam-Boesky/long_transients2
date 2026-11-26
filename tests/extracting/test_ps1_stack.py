"""ps1_stack: reducing PS1 stack measurements to one row per object, without losing or inventing objects."""
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from Extracting import ps1_stack
from Extracting.ps1_stack import BANDS, BAND_COLS, OBJECT_COLS, STACK_PHOT_SRC, STACK_PRIMARY

DETECTED = dict(PSFMag=20.0, PSFFlux=1e-5, PSFFluxErr=1e-6, infoFlag2=0)      # S/N 10
FORCED = dict(PSFMag=23.5, PSFFlux=1e-7, PSFFluxErr=1e-7, infoFlag2=4)
MISSING = dict(PSFMag=-999.0, PSFFlux=-999.0, PSFFluxErr=-999.0, infoFlag2=0)


def row(obj_id, sid, primary=1, phot_src=None, ra=10.0, dec=20.0, **bands):
    """One stack measurement. `primary` sets STACK_PRIMARY and `phot_src` (default: as `primary`) STACK_PHOT_SRC in
    every band's infoFlag3; `bands` maps a band to a dict of overrides (default: DETECTED)."""
    phot_src = primary if phot_src is None else phot_src
    r = {c: 0 for c in OBJECT_COLS}
    r.update(objID=obj_id, uniquePspsSTid=sid, _healpix_29=0, raMean=ra, decMean=dec, raStack=ra, decStack=dec,
             primaryDetection=primary, epochMean=56000.0)
    for b in BANDS:
        r.update({f'{b}{c}': 1.0 for c in BAND_COLS})
        r[f'{b}infoFlag3'] = (STACK_PRIMARY if primary else 0) | (STACK_PHOT_SRC if phot_src else 0)
        r.update({f'{b}{k}': v for k, v in {**DETECTED, **bands.get(b, {})}.items()})
    return r


def clean(*rows):
    return ps1_stack.clean(pd.DataFrame(list(rows))).set_index('PanSTARR_ID')


def test_single_measurement_passes_through():
    out = clean(row(1, 101))
    assert list(out.index) == [1]
    assert out.loc[1, 'n_rows'] == 1 and out.loc[1, 'n_primary_rows'] == 1
    for b in BANDS:
        assert out.loc[1, f'{b}_row'] == 101 and out.loc[1, f'{b}_detected']


def test_overlap_keeps_the_primary_detection_even_when_a_secondary_has_higher_snr():
    out = clean(row(1, 101, primary=1, g=dict(PSFFlux=5e-6)),                     # S/N 5
                row(1, 102, primary=0, g=dict(PSFFlux=5e-5)))                     # S/N 50
    assert out.loc[1, 'g_row'] == 101
    assert out.loc[1, 'n_rows'] == 2


@pytest.mark.parametrize('primary_band', [FORCED, MISSING], ids=['forced', 'missing'])
def test_band_comes_from_a_detected_secondary_when_the_primary_is_not_a_detection(primary_band):
    out = clean(row(1, 101, primary=1, phot_src=0, g=primary_band),          # no row is PS1's photometry source
                row(1, 102, primary=0))
    assert out.loc[1, 'g_row'] == 102 and out.loc[1, 'g_detected']
    assert out.loc[1, 'r_row'] == 101 and out.loc[1, 'i_row'] == 101           # other bands stay on the primary
    assert out.loc[1, 'qualityFlag'] == 0 and out.loc[1, 'ra'] == 10.0


def test_forced_only_band_keeps_the_primary_forced_measurement():
    out = clean(row(1, 101, primary=1, r=FORCED), row(1, 102, primary=0, r=FORCED))
    assert out.loc[1, 'r_row'] == 101 and not out.loc[1, 'r_detected']
    assert out.loc[1, 'rPSFMag'] == FORCED['PSFMag']


def test_object_without_a_primary_row_is_kept():
    out = clean(row(1, 101, primary=0, g=FORCED, r=FORCED), row(1, 102, primary=0))
    assert list(out.index) == [1]
    assert out.loc[1, 'n_primary_rows'] == 0
    assert out.loc[1, 'g_row'] == 102 and out.loc[1, 'g_detected']


def test_two_primary_rows_take_the_higher_snr_detection_and_are_counted():
    out = clean(row(1, 101, primary=1, g=dict(PSFFlux=3e-6)), row(1, 102, primary=1, g=dict(PSFFlux=4e-5)))
    assert out.loc[1, 'n_primary_rows'] == 2
    assert out.loc[1, 'g_row'] == 102


def test_object_columns_come_from_the_primary_row():
    out = clean(row(1, 101, primary=0, ra=10.001), row(1, 102, primary=1, ra=10.0))
    assert out.loc[1, 'ra'] == 10.0 and out.loc[1, 'primaryDetection'] == 1


def test_position_is_the_stack_position_with_the_mean_as_fallback():
    out = clean({**row(1, 101), 'raStack': 10.0001, 'decStack': 20.0002},
                {**row(2, 201), 'raStack': -999.0, 'decStack': -999.0})
    assert (out.loc[1, 'ra'], out.loc[1, 'dec']) == (10.0001, 20.0002)
    assert (out.loc[1, 'raMean'], out.loc[1, 'decMean']) == (10.0, 20.0)
    assert (out.loc[2, 'ra'], out.loc[2, 'dec']) == (10.0, 20.0)


def test_a_stack_row_joined_twice_counts_once():
    dup = row(1, 101)
    out = clean(dup, {**dup, 'raMean': 10.00001}, row(1, 102, primary=0))
    assert out.loc[1, 'n_rows'] == 2


def test_missing_values_become_nan_but_forced_values_stay():
    out = clean(row(1, 101, g=MISSING, r=FORCED))
    assert np.isnan(out.loc[1, 'gPSFMag']) and np.isnan(out.loc[1, 'gPSFFlux'])
    assert out.loc[1, 'rPSFMag'] == FORCED['PSFMag'] and not out.loc[1, 'r_detected']


def test_every_object_appears_exactly_once_and_ps1s_photometry_row_is_taken_when_there_is_one():
    rng = np.random.default_rng(0)
    rows, sid = [], 0
    for obj_id in range(1, 401):
        n = rng.integers(1, 5)
        primaries, sources = rng.random(n) < 0.6, rng.random(n) < 0.5
        for k in range(n):
            sid += 1
            kinds = {b: [DETECTED, FORCED, MISSING][rng.integers(0, 3)] for b in BANDS}
            rows.append(row(obj_id, sid, primary=int(primaries[k]), phot_src=int(sources[k]), **kinds))
            if rng.random() < 0.05:                                              # an occasional duplicate join
                rows.append(dict(rows[-1]))
    raw = pd.DataFrame(rows)
    out = ps1_stack.clean(raw.copy()).set_index('PanSTARR_ID')
    assert out.index.is_unique and set(out.index) == set(raw.objID)
    by_id = raw.drop_duplicates('uniquePspsSTid').set_index('uniquePspsSTid')
    for b in BANDS:
        src = (by_id[f'{b}infoFlag3'] & STACK_PHOT_SRC) != 0
        detected = (by_id[f'{b}infoFlag2'] & 4 == 0) & (by_id[f'{b}PSFFluxErr'] > 0)
        has_src = src.groupby(by_id.objID).any().reindex(out.index)
        chosen = by_id.loc[out[f'{b}_row']].set_index(out.index)
        assert ((chosen[f'{b}infoFlag3'] & STACK_PHOT_SRC) != 0)[has_src].all()      # PS1's own choice wins
        no_src = ~has_src
        has_detection = detected.groupby(by_id.objID).any().reindex(out.index)
        assert (out[f'{b}_detected'][no_src] == has_detection[no_src]).all()        # else a detection is never passed over


def test_primary_comes_from_infoflag3_not_the_primarydetection_column():
    out = clean({**row(1, 101, primary=0, ra=10.001), 'primaryDetection': 1},      # flagged primary only by the bad column
                row(1, 102, primary=1, ra=10.0))
    assert out.loc[1, 'ra'] == 10.0 and out.loc[1, 'n_primary_rows'] == 1


def _write_partition(root, norder, npix, rows):
    path = os.path.join(root, f'Norder={norder}', f'Dir={npix // 10000 * 10000}', f'Npix={npix}.parquet')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False), path)


def test_objects_reads_the_covering_partitions_and_keeps_an_object_split_across_them_once(tmp_path):
    import hpgeom
    pix = lambda ra, dec: int(hpgeom.angle_to_pixel(2 ** 4, ra, dec, nest=True))
    a, b = pix(10.0, 20.0), pix(200.0, -20.0)
    _write_partition(tmp_path, 4, a, [row(1, 101, ra=10.0), row(2, 201, ra=10.01)])
    _write_partition(tmp_path, 4, b, [row(2, 202, ra=200.0, dec=-20.0), row(3, 301, ra=200.0, dec=-20.0)])
    one = ps1_stack.objects(np.array([10.0]), np.array([20.0]), str(tmp_path))
    assert sorted(one.PanSTARR_ID) == [1, 2]
    both = ps1_stack.objects(np.array([10.0, 200.0]), np.array([20.0, -20.0]), str(tmp_path))
    assert sorted(both.PanSTARR_ID) == [1, 2, 3]
    with pytest.raises(FileNotFoundError):
        ps1_stack.objects(np.array([100.0]), np.array([0.0]), str(tmp_path))


def test_each_band_records_the_skycell_its_row_came_from():
    out = clean({**row(1, 101, primary=1, phot_src=0, g=FORCED), 'skyCellID': 46}, {**row(1, 102, primary=0), 'skyCellID': 56})
    assert out.loc[1, 'g_skycell'] == 56 and out.loc[1, 'g_row'] == 102         # g from the detected secondary
    assert out.loc[1, 'r_skycell'] == 46 and out.loc[1, 'r_row'] == 101         # r from the primary
