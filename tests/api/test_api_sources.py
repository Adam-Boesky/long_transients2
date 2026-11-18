"""The read endpoints: candidate list, detail, light curves, navigation."""
from __future__ import annotations

import json

import pytest

from tests.conftest import LC_UUIDS, NO_LC_UUID, UUIDS


# ── health ────────────────────────────────────────────────────────────────────

def test_health_reports_ok(client):
    r = client.get('/api/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'


# ── list ──────────────────────────────────────────────────────────────────────

def test_list_returns_every_source(client):
    r = client.get('/api/sources')
    assert r.status_code == 200
    body = r.json()
    assert body['total'] == len(UUIDS)
    assert {i['source_uuid'] for i in body['items']} == set(UUIDS)


def test_list_paginates(client):
    first = client.get('/api/sources?limit=2&offset=0').json()
    second = client.get('/api/sources?limit=2&offset=2').json()
    assert len(first['items']) == 2 and len(second['items']) == 2
    # Pages must not overlap, or the click-through would show duplicates.
    assert not ({i['source_uuid'] for i in first['items']}
                & {i['source_uuid'] for i in second['items']})


def test_list_rejects_oversized_page(client):
    assert client.get('/api/sources?limit=100000').status_code == 422


def test_list_sorts_by_epochs_descending(client):
    items = client.get('/api/sources?sort=epochs&desc=true').json()['items']
    counts = [i['n_epochs'] for i in items]
    assert counts == sorted(counts, reverse=True)


def test_list_sort_key_is_whitelisted(client):
    """An unknown sort key must be rejected, not interpolated into SQL."""
    assert client.get('/api/sources?sort=source_uuid;DROP TABLE sources').status_code == 422


def test_list_filters_by_survey(client):
    items = client.get('/api/sources?survey=ztf').json()['items']
    assert items, 'expected at least one ztf source'
    assert all('ztf' in i['surveys'] for i in items)


def test_survey_filter_matches_whole_names_only(client):
    """`surveys` is a comma-joined string; a substring must not match."""
    assert client.get('/api/sources?survey=zt').json()['total'] == 0


def test_list_filters_by_min_epochs(client):
    items = client.get('/api/sources?min_epochs=5').json()['items']
    assert all(i['n_epochs'] >= 5 for i in items)


def test_has_lightcurve_excludes_empty_sources(client):
    ids = {i['source_uuid'] for i in
           client.get('/api/sources?has_lightcurve=true').json()['items']}
    assert NO_LC_UUID not in ids
    assert set(LC_UUIDS) <= ids


def test_unlabelled_only_requires_author(client):
    assert client.get('/api/sources?unlabelled_only=true').status_code == 400


def test_unlabelled_only_hides_labelled_sources(client):
    target = LC_UUIDS[0]
    client.put(f'/api/sources/{target}/annotations/adam', json={'label': 'SN'})
    ids = {i['source_uuid'] for i in client.get(
        '/api/sources?unlabelled_only=true&author=adam').json()['items']}
    assert target not in ids
    # ...but only for that author.
    ids_other = {i['source_uuid'] for i in client.get(
        '/api/sources?unlabelled_only=true&author=someone_else').json()['items']}
    assert target in ids_other


def test_list_surfaces_this_authors_label(client):
    target = LC_UUIDS[0]
    client.put(f'/api/sources/{target}/annotations/adam', json={'label': 'AGN'})
    row = next(i for i in client.get('/api/sources?author=adam').json()['items']
               if i['source_uuid'] == target)
    assert row['label'] == 'AGN'
    assert row['n_annotations'] == 1


# ── detail ────────────────────────────────────────────────────────────────────

def test_detail_returns_catalog_columns(client):
    body = client.get(f'/api/sources/{UUIDS[0]}').json()
    assert body['source_uuid'] == UUIDS[0]
    # Promoted columns are fields; everything else lands in `catalog`.
    assert body['ra'] == 10.0
    assert body['catalog']['some_float'] == 1.5
    assert 'ra' not in body['catalog']


def test_detail_404s_for_unknown_source(client):
    assert client.get('/api/sources/not-a-uuid').status_code == 404


# ── light curves ──────────────────────────────────────────────────────────────

def test_lightcurve_is_served_gzipped(client):
    r = client.get(f'/api/sources/{UUIDS[0]}/lightcurve',
                   headers={'accept-encoding': 'gzip'})
    assert r.status_code == 200
    doc = r.json()
    assert doc['source_uuid'] == UUIDS[0]
    assert set(doc['surveys']) == {'ztf', 'panstarrs'}


def test_lightcurve_is_decompressed_when_client_cannot_accept_gzip(client):
    r = client.get(f'/api/sources/{UUIDS[0]}/lightcurve',
                   headers={'accept-encoding': 'identity'})
    assert r.status_code == 200
    assert r.headers.get('content-encoding') != 'gzip'
    # Body must be plain JSON the client can read directly.
    assert json.loads(r.content)['source_uuid'] == UUIDS[0]


def test_lightcurve_payload_is_columnar_and_ordered(client):
    doc = client.get(f'/api/sources/{UUIDS[0]}/lightcurve').json()
    g = doc['surveys']['ztf']['g']
    assert set(g) >= {'mjd', 'mag', 'magerr'}
    assert len(g['mjd']) == len(g['mag']) == len(g['magerr'])
    assert g['mjd'] == sorted(g['mjd']), 'epochs must be time-ordered for plotting'


def test_lightcurve_carries_reduction_inputs(client):
    """catflags must survive to the client: the cut is applied at render time."""
    doc = client.get(f'/api/sources/{UUIDS[0]}/lightcurve').json()
    assert 'catflags' in doc['surveys']['ztf']['g']
    assert 6141 in doc['surveys']['ztf']['g']['catflags'], (
        'the flagged epoch must be stored, not filtered out at build time')


def test_lightcurve_of_source_without_epochs_is_empty_not_404(client):
    r = client.get(f'/api/sources/{NO_LC_UUID}/lightcurve')
    assert r.status_code == 200
    assert r.json() == {'source_uuid': NO_LC_UUID, 'n_epochs': 0, 'surveys': {}}


def test_lightcurve_404s_for_unknown_source(client):
    assert client.get('/api/sources/not-a-uuid/lightcurve').status_code == 404


def test_reported_epoch_count_matches_payload(client):
    for uuid in LC_UUIDS:
        doc = client.get(f'/api/sources/{uuid}/lightcurve').json()
        counted = sum(len(s['mjd']) for bands in doc['surveys'].values()
                      for s in bands.values())
        assert doc['n_epochs'] == counted
        summary = client.get(f'/api/sources/{uuid}').json()
        assert summary['n_epochs'] == counted


# ── sub-catalogue filter ──────────────────────────────────────────────────

@pytest.fixture
def tiered_client(tmp_path):
    """A science DB with a known spread of `catalog_tier` values.

    The shared fixture's synthetic sources carry no tier at all, so the
    filter would be exercised only against zero rows there.

    Both tiers are built from one list of rows: the list endpoint filters and
    pages in the catalog engine (over parquet) and then reads the surviving
    rows out of SQLite, so the two have to describe the same sources or the
    page comes back empty for reasons that have nothing to do with the test.
    """
    import sqlite3
    import orjson
    import pandas as pd
    from fastapi.testclient import TestClient
    from api import catalog_filters as api_filters
    from api import db as api_db
    from api.main import app

    con = sqlite3.connect(':memory:', check_same_thread=False)
    con.row_factory = sqlite3.Row
    # `catalog_tier` is both a real column and a key in the blob, as the build
    # writes it: the list filters on the column, the detail page reads the row.
    con.execute('CREATE TABLE sources (source_uuid TEXT PRIMARY KEY, coord_slug TEXT,'
                ' ra REAL, dec REAL, n_epochs INT, surveys TEXT, n_spectra INT,'
                ' spec_surveys TEXT, catalog_tier TEXT, catalog_json TEXT)')
    # Two in_both, two in_ztf, one in_ztf_wide, and one with no tier at all.
    tiers = ['0', '0', '1', '1', '1_wide_association', None]
    for i, tier in enumerate(tiers):
        con.execute('INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?,?)',
                    (f'{i:08d}-0000-4000-8000-000000000000', f'slug{i}',
                     10.0 + i, -5.0 + i, 3, 'ztf', 0, '', tier,
                     orjson.dumps({} if tier is None
                                  else {'catalog_tier': tier}).decode()))
    parquet = tmp_path / 'sources.parquet'
    pd.DataFrame({
        'source_uuid': [f'{i:08d}-0000-4000-8000-000000000000'
                        for i in range(len(tiers))],
        'coord_slug': [f'slug{i}' for i in range(len(tiers))],
        'ra': [10.0 + i for i in range(len(tiers))],
        'dec': [-5.0 + i for i in range(len(tiers))],
        'catalog_tier': tiers,
    }).to_parquet(parquet, index=False)
    engine = api_filters.startup(con, parquet)

    app.dependency_overrides[api_db.get_science] = lambda: con
    app.dependency_overrides[api_filters.get_filters] = lambda: engine
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    api_filters.shutdown()
    con.close()


def test_catalog_filter_rejects_an_unknown_name(client):
    assert client.get('/api/sources',
                      params={'catalog': 'in_hubble'}).status_code == 422


def test_catalog_filter_splits_the_tiers(tiered_client):
    """The named groups partition the tiers: no source is in two of them."""
    totals = {name: tiered_client.get('/api/sources',
                                      params={'catalog': name}).json()['total']
              for name in ('in_both', 'in_ztf', 'in_ztf_wide')}
    assert totals == {'in_both': 2, 'in_ztf': 2, 'in_ztf_wide': 1}


def test_the_named_groups_do_not_overlap(tiered_client):
    """What the segmented control that offers them promises."""
    seen = []
    for name in ('in_both', 'in_ztf', 'in_ztf_wide'):
        page = tiered_client.get('/api/sources', params={'catalog': name}).json()
        seen += [i['source_uuid'] for i in page['items']]
    assert len(seen) == len(set(seen))


def test_each_row_names_its_sub_catalogue(tiered_client):
    """The classify header shows this, in the filter's own vocabulary."""
    items = tiered_client.get('/api/sources').json()['items']
    got = [i['catalog'] for i in sorted(items, key=lambda i: i['source_uuid'])]
    assert got == ['in_both', 'in_both', 'in_ztf', 'in_ztf', 'in_ztf_wide', None]


def test_catalog_filter_omitted_returns_every_tier(tiered_client):
    assert tiered_client.get('/api/sources').json()['total'] == 6


def test_source_without_a_tier_appears_only_under_everything(tiered_client):
    """Not reachable from either sub-catalogue, by construction.

    Every row the real build writes is tagged, so this is a guard against a
    malformed blob silently landing in one of the named buckets.
    """
    seen = set()
    for cat in ('in_both', 'in_ztf', 'in_ztf_wide'):
        page = tiered_client.get('/api/sources', params={'catalog': cat}).json()
        seen |= {i['source_uuid'] for i in page['items']}
    everything = {i['source_uuid'] for i
                  in tiered_client.get('/api/sources').json()['items']}
    assert len(everything - seen) == 1


def test_catalog_filter_composes_with_other_filters(tiered_client):
    """The clause is ANDed, not substituted for the rest of the WHERE."""
    page = tiered_client.get('/api/sources',
                             params={'catalog': 'in_ztf', 'min_epochs': 99}).json()
    assert page['total'] == 0


def test_wise_colors_are_filterable_when_built(built_db, science_con, tmp_path):
    """`wise_colors.parquet`, found next to the catalog, adds two filter columns."""
    import shutil

    import pandas as pd

    from api import catalog_filters as api_filters

    parquet = tmp_path / 'sources.parquet'
    shutil.copy(built_db.parent / 'sources.parquet', parquet)
    pd.DataFrame({'source_uuid': UUIDS[:2], 'wise_w1w2_mean': [0.9, 0.3],
                  'wise_w1w2_median': [0.85, 0.2]}).to_parquet(
        tmp_path / 'wise_colors.parquet', index=False)
    engine = api_filters.startup(science_con, parquet)
    try:
        assert engine.count({'column': 'wise_w1w2_mean', 'op': '>', 'value': 0.8}) == 1
        assert engine.count({'column': 'wise_w1w2_median', 'op': 'is_null'}) == len(UUIDS) - 2
    finally:
        api_filters.shutdown()


# ── shared verdict filters ────────────────────────────────────────────────────

def test_sources_honours_bucket_include_exclude(client):
    client.put(f'/api/sources/{UUIDS[0]}/annotations/beth', json={'label': 'SN'})
    inc = client.get('/api/sources?limit=50&bucket_in=SN').json()
    assert [i['source_uuid'] for i in inc['items']] == [UUIDS[0]]
    exc = client.get('/api/sources?limit=50&bucket_out=SN').json()
    assert UUIDS[0] not in [i['source_uuid'] for i in exc['items']]


def test_sources_honours_status_include(client):
    client.put(f'/api/sources/{UUIDS[1]}/followup',
               json={'status': 'queued', 'author': 'adam'})
    body = client.get('/api/sources?limit=50&status_in=queued').json()
    assert [i['source_uuid'] for i in body['items']] == [UUIDS[1]]


def test_none_replaces_only_unlabelled(client):
    """`bucket_in=none` is what the removed checkbox used to ask, but
    any-author rather than per-author."""
    client.put(f'/api/sources/{UUIDS[0]}/annotations/beth', json={'label': 'SN'})
    total = client.get('/api/sources?limit=1').json()['total']
    body = client.get('/api/sources?limit=50&bucket_in=none').json()
    assert body['total'] == total - 1
    assert UUIDS[0] not in [i['source_uuid'] for i in body['items']]


def test_none_status_is_everything_when_nobody_has_set_one(client):
    total = client.get('/api/sources?limit=1').json()['total']
    assert client.get('/api/sources?limit=1&status_in=none').json()['total'] == total


def test_excluding_none_keeps_only_the_ones_with_a_value(client):
    client.put(f'/api/sources/{UUIDS[1]}/followup',
               json={'status': 'observed', 'author': 'adam'})
    body = client.get('/api/sources?limit=50&status_out=none').json()
    assert [i['source_uuid'] for i in body['items']] == [UUIDS[1]]


def test_locate_uses_the_same_cut_as_the_window(client):
    """Otherwise the offset a link resolves to indexes a different list."""
    client.put(f'/api/sources/{UUIDS[0]}/annotations/beth', json={'label': 'SN'})
    r = client.get('/api/sources-locate', params={
        'target': UUIDS[0], 'bucket_out': ['SN']}).json()
    assert r['offset'] is None, 'a filtered-out source has no position'
    r = client.get('/api/sources-locate', params={
        'target': UUIDS[0], 'bucket_in': ['SN']}).json()
    assert r['offset'] == 0


def test_table_and_sources_agree_under_the_same_filter(client):
    """The two pages must narrow the same sample."""
    client.put(f'/api/sources/{UUIDS[0]}/annotations/beth', json={'label': 'SN'})
    for q in ('bucket_in=SN', 'bucket_out=SN', 'bucket_in=none'):
        a = client.get(f'/api/table?limit=1&{q}').json()['total']
        b = client.get(f'/api/sources?limit=1&{q}').json()['total']
        assert a == b, q


# ── next-unlabelled ───────────────────────────────────────────────────────────
#
# The endpoint the classify page actually navigates with. It had no tests; the
# properties below were covered only against `/queue/next`, which this
# superseded and which has been removed.

def test_next_unlabelled_skips_what_the_author_filed(client):
    first = client.get('/api/sources-next-unlabelled', params={'author': 'adam'}).json()
    uuid = client.get('/api/sources', params={'limit': 1, 'offset': first['offset']}
                      ).json()['items'][0]['source_uuid']
    client.put(f'/api/sources/{uuid}/annotations/adam', json={'label': 'SN'})
    after = client.get('/api/sources-next-unlabelled', params={'author': 'adam'}).json()
    nxt = client.get('/api/sources', params={'limit': 1, 'offset': after['offset']}
                     ).json()['items'][0]['source_uuid']
    assert nxt != uuid


def test_one_authors_label_does_not_advance_another(client):
    """Each member works their own queue; a verdict is not a global 'done'."""
    start = client.get('/api/sources-next-unlabelled', params={'author': 'adam'}).json()
    uuid = client.get('/api/sources', params={'limit': 1, 'offset': start['offset']}
                      ).json()['items'][0]['source_uuid']
    client.put(f'/api/sources/{uuid}/annotations/adam', json={'label': 'SN'})
    beth = client.get('/api/sources-next-unlabelled', params={'author': 'beth'}).json()
    assert beth['offset'] == start['offset']


def test_next_unlabelled_requires_an_author(client):
    assert client.get('/api/sources-next-unlabelled').status_code == 422


def test_group_lists_every_detection_of_the_object(client):
    members = client.get(f'/api/sources/{UUIDS[0]}/group').json()['members']
    assert [m['source_uuid'] for m in members] == sorted([UUIDS[0], UUIDS[2]])
    # Asked from the other member, the same group comes back.
    assert client.get(f'/api/sources/{UUIDS[2]}/group').json()['members'] == members


def test_an_ungrouped_source_has_no_members(client):
    assert client.get(f'/api/sources/{UUIDS[1]}/group').json() == {'members': []}


def test_one_per_object_lists_each_group_once(client):
    """UUIDS[0] and UUIDS[2] are one object; the lower uuid stands for it."""
    page = client.get('/api/sources', params={'one_per_object': True, 'limit': 50}).json()
    shown = {s['source_uuid'] for s in page['items']}
    assert page['total'] == len(UUIDS) - 1
    assert UUIDS[0] in shown and UUIDS[2] not in shown
    loc = client.get('/api/sources-locate', params={'target': UUIDS[2], 'one_per_object': True})
    assert loc.json()['offset'] is None


def test_a_filtered_out_detection_hands_over_to_its_group_mate(client):
    """The pick happens after the filter, so the object does not vanish."""
    doc = {'column': 'source_uuid', 'op': 'not_in', 'value': [UUIDS[0]]}
    page = client.get('/api/sources', params={'one_per_object': True, 'limit': 50,
                                              'filter': json.dumps(doc)}).json()
    assert UUIDS[2] in {s['source_uuid'] for s in page['items']}



# ── the query regions drawn on the stamps ─────────────────────────────────────

def test_cutouts_carry_the_query_regions(client):
    body = client.get(f'/api/sources/{UUIDS[0]}/cutouts').json()
    by_key = {r['key']: r for r in body['regions']}
    assert [r['key'] for r in body['regions']] == ['lightcurve', 'catalog', 'gaia',
                                                   'wise_beam', 'erosita']
    assert by_key['lightcurve']['radius_arcsec'] == 1.5
    assert all(r['label'] and r['help'] for r in body['regions'])


def test_every_region_matches_its_module():
    """`QUERY_REGIONS` copies numbers the pipeline modules own; the stamps draw
    the copies, so each must still equal its source."""
    import inspect
    from source_db.constants import MATCH_RADIUS_ARCSEC, QUERY_REGIONS
    from source_db.lightcurves import irsa, panstarrs, sdss, ztf
    from Source_Analysis import enrich_combined_tabs, filter_field
    from Source_Analysis.catalogs import erosita, wise

    for mod in (irsa, panstarrs, sdss, ztf, enrich_combined_tabs):
        assert mod.MATCH_RADIUS_ARCSEC == MATCH_RADIUS_ARCSEC, mod.__name__
    default = lambda fn, name: inspect.signature(fn).parameters[name].default
    radius = {r['key']: r['radius_arcsec'] for r in QUERY_REGIONS}
    assert radius['lightcurve'] == radius['catalog'] == MATCH_RADIUS_ARCSEC
    assert radius['gaia'] == default(filter_field.annotate_gaia, 'max_arcsec') == filter_field.GAIA_MATCH_ARCSEC
    assert radius['wise_beam'] == wise.BEAM_ARCSEC
    assert radius['erosita'] == erosita.MATCH_RADIUS_ARCSEC
