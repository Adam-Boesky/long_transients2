"""Cutout rendering and the endpoints that serve it.

Nothing here touches the network. The fetchers are the only part that does,
and they are monkeypatched: what is worth testing is the arithmetic around
them -- the IBE path construction, the cache key, which bands get offered,
and the status codes -- not that IRSA is up.
"""
from __future__ import annotations

import io
import os
import sqlite3
import time

import numpy as np
import orjson
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import db as api_db
from api.main import app
from source_db.cutouts import render

UUID = 'aaaa1111-0000-4000-8000-000000000001'
BARE = 'bbbb2222-0000-4000-8000-000000000002'   # no pointing, no PanSTARRS


# ── rendering ─────────────────────────────────────────────────────────────

def test_render_png_is_a_paletted_png_of_the_input_shape():
    img = np.arange(64, dtype=np.float32).reshape(8, 8)
    im = Image.open(io.BytesIO(render.render_png(img)))
    assert im.format == 'PNG'
    assert im.size == (8, 8)
    # Paletted, not RGB: a colour-mapped image has only 256 distinct colours,
    # and encoding it as RGB triples the bytes for no extra information.
    assert im.mode == 'P'


def test_render_png_reflects_stretch_and_percentile():
    rng = np.random.default_rng(0)
    img = rng.lognormal(0.0, 1.0, (32, 32)).astype(np.float32)
    log = render.render_png(img, 'log', 99.5)
    assert render.render_png(img, 'asinh', 99.5) != log
    assert render.render_png(img, 'log', 99.9) != log


@pytest.mark.parametrize('img', [
    None,
    np.zeros((0, 0), dtype=np.float32),
    np.full((4, 4), np.nan, dtype=np.float32),
])
def test_has_signal_rejects_nothing_to_draw(img):
    """An off-the-edge cutout is the right shape but entirely NaN."""
    assert render._has_signal(img) is False


def test_has_signal_accepts_a_partially_finite_image():
    img = np.full((4, 4), np.nan, dtype=np.float32)
    img[0, 0] = 1.0
    assert render._has_signal(img) is True


# ── URLs ──────────────────────────────────────────────────────────────────

# The IBE directory prefix is the first three characters of the field padded
# to six, so it is always three characters -- it is not "the leading zeros".
@pytest.mark.parametrize('field,expected', [
    (501, '/deep/000/field000501/'),
    (690, '/deep/000/field000690/'),
    (1234, '/deep/001/field001234/'),
    (10234, '/deep/010/field010234/'),
])
def test_ztf_url_pads_field_and_prefix(field, expected):
    url = render.ztf_cutout_url(field, 6, 2, 'g', 10.0, -5.0)
    assert expected in url


def test_ztf_url_asks_for_an_uncompressed_cutout():
    url = render.ztf_cutout_url(501, 15, 3, 'r', 10.0, -5.0)
    assert 'ztf_000501_zr_c15_q3_refimg.fits' in url
    assert 'center=10.0,-5.0' in url
    assert f'size={render.CUTOUT_ARCSEC}arcsec' in url
    assert 'gzip=false' in url


def test_ps1_url_requests_fits_not_jpeg():
    url = render.ps1_cutout_url(10.0, -5.0, '/rings/stk.g.fits')
    assert 'format=fits' in url
    assert 'red=/rings/stk.g.fits' in url


# ── cache ─────────────────────────────────────────────────────────────────

def test_cache_path_separates_every_input():
    base = render.cache_path(UUID, 'ztf', 'g', 'log', 99.5)
    for other in [render.cache_path(UUID, 'ztf', 'g', 'asinh', 99.5),
                  render.cache_path(UUID, 'ztf', 'g', 'log', 99.9),
                  render.cache_path(UUID, 'panstarrs', 'g', 'log', 99.5),
                  render.cache_path(UUID, 'ztf', 'r', 'log', 99.5),
                  render.cache_path(BARE, 'ztf', 'g', 'log', 99.5)]:
        assert other != base


def test_cache_path_changes_when_the_source_moves():
    """A rebuild that repoints a source must miss, not serve its old sky."""
    here = render.source_fingerprint(10.0, -5.0, (501, 15, 3))
    moved = render.source_fingerprint(10.001, -5.0, (501, 15, 3))
    repointed = render.source_fingerprint(10.0, -5.0, (502, 15, 3))
    assert render.cache_path(UUID, 'ztf', 'g', 'log', 99.5, here) \
        != render.cache_path(UUID, 'ztf', 'g', 'log', 99.5, moved)
    assert render.cache_path(UUID, 'ztf', 'g', 'log', 99.5, here) \
        != render.cache_path(UUID, 'ztf', 'g', 'log', 99.5, repointed)


def test_default_and_alternate_stretches_live_in_separate_tiers():
    """Eviction clears the alternates first, so they must be separable."""
    d = render.cache_path(UUID, 'ztf', 'g', render.DEFAULT_STRETCH,
                          render.DEFAULT_PERCENT)
    a = render.cache_path(UUID, 'ztf', 'g', 'asinh', 99.9)
    assert d.parent.parent.name == render.DEFAULT_TIER
    assert a.parent.parent.name == render.ALT_TIER


def test_cache_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(render, 'CACHE_DIR', tmp_path)
    path = render.cache_path(UUID, 'ztf', 'g', 'log', 99.5)
    assert render.cached(path) is None
    render.store(path, b'\x89PNG-not-really')
    assert render.cached(path) == b'\x89PNG-not-really'
    # No stray temporary files left behind by the write-then-rename.
    assert [q.name for q in tmp_path.rglob('*.tmp')] == []


# ── the endpoints ─────────────────────────────────────────────────────────

@pytest.fixture
def cutout_client(tmp_path, monkeypatch):
    """A science DB holding just what the cutout endpoints read."""
    monkeypatch.setattr(render, 'CACHE_DIR', tmp_path / 'cache')
    # `check_same_thread=False` as `api_db.connect_science` does: FastAPI
    # runs these sync endpoints in a threadpool, so the request never
    # touches the connection from the thread that opened it.
    con = sqlite3.connect(':memory:', check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute('CREATE TABLE sources (source_uuid TEXT PRIMARY KEY, ra REAL,'
                ' dec REAL, catalog_json TEXT)')
    con.execute('CREATE TABLE cutout_sources (source_uuid TEXT, survey TEXT,'
                ' band TEXT, origin TEXT, PRIMARY KEY (source_uuid, survey, band))')
    con.execute('INSERT INTO sources VALUES (?,?,?,?)', (UUID, 10.0, -5.0, '{}'))
    con.execute('INSERT INTO sources VALUES (?,?,?,?)', (BARE, 20.0, 5.0, '{}'))
    # Both surveys look the same here now: a row means the stamp exists.
    # This field has no i-band reference, so ZTF has only g and r.
    con.executemany(
        'INSERT INTO cutout_sources VALUES (?,?,?,?)',
        [(UUID, 'ztf', 'g', '501/15/3'),
         (UUID, 'ztf', 'r', '501/15/3'),
         (UUID, 'panstarrs', 'g', '/stk.g.fits'),
         (UUID, 'panstarrs', 'r', '/stk.r.fits')])
    app.dependency_overrides[api_db.get_science] = lambda: con
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    con.close()


def test_cutouts_lists_only_bands_that_exist(cutout_client):
    body = cutout_client.get(f'/api/sources/{UUID}/cutouts').json()
    assert sorted(body['images']['ztf']) == ['g', 'r']
    assert sorted(body['images']['panstarrs']) == ['g', 'r']


def test_cutouts_returns_our_own_urls_carrying_the_stretch(cutout_client):
    body = cutout_client.get(f'/api/sources/{UUID}/cutouts',
                             params={'stretch': 'asinh', 'percent': 99.9}).json()
    url = body['images']['ztf']['g']
    assert url.startswith(f'/api/sources/{UUID}/cutout/ztf/g.png')
    # The stretch is part of the address, so changing it cannot be served
    # from a stale browser cache.
    assert 'stretch=asinh' in url and 'percent=99.9' in url


def test_cutouts_reports_the_pixel_scale_per_survey(cutout_client):
    """The page sizes the 10" bar with this; the surveys differ."""
    body = cutout_client.get(f'/api/sources/{UUID}/cutouts').json()
    assert body['pixel_arcsec'] == {'ztf': 1.0, 'panstarrs': 0.25}


def test_cutouts_is_empty_without_a_pointing_or_stack(cutout_client):
    body = cutout_client.get(f'/api/sources/{BARE}/cutouts').json()
    assert body['images'] == {}


def test_cutouts_404s_for_an_unknown_source(cutout_client):
    assert cutout_client.get('/api/sources/nope/cutouts').status_code == 404


def test_png_is_served_from_the_cache_without_fetching(cutout_client, monkeypatch):
    def explode(*a, **k):                      # noqa: ANN001
        raise AssertionError('should not have fetched')
    monkeypatch.setattr(render, 'fetch_ztf', explode)

    png = render.render_png(np.arange(16, dtype=np.float32).reshape(4, 4))
    fp = render.source_fingerprint(10.0, -5.0, (501, 15, 3))
    render.store(render.cache_path(UUID, 'ztf', 'g', 'log', 99.5, fp), png)

    r = cutout_client.get(f'/api/sources/{UUID}/cutout/ztf/g.png')
    assert r.status_code == 200
    assert r.headers['content-type'] == 'image/png'
    assert r.content == png
    assert 'immutable' in r.headers['cache-control']


def test_png_404s_when_the_band_has_no_image(cutout_client, monkeypatch):
    monkeypatch.setattr(render, 'fetch_ztf', lambda *a, **k: None)
    r = cutout_client.get(f'/api/sources/{UUID}/cutout/ztf/i.png')
    assert r.status_code == 404


def test_png_502s_when_the_service_gives_up(cutout_client, monkeypatch):
    from source_db.cutouts import fetch

    def down(*a, **k):                   # noqa: ANN001
        raise fetch.FetchError('HTTPError after 5 attempts')
    monkeypatch.setattr(render, 'fetch_ztf', down)
    r = cutout_client.get(f'/api/sources/{UUID}/cutout/ztf/g.png')
    assert r.status_code == 502


def test_png_renders_and_caches_on_a_miss(cutout_client, monkeypatch):
    calls = []

    def fake(*args, **kwargs):                 # noqa: ANN001
        calls.append(args)
        return np.arange(36, dtype=np.float32).reshape(6, 6)

    monkeypatch.setattr(render, 'fetch_ps1', fake)
    url = f'/api/sources/{UUID}/cutout/panstarrs/g.png'
    first = cutout_client.get(url)
    assert first.status_code == 200
    # Second request must come from disk, not a second fetch.
    second = cutout_client.get(url)
    assert second.content == first.content
    assert len(calls) == 1


@pytest.mark.parametrize('path,status', [
    (f'/api/sources/{UUID}/cutout/hubble/g.png', 404),
    (f'/api/sources/{UUID}/cutout/ztf/z.png', 404),
])
def test_png_rejects_unknown_survey_or_band(cutout_client, path, status):
    assert cutout_client.get(path).status_code == status


def test_png_rejects_an_unsupported_stretch(cutout_client):
    r = cutout_client.get(f'/api/sources/{UUID}/cutout/ztf/g.png',
                          params={'stretch': 'bogus'})
    assert r.status_code == 422


def test_render_matches_plot_cutout_pixel_for_pixel():
    """The default stretch is the analysis code's, not an approximation.

    `Sources.plot_cutout` draws `simple_norm(img, 'log', percent=99.5)` with
    viridis, so a stamp on the web page has to come out the same array
    matplotlib would rasterise. This pins both halves of that: the limits
    astropy derives, and the quantisation into the 256-entry palette --
    scaling by 255 rather than 256 shifts about a third of the pixels by one
    colour, which is invisible per pixel and wrong everywhere.
    """
    from astropy.visualization import simple_norm
    from matplotlib import colormaps

    rng = np.random.default_rng(0)
    img = rng.lognormal(3.0, 1.5, (48, 48)).astype(np.float32)

    ours = np.array(Image.open(io.BytesIO(render.render_png(img))).convert('RGB'))
    norm = simple_norm(img, 'log', percent=99.5)
    theirs = (np.asarray(colormaps['viridis'](norm(img)))[:, :, :3] * 255).astype(np.uint8)

    assert np.array_equal(ours, theirs)


def test_blank_edges_do_not_shift_the_stretch():
    """NaN reaches `simple_norm` unsubstituted.

    astropy drops non-finite values before taking the percentile. Replacing
    them with zeros first would pull the limits toward zero and quietly
    restretch every cutout that runs off the edge of a survey.
    """
    from astropy.visualization import simple_norm
    from matplotlib import colormaps

    rng = np.random.default_rng(1)
    img = rng.lognormal(3.0, 1.5, (48, 48)).astype(np.float32)
    img[:12, :] = np.nan

    ours = np.array(Image.open(io.BytesIO(render.render_png(img))).convert('RGB'))
    norm = simple_norm(img, 'log', percent=99.5)
    theirs = (np.asarray(colormaps['viridis'](norm(img)))[:, :, :3] * 255).astype(np.uint8)

    real = ~np.isnan(img)
    assert np.array_equal(ours[real], theirs[real])


def test_pixels_with_no_data_are_white():
    """Blank sky off the edge of a survey must not read as faint sky.

    Mapping NaN through the colour map puts it at the bottom of viridis,
    which is exactly what a real but very faint pixel looks like.
    """
    rng = np.random.default_rng(2)
    img = rng.lognormal(3.0, 1.5, (32, 32)).astype(np.float32)
    img[:8, :] = np.nan

    px = np.array(Image.open(io.BytesIO(render.render_png(img))).convert('RGB'))
    assert (px[:8] == 255).all()
    assert (px[8:] != 255).any()


def test_a_clean_cutout_stays_paletted():
    """RGB only where it buys something.

    A colour-mapped image holds 256 colours, so the paletted encoding is
    about a third the bytes; the RGB fallback exists solely because a
    palette has no room for a 257th "no data" entry.
    """
    rng = np.random.default_rng(3)
    img = rng.lognormal(3.0, 1.5, (64, 64)).astype(np.float32)

    clean = render.render_png(img)
    assert Image.open(io.BytesIO(clean)).mode == 'P'

    holed = img.copy()
    holed[0, 0] = np.nan
    assert Image.open(io.BytesIO(render.render_png(holed))).mode == 'RGB'
    assert len(clean) < len(render.render_png(holed))


# ── cache lifetime ────────────────────────────────────────────────────────

def test_render_version_is_bumped_when_output_changes():
    """A renderer change must come with a `RENDER_VERSION` bump.

    The cache key embeds the version, so forgetting to bump it leaves every
    existing entry reachable and the app keeps serving pixels the current
    code would not produce. That has already happened twice. If this fails,
    the fix is to bump `RENDER_VERSION` and paste the new digest in here --
    not to relax the assertion.
    """
    import hashlib

    rng = np.random.default_rng(20260923)
    img = rng.lognormal(3.0, 1.5, (24, 24)).astype(np.float32)
    img[0, :4] = np.nan                       # exercise the no-data path too
    digest = hashlib.sha256(render.render_png(img)).hexdigest()

    known = {
        2: '317f16b000487919401c04a2e3a67b61ea5e7319007eabe6665e7acddb913d1f',
    }
    assert render.RENDER_VERSION in known, (
        f'RENDER_VERSION {render.RENDER_VERSION} has no recorded output digest')
    assert digest == known[render.RENDER_VERSION], (
        'render_png output changed without a RENDER_VERSION bump')


def test_evict_is_a_noop_under_the_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(render, 'CACHE_DIR', tmp_path)
    render.store(render.cache_path(UUID, 'ztf', 'g', 'log', 99.5), b'x' * 100)
    assert render.evict(10_000)['removed'] == 0
    assert render.cache_size()[0] == 1


def test_evict_drops_alternate_stretches_before_default_ones(tmp_path, monkeypatch):
    """The default renders are the precomputed hot path.

    Recreating one alternate stretch is a few hundred milliseconds; the
    default tier is the whole precompute, hours of it at full scale.
    """
    monkeypatch.setattr(render, 'CACHE_DIR', tmp_path)
    default = render.cache_path(UUID, 'ztf', 'g', 'log', 99.5)
    alt = render.cache_path(UUID, 'ztf', 'g', 'asinh', 99.9)
    render.store(default, b'd' * 500)
    render.store(alt, b'a' * 500)

    # Make the *default* look colder, so age alone would evict it first.
    old = time.time() - 86_400
    os.utime(default, (old, old))

    render.evict(600)
    assert default.exists()
    assert not alt.exists()


def test_evict_takes_the_coldest_first_within_a_tier(tmp_path, monkeypatch):
    monkeypatch.setattr(render, 'CACHE_DIR', tmp_path)
    paths = []
    for i, band in enumerate(('g', 'r', 'i')):
        p = render.cache_path(UUID, 'ztf', band, 'asinh', 99.9)
        render.store(p, b'x' * 400)
        stamp = time.time() - (3 - i) * 3600      # g oldest, i newest
        os.utime(p, (stamp, stamp))
        paths.append(p)

    render.evict(900)                              # room for two
    assert not paths[0].exists()
    assert paths[1].exists() and paths[2].exists()


def test_evict_reports_what_it_freed(tmp_path, monkeypatch):
    monkeypatch.setattr(render, 'CACHE_DIR', tmp_path)
    for band in ('g', 'r', 'i'):
        render.store(render.cache_path(UUID, 'ztf', band, 'asinh', 99.9), b'x' * 1000)
    before = render.cache_size()[1]
    r = render.evict(1500)
    assert r['bytes_before'] == before
    assert r['bytes_after'] <= 1500
    assert r['freed'] == before - r['bytes_after']
    assert r['removed'] == 2


def test_ledger_starts_from_what_is_already_on_disk(tmp_path, monkeypatch):
    """A cache that predates the ledger must still count against the budget.

    Counting only writes made after the ledger appeared would leave an
    existing 60 GB invisible and no sweep would ever fire.
    """
    monkeypatch.setattr(render, 'CACHE_DIR', tmp_path)
    monkeypatch.setattr(render, 'CACHE_MAX_BYTES', 10_000_000)
    # Files written before any accounting exists.
    for band in ('g', 'r', 'i'):
        p = render.cache_path(UUID, 'ztf', band, 'log', 99.5)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'x' * 1000)
    assert not (tmp_path / render._LEDGER).exists()

    render.store(render.cache_path(BARE, 'ztf', 'g', 'log', 99.5), b'y' * 500)
    assert int((tmp_path / render._LEDGER).read_text()) == 3500


def test_store_sweeps_itself_when_it_crosses_the_budget(tmp_path, monkeypatch):
    """The cap holds without anything external running eviction."""
    monkeypatch.setattr(render, 'CACHE_DIR', tmp_path)
    monkeypatch.setattr(render, 'CACHE_MAX_BYTES', 5_000)
    for i in range(40):
        render.store(render.cache_path(f'u{i}', 'ztf', 'g', 'asinh', 99.9),
                     b'z' * 500)
    _, total = render.cache_size()
    assert total <= 5_000
    # And the ledger agrees with the disk after a sweep re-bases it.
    assert int((tmp_path / render._LEDGER).read_text()) == total


def test_a_zero_budget_disables_the_sweep(tmp_path, monkeypatch):
    monkeypatch.setattr(render, 'CACHE_DIR', tmp_path)
    monkeypatch.setattr(render, 'CACHE_MAX_BYTES', 0)
    for i in range(5):
        render.store(render.cache_path(f'u{i}', 'ztf', 'g', 'log', 99.5), b'q' * 100)
    assert render.cache_size()[0] == 5


def test_concurrent_requests_for_one_stamp_fetch_it_once(cutout_client, monkeypatch):
    # A prefetch and the real view asking for the same uncached stamp at once
    # must share one upstream fetch.
    import threading
    calls = []

    def slow(*a, **k):                   # noqa: ANN001
        calls.append(a)
        time.sleep(0.3)
        return np.arange(61 * 61, dtype=np.float32).reshape(61, 61)
    monkeypatch.setattr(render, 'fetch_ztf', slow)
    # The fixture's one in-memory connection is not safe across threads;
    # the DB lookup is not what this test is about.
    from api.routers import cutouts as router
    monkeypatch.setattr(router, '_cutout_inputs', lambda *a: (10.0, -5.0, (501, 15, 3)))
    codes = []
    url = f'/api/sources/{UUID}/cutout/ztf/g.png'

    def hit():
        try:
            codes.append(cutout_client.get(url).status_code)
        except BaseException as e:       # noqa: BLE001
            codes.append(repr(e))
    threads = [threading.Thread(target=hit) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert codes == [200] * 4
    assert len(calls) == 1


def test_waiters_share_the_leaders_error(tmp_path, monkeypatch):
    import threading
    monkeypatch.setattr(render, 'CACHE_DIR', tmp_path)
    path = tmp_path / 'x.png'
    calls, errors = [], []

    def failing():
        calls.append(1)
        time.sleep(0.2)
        raise RuntimeError('upstream gave up')

    def hit():
        try:
            render.cached_or_render(path, failing)
        except RuntimeError as e:
            errors.append(str(e))
    threads = [threading.Thread(target=hit) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert errors == ['upstream gave up'] * 3


def test_a_foreground_waiter_promotes_a_background_render(tmp_path, monkeypatch):
    import threading
    from source_db.cutouts import fetch
    monkeypatch.setattr(render, 'CACHE_DIR', tmp_path)
    path = tmp_path / 'y.png'
    started, release, seen = threading.Event(), threading.Event(), []

    def produce():
        started.set()
        release.wait(2)
        seen.append(fetch.current_priority().background)
        return b'png'

    def prefetch():
        with fetch.priority(background=True):
            render.cached_or_render(path, produce)
    t = threading.Thread(target=prefetch)
    t.start()
    started.wait(2)
    view = threading.Thread(target=lambda: render.cached_or_render(path, produce))
    view.start()
    time.sleep(0.1)
    release.set()
    t.join()
    view.join()
    assert seen == [False]                # rendered once, promoted to foreground


def test_prefetch_header_runs_the_render_as_background(cutout_client, monkeypatch):
    from source_db.cutouts import fetch
    seen = []

    def fake(*a, **k):                   # noqa: ANN001
        seen.append(fetch.current_priority().background)
        return np.arange(61 * 61, dtype=np.float32).reshape(61, 61)
    monkeypatch.setattr(render, 'fetch_ztf', fake)
    cutout_client.get(f'/api/sources/{UUID}/cutout/ztf/g.png', headers={'X-Prefetch': '1'})
    cutout_client.get(f'/api/sources/{UUID}/cutout/ztf/r.png')
    assert seen == [True, False]


def test_bin_image_sums_blocks_and_keeps_the_cache_key_of_unbinned_stamps():
    img = np.arange(16, dtype=float).reshape(4, 4); img[0, 0] = np.nan
    out = render.bin_image(img, 2)
    assert out.shape == (2, 2) and out[0, 0] == 1 + 4 + 5 and out[1, 1] == 10 + 11 + 14 + 15     # NaN ignored inside a block
    assert np.isnan(render.bin_image(np.full((2, 2), np.nan), 2)).all()                          # ... but a block with no data is NaN
    assert render.bin_image(img, 1) is img
    assert render.cache_path(UUID, 'panstarrs', 'g', 'log', 99.5) == render.cache_path(UUID, 'panstarrs', 'g', 'log', 99.5, bin=1)
    assert render.cache_path(UUID, 'panstarrs', 'g', 'log', 99.5, bin=4) != render.cache_path(UUID, 'panstarrs', 'g', 'log', 99.5)
    assert render.ALT_TIER in str(render.cache_path(UUID, 'panstarrs', 'g', 'log', 99.5, bin=4))
