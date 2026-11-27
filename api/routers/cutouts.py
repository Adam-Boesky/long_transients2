"""Postage stamps: which ones exist, and the rendered PNGs."""
from __future__ import annotations

import sqlite3
import urllib.error
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from api.db import get_science
from source_db.constants import QUERY_REGIONS
from source_db.cutouts import epochs, fetch, render

router = APIRouter()


# Postage stamps are rendered from FITS rather than fetched as ready-made
# JPEGs. The JPEGs were cheaper but the stretch is fixed by the service, which
# buries a faint source in the noise; `source_db.cutouts.render` applies the
# same `simple_norm(img, 'log', percent=99.5)` + viridis the analysis code
# uses. See that module for the measured cost -- it lands within ~0.3 s of
# the JPEGs and is smaller on the wire for ZTF.
#
# It has to happen here rather than in the browser: the ZTF reference images
# need IRSA credentials, and PanSTARRS stack FITS need an asinh flux rescale.

SURVEYS = ('ztf', 'panstarrs')
BANDS = ('g', 'r', 'i')


def _origins(sci: sqlite3.Connection, source_uuid: str) -> dict[tuple[str, str], str]:
    """(survey, band) -> origin, for the stamps this source actually has.

    A row exists only where the stamp can be made, so the absence of a key
    is the availability answer -- there is no flag to consult.
    """
    return {(r['survey'], r['band']): r['origin'] for r in sci.execute(
        'SELECT survey, band, origin FROM cutout_sources WHERE source_uuid=?',
        (source_uuid,))}


def _parse_origin(survey: str, origin: str):
    """The ZTF pointing back as integers; PanSTARRS passes through."""
    if survey != 'ztf':
        return origin
    field, ccd, qid = origin.split('/')
    return int(field), int(ccd), int(qid)


@router.get('/sources/{source_uuid}/cutouts')
def get_cutouts(source_uuid: str,
                stretch: str = Query(render.DEFAULT_STRETCH),
                percent: float = Query(render.DEFAULT_PERCENT, gt=0, le=100),
                bin: int = Query(1, ge=1, le=8),
                sci: sqlite3.Connection = Depends(get_science)):
    """Image URLs for the source, by survey and band.

    URLs rather than bytes, so the six stamps download in parallel off the
    browser's own connection pool and land in its cache. Each one points at
    the renderer below; `stretch` and `percent` ride along so a change to
    them is a different URL and cannot be served from a stale cache.
    """
    row = sci.execute('SELECT ra, dec FROM sources WHERE source_uuid=?',
                      (source_uuid,)).fetchone()
    if row is None:
        raise HTTPException(404, f'no source {source_uuid}')
    if stretch not in render.STRETCHES:
        raise HTTPException(422, f'stretch must be one of {render.STRETCHES}')

    images: dict[str, dict[str, str]] = {}
    if row['ra'] is None or row['dec'] is None:
        return {'source_uuid': source_uuid, 'images': images, 'regions': QUERY_REGIONS}

    qs = f'?stretch={quote(stretch)}&percent={percent}' + (f'&bin={bin}' if bin > 1 else '')
    have = _origins(sci, source_uuid)
    for survey in SURVEYS:
        bands = sorted(b for s, b in have if s == survey)
        if bands:
            images[survey] = {
                b: f'/api/sources/{source_uuid}/cutout/{survey}/{b}.png{qs}'
                for b in bands}

    # The page draws the 10" scale bar itself, and the two surveys do not
    # cover the same angle: PanSTARRS is 240 px at 0.25" (60.0") but ZTF is
    # 61 px at 1.0" (61.0"). Sending the pixel scale lets it size the bar
    # from each PNG's own width rather than assuming a common field of view.
    # The radii everything about the source was gathered within, so the page
    # can draw each one on the stamps.
    return {'source_uuid': source_uuid, 'images': images,
            # Binning applies to PanSTARRS only (ZTF's pixels are already 1"), so its scale grows with it.
            'pixel_arcsec': {s: render.PIXEL_ARCSEC[s] * (bin if s == 'panstarrs' else 1) for s in images},
            'regions': QUERY_REGIONS}


@router.get('/sources/{source_uuid}/cutout/{survey}/{band}.png')
def get_cutout_png(source_uuid: str, survey: str, band: str, request: Request,
                   stretch: str = Query(render.DEFAULT_STRETCH),
                   percent: float = Query(render.DEFAULT_PERCENT, gt=0, le=100),
                   bin: int = Query(1, ge=1, le=8),
                   sci: sqlite3.Connection = Depends(get_science)):
    """One rendered postage stamp.

    A plain `def`, so FastAPI runs it in the threadpool: the body is entirely
    blocking network IO and the six stamps for a source are requested at
    once. Rendered PNGs are cached on disk, so the upstream fetch is paid
    once per (source, band, stretch).

    A prefetch says so in a header, not the URL, so it warms the browser
    cache under the address the page will ask for; it then runs as
    background work upstream.
    """
    if survey not in SURVEYS:
        raise HTTPException(404, f'unknown survey {survey}')
    if band not in BANDS:
        raise HTTPException(404, f'unknown band {band}')
    if stretch not in render.STRETCHES:
        raise HTTPException(422, f'stretch must be one of {render.STRETCHES}')

    # The fingerprint folds in the position and the stack file, so a rebuild
    # that moves a source misses the cache instead of serving its old sky.
    ident = _cutout_inputs(source_uuid, survey, band, sci)
    if ident is None:
        raise HTTPException(404, f'no {survey} {band} image for {source_uuid}')
    ra, dec, origin = ident
    k = bin if survey == 'panstarrs' else 1                  # ZTF pixels are 1" already
    path = render.cache_path(source_uuid, survey, band, stretch, percent,
                             render.source_fingerprint(ra, dec, origin), bin=k)
    try:
        with fetch.priority(background=_is_prefetch(request)):
            png = render.cached_or_render(
                path, lambda: _render_cutout(survey, band, ra, dec, origin, stretch, percent, k))
    except urllib.error.URLError as e:
        raise HTTPException(502, f'{survey} image service did not answer: {e}')
    if png is None:
        raise HTTPException(404, f'no {survey} {band} image for {source_uuid}')

    return Response(content=png, media_type='image/png',
                    # The pixels for a given source and stretch never change,
                    # so the browser should never ask twice.
                    headers={'Cache-Control': 'public, max-age=31536000, immutable'})


def _is_prefetch(request: Request) -> bool:
    """Our own `X-Prefetch: 1`, or the browser's `Sec-Purpose: prefetch`."""
    return (request.headers.get('x-prefetch') == '1'
            or 'prefetch' in request.headers.get('sec-purpose', ''))


def _cutout_inputs(source_uuid: str, survey: str, band: str,
                   sci: sqlite3.Connection):
    """(ra, dec, origin) for one stamp, or None if it cannot be made."""
    row = sci.execute('SELECT ra, dec FROM sources WHERE source_uuid=?',
                      (source_uuid,)).fetchone()
    if row is None:
        raise HTTPException(404, f'no source {source_uuid}')
    if row['ra'] is None or row['dec'] is None:
        return None
    origin = _origins(sci, source_uuid).get((survey, band))
    if origin is None:
        return None
    return row['ra'], row['dec'], _parse_origin(survey, origin)


def _position(sci: sqlite3.Connection, source_uuid: str) -> tuple[float, float]:
    row = sci.execute('SELECT ra, dec FROM sources WHERE source_uuid=?',
                      (source_uuid,)).fetchone()
    if row is None:
        raise HTTPException(404, f'no source {source_uuid}')
    if row['ra'] is None or row['dec'] is None:
        raise HTTPException(404, f'{source_uuid} has no position')
    return row['ra'], row['dec']


def _ztf_pointing(sci: sqlite3.Connection, source_uuid: str, band: str,
                  objectid: str | None):
    """The (field, ccd, qid) an epoch was taken on, as best we know it.

    The epoch's own objectid is exact. Without it (a serving DB built before
    the payload carried it), the source's reference pointing in that band is
    right for every epoch from its main field.
    """
    parsed = epochs.parse_objectid(objectid) if objectid else None
    if parsed is not None:
        field, _fid, ccd, qid = parsed
        return field, ccd, qid
    have = _origins(sci, source_uuid)
    origin = have.get(('ztf', band)) or next(
        (o for (s, _b), o in have.items() if s == 'ztf'), None)
    return _parse_origin('ztf', origin) if origin else None


def _exposure(survey: str, band: str, ra: float, dec: float, mjd: float,
              pointing=None):
    """The exposure behind one light-curve point; 404 if no survey image matches."""
    if survey not in epochs.EPOCH_SURVEYS:
        raise HTTPException(404, f'no single-epoch images for {survey}')
    if band not in epochs.EPOCH_BANDS[survey]:
        raise HTTPException(404, f'unknown {survey} band {band}')
    try:
        exp = epochs.find_exposure(survey, band, ra, dec, mjd, pointing)
    except (urllib.error.URLError, TimeoutError) as e:
        raise HTTPException(502, f'{survey} image service did not answer: {e}')
    if exp is None:
        raise HTTPException(404, f'no {survey} {band} exposure found at MJD {mjd}')
    return exp


@router.get('/sources/{source_uuid}/epoch/{survey}/{band}')
def get_epoch(source_uuid: str, survey: str, band: str,
              mjd: float = Query(..., description='the light-curve point\'s MJD'),
              objectid: str | None = Query(None, description='ZTF DR objectid of the epoch'),
              stretch: str = Query(render.DEFAULT_STRETCH),
              percent: float = Query(render.DEFAULT_PERCENT, gt=0, le=100),
              sci: sqlite3.Connection = Depends(get_science)):
    """Which exposure a light-curve point came from, and where its image is.

    Separate from the PNG so the page can show what was matched (and the
    links out) while the pixels are still being fetched.
    """
    if stretch not in render.STRETCHES:
        raise HTTPException(422, f'stretch must be one of {render.STRETCHES}')
    ra, dec = _position(sci, source_uuid)
    pointing = _ztf_pointing(sci, source_uuid, band, objectid) if survey == 'ztf' else None
    exp = _exposure(survey, band, ra, dec, mjd, pointing)
    qs = f'?mjd={mjd!r}&stretch={quote(stretch)}&percent={percent}'
    if objectid:
        qs += f'&objectid={quote(objectid)}'
    return {
        'survey': survey, 'band': band, 'mjd': mjd,
        'image_id': exp.image_id, 'obs_mjd': exp.obs_mjd, 'details': exp.details,
        'png': f'/api/sources/{source_uuid}/epoch/{survey}/{band}/image.png{qs}',
        'pixel_arcsec': render.PIXEL_ARCSEC[survey],
        'fits_url': exp.cutout_url, 'full_url': exp.full_url,
    }


@router.get('/sources/{source_uuid}/epoch/{survey}/{band}/image.png')
def get_epoch_png(source_uuid: str, survey: str, band: str,
                  mjd: float = Query(...),
                  objectid: str | None = Query(None),
                  stretch: str = Query(render.DEFAULT_STRETCH),
                  percent: float = Query(render.DEFAULT_PERCENT, gt=0, le=100),
                  sci: sqlite3.Connection = Depends(get_science)):
    """One exposure's cutout, on the same grid as the reference stamp."""
    if stretch not in render.STRETCHES:
        raise HTTPException(422, f'stretch must be one of {render.STRETCHES}')
    ra, dec = _position(sci, source_uuid)
    path = render.cache_path(source_uuid, f'{survey}-epoch', band, stretch, percent,
                             f'{ra:.6f},{dec:.6f},{mjd:.6f}')

    def produce() -> bytes:
        pointing = _ztf_pointing(sci, source_uuid, band, objectid) if survey == 'ztf' else None
        exp = _exposure(survey, band, ra, dec, mjd, pointing)
        origin = _origins(sci, source_uuid).get((survey, band)) if survey == 'ztf' else None
        try:
            img = epochs.fetch_epoch(exp, ra, dec,
                                     _parse_origin(survey, origin) if origin else None)
        except (urllib.error.URLError, TimeoutError) as e:
            raise HTTPException(502, f'{survey} image service did not answer: {e}')
        if img is None:
            raise HTTPException(404, f'{exp.image_id} has no pixels at this position')
        return render.render_png(img, stretch, percent)

    png = render.cached_or_render(path, produce)
    return Response(content=png, media_type='image/png',
                    headers={'Cache-Control': 'public, max-age=31536000, immutable'})


def _render_cutout(survey: str, band: str, ra: float, dec: float, origin,
                   stretch: str, percent: float, bin: int = 1) -> bytes | None:
    img = (render.fetch_ztf(*origin, band, ra, dec) if survey == 'ztf'
           else render.fetch_ps1(ra, dec, origin))
    if img is None or img.size == 0:
        return None
    return render.render_png(img, stretch, percent, bin)
