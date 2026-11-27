"""Postage stamps rendered from FITS, the way `Sources.plot_cutout` draws them.

The browser used to fetch ready-made JPEGs (PanSTARRS `fitscut.cgi`, ZTF via
the CDS HiPS). They are cheap but hard to read: the stretch is fixed by the
service and a faint source sits invisibly in the noise. This module fetches
the pixels instead and applies the same scaling the analysis notebooks use --
`simple_norm(img, 'log', percent=99.5)` with viridis -- so a stamp on the web
page looks like a stamp in the paper.

The reason that is affordable at all is server-side cutouts. `Sources` gets
its ZTF pixels by downloading the whole 41 MB reference quadrant and cropping
it; IRSA's IBE will crop server-side for 25 kB, which is 1600x less and, at
60", the identical pixels. PanSTARRS already had a cutout service; asking it
for `format=fits` instead of `format=jpg` costs 245 kB rather than 13 kB.

Measured per band, cold:

    ZTF   IBE cutout  25 kB   0.42 s  ->  7.9 kB PNG   (61 x 61 native)
    PS1   fitscut     245 kB  0.86 s  ->  44 kB PNG    (240 x 240)

which lands within ~0.3 s of the JPEGs it replaces, and is *smaller* on the
wire for ZTF. Rendered PNGs are cached on disk, so that cost is paid once.

Both surveys need work we cannot do in the browser -- IRSA needs credentials
and PanSTARRS FITS needs the asinh flux rescaling -- so this runs server-side
and the API hands out its own URLs.
"""
from __future__ import annotations

import fcntl
import hashlib
import io
import os
import threading
from base64 import b64encode
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from source_db.cutouts import fetch
from source_db.paths import DEFAULT_OUT_DIR

# 60" across, matching the 240 px PanSTARRS stamp at 0.25"/px that the old
# JPEG path used, so swapping renderers does not change the field of view.
CUTOUT_ARCSEC = 60.0
PS1_PIXEL_ARCSEC = 0.25
PS1_SIZE_PX = int(round(CUTOUT_ARCSEC / PS1_PIXEL_ARCSEC))

# `simple_norm` defaults lifted from `Postage_Stamp.plot_cutout`.
DEFAULT_STRETCH = 'log'
DEFAULT_PERCENT = 99.5
STRETCHES = ('linear', 'sqrt', 'power', 'log', 'asinh')

# Pixels per arcsecond, for the scale-bar overlay. PanSTARRS stacks are
# 0.25"/px by definition; ZTF *deep reference* images are resampled to
# exactly 1.0"/px, checked across six field/ccd/quadrant combinations, so a
# 60" request comes back 61 x 61 covering 61.0". Sizing the bar as a fixed
# fraction of the frame would therefore read 10.17" on ZTF; the frontend
# multiplies this by the PNG's own pixel count instead.
PIXEL_ARCSEC = {'ztf': 1.0, 'panstarrs': PS1_PIXEL_ARCSEC}

PS1_FITSCUT = 'https://ps1images.stsci.edu/cgi-bin/fitscut.cgi'
IBE_DEEP = 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/deep'
ZTF_FILTERCODE = {'g': 'zg', 'r': 'zr', 'i': 'zi'}

CACHE_DIR = Path(os.environ.get('LT_CUTOUT_CACHE', DEFAULT_OUT_DIR / 'cutout_cache'))
# Per socket operation, so it also bounds a stalled transfer; IBE's slowest
# answer (the metadata search) is ~35 s.
TIMEOUT = 60

# The cache bounds itself rather than waiting for a cron nobody set up. A
# running byte total sits beside it and a sweep runs when that total crosses
# the budget; measuring the true size costs a full directory walk, so it
# happens only when the cheap counter says it is worth doing. 0 disables.
CACHE_MAX_BYTES = int(float(os.environ.get('LT_CUTOUT_MAX_GB', '20')) * 1e9)
_LEDGER = 'bytes'


# ── fetching ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _irsa_auth() -> Optional[str]:
    """Basic-auth header for IRSA, or None if no credentials are installed.

    The deep reference images are not public, which is the reason ZTF cannot
    be an `<img src>` straight to IRSA and has to come through us.
    """
    path = Path.home() / 'vault' / 'irsa_login.txt'
    if path.exists():
        parts = path.read_text().split()
    else:
        # On a host there is no vault; the same login ztforce uses, from .env.
        parts = [os.environ.get('ZTFORCE_IRSA_USER', ''),
                 os.environ.get('ZTFORCE_IRSA_PASS', '')]
    if len(parts) < 2 or not all(parts[:2]):
        return None
    return 'Basic ' + b64encode(f'{parts[0]}:{parts[1]}'.encode()).decode()


def _get(url: str, headers: Optional[dict] = None) -> Optional[bytes]:
    """Fetch with retries (`fetch.get`), returning None for a 400/404.

    A missing band is ordinary: ZTF has no i-band reference for most fields,
    and the caller renders "not available" for it.
    """
    return fetch.get(url, headers, timeout=TIMEOUT)


def _get_fits(url: str, headers: Optional[dict] = None) -> Optional[bytes]:
    """`_get`, also retrying a 200 whose body is not a complete FITS file."""
    return fetch.get(url, headers, timeout=TIMEOUT, validate=fetch.is_fits)


def ztf_cutout_url(field: int, ccd: int, qid: int, band: str,
                   ra: float, dec: float, arcsec: float = CUTOUT_ARCSEC) -> str:
    """IBE URL for a server-side crop of the ZTF deep reference image."""
    pf = str(int(field)).zfill(6)
    prefix = pf[:3]  # first three digits (000/, 001/), not the leading zeros
    fc = ZTF_FILTERCODE[band]
    pc = str(int(ccd)).zfill(2)
    return (f'{IBE_DEEP}/{prefix}/field{pf}/{fc}/ccd{pc}/q{int(qid)}/'
            f'ztf_{pf}_{fc}_c{pc}_q{int(qid)}_refimg.fits'
            # `gzip=false` matters: the gzipped variant is smaller on the wire
            # but astropy cannot open it from memory without a decompression
            # step that costs more than the bytes saved.
            f'?center={ra},{dec}&size={arcsec}arcsec&gzip=false')


def fetch_ztf(field: int, ccd: int, qid: int, band: str,
              ra: float, dec: float,
              arcsec: float = CUTOUT_ARCSEC) -> Optional[np.ndarray]:
    """ZTF reference pixels around a position, background-subtracted.

    `ZTF_Postage_Stamp` subtracts `sep`'s 2D background fitted over the whole
    quadrant. Over 60" that surface is flat to well under the noise, so a
    median of the stamp stands in for it; the log stretch is set by
    percentiles either way, which makes the two visually indistinguishable.
    """
    auth = _irsa_auth()
    if auth is None:
        return None
    raw = _get_fits(ztf_cutout_url(field, ccd, qid, band, ra, dec, arcsec),
                    {'Authorization': auth})
    if not raw:
        return None
    img = _primary_image(raw)
    if not _has_signal(img):
        return None
    img = img - np.nanmedian(img)
    # `get_images` flips both axes to match PanSTARRS, then draws with
    # `origin='lower'`. A PNG's first row is its top, so writing it out costs
    # another vertical flip -- the two cancel and only the horizontal one is
    # left. `_to_png` does not flip again for this reason.
    return np.fliplr(img)


def ps1_cutout_url(ra: float, dec: float, filename: str,
                   size_px: int = PS1_SIZE_PX) -> str:
    """`fitscut.cgi` URL for the FITS (not JPEG) form of a stack cutout."""
    return (f'{PS1_FITSCUT}?ra={ra}&dec={dec}&size={size_px}'
            f'&format=fits&red={filename}')


def fetch_ps1(ra: float, dec: float, filename: str,
              size_px: int = PS1_SIZE_PX) -> Optional[np.ndarray]:
    """PanSTARRS stack pixels, converted back to linear flux.

    Stack FITS are stored asinh-compressed, so the header's BSOFTEN/BOFFSET
    have to be undone or the stretch is applied to the wrong quantity. This
    is the same arithmetic as `get_pstarr_cutout`.
    """
    raw = _get_fits(ps1_cutout_url(ra, dec, filename, size_px))
    if not raw:
        return None
    from astropy.io import fits
    hdu = fits.open(io.BytesIO(raw))[0]
    if hdu.data is None:
        return None
    flux = hdu.data.astype(np.float32)
    if not _has_signal(flux):
        return None
    # Drawn with `origin='lower'` and no flips, so only the PNG row order.
    return np.flipud(ps1_linear_flux(flux, hdu.header))


def ps1_linear_flux(flux: np.ndarray, h) -> np.ndarray:
    """Undo the asinh compression PanSTARRS stores some images with."""
    if 'BSOFTEN' in h and 'BOFFSET' in h:
        v = h['BZERO'] + h['BSCALE'] * flux
        a = 2.5 / np.log(10)
        x = v / a
        flux = h['BOFFSET'] + h['BSOFTEN'] * (np.exp(x) - np.exp(-x))
    return flux


def _has_signal(img: Optional[np.ndarray]) -> bool:
    """Is there anything to draw?

    A cutout that falls off the edge of a reference image comes back the
    right shape but entirely NaN. Taking a median of that warns and yields
    NaN, and the stretch then renders a uniform block that looks like real
    but empty sky -- so it is treated as a missing band instead.
    """
    return (img is not None and img.size > 0
            and bool(np.isfinite(img).any()))


def _primary_image(raw: bytes) -> Optional[np.ndarray]:
    from astropy.io import fits
    hdu = fits.open(io.BytesIO(raw))[0]
    return None if hdu.data is None else hdu.data.astype(np.float32)


# ── rendering ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _viridis_lut() -> np.ndarray:
    """viridis as 256 RGB triples."""
    from matplotlib import colormaps
    lut = (np.asarray(colormaps['viridis'](np.linspace(0, 1, 256)))[:, :3] * 255)
    return lut.astype(np.uint8)


BAD_RGB = (255, 255, 255)


def bin_image(img: np.ndarray, k: int) -> np.ndarray:
    """`img` summed over k x k blocks (NaN where a block has no data). At 0.25"/px a 6-sigma PanSTARRS source puts
    about one sigma into its brightest pixel, so it is invisible at any stretch; binning 4 x 4 to ZTF-like 1" pixels
    adds its pixels the way the catalog's PSF fit does. Trailing rows and columns that do not fill a block are dropped."""
    if k <= 1:
        return img
    ny, nx = (img.shape[0] // k) * k, (img.shape[1] // k) * k
    blocks = np.asarray(img[:ny, :nx], dtype=np.float64).reshape(ny // k, k, nx // k, k)
    with np.errstate(invalid='ignore'):
        out = np.nansum(blocks, axis=(1, 3))
    out[np.isnan(blocks).all(axis=(1, 3))] = np.nan
    return out


def render_png(img: np.ndarray, stretch: str = DEFAULT_STRETCH,
               percent: float = DEFAULT_PERCENT, bin: int = 1) -> bytes:
    """Scale to bytes with `simple_norm` and encode as a paletted PNG; `bin` > 1 sums the image over bin x bin
    blocks first (`bin_image`)."""
    img = bin_image(img, bin)
    from astropy.visualization import simple_norm
    from PIL import Image

    data = np.asarray(img, dtype=np.float64)
    # `simple_norm` gets the array as it is, NaNs included: astropy drops
    # non-finite values when it takes the percentile, so substituting zeros
    # first would drag the limits toward zero and change the stretch on any
    # cutout with blank edges.
    norm = simple_norm(data, stretch, percent=percent)
    scaled = np.nan_to_num(np.asarray(norm(np.nan_to_num(data)), dtype=np.float64))
    # Index the palette the way matplotlib's `Colormap.__call__` does --
    # multiply by the entry count and truncate, with 1.0 pulled back into the
    # last entry. Scaling by 255 instead is off by one over most of the range
    # and shifts roughly a third of the pixels by one colour.
    idx = np.clip((scaled * 256).astype(np.int64), 0, 255).astype(np.uint8)

    # Pixels with no data are painted white rather than mapped to the bottom
    # of the colour map, where blank sky off the edge of a survey is
    # indistinguishable from real faint sky. This is also what matplotlib
    # does: its default "bad" colour is transparent, which on the white
    # figure `plot_cutout` draws into comes out white.
    bad = ~np.isfinite(data)
    if bad.any():
        # RGB only when it is actually needed. A colour-mapped image holds
        # 256 colours, so the paletted form is a third the size (44 kB
        # rather than 143 kB for a 240 px PanSTARRS stamp) -- but a palette
        # cannot carry a 257th entry for "no data".
        rgb = _viridis_lut()[idx]
        rgb[bad] = BAD_RGB
        im = Image.fromarray(rgb, mode='RGB')
    else:
        im = Image.fromarray(idx, mode='P')
        im.putpalette(_viridis_lut().tobytes())
    buf = io.BytesIO()
    im.save(buf, 'PNG', optimize=True)
    return buf.getvalue()


# ── disk cache ────────────────────────────────────────────────────────────

# Bump whenever `render_png` produces different bytes for the same input.
#
# The key includes this, so a bump orphans every existing entry rather than
# serving it: old files become unreachable and the next eviction reclaims
# them. Without it a renderer change silently keeps serving the old pixels,
# which has already happened twice -- once for the palette quantisation and
# once for white no-data pixels -- and was caught both times only by
# remembering to purge by hand.
#
# `test_render_version_is_bumped_when_output_changes` fails if the output
# moves and this does not, so forgetting is a test failure rather than a
# silent regression.
RENDER_VERSION = 2


def source_fingerprint(ra: float, dec: float, origin: object) -> str:
    """The inputs that decide the pixels, beyond the source's identity.

    A rebuild can move a source or repoint it at a different stack file.
    Folding those into the key invalidates exactly the sources that changed,
    where a global release id would throw away the whole cache on every
    build.
    """
    return f'{ra:.6f},{dec:.6f},{origin}'


def cache_path(source_uuid: str, survey: str, band: str, stretch: str,
               percent: float, fingerprint: str = '', bin: int = 1) -> Path:
    raw = (f'v{RENDER_VERSION}|{source_uuid}|{survey}|{band}'
           f'|{stretch}|{percent}|{fingerprint}' + (f'|bin{bin}' if bin > 1 else ''))     # unbinned keys unchanged: the precomputed cache stays valid
    key = hashlib.sha1(raw.encode()).hexdigest()
    # Default-stretch renders are the precomputed hot path -- roughly twelve
    # hours to rebuild at full scale -- so they live in their own subtree and
    # eviction clears everything else first. Two-character fan-out below
    # that, because a flat directory of a million PNGs is slow to stat.
    tier = DEFAULT_TIER if is_default(stretch, percent) and bin <= 1 else ALT_TIER
    return CACHE_DIR / tier / key[:2] / f'{key}.png'


DEFAULT_TIER = 'default'
ALT_TIER = 'alt'


def is_default(stretch: str, percent: float) -> bool:
    return stretch == DEFAULT_STRETCH and float(percent) == DEFAULT_PERCENT


def cached(path: Path) -> Optional[bytes]:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


class _Flight:
    def __init__(self, priority: Optional[fetch.Priority]):
        self.done = threading.Event()
        self.priority = priority
        self.png: Optional[bytes] = None
        self.error: Optional[BaseException] = None


_flights: dict[Path, _Flight] = {}
_flights_lock = threading.Lock()


def cached_or_render(path: Path, produce: Callable[[], Optional[bytes]]) -> Optional[bytes]:
    """The PNG at `path`, or `produce()`d and stored on a miss; None if nothing to draw.

    Single-flight: concurrent misses for one stamp (a prefetch and the view
    itself, say) share the first one's render and its outcome, error
    included, so the upstream service is asked once. A foreground caller
    joining a background render promotes it.
    """
    png = cached(path)
    if png is not None:
        return png
    mine = fetch.current_priority()
    with _flights_lock:
        flight = _flights.get(path)
        leader = flight is None
        if leader:
            flight = _flights[path] = _Flight(mine)
    if not leader:
        if flight.priority is not None and not (mine is not None and mine.background):
            flight.priority.background = False
        flight.done.wait()
        if flight.error is not None:
            raise flight.error
        return flight.png
    try:
        # A flight that finished between the miss above and taking the lead.
        flight.png = cached(path)
        if flight.png is None:
            flight.png = produce()
            if flight.png is not None:
                store(path, flight.png)
        return flight.png
    except BaseException as e:
        flight.error = e
        raise
    finally:
        with _flights_lock:
            del _flights[path]
        flight.done.set()


def store(path: Path, png: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename, so a reader never sees a half-written PNG and two
    # workers rendering the same stamp cannot corrupt each other. Per thread,
    # not just per process: the API renders on a thread pool.
    tmp = path.with_suffix(f'.{os.getpid()}-{threading.get_ident()}.tmp')
    tmp.write_bytes(png)
    tmp.replace(path)
    _account(len(png))


def _account(added: int) -> None:
    """Add to the running total, and sweep if it has crossed the budget.

    Approximate by design. The counter drifts when files are removed by
    hand, and every sweep re-bases it on a real measurement. Locked because
    the renderers are a thread pool, and a lost update there would delay a
    sweep rather than corrupt anything.
    """
    if CACHE_MAX_BYTES <= 0:
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        seed = None if (CACHE_DIR / _LEDGER).exists() else cache_size()[1]
        with open(CACHE_DIR / _LEDGER, 'a+') as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.seek(0)
            raw = fh.read().strip()
            try:
                total = int(raw) + added
            except ValueError:
                # No ledger yet, or a corrupt one: start from what is
                # actually on disk. Counting only writes made after the
                # ledger appeared would leave an existing cache invisible
                # to the budget and no sweep would ever fire. The
                # measurement is taken after the write, so it already
                # includes it -- adding `added` here would double-count.
                total = seed if seed is not None else cache_size()[1]
            fh.seek(0)
            fh.truncate()
            fh.write(str(total))
            over = total > CACHE_MAX_BYTES
    except OSError:
        return
    if over:
        evict(CACHE_MAX_BYTES)


def cache_size() -> tuple[int, int]:
    """(files, bytes) currently held."""
    n = total = 0
    for p in CACHE_DIR.rglob('*.png'):
        n += 1
        total += p.stat().st_size
    return n, total


def evict(max_bytes: int) -> dict:
    """Delete the coldest entries until the cache fits in `max_bytes`.

    Size-bounded and least-recently-used rather than time-expiring: a stamp
    does not go stale with age. The pixels are a deterministic function of a
    static data release, so an old entry is exactly as correct as a fresh
    one and expiring it only re-pays the fetch. What needs bounding is the
    disk, and what makes an entry worthless is that nobody looks at it.

    Non-default stretches go first regardless of age -- they are a few
    hundred milliseconds to recreate on demand, where the default renders
    back the whole precompute.

    Recency is the later of atime and mtime. Under `relatime` that is only
    refreshed about daily, which is ample for deciding what to drop.
    """
    entries = []
    for tier in (ALT_TIER, DEFAULT_TIER):
        for p in (CACHE_DIR / tier).rglob('*.png'):
            st = p.stat()
            entries.append((tier == DEFAULT_TIER, max(st.st_atime, st.st_mtime),
                            st.st_size, p))
    total = sum(e[2] for e in entries)
    before = total
    removed = 0
    if total > max_bytes:
        # Protected tier last, coldest first within each.
        entries.sort(key=lambda e: (e[0], e[1]))
        for _, _, size, p in entries:
            if total <= max_bytes:
                break
            try:
                p.unlink()
            except FileNotFoundError:
                continue
            total -= size
            removed += 1
    # A sweep is the one moment the true size is known, so the running
    # counter is re-based here rather than left to drift.
    try:
        (CACHE_DIR / _LEDGER).write_text(str(total))
    except OSError:
        pass
    return {'removed': removed, 'freed': before - total,
            'bytes_before': before, 'bytes_after': total}
