"""kron.measure_all (the compiled loop) against the per-source reference functions."""
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from Extracting import kron
from Extracting.images import Image


@pytest.fixture(scope='module')
def frame():
    """A 400 x 400 image of noise with 60 Gaussian sources of assorted widths and fluxes, a masked block and a bright
    neighbour pair, as a KronFrame with its floor set."""
    rng = np.random.default_rng(3)
    ny = nx = 400
    yy, xx = np.mgrid[0:ny, 0:nx]
    data = rng.normal(100.0, 2.0, (ny, nx)).astype(np.float32)
    n = 60
    x, y = rng.uniform(30, nx - 30, n), rng.uniform(30, ny - 30, n)
    x[1], y[1] = x[0] + 4.0, y[0] + 1.0                                     # a close pair: segments must separate them
    sigma = rng.choice([1.0, 1.4, 2.5, 4.0], n)
    flux = 10 ** rng.uniform(2.0, 4.5, n)
    for k in range(n):
        data += (flux[k] / (2 * np.pi * sigma[k] ** 2) * np.exp(-((xx - x[k]) ** 2 + (yy - y[k]) ** 2) / (2 * sigma[k] ** 2))).astype(np.float32)
    data[200:215, 100:140] = np.nan                                          # a masked block some apertures reach into
    w = WCS(naxis=2); w.wcs.ctype = ['RA---TAN', 'DEC--TAN']; w.wcs.crval = [10.0, 20.0]; w.wcs.crpix = [nx / 2, ny / 2]; w.wcs.cdelt = [-1.0 / 3600, 1.0 / 3600]
    im = Image(band='r', name='synthetic', data=data, wcs=w, zero_point=26.0, fwhm_px=2.4, pixel_scale=1.0, gain=6.0, mag_limit=None, header=fits.Header())
    ra, dec = w.pixel_to_world_values(x, y)
    f = kron.prepare(im, ra, dec)
    kron.moment_floor(f, np.full(n, 100.0))
    return f


def test_the_compiled_loop_reproduces_the_reference_functions(frame):
    idx = np.arange(len(frame.x))
    fast = kron.measure_all(frame, idx)
    slow = [kron.measure(frame, k) for k in idx]
    assert list(fast['flag']) == [r['flag'] for r in slow]
    for col in ('sky_radius', 'window', 'r1', 'r_ap', 'flux', 'fluxerr', 'masked_frac', 'peak_snr'):
        a, b = fast[col].values, np.array([r[col] for r in slow])
        assert np.array_equal(np.isnan(a), np.isnan(b)), col
        ok = np.isfinite(b)
        assert np.allclose(a[ok], b[ok], rtol=1e-6, atol=1e-6), col
    assert fast['flag'].ne(0).any() and np.isfinite(fast['flux']).sum() > 50     # the fixture exercises both the flagged and the plain paths


def test_kron_photometry_runs_a_subset_and_reports_the_rest_unmeasured(frame):
    which = np.zeros(len(frame.x), bool); which[:5] = True
    out = kron.kron_photometry(frame, which)
    assert len(out) == len(frame.x) and np.isfinite(out['mag'].values[:5]).any() and out['flag'].values[5:].tolist() == [-1] * (len(frame.x) - 5)
    assert np.isnan(out['mag'].values[5:]).all()
