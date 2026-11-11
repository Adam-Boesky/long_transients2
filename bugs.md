# Known bugs

## Swapped PSF-fit results in the ZTF extraction (threaded band photometry)

**Status:** open. Found 2026-11-11. Affects every quadrant extracted with `Tile(parallel=True)`
(the local-PS1 path, `USE_LOCAL_PS1 = True`, since 2026-09-01), and any earlier run that used it.

**Symptom.** `ZTF_{g,r,i}PSFMag`, `PSFMagErr`, `qfit` and `cfit` for a small fraction of sources
are the fit results of a *different* source, usually in a different band of the same quadrant.
The Kron and sep columns in the same row are correct. Swaps go both ways: a 21.5-mag source
gets a 16.8 ± 0.0006 result, a 15.5-mag star gets 22.9 ± 0.16. `PSFFlags` is 0 for all of them,
so `psf_fit_filter` cannot catch them. Example: 322p538286_27p776446, quadrant 000643_11_1,
whose stored r-band PSF value is exactly the i-band fit of the star at pixel (600, 651).

**Mechanism.** Three things combine:

1. `Tile.data_dicts` runs the g, r, i `Source_Extractor.get_data_table` calls concurrently in a
   `ThreadPoolExecutor` when `parallel=True`.
2. `photutils.psf.PSFPhotometry.__init__` has `fitter=TRFLSQFitter()` as a *default argument*,
   so every `PSFPhotometry` object in the process shares one fitter instance.
3. astropy's `_NLLSQFitter._run_fitter` stores the scipy result on the instance
   (`self.fit_info = optimize.least_squares(...)`) and then copies the solution into the model
   from `self.fit_info.x`. If another thread's fit finishes in that gap, the model receives the
   other thread's parameters. photutils then reads `self.fitter.fit_info` for the covariance and
   residuals, which is why error, qfit and cfit travel with the flux.

The window is microseconds per fit, so the rate is low and depends on node load: it rises when
the node is oversubscribed (8 worker processes × 3 threads).

**How widespread.** Measured on Cannon with the sep-flux test below (random 150 quadrants,
~7 M fits): ~60% of band tables at or below the 0.02% floor, ~30% at 0.02–0.1%, ~10% at
0.1–2%. Worst batches: extractions dated 2026-10-14 to 10-22 and 2026-04. Overall ~0.06% of
fits are provably bad, so ~0.1% swapped in total. In the candidate catalog (release 6,
249,212 sources) 1,319 have a provably bad band and ~2,600 are estimated affected (1%), because
a swap manufactures a large dmag and is preferentially selected. Follow-up targets, the queue
and the exciting-sources list contain none; the golden cut has ~34 of 5,172.

**Caveat on counting.** The only test that works on the stored tables is physical impossibility:
PSF flux more than 2.5× the whole sep flux of the same row (`m_sep - PSFMag > 1`). Swaps between
two sources of similar flux produce plausible values and cannot be detected without a rerun, so
every count above is a lower bound on swaps and the ×2 is an assumption of symmetry. Those
undetectable swaps also cannot produce a large dmag, so they do not create false candidates,
but they do perturb any per-source PSF magnitude by an unknown amount.

**Fix.** Give each `PSFPhotometry` its own fitter in `Source_Extractor.get_psf_mags`:

```python
from astropy.modeling.fitting import TRFLSQFitter
self.psfphot = PSFPhotometry(self.epsf, fit_shape=psf_fit_width, fitter=TRFLSQFitter())
```

Then rerun the PSF step for affected quadrants (EPSFs, nan masks and WCSs are saved per
quadrant, so only `get_psf_mags` and the downstream association and filtering need to run).
Add a post-fit guard: flag or raise when PSF flux exceeds 2.5× sep flux for more than ~0.02% of
sources in a quadrant.

**Related, separate failure.** Quadrant 000337_12_4 has a broken PSF model in all three bands
(g bright stars offset by a constant 1.1 mag; r and i scatter ±1.5 mag vs Kron; 83/30/14% of
fits impossibly bright; no swap partners). It produced 333 candidates vs a typical 8. None of
the 150 sampled quadrants show this, so it is rare. `_build_epsf` has no check that the model
reproduces the PS1 magnitudes of the stars it was built from; it should.

**Artifacts.** `Data/catalog_results/000643_11_1/` (stored r table, EPSF, comparison scripts,
`impossible_psf_sources.csv` with the 1,319 provably bad candidates),
`Data/catalog_results/_field_lists/psf_rate_sample.csv` (per-quadrant rates).

## Fake upper limits for PS1 sources inside bright galaxies (association, not detection)

**Status:** open. Found 2026-11-11. Affects the in_both and in_ztf selections in every release.

**Symptom.** A candidate with two ZTF detections and one "missing" band, where the missing band's
`{b}_dmag` is the ZTF limiting magnitude minus a bright (< 19) PS1 PSF mag, i.e. a +5 to +6 mag
"change" at the capped ±8σ. That band then supplies the second |σ| > 3 band for
`at_least_n_big_dmag_bands`. Examples: 158p862021_n14p129408 (i), 57p825971_n7p411841 (r),
163p119164_45p889120 (g), 268p228400_18p592894 (g), 168p668036_n3p727797 (g).

**Mechanism.** Pan-STARRS resolves a bright galaxy into several catalog entries (nucleus, knots,
the galaxy itself); the ZTF reference image gives one merged sep segment. Two ways the PS1 entry
ends up with no ZTF row:

1. *Deblending* (158p862021, i): the knot has too little contrast against the disk in that band and
   never becomes a child; the segment centroid is the nucleus, 6.6" away, outside the 1" radius.
   `deblend_nthresh = 32` leaves no level between saddle and knot peak; 64 separates it.
2. *One-to-one association* (the other four): sep does produce a segment whose centroid is only
   0.2–0.8" from the PS1 position, but `associate_tables` hands it to a different PS1 entry
   (or to none), and the PS1 row for our source is left `PSTARR`-only.

Either way the PS1-only row gets `Catalog_Flag = 2`, `annotate_dmag_sigma` substitutes
`ZTF_{b}_mag_limit`, and the result is a capped +8σ that counts toward the two-band requirement.
`at_least_n_big_dmag_bands` checks catalog membership (two bands with a ZTF+PS1 association) on
one set of bands but counts |σ| > 3 over all three, so a PS1-only band can supply the second.

**How widespread.** Release 6: 59,791 candidates have exactly two ZTF detections plus one
substituted band; in 52,139 of them the substituted band is above 3σ. Restricting to PS1 < 19 in
the missing band (where a limit is unphysical on its face): 1,537 candidates, 1,008 of which
depend on that band to pass.

**Test (100 random of the 1,537).** Pipeline-style PSF photometry at the PS1 position with the
quadrant's saved EPSF, converted to σ with the field's envelope:
- measured dmag median −0.63 mag (ZTF brighter), σ 10/50/90% = −3.6 / −2.6 / +1.7; none near +8σ;
- 69 of 100 depended on the substituted band; **44 of those fail the n ≥ 2 cut with real
  photometry** (64%); scaled, ~680 of the 1,537 would leave the catalog;
- the 25 that still pass do so because the measured band is itself at ≲ −3σ: ZTF's 2" PSF
  includes host light a 1" PS1 PSF does not, so every band of a source on a galaxy reads
  −0.5 to −1 mag. That host-contamination systematic is a separate issue and is what the other
  31 (already two real bands > 3σ) are passing on too.

**Fix.** (a) In `at_least_n_big_dmag_bands`, count |σ| > 3 only over bands that satisfy the catalog
membership being filtered, so a substituted band can never be the second. (b) Do not substitute a
limiting magnitude when the PS1 position falls inside a ZTF sep segment (the nan mask / segmap are
saved per quadrant); treat it as blended, not undetected. (c) Longer term, forced PSF photometry
at the PS1 position for every unmatched PS1 source brighter than the limit, instead of a limit.
The deblending side (raising `deblend_nthresh`) only helps the minority of cases like 158p862021.

**Artifacts.** `Data/catalog_results/_field_lists/sample100.json` (the sample),
`missing100_results.csv` (per-source measured mag, σ, before/after band counts).
