# Additional filters

Cuts we plan to add to the ZTF–PS1 filtering pipeline (`Source_Analysis/filter_field.py`), one entry per cut.

## PS1

- **`{b}psfQfPerfect > 0.85`** *Implemented.* (per band, on the PS1 measurement each band uses). `psfQfPerfect` is the PSF-weighted
  fraction of the source's pixels that are unmasked, counting both bad and "suspect" pixels (spikes, ghosts, star
  cores, burntool), so it flags measurements compromised by masking rather than selecting by morphology. 0.85 is the
  PS1 team's "generous" quality threshold (Magnier et al. 2020, pixel analysis §4.5; 0.95 is their strict one), also
  used in MAST's example queries and the PS1 quasar searches (Bañados et al. 2014, 2016).
- **`{b}infoFlag2 & 8 == 0` (ON_SPIKE)** *Implemented.* (per band, as above). PS1 sets ON_SPIKE when more than 25% of the source's
  PSF-weighted pixels lie on a diffraction spike (Magnier et al. 2020, pixel analysis §4.5); it complements
  PEAK_ON_SPIKE, which we already cut and which only tests the peak pixel. The PS1 quasar searches exclude it together
  with ON_STARCORE and ON_BURNTOOL (Bañados et al. 2014, 2016); in field 860 it is set on 0.09% of PS1-only candidates
  and on no real matched objects.
- **`nStackDetections >= 2`, in_pstarr only.** *Implemented.* A PS1-only object must be detected (not forced) in at least two PS1
  stack bands, since nothing from ZTF confirms it. Chambers et al. (2016) find that most single faint detections are not
  real, and the GHOST host catalog sees artefacts rise sharply without a multi-band detection requirement (Gagliano et
  al. 2021). In field 860, after the stack-quality cut, 33–43% of PS1-only objects are single-band at every magnitude
  against 0–7% of matched (in_both) objects, so the cut removes about half of the in_pstarr candidates (52% are
  single-band) at little cost to real sources; in_both is exempt, its ZTF match standing in as the second confirmation.

## ZTF / all catalogs

- **Bright-star mask: drop objects within R = 1630″ × 1.396^(−G) of a Gaia G < 16 star** (the Legacy Surveys DR9
  MEDIUM-bit radius). ZTF's DAOPhot catalogs have no per-pixel spike/halo flags (the ZTF pipeline masks halos only
  around V ≤ 6.5 stars) and over-deblend star wings, so fragments and halo detections reach the in_ztf candidates. In
  field 860, candidates are 10–60× overdense inside R, and the mask removes 48% of in_ztf, 23% of in_pstarr and 16% of
  in_both candidates, against 5% of labelled real sources. PS1's flag cuts already clear most bright-star debris from
  in_pstarr, so the main gain is in_ztf. Still open: apply it to in_ztf only, or to all three catalogs.
  *Implemented* in all three catalogs (`bright_star_masked`), with four diffraction-spike boxes added 2026-11-18 after
  calibrating on field 860's images (`Notebooks/star_mask_860.ipynb`): PA 0, 90, 30 and 150°, 12″ wide, half-length
  R × 1.15^(16 − G), so the boxes collapse to the circle at G = 16 and reach ~3 R for the brightest stars. ZTF's spikes
  sit at those angles in every quadrant; PS1 stacks rotate them away, so the boxes mainly matter for in_ztf.
- **ZTF DAOPhot `|sharp| ≤ 0.5`, object-wide, point sources only.** *Implemented* (`Cuts.sharp_max`, `Cuts.sharp_point_only`;
  2026-11-17: on all objects the cut removed 4.9k in_both galaxies (|sharp| median 0.23, the tail of the extended
  population) including two known AGN, and the point sources it removes are diffuse blobs, single-band spikes and
  poor-fit blends with no known or labelled object among them; limited to point sources it adds 3 in_both candidates
  and loses none). The original note:** Sharpness is FWHM²(observed) − FWHM²(PSF) (ZTF
  Explanatory Supplement §10.6). The ±0.5 range is ZTF's recommended point-source test for the reference source matched
  to an alert (§13.8 note 13); the forced-photometry guide (Masci et al. 2023) and IPAC's own reference-star selection
  use χ and sharpness the same way, and a ZTF study of planetary-nebula central stars (arXiv:2410.03589) cuts at
  |sharp| < 0.25 with χ < 2.5. Classic DAOPhot crowded-field work (e.g. Annibali et al. on NGC 1705) uses |sharp| ≤ 0.5
  to reject non-stellar detections. Every precedent uses it to select point sources, not to clean artefacts while
  keeping galaxies. In field 860 it keeps 94% of extended and 99% of real objects and removes about half of the in_ztf
  candidates, but we'd rather find something more principled for extended sources before adopting it (χ is worse: it
  keeps only 16% of extended objects).

# Filters by purpose

Every existing (E), proposed (P) and deferred (D) cut, sorted by what it decides. The scope column says what a failure
removes: one band, or the whole object.

**Structure, not a filter.** Catalog assignment (in_both / in_ztf / in_pstarr by a ZTF–PS1 object match within 1″);
an object leaves once fewer than two usable bands remain; morphology (`point_source`, PSF − Kron vote) picks PSF or
Kron magnitudes and decides whether the Kron-contamination cut applies.

## 1. Quality: is this band's measurement reliable?

| Cut | Catalogs | Scope | Status |
|---|---|---|---|
| ZTF image covers the position | in_both, in_pstarr | band | E |
| ZTF DAOPhot flags = 0 | in_both, in_ztf | band | E |
| ZTF > 5 px from a masked pixel | in_both, in_ztf | band | E |
| PS1 position on valid ZTF pixels, > 5 px from masked | in_both, in_pstarr | object | E |
| PS1 not BAD_STACK / SUSPECT_STACK (`qualityFlag`) | in_both, in_pstarr | object | E |
| PS1 position not in a MAST-flagged mis-registered polar skycell (`Extracting/ps1_skycells.py`, `badcells_near_north_pole_v6.dat`; applies without a PS1 object too) | all | object | E |
| PS1 not DEFECT, PEAK_ON_SPIKE, SATSTAR (`infoFlag`) | in_both, in_pstarr | object* | E |
| PS1 not ON_STARCORE, ON_BURNTOOL, SATSTAR_PROFILE (`infoFlag2`) | in_both, in_pstarr | object* | E |
| Extended objects: ZTF Kron aperture ≤ 10% masked | in_both, in_ztf | band | E |
| PS1 `psfQfPerfect > 0.85` | in_both, in_pstarr | band | E |
| PS1 not ON_SPIKE (`infoFlag2 & 8`) | in_both, in_pstarr | band | E |

\* Currently object-level: a flag on any PS1-measured band, forced bands included, removes every band of the object.
Under this scheme they would become per band.

## 2. Confident: are we confident the object is real?

| Cut | Catalogs | Scope | Status |
|---|---|---|---|
| `nStackDetections ≥ 2` | in_pstarr | object | E |
| Bright-star mask, R = 1630″ × 1.396^(−G), Gaia G < 16 | in_ztf (all?) | object | P |
| ZTF DAOPhot \|sharp\| ≤ 0.5 | in_ztf (all?) | band | D |

## 3. Is it detected?

| Cut | Catalogs | Scope | Status |
|---|---|---|---|
| PSF S/N ≥ 5 on at least one side (also sets the comparison: both measured, or one-sided against a limit) | all | band | E |

## 4. Population: is it from a population we want?

| Cut | Catalogs | Scope | Status |
|---|---|---|---|
| Gaia: no proper motion at 5σ | all | object | E |
| Gaia: no parallax at 5σ | all | object | E |

## 5. Interesting

| Cut | Catalogs | Scope | Status |
|---|---|---|---|
| \|σ\| > 3 in ≥ 2 bands | all | object | E |
| … ≥ 2 of them the same sign | all | object | E |

## 6. Best (last)

| Cut | Catalogs | Scope | Status |
|---|---|---|---|
| Every side measured in a band's comparison (not a limit) has S/N ≥ 10; the \|σ\| rule re-checked on the bands left | all | band | E |

The envelope is built from the S/N ≥ 5 population before this step; the step only selects among the candidates.

**Order (implemented):** 1 quality → 2 confident → 3 detected → 4 population → 5 interesting → 6 best. Each
waterfall step carries its group (`ff.GROUPS`, `Waterfall.groups`, the `group` column of `wf.stats`), and
`Waterfall.flowchart` frames each group.

# Band statuses, comparisons and catalogs (implemented)

Each band gets one status per survey, stored as `{b}_ztf_status` and `{b}_ps1_status`; the pair sets the band's
comparison, `{b}_cat` (`COMPARISON` in `filter_field.py`). An object's catalog is which surveys have a **D** in at least
one band. An object needs two compared bands to stay in. These cases are the unit tests in
`tests/source_analysis/test_filter_field_catalogs.py`.

| Status | ZTF | PS1 |
|:-:|---|---|
| **D** | DAOPhot detection, S/N ≥ 5 | PSF flux S/N ≥ 5 (detected or forced) |
| **L** | DAOPhot detection, S/N < 5 | PSF flux S/N < 5, forced fluxes ≤ 0 included |
| **U** | no DAOPhot detection in the band | no PS1 object within 1″ |
| **X** | fails a trust check (incl. not covered by ZTF) | fails a trust check (incl. no measurement in the band) |

Comparisons: **ZTF vs PS1** (D, D) · **ZTF vs PS1 limit** (D, L or D, U; PS1's forced measurement where that is
brighter) · **PS1 vs ZTF limit** (L, D or U, D; the sub-threshold ZTF detection where that is brighter) · every
other pair: band dropped.

**Limits, symmetric** (`completeness_limits`): each survey's limit per ZTF quadrant is the magnitude at which its
recovery of the other survey's detections falls to 90% (recovered = a measurement in the band at the 1″ match: a ZTF detection at any S/N, a PS1 detection or forced flux > 0, what the pipeline uses in place of the limit), computed in PSF and
in Kron magnitude so a band uses the kind it compares; never deeper than the survey's nominal depth (ZTF: MAGLIM;
PS1: the nFrames stack depth). The reference is the other survey's detections that pass their own trust and real
checks in ≥ 2 bands — the population the one-sided candidates come from. Raw counts do not work: ~10% of all ZTF
detections have no PS1 object at any magnitude (fragments, star wings, single-band junk), so PS1's recovery of them
never reaches 90% and the limit collapses to the 19 mag guard (field 860: 19.5 instead of a measured ≈ 22.4 in r,
against a nominal 23.1). No PS1 coverage or mask information is used (it would need PS1's image products).

**The catalog sets the evidence.** An in_both object compares only its (D, D) bands; its one-sided bands are dropped
rather than compared against a limit, so every σ it carries is a two-sided measurement. Limits enter only in in_ztf
and in_pstarr, where there is nothing else. A sub-threshold side (L) is a limit on either survey, symmetrically.

| # | ZTF g | ZTF r | ZTF i | PS1 g | PS1 r | PS1 i | Catalog | Outcome |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|---|
| 1 | D | D | D | D | D | D | in_both | g, r, i: ZTF vs PS1 |
| 2 | D | D | D | D | L | L | in_both | g only → **object dropped** |
| 3a | D | D | D | D | X (no measurement) | X (no measurement) | in_both | g only → **object dropped** |
| 3b | D | D | D | D | L (forced ≤ 0) | L (forced ≤ 0) | in_both | g only → **object dropped** |
| 4 | D | D | D | L | L | L | in_ztf | g, r, i: ZTF vs PS1 limit |
| 4′ | L | L | L | D | D | D | in_pstarr | g, r, i: PS1 vs ZTF limit (the < 5σ ZTF value where brighter) |
| 5 | L | D | D | D | D | D | in_both | g dropped · r, i: ZTF vs PS1 |
| 6 | U | D | D | D | D | D | in_both | g dropped · r, i: ZTF vs PS1 |
| 7 | X | D | D | D | D | D | in_both | g dropped · r, i: ZTF vs PS1 |
| 8 | D | D | D | D | D | X | in_both | g, r: ZTF vs PS1 · i dropped |
| 9 | L | L | D | L | L | D | in_both | i only → **object dropped** |
| 10 | D | D | D | U | U | U | in_ztf | g, r, i: ZTF vs PS1 limit |
| 11 | D | D | U | U | U | U | in_ztf | g, r: ZTF vs PS1 limit |
| 12 | D | L | U | U | U | U | in_ztf | g only → **object dropped** |
| 13 | U | U | U | D | D | D | in_pstarr | g, r, i: PS1 vs ZTF limit |
| 14 | U | U | U | D | D | L | in_pstarr | g, r: PS1 vs ZTF limit |
| 15 | U | U | U | D | X (no measurement) | X (no measurement) | in_pstarr | g only → **object dropped** |
| 16 | U | U | U | D | D | X | in_pstarr | g, r: PS1 vs ZTF limit · i dropped |
| 17 | X (not covered) | X (not covered) | X (not covered) | D | D | D | in_pstarr | no band compared → **object dropped** |
| 18 | L | L | L | L | L | L | none (`undetected`) | **object dropped** |

Changes from the previous pipeline: 4 and 4′ (catalog by detection), 8 and 16 (PS1 flags per band instead of per
object), and 2, 3b, 5, 6 (in_both drops one-sided bands instead of comparing them against a limit; an in_both object
that faded out of two bands and kept one ZTF detection is lost rather than moved to in_pstarr). 4′ uses the < 5σ ZTF
value where it is brighter than the completeness limit, which it rarely is.

Open: BAD_STACK / SUSPECT_STACK, an object-level PS1 flag, marks every PS1 band X (≈230k ZTF-detected objects in field
860).

Caveat on `nStackDetections`: it sums detections over all of an object's stack rows and over grizy, so an object in a
skycell overlap detected in one band counts 2. In field 860, 1,029 of the 3,534 in_pstarr candidates that pass have at
most one detected g/r/i band (768 of them in overlaps); some may be detected in z or y.
