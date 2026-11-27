# Filtering review: open items (2026-11-27)

State: pipeline at commit 11c6b83 (per-morphology envelopes, diffuse-light check, rolling clip, model limits at 75 %,
nStackDetections for in_pstarr only, one two-sided band + D/L bands as in_both, forced fluxes at S/N >= 3).
Field 860: in_both 303, in_ztf 19 (171 before the in_ztf profile checks), in_pstarr 116. The 24 follow-up fields: 11,007 / 6,203 / 3,410 (~400k all-sky).
Queue sources recovered 22/26; Follow-up label 22/29; TNS matches 48/82; AGN-DB 60 %; Trash 2/86.

## Spot-check of every filter (field 860)
`Data/fields/000860/review/experiments/unique_by_check/index.md` (counts, figures, a review column per check).
The three ZTF quality checks that remove most of the lost queue / Follow-up / TNS sources on the 24 fields, and the
largest unique removers on 860:
- bright-star mask: spike-box reach for G 12-15 stars (68 in_both, 408 in_ztf unique on 860)
- Kron aperture <= 10 % masked: removes real galaxies with contaminated apertures (388 in_both); a PSF fallback?
- DAOPhot chi <= 6 on galaxies (13 in_both, 25 in_ztf); point sources only?
Also: isolation at 3" now removes 1,338 in_ztf / 527 in_pstarr uniquely (the 75 % limits enlarge the one-sided
catalogs); PS1 residual sky threshold (24 in_both, ordinary objects just over 1e-7).

## Other
- in_pstarr objects 3-4" from a brighter star: PS1 deblends a companion 3-4 mag fainter that ZTF (2" FWHM) folds into
  the star; DAOPhot never splits it, so there is no ZTF entry at all. On the 24 fields 69 % of in_pstarr candidates
  have another object within 4" and 68 % sit inside a brighter object's Kron aperture (the nearest-other histogram
  peaks at 3.0-3.5", right at the isolation radius). Examples and stamps: `Data/fields/review_followup/
  in_pstarr_offset_examples.png`. PLAN: forced ZTF photometry at the PS1 position for PS1-only objects without a ZTF
  match, as a simultaneous two-source PSF fit (the ZTF-detected neighbour's PSF at its position plus one at the PS1
  position, the builder's ePSF, linear least squares, uncertainty from the covariance). This is how psphot measures
  blends (BLEND_FIT) and its forced MATCHED rows. The ZTF side then becomes a measurement (L or D) instead of U, and
  these objects sort as in_both and are judged by sigma. A single forced PSF fit would not do: it measures the star's
  wing. Blends inside ~1 FWHM stay degenerate; flag them.
- in_ztf over-deblends (DONE in the pipeline, not yet rerun on the 24 fields): nearly all in_ztf candidates were DAOPhot
  fragments of galaxies (87 % inside a brighter object's Kron aperture on 860; `in_ztf_random20.png`,
  `in_ztf_sharp_or_chi_removed.png`). The ZTF shape cuts (|sharp| <= 0.5 point sources, chi <= 6) were set to keep
  galaxies, so fragments passed. New in_ztf-only confident checks: S/N^2-weighted |sharp| <= 0.25 and median DAOPhot
  chi <= 2.5 (`Cuts.in_ztf_sharp_max`, `Cuts.in_ztf_chi_max`), the pair a ZTF planetary-nebula study used, inside the
  ZSDS point-source ranges (|sharp| <~ 0.5, chi <~ 1.5) and at the 99th percentiles of 860's in_both point sources
  (per S/N bin: 0.28/1.84 at S/N 5-10, 0.25/2.39 at 10-20, 0.17/1.96 at 20-50, 0.05/0.95 above 50; figure
  `Data/fields/review_followup/in_ztf_sharp_chi_vs_reference.png`). On the 25 fields' candidate tables it drops 85 %
  of in_ztf (5405/6374), none of the queue (20 in in_ztf), TNS (5), Follow-up (16), Interesting? (7), Known
  Transient (1) or AGN Candidate (5) objects, and 6 of 13 in_ztf AGN-DB matches (all inside a brighter Kron
  aperture); applied to in_both for reference it would drop 0.8 % of 5,722 AGN-DB matches and 4 % of 860's in_both
  point sources. The tighter (0.2, 2) pair dropped 92 % but also the two in_ztf AGN-DB matches that could be real
  nuclei (z = 0.75 QSO at 0.18/2.03, z = 0.76 at 0.12/2.4). The PS1 side has no sharpness; psfChiSq (99 % of in_both
  point sources 4.0, in_pstarr candidates 95 % 2.8) is too weak to tune. TODO: rerun 860 and the 24 fields; stamps of
  the 6 removed AGN-DB matches from Cannon; Kron containment (`neighbour_kron`) on top would leave ~8 per field
  instead of ~40.
- Rebuild the follow-up site DB on the current candidate tables (the site serves the 2026-11-26 catalog).
- 234p588979_71p430729 is now a candidate; 210p089826 and 183p543919 fail the sigma rule against the PS1 limit;
  199p872494 (Kron masking) and 94p007109 (spike box) leave at quality.
