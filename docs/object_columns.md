# Per-object table columns

Every column of the field tables and the filter's object table, in pipeline order. **N** = survey-native value
(possibly renamed or cast; the source column is given). **N\*** = native value we transformed (calibrated, error-floored).
**O** = ours (computed by this pipeline). `{b}` is a band, g/r/i. Dtypes are as saved in parquet.

Files per field, under `Data/fields/<field>/`: `ztf_{b}.parquet`, `ztf_objects.parquet`, `ps1.parquet`
and `matches.parquet` (`Extracting/field_tables.py:build_field`), then `filter/objects.parquet` and `filter/<catalog>_candidates.parquet`
(`Source_Analysis/filter_field.py:filter_field`).

## 1. `ztf_{b}.parquet`: one row per DAOPhot source in band b

`band_catalog` (per quadrant, cached in `quadrants/`), concatenated and deduplicated by `dedupe_band`.

Native, from IRSA's `refpsfcat` (`PSF_COLS`):

| column | dtype | | source |
|---|---|---|---|
| `ra`, `dec` | float64 | N | `ra`, `dec` (deg) |
| `x`, `y` | float64 | N | `xpos`, `ypos` (1-based px in the reference image) |
| `mag` | float64 | N\* | `mag` + header `MAGZP` (refpsfcat is instrumental) |
| `magerr` | float64 | N | `sigmag` |
| `snr`, `chi`, `sharp` | float64 | N | `snr`, `chi`, `sharp` |
| `flags` | int64 | N | `flags` |
| `magzp` | float64 | N | refpsfcat header `MAGZP` |
| `maglim` | float64 | N | refimg header `MAGLIM` (`load_refimg`) |

Ours:

| column | dtype | meaning | computed by |
|---|---|---|---|
| `ccd`, `qid` | int64 | quadrant the row was measured in | `band_catalog` |
| `edge_px` | float64 | distance to the nearest edge of the 3200 px reference image, px | `band_catalog` |
| `nan_dist_px` | float64 | distance to the nearest bad (NaN) reference pixel, px | `band_catalog` (`distance_transform_edt`) |
| `zkron`, `zkron_err` | float64 | our Kron magnitude (AB, image MAGZP) and error; NaN if flux <= 0 | `kron.kron_photometry` |
| `zkron_r1` | float64 | first-moment Kron radius R1, arcsec (after floor / faint cap) | `kron.measure` |
| `zkron_rap` | float64 | Kron aperture radius 2.5 R1, arcsec, capped at 15" | `kron.measure` |
| `zkron_sky` | float64 | sky radius (first-pass window), arcsec | `kron.sky_radius` |
| `zkron_flag` | int64 | bits: 1 no radius, 2 at floor, 4 faint cap, 8 truncated at 15", 16 flux <= 0; -1 not measured (off frame) | `kron.measure` |
| `zkron_masked` | float64 | fraction of the Kron aperture masked (other segments, bad pixels) | `kron.aperture_sum` |
| `zkron_floor_arcsec` | float64 | the quadrant's R1 floor (median moment of S/N >= 40 sources), arcsec; constant per quadrant | `kron.moment_floor` |
| `n_duplicates_dropped` | int64 | overlap-strip duplicates dropped from this band table; constant per table | `dedupe_band` |

Tables built before commit 234ec43 also carry SExtractor `kron_auto`, `kron_auto_err`, `sexflags`, `class_star`
(and `{b}_kron_auto`, `{b}_kron_auto_err` in `ztf_objects.parquet`); the builder no longer writes them and nothing reads them.

## 2. `ztf_objects.parquet`: one row per ZTF object (bands merged)

| column | dtype | meaning | computed by |
|---|---|---|---|
| `ra`, `dec` | float64 | S/N-weighted mean position of the object's detections | `merge_bands` |
| `{b}_idx` | int64 | row in `ztf_{b}.parquet`, -1 if not detected in b | `merge_bands` |
| `{b}_sep` | float64 | separation of the b row from the object position, arcsec; NaN if absent | `merge_bands` |
| `{b}_covered` | bool | on a valid pixel of some quadrant's b reference image | `object_coverage` |

The photometry stays in the band tables; a table holds only its own data and the row indices of the tables it was made
from, never anything about tables made from it.

## 3. `ps1.parquet`: one row per PS1 stack object inside some reference image

`ps1_stack.clean` reduces the stack mirror to one row per objID; `field_tables.ps1_field` restricts it to the field.

Native (values of the selected stack row; selection rules in the `ps1_stack` docstring):

| column | dtype | | source |
|---|---|---|---|
| `PanSTARR_ID` | int64 | N | `objID` |
| `raMean`, `decMean`, `raMeanErr`, `decMeanErr`, `raStack`, `decStack`, `raStackErr`, `decStackErr`, `epochMean`, `qualityFlag`, `objInfoFlag`, `primaryDetection`, `nStackObjectRows`, `nStackDetections`, `nDetections`, `ng`, `nr`, `ni`, `projectionID`, `skyCellID` | various | N | same names (`OBJECT_COLS`), from the object's best row; -999 -> NaN |
| `{b}PSFMag`, `{b}PSFMagErr`, `{b}PSFFlux`, `{b}PSFFluxErr`, `{b}KronMag`, `{b}KronMagErr`, `{b}KronFlux`, `{b}KronFluxErr`, `{b}KronRad`, `{b}infoFlag`, `{b}infoFlag2`, `{b}infoFlag3`, `{b}psfQfPerfect`, `{b}ExtNSigma`, `{b}nFrames`, `{b}psfChiSq`, `{b}sky`, `{b}Epoch` | various | N | same names (`BAND_COLS`), from the band's best row; -999 -> NaN |
| `{b}_row` | int64 | N | `uniquePspsSTid` of the row band b came from |
| `{b}_projcell`, `{b}_skycell` | int16 | N | `projectionID`, `skyCellID` of the row band b came from |

Ours:

| column | dtype | meaning | computed by |
|---|---|---|---|
| `ra`, `dec` | float64 | stack position, mean position where there is none | `ps1_stack.clean` |
| `n_rows` | int64 | stack rows of the object (after dropping repeated `uniquePspsSTid`) | `ps1_stack.clean` |
| `n_primary_rows` | int64 | of those, rows with STACK_PRIMARY in any band | `ps1_stack.clean` |
| `{b}_detected` | bool | not forced (`{b}infoFlag2 & 4` == 0), PSFFluxErr > 0, PSFFlux finite | `ps1_stack.clean` |
| `ccd`, `qid` | int16 / int8 | ZTF quadrant chosen for the object (farthest from a masked pixel) | `ps1_field` |
| `nan_dist_px` | float64 | distance to the nearest bad ZTF pixel in that quadrant, min over bands whose image contains it, px | `ps1_field` |
| `{b}_covered` | bool | on a valid pixel of that quadrant's b image | `ps1_field` |
| `on_ztf` | bool | `{b}_covered` in any band | `ps1_field` |

## 3b. `matches.parquet`: one row per ZTF-PS1 match

| column | dtype | meaning | computed by |
|---|---|---|---|
| `ztf_obj` | int64 | row in `ztf_objects.parquet` | `match_ps1` (mutual nearest within 1") |
| `ps1` | int64 | row in `ps1.parquet` | `match_ps1` |
| `sep_arcsec` | float64 | separation of the pair | `match_ps1` |

## 4. `filter/objects.parquet`: the object table

One row per ZTF object (with its PS1 match) followed by every unmatched PS1 object. Errors and S/N in this table carry
the 0.01 mag calibration floor (`floor_err`, `floor_flux_err`, `floor_snr`); the field tables do not.

### 4a. Base table (`object_table`)

Object-level:

| column | dtype | | meaning / source |
|---|---|---|---|
| `ra`, `dec` | float64 | O | ZTF object position; PS1 position for PS1-only rows |
| `ztf_idx` | int64 | O | row in `ztf_objects.parquet`, -1 for PS1-only rows |
| `ps1_idx` | int64 | O | row in `ps1.parquet`, -1 without a PS1 object |
| `ccd`, `qid` | int16 / int8 | O | quadrant of the first detected band (g, r, i order); `ps1_field`'s quadrant for PS1-only rows |
| `ps1_clear` | bool | O | the PS1 position is on valid ZTF pixels and > 5 px (`NAN_RADIUS_PX`) from a masked one (`on_ztf & nan_dist_px > 5`) |
| `ps1_objid` | int64 | N | PS1 `objID`, -1 without a PS1 object |
| `ps1_ra`, `ps1_dec` | float64 | O | PS1 object's own position (`ps1.ra/dec`: stack, else mean) |
| `ps1_qualityFlag` | int64 | N | `qualityFlag` (0 without a PS1 object) |
| `ps1_nStackDetections` | int16 | N | `nStackDetections` (0 without) |
| `ps1_projcell`, `ps1_skycell` | int32 / int16 | N | `projectionID`, `skyCellID` of the object's best row (-1 without) |

Per band, ZTF side (from `ztf_{b}.parquet` via `{b}_idx`; NaN where not detected):

| column | dtype | | meaning / source |
|---|---|---|---|
| `{b}_ztf_det` | bool | O | DAOPhot detection in b |
| `{b}_ztf_psf` | float64 | N\* | `mag` (DAOPhot + MAGZP) |
| `{b}_ztf_psf_err` | float64 | N\* | `sigmag` with error floor |
| `{b}_ztf_snr` | float64 | N\* | DAOPhot `snr` with error floor |
| `{b}_ztf_chi`, `{b}_ztf_sharp` | float32 | N | DAOPhot `chi`, `sharp` |
| `{b}_ztf_flags` | int16 | N | DAOPhot `flags` (0 where not detected) |
| `{b}_ztf_kron`, `{b}_ztf_kron_err` | float64 | O | `zkron`, `zkron_err` (+ floor) |
| `{b}_ztf_kron_masked` | float32 | O | `zkron_masked` |
| `{b}_ztf_kron_rap` | float32 | O | `zkron_rap`, arcsec |
| `{b}_ztf_nan_dist` | float32 | O | `nan_dist_px` of the detection |
| `{b}_ztf_covered` | bool | O | `{b}_covered` (ZTF objects from `object_coverage`, PS1-only rows from `ps1_field`) |
| `{b}_ztf_maglim` | float64 | N | `MAGLIM` of the object's quadrant in b, NaN if the quadrant has no b image |

Per band, PS1 side (from the PS1 object; NaN / 0 / -1 without one):

| column | dtype | | meaning / source |
|---|---|---|---|
| `{b}_ps1_detected` | bool | O | `{b}_detected` (a detection, not forced) |
| `{b}_ps1_psf`, `{b}_ps1_kron` | float32 | N | `{b}PSFMag`, `{b}KronMag` |
| `{b}_ps1_psf_err`, `{b}_ps1_kron_err` | float32 | N\* | `{b}PSFMagErr`, `{b}KronMagErr` with floor |
| `{b}_ps1_flux` | float32 | N | `{b}PSFFlux` (Jy) |
| `{b}_ps1_flux_err` | float32 | N\* | `{b}PSFFluxErr` with floor |
| `{b}_ps1_kron_rad` | float32 | N | `{b}KronRad`, arcsec |
| `{b}_ps1_psfQfPerfect`, `{b}_ps1_sky` | float32 | N | `{b}psfQfPerfect`, `{b}sky` (Jy/arcsec²) |
| `{b}_ps1_infoFlag`, `{b}_ps1_infoFlag2` | int64 | N | `{b}infoFlag`, `{b}infoFlag2` (0 without) |
| `{b}_ps1_nFrames` | float32 | N | `{b}nFrames` (NaN without) |
| `{b}_ps1_projcell`, `{b}_ps1_skycell` | int32 / int16 | N | `{b}_projcell`, `{b}_skycell` (-1 without) |
| `{b}_ps1_meas` | bool | O | finite flux with error > 0 (forced included) |
| `{b}_ps1_snr` | float32 | O | `flux / flux_err` (floored) where `_meas` |
| `{b}_ps1_limit_nframes` | float32 | O | own nFrames if >= 1, else median nFrames of the 5 nearest PS1 objects within 60" (`nframes_at`) |
| `{b}_ps1_depth` | float32 | O | nominal 5σ stack depth for that nFrames, mag (`ps1_stack.stack_depth`) |

### 4b. Gaia (`annotate_gaia`, Gaia DR3 propagated to J2021.0, nearest within 5")

| column | dtype | meaning |
|---|---|---|
| `gaia_sep_arcsec` | float64 | separation to the Gaia match, arcsec; NaN without |
| `gaia_pm_sig` | float32 | `pm / (pmra_error + pmdec_error)` |
| `gaia_parallax_over_error` | float32 | Gaia `parallax_over_error` (N) |

### 4c. PS1 skycells and catalog holes (`annotate_skycells`)

| column | dtype | meaning |
|---|---|---|
| `{b}_ps1_skycell_bad` | bool | band b's PS1 row comes from a MAST-flagged polar skycell; without a PS1 object, any skycell covering the position is flagged |
| `ps1_skycell_bad` | bool | any band |
| `ps1_catalog_missing` | bool | no PS1 object and the position is in a DR2 missing-catalog region (`ps1_skycells.catalog_missing`) |

### 4d. Bright-star mask (`annotate_bright_stars`, Gaia DR2 G < 16)

| column | dtype | meaning |
|---|---|---|
| `bright_star_d_over_R` | float64 | distance to the nearest bright star in units of its mask radius (`bright_star_radius`); inf beyond 3 R; a star within 1" is the object itself |
| `in_star_spike` | bool | inside one of a star's four diffraction-spike boxes (`spike_half_length`) |
| `bright_star_masked` | bool | `bright_star_d_over_R < 1` or `in_star_spike` |

### 4e. Neighbours and PS1 coverage (`annotate_neighbours`)

| column | dtype | meaning |
|---|---|---|
| `nearest_other_arcsec` | float32 | distance to the nearest other ZTF or PS1 entry (not the object's own), arcsec; NaN beyond 60" |
| `inside_brighter_kron` | bool | inside the Kron aperture (ZTF `kron_rap` or PS1 `KronRad`, capped at 40") of a brighter object |
| `{b}_ps1_covered` | bool | some PS1 object with `{b}_ps1_meas` within 1' (PS1's b stack reaches the position) |

### 4f. Completeness limits (`completeness_limits`, called from `assign_catalogs`)

| column | dtype | meaning |
|---|---|---|
| `{b}_ztf_limit_psf`, `{b}_ztf_limit_kron` | float32 | ZTF's 90% recovery limit of clean PS1 detections in the object's quadrant, in PS1 PSF / Kron mag; capped at MAGLIM |
| `{b}_ps1_limit_psf`, `{b}_ps1_limit_kron` | float32 | PS1's 90% recovery limit of clean ZTF detections in the quadrant, in ZTF PSF / Kron mag; capped at the published stack depth and at `{b}_ps1_depth` |

### 4g. Statuses, catalog, magnitudes used (`assign_catalogs`)

| column | dtype | meaning |
|---|---|---|
| `catalog` | int8 | 0 in_both, 1 in_ztf, 2 in_pstarr, -1 not comparable (`object_catalog`) |
| `catalog_reason` | string | why -1 (`REASONS`); empty otherwise |
| `{b}_ztf_status`, `{b}_ps1_status` | string | D (S/N >= 5), L (below), U (nothing measured), X (side fails quality checks) (`detection_status`, `usable_sides`) |
| `{b}_cat` | int8 | comparison: 0 ZTF vs PS1, 1 ZTF vs PS1 limit, 2 PS1 vs ZTF limit, -1 none (`COMPARISON`; in_both keeps only 0) |
| `{b}_extendedness` | float64 | inverse-variance (PSF − Kron − 0.05)/σ over the two surveys; -inf with no Kron (`extendedness`) |
| `{b}_point` | bool | object-wide morphology (`point_source`), identical in every band |
| `{b}_kron_contaminated` | bool | ZTF-detected, extended, Kron aperture > 10% masked |
| `{b}_use_kron` | bool | the band compares Kron magnitudes (extended, compared, Kron on each needed side) |
| `{b}_ztf_limit`, `{b}_ps1_limit` | float32 | the `_limit_psf` or `_limit_kron` matching `_use_kron` |
| `{b}_ztf_use`, `{b}_ztf_use_err` | float64 | ZTF magnitude entering dmag (PSF or Kron; for `_cat` 2 the sub-threshold measurement or the limit, error 0 for a limit) |
| `{b}_ztf_use_meas` | bool | `_cat` 2 and ZTF's sub-threshold measurement is used instead of the limit |
| `{b}_ps1_use`, `{b}_ps1_use_err` | float32 | PS1 magnitude entering dmag (same rules, `_cat` 1) |
| `{b}_ps1_use_forced` | bool | `_cat` 1 and PS1's forced / sub-threshold measurement is used instead of the limit |

### 4h. Δmag and σ (`annotate_sigma`)

| column | dtype | meaning |
|---|---|---|
| `{b}_dmag` | float64 | `{b}_ztf_use − {b}_ps1_use`, mag (negative: brighter in ZTF) |
| `{b}_sigma` | float64 | signed σ against the band's KDE envelope; clipped one-sided against a limit; NaN for `_cat` -1 or `{b}_ps1_use` not > 14 |
| `{b}_capped` | bool | \|σ\| beyond the grid, reported as ±`SIGMA_CAP` (8) |

### 4i. Outcome record (`annotate_outcomes`)

| column | dtype | meaning |
|---|---|---|
| `{b}_dropped_by` | category | label of the first waterfall step that dropped band b; `kept` if none |
| `outcome` | category | `candidate`, `not comparable: <reason>`, or `dropped: <step>` (where fewer than two bands remained) |

## 5. `filter/<catalog>_candidates.parquet`

The object-table rows that pass `judge_bands` (all columns above), with:

| column | dtype | meaning | computed by |
|---|---|---|---|
| `catalog` | string | **overwritten** with the catalog name (`in_both`, ...); int8 code in `objects.parquet` | `judge_bands` |
| `n_big_dmag_bands` | int64 | bands with \|σ\| > 3 among those that reached the σ step | `judge_bands` |
