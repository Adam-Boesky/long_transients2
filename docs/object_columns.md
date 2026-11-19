# Table columns

Per field, under `Data/fields/<field>/`. Builder (`Extracting/field_tables.py`): `ztf_{b}`, `ztf_objects`, `ps1`,
`matches`. Filter (`Source_Analysis/filter_field.py`): `filter/objects`, `filter/<catalog>_candidates`. `{b}` is g, r
or i. **N** marks a survey-native value (source in brackets); everything else is ours. A table holds its own data plus
row indices of its parents, never anything about tables made from it.

## `ztf_{b}`: one row per DAOPhot source (`band_catalog`, `dedupe_band`)

| column | meaning |
|---|---|
| `ra`, `dec`, `x`, `y` | N (`refpsfcat`; x, y in 1-based reference px) |
| `mag`, `magerr`, `snr`, `chi`, `sharp`, `flags` | N (`refpsfcat`; `mag` + header MAGZP) |
| `magzp`, `maglim` | N (refpsfcat MAGZP, refimg MAGLIM) |
| `ccd`, `qid` | quadrant measured in |
| `edge_px`, `nan_dist_px` | px to the image edge, to the nearest masked pixel |
| `zkron`, `zkron_err` | our Kron magnitude (`kron`) |
| `zkron_r1`, `zkron_rap`, `zkron_sky` | Kron radius R1, aperture 2.5 R1, sky radius; arcsec |
| `zkron_flag` | bits: 1 no radius, 2 floor, 4 faint cap, 8 truncated, 16 flux ≤ 0; −1 not measured |
| `zkron_masked` | masked fraction of the Kron aperture |
| `zkron_floor_arcsec` | the quadrant's R1 floor |
| `n_duplicates_dropped` | overlap duplicates dropped from the table |

## `ztf_objects`: one row per ZTF object (`merge_bands`, `ztf_coverage`, `ps1_neighbourhood`)

| column | meaning |
|---|---|
| `ra`, `dec` | S/N-weighted mean of the detections |
| `{b}_idx`, `{b}_sep` | row in `ztf_{b}` (−1 none); its offset, arcsec |
| `{b}_covered` | on a valid pixel of some b image |
| `{b}_ccd`, `{b}_qid` | chosen covering image: valid, then ≥ 5 px from a mask, then deepest (−1 none) |
| `{b}_nan_dist_px`, `{b}_maglim` | that image's px to a mask, MAGLIM |
| `{b}_ps1_covered` | a PS1 object measured in b within 1′ |
| `{b}_ps1_nframes_local` | median nFrames of the 5 nearest PS1 objects within 60″ |

## `ps1`: one row per PS1 stack object in the field (`ps1_stack.clean`, `ps1_field`)

| column | meaning |
|---|---|
| `PanSTARR_ID` | N (`objID`) |
| `raMean` … `skyCellID` | N (ObjectThin columns, `ps1_stack.OBJECT_COLS`; −999 → NaN) |
| `{b}PSFMag` … `{b}Epoch` | N (stack band columns, `ps1_stack.BAND_COLS`, from the band's best row) |
| `{b}_row`, `{b}_projcell`, `{b}_skycell` | N (`uniquePspsSTid`, `projectionID`, `skyCellID` of the band's row) |
| `ra`, `dec` | stack position, mean where none |
| `n_rows`, `n_primary_rows` | stack rows, of which primary |
| `{b}_detected` | not forced (`infoFlag2 & 4` clear), finite flux, error > 0 |
| `{b}_covered`, `{b}_ccd`, `{b}_qid`, `{b}_nan_dist_px`, `{b}_maglim` | ZTF at the PS1 position, as in `ztf_objects` |
| `{b}_nframes_local` | median nFrames of the 5 nearest other PS1 objects within 60″ |

## `matches`: one row per ZTF–PS1 pair (`match_ps1`, mutual nearest within 1″)

| column | meaning |
|---|---|
| `ztf_obj`, `ps1` | rows in `ztf_objects`, `ps1` |
| `sep_arcsec` | separation |

## `filter/objects`: one row per ZTF object, then per unmatched PS1 object

Errors and S/N carry a 0.01 mag floor (`floor_err`).

**Assembled** (`object_table`)

| column | meaning |
|---|---|
| `ra`, `dec` | ZTF position; PS1 position for PS1-only rows |
| `ztf_idx`, `ps1_idx` | rows in `ztf_objects`, `ps1` (−1 none) |
| `ps1_ra`, `ps1_dec` | the PS1 object's position |
| `ps1_objid`, `ps1_qualityFlag`, `ps1_nStackDetections` | N (`objID`, `qualityFlag`, `nStackDetections`) |
| `{b}_ccd`, `{b}_qid` | image the band is judged in: the detection's, else the chosen covering one |
| `{b}_ztf_det` | DAOPhot detection in b |
| `{b}_ztf_psf`, `{b}_ztf_psf_err`, `{b}_ztf_snr`, `{b}_ztf_sharp`, `{b}_ztf_flags` | N (from `ztf_{b}`) |
| `{b}_ztf_kron`, `{b}_ztf_kron_err`, `{b}_ztf_kron_masked`, `{b}_ztf_kron_rap` | our Kron (from `ztf_{b}`) |
| `{b}_ztf_nan_dist`, `{b}_ztf_maglim` | the band image's px to a mask, MAGLIM |
| `{b}_ztf_covered` | on a valid pixel of some b image |
| `{b}_ps1_psf`, `{b}_ps1_kron`, `{b}_ps1_psf_err`, `{b}_ps1_kron_err`, `{b}_ps1_flux`, `{b}_ps1_flux_err` | N (`PSFMag`, `KronMag`, errors, `PSFFlux` and error) |
| `{b}_ps1_kron_rad`, `{b}_ps1_psfQfPerfect`, `{b}_ps1_sky`, `{b}_ps1_infoFlag`, `{b}_ps1_infoFlag2`, `{b}_ps1_nFrames` | N |
| `{b}_ps1_projcell`, `{b}_ps1_skycell` | N (the band row's skycell; −1 none) |
| `{b}_ps1_detected` | from `ps1.{b}_detected` |
| `{b}_ps1_meas`, `{b}_ps1_snr` | finite flux with error > 0 (forced included); flux / error |
| `{b}_ps1_covered`, `{b}_ps1_nframes_local` | from `ztf_objects` / `ps1` |

**Annotations**

| column | meaning | by |
|---|---|---|
| `gaia_sep_arcsec` | to the nearest Gaia DR3 source within 5″ | `annotate_gaia` |
| `gaia_pm`, `gaia_pmra`, `gaia_pmdec`, `gaia_pmra_err`, `gaia_pmdec_err`, `gaia_parallax_over_error` | N (Gaia DR3) | `annotate_gaia` |
| `ps1_catalog_missing` | no PS1 object, in a DR2 missing-data box or Dec < −30 | `annotate_skycells` |
| `bright_star_d_over_R` | to the nearest G < 16 star, in its mask radii | `annotate_bright_stars` |
| `in_star_spike` | in one of its four spike boxes | `annotate_bright_stars` |
| `bright_star_masked` | `d_over_R < 1` or `in_star_spike` | `annotate_bright_stars` |
| `nearest_other_arcsec` | to the nearest other ZTF or PS1 entry (NaN beyond 60″) | `annotate_neighbours` |
| `inside_brighter_kron` | inside a brighter object's Kron aperture | `annotate_neighbours` |

**Catalogs and comparison** (`completeness_limits`, `assign_catalogs`, `annotate_sigma`)

| column | meaning |
|---|---|
| `{b}_ztf_limit`, `{b}_ps1_limit` | 90% recovery limit in the band's quadrant, in each object's compared magnitude |
| `{b}_ztf_status`, `{b}_ps1_status` | D (S/N ≥ 5), L (below), U (nothing measured), X (fails quality) |
| `{b}_cat` | 0 ZTF vs PS1, 1 ZTF vs PS1 limit, 2 PS1 vs ZTF limit, −1 none |
| `catalog`, `catalog_reason` | 0 in_both, 1 in_ztf, 2 in_pstarr, −1 not comparable; why |
| `{b}_extendedness` | (PSF − Kron − 0.05)/σ, inverse-variance over both surveys |
| `point_source` | not extended in ≥ 2 measured bands |
| `{b}_use_psf` | compares PSF (else Kron) |
| `{b}_ztf_use_meas`, `{b}_ps1_use_forced` | a limit side uses its own sub-threshold measurement |
| `{b}_dmag`, `{b}_sigma` | ZTF − PS1 of the compared magnitudes (`compared_magnitudes`); σ against the envelope |

**Outcome** (`annotate_outcomes`)

| column | meaning |
|---|---|
| `{b}_dropped_by` | first step that dropped the band, or `kept` |
| `outcome` | `candidate`, `not comparable: <reason>`, or `dropped: <step>` |

Derived in the filter, not stored: compared magnitudes (`compared_magnitudes`), PS1 depth (`ps1_depth`), polar-skycell
badness (`ps1_skycell_bad`), Gaia significances (`gaia_population_ok`).

## `filter/<catalog>_candidates`

The `objects` rows that pass, plus `n_big_dmag_bands` (bands with |σ| > 3). `catalog` here is the name, not the code.
