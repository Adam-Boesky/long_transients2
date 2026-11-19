# Table columns

`{b}` is g, r or i. **N** marks a survey-native value (source in brackets); everything else is ours. A table holds its
own data plus row indices of its parents, never anything about tables made from it, and no column that only restates
another.

## Files per field

```
Data/fields/<field>/
├── quadrants/                     builder caches, per quadrant and band
│   ├── ztf_c<cc>_q<q>_<b>.parquet   DAOPhot + our Kron for that image
│   └── mask_c<cc>_q<q>_<b>.npz      its bad-pixel mask
├── ztf_g.parquet                  ┐
├── ztf_r.parquet                  │ builder (Extracting.field_tables)
├── ztf_i.parquet                  │
├── ztf.parquet                    │
├── ps1.parquet                    │
├── merged.parquet                 ┘ the ZTF objects and PS1 objects joined on their match: what the filter reads
├── filter/                        filter (Source_Analysis.filter_field run), rewritten each run
│   ├── objects.parquet              merged + annotations + outcome
│   ├── in_both_candidates.parquet
│   ├── in_ztf_candidates.parquet
│   ├── in_pstarr_candidates.parquet
│   ├── all_stats.csv, in_both_stats.csv, in_ztf_stats.csv, in_pstarr_stats.csv
│   └── envelope_g.npz, envelope_r.npz, envelope_i.npz
└── review/                        notebooks only
```

## `ztf_{b}`: one row per DAOPhot source (`band_catalog`, `dedupe_band`)

| column | meaning |
|---|---|
| `ra`, `dec` | N (`refpsfcat` `ra`, `dec`) |
| `x`, `y` | N (`refpsfcat` `xpos`, `ypos`; 1-based reference px) |
| `mag` | N (`refpsfcat` `mag` + header MAGZP) |
| `magerr` | N (`refpsfcat` `sigmag`) |
| `snr` | N (`refpsfcat` `snr`) |
| `chi` | N (`refpsfcat` `chi`) |
| `sharp` | N (`refpsfcat` `sharp`) |
| `flags` | N (`refpsfcat` `flags`) |
| `magzp` | N (refpsfcat header MAGZP) |
| `maglim` | N (refimg header MAGLIM) |
| `ccd` | CCD measured in |
| `qid` | quadrant measured in |
| `nan_dist_px` | px to the nearest masked pixel |
| `zkron` | our Kron magnitude |
| `zkron_err` | its error |
| `zkron_r1` | Kron radius R1, arcsec |
| `zkron_rap` | Kron aperture 2.5 R1, arcsec |
| `zkron_sky` | sky radius, arcsec |
| `zkron_flag` | bits: 1 no radius, 2 floor, 4 faint cap, 8 truncated, 16 flux ≤ 0; −1 not measured |
| `zkron_masked` | masked fraction of the Kron aperture |
| `zkron_floor_arcsec` | the quadrant's R1 floor |
| `n_duplicates_dropped` | overlap duplicates dropped from the table |

## `ztf`: one row per ZTF object (`merge_bands`, `ztf_coverage`, `ps1_neighbourhood`)

| column | meaning |
|---|---|
| `ra`, `dec` | S/N-weighted mean of the detections |
| `{b}_idx` | row in `ztf_{b}` (−1 none) |
| `{b}_sep` | that row's offset, arcsec |
| `{b}_covered` | on a valid pixel of some b image |
| `{b}_ccd`, `{b}_qid` | chosen covering image: valid, then ≥ 5 px from a mask, then deepest (−1 none) |
| `{b}_nan_dist_px` | that image's px to a mask |
| `{b}_maglim` | that image's MAGLIM |
| `{b}_ps1_covered` | a PS1 object measured in b within 1′ |
| `{b}_ps1_nframes_local` | median nFrames of the 5 nearest PS1 objects within 60″ |

## `ps1`: one row per PS1 stack object in the field (`ps1_stack.clean`, `ps1_field`)

| column | meaning |
|---|---|
| `PanSTARR_ID` | N (`objID`) |
| `ra`, `dec` | stack position, mean where none |
| `raMean`, `decMean`, `raMeanErr`, `decMeanErr` | N |
| `raStack`, `decStack`, `raStackErr`, `decStackErr` | N |
| `epochMean` | N |
| `qualityFlag`, `objInfoFlag` | N |
| `primaryDetection` | N (wrong in DR2 per MAST; unused) |
| `nStackObjectRows`, `nStackDetections`, `nDetections`, `ng`, `nr`, `ni` | N |
| `projectionID`, `skyCellID` | N (the object's best row) |
| `n_rows`, `n_primary_rows` | stack rows of the object, of which primary |
| `{b}PSFMag`, `{b}PSFMagErr`, `{b}PSFFlux`, `{b}PSFFluxErr` | N (the band's best row) |
| `{b}KronMag`, `{b}KronMagErr`, `{b}KronFlux`, `{b}KronFluxErr`, `{b}KronRad` | N |
| `{b}infoFlag`, `{b}infoFlag2`, `{b}infoFlag3` | N |
| `{b}psfQfPerfect`, `{b}ExtNSigma`, `{b}psfChiSq` | N |
| `{b}nFrames` | N (stack input frames) |
| `{b}sky`, `{b}Epoch` | N |
| `{b}_row` | N (`uniquePspsSTid` of the band's row) |
| `{b}_projcell`, `{b}_skycell` | N (`projectionID`, `skyCellID` of the band's row) |
| `{b}_detected` | not forced (`infoFlag2 & 4` clear), finite flux, error > 0 |
| `{b}_covered`, `{b}_ccd`, `{b}_qid`, `{b}_nan_dist_px`, `{b}_maglim` | ZTF at the PS1 position, as in `ztf` |
| `{b}_nframes_local` | median nFrames of the 5 nearest other PS1 objects within 60″ |

## `merged`: one row per object (`match_ps1`, mutual nearest within 1″; `merge_tables`)

Every ZTF object with its PS1 match, then every unmatched PS1 object. Values as measured or catalogued, no error floor.
−1 / NaN / 0 where a survey has nothing.

| column | meaning |
|---|---|
| `ra`, `dec` | ZTF object position; PS1 position for PS1-only rows |
| `ztf_idx` | row in `ztf` (−1 none) |
| `ps1_idx` | row in `ps1` (−1 none) |
| `match_sep_arcsec` | ZTF–PS1 separation (NaN unmatched) |
| `ps1_ra`, `ps1_dec` | the PS1 object's position |
| `ps1_objid` | N (`objID`) |
| `ps1_qualityFlag` | N (`qualityFlag`) |
| `ps1_nStackDetections` | N (`nStackDetections`) |
| `{b}_ccd`, `{b}_qid` | image the band is judged in: the detection's, else the chosen covering one |
| `{b}_ztf_covered` | on a valid pixel of some b image |
| `{b}_ztf_nan_dist` | that image's px to a mask |
| `{b}_ztf_maglim` | that image's MAGLIM |
| `{b}_ztf_psf` | N (DAOPhot `mag`) |
| `{b}_ztf_psf_err` | N (DAOPhot `sigmag`) |
| `{b}_ztf_snr` | N (DAOPhot `snr`) |
| `{b}_ztf_sharp` | N (DAOPhot `sharp`) |
| `{b}_ztf_flags` | N (DAOPhot `flags`) |
| `{b}_ztf_kron`, `{b}_ztf_kron_err` | our Kron magnitude and error |
| `{b}_ztf_kron_masked` | masked fraction of the Kron aperture |
| `{b}_ztf_kron_rap` | Kron aperture radius, arcsec |
| `{b}_ps1_detected` | not forced photometry |
| `{b}_ps1_psf`, `{b}_ps1_psf_err` | N (`PSFMag`, `PSFMagErr`) |
| `{b}_ps1_kron`, `{b}_ps1_kron_err` | N (`KronMag`, `KronMagErr`) |
| `{b}_ps1_flux`, `{b}_ps1_flux_err` | N (`PSFFlux`, `PSFFluxErr`) |
| `{b}_ps1_kron_rad` | N (`KronRad`) |
| `{b}_ps1_psfQfPerfect` | N |
| `{b}_ps1_sky` | N |
| `{b}_ps1_infoFlag`, `{b}_ps1_infoFlag2` | N |
| `{b}_ps1_projcell`, `{b}_ps1_skycell` | N (the band row's skycell) |
| `{b}_ps1_nFrames` | N (the object's own stack frames; NaN without a band row) |
| `{b}_ps1_nframes_local` | median nFrames of the 5 nearest PS1 objects; used where the object has no frames of its own |
| `{b}_ps1_covered` | PS1 measured something in b within 1′ |


## `filter/objects`: `merged`, floored and annotated (`prepare_field`, `filter_prepared`)

Every `merged` column (errors and S/N with a 0.01 mag floor, `floor_err`), plus:

| column | meaning |
|---|---|
| `{b}_ps1_meas` | finite PS1 flux with error > 0 (forced included) |
| `{b}_ps1_snr` | PS1 flux / error |
| `gaia_sep_arcsec` | to the nearest Gaia DR3 source within 5″ (`annotate_gaia`) |
| `gaia_pm`, `gaia_pmra`, `gaia_pmdec` | N (Gaia DR3), mas/yr |
| `gaia_pmra_err`, `gaia_pmdec_err` | N (Gaia DR3), mas/yr |
| `gaia_parallax_over_error` | N (Gaia DR3) |
| `ps1_catalog_missing` | no PS1 object, in a DR2 missing-data box or Dec < −30 (`annotate_skycells`) |
| `bright_star_d_over_R` | to the nearest G < 16 star, in its mask radii (`annotate_bright_stars`) |
| `in_star_spike` | in one of its four spike boxes |
| `nearest_other_arcsec` | to the nearest other ZTF or PS1 entry, NaN beyond 60″ (`annotate_neighbours`) |
| `inside_brighter_kron` | inside a brighter object's Kron aperture |
| `{b}_ztf_limit` | ZTF's 90% recovery limit in the band's quadrant, in the object's compared magnitude (`completeness_limits`) |
| `{b}_ps1_limit` | PS1's, likewise |
| `{b}_ztf_status`, `{b}_ps1_status` | D (S/N ≥ 5), L (below), U (nothing measured), X (fails quality) (`assign_catalogs`) |
| `{b}_cat` | 0 ZTF vs PS1, 1 ZTF vs PS1 limit, 2 PS1 vs ZTF limit, −1 none |
| `catalog` | 0 in_both, 1 in_ztf, 2 in_pstarr, −1 not comparable |
| `catalog_reason` | why −1 |
| `{b}_extendedness` | (PSF − Kron − 0.05)/σ, inverse-variance over both surveys |
| `point_source` | not extended in ≥ 2 measured bands |
| `{b}_use_psf` | compares PSF (else Kron) |
| `{b}_ztf_use_meas` | ZTF's limit side uses its own sub-threshold measurement |
| `{b}_ps1_use_forced` | PS1's limit side uses its own forced / sub-threshold measurement |
| `{b}_dmag` | ZTF − PS1 of the compared magnitudes (`annotate_sigma`) |
| `{b}_sigma` | σ against the band's envelope |
| `{b}_dropped_by` | first step that dropped the band, or `kept` (`annotate_outcomes`) |
| `outcome` | `candidate`, `not comparable: <reason>`, or `dropped: <step>` |

Derived in the filter, not stored: ZTF detection (`ztf_detected`), compared magnitudes (`compared_magnitudes`), PS1
depth (`ps1_depth`), polar-skycell badness (`ps1_skycell_bad`), the bright-star mask (`bright_star_masked`), Gaia
significances (`gaia_population_ok`).

## `filter/<catalog>_candidates`

The `objects` rows that pass, plus:

| column | meaning |
|---|---|
| `n_big_dmag_bands` | bands with \|σ\| > 3 |

`catalog` here is the name, not the code.
