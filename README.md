# long_transients

## The filtering pipeline at a glance

```mermaid
flowchart TD
    IN["<b>Input</b><br/>one row per object: ZTF detections in g, r, i (DAOPhot + our Kron photometry)<br/>+ the matched PS1 stack object, or a PS1-only object"]
    Q["<b>1 · Quality</b> — per band, per survey<br/>is the data here usable?<br/><i>usable, or X</i>"]
    D["<b>2 · Detection</b> — per band, per usable side<br/>what does the survey have?<br/><i>D measured at S/N ≥ 5 · L measured below · U nothing: its limit stands in</i>"]
    S["<b>3 · Comparison and sorting</b><br/>per band: (D, D) two-sided · (D, L/U) ZTF vs PS1 limit · (L/U, D) PS1 vs ZTF limit · an X or (L/U, L/U): nothing<br/>per object: ≥ 2 bands of one kind → that catalog, else not comparable (with the reason)"]
    NC["not comparable"]
    IN --> Q --> D --> S
    S -- no --> NC

    subgraph BOTH ["in_both — ZTF vs PS1"]
        direction TB
        B4["<b>4 · Confident</b><br/>bright-star mask · sharpness (point sources)<br/>PS1 nStackDetections ≥ 2, not BAD_STACK, not SUSPECT_STACK"]
        B5["<b>5 · Population</b><br/>no Gaia proper motion or parallax at 5σ"]
        B7["<b>7 · Interesting</b><br/>σ from the envelope, either sign<br/>|σ| > 3 in ≥ 2 bands, ≥ 2 the same sign"]
        B8["in_both candidates"]
        B4 --> B5 --> B7 --> B8
    end
    subgraph ZTF ["in_ztf — ZTF vs the PS1 limit"]
        direction TB
        Z4["<b>4 · Confident</b><br/>bright-star mask · sharpness (point sources)<br/>no object within 3″ · outside brighter objects' Kron apertures"]
        Z5["<b>5 · Population</b><br/>no Gaia proper motion or parallax at 5σ"]
        Z7["<b>7 · Interesting</b><br/>σ against the PS1 limit, ZTF brighter only<br/>|σ| > 3 in ≥ 2 bands"]
        Z8["in_ztf candidates"]
        Z4 --> Z5 --> Z7 --> Z8
    end
    subgraph PS1 ["in_pstarr — PS1 vs the ZTF limit"]
        direction TB
        P4["<b>4 · Confident</b><br/>bright-star mask · sharpness (point sources)<br/>no object within 3″ · outside brighter objects' Kron apertures<br/>PS1 nStackDetections ≥ 2, not BAD_STACK, not SUSPECT_STACK"]
        P5["<b>5 · Population</b><br/>no Gaia proper motion or parallax at 5σ"]
        P7["<b>7 · Interesting</b><br/>σ against the ZTF limit, PS1 brighter only<br/>|σ| > 3 in ≥ 2 bands"]
        P8["in_pstarr candidates"]
        P4 --> P5 --> P7 --> P8
    end
    S -- two-sided --> B4
    S -- ZTF vs limit --> Z4
    S -- PS1 vs limit --> P4

    CAL["<b>6 · Calibrate</b> — once, from what survives 4 and 5 in every track<br/>each survey's 90 % recovery limit per quadrant, PSF and Kron<br/>the Δmag envelope from the two-sided bands"]
    B5 -.-> CAL
    Z5 -.-> CAL
    P5 -.-> CAL
    CAL -.-> B7
    CAL -.-> Z7
    CAL -.-> P7
```

Every band of every object ends step 3 with one of these on each survey's side:

| status | meaning |
|---|---|
| X | a quality check failed: this survey's side of the band is unusable, so the band compares nothing |
| D | measured at S/N ≥ 5 |
| L | measured below S/N 5: used if brighter than the survey's limit, else the limit stands in |
| U | nothing measured: the survey's completeness limit stands in |

The quality checks (step 1), each answering "is the data at this position sound?" and nothing about the object:

| ZTF side | PS1 side |
|---|---|
| the reference image covers the position | not one of MAST's mis-registered polar skycells (with or without a PS1 object) |
| DAOPhot flags = 0 | stack nFrames ≥ 4 |
| > 5 px from a masked pixel (at the detection, or at the PS1 position for a limit) | residual sky ≤ 1e-7 |
| extended objects: Kron aperture ≤ 10 % masked | not DEFECT, PEAK_ON_SPIKE, ON_SPIKE |
| | measured in the band (a PS1 object with no row for the band tells us nothing) |
| | psfQfPerfect > 0.85 |

Code: `Source_Analysis/filter_field.py` (`usable_sides`, `detection_status`, `COMPARISON`, `object_catalog`, `real_checks`, `completeness_limits`, `kde_envelopes`,
`judge_bands`); every object's record of what happened to it is in `objects.parquet` (`{b}_dropped_by`, `outcome`; `python -m Source_Analysis.filter_field explain FIELD RA DEC`).
Walkthrough with field 860: `Notebooks/pipeline_review_860.ipynb`.

## Workflow Overview
### 1. Extract
This step extracts sources from ZTF deep stack reference images and the Pan-STARRS source catalog, and stores them.
- Run `run_extraction.py` with the option of the bash script `run_extraction.sh`.
- Ensure that directories populate into the `path/to/Data/catalog_results/field_results` directory with names like `<field_id>_<ccd_id>_<q_id>` (e.g. `000245_15_4`) that contain the following files:
    - EPSFs/
        - ZTF_g_EPSF.npy
        - ZTF_r_EPSF.npy
        - ZTF_i_EPSF.npy
    - nan_masks/
        - ZTF_g_nan_mask.npy
        - ZTF_r_nan_mask.npy
        - ZTF_i_nan_mask.npy 
    - WCSs/
        - ZTF_g_wcs.pkl
        - ZTF_r_wcs.pkl
        - ZTF_i_wcs.pkl
    - PSTARR.ecsv
    - ZTF_g.ecsv  
    - ZTF_r.ecsv
    - ZTF_i.ecsv


### 2. Match sources between ZTF and Pan-STARRS
This step will associate the sources stored in the ZTF and Pan-STARRS `.ecsv` files that were stored in step 1.
- Run `cross_match.py -aq -mf` with the option of the bash script `cross_match.sh`.
    - If you don't want to associate quadrants and merge fields, consider removing the kwargs.
- Ensure that files populate into the `path/to/Data/catalog_results/field_results` directory with names like `<field_id>_<band>.ecsv`. Note that you'll also find files like `<band>_associated.ecsv` in the `<field_id>_<ccd_id>_<q_id>` directories created in step 1.
It is important to realize that sources may have non-detections in Pan-STARRS or ZTF!

### 3. Filter sources
This step takes the sources and passes them through filtration pipelines. You'll just need to look at the `filter_fields.py` file.
- Run `filter_fields.py` with the option of the bash script `filter_fields.sh`.
- Ensure that directories labeled by fieldid (e.g. `000313`) populate into the `filter_results/` directory with files `0.ecsv`, `1.ecsv`, and `2.ecsv` corresponding to in_both, in_ztf, and in_pstarr.

### 4. Combine filtration results into one table
This step takes the filtration results and combines them into a single table.
- Run `combined_filtered_tabs.py` with the option of the bash script `combined_filtered_tabs.sh`.
- Ensure that a new folder `combined/` appears in the `filter_results/` directory with files `0.ecsv`, `1.ecsv`, and `2.ecsv` and the corresponding flowcharts and stats. These files have all the candidates for each of the three catalogs!

### 5. Make (and sort) analysis pages
Make analysis pages for each source by running `store_src_plots.py` which does something like:
```
import matplotlib.pyplot as plt
from Source_Analysis.Sources import Sources

coords = Sources.from_file('path/to/file.ecsv')
for c in coords:
    c.plot_everything()
    plt.show()
```
