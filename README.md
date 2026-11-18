# long_transients

## The filtering pipeline at a glance

```
ZTF deep-reference DAOPhot catalogs (g, r, i)        PS1 DR2 stack catalog (grizy, one row per object)
   + our Kron photometry on the reference images        + MAST's bad-skycell list
            │                                                   │
            └──────────── 1″ cross-match ───────────────────────┘
                                 │
                   one row per object: ZTF object (+ PS1 match) or PS1-only object
                                 │
                 ┌───────────────▼────────────────┐
                 │  SHARED STAGE (every object)    │
                 │  per band, per survey:          │
                 │   characterize  → D / L / U     │  measurement at S/N ≥ 5 / below / none
                 │   quality       → X             │  coverage, masks, flags, bad stacks, bad skycells
                 │   compare?      → kind          │  two-sided · vs PS1 limit · vs ZTF limit · nothing
                 └───────────────┬────────────────┘
                                 │  ≥ 2 comparable bands of one kind?
            ┌────────────────────┼────────────────────┐───────────────► not comparable (exit, with reason)
            ▼                    ▼                    ▼
         in_both              in_ztf              in_pstarr
      (two-sided)       (ZTF vs PS1 limit)    (PS1 vs ZTF limit)
            │                    │                    │
            ├─ confident: real object? (star mask, sharpness; isolation & Kron containment for one-sided; nStack ≥ 2)
            ├─ population: not Galactic (Gaia proper motion, parallax)
            │
            ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ CALIBRATE on what is left                                      │
   │  recovery limits: 90% recovery of the other survey's objects   │  per quadrant, PSF & Kron
   │  envelope: KDE of ZTF − PS1 vs mag from the two-sided bands    │  clipped
   └────────────────────────────┬───────────────────────────────────┘
                                ▼
            σ per band  (limit bands clipped to the direction the limit supports)
                                │
            ├─ interesting: |σ| > 3 in ≥ 2 bands, ≥ 2 of the same sign
            ├─ best: every measured side at S/N ≥ 10, rule re-checked
                                ▼
                 candidates  (in_both / in_ztf / in_pstarr)
```

Five ideas, top to bottom: **match** → **characterize & sort** (the shared stage: what each band is, whether it's
trustworthy, what it can compare; the catalog is just the kind of comparison) → **clean** the catalogs (real, not
Galactic) → **calibrate** on the survivors (limits, envelope) → **select** (σ, the ≥ 2-band rules, S/N ≥ 10).
Code: `Source_Analysis/filter_field.py`; walkthrough with field 860: `Notebooks/pipeline_review_860.ipynb`.

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
