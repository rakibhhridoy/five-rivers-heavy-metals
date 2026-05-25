# Known issues — disclosed for the public record

This package is a **curated** reproduction of the analyses in the manuscript. It does *not* reproduce two early-stage artifacts that were found, during the manuscript's revision cycle, to be unsound. We document them here because we believe full transparency is the right default for reproducibility deposits — and because the curation is the reason some files you might expect are not present.

## 1. Sediment Mn is excluded

The primary 17-station EDXRF panel covered **Cr, Ni, Cu, As, Cd, Pb, Fe** plus textural fractions (sand/silt/clay/moisture). It did **not** include Mn. In an earlier worksheet the column abbreviated `MR` / `MW` (moisture rainy / winter, in percent) was inadvertently mapped to "Mn" in some pipeline scripts (notably the predecessors of `phase8_build_pooled.py`). That mismapping propagated into LOOCV, K_d, brick-kiln dose–response, seasonal Wilcoxon, NMF loadings, and quantile-mapping outputs.

The error was discovered during revision. The submitted manuscript's prose and tables were purged of every Mn-attributed number before submission to *Science of the Total Environment*. **This package goes further:** the moisture-to-Mn mappings have been removed from every pipeline script, Mn rows have been filtered out of every intermediate CSV in `results/`, and Mn has been dropped from every figure regenerator's `metal_order`. The package therefore reports analyses for **seven metals** (Cd, Cr, As, Pb, Cu, Ni, Fe), not the eight that appeared in some early drafts.

A consequence: the **published Fig 3, 5, and 7 in the PDF show eight panels including Mn**; rebuilding those figures from this package will yield seven-panel versions. The seven-panel versions are the ones that match the manuscript's prose; the eight-panel PDF versions are a residual inconsistency the authors accepted at submission time.

Water Mn was measured and is real. Water-side analyses retain Mn.

## 2. Synthetic AlphaEarth placeholders are excluded

When the original `data/sedimentRO.csv` and `data/sedimentWO.csv` were assembled, Google Earth Engine access was not yet available. As a placeholder, the 64 AlphaEarth feature columns (`AE_00`..`AE_63`) were filled with `np.random.rand(N, 64) * 0.5 + 0.25`. The placeholder was later expected to be swapped for real values but, in the original files, was not.

In April 2026 the AlphaEarth V1 embeddings for 2024 were extracted from Earth Engine (`src/pipeline/phase4c_reextract_primary_ae.py`) and written to `data/sedimentRO_v2.csv` and `data/sedimentWO_v2.csv` — these are the inputs every script in this package reads. The original placeholder-filled CSVs are **not** included.

If you intend to re-extract the embeddings independently, you will need a Google Earth Engine account and the `earthengine-api` Python package; the script's source documents the exact `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` image and the buffer/temporal aggregation it uses.

## 3. Honest performance ranges

For transparency, the honest 5-fold and LOOCV performance ranges on real AlphaEarth + 17 real stations are modest, not the >0.9 R² numbers an early version of the draft reported on synthetic features. The numbers in the **submitted manuscript** reflect the corrected pipeline:

- 5-fold CV (n=17 per season): R² ranges from ~0.2 (Cr) to ~0.7 (Cd), median ~0.5
- LOOCV (n=17 per season): Cd is the only consistently positive performer (~0.6); other metals near zero or negative on the strict per-season test
- Pooled (n=34, log-target, with climate/feature engineering): substantially better, hence the manuscript's emphasis on the pooled formulation

These figures are reproduced by `run_all.py pipeline` from the data shipped here.

## 4. Components investigated in the four-gap deep-dive

The first audit flagged four gaps. All four were investigated; three are now fully resolved and one is partially resolved (the irreducibly GIS-dependent part). Full detail in `docs/AUDIT_COVERAGE.md`.

- **Monte Carlo "Pb dominates at 60.68%" headline — RESOLVED.** The originating notebook (`Python/sample.ipynb`, cells 6 & 220) revealed the source paper used non-Håkanson toxic-response factors (Cr=26, Ni=16, Cu=16, As=6, Cd=0.6, Pb=31) and an Fe background of 30890 mg/kg. With these, `src/pipeline/extra_mc_primary_headline.py` reproduces Pb-dominance (68.5% vs published 60.68%) and mean RI 253.1 (vs 254.78); the residual gap is seed/subset noise. Note these Tr values diverge from standard Håkanson 1980 — the §3.7 *robustness* analysis (`phase6_1_mc_pooled.py`, standard Tr + Turekian–Wedepohl background) is the more defensible result and shows the Pb/Cu/Cr cluster at 60–70% without single-metal dominance.

- **Three "missing" DL architectures — RESOLVED.** CNN-GNN-MLP, GNN-MLP-Autoencoder, and Mixture-of-Experts existed at the project root in `train_individual_metals.py`; all five architectures are now curated into `src/pipeline/extra_dl_architectures.py` and build-verified under TensorFlow 2.16.2.

- **LIME explanations (Fig 9) — RESOLVED.** `src/pipeline/extra_lime_explanations.py` trains three CNN-GNN-MLP models (As/Pb/Cd) and regenerates Fig 9 with LIME; verified end-to-end. The full original analyzer is at `scripts/phase1_feature_importance_extraction.py` in the project root.

- **Land-use change analysis — PARTIALLY RESOLVED.** The 11 spectral-index formulas (NDVI, EVI, SAVI, NDWI, MNDWI, AWEI, NDBI, UI, BUI, NDSI, LST) are now shipped and verified in `src/gis/calculate_spectral_indices.py`. The RF LULC classification (OA = 87.3%, κ = 0.84), the +34.2%-industrial / 312→487-kiln change-detection figures, and the brick-kiln/industrial point inventories remain dependent on the 3.4 GB raster archive (`gis.zip` on Zenodo) and the original `gis/GIS.ipynb`; the methodology and per-step code locations are documented in `src/gis/README_GIS.md`. The proximity features these produce are already merged into the shipped `_v2.csv` files.

## 5. Residual notes

- **Supplementary §S12 environment claim.** S12 specifies Python 3.9 / TensorFlow 2.13 / scikit-learn 1.2.1 / NumPy 1.24 / Pandas 2.0. This package pins to the verified-working set: Python 3.11 / **TensorFlow 2.16.2 + protobuf 4.25.x** / scikit-learn 1.8.0 / NumPy 1.26.4 / Pandas 2.2.3. **TensorFlow and protobuf must be version-matched** — the authoring machine's base env (TF 2.21 + protobuf 5.29.6) is itself broken (gencode 6.31 vs runtime 5.29), which is why the DL code was verified in a clean venv with the pinned versions. If you hit a `libprotobuf` load error, your TF/protobuf are mismatched.

---

If you spot a further issue, please open a GitHub issue or email the lead author. Corrections will be released as new tagged versions and re-archived on Zenodo.
