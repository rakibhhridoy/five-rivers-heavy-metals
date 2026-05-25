# Data availability

## What ships with this repository

Everything needed to rebuild the manuscript's figures and tables from the cached intermediate results, and to re-run the analysis pipeline end-to-end, lives in this repository:

| Path | Contents | Size |
|------|----------|------|
| `data/sedimentRO_v2.csv`, `data/sedimentWO_v2.csv` | Primary 17-station sediment chemistry (Cr, Ni, Cu, As, Cd, Pb, Fe) + 64 AlphaEarth V1 features + proximity (brick kiln / industry distance & count) | ~240 KB each |
| `data/waterR*.csv`, `data/waterW*.csv` | Primary 17-station water chemistry (rainy + winter) | <130 KB each |
| `data/climate_covariates_v2.csv` | ERA5-derived covariates aligned to the 17 stations | <130 KB |
| `data/secondary/MAIN DATA.csv` | 111-row literature dataset (heavy-metal concentrations in the five rivers, compiled from published papers) | ~28 KB |
| `data/secondary/harmonized_{sediment,water}.csv` | Audited, unit-normalized long-format version of the above | ~50 KB each |
| `data/secondary/stations_geocoded.csv` | Geocoded secondary stations (lat/lon) | <30 KB |
| `data/secondary/secondary_{proximity,ae}_features.csv` | Proximity + AlphaEarth features for the 56 geocoded literature stations | ~80 KB combined |
| `data/Sediments Data.docx`, `data/Shitalakshya.xlsx` | Raw lab record (read-only reference) | ~1 MB |
| `results/*.csv` | 26 intermediate analysis tables — the immediate inputs to every figure regenerator | <2 MB total |

The complete repository is ~40 MB.

## What is on Zenodo

The Zenodo deposit (https://doi.org/10.5281/zenodo.20378184) provides the **complete data archive** in a single immutable snapshot, including items too large to host on GitHub:

- A mirror of this repository's `data/` (so the Zenodo record stands alone, with no GitHub dependency)
- `gis/` — five-river basin boundaries, brick-kiln point shapefiles, industrial site shapefiles, study-area background layers (~3.4 GB)
- `models/` — trained Keras model weights for the MultiTask deep-learning model used in `phase8_unified_loocv.py` (if you do not want to retrain)
- `analysis_results_full/` — the full 43 MB tree of exploratory intermediate analyses (the repository only ships the curated subset used by the manuscript)

The `zenodo_manifest/MANIFEST.md` in this repository lists the exact files in the Zenodo deposit with SHA-256 checksums (filled in at deposit time).

## Licensing

- Code in `src/`, `run_all.py`, scripts in `src/figures/regenerate_*.py`: **MIT** (see `LICENSE`).
- Data in `data/`: **Creative Commons Attribution 4.0 International** (CC BY 4.0).
- The harmonized secondary dataset in `data/secondary/` is a derived work compiled from published papers; the **values** are sourced from the original publications cited in `data/secondary/MAIN DATA.csv` (column `Paper_ID` keys into the manuscript's bibliography). Please cite the original publications when using individual measurements; cite this package when using the harmonized compilation.

## Re-extracting AlphaEarth from scratch

`data/sedimentRO_v2.csv` and `data/sedimentWO_v2.csv` ship with the AlphaEarth V1 embeddings pre-extracted. If you want to verify the extraction or extend it to other stations:

1. Install the Earth Engine client: `pip install earthengine-api`
2. Authenticate: `earthengine authenticate`
3. Run `python src/pipeline/phase4c_reextract_primary_ae.py`

The script documents the image (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`), buffer radius, and temporal aggregation choices. Outputs are byte-identical to the shipped files modulo Earth Engine internal versioning.
