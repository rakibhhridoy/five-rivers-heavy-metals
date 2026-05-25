# Zenodo deposit — manifest

> **DEPOSITED 2026-05-25.** The archive is live at **https://doi.org/10.5281/zenodo.20378184**.
> A single combined archive `five-rivers-heavy-metals-data.zip` was uploaded (1.82 GB,
> 508 files: `README.txt` + curated `data/` + full `gis/` tree, junk excluded).
> SHA-256: `69fc9eed844c15cad7307d9731f39dc081ee36baaf5c1ac9d12c6dce8e31c2d5`.
> The per-directory split below was the original plan; the single-archive form was used instead.

Items to upload to the Zenodo record paired with this GitHub repository. Build the deposit by archiving the items listed here into a single `.zip` (or one `.zip` per top-level directory if the total exceeds Zenodo's per-file limit).

## Suggested Zenodo metadata

- **Title:** *five-rivers-heavy-metals — data archive for "Geochemistry Source Attribution and Satellite-Prediction Boundaries for Heavy Metals in Dhaka's Five Rivers"*
- **Authors:** Md Rakib Hasan; Rahman; Zubyer; Rupa; Arabi; Yeasmin Nahar Jolly
- **Upload type:** Dataset
- **Description (paste):** see `docs/DATA_AVAILABILITY.md` in the GitHub repo
- **License:** CC BY 4.0
- **Related identifiers:**
  - *cites* the manuscript DOI (on acceptance)
  - *is supplement to* GitHub repository URL
- **Keywords:** heavy metals; sediment; source apportionment; AlphaEarth; remote sensing; Bangladesh; Dhaka; reproducibility

## Items to include

### 1. Repository data mirror (so the record stands alone)
Source: `data/` directory of the GitHub repo.
Recommended archive: `data.zip` (~5 MB)

### 2. GIS layers — too large for GitHub
Source: `/Volumes/SSD Rx/Research/Five_Rivers/gis/`
Recommended archive: `gis.zip` (~3.4 GB)
Includes: five-river basin polygons, brick-kiln point shapefiles, industrial-site shapefiles, OSM background layers used to render Fig 1 (study area).

### 3. Raw lab record
Source: `data/Sediments Data.docx` (also in repo)
Optionally include the original spreadsheet exports under `data/` (e.g. `Shitalakshya.xlsx`, `data.csv`).

### 4. Trained model weights (optional)
Source: any `*.keras` files produced by `phase8_unified_loocv.py` and other model-training phases.
Without these, running `python run_all.py pipeline` will retrain from scratch on a CPU in 30–60 minutes; with these, figures rebuild in seconds.

### 5. Full intermediate results tree (optional)
Source: `/Volumes/SSD Rx/Research/Five_Rivers/analysis_results/` (~43 MB; the repo ships only the curated subset used by the manuscript).

## Checksums (fill at deposit time)

After bundling each archive, run:

```bash
shasum -a 256 data.zip gis.zip models.zip > zenodo_checksums.txt
```

and replace the table below.

| File | SHA-256 | Size |
|------|---------|------|
| `data.zip` | _to be filled_ | _to be filled_ |
| `gis.zip` | _to be filled_ | _to be filled_ |
| `models.zip` | _to be filled_ | _to be filled_ |
| `analysis_results_full.zip` | _to be filled_ | _to be filled_ |

## After publication

1. ✅ DONE — `CITATION.cff` updated with the Zenodo DOI (top-level `doi` + `identifiers`).
2. ✅ DONE — `README.md` and `docs/DATA_AVAILABILITY.md` updated with https://doi.org/10.5281/zenodo.20378184.
3. TODO — tag the GitHub repo with a release matching the Zenodo version (e.g. `v1.0.0`).
4. TODO (optional) — enable GitHub-Zenodo integration so future GitHub releases auto-archive as new Zenodo versions citing back to this one.
