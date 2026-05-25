# Zenodo deposit — manifest

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

1. Update `CITATION.cff` in the GitHub repo with the Zenodo DOI.
2. Update `README.md` "DOI to be inserted upon deposit" with the real DOI.
3. Tag the GitHub repo with a release matching the Zenodo version (e.g. `v1.0.0`).
4. If GitHub-Zenodo integration is enabled, the next GitHub release will be archived automatically — a useful arrangement that lets *future* edits to the repo each generate a new Zenodo version that cites back to this one.
