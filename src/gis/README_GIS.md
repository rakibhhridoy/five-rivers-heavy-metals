# GIS / land-use-change workflow (Methodology §5)

This directory documents the geospatial workflow behind the manuscript's land-use-change
and brick-kiln-inventory claims. The full pipeline depends on the **3.4 GB raster tree
archived on Zenodo** (`gis.zip`), not on the tabular data shipped in the GitHub repo. This
README records the methodology precisely so the workflow is auditable and re-runnable for
anyone who downloads the Zenodo raster archive.

## Manuscript claims this workflow produces

From Methodology §5 and Results §3.1:

- **+34.2% industrial footprint** (2017 → 2025)
- **Brick kilns 312 → 487** (+56%) active kilns
- **Random Forest LULC classification: OA = 87.3%, κ = 0.84**
- **11 spectral indices**: NDVI, EVI, SAVI, NDWI, MNDWI, AWEI, NDBI, UI, BUI, NDSI, LST
- The four hydrological-proximity features (`dist_brick`, `num_brick`, `dist_ind`,
  `num_ind`) that feed every prediction model — these are the *output* of this workflow,
  already merged into `data/sedimentRO_v2.csv` / `data/sedimentWO_v2.csv`.

## Data sources

| Layer | Source | Resolution | Notes |
|-------|--------|------------|-------|
| Surface reflectance | Sentinel-2 L2A (tile T45QZG) | 10 m / 20 m | bands B02/B03/B04/B08 (10 m), B11/B12 (20 m, resampled) |
| Thermal (LST) | Landsat 8/9 Collection 2 L2 | 30 m → 100 m | band B10; conversion constants from the scene MTL |
| Land-cover baseline | Impact Observatory / ESA 10 m LULC time series | 10 m | downloaded per-year from the Azure blob `lulctimeseries` (gis/GIS.ipynb cell 27) |
| Elevation (flow routing) | NASA SRTM 30 m DEM (`DEMF.tif`) | 30 m | D8 flow accumulation via pysheds / skimage route_through_array |
| Brick-kiln points | Manual + semi-automated digitisation over Sentinel-2 true-colour and NDBI | vector | validated against the LULC built-up class |
| Industrial sites | OSM landuse=industrial + manual digitisation | vector | |

## Workflow steps (and where the code lives)

1. **Spectral indices** — `calculate_spectral_indices.py` (in this directory; ported from
   `gis/GIS.ipynb` cell 32). Runnable today against any Sentinel-2 band directory:
   ```bash
   python calculate_spectral_indices.py --bands <sentinel_band_dir> --out <indices_dir>
   ```
   Produces 10 reflectance indices as GeoTIFFs; LST is computed separately via
   `compute_lst()` from a Landsat thermal band.

2. **LULC classification** — Random Forest classifier trained on labelled samples over the
   index stack + raw bands. Original code: `gis/GIS.ipynb` cells 35–42. Reported accuracy
   OA = 87.3%, κ = 0.84 on a held-out sample. *Not ported here* because it depends on the
   labelled training polygons and full raster stack in `gis.zip`.

3. **Change detection** — per-class area differencing between the 2017 and 2025 LULC rasters
   yields the +34.2% industrial-footprint figure. Original: `gis/GIS.ipynb` cells 36–38.

4. **Brick-kiln inventory** — kiln points digitised per year; the 312 (2017) → 487 (2025)
   count is the cardinality of the two point layers. Stored as shapefiles in `gis.zip`.

5. **Hydrological proximity features** — `src/pipeline/phase4a_proximity_features.py`
   (already in the runnable pipeline) consumes the kiln/industrial point layers + the SRTM
   DEM, performs D8 flow routing, and writes `dist_brick / num_brick / dist_ind / num_ind`
   into the primary `_v2.csv` files. This is the one GIS step already wired into `run_all.py`
   (it needs the shapefiles from `gis.zip` to re-run; outputs already shipped in the CSVs).

## What is and isn't reproducible without the Zenodo raster archive

| Step | Reproducible from GitHub repo alone? |
|------|--------------------------------------|
| Spectral-index formulas | ✅ code shipped; needs your own Sentinel-2 bands |
| LST formula | ✅ `compute_lst()` shipped; needs a Landsat thermal band |
| LULC RF classification (OA/κ) | ❌ needs `gis.zip` raster stack + training polygons |
| +34.2% / 312→487 change figures | ❌ needs the 2017 & 2025 LULC rasters in `gis.zip` |
| Proximity features (model inputs) | ⚠️ outputs already in the CSVs; re-running needs `gis.zip` shapefiles |

## Why this is documented rather than fully scripted

The LULC analysis is a multi-notebook GIS workflow (`gis/GIS.ipynb`, 63 cells) tightly
coupled to a 3.4 GB raster tree and external per-year LULC downloads. Bundling it as a
"run from the shipped CSVs" script would be misleading — it cannot run without the rasters.
The honest packaging is: ship the reusable, self-contained index code; document the rest
precisely; point to the raster archive on Zenodo and to the original notebook
(`gis/GIS.ipynb`, also included in `gis.zip`) for the classification and change-detection
steps.
