# five-rivers-heavy-metals

Reproducibility package for:

> **Geochemistry Source Attribution and Satellite-Prediction Boundaries for Heavy Metals in Dhaka's Five Rivers.**
> Hasan, Rahman, Zubyer, Rupa, Arabi, and Jolly. Submitted to *Science of the Total Environment*, 2026.

This repository contains the **code**, **small derived data**, and **rebuild scripts** for every figure and table in the manuscript and supplementary material. Large raw inputs (raw lab record, GIS shapefiles, trained model weights) are archived on Zenodo: **DOI to be inserted upon deposit**.

---

## Quickstart — reproduce the figures (no model training)

```bash
git clone https://github.com/rakibhhridoy/five-rivers-heavy-metals.git
cd five-rivers-heavy-metals
python -m venv .venv && source .venv/bin/activate    # optional but recommended
pip install -r requirements.txt
python run_all.py figures
```

This regenerates every chart-based figure under `figures/` from the cached intermediate results in `results/`. Expected runtime: **under 30 seconds** on a laptop.

## Full pipeline rerun (model training)

```bash
python run_all.py pipeline      # ~30–60 minutes on a CPU, faster with GPU
python run_all.py figures
```

The pipeline phases consume the primary-station chemistry + AlphaEarth features in `data/sedimentRO_v2.csv` / `data/sedimentWO_v2.csv`, the harmonized 111-row literature dataset in `data/secondary/`, and the climate covariates in `data/climate_covariates_v2.csv`. They emit the intermediate CSVs the figure scripts then read.

## Repository layout

```
five-rivers-heavy-metals/
├── README.md                  this file
├── LICENSE                    MIT (code), CC BY 4.0 (data)
├── CITATION.cff               machine-readable citation
├── requirements.txt           pinned Python deps
├── run_all.py                 master reproducibility script
├── src/
│   ├── pipeline/              38 scripts: 30 phase (phase0 → phase11) + 8 extra
│   │                          (Kd, PCA, indices, Mann-Kendall, Monte Carlo,
│   │                           health risk, 5 DL architectures, LIME)
│   ├── figures/               11 figure regeneration scripts
│   └── gis/                   spectral-index calculation + GIS workflow docs
├── results/                   40 intermediate CSVs (small derived artifacts)
├── figures/                   14 published PNGs + regenerated outputs
├── data/                      input data (primary + harmonized secondary)
│   └── secondary/             harmonized 111-row literature dataset + audit
├── docs/
│   ├── FIGURE_TABLE_MAP.md    figure/table → script → data chain
│   ├── AUDIT_COVERAGE.md      manuscript ↔ package coverage matrix
│   ├── DATA_AVAILABILITY.md   pointer to the Zenodo deposit + provenance
│   └── KNOWN_ISSUES.md        candid notes (Mn exclusion, gap deep-dive, env)
└── zenodo_manifest/
    └── MANIFEST.md            what goes on Zenodo, with checksums (to be filled at deposit time)
```

## What this package *does not* include — by design

- **Synthetic-AlphaEarth artifacts.** An early version of `data/sedimentRO.csv` / `sedimentWO.csv` carried `np.random.rand`-generated placeholders in the `AE_00..AE_63` columns (Earth Engine access was unavailable when those CSVs were first created). The corrected files — with real Google AlphaEarth V1 embeddings extracted by `src/pipeline/phase4c_reextract_primary_ae.py` — are shipped here as `data/sedimentRO_v2.csv` and `data/sedimentWO_v2.csv`. The synthetic versions are intentionally excluded.
- **Sediment Mn.** The primary 17-station campaign measured Cr, Ni, Cu, As, Cd, Pb, and Fe by EDXRF. Sediment Mn was not measured; a column carried over from an earlier worksheet contained moisture %, not Mn. The manuscript text was purged of Mn claims before submission; this package mirrors that decision and drops Mn from the pipeline and figures end-to-end. See `docs/KNOWN_ISSUES.md` for the full record.
- **Third-party publication PDFs.** The PDFs of cited papers are not redistributed. The harmonized data extracted from them sits in `data/secondary/harmonized_sediment.csv` and `data/secondary/harmonized_water.csv` with per-paper provenance preserved.

## Citing this package

If you use the code or data, please cite **both** the paper (DOI on acceptance) and this package (`CITATION.cff` resolves to a Zenodo DOI once the deposit is published). The recommended attribution is:

> Hasan, M. R., et al. (2026). *five-rivers-heavy-metals: Reproducibility package for "Geochemistry Source Attribution and Satellite-Prediction Boundaries for Heavy Metals in Dhaka's Five Rivers"*. Zenodo. https://doi.org/10.xxxx/xxxxxx

## Contact

Md Rakib Hasan — `rakibhridoy63@gmail.com` (lead author)
Yeasmin Nahar Jolly — Atomic Energy Centre, Dhaka (corresponding author)
