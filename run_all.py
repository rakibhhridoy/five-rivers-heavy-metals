#!/usr/bin/env python3
"""Master reproducibility script for five-rivers-heavy-metals.

Two entry points are supported:

  python run_all.py figures      # rebuild figures from cached results/*.csv
                                 # (fast: no model training, no GEE access required)

  python run_all.py pipeline     # rerun the full analysis pipeline end-to-end
                                 # (slow: trains ML/DL models from scratch;
                                 #  data/sedimentRO_v2.csv and data/sedimentWO_v2.csv
                                 #  already contain pre-extracted AlphaEarth features)

  python run_all.py all          # pipeline followed by figures

Each subcommand prints the script it is about to invoke before running it,
so partial failure is easy to diagnose.

See docs/FIGURE_TABLE_MAP.md for the figure/table -> script -> data chain.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "src" / "pipeline"
FIGURES = ROOT / "src" / "figures"

# Run order matters: each phase consumes outputs of earlier phases.
# See docs/FIGURE_TABLE_MAP.md.
PIPELINE_ORDER = [
    "phase0_harmonize.py",
    "phase1_distribution_check.py",
    "phase1_2_harmonized_indices.py",
    "phase2_temporal_trends.py",
    "phase3_geocode.py",
    "phase4a_proximity_features.py",
    # phase4b/4c extract AlphaEarth from Google Earth Engine.
    # Skipped by default — outputs already shipped as data/sedimentRO_v2.csv,
    # data/sedimentWO_v2.csv, data/secondary/secondary_ae_features.csv.
    # Uncomment to re-extract (requires GEE auth + the `earthengine-api` package).
    # "phase4b_extract_ae.py",
    # "phase4c_reextract_primary_ae.py",
    "phase4d_external_validation.py",
    "phase4e_v2_baseline.py",
    "phase4f_full_grid.py",
    "phase4f_loocv_only.py",
    "phase4f_loocv_predictions.py",
    "phase4f_cd_bootstrap.py",
    "phase4g_ablation.py",
    "phase5_augmented_retrain.py",
    "phase6_1_mc_pooled.py",
    "phase7_wilcoxon_seasonal.py",
    "phase8_build_pooled.py",
    "phase8_unified_loocv.py",
    "phase8_xgb_rf_pooled.py",
    "phase8_compile_comparison.py",
    "phase9_climate_covariates.py",
    "phase9_loocv_with_climate.py",
    "phase9_pb_bootstrap_svr.py",
    "phase9_winner_predictions.py",
    "phase10_mk_pooled.py",
    "phase11_kernel_check.py",
    "phase11_pmf_nmf.py",
    "phase11_quantile_mapping.py",
    # Extra phases added during reproducibility audit (see docs/AUDIT_COVERAGE.md).
    # These cover analyses described in the manuscript but not previously scripted:
    # K_d, primary contamination indices (Tab 1), PCA, per-river MK,
    # primary-only Monte Carlo headline, and HQ/ILCR health risk.
    "extra_kd_compute.py",
    "extra_primary_indices.py",
    "extra_pca_source_id.py",
    "extra_per_river_mk.py",
    "extra_mc_primary_headline.py",
    "extra_health_risk.py",
    # DL-dependent (need a working TensorFlow — see requirements.txt env note):
    "extra_lime_explanations.py",   # Fig 9; trains 3 CNN-GNN-MLP models
]

# extra_dl_architectures.py is an importable library (used by extra_lime_explanations.py
# and available for phase4e/4f/5/4g), not a standalone pipeline step — not run directly.
# src/gis/calculate_spectral_indices.py is a GIS utility requiring the Zenodo raster
# archive; see src/gis/README_GIS.md. Neither is part of the default run.

# `regenerate_all_themed.py` consolidates Figs 3, 5, 7, 8. The per-figure
# scripts (`regenerate_fig*.py`) are kept for users who want a single figure.
FIGURE_ORDER = [
    "regenerate_all_themed.py",
    "regenerate_fig_kd_partition.py",
    "regenerate_fig2_heatmap.py",
    "regenerate_fig6_ri.py",
]


def _run(script_path: Path) -> None:
    print(f"\n>>> {script_path.relative_to(ROOT)}", flush=True)
    res = subprocess.run([sys.executable, str(script_path)], cwd=ROOT)
    if res.returncode != 0:
        sys.exit(f"!! {script_path.name} failed with exit code {res.returncode}")


def cmd_pipeline() -> None:
    for name in PIPELINE_ORDER:
        _run(PIPELINE / name)


def cmd_figures() -> None:
    for name in FIGURE_ORDER:
        _run(FIGURES / name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["figures", "pipeline", "all"])
    args = ap.parse_args()
    if args.step in ("pipeline", "all"):
        cmd_pipeline()
    if args.step in ("figures", "all"):
        cmd_figures()


if __name__ == "__main__":
    main()
