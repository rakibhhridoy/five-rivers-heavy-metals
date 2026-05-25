"""Regenerate Fig 2 — full model × metal × season performance heatmap (Supplementary S4.5).

5-fold CV R² for all evaluated models × 7 metals × 2 seasons under the corrected
GEE V1 AlphaEarth feature pipeline. DL architectures above the divider; ML below.

Input
-----
results/s8_model_comparison.csv  — long-format rows: Metal, Season, Model, R2 (or similar)

Output
------
figures/Fig2_performance_heatmap.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "figures" / "Fig2_performance_heatmap.png"

# Manuscript-ordered metals (Mn-purged)
METAL_ORDER = ["Cd", "Cu", "Pb", "Ni", "Fe", "As", "Cr"]
# DL above the divider, ML below.
DL_MODELS = ["Transformer CNN", "Dual Attention", "CNN-GNN-MLP",
             "GNN-MLP-Autoencoder", "Mixture of Experts"]
ML_MODELS = ["RandomForest", "XGBoost", "SVR"]


def main() -> None:
    df = pd.read_csv(RES / "s8_model_comparison.csv")
    # Detect schema: locate the R² column
    r2_col = next((c for c in df.columns if c.lower() in ("r2", "r²", "loocv_r2",
                                                          "cv5_r2", "r2_mean")), None)
    if r2_col is None:
        # Wide format: columns like "RandomForest", "XGBoost", "SVR_pooled_log", ...
        # Long-melt the model columns.
        id_cols = [c for c in df.columns if c in ("Metal", "Season")]
        value_cols = [c for c in df.columns if c not in id_cols]
        df = df.melt(id_vars=id_cols, value_vars=value_cols,
                     var_name="Model", value_name="R2")
        r2_col = "R2"

    # Build the pivot: rows = model, cols = (metal, season)
    df["MetalSeason"] = df["Metal"] + " — " + df["Season"]
    pivot = df.pivot_table(index="Model", columns="MetalSeason", values=r2_col,
                           aggfunc="first")

    # Order metals × seasons (rainy first, then winter)
    col_order = [f"{m} — Rainy" for m in METAL_ORDER] + [f"{m} — Winter" for m in METAL_ORDER]
    col_order = [c for c in col_order if c in pivot.columns]
    pivot = pivot.reindex(columns=col_order)

    # Order rows: DL first, then ML, then anything else (e.g. pooled variants)
    known = [m for m in DL_MODELS if m in pivot.index] + \
            [m for m in ML_MODELS if m in pivot.index]
    other = [m for m in pivot.index if m not in known]
    pivot = pivot.reindex(known + other)

    fig, ax = plt.subplots(figsize=(max(8, len(col_order) * 0.55),
                                    max(4, len(pivot.index) * 0.45)))
    arr = pivot.values.astype(float)
    arr_masked = np.ma.masked_invalid(arr)
    im = ax.imshow(arr_masked, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(v) < 0.6 else "white")

    ax.set_xticks(range(arr.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(arr.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    # Divider between DL and ML rows
    n_dl = sum(1 for m in pivot.index if m in DL_MODELS)
    if 0 < n_dl < len(pivot.index):
        ax.axhline(n_dl - 0.5, color="black", linewidth=1.2)
    # Divider between Rainy and Winter columns
    n_rainy = sum(1 for c in pivot.columns if "Rainy" in c)
    if 0 < n_rainy < len(pivot.columns):
        ax.axvline(n_rainy - 0.5, color="black", linewidth=1.2)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("5-fold CV R² (n=117 IDW-augmented)")
    ax.set_title("Model × Metal × Season performance heatmap (Mn excluded — see KNOWN_ISSUES.md)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
