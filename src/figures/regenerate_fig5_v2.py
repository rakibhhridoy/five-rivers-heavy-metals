"""Regenerate Fig 5 (feature importance / ablation) under v2 (real GEE V1 AE).
Story: proximity-only vs AE-only vs combined 5-fold CV R^2 per metal-season.
Source: analysis_results/enrichment/s4_v2_ablation.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "SubmissionV2/LaTeX_Source/figures/Fig5_feature_importance_rainy.png"
OUT_W = ROOT / "SubmissionV2/LaTeX_Source/figures/Fig5_feature_importance_winter.png"

df = pd.read_csv(ROOT / "results/s4_v2_ablation.csv")
metal_order = ["Cd", "Cu", "Pb", "Ni", "Fe", "As", "Cr"]


def plot_season(season, out_path):
    sub = df[df["Season"] == season].set_index("Metal").reindex(metal_order)
    x = np.arange(len(metal_order))
    w = 0.27

    fig, ax = plt.subplots(figsize=(10, 4.8))
    b1 = ax.bar(x - w, sub["Prox_only_R2"], w, label="Proximity only (4 features)",
                color="#4eb265", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x,     sub["AE_only_R2"],   w, label="AlphaEarth only (64 dims)",
                color="#cae0ab", edgecolor="black", linewidth=0.6)
    b3 = ax.bar(x + w, sub["Both_R2"],      w, label="Combined (68 features)",
                color="#1965b0", edgecolor="black", linewidth=0.6)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle="--",
               label=r"Useful-prediction threshold ($R^2 = 0.5$)")

    for bars in [b1, b2, b3]:
        for bar in bars:
            v = bar.get_height()
            offset = 0.025 if v >= 0 else -0.04
            ax.text(bar.get_x() + bar.get_width()/2, v + offset, f"{v:+.2f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metal_order)
    ax.set_ylabel(r"5-fold CV $R^2$ ($n=117$, per-metal best architecture)")
    ax.set_title(f"Same-model ablation under corrected GEE V1 AlphaEarth pipeline --- {season} season")
    ax.set_ylim(-0.35, 0.85)
    ax.legend(loc="upper right", fontsize=8, frameon=True, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


plot_season("Rainy", OUT)
plot_season("Winter", OUT_W)
