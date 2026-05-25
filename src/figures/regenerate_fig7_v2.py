"""Regenerate Fig 7 (LOOCV scatter, predicted vs observed) under v2.
Shows the four metal-season cases with positive LOOCV R^2.
Source: analysis_results/enrichment/s4_v2_loocv_preds.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "SubmissionV2/LaTeX_Source/figures/Fig7_scatter_loocv.png"

df = pd.read_csv(ROOT / "results/s4_v2_loocv_preds.csv")

panels = [
    ("Cd", "Rainy"),  ("Cd", "Winter"),
    ("Cu", "Rainy"),  ("Ni", "Rainy"),
]

river_color = {
    "Buriganga": "#d7191c", "Shitalakshya": "#fdae61", "Turag": "#2c7bb6",
    "Dhaleshwari": "#abd9e9", "Balu": "#1a9641",
}

fig, axes = plt.subplots(2, 2, figsize=(9, 8))
axes = axes.flatten()

for ax, (metal, season) in zip(axes, panels):
    sub = df[(df["Metal"] == metal) & (df["Season"] == season)]
    obs = sub["Observed"].values
    pred = sub["Predicted"].values
    r2 = sub["LOOCV_R2"].iloc[0]
    best = sub["Best_ML"].iloc[0]
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))

    lo = min(obs.min(), pred.min())
    hi = max(obs.max(), pred.max())
    pad = 0.08 * (hi - lo) if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=0.8, label="1:1")

    for river, color in river_color.items():
        m = sub["River"].values == river
        if m.any():
            ax.scatter(obs[m], pred[m], s=55, c=color, edgecolor="black",
                       linewidth=0.6, label=river, zorder=3)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Observed {metal} (mg/kg)")
    ax.set_ylabel(f"Predicted {metal} (mg/kg)")
    ax.set_title(f"{metal} -- {season}    "
                 rf"$R^2 = {r2:+.3f}$, RMSE = {rmse:.2f}, {best}",
                 fontsize=10)
    ax.grid(linestyle=":", alpha=0.5)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6,
           bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)

plt.suptitle("LOOCV predicted vs observed (n=17 field stations) "
             "under the corrected GEE V1 AlphaEarth pipeline",
             fontsize=11, y=1.0)
plt.tight_layout(rect=[0, 0.04, 1, 0.98])
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT}")
