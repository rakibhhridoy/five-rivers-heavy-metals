"""Fig 7 v3: LOOCV scatter with WINNING models for Cd, Pb (top performers).
Plus Cu Rainy and Ni Rainy as the next-best metal-seasons.

Cd Rainy/Winter: GP_pooled_log     (s9_winner_loocv_preds.csv)
Pb Rainy/Winter: GP_pooled_log_climate
Cu Rainy:        GP_pooled_log_climate
Ni Rainy:        XGBoost (per season)  -- from s4_v2_loocv_preds.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "SubmissionV2/LaTeX_Source/figures/Fig7_scatter_loocv.png"

df_winners = pd.read_csv(ROOT / "results/s9_winner_loocv_preds.csv")
df_v2 = pd.read_csv(ROOT / "results/s4_v2_loocv_preds.csv")

panels = [
    ("Cd", "Rainy",  df_winners, "GP pooled+log"),
    ("Cd", "Winter", df_winners, "GP pooled+log"),
    ("Pb", "Rainy",  df_winners, "GP pooled+log+climate"),
    ("Pb", "Winter", df_winners, "GP pooled+log+climate"),
    ("Cu", "Rainy",  df_winners, "GP pooled+log+climate"),
    ("Ni", "Rainy",  df_v2,      "XGBoost (per season)"),
]

river_color = {
    "Buriganga": "#d7191c", "Shitalakshya": "#fdae61", "Turag": "#2c7bb6",
    "Dhaleshwari": "#abd9e9", "Balu": "#1a9641",
}

fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5))
axes = axes.flatten()

for ax, (metal, season, source_df, model_label) in zip(axes, panels):
    sub = source_df[(source_df["Metal"] == metal) & (source_df["Season"] == season)]
    obs = sub["Observed"].values
    pred = sub["Predicted"].values
    if len(obs) == 0:
        ax.text(0.5, 0.5, "No predictions found",
                transform=ax.transAxes, ha="center")
        continue
    r2 = sub["LOOCV_R2"].iloc[0]
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))

    lo = float(min(obs.min(), pred.min()))
    hi = float(max(obs.max(), pred.max()))
    pad = 0.08 * (hi - lo) if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--",
            linewidth=0.8, label="1:1")

    if "River" in sub.columns:
        for river, color in river_color.items():
            m = sub["River"].values == river
            if m.any():
                ax.scatter(obs[m], pred[m], s=55, c=color, edgecolor="black",
                           linewidth=0.6, label=river, zorder=3)
    else:
        ax.scatter(obs, pred, s=55, c="#2c7bb6", edgecolor="black",
                   linewidth=0.6, zorder=3)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Observed {metal} (mg/kg)")
    ax.set_ylabel(f"Predicted {metal} (mg/kg)")
    ax.set_title(f"{metal} -- {season}    "
                 rf"$R^2 = {r2:+.3f}$, RMSE = {rmse:.2f}" + "\n"
                 + model_label,
                 fontsize=9)
    ax.grid(linestyle=":", alpha=0.5)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6,
           bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)

plt.suptitle("LOOCV predicted vs observed under per-metal-season winning model "
             "(n=17 field stations, corrected GEE V1 AlphaEarth pipeline)",
             fontsize=11, y=1.0)
plt.tight_layout(rect=[0, 0.04, 1, 0.97])
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT}")
