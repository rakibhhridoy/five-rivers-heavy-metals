"""Fig 8 v3: Spatial prediction maps for Cd and Pb (the R^2 >= 0.5 metals).
Source: s9_winner_spatial_preds.csv (117 stations x metal x season).
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "SubmissionV2/LaTeX_Source/figures/Fig8_spatial_cd_pb.png"

df = pd.read_csv(ROOT / "results/s9_winner_spatial_preds.csv")

bootstrap = pd.read_csv(ROOT / "results/s4_v2_cd_bootstrap.csv")
pb_boot = pd.read_csv(ROOT / "results/s9_pb_bootstrap.csv")
boot_lookup = {}
for _, r in bootstrap.iterrows():
    boot_lookup[(r.Metal, r.Season)] = (r.Point_LOOCV_R2, r.CI95_low, r.CI95_high)
for _, r in pb_boot.iterrows():
    boot_lookup[(r.Metal, r.Season)] = (r.Point_LOOCV_R2_SVR, r.CI95_low, r.CI95_high)

# Use GP point estimates from s8_winners
winners = pd.read_csv(ROOT / "results/s8_winners.csv")
winner_r2 = {(r.Metal, r.Season): r.Winner_R2 for _, r in winners.iterrows()}

river_color = {
    "Buriganga": "#d7191c", "Shitalakshya": "#fdae61", "Turag": "#2c7bb6",
    "Dhaleshwari": "#abd9e9", "Balu": "#1a9641",
}

panels = [
    ("Cd", "Rainy"),  ("Cd", "Winter"),
    ("Pb", "Rainy"),  ("Pb", "Winter"),
]

fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[10, 10, 1], hspace=0.42, wspace=0.22)
ax_panels = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
ax_legend = fig.add_subplot(gs[2, :]); ax_legend.axis("off")

cmaps = {"Cd": plt.cm.YlOrRd, "Pb": plt.cm.Purples}

for ax, (metal, season) in zip(ax_panels, panels):
    sub = df[(df["Metal"] == metal) & (df["Season"] == season)]
    pred = sub["Predicted"].values
    vmax = float(np.percentile(pred, 95))
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = cmaps[metal]

    sc = ax.scatter(sub["Long"], sub["Lat"], c=pred, s=55, cmap=cmap,
                    norm=norm, edgecolor="black", linewidth=0.4,
                    alpha=0.95, zorder=2)

    real_mask = sub["is_real_station"] == 1
    real = sub[real_mask]
    for _, r in real.iterrows():
        ax.scatter(r["Long"], r["Lat"], facecolor="none", edgecolor="black",
                   linewidth=1.6, s=180, zorder=3)
    for river, color in river_color.items():
        m = real["River"].values == river
        if m.any():
            ax.scatter(real["Long"].values[m], real["Lat"].values[m],
                       c=color, s=22, edgecolor="black", linewidth=0.4,
                       zorder=4)

    boot = boot_lookup.get((metal, season))
    r2 = winner_r2.get((metal, season))
    title = f"{metal} -- {season}\n" + rf"LOOCV $R^2 = {r2:+.3f}$"
    if boot is not None:
        _, ci_lo, ci_hi = boot
        title += rf" (95% CI $[{ci_lo:+.3f}, {ci_hi:+.3f}]$, $n=17$)"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(linestyle=":", alpha=0.5)

    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(f"Predicted {metal} (mg/kg)", fontsize=9)

legend_elems = [
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor="none", markeredgecolor="black",
               markersize=12, markeredgewidth=1.6,
               label="Field-sampled station (n=17)"),
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor="lightgray", markeredgecolor="black",
               markersize=8, markeredgewidth=0.4,
               label="IDW spatial scaffold (n=100)"),
]
for river, color in river_color.items():
    legend_elems.append(plt.Line2D([0], [0], marker="o", color="w",
                                   markerfacecolor=color, markeredgecolor="black",
                                   markersize=8, markeredgewidth=0.4,
                                   label=river))
ax_legend.legend(handles=legend_elems, loc="center", ncol=7,
                 frameon=False, fontsize=10)

fig.suptitle("Spatial prediction of sediment Cd and Pb under the corrected GEE V1 "
             "AlphaEarth pipeline. Cd uses GP pooled+log; Pb uses GP "
             "pooled+log+climate. Spatial predictions for the remaining six metals "
             "are not shown because LOOCV $R^2 < 0.5$ does not support spatial inference.",
             fontsize=11, y=0.99)

plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT}")
