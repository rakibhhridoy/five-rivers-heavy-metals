"""Regenerate Fig 8 (spatial Cd prediction maps) under v2.
Cd is the only metal meeting the LOOCV R^2 >= 0.5 threshold (Table 5);
spatial prediction is therefore shown only for Cd in both seasons.

Train SVR on 17 real stations using the 68-feature satellite-only set;
predict at all 117 station locations (real + IDW spatial scaffold).
Source: data/sedimentRO_v2.csv, data/sedimentWO_v2.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "SubmissionV2/LaTeX_Source/figures/Fig8_spatial_cd.png"

N_REAL = 17
METADATA = ["Stations", "River", "Lat", "Long", "geometry"]
RAINY_CHEM  = ["CrR", "NiR", "CuR", "AsR", "CdR", "PbR", "MR",
               "SandR", "SiltR", "ClayR", "FeR", "RI"]
WINTER_CHEM = ["CdW", "CrW", "AsW", "PbW", "CuW", "NiW", "FeW", "MW",
               "SandW", "SiltW", "ClayW", "RI"]

river_color = {
    "Buriganga": "#d7191c", "Shitalakshya": "#fdae61", "Turag": "#2c7bb6",
    "Dhaleshwari": "#abd9e9", "Balu": "#1a9641",
}


def predict_cd(df, target_col, chem_drop):
    feat = [c for c in df.columns if c not in METADATA + chem_drop]
    X = df[feat].fillna(0).values.astype(np.float32)
    y_real = df[target_col].values.astype(np.float32)[:N_REAL]
    sc = StandardScaler()
    X_real_s = sc.fit_transform(X[:N_REAL])
    X_all_s = sc.transform(X)
    m = SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")
    m.fit(X_real_s, y_real)
    return m.predict(X_all_s), y_real


def main():
    rainy = pd.read_csv(DATA / "sedimentRO_v2.csv")
    winter = pd.read_csv(DATA / "sedimentWO_v2.csv")

    pred_r, y_r = predict_cd(rainy, "CdR", RAINY_CHEM)
    pred_w, y_w = predict_cd(winter, "CdW", WINTER_CHEM)

    vmax = float(max(np.percentile(pred_r, 95), np.percentile(pred_w, 95),
                     y_r.max(), y_w.max()))
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.cm.YlOrRd

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[10, 1], hspace=0.35, wspace=0.20)
    ax_r = fig.add_subplot(gs[0, 0])
    ax_w = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, :]); ax_legend.axis("off")

    panels = [
        (ax_r, rainy, pred_r, "Rainy", 0.607, 0.748, 0.114, 0.959),
        (ax_w, winter, pred_w, "Winter", 0.527, 0.727, 0.143, 0.956),
    ]

    for ax, df, pred, season, r2, med, lo, hi in panels:
        sc = ax.scatter(df["Long"], df["Lat"], c=pred, s=55, cmap=cmap,
                        norm=norm, edgecolor="black", linewidth=0.4,
                        alpha=0.95, zorder=2)
        for i in range(N_REAL):
            ax.scatter(df["Long"].iloc[i], df["Lat"].iloc[i],
                       facecolor="none", edgecolor="black",
                       linewidth=1.8, s=200, zorder=3)
        for river, color in river_color.items():
            mask = df["River"].iloc[:N_REAL].values == river
            xs = df["Long"].iloc[:N_REAL].values[mask]
            ys = df["Lat"].iloc[:N_REAL].values[mask]
            for x, y in zip(xs, ys):
                ax.scatter(x, y, c=color, s=22, edgecolor="black",
                           linewidth=0.4, zorder=4)
        ax.set_title(
            f"Cd -- {season}\n"
            rf"LOOCV $R^2 = {r2:+.3f}$  "
            rf"(bootstrap median $= {med:+.3f}$, 95% CI $[{lo:+.3f}, {hi:+.3f}]$)",
            fontsize=10,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(linestyle=":", alpha=0.5)

        cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("Predicted Cd (mg/kg)", fontsize=9)

    legend_elems = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="none", markeredgecolor="black",
                   markersize=12, markeredgewidth=1.8,
                   label="Field-sampled station (n=17)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="lightgray", markeredgecolor="black",
                   markersize=8, markeredgewidth=0.4,
                   label="IDW spatial scaffold (n=100)"),
    ]
    for river, color in river_color.items():
        legend_elems.append(plt.Line2D([0], [0], marker="o", color="w",
                                       markerfacecolor=color,
                                       markeredgecolor="black",
                                       markersize=8, markeredgewidth=0.4,
                                       label=river))
    ax_legend.legend(handles=legend_elems, loc="center", ncol=7,
                     frameon=False, fontsize=10)

    fig.suptitle(
        "Cadmium spatial prediction (sediment) under the corrected GEE V1 "
        "AlphaEarth pipeline. Cd is the only metal meeting LOOCV $R^2 \geq "
        "0.5$; predictions for the other seven metals are not shown.",
        fontsize=11, y=0.99,
    )
    plt.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
