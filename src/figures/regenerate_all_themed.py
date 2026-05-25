"""Regenerate Fig 3, 5, 7, 8 with consistent matplotlib tab10 theme.

Theme reference: Fig_Kd_partition uses Rainy = #1f77b4, Winter = #ff7f0e.
Fig 6 uses tab10 categorical for models. We adopt:
  - Rainy bars/markers: #1f77b4 (tab:blue)
  - Winter bars/markers: #ff7f0e (tab:orange)
  - 3-way categorical (Fig 5): Proximity #1f77b4, AE #ff7f0e, Combined #2ca02c (green)
  - River palette (Fig 7): align with existing study-area legend (Fig 1)
  - Fig 8 spatial: Blues colormap for Cd, Oranges colormap for Pb, both seasons
    same colormap but separate panels.
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
RES = ROOT / "results"
DATA = ROOT / "data"

# THEME
RAINY  = "#1f77b4"  # tab:blue
WINTER = "#ff7f0e"  # tab:orange
COMBINED = "#2ca02c"  # tab:green
THEME_RIVERS = {
    "Buriganga":   "#1f77b4",   # blue
    "Shitalakshya":"#ff7f0e",   # orange
    "Turag":       "#2ca02c",   # green
    "Dhaleshwari": "#d62728",   # red
    "Balu":        "#9467bd",   # purple
}

N_REAL = 17


# ============================================================
# FIG 3: LOOCV winners bar chart
# ============================================================
def fig3():
    df = pd.read_csv(RES / "s8_winners.csv")
    metal_order = ["Cd", "Pb", "Cu", "Ni", "Cr", "Fe", "As"]
    df["Metal"] = pd.Categorical(df["Metal"], categories=metal_order, ordered=True)
    df = df.sort_values(["Metal", "Season"])
    rainy = df[df["Season"] == "Rainy"].set_index("Metal")["Winner_R2"].reindex(metal_order)
    winter = df[df["Season"] == "Winter"].set_index("Metal")["Winner_R2"].reindex(metal_order)

    x = np.arange(len(metal_order)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    br = ax.bar(x - w/2, rainy.values, w, label="Rainy",
                color=RAINY, edgecolor="black", linewidth=0.6)
    bw = ax.bar(x + w/2, winter.values, w, label="Winter",
                color=WINTER, edgecolor="black", linewidth=0.6)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle="--",
               label=r"Useful-prediction threshold ($R^2 = 0.5$)")
    ax.axhline(0.2, color="lightgray", linewidth=0.7, linestyle=":",
               label=r"Substantive-positive threshold ($R^2 = 0.2$)")

    for bars, vals in [(br, rainy.values), (bw, winter.values)]:
        for bar, v in zip(bars, vals):
            y = bar.get_height()
            offset = 0.025 if y >= 0 else -0.04
            ax.text(bar.get_x() + bar.get_width()/2, y + offset, f"{v:+.2f}",
                    ha="center", va="bottom" if y >= 0 else "top", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(metal_order)
    ax.set_ylabel(r"LOOCV $R^2$ (per-metal-season winning model, $n = 17$)")
    ax.set_title("LOOCV performance under per-metal-season winning model selection",
                 fontsize=11)
    ax.set_ylim(-0.5, 0.8)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIG / "Fig3_loocv_performance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Fig3")


# ============================================================
# FIG 5: 3-way ablation (rainy + winter)
# ============================================================
def fig5():
    df = pd.read_csv(RES / "s4_v2_ablation.csv")
    metal_order = ["Cd", "Cu", "Pb", "Ni", "Fe", "As", "Cr"]

    def plot(season, out):
        sub = df[df["Season"] == season].set_index("Metal").reindex(metal_order)
        x = np.arange(len(metal_order)); w = 0.27
        fig, ax = plt.subplots(figsize=(10, 4.8))
        b1 = ax.bar(x - w, sub["Prox_only_R2"], w, label="Proximity only (4 features)",
                    color=RAINY, edgecolor="black", linewidth=0.6)
        b2 = ax.bar(x,     sub["AE_only_R2"],   w, label="AlphaEarth only (64 dims)",
                    color=WINTER, edgecolor="black", linewidth=0.6)
        b3 = ax.bar(x + w, sub["Both_R2"],      w, label="Combined (68 features)",
                    color=COMBINED, edgecolor="black", linewidth=0.6)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(0.5, color="gray", linewidth=0.7, linestyle="--",
                   label=r"Useful-prediction threshold ($R^2 = 0.5$)")
        for bars in [b1, b2, b3]:
            for bar in bars:
                v = bar.get_height()
                offset = 0.025 if v >= 0 else -0.04
                ax.text(bar.get_x() + bar.get_width()/2, v + offset, f"{v:+.2f}",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(metal_order)
        ax.set_ylabel(r"5-fold CV $R^2$ ($n=117$)")
        ax.set_title(f"Same-model feature ablation --- {season} season", fontsize=11)
        ax.set_ylim(-0.35, 0.85)
        ax.legend(loc="upper right", fontsize=8, frameon=True, ncol=2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

    plot("Rainy",  FIG / "Fig5_feature_importance_rainy.png")
    plot("Winter", FIG / "Fig5_feature_importance_winter.png")
    print("Saved Fig5")


# ============================================================
# FIG 7: LOOCV scatter, 6 panels, river-coloured tab10 palette
# ============================================================
def fig7():
    df_w = pd.read_csv(RES / "s9_winner_loocv_preds.csv")
    df_v = pd.read_csv(RES / "s4_v2_loocv_preds.csv")
    panels = [
        ("Cd", "Rainy",  df_w, "GP pooled+log"),
        ("Cd", "Winter", df_w, "GP pooled+log"),
        ("Pb", "Rainy",  df_w, "GP pooled+log+climate"),
        ("Pb", "Winter", df_w, "GP pooled+log+climate"),
        ("Cu", "Rainy",  df_w, "GP pooled+log+climate"),
        ("Ni", "Rainy",  df_v, "XGBoost (per season)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5))
    axes = axes.flatten()
    for ax, (metal, season, src, label) in zip(axes, panels):
        sub = src[(src["Metal"] == metal) & (src["Season"] == season)]
        obs = sub["Observed"].values; pred = sub["Predicted"].values
        if len(obs) == 0: continue
        r2 = sub["LOOCV_R2"].iloc[0]
        rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
        lo = float(min(obs.min(), pred.min())); hi = float(max(obs.max(), pred.max()))
        pad = 0.08 * (hi - lo) if hi > lo else 1.0; lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=0.8, label="1:1")
        for river, color in THEME_RIVERS.items():
            m = sub["River"].values == river
            if m.any():
                ax.scatter(obs[m], pred[m], s=55, c=color, edgecolor="black",
                           linewidth=0.6, label=river, zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel(f"Observed {metal} (mg/kg)")
        ax.set_ylabel(f"Predicted {metal} (mg/kg)")
        ax.set_title(f"{metal} -- {season}    "
                     rf"$R^2 = {r2:+.3f}$, RMSE = {rmse:.2f}" + "\n" + label,
                     fontsize=9)
        ax.grid(linestyle=":", alpha=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)
    plt.suptitle("LOOCV predicted vs observed under per-metal-season winning model",
                 fontsize=11, y=1.0)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(FIG / "Fig7_scatter_loocv.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Fig7")


# ============================================================
# FIG 8: spatial Cd + Pb maps. Use Blues for Cd, Oranges for Pb
# ============================================================
def fig8():
    df = pd.read_csv(RES / "s9_winner_spatial_preds.csv")
    winners = pd.read_csv(RES / "s8_winners.csv")
    winner_r2 = {(r.Metal, r.Season): r.Winner_R2 for _, r in winners.iterrows()}
    cd_boot = pd.read_csv(RES / "s4_v2_cd_bootstrap.csv")
    pb_boot = pd.read_csv(RES / "s9_pb_bootstrap.csv")
    boot = {}
    for _, r in cd_boot.iterrows():
        boot[(r.Metal, r.Season)] = (r.CI95_low, r.CI95_high)
    for _, r in pb_boot.iterrows():
        boot[(r.Metal, r.Season)] = (r.CI95_low, r.CI95_high)

    panels = [("Cd", "Rainy"), ("Cd", "Winter"),
              ("Pb", "Rainy"), ("Pb", "Winter")]
    cmaps = {"Cd": plt.cm.Blues, "Pb": plt.cm.Oranges}

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[10, 10, 1], hspace=0.42, wspace=0.22)
    ax_panels = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    ax_legend = fig.add_subplot(gs[2, :]); ax_legend.axis("off")

    for ax, (metal, season) in zip(ax_panels, panels):
        sub = df[(df["Metal"] == metal) & (df["Season"] == season)]
        pred = sub["Predicted"].values
        vmax = float(np.percentile(pred, 95))
        norm = Normalize(vmin=0.0, vmax=vmax)
        cmap = cmaps[metal]
        sc = ax.scatter(sub["Long"], sub["Lat"], c=pred, s=55, cmap=cmap,
                        norm=norm, edgecolor="black", linewidth=0.4,
                        alpha=0.95, zorder=2)
        real = sub[sub["is_real_station"] == 1]
        for _, r in real.iterrows():
            ax.scatter(r["Long"], r["Lat"], facecolor="none", edgecolor="black",
                       linewidth=1.6, s=180, zorder=3)
        for river, color in THEME_RIVERS.items():
            m = real["River"].values == river
            if m.any():
                ax.scatter(real["Long"].values[m], real["Lat"].values[m],
                           c=color, s=22, edgecolor="black", linewidth=0.4, zorder=4)
        r2 = winner_r2.get((metal, season))
        ci = boot.get((metal, season))
        title = f"{metal} -- {season}\n" + rf"LOOCV $R^2 = {r2:+.3f}$"
        if ci is not None:
            title += rf" (95% CI $[{ci[0]:+.3f}, {ci[1]:+.3f}]$, $n=17$)"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.grid(linestyle=":", alpha=0.5)
        cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label(f"Predicted {metal} (mg/kg)", fontsize=9)

    legend_elems = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor="black", markersize=12, markeredgewidth=1.6,
                   label="Field-sampled station (n=17)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray",
                   markeredgecolor="black", markersize=8, markeredgewidth=0.4,
                   label="IDW spatial scaffold (n=100)"),
    ]
    for river, color in THEME_RIVERS.items():
        legend_elems.append(plt.Line2D([0], [0], marker="o", color="w",
                                       markerfacecolor=color, markeredgecolor="black",
                                       markersize=8, markeredgewidth=0.4, label=river))
    ax_legend.legend(handles=legend_elems, loc="center", ncol=7,
                     frameon=False, fontsize=10)

    fig.suptitle("Spatial prediction of sediment Cd (Blues) and Pb (Oranges) under per-metal "
                 "winning model. Predictions for the remaining six metals are not shown "
                 r"because LOOCV $R^2 < 0.5$ does not support spatial inference.",
                 fontsize=11, y=0.99)
    plt.savefig(FIG / "Fig8_spatial_cd_pb.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Fig8")


if __name__ == "__main__":
    fig3()
    fig5()
    fig7()
    fig8()
    print("\nAll four themed figures regenerated.")
