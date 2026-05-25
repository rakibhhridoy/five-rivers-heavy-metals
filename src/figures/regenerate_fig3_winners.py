"""Fig 3 v3: LOOCV R^2 bars using per-metal-season WINNING model
(after pooled+log+climate model search). Source: s8_winners.csv.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "SubmissionV2/LaTeX_Source/figures/Fig3_loocv_performance.png"

df = pd.read_csv(ROOT / "results/s8_winners.csv")

metal_order = ["Cd", "Pb", "Cu", "Ni", "Cr", "Fe", "As"]
df["Metal"] = pd.Categorical(df["Metal"], categories=metal_order, ordered=True)
df = df.sort_values(["Metal", "Season"])

rainy = df[df["Season"] == "Rainy"].set_index("Metal")["Winner_R2"].reindex(metal_order)
winter = df[df["Season"] == "Winter"].set_index("Metal")["Winner_R2"].reindex(metal_order)

x = np.arange(len(metal_order))
w = 0.38

fig, ax = plt.subplots(figsize=(10, 5))
bar_r = ax.bar(x - w/2, rainy.values, w, label="Rainy",
               color="#2b8cbe", edgecolor="black", linewidth=0.6)
bar_w = ax.bar(x + w/2, winter.values, w, label="Winter",
               color="#e9a3c9", edgecolor="black", linewidth=0.6)

ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(0.5, color="gray", linewidth=0.7, linestyle="--",
           label=r"Useful-prediction threshold ($R^2 = 0.5$)")
ax.axhline(0.2, color="lightgray", linewidth=0.7, linestyle=":",
           label=r"Substantive-positive threshold ($R^2 = 0.2$)")

for bars, vals in [(bar_r, rainy.values), (bar_w, winter.values)]:
    for bar, v in zip(bars, vals):
        y = bar.get_height()
        offset = 0.025 if y >= 0 else -0.04
        ax.text(bar.get_x() + bar.get_width()/2, y + offset, f"{v:+.2f}",
                ha="center", va="bottom" if y >= 0 else "top", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(metal_order)
ax.set_ylabel(r"LOOCV $R^2$ (per-metal-season winning model, $n = 17$)")
ax.set_title("Honest LOOCV performance under per-metal-season winning model selection\n"
             "(per-season family: DL+RF+XGB+SVR; pooled+log+climate family: GP, SVR, MultiTask DL)",
             fontsize=10)
ax.set_ylim(-0.5, 0.8)
ax.legend(loc="lower right", fontsize=8, frameon=True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT}")
