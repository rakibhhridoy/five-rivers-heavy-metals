"""Regenerate Fig 6 — Monte Carlo RI per-metal contribution (§3.7).

Reads the primary-headline Monte Carlo (extra_mc_primary_headline.py) and renders
the per-metal contribution to total RI as a bar chart, annotated with %.

Inputs
------
results/s_mc_primary_headline.csv  — distribution × metal × {Er_mean, Contribution_pct}

Output
------
figures/Fig6_ri_performance.png
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "s_mc_primary_headline.csv"
OUT = ROOT / "figures" / "Fig6_ri_performance.png"


def main() -> None:
    df = pd.read_csv(SRC)
    ln = df[df["Distribution"] == "lognormal"].copy().sort_values("Contribution_pct",
                                                                  ascending=False)
    metals = ln["Metal"].tolist()
    pct = ln["Contribution_pct"].values

    cmap = plt.cm.tab10
    colors = [cmap(i) for i in range(len(metals))]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(metals, pct, color=colors, edgecolor="black", linewidth=0.7)
    for b, v in zip(bars, pct):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.2f}%",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Contribution to mean RI (%)")
    ax.set_title("Monte Carlo RI — per-metal contribution (lognormal, primary n=34, "
                 "USEPA regional shale background, 10,000 iter)", fontsize=10)
    ax.set_ylim(0, max(pct.max() * 1.15, 80))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
