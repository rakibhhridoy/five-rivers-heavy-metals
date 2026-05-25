"""
Regenerate Fig_Kd_partition.png.

Reads computed K_d summary from results/s_kd_summary.csv (produced by
extra_kd_compute.py). Run extra_kd_compute.py first if the file is missing.

Theme: Rainy = #1f77b4, Winter = #ff7f0e (matches Fig 3, 5, 7, 8).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"
SRC = ROOT / "results" / "s_kd_summary.csv"

RAINY = "#1f77b4"
WINTER = "#ff7f0e"

if not SRC.exists():
    raise SystemExit(
        f"{SRC} not found. Run src/pipeline/extra_kd_compute.py first."
    )

_kd = pd.read_csv(SRC)
DATA = {
    m: {s[0]: tuple(_kd[(_kd["Metal"] == m) & (_kd["Season"] == season)]
                    [["Mean", "Median", "Min", "Max"]].iloc[0].values)
        for s, season in [("R", "Rainy"), ("W", "Winter")]}
    for m in _kd["Metal"].unique()
}

trace = ["Cr", "Ni", "Cu", "As", "Cd", "Pb"]

fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [4, 1]}
)

bw = 0.38
xs = np.arange(len(trace))


def stats_arr(metals, season):
    means = np.array([DATA[m][season][0] for m in metals])
    medians = np.array([DATA[m][season][1] for m in metals])
    mins = np.array([DATA[m][season][2] for m in metals])
    maxs = np.array([DATA[m][season][3] for m in metals])
    err_lo = means - mins
    err_hi = maxs - means
    return means, medians, np.vstack([err_lo, err_hi])


# Panel (a) — trace metals, common scale
mR, medR, errR = stats_arr(trace, "R")
mW, medW, errW = stats_arr(trace, "W")
axA.bar(xs - bw / 2, mR, bw, yerr=errR, capsize=3, color=RAINY,
        edgecolor="black", linewidth=0.6, label="Rainy",
        error_kw={"elinewidth": 0.8, "ecolor": "black"})
axA.bar(xs + bw / 2, mW, bw, yerr=errW, capsize=3, color=WINTER,
        edgecolor="black", linewidth=0.6, label="Winter",
        error_kw={"elinewidth": 0.8, "ecolor": "black"})
# Median tick marks
mark_w = bw * 0.7
for i, (mr, mw) in enumerate(zip(medR, medW)):
    axA.hlines(mr, i - bw / 2 - mark_w / 2, i - bw / 2 + mark_w / 2,
               colors="black", linewidth=1.4)
    axA.hlines(mw, i + bw / 2 - mark_w / 2, i + bw / 2 + mark_w / 2,
               colors="black", linewidth=1.4)
axA.set_xticks(xs)
axA.set_xticklabels(trace)
axA.set_xlabel("Metal")
axA.set_ylabel(r"$K_d$  (L/kg)")
axA.set_title("(a) Trace Metals")
axA.legend(loc="upper right", framealpha=0.9)
axA.grid(axis="y", alpha=0.25)
axA.set_axisbelow(True)

# Panel (b) — Fe alone
fe_R = DATA["Fe"]["R"]
fe_W = DATA["Fe"]["W"]
errFeR = np.array([[fe_R[0] - fe_R[2]], [fe_R[3] - fe_R[0]]])
errFeW = np.array([[fe_W[0] - fe_W[2]], [fe_W[3] - fe_W[0]]])
axB.bar(0 - bw / 2, fe_R[0], bw, yerr=errFeR, capsize=3, color=RAINY,
        edgecolor="black", linewidth=0.6,
        error_kw={"elinewidth": 0.8, "ecolor": "black"})
axB.bar(0 + bw / 2, fe_W[0], bw, yerr=errFeW, capsize=3, color=WINTER,
        edgecolor="black", linewidth=0.6,
        error_kw={"elinewidth": 0.8, "ecolor": "black"})
axB.hlines(fe_R[1], 0 - bw / 2 - mark_w / 2, 0 - bw / 2 + mark_w / 2,
           colors="black", linewidth=1.4)
axB.hlines(fe_W[1], 0 + bw / 2 - mark_w / 2, 0 + bw / 2 + mark_w / 2,
           colors="black", linewidth=1.4)
axB.set_xticks([0])
axB.set_xticklabels(["Fe"])
axB.set_ylabel(r"$K_d$  (L/kg)")
axB.set_title("(b) Fe")
axB.grid(axis="y", alpha=0.25)
axB.set_axisbelow(True)

fig.suptitle(
    r"Sediment$-$Water Partition Coefficients ($K_d$) by Metal and Season "
    "(n=17 stations)",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "Fig_Kd_partition.png", dpi=200, bbox_inches="tight")
print("Wrote Fig_Kd_partition.png")
