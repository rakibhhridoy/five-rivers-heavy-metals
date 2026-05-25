"""Wilcoxon signed-rank tests on winter-vs-rainy seasonal gradient.

For each metal, paired by station (n=17), tests whether winter > rainy.
Reports paired median ratio (winter/rainy), Wilcoxon p-value, and effect direction.

Sediment matrix only (the seasonal-gradient claim is on sediment in Section 4.1).

Output: analysis_results/enrichment/s7_seasonal_wilcoxon.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "results/s7_seasonal_wilcoxon.csv"

N_REAL = 17

TARGETS = {
    "Cd": ("CdR", "CdW"), "Cr": ("CrR", "CrW"), "As": ("AsR", "AsW"),
    "Pb": ("PbR", "PbW"), "Cu": ("CuR", "CuW"), "Ni": ("NiR", "NiW"),
    "Fe": ("FeR", "FeW"),
}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rainy = pd.read_csv(DATA / "sedimentRO_v2.csv").iloc[:N_REAL]
    winter = pd.read_csv(DATA / "sedimentWO_v2.csv").iloc[:N_REAL]

    rows = []
    for metal, (col_r, col_w) in TARGETS.items():
        r = rainy[col_r].values.astype(float)
        w = winter[col_w].values.astype(float)
        diff = w - r
        med_r, med_w = float(np.median(r)), float(np.median(w))
        med_ratio = med_w / med_r if med_r > 0 else np.nan
        # one-sided Wilcoxon: H1 winter > rainy
        try:
            stat, p = wilcoxon(diff, alternative="greater", zero_method="wilcox")
        except Exception:
            stat, p = np.nan, np.nan
        sig = ("***" if p < 0.001 else "**" if p < 0.01 else
               "*"  if p < 0.05  else "ns")
        rows.append({
            "Metal": metal,
            "Median_Rainy": round(med_r, 3),
            "Median_Winter": round(med_w, 3),
            "Median_Ratio_W_over_R": round(med_ratio, 2) if med_ratio == med_ratio else None,
            "Wilcoxon_stat": round(float(stat), 2) if stat == stat else None,
            "p_one_sided_W_gt_R": round(float(p), 4) if p == p else None,
            "Significance": sig,
            "n_pairs": int(N_REAL),
        })
        print(f"  {metal:3s}  median W/R = {med_ratio:.2f}x  "
              f"Wilcoxon p = {p:.4f}  {sig}")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
