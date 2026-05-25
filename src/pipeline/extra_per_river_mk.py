"""Per-river Mann–Kendall trends in pooled secondary literature (§3.4 detail).

Augments phase10_mk_pooled.py (pooled-across-rivers MK) with per-river breakdowns.
Reproduces the per-river Pb percentage-change claims:
  Buriganga 68 → 218 mg/kg (+218%), Turag +503%, Shitalakshya +393%, Dhaleshwari +83%.

Inputs
------
data/secondary/harmonized_sediment.csv (long format with Year, River, Metal, Concentration)

Output
------
results/s_per_river_mk.csv  — per (River × Metal): year span, n_years, first/last median, %Δ, MK τ + p
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pymannkendall as mk

ROOT = Path(__file__).resolve().parents[2]
SEC = ROOT / "data" / "secondary"
OUT = ROOT / "results" / "s_per_river_mk.csv"

METALS = ["Cr", "Ni", "Cu", "As", "Cd", "Pb", "Fe"]


def main() -> None:
    long = pd.read_csv(SEC / "harmonized_sediment.csv")
    # Schema: Year, River, Metal, Concentration (long-format from phase0)
    needed = {"Year", "River", "Metal", "Concentration"}
    if not needed.issubset(long.columns):
        raise SystemExit(f"harmonized_sediment.csv missing columns; have {list(long.columns)}")

    rows = []
    for (river, metal), g in long.groupby(["River", "Metal"]):
        if metal not in METALS:
            continue
        med_by_year = (g.dropna(subset=["Concentration"])
                        .groupby("Year")["Concentration"].median()
                        .sort_index())
        n = len(med_by_year)
        if n < 2:
            continue
        first_yr, last_yr = med_by_year.index.min(), med_by_year.index.max()
        first_v, last_v = float(med_by_year.iloc[0]), float(med_by_year.iloc[-1])
        pct_change = 100.0 * (last_v - first_v) / first_v if first_v != 0 else np.nan
        if n >= 3:
            try:
                r = mk.original_test(med_by_year.values)
                tau, pval, trend = r.Tau, r.p, r.trend
            except Exception:
                tau, pval, trend = np.nan, np.nan, "insufficient"
        else:
            tau, pval, trend = np.nan, np.nan, "n<3 (descriptive only)"
        rows.append(dict(
            River=river, Metal=metal,
            n_years=n, Year_first=first_yr, Year_last=last_yr,
            Median_first=first_v, Median_last=last_v,
            Pct_change=pct_change, MK_tau=tau, MK_pvalue=pval, MK_trend=trend,
        ))

    df = pd.DataFrame(rows).sort_values(["Metal", "River"]).reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT.name} ({len(df)} rows)")
    print("\nPer-river Pb trends (§3.4 prose):")
    pb = df[df["Metal"] == "Pb"].sort_values("River")
    print(pb[["River", "n_years", "Year_first", "Year_last",
              "Median_first", "Median_last", "Pct_change",
              "MK_tau", "MK_pvalue"]].to_string(
        index=False, float_format=lambda x: f"{x:.2f}" if pd.notna(x) else "n/a"))


if __name__ == "__main__":
    main()
