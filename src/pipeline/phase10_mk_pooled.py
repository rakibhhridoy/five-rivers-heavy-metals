"""Mann-Kendall temporal trend test on pooled (all-river) literature data + 2024 primary.

For each metal x matrix, pools all rivers and computes per-year median, then
runs Mann-Kendall on the year-median series. Reports tau, p-value, and Sen's slope.

Output: analysis_results/enrichment/s10_mk_pooled.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SECDATA = ROOT / "data/secondary"
PRIMARY = ROOT / "data"
OUT = ROOT / "results/s10_mk_pooled.csv"

METALS = ["Cd", "Cr", "As", "Pb", "Cu", "Ni", "Fe"]

PRIM_RAINY = {"Cd": "CdR", "Cr": "CrR", "As": "AsR", "Pb": "PbR",
              "Cu": "CuR", "Ni": "NiR", "Fe": "FeR"}
PRIM_WINTER = {"Cd": "CdW", "Cr": "CrW", "As": "AsW", "Pb": "PbW",
               "Cu": "CuW", "Ni": "NiW", "Fe": "FeW"}


def mann_kendall(years, vals):
    n = len(years)
    if n < 4:
        return np.nan, np.nan, np.nan
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(vals[j] - vals[i])
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (n * (n - 1) / 2)
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if years[j] - years[i] != 0:
                slopes.append((vals[j] - vals[i]) / (years[j] - years[i]))
    sen = float(np.median(slopes)) if slopes else np.nan
    return float(tau), float(p), sen


def main():
    sed_lit = pd.read_csv(SECDATA / "harmonized_sediment.csv")
    wat_lit = pd.read_csv(SECDATA / "harmonized_water.csv")
    rainy = pd.read_csv(PRIMARY / "sedimentRO_v2.csv").iloc[:17]
    winter = pd.read_csv(PRIMARY / "sedimentWO_v2.csv").iloc[:17]

    rows = []
    for matrix, df_lit in [("Sediment", sed_lit), ("Water", wat_lit)]:
        for metal in METALS:
            sub = df_lit[df_lit["Metal"] == metal][["Year", "Concentration"]].copy()
            year_med = sub.groupby("Year")["Concentration"].median()
            if matrix == "Sediment":
                primary_med = float(np.median(np.concatenate([
                    rainy[PRIM_RAINY[metal]].values,
                    winter[PRIM_WINTER[metal]].values,
                ])))
                year_med.loc[2024] = primary_med
            year_med = year_med.sort_index().dropna()
            years = year_med.index.values.astype(float)
            vals = year_med.values.astype(float)
            mask = ~np.isnan(vals)
            years, vals = years[mask], vals[mask]

            tau, p, sen = mann_kendall(years, vals)
            yr0, yr1 = float(years.min()), float(years.max())
            v0, v1 = float(vals[0]), float(vals[-1])
            pct = 100.0 * (v1 - v0) / v0 if v0 != 0 else np.nan
            sig = ("***" if p < 0.001 else "**" if p < 0.01 else
                   "*" if p < 0.05 else "ns")
            rows.append({
                "Matrix": matrix, "Metal": metal,
                "n_years": int(len(years)),
                "year_first": int(yr0), "year_last": int(yr1),
                "median_first": round(v0, 2), "median_last": round(v1, 2),
                "pct_change": round(pct, 1) if not np.isnan(pct) else None,
                "MK_tau": round(tau, 3) if not np.isnan(tau) else None,
                "MK_p": round(p, 4) if not np.isnan(p) else None,
                "Sens_slope": round(sen, 3) if not np.isnan(sen) else None,
                "Significance": sig,
            })
            print(f"  {matrix:8s} {metal:3s}  n={len(years)}  "
                  f"{yr0:.0f}-{yr1:.0f}  pct={pct:+.0f}%  "
                  f"tau={tau:+.2f}  p={p:.3f}  {sig}")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
