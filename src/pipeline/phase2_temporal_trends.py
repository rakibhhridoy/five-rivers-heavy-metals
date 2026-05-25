"""Phase 2 — Decadal temporal trends (Strategy S1).

Show metal concentration trajectories 2015 → 2025 per river × metal × matrix.
Literature provides 2015, 2019, 2021, 2023, 2025 snapshots; primary provides 2024.

Outputs:
  analysis_results/enrichment/s1_temporal_trends.png
  analysis_results/enrichment/s1_trend_tests.csv
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import pymannkendall as mk

ROOT = Path(__file__).resolve().parents[2]
SECONDARY = ROOT / "data/secondary"
PRIMARY = ROOT / "data"
OUT = ROOT / "results"

METALS = ["Cr", "Ni", "Cu", "As", "Cd", "Pb", "Fe"]
PRIMARY_METAL_COL = {"Cr": "Cr", "Ni": "Ni", "Cu": "Cu", "As": "As",
                     "Cd": "Cd", "Pb": "Pb", "Fe": "Fe"}

# Rivers with enough temporal coverage (≥2 time points)
FOCUS_RIVERS = ["Buriganga", "Turag", "Dhaleshwari", "Shitalakshya", "Balu"]
PRIMARY_YEAR = 2024


def load_all_long() -> pd.DataFrame:
    """Combined long dataset: one row per (source, paper, station, year, river, season, matrix, metal, conc)."""
    # Secondary
    w = pd.read_csv(SECONDARY / "harmonized_water.csv")
    s = pd.read_csv(SECONDARY / "harmonized_sediment.csv")
    sec = pd.concat([w, s], ignore_index=True)
    sec = sec[["Paper_ID", "First_Author", "Year", "River", "Season_Norm",
               "Matrix", "Metal", "Concentration"]].copy()
    sec = sec[sec["Concentration"].notna()]
    sec["Source"] = "Literature"

    # Primary
    rows = []
    for season, suffix, fn, matrix in [
        ("Rainy", "R", "waterR_original_17.csv", "Water"),
        ("Winter", "W", "waterW_original_17.csv", "Water"),
        ("Rainy", "R", "sedimentRO_v2.csv", "Sediment"),
        ("Winter", "W", "sedimentWO_v2.csv", "Sediment"),
    ]:
        df = pd.read_csv(PRIMARY / fn)
        if matrix == "Sediment":
            df = df[df["Stations"].str.match(r"^S\d+$")].copy()
            df = df[df["Stations"].str.extract(r"S(\d+)").astype(int).iloc[:, 0] <= 17]
        for _, r in df.iterrows():
            for metal in METALS:
                col = f"{PRIMARY_METAL_COL[metal]}{suffix}"
                if col not in df.columns or pd.isna(r[col]):
                    continue
                val = float(r[col])
                if matrix == "Water":
                    val *= 1000.0  # mg/L → µg/L
                rows.append({
                    "Paper_ID": 0, "First_Author": "Hasan", "Year": PRIMARY_YEAR,
                    "River": r["River"], "Season_Norm": season,
                    "Matrix": matrix, "Metal": metal, "Concentration": val,
                    "Source": "Primary",
                })
    prim = pd.DataFrame(rows)
    return pd.concat([sec, prim], ignore_index=True)


def trend_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Per (River, Metal, Matrix): Mann-Kendall where ≥4 time points; Sen's slope."""
    rows = []
    for river in FOCUS_RIVERS:
        for matrix in ["Water", "Sediment"]:
            for metal in METALS:
                sub = df[(df.River == river) & (df.Matrix == matrix) & (df.Metal == metal)]
                if sub.empty:
                    continue
                yearly = sub.groupby("Year")["Concentration"].median().reset_index()
                yearly = yearly.sort_values("Year")
                n_points = len(yearly)
                n_obs = len(sub)
                result = {
                    "River": river, "Matrix": matrix, "Metal": metal,
                    "n_years": n_points, "n_obs": n_obs,
                    "years": ",".join(str(int(y)) for y in yearly["Year"]),
                    "medians": ",".join(f"{v:.2f}" for v in yearly["Concentration"]),
                    "min_year": int(yearly["Year"].min()),
                    "max_year": int(yearly["Year"].max()),
                    "pct_change": np.nan,
                    "mk_trend": "insufficient_data",
                    "mk_p": np.nan,
                    "sens_slope": np.nan,
                }
                if n_points >= 2:
                    first = yearly["Concentration"].iloc[0]
                    last = yearly["Concentration"].iloc[-1]
                    if first > 0:
                        result["pct_change"] = round((last - first) / first * 100, 1)
                if n_points >= 4:
                    try:
                        r = mk.original_test(yearly["Concentration"].values)
                        result["mk_trend"] = r.trend
                        result["mk_p"] = round(r.p, 4)
                        s = mk.sens_slope(yearly["Concentration"].values)
                        result["sens_slope"] = round(s.slope, 4)
                    except Exception as e:
                        result["mk_trend"] = f"error: {e}"
                rows.append(result)
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame) -> None:
    """Grid: rows = metals (8), cols = matrix (Water, Sediment). Colors = rivers."""
    fig, axes = plt.subplots(8, 2, figsize=(14, 24), sharex=True)
    fig.suptitle("Temporal trajectory of heavy metals in Dhaka rivers, 2015–2025\n"
                 "Literature medians (colored lines) with IQR shading; primary 2024 = star",
                 fontsize=13, y=0.998)

    river_colors = {
        "Buriganga": "#e41a1c", "Turag": "#377eb8", "Dhaleshwari": "#4daf4a",
        "Shitalakshya": "#984ea3", "Balu": "#ff7f00",
    }

    for row_idx, metal in enumerate(METALS):
        for col_idx, matrix in enumerate(["Water", "Sediment"]):
            ax = axes[row_idx, col_idx]
            sub = df[(df.Matrix == matrix) & (df.Metal == metal)]
            for river in FOCUS_RIVERS:
                sr = sub[sub.River == river]
                if sr.empty:
                    continue
                # Yearly median and IQR
                grp = sr.groupby("Year")["Concentration"].agg(
                    med="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
                ).reset_index().sort_values("Year")
                color = river_colors[river]
                # Literature (non-primary) solid line
                lit = sr[sr.Source == "Literature"]
                prim = sr[sr.Source == "Primary"]
                if not lit.empty:
                    lgrp = lit.groupby("Year")["Concentration"].agg(
                        med="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
                    ).reset_index().sort_values("Year")
                    ax.plot(lgrp["Year"], lgrp["med"], color=color, marker="o",
                            linewidth=1.5, markersize=5, label=river)
                    ax.fill_between(lgrp["Year"], lgrp["q25"], lgrp["q75"],
                                    color=color, alpha=0.15)
                if not prim.empty:
                    pmed = prim["Concentration"].median()
                    ax.scatter([PRIMARY_YEAR], [pmed], marker="*", s=220,
                               color=color, edgecolors="black", linewidth=1, zorder=10)
            ax.set_yscale("log")
            ax.set_title(f"{metal} — {matrix}", fontsize=10, pad=3)
            unit = "µg/L" if matrix == "Water" else "mg/kg"
            ax.set_ylabel(unit, fontsize=9)
            ax.grid(True, which="both", alpha=0.3)
            if row_idx == 0 and col_idx == 1:
                ax.legend(loc="upper right", fontsize=7, frameon=True)
            if row_idx == 7:
                ax.set_xlabel("Year", fontsize=10)

    plt.tight_layout()
    fig.savefig(OUT / "s1_temporal_trends.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_all_long()
    print(f"Combined long dataset: {len(df)} rows")
    print(f"  Years present: {sorted(df['Year'].unique())}")
    print(f"  Rivers: {df.groupby(['River']).size().to_dict()}")

    trends = trend_tests(df)
    trends.to_csv(OUT / "s1_trend_tests.csv", index=False)

    print("\n" + "=" * 80)
    print("PHASE 2 — TEMPORAL TREND TESTS")
    print("=" * 80)
    # Focus on rows with ≥2 time points (actually interesting)
    interesting = trends[trends.n_years >= 2].copy()
    print(f"Combinations with ≥2 time points: {len(interesting)}")

    print("\nTop % changes (|pct_change| ≥ 50%):")
    big = interesting[interesting["pct_change"].abs() >= 50].sort_values("pct_change", ascending=False)
    if not big.empty:
        print(big[["River", "Matrix", "Metal", "n_years", "years", "medians", "pct_change"]].to_string(index=False))
    else:
        print("  (none)")

    print("\nMann-Kendall results (rivers with ≥4 time points):")
    mk_rows = trends[trends.mk_trend.isin(["increasing", "decreasing", "no trend"])]
    if not mk_rows.empty:
        print(mk_rows[["River", "Matrix", "Metal", "n_years", "mk_trend", "mk_p", "sens_slope"]].to_string(index=False))
    else:
        print("  (none — no river has ≥4 time points in our compilation)")

    make_figure(df)
    print(f"\nOutputs:")
    for f in ["s1_temporal_trends.png", "s1_trend_tests.csv"]:
        p = OUT / f
        print(f"  {f}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
