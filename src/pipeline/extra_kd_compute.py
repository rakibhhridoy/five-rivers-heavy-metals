"""Compute sediment-water partition coefficients K_d for Table 2 and Fig K_d_partition.

K_d = C_sediment (mg/kg) / C_water (mg/L), units L/kg.

Manuscript scope: 7 metals (Cr, Ni, Cu, As, Cd, Pb, Fe) × 2 seasons × 17 stations.
Mn is excluded (see docs/KNOWN_ISSUES.md).

Inputs
------
data/sedimentRO_v2.csv, data/sedimentWO_v2.csv  — primary sediment (mg/kg)
data/waterR_original_17.csv, data/waterW_original_17.csv  — primary water (mg/L)

Output
------
results/s_kd_values.csv  — long-format Kd per (Metal, Season, Station)
results/s_kd_summary.csv  — Mean/Median/Min/Max per (Metal, Season) — reproduces Table 2
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT_LONG = ROOT / "results" / "s_kd_values.csv"
OUT_SUMM = ROOT / "results" / "s_kd_summary.csv"

# (sediment col, water col) — same suffix scheme R=Rainy, W=Winter.
METAL_COLS = {
    "Cr": ("CrR", "CrW"),
    "Ni": ("NiR", "NiW"),
    "Cu": ("CuR", "CuW"),
    "As": ("AsR", "AsW"),
    "Cd": ("CdR", "CdW"),
    "Pb": ("PbR", "PbW"),
    "Fe": ("FeR", "FeW"),
}


def kd_for_season(sed_df: pd.DataFrame, wat_df: pd.DataFrame, suffix: str, season: str) -> pd.DataFrame:
    """Compute Kd for one season; returns long-format rows."""
    rows = []
    for metal, (col_r, col_w) in METAL_COLS.items():
        col = col_r if suffix == "R" else col_w
        sed = sed_df[col].values.astype(float)
        wat = wat_df[col].values.astype(float)
        # NB: K_d = mg/kg ÷ mg/L = L/kg
        kd = sed / wat
        for st, k, s, w in zip(sed_df["Stations"], kd, sed, wat):
            rows.append(dict(Metal=metal, Season=season, Station=st,
                             Sed_mg_per_kg=s, Water_mg_per_L=w, Kd_L_per_kg=k))
    return pd.DataFrame(rows)


def main() -> None:
    sed_r = pd.read_csv(DATA / "sedimentRO_v2.csv").iloc[:17]
    sed_w = pd.read_csv(DATA / "sedimentWO_v2.csv").iloc[:17]
    wat_r = pd.read_csv(DATA / "waterR_original_17.csv")
    wat_w = pd.read_csv(DATA / "waterW_original_17.csv")

    long = pd.concat([
        kd_for_season(sed_r, wat_r, "R", "Rainy"),
        kd_for_season(sed_w, wat_w, "W", "Winter"),
    ], ignore_index=True)
    long.to_csv(OUT_LONG, index=False)

    summ = (long.groupby(["Metal", "Season"])["Kd_L_per_kg"]
                .agg(["mean", "median", "min", "max"])
                .reset_index()
                .rename(columns=dict(mean="Mean", median="Median", min="Min", max="Max")))
    # Preserve manuscript metal order
    metal_order = ["Cr", "Ni", "Cu", "As", "Cd", "Pb", "Fe"]
    summ["Metal"] = pd.Categorical(summ["Metal"], categories=metal_order, ordered=True)
    summ = summ.sort_values(["Metal", "Season"]).reset_index(drop=True)
    summ.to_csv(OUT_SUMM, index=False)
    print(f"Wrote {OUT_LONG.name} ({len(long)} rows) and {OUT_SUMM.name} ({len(summ)} rows)")
    print("\nTable 2 reproduction:")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))


if __name__ == "__main__":
    main()
