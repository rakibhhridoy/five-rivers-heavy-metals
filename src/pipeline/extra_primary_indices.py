"""Primary-data contamination indices for Table 1 (§3.1).

Computes Igeo, EF, CF, mCd, PLI on the 17-station primary dataset using the
**USEPA regional screening backgrounds** (manuscript methodology §3.2),
distinct from the Turekian–Wedepohl harmonised backgrounds used in
phase1_2_harmonized_indices.py for the cross-paper comparison.

Backgrounds (USEPA regional, mg/kg, manuscript §3.2):
  Cr=90, Ni=47, Cu=36, As=6.72, Cd=0.36, Pb=19, Fe=23000

Inputs
------
data/sedimentRO_v2.csv, data/sedimentWO_v2.csv

Output
------
results/s_primary_indices.csv     — per-station Igeo/CF/EF/PLI per (Station, Season, Metal)
results/s_primary_indices_river.csv — Table 1 ready: max Igeo, EF range, mCd, PLI per (River, Season)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT_STN = ROOT / "results" / "s_primary_indices.csv"
OUT_RIVER = ROOT / "results" / "s_primary_indices_river.csv"

METALS = ["Cr", "Ni", "Cu", "As", "Cd", "Pb", "Fe"]
TRACE = ["Cr", "Ni", "Cu", "As", "Cd", "Pb"]  # for mCd; Fe is reference

BACKGROUND_USEPA = {
    "Cr": 90.0,  "Ni": 47.0,  "Cu": 36.0,   "As": 6.72,
    "Cd": 0.36,  "Pb": 19.0,  "Fe": 23000.0,
}

METAL_COLS_R = {m: f"{m if m != 'Cr' and m != 'Ni' and m != 'Cu' and m != 'As' and m != 'Cd' and m != 'Pb' and m != 'Fe' else m}R"
                for m in METALS}
METAL_COLS_R = {m: f"{m}R" for m in METALS}
METAL_COLS_W = {m: f"{m}W" for m in METALS}


def compute_indices(df: pd.DataFrame, season_suffix: str) -> pd.DataFrame:
    cols = METAL_COLS_R if season_suffix == "R" else METAL_COLS_W
    rows = []
    Fe_samp = df[cols["Fe"]].values
    for m in METALS:
        Cn = df[cols[m]].values
        Bn = BACKGROUND_USEPA[m]
        cf = Cn / Bn
        igeo = np.log2(np.clip(Cn / (1.5 * Bn), 1e-12, None))
        ef = (Cn / Fe_samp) / (Bn / BACKGROUND_USEPA["Fe"])
        for i, st in enumerate(df["Stations"].values):
            rows.append(dict(
                Station=st, River=df["River"].values[i],
                Season="Rainy" if season_suffix == "R" else "Winter",
                Metal=m, Conc_mg_per_kg=float(Cn[i]),
                CF=float(cf[i]), Igeo=float(igeo[i]), EF=float(ef[i]),
            ))
    return pd.DataFrame(rows)


def main() -> None:
    sed_r = pd.read_csv(DATA / "sedimentRO_v2.csv").iloc[:17]
    sed_w = pd.read_csv(DATA / "sedimentWO_v2.csv").iloc[:17]

    long = pd.concat([compute_indices(sed_r, "R"), compute_indices(sed_w, "W")],
                     ignore_index=True)
    long.to_csv(OUT_STN, index=False)

    # PLI per station-season: geometric mean of CF over trace metals
    pli_rows = []
    for season, df in [("Rainy", sed_r), ("Winter", sed_w)]:
        cols = METAL_COLS_R if season == "Rainy" else METAL_COLS_W
        for i in range(len(df)):
            cfs = [df[cols[m]].values[i] / BACKGROUND_USEPA[m] for m in TRACE]
            pli = float(np.exp(np.mean(np.log(np.clip(cfs, 1e-12, None)))))
            mcd = float(np.mean(cfs))  # modified contamination degree
            pli_rows.append(dict(Station=df["Stations"].values[i],
                                 River=df["River"].values[i],
                                 Season=season, PLI=pli, mCd=mcd))
    pli_df = pd.DataFrame(pli_rows)

    # River-level summary (Table 1)
    river_rows = []
    for (river, season), g in long.groupby(["River", "Season"]):
        row = {"River": river, "Season": season}
        for m in TRACE:
            sub = g[g["Metal"] == m]
            row[f"{m}_Igeo_max"] = sub["Igeo"].max()
            row[f"{m}_EF_min"] = sub["EF"].min()
            row[f"{m}_EF_max"] = sub["EF"].max()
        pli_sub = pli_df[(pli_df["River"] == river) & (pli_df["Season"] == season)]
        row["PLI_max"] = pli_sub["PLI"].max()
        row["mCd_max"] = pli_sub["mCd"].max()
        river_rows.append(row)
    river_df = pd.DataFrame(river_rows).sort_values(["River", "Season"])
    river_df.to_csv(OUT_RIVER, index=False)

    print(f"Wrote {OUT_STN.name} ({len(long)} rows) and {OUT_RIVER.name} ({len(river_df)} rows)")
    print("\nTable 1 reproduction — Cd Igeo max, EF range, mCd, PLI per river:")
    cols_to_show = ["River", "Season", "Cd_Igeo_max", "Cd_EF_min", "Cd_EF_max",
                    "mCd_max", "PLI_max"]
    print(river_df[cols_to_show].to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
