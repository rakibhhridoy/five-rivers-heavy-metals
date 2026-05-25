"""Build pooled long-format dataset (rainy + winter stacked).

n=17 stations x 2 seasons = 34 station-season rows (real only; IDW excluded).
Adds is_winter binary feature, applies log1p to all 7 metal targets.

Note: sediment Mn was never measured at the 17 primary stations; the source
columns previously mapped to "Mn" (MR/MW) are actually moisture %. Mn is
therefore excluded from the sediment pipeline. See docs/KNOWN_ISSUES.md.

Output: results/sediment_pooled_v2.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
OUT = RESULTS / "sediment_pooled_v2.csv"

N_REAL = 17

PROX = ["dist_brick", "num_brick", "dist_ind", "num_ind"]
AE_COLS = [f"AE_{i:02d}" for i in range(64)]
META = ["Stations", "River", "Lat", "Long"]

PROX_RAINY = {
    "hydro_dist_brick": "dist_brick", "num_brick_field": "num_brick",
    "hydro_dist_ind": "dist_ind",     "num_industry": "num_ind",
}
PROX_WINTER = {
    "hydrological_dist_to_nearest_BF":  "dist_brick", "num_upstream_BF":  "num_brick",
    "hydrological_dist_to_nearest_IND": "dist_ind",   "num_upstream_IND": "num_ind",
}

METAL_RAINY = {
    "Cd": "CdR", "Cr": "CrR", "As": "AsR", "Pb": "PbR",
    "Cu": "CuR", "Ni": "NiR", "Fe": "FeR",
}
METAL_WINTER = {
    "Cd": "CdW", "Cr": "CrW", "As": "AsW", "Pb": "PbW",
    "Cu": "CuW", "Ni": "NiW", "Fe": "FeW",
}


def melt(df, mapping, prox_rename, is_winter):
    df = df.rename(columns=prox_rename)
    keep = META + PROX + AE_COLS
    out = df[keep].iloc[:N_REAL].copy()
    for metal, src in mapping.items():
        out[metal] = df[src].iloc[:N_REAL].values.astype(float)
        out[f"log_{metal}"] = np.log1p(out[metal])
    out["is_winter"] = int(is_winter)
    out["station_id"] = out["Stations"].values
    return out


def main():
    rainy = pd.read_csv(DATA / "sedimentRO_v2.csv")
    winter = pd.read_csv(DATA / "sedimentWO_v2.csv")
    pooled = pd.concat([
        melt(rainy, METAL_RAINY, PROX_RAINY, is_winter=0),
        melt(winter, METAL_WINTER, PROX_WINTER, is_winter=1),
    ], ignore_index=True)

    metals = list(METAL_RAINY.keys())
    cols = ["station_id", "River", "Lat", "Long", "is_winter"] + PROX + AE_COLS \
           + metals + [f"log_{m}" for m in metals]
    pooled = pooled[cols]

    pooled.to_csv(OUT, index=False)
    print(f"Wrote {OUT}")
    print(f"Shape: {pooled.shape}")
    print(f"Stations per season: {pooled.groupby('is_winter').size().to_dict()}")
    print("\nLog-target summary (mean +/- std):")
    for m in metals:
        col = f"log_{m}"
        print(f"  {m:3s}  log range: [{pooled[col].min():.2f}, {pooled[col].max():.2f}]  "
              f"raw range: [{pooled[m].min():.2f}, {pooled[m].max():.2f}]")


if __name__ == "__main__":
    main()
