"""
Phase 4b — Extract AlphaEarth embeddings for secondary stations.

Uses Google Earth Engine `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (64-dim, A00..A63).
For each secondary station, queries the embedding for the paper's sampling year
(or the closest available year, since AE coverage is 2017-2025).

Output: SecondaryData/secondary_ae_features.csv
        Stations | Lat | Lon | AE_00 ... AE_63
"""

from pathlib import Path
import time
import pandas as pd
import ee

ROOT = Path(__file__).resolve().parents[2]
SECONDARY = ROOT / "data/secondary"
OUT = SECONDARY / "secondary_ae_features.csv"

# Map paper year to AE year (AE collection covers 2017-2025).
# For pre-2017 papers we use the earliest available year (2017).
PAPER_AE_YEAR = {
    ("Bhuiyan", 2015): 2017,   # paper sampled 2010 — use earliest AE
    ("Hossain", 2021): 2019,   # paper sampled 2018-2019
    ("Islam",   2021): 2021,   # paper sampled 2020-2021
    ("Jolly",   2023): 2020,   # paper sampled March/Oct 2020
    ("Akter",   2025): 2022,   # paper sampled Oct 2021 - Sep 2022
    ("Mohanta", 2019): 2019,   # paper sampled Jul 2018 - Jun 2019
}


def main():
    ee.Initialize(project="five-rivers-alphaearth")
    print("Earth Engine initialised.")

    stations = pd.read_csv(SECONDARY / "secondary_proximity_features.csv")
    col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

    rows = []
    for i, r in stations.iterrows():
        ae_year = PAPER_AE_YEAR[(r["First_Author"], r["Year"])]
        pt = ee.Geometry.Point([r["Lon"], r["Lat"]])

        img = (
            col.filterBounds(pt)
               .filterDate(f"{ae_year}-01-01", f"{ae_year + 1}-01-01")
               .first()
        )

        try:
            sample = img.sample(region=pt, scale=10, numPixels=1).first()
            d = sample.toDictionary().getInfo()
        except Exception as e:
            print(f"  ! station {r['Station_ID']} ({r['First_Author']} {r['Year']}, AE={ae_year}): {e}")
            d = {f"A{i:02d}": None for i in range(64)}

        out_row = {"Station_ID": r["Station_ID"], "Lat": r["Lat"], "Lon": r["Lon"], "AE_year": ae_year}
        for k in range(64):
            out_row[f"AE_{k:02d}"] = d.get(f"A{k:02d}")
        rows.append(out_row)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(stations)} stations done")
        time.sleep(0.05)  # gentle on the API

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    n_complete = out[[f"AE_{k:02d}" for k in range(64)]].notna().all(axis=1).sum()
    print(f"\nWrote {len(out)} stations to {OUT}")
    print(f"  Complete (all 64 AE dims non-null): {n_complete}/{len(out)}")
    print(f"  AE_00 sample range: {out['AE_00'].min():.4f} .. {out['AE_00'].max():.4f}")


if __name__ == "__main__":
    main()
