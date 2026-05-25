"""
Phase 4c — Re-extract AlphaEarth embeddings for the 117 primary stations from
current GEE V1, so primary and secondary use the same scale.

Primary sampling: 2024 (per manuscript).
AE source: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL, year=2024.

Output: data/sedimentRO_v2.csv  — same schema as sedimentRO.csv, with raw
(unscaled) AE_00..AE_63 columns from current V1. Chemistry, proximity, and
station IDs are copied through unchanged.

Same is done for sedimentWO.csv (winter sediment) and waterS.csv (water both seasons).
"""

from pathlib import Path
import time
import pandas as pd
import ee

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

AE_YEAR = 2024


def extract_ae(stations: pd.DataFrame, year: int) -> pd.DataFrame:
    col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

    rows = []
    for i, r in stations.iterrows():
        pt = ee.Geometry.Point([r["Long"], r["Lat"]])
        img = (
            col.filterBounds(pt)
               .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
               .first()
        )
        try:
            sample = img.sample(region=pt, scale=10, numPixels=1).first()
            d = sample.toDictionary().getInfo()
        except Exception as e:
            print(f"  ! station {r['Stations']}: {e}")
            d = {f"A{k:02d}": None for k in range(64)}

        out = {f"AE_{k:02d}": d.get(f"A{k:02d}") for k in range(64)}
        rows.append(out)

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(stations)}")
        time.sleep(0.05)

    return pd.DataFrame(rows, index=stations.index)


def reextract_one(in_path: Path, out_path: Path):
    df = pd.read_csv(in_path)
    print(f"\n=== {in_path.name} ({len(df)} stations) ===")

    ae_cols = [f"AE_{k:02d}" for k in range(64)]
    new_ae = extract_ae(df[["Stations", "Lat", "Long"]], AE_YEAR)

    out = df.copy()
    out[ae_cols] = new_ae[ae_cols].values
    out.to_csv(out_path, index=False)

    n_complete = out[ae_cols].notna().all(axis=1).sum()
    print(f"  {n_complete}/{len(out)} stations have complete AE")
    print(f"  Wrote {out_path.name}")
    print(f"  AE_00 range: {out['AE_00'].min():.4f} .. {out['AE_00'].max():.4f}  median={out['AE_00'].median():.4f}")


def main():
    ee.Initialize(project="five-rivers-alphaearth")
    print(f"Earth Engine initialised. Using AE year = {AE_YEAR}")

    # Re-extract for all four primary CSVs
    reextract_one(DATA / "sedimentRO.csv", DATA / "sedimentRO_v2.csv")
    reextract_one(DATA / "sedimentWO.csv", DATA / "sedimentWO_v2.csv")
    reextract_one(DATA / "waterS.csv",     DATA / "waterS_v2.csv")


if __name__ == "__main__":
    main()
