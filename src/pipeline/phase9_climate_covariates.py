"""Extract climate covariates from GEE per station x season.

Covariates: 30-day and 7-day cumulative rainfall (CHIRPS) and 30-day mean
2m temperature (ERA5-Land), each computed over the window ending at the
representative sampling date for the season.

Sampling windows (manuscript: rainy = Jun-Sep, winter = Nov-Feb):
- Rainy:  end-date 2024-07-15 (peak monsoon)
- Winter: end-date 2024-01-15 (mid dry season)

Output: data/climate_covariates_v2.csv  (one row per station-season, 34 rows)
"""
from pathlib import Path
import time
import ee
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = DATA / "climate_covariates_v2.csv"

ee.Initialize(project="five-rivers-alphaearth")

WINDOWS = {
    "Rainy":  {"end": "2024-07-15"},
    "Winter": {"end": "2024-01-15"},
}


def end_minus(end_date_str, days):
    return ee.Date(end_date_str).advance(-days, "day")


def extract_one(lat, lon, end_date):
    pt = ee.Geometry.Point([lon, lat])
    rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
    temp = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")

    rain_30 = (rain.filterDate(end_minus(end_date, 30), end_date).sum()
               .reduceRegion(ee.Reducer.first(), pt, scale=5000)
               .get("precipitation"))
    rain_7 = (rain.filterDate(end_minus(end_date, 7), end_date).sum()
              .reduceRegion(ee.Reducer.first(), pt, scale=5000)
              .get("precipitation"))
    temp_30 = (temp.filterDate(end_minus(end_date, 30), end_date)
               .select("temperature_2m").mean()
               .reduceRegion(ee.Reducer.first(), pt, scale=10000)
               .get("temperature_2m"))
    temp_7 = (temp.filterDate(end_minus(end_date, 7), end_date)
              .select("temperature_2m").mean()
              .reduceRegion(ee.Reducer.first(), pt, scale=10000)
              .get("temperature_2m"))
    return ee.Dictionary({
        "rain_30d_mm": rain_30, "rain_7d_mm": rain_7,
        "temp_30d_K": temp_30, "temp_7d_K": temp_7,
    })


def main():
    pooled = pd.read_csv(DATA / "sediment_pooled_v2.csv")
    rows = []
    for i, r in pooled.iterrows():
        season = "Winter" if r.is_winter else "Rainy"
        end = WINDOWS[season]["end"]
        d = extract_one(r.Lat, r.Long, end).getInfo()
        rows.append({
            "station_id": r.station_id, "is_winter": int(r.is_winter),
            "season": season, "end_date": end,
            "rain_30d_mm": d.get("rain_30d_mm"),
            "rain_7d_mm":  d.get("rain_7d_mm"),
            "temp_30d_K":  d.get("temp_30d_K"),
            "temp_7d_K":   d.get("temp_7d_K"),
        })
        if i % 5 == 0:
            print(f"  [{i+1}/{len(pooled)}] {r.station_id} {season}  "
                  f"rain30={d.get('rain_30d_mm')}  temp30={d.get('temp_30d_K')}",
                  flush=True)
        time.sleep(0.3)

    out_df = pd.DataFrame(rows)
    out_df["temp_30d_C"] = out_df["temp_30d_K"] - 273.15
    out_df["temp_7d_C"] = out_df["temp_7d_K"] - 273.15
    out_df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")
    print(f"Shape: {out_df.shape}")
    print("\nSummary by season:")
    print(out_df.groupby("season")[["rain_30d_mm", "rain_7d_mm",
                                     "temp_30d_C", "temp_7d_C"]].mean().round(1))


if __name__ == "__main__":
    main()
