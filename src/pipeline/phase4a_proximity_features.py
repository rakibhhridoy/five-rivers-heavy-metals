"""
Phase 4a — Compute proximity features for secondary stations.

Reproduces the four primary-data proximity columns:
  hydro_dist_brick — Euclidean distance to nearest brick kiln (degrees, matches primary scale)
  num_brick_field  — count of brick kilns within 1000 m
  hydro_dist_ind   — Euclidean distance to nearest industry (degrees)
  num_industry     — count of industries within 1000 m

The primary data computes hydro_dist in DEM-pixel units × resolution. The DEM
transform is in geographic coords (~0.000278° per pixel ≈ 30 m), so primary
hydro_dist values are in DEGREES. We reproduce that here by working directly
in EPSG:4326 with euclidean distance, and project to UTM only for the 1000 m
buffer count.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
SECONDARY = ROOT / "data/secondary"
GIS = ROOT / "gis"
OUT = SECONDARY / "secondary_proximity_features.csv"

UTM_BANGLADESH = "EPSG:32646"  # UTM Zone 46N — covers Dhaka


def main():
    stations = pd.read_csv(SECONDARY / "stations_geocoded.csv").dropna(subset=["Lat", "Lon"])
    brick = gpd.read_file(GIS / "brick_field_point.shp").to_crs("EPSG:4326").explode(index_parts=False)
    ind   = gpd.read_file(GIS / "industry_point.shp").to_crs("EPSG:4326").explode(index_parts=False)

    # Stations as gdf
    stations_gdf = gpd.GeoDataFrame(
        stations,
        geometry=[Point(xy) for xy in zip(stations["Lon"], stations["Lat"])],
        crs="EPSG:4326",
    )

    # --- nearest distance in degrees (matches primary scale) ---
    brick_xy = np.column_stack([brick.geometry.x.values, brick.geometry.y.values])
    ind_xy   = np.column_stack([ind.geometry.x.values, ind.geometry.y.values])
    brick_tree = cKDTree(brick_xy)
    ind_tree   = cKDTree(ind_xy)
    s_xy = np.column_stack([stations_gdf.geometry.x.values, stations_gdf.geometry.y.values])

    stations["hydro_dist_brick"], _ = brick_tree.query(s_xy)
    stations["hydro_dist_ind"],   _ = ind_tree.query(s_xy)

    # --- counts within 1000 m buffer (project to UTM for accurate metric buffer) ---
    s_utm     = stations_gdf.to_crs(UTM_BANGLADESH)
    brick_utm = brick.to_crs(UTM_BANGLADESH)
    ind_utm   = ind.to_crs(UTM_BANGLADESH)

    s_utm["geometry"] = s_utm.geometry.buffer(1000.0)

    sjoin_b = gpd.sjoin(s_utm, brick_utm, how="left", predicate="intersects")
    matched_b = sjoin_b[sjoin_b["index_right"].notna()]
    cnt_b = matched_b.groupby(matched_b.index).size()
    stations["num_brick_field"] = stations.index.map(cnt_b).fillna(0).astype(int)

    sjoin_i = gpd.sjoin(s_utm, ind_utm, how="left", predicate="intersects")
    matched_i = sjoin_i[sjoin_i["index_right"].notna()]
    cnt_i = matched_i.groupby(matched_i.index).size()
    stations["num_industry"] = stations.index.map(cnt_i).fillna(0).astype(int)

    out = stations[[
        "Paper_ID", "First_Author", "Year", "River", "Station_ID", "Station_Name",
        "Lat", "Lon", "Geocode_Tier",
        "hydro_dist_brick", "num_brick_field", "hydro_dist_ind", "num_industry",
    ]]
    out.to_csv(OUT, index=False)

    print(f"Wrote {len(out)} stations to {OUT}\n")
    print("Sanity vs primary-data ranges:")
    prim = pd.read_csv(ROOT / "data/sedimentRO_v2.csv")
    for col in ["hydro_dist_brick", "num_brick_field", "hydro_dist_ind", "num_industry"]:
        p = prim[col].describe()
        s = out[col].describe()
        print(f"  {col}:")
        print(f"    primary   min={p['min']:.4f}  median={p['50%']:.4f}  max={p['max']:.4f}")
        print(f"    secondary min={s['min']:.4f}  median={s['50%']:.4f}  max={s['max']:.4f}")


if __name__ == "__main__":
    main()
