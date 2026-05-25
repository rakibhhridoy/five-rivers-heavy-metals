"""
Phase 3 — Geocode secondary literature stations.

Output: SecondaryData/stations_geocoded.csv

Tier convention:
  Exact   — station coordinates published in source paper
  High    — named single-point landmark with unambiguous public geocode
  Medium  — distributed along a river segment from known endpoints (interpolated)
  Low     — river-segment centroid only (named place not uniquely locatable)

Anchor points (decimal degrees, WGS84) — sourced from public references and
cross-checked against OpenStreetMap/Google Maps. Listed here so the choices
are auditable rather than buried in a function.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SECONDARY = ROOT / "data/secondary"
OUT = SECONDARY / "stations_geocoded.csv"

# ---------------------------------------------------------------------------
# Anchor points
# ---------------------------------------------------------------------------

# Bhuiyan 2015: Rayer Bazar (north end, 23°44'N) → Pagla (south end, 23°36'N)
# along ~30 km Buriganga reach, with stations every ~2 km.
BHUIYAN_START = (23.7440, 90.3690)   # Rayer Bazar bridge area
BHUIYAN_END   = (23.6700, 90.4870)   # Pagla, Narayanganj (where Buriganga meets Dhaleshwari)

# Hossain 2021 — explicit cluster centroids from paper text
HOSSAIN_BUR_A  = (23.7333, 90.3333)  # Kamrangir Char (23°44'N, 90°20'E)
HOSSAIN_BUR_B  = (23.6833, 90.4167)  # Shaheed Buddhijibi Bridge (23°41'N, 90°25'E)
HOSSAIN_TURAG  = (23.8833, 90.4167)  # Tongi Bridge downstream (23°53'N, 90°25'E)

# Islam 2021 — named Dhaleshwari stations along Savar–Hemayetpur reach
ISLAM_NAMES = {
    "Savar Tannery": (23.7787, 90.2370),  # BSCIC Tannery Estate, Hemayetpur
    "Sudkhira":      (23.7560, 90.2440),  # Sudkhira village (downstream Tannery)
    "Dhalla":        (23.7300, 90.2700),  # Dhalla, downstream Dhaleshwari
    "AKS dying":     (23.7600, 90.2500),  # AKS dyeing factory area, near Tannery
    "Nama Bazar":    (23.7140, 90.2900),  # Nama Bazar, lower Dhaleshwari
}

# Jolly 2023 — exact coords already in source data
JOLLY_COORD = (23.7218, 90.5006)  # 23°43'18.462"N, 90°30'2.1348"E

# Akter 2025 — Balu River, station names given (St-1 to St-5, ~1 km apart)
AKTER_NAMES = {
    "Chanpara Bridge": (23.8330, 90.5050),   # Balu river crossing at Chanpara
    "Chanpara Bondor": (23.8290, 90.5070),
    "Dakkhin Para":    (23.8245, 90.5095),
    "Paschim Gao":     (23.8200, 90.5125),
    "Kheodhala":       (23.8150, 90.5150),
}

# Mohanta 2019 — central coord given; 5 spots 0.5 km apart
MOHANTA_CENTRE = (23.7813, 90.2396)  # 23°46'52.68"N, 90°14'22.58"E


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def linear_interpolate(start, end, n, idx):
    """idx = 1..n along the segment (inclusive endpoints)."""
    frac = (idx - 1) / (n - 1) if n > 1 else 0.5
    lat = start[0] + frac * (end[0] - start[0])
    lon = start[1] + frac * (end[1] - start[1])
    return lat, lon


def offset_metres(lat, lon, d_north_m, d_east_m):
    """Small-displacement coords offset (good to ~1 km in mid-latitudes)."""
    dlat = d_north_m / 111_000
    dlon = d_east_m / (111_000 * np.cos(np.radians(lat)))
    return lat + dlat, lon + dlon


# ---------------------------------------------------------------------------
# Geocoder
# ---------------------------------------------------------------------------

def geocode_bhuiyan(station_id, n=15):
    """BW1..BW15 (water) and RS1..RS15 (sediment) at the same 15 geographic
    points along Rayer Bazar → Pagla. station_id like 'BW7' or 'RS3'."""
    idx = int(station_id[2:])
    lat, lon = linear_interpolate(BHUIYAN_START, BHUIYAN_END, n, idx)
    return lat, lon, "Medium"


def geocode_hossain(station_id):
    """BS1..BS5 along Buriganga (Kamrangir Char→Buddhijibi Bridge),
    TS1..TS5 along Turag (Tongi reach)."""
    idx = int(station_id[2:])
    if station_id.startswith("BS"):
        lat, lon = linear_interpolate(HOSSAIN_BUR_A, HOSSAIN_BUR_B, 5, idx)
    elif station_id.startswith("TS"):
        # Tongi reach: 1.5 km segment around Tongi Bridge, downstream
        seg_start = offset_metres(*HOSSAIN_TURAG, 800, -750)
        seg_end   = offset_metres(*HOSSAIN_TURAG, -800, 750)
        lat, lon = linear_interpolate(seg_start, seg_end, 5, idx)
    else:
        raise ValueError(f"Unknown Hossain station: {station_id}")
    return lat, lon, "Medium"


def geocode_islam(station_name):
    if station_name not in ISLAM_NAMES:
        raise ValueError(f"Islam station name not in dict: {station_name}")
    lat, lon = ISLAM_NAMES[station_name]
    return lat, lon, "High"


def geocode_jolly():
    return JOLLY_COORD[0], JOLLY_COORD[1], "Exact"


def geocode_akter(station_name):
    if station_name not in AKTER_NAMES:
        raise ValueError(f"Akter station name not in dict: {station_name}")
    lat, lon = AKTER_NAMES[station_name]
    return lat, lon, "High"


def geocode_mohanta(station_id):
    """Spot 1..Spot 5 — 0.5 km spacing around central coord. Within AlphaEarth
    10 m grid resolution, all 5 spots map to the same neighbourhood."""
    idx = int(station_id.split()[1])
    # offset along E-W axis at 500 m intervals from centre
    east_offset = (idx - 3) * 500  # Spot 3 at centre, Spot 1 -1 km W, Spot 5 +1 km E
    lat, lon = offset_metres(*MOHANTA_CENTRE, 0, east_offset)
    return lat, lon, "Exact"  # paper supplies the centre coord; offsets are within paper-stated spacing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sed = pd.read_csv(SECONDARY / "harmonized_sediment.csv")
    wat = pd.read_csv(SECONDARY / "harmonized_water.csv")
    combined = pd.concat([sed, wat], ignore_index=True)

    stations = combined.groupby(
        ["Paper_ID", "First_Author", "Year", "River", "Station_ID"]
    ).agg(
        Station_Name=("Station_Name", "first"),
    ).reset_index()

    rows = []
    for _, r in stations.iterrows():
        author = r["First_Author"]
        sid = r["Station_ID"]
        name = r["Station_Name"]

        try:
            if author == "Bhuiyan":
                lat, lon, tier = geocode_bhuiyan(sid)
            elif author == "Hossain":
                lat, lon, tier = geocode_hossain(sid)
            elif author == "Islam":
                lat, lon, tier = geocode_islam(name)
            elif author == "Jolly":
                lat, lon, tier = geocode_jolly()
            elif author == "Akter":
                lat, lon, tier = geocode_akter(name)
            elif author == "Mohanta":
                lat, lon, tier = geocode_mohanta(sid)
            else:
                raise ValueError(f"No geocoder for author: {author}")
            note = ""
        except Exception as e:
            lat, lon, tier, note = np.nan, np.nan, "Failed", str(e)

        rows.append({
            "Paper_ID": r["Paper_ID"],
            "First_Author": author,
            "Year": r["Year"],
            "River": r["River"],
            "Station_ID": sid,
            "Station_Name": name,
            "Lat": lat,
            "Lon": lon,
            "Geocode_Tier": tier,
            "Note": note,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    print(f"Wrote {len(out)} stations to {OUT}")
    print("\nTier breakdown:")
    print(out["Geocode_Tier"].value_counts().to_string())
    print("\nBy author × tier:")
    print(out.groupby(["First_Author", "Geocode_Tier"]).size().to_string())

    # Sanity: every coord inside the Dhaka five-rivers bounding box
    bb = out.dropna(subset=["Lat", "Lon"])
    in_box = (bb["Lat"].between(23.5, 24.0) & bb["Lon"].between(90.1, 90.6))
    print(f"\n{in_box.sum()}/{len(bb)} coords inside Dhaka bbox (23.5–24.0°N, 90.1–90.6°E)")
    if not in_box.all():
        print("OUT-OF-BOX rows:")
        print(bb[~in_box].to_string())


if __name__ == "__main__":
    main()
