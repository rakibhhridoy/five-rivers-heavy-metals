"""Compute the 11 spectral indices from Sentinel-2 bands (Methodology §5).

Closes the reusable part of Gap 1 (docs/AUDIT_COVERAGE.md). The full LULC-change
workflow depends on the 3.4 GB raster tree archived on Zenodo (see src/gis/README_GIS.md);
this script is the self-contained, documented index-calculation step ported from
gis/GIS.ipynb (cell 32).

Indices (manuscript Methodology §5): NDVI, EVI, SAVI, NDWI, MNDWI, AWEI, NDBI,
UI, BUI, NDSI, LST.

Sentinel-2 band convention (after scale factor 1e-4 applied to L2A DN):
  B02 Blue (10 m), B03 Green (10 m), B04 Red (10 m), B08 NIR (10 m),
  B11 SWIR1 (20 m, resampled to 10 m), B12 SWIR2 (20 m, resampled to 10 m).
LST is derived from Landsat 8/9 thermal band (B10) — see compute_lst().

Usage
-----
    python calculate_spectral_indices.py --bands /path/to/sentinel_band_dir --out /path/to/indices_out

Requires: rasterio, numpy (install separately; not in the core requirements.txt
because the index step is GIS-stack-heavy and optional for tabular reproduction).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
except ImportError:  # pragma: no cover
    raise SystemExit(
        "rasterio is required for spectral-index calculation. "
        "Install with: pip install rasterio"
    )

L_SAVI = 0.5  # soil-adjustment factor


def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Element-wise division returning NaN where the denominator is zero."""
    out = np.full_like(num, np.nan, dtype=np.float32)
    np.divide(num, den, out=out, where=(den != 0))
    return out


def compute_indices(B2, B3, B4, B8, B11, B12) -> dict:
    """Return a dict of the 10 reflectance-based indices.

    LST (the 11th) needs a thermal band and is handled by compute_lst().
    All inputs are surface-reflectance arrays (float32, scaled to [0, 1]).
    """
    idx = {}
    idx["NDVI"] = safe_divide(B8 - B4, B8 + B4)
    idx["EVI"] = np.clip(2.5 * safe_divide(B8 - B4, B8 + 6 * B4 - 7.5 * B2 + 1), -1, 1)
    idx["SAVI"] = (1 + L_SAVI) * safe_divide(B8 - B4, B8 + B4 + L_SAVI)
    idx["NDWI"] = safe_divide(B3 - B8, B3 + B8)        # McFeeters
    idx["MNDWI"] = safe_divide(B3 - B11, B3 + B11)     # Xu
    idx["NDBI"] = safe_divide(B11 - B8, B11 + B8)
    idx["UI"] = safe_divide(B12 - B8, B12 + B8)
    idx["BUI"] = np.clip(idx["NDBI"] - idx["NDVI"], -1, 1)
    idx["NDSI"] = safe_divide(B3 - B11, B3 + B11)
    idx["AWEI"] = np.clip(4 * (B3 - B11) - (0.25 * B8 + 2.75 * B12), -1, 1)
    return idx


def compute_lst(thermal_band: np.ndarray, ndvi: np.ndarray,
                ml: float = 3.42e-4, al: float = 0.1,
                k1: float = 774.89, k2: float = 1321.08) -> np.ndarray:
    """Land Surface Temperature from a Landsat 8/9 thermal band (B10).

    ml/al: radiance multiplicative/additive rescaling (band-specific, from MTL).
    k1/k2: thermal conversion constants (band-specific, from MTL).
    Emissivity is estimated from NDVI via the proportion-of-vegetation method.
    Returns LST in degrees Celsius.
    """
    radiance = ml * thermal_band + al
    bt = k2 / np.log((k1 / np.clip(radiance, 1e-6, None)) + 1) - 273.15  # brightness temp (°C)
    # NDVI-based emissivity
    ndvi_min, ndvi_max = np.nanmin(ndvi), np.nanmax(ndvi)
    pv = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min)) ** 2
    emissivity = 0.004 * pv + 0.986
    lambda_ = 10.895e-6  # B10 central wavelength (m)
    rho = 1.438e-2       # h*c/sigma (m·K)
    lst = bt / (1 + (lambda_ * bt / rho) * np.log(np.clip(emissivity, 1e-6, None)))
    return lst


def _load(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32) * 1e-4, src.profile


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bands", required=True, type=Path,
                    help="Directory with B02/B03/B04/B08/B11/B12 GeoTIFFs or JP2s")
    ap.add_argument("--out", required=True, type=Path, help="Output directory for index rasters")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    def find(band: str) -> Path:
        hits = list(args.bands.glob(f"*{band}*"))
        if not hits:
            raise SystemExit(f"Band {band} not found under {args.bands}")
        return hits[0]

    B2, prof = _load(find("B02"))
    B3, _ = _load(find("B03"))
    B4, _ = _load(find("B04"))
    B8, _ = _load(find("B08"))
    B11, _ = _load(find("B11"))
    B12, _ = _load(find("B12"))

    indices = compute_indices(B2, B3, B4, B8, B11, B12)
    prof.update(dtype="float32", count=1, nodata=np.nan)
    for name, arr in indices.items():
        out_path = args.out / f"{name.lower()}.tif"
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(arr.astype(np.float32), 1)
        print(f"  wrote {out_path.name}")
    print(f"\nWrote {len(indices)} indices to {args.out}. "
          "LST requires a Landsat thermal band — see compute_lst() in this module.")


if __name__ == "__main__":
    main()
