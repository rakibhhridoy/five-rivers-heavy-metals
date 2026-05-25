"""Phase 1.2 — Harmonized contamination indices (Strategy S4).

Recompute Igeo, CF, EF (Fe-normalized), PLI for ALL sediment stations
(primary 17 real + secondary geocoded literature) on a single background
(Turekian & Wedepohl 1961, average shale).

Outputs:
  analysis_results/enrichment/s4_harmonized_indices.csv
  analysis_results/enrichment/s4_index_deltas_vs_original.csv
  analysis_results/enrichment/s4_class_changes.csv
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SECONDARY = ROOT / "data/secondary"
PRIMARY = ROOT / "data"
OUT = ROOT / "results"

METALS = ["Cr", "Ni", "Cu", "As", "Cd", "Pb", "Fe"]
PRIMARY_METAL_COL = {"Cr": "Cr", "Ni": "Ni", "Cu": "Cu", "As": "As",
                     "Cd": "Cd", "Pb": "Pb", "Fe": "Fe"}

BACKGROUND = pd.read_csv(SECONDARY / "background_std.csv").set_index("Metal")["Background_mg_per_kg"].to_dict()


def igeo_class(igeo: float) -> str:
    """Müller (1969) classification."""
    if pd.isna(igeo):
        return "NA"
    if igeo <= 0:
        return "0_uncontaminated"
    if igeo <= 1:
        return "1_unconta_to_moderate"
    if igeo <= 2:
        return "2_moderate"
    if igeo <= 3:
        return "3_moderate_to_strong"
    if igeo <= 4:
        return "4_strong"
    if igeo <= 5:
        return "5_strong_to_extreme"
    return "6_extreme"


def cf_class(cf: float) -> str:
    """Hakanson (1980) CF classification."""
    if pd.isna(cf):
        return "NA"
    if cf < 1:
        return "low"
    if cf < 3:
        return "moderate"
    if cf < 6:
        return "considerable"
    return "very_high"


def ef_class(ef: float) -> str:
    """Sutherland (2000) EF classification."""
    if pd.isna(ef):
        return "NA"
    if ef < 2:
        return "minimal"
    if ef < 5:
        return "moderate"
    if ef < 20:
        return "significant"
    if ef < 40:
        return "very_high"
    return "extremely_high"


def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Wide df with metal columns named exactly METALS (Cr, Ni, ... Fe in mg/kg).
    Returns same df with Igeo_*, CF_*, EF_*, PLI columns added."""
    out = df.copy()
    cf_cols = []
    for m in METALS:
        Bn = BACKGROUND[m]
        c = pd.to_numeric(out[m], errors="coerce")
        out[f"Igeo_{m}"] = np.log2(c / (1.5 * Bn))
        out[f"CF_{m}"] = c / Bn
        cf_cols.append(f"CF_{m}")
    # PLI = geometric mean of CF over the metals where CF is available (per row)
    cf_arr = out[cf_cols].apply(pd.to_numeric, errors="coerce")
    out["PLI"] = cf_arr.apply(
        lambda r: np.exp(np.mean(np.log(r.dropna()))) if r.notna().sum() >= 4 else np.nan, axis=1
    )
    # EF: (C_metal / C_Fe)_sample / (B_metal / B_Fe), Fe is reference
    Fe_bg = BACKGROUND["Fe"]
    Fe_samp = pd.to_numeric(out["Fe"], errors="coerce")
    for m in METALS:
        if m == "Fe":
            out[f"EF_{m}"] = np.nan
            continue
        c = pd.to_numeric(out[m], errors="coerce")
        Bn = BACKGROUND[m]
        out[f"EF_{m}"] = (c / Fe_samp) / (Bn / Fe_bg)
    return out


def load_primary_sediment_real_17() -> pd.DataFrame:
    """Both seasons; one row per (station, season) with metal columns Cr..Fe in mg/kg."""
    rows = []
    for season, suffix, fn in [("Rainy", "R", "sedimentRO_v2.csv"),
                                ("Winter", "W", "sedimentWO_v2.csv")]:
        df = pd.read_csv(PRIMARY / fn)
        df = df[df["Stations"].str.match(r"^S\d+$")].copy()
        df = df[df["Stations"].str.extract(r"S(\d+)").astype(int).iloc[:, 0] <= 17]
        for _, r in df.iterrows():
            row = {
                "Source": "Primary",
                "Paper_ID": 0,
                "First_Author": "Hasan",
                "Year": 2024,
                "Station_ID": r["Stations"],
                "River": r["River"],
                "Season_Norm": season,
            }
            for m in METALS:
                col = f"{PRIMARY_METAL_COL[m]}{suffix}"
                row[m] = float(r[col]) if pd.notna(r[col]) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def load_secondary_sediment_wide() -> pd.DataFrame:
    """Pivot harmonized_sediment.csv (long) back to wide format for index computation."""
    long_df = pd.read_csv(SECONDARY / "harmonized_sediment.csv")
    # pivot
    idx_cols = ["Paper_ID", "First_Author", "Year", "River", "Station_ID",
                "Season_Norm", "Background_standard"]
    wide = long_df.pivot_table(
        index=idx_cols, columns="Metal", values="Concentration", aggfunc="first"
    ).reset_index()
    wide["Source"] = "Literature"
    # ensure all metal columns exist
    for m in METALS:
        if m not in wide.columns:
            wide[m] = np.nan
    # also keep originals for delta comparison
    orig_indices = (
        long_df.groupby(idx_cols + ["Metal"])
        .agg(Igeo_orig=("Igeo_orig", "first"),
             CF_orig=("CF_orig", "first"),
             EF_orig=("EF_orig", "first"))
        .reset_index()
    )
    return wide, orig_indices, long_df


def main() -> None:
    primary = load_primary_sediment_real_17()
    secondary, secondary_orig, secondary_long = load_secondary_sediment_wide()

    # Align columns
    keep = ["Source", "Paper_ID", "First_Author", "Year", "River", "Station_ID",
            "Season_Norm"] + METALS
    secondary_keep = secondary.copy()
    secondary_keep["Background_standard"] = secondary["Background_standard"]
    secondary_keep = secondary_keep[keep + ["Background_standard"]]
    primary_keep = primary[keep].copy()
    primary_keep["Background_standard"] = "(field measurement, no original index)"

    combined = pd.concat([primary_keep, secondary_keep], ignore_index=True)
    combined = compute_indices(combined)

    # Save full harmonized index table
    keep_out = (["Source", "Paper_ID", "First_Author", "Year", "River", "Station_ID",
                 "Season_Norm", "Background_standard"]
                + METALS
                + [f"Igeo_{m}" for m in METALS]
                + [f"CF_{m}" for m in METALS]
                + [f"EF_{m}" for m in METALS if m != "Fe"]
                + ["PLI"])
    combined[keep_out].to_csv(OUT / "s4_harmonized_indices.csv", index=False)

    # Delta vs original for secondary rows
    deltas = []
    class_changes = []
    sec_h = combined[combined.Source == "Literature"].copy()
    for _, r in sec_h.iterrows():
        for m in METALS:
            orig_match = secondary_orig[
                (secondary_orig.Paper_ID == r.Paper_ID)
                & (secondary_orig.Station_ID == r.Station_ID)
                & (secondary_orig.Season_Norm == r.Season_Norm)
                & (secondary_orig.Metal == m)
            ]
            if orig_match.empty:
                continue
            orig = orig_match.iloc[0]
            new_igeo = r[f"Igeo_{m}"]
            new_cf = r[f"CF_{m}"]
            new_ef = r[f"EF_{m}"] if m != "Fe" else np.nan
            deltas.append({
                "Paper_ID": r.Paper_ID, "Station": r.Station_ID, "Season": r.Season_Norm,
                "Metal": m, "Original_BG": r.Background_standard,
                "Igeo_orig": orig.Igeo_orig, "Igeo_new": new_igeo,
                "Igeo_delta": (new_igeo - orig.Igeo_orig) if pd.notna(orig.Igeo_orig) and pd.notna(new_igeo) else np.nan,
                "CF_orig": orig.CF_orig, "CF_new": new_cf,
                "CF_delta": (new_cf - orig.CF_orig) if pd.notna(orig.CF_orig) and pd.notna(new_cf) else np.nan,
                "EF_orig": orig.EF_orig, "EF_new": new_ef,
            })
            if pd.notna(orig.Igeo_orig) and pd.notna(new_igeo):
                old_c = igeo_class(orig.Igeo_orig)
                new_c = igeo_class(new_igeo)
                if old_c != new_c:
                    class_changes.append({
                        "Paper_ID": r.Paper_ID, "Station": r.Station_ID, "Season": r.Season_Norm,
                        "Metal": m, "Old_BG": r.Background_standard,
                        "Igeo_orig": round(orig.Igeo_orig, 2), "Old_class": old_c,
                        "Igeo_new_TW1961": round(new_igeo, 2), "New_class": new_c,
                    })
    pd.DataFrame(deltas).to_csv(OUT / "s4_index_deltas_vs_original.csv", index=False)
    pd.DataFrame(class_changes).to_csv(OUT / "s4_class_changes.csv", index=False)

    # Print summary
    print("=" * 72)
    print("PHASE 1.2 — HARMONIZED CONTAMINATION INDICES")
    print("=" * 72)
    print(f"Combined stations with index calc: {len(combined)}")
    print(f"  Primary (n=17 × 2 seasons):    {(combined.Source == 'Primary').sum()}")
    print(f"  Literature:                     {(combined.Source == 'Literature').sum()}")

    print(f"\nIndex deltas saved: {len(deltas)} (metal-station rows)")
    n_with_orig = sum(1 for d in deltas if pd.notna(d["Igeo_orig"]))
    print(f"  With original Igeo for comparison: {n_with_orig}")

    print(f"\nContamination-class changes when switched to T&W 1961: {len(class_changes)}")
    if class_changes:
        cc = pd.DataFrame(class_changes)
        print("\nFirst 15 class changes:")
        print(cc.head(15).to_string(index=False))

    # Per-metal summary across the harmonized dataset
    print("\nPer-metal Igeo class distribution (combined dataset):")
    for m in METALS:
        vc = combined[f"Igeo_{m}"].apply(igeo_class).value_counts().to_dict()
        print(f"  {m}: {vc}")

    print(f"\nOutputs in {OUT}/")
    for f in ["s4_harmonized_indices.csv", "s4_index_deltas_vs_original.csv", "s4_class_changes.csv"]:
        p = OUT / f
        print(f"  {f}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
