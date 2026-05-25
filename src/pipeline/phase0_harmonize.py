"""Phase 0 — Data hygiene for SecondaryData/MAIN DATA.csv.

Produces:
  SecondaryData/audit_report.txt
  SecondaryData/season_map.csv
  SecondaryData/bdl_substitutions.csv
  SecondaryData/background_std.csv
  SecondaryData/harmonized_water.csv   (long format: one row per station × metal)
  SecondaryData/harmonized_sediment.csv

Run from project root:
  python enrichment_scripts/phase0_harmonize.py
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/secondary" / "MAIN DATA.csv"
OUT = ROOT / "data/secondary"

METALS = ["Cr", "Ni", "Cu", "As", "Cd", "Pb", "Fe"]

# Turekian & Wedepohl (1961) average shale background (mg/kg)
BACKGROUND_TW1961 = {
    "Cr": 90.0,
    "Ni": 68.0,
    "Cu": 45.0,
    "As": 13.0,
    "Cd": 0.3,
    "Pb": 20.0,
    "Fe": 47200.0,
}

SEASON_MAP = {
    "Wet": "Rainy",
    "Monsoon": "Rainy",
    "Pre-monsoon": "Rainy",
    "Rainy": "Rainy",
    "Summer": "Rainy",
    "Dry": "Winter",
    "Post-monsoon": "Winter",
    "Winter": "Winter",
}


def audit(df: pd.DataFrame, lines: list[str]) -> None:
    lines.append("=" * 72)
    lines.append("PHASE 0.1 — LOAD & AUDIT")
    lines.append("=" * 72)
    lines.append(f"Source: {SRC}")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Columns: {len(df.columns)}")
    lines.append("")

    lines.append("Papers present (ID / Author / Year):")
    papers = df[["Paper_ID", "First_Author", "Year"]].drop_duplicates().sort_values("Paper_ID")
    for _, r in papers.iterrows():
        lines.append(f"  {r.Paper_ID:>3}  {r.First_Author:<12}  {r.Year}")

    ids = sorted(df["Paper_ID"].unique())
    missing_ids = [i for i in range(min(ids), max(ids) + 1) if i not in ids]
    lines.append(f"\nPaper ID sequence: {ids}")
    lines.append(f"Missing IDs in range: {missing_ids or 'none'}")

    lines.append("\nRows per (paper, river):")
    counts = df.groupby(["Paper_ID", "First_Author", "River"]).size().reset_index(name="n")
    for _, r in counts.iterrows():
        lines.append(f"  P{r.Paper_ID} {r.First_Author:<12} {r.River:<14} n={r.n}")

    lines.append("\nMatrix × Season:")
    lines.append(df.groupby(["Matrix", "Season"]).size().to_string())

    lines.append("\nAccess_status:")
    lines.append(df["Access_status"].value_counts().to_string())

    lines.append("\nAnalytical_method:")
    lines.append(df["Analytical_method"].value_counts().to_string())

    lines.append("\nBackground_standard:")
    lines.append(df["Background_standard"].value_counts().to_string())

    lines.append("\nUnit distribution:")
    lines.append(df["Unit"].value_counts().to_string())

    lines.append("\nNaN counts — metal columns:")
    metal_cols = [f"{m}_conc" for m in METALS]
    for c in metal_cols:
        lines.append(f"  {c:<10} NaN={df[c].isna().sum():>4}  present={df[c].notna().sum():>4}")

    lines.append("\nNaN counts — coord columns:")
    for c in ["Lat", "Lon"]:
        lines.append(f"  {c:<10} NaN={df[c].isna().sum():>4}  present={df[c].notna().sum():>4}")


def normalize_seasons(df: pd.DataFrame) -> pd.DataFrame:
    df["Season_Norm"] = df["Season"].map(SEASON_MAP)
    # NaN or unmapped stay NaN
    mapping_log = (
        df.groupby("Season")["Season_Norm"]
        .agg(lambda s: s.dropna().unique().tolist() if s.dropna().any() else [])
        .reset_index()
        .rename(columns={"Season_Norm": "Mapped_To"})
    )
    mapping_log["Mapped_To"] = mapping_log["Mapped_To"].apply(
        lambda xs: xs[0] if xs else "NaN (excluded from seasonal analyses)"
    )
    mapping_log["Count"] = df.groupby("Season").size().values
    mapping_log.to_csv(OUT / "season_map.csv", index=False)
    return df


def normalize_units(df: pd.DataFrame, lines: list[str]) -> pd.DataFrame:
    """Target: water=µg/L, sediment=mg/kg. Flag and convert where possible."""
    df["Unit_Norm"] = df["Unit"]
    conversions = []
    for i, row in df.iterrows():
        unit = str(row["Unit"]).strip() if pd.notna(row["Unit"]) else ""
        mat = row["Matrix"]
        if mat == "Water":
            if unit.lower() == "mg/l":
                for m in METALS:
                    col = f"{m}_conc"
                    if pd.notna(df.at[i, col]):
                        df.at[i, col] = float(df.at[i, col]) * 1000.0  # mg/L → µg/L
                df.at[i, "Unit_Norm"] = "µg/L"
                conversions.append((i, mat, unit, "µg/L", "×1000"))
            elif unit.replace("μ", "µ").lower() == "µg/l":
                df.at[i, "Unit_Norm"] = "µg/L"
            else:
                conversions.append((i, mat, unit, "CHECK", "unrecognized"))
        elif mat == "Sediment":
            if unit.lower() == "µg/g" or unit.replace("μ", "µ").lower() == "µg/g":
                # 1 µg/g == 1 mg/kg, no numeric change
                df.at[i, "Unit_Norm"] = "mg/kg"
                conversions.append((i, mat, unit, "mg/kg", "×1 (1 µg/g = 1 mg/kg)"))
            elif unit.lower() == "mg/kg":
                df.at[i, "Unit_Norm"] = "mg/kg"
            else:
                conversions.append((i, mat, unit, "CHECK", "unrecognized"))

    lines.append("\n" + "=" * 72)
    lines.append("PHASE 0.3 — UNIT NORMALIZATION")
    lines.append("=" * 72)
    if conversions:
        lines.append(f"{len(conversions)} row-level actions:")
        for row_idx, mat, orig, norm, note in conversions[:30]:
            lines.append(f"  row {row_idx:>3} {mat:<8} '{orig}' -> '{norm}' ({note})")
        if len(conversions) > 30:
            lines.append(f"  ... and {len(conversions) - 30} more")
    else:
        lines.append("All units already in target form (µg/L water, mg/kg sediment).")
    lines.append("\nAfter normalization — Unit_Norm × Matrix:")
    lines.append(df.groupby(["Matrix", "Unit_Norm"]).size().to_string())
    return df


def handle_bdl(df: pd.DataFrame, lines: list[str]) -> pd.DataFrame:
    """Find any non-numeric entries in metal columns; log."""
    metal_cols = [f"{m}_conc" for m in METALS]
    bdl_rows = []
    for c in metal_cols:
        coerced = pd.to_numeric(df[c], errors="coerce")
        # Find entries that WERE non-null but became null after coercion (= strings)
        was_str = df[c].notna() & coerced.isna() & (df[c] != "NaN")
        for i in df.index[was_str]:
            bdl_rows.append((i, c, df.at[i, c], "coerced to NaN"))
        # Also handle the literal string 'NaN' which should be NaN
        df[c] = coerced
    bdl_df = pd.DataFrame(bdl_rows, columns=["row", "column", "original", "action"])
    bdl_df.to_csv(OUT / "bdl_substitutions.csv", index=False)

    lines.append("\n" + "=" * 72)
    lines.append("PHASE 0.4 — BDL / NON-NUMERIC HANDLING")
    lines.append("=" * 72)
    if bdl_df.empty:
        lines.append("No non-numeric concentration entries detected in MAIN DATA.csv.")
        lines.append("(Uddin & Jeong review sheet, if later merged, contains BDL strings.)")
    else:
        lines.append(f"{len(bdl_df)} non-numeric entries in metal columns:")
        for _, r in bdl_df.iterrows():
            lines.append(f"  row {r.row:>3} col {r.column:<10} '{r.original}' -> NaN")
    return df


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Wide → long: one row per (paper, station, season, matrix, metal)."""
    id_cols = [
        "Paper_ID", "First_Author", "Year", "River", "Station_ID", "Station_Name",
        "Lat", "Lon", "Season", "Season_Norm", "Matrix", "Unit_Norm",
        "Background_standard", "Analytical_method", "Access_status",
        "WHO_exceedance", "GNN_usability",
    ]
    records = []
    for _, r in df.iterrows():
        for m in METALS:
            conc = r[f"{m}_conc"]
            records.append({
                **{c: r[c] for c in id_cols},
                "Metal": m,
                "Concentration": conc,
                "Igeo_orig": r.get(f"Igeo_{m}", np.nan),
                "EF_orig": r.get(f"EF_{m}", np.nan),
                "CF_orig": r.get(f"CF_{m}", np.nan),
            })
    return pd.DataFrame(records)


def write_background(lines: list[str]) -> None:
    bg = pd.DataFrame(
        [{"Metal": m, "Background_mg_per_kg": v, "Source": "Turekian & Wedepohl 1961 (average shale)"}
         for m, v in BACKGROUND_TW1961.items()]
    )
    bg.to_csv(OUT / "background_std.csv", index=False)
    lines.append("\n" + "=" * 72)
    lines.append("PHASE 0.6 — BACKGROUND STANDARD DECISION")
    lines.append("=" * 72)
    lines.append("Chosen: Turekian & Wedepohl 1961 (average shale composition).")
    lines.append("Rationale: most widely cited global sediment background; already used by")
    lines.append("Hossain 2021 in our compilation; enables direct comparison with prior lit.")
    lines.append("\nValues (mg/kg):")
    lines.append(bg.to_string(index=False))


def main() -> None:
    lines: list[str] = []
    df = pd.read_csv(SRC, na_values=["NaN", "nan", "NA", ""], keep_default_na=True)

    audit(df, lines)

    # 0.2 seasons
    df = normalize_seasons(df)
    lines.append("\n" + "=" * 72)
    lines.append("PHASE 0.2 — SEASON NORMALIZATION")
    lines.append("=" * 72)
    lines.append("Mapping written to SecondaryData/season_map.csv")
    lines.append(df.groupby(["Season", "Season_Norm"], dropna=False).size().to_string())

    # 0.3 units (must come BEFORE BDL since we mutate numeric columns)
    df = normalize_units(df, lines)

    # 0.4 BDL
    df = handle_bdl(df, lines)

    # 0.5 matrix split + long format
    long_df = to_long(df)

    water = long_df[long_df["Matrix"] == "Water"].copy()
    sediment = long_df[long_df["Matrix"] == "Sediment"].copy()

    water.to_csv(OUT / "harmonized_water.csv", index=False)
    sediment.to_csv(OUT / "harmonized_sediment.csv", index=False)

    lines.append("\n" + "=" * 72)
    lines.append("PHASE 0.5 — MATRIX SEPARATION (LONG FORMAT)")
    lines.append("=" * 72)
    lines.append(f"harmonized_water.csv:    {len(water):>5} rows  ({water['Concentration'].notna().sum()} with value)")
    lines.append(f"harmonized_sediment.csv: {len(sediment):>5} rows  ({sediment['Concentration'].notna().sum()} with value)")

    # 0.6 background
    write_background(lines)

    # Gate G0 summary
    lines.append("\n" + "=" * 72)
    lines.append("GATE G0 — PHASE 0 EXIT CHECKS")
    lines.append("=" * 72)
    checks = [
        ("harmonized_water.csv exists", (OUT / "harmonized_water.csv").exists()),
        ("harmonized_sediment.csv exists", (OUT / "harmonized_sediment.csv").exists()),
        ("season_map.csv exists", (OUT / "season_map.csv").exists()),
        ("background_std.csv exists", (OUT / "background_std.csv").exists()),
        ("all water rows Unit_Norm=µg/L",
            (water["Unit_Norm"].dropna() == "µg/L").all() if not water.empty else False),
        ("all sediment rows Unit_Norm=mg/kg",
            (sediment["Unit_Norm"].dropna() == "mg/kg").all() if not sediment.empty else False),
        ("Season_Norm ⊆ {Rainy, Winter, NaN}",
            set(df["Season_Norm"].dropna().unique()).issubset({"Rainy", "Winter"})),
    ]
    for name, ok in checks:
        lines.append(f"  [{'✓' if ok else '✗'}] {name}")
    lines.append("")
    lines.append("GATE G0: " + ("PASS" if all(ok for _, ok in checks) else "FAIL"))

    report_path = OUT / "audit_report.txt"
    report_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
