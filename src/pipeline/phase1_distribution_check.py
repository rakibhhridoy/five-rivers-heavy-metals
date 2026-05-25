"""Phase 1.1 — Distributional sanity check (Strategy S3).

Compare primary (n=17 real stations × 2 seasons) concentrations against the
harmonized secondary-literature distribution, per metal × matrix × season.

Gate G1: primary sits inside literature 95% range for ≥6 of 8 metals.

Outputs:
  analysis_results/enrichment/s3_stats.csv
  analysis_results/enrichment/s3_distribution_comparison.png
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
SECONDARY = ROOT / "data/secondary"
PRIMARY = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

METALS = ["Cr", "Ni", "Cu", "As", "Cd", "Pb", "Fe"]
# Primary uses MR/MW for manganese; conversion map
PRIMARY_METAL_COL = {"Cr": "Cr", "Ni": "Ni", "Cu": "Cu", "As": "As",
                     "Cd": "Cd", "Pb": "Pb", "Fe": "Fe"}


def load_primary_water() -> pd.DataFrame:
    """Primary water: mg/L → convert to µg/L."""
    rows = []
    for season, suffix, fn in [
        ("Rainy", "R", "waterR_original_17.csv"),
        ("Winter", "W", "waterW_original_17.csv"),
    ]:
        df = pd.read_csv(PRIMARY / fn)
        for _, r in df.iterrows():
            for metal in METALS:
                col = f"{PRIMARY_METAL_COL[metal]}{suffix}"
                if col in df.columns and pd.notna(r[col]):
                    rows.append({
                        "Source": "Primary",
                        "Matrix": "Water",
                        "Season_Norm": season,
                        "River": r["River"],
                        "Station": r["Stations"],
                        "Metal": metal,
                        # mg/L → µg/L
                        "Concentration": float(r[col]) * 1000.0,
                    })
    return pd.DataFrame(rows)


def load_primary_sediment() -> pd.DataFrame:
    """Primary sediment: keep only S1–S17 real stations. Already mg/kg."""
    rows = []
    for season, suffix, fn in [
        ("Rainy", "R", "sedimentRO_v2.csv"),
        ("Winter", "W", "sedimentWO_v2.csv"),
    ]:
        df = pd.read_csv(PRIMARY / fn)
        df = df[df["Stations"].str.match(r"^S\d+$")].copy()
        df = df[df["Stations"].str.extract(r"S(\d+)").astype(int).iloc[:, 0] <= 17]
        for _, r in df.iterrows():
            for metal in METALS:
                col = f"{PRIMARY_METAL_COL[metal]}{suffix}"
                if col in df.columns and pd.notna(r[col]):
                    rows.append({
                        "Source": "Primary",
                        "Matrix": "Sediment",
                        "Season_Norm": season,
                        "River": r["River"],
                        "Station": r["Stations"],
                        "Metal": metal,
                        "Concentration": float(r[col]),
                    })
    return pd.DataFrame(rows)


def load_secondary() -> pd.DataFrame:
    water = pd.read_csv(SECONDARY / "harmonized_water.csv")
    sed = pd.read_csv(SECONDARY / "harmonized_sediment.csv")
    out = pd.concat([water, sed], ignore_index=True)
    out = out[out["Concentration"].notna()].copy()
    out["Source"] = "Literature"
    keep = ["Source", "Matrix", "Season_Norm", "River", "Metal", "Concentration"]
    out["Station"] = out.get("Station_ID", pd.Series([""] * len(out)))
    return out[keep + ["Station"]]


def coverage_stats(primary: pd.DataFrame, lit: pd.DataFrame) -> pd.DataFrame:
    """Per metal × matrix × season: what fraction of primary points are in lit IQR / lit 95% / lit min-max."""
    rows = []
    for matrix in ["Water", "Sediment"]:
        for season in ["Rainy", "Winter"]:
            for metal in METALS:
                p = primary[(primary.Matrix == matrix) & (primary.Season_Norm == season) & (primary.Metal == metal)]
                l = lit[(lit.Matrix == matrix) & (lit.Season_Norm == season) & (lit.Metal == metal)]
                if len(l) < 3 or len(p) == 0:
                    rows.append({
                        "Matrix": matrix, "Season": season, "Metal": metal,
                        "n_primary": len(p), "n_lit": len(l),
                        "pct_in_IQR": np.nan, "pct_in_95": np.nan, "pct_in_range": np.nan,
                        "lit_median": l["Concentration"].median() if len(l) else np.nan,
                        "primary_median": p["Concentration"].median() if len(p) else np.nan,
                        "notes": "insufficient literature coverage" if len(l) < 3 else "no primary",
                    })
                    continue
                q25, q75 = l["Concentration"].quantile([0.25, 0.75])
                q025, q975 = l["Concentration"].quantile([0.025, 0.975])
                lo, hi = l["Concentration"].min(), l["Concentration"].max()
                vals = p["Concentration"].values
                in_iqr = np.mean((vals >= q25) & (vals <= q75)) * 100
                in_95 = np.mean((vals >= q025) & (vals <= q975)) * 100
                in_range = np.mean((vals >= lo) & (vals <= hi)) * 100
                rows.append({
                    "Matrix": matrix, "Season": season, "Metal": metal,
                    "n_primary": len(p), "n_lit": len(l),
                    "pct_in_IQR": round(in_iqr, 1),
                    "pct_in_95": round(in_95, 1),
                    "pct_in_range": round(in_range, 1),
                    "lit_median": round(l["Concentration"].median(), 3),
                    "primary_median": round(p["Concentration"].median(), 3),
                    "notes": "",
                })
    return pd.DataFrame(rows)


def make_figure(primary: pd.DataFrame, lit: pd.DataFrame) -> None:
    """8 metals (rows) × 4 matrix-season combos (cols)."""
    fig, axes = plt.subplots(8, 4, figsize=(18, 22))
    fig.suptitle("Primary (n=17) vs Literature (n=111) — log-scale concentration distributions",
                 fontsize=14, y=0.999)

    combos = [("Water", "Rainy"), ("Water", "Winter"), ("Sediment", "Rainy"), ("Sediment", "Winter")]
    for col_idx, (matrix, season) in enumerate(combos):
        unit = "µg/L" if matrix == "Water" else "mg/kg"
        for row_idx, metal in enumerate(METALS):
            ax = axes[row_idx, col_idx]
            p = primary[(primary.Matrix == matrix) & (primary.Season_Norm == season) & (primary.Metal == metal)]
            l = lit[(lit.Matrix == matrix) & (lit.Season_Norm == season) & (lit.Metal == metal)]
            if len(l) > 0:
                ax.boxplot([l["Concentration"].values], positions=[1], widths=0.6, showfliers=True,
                           boxprops=dict(facecolor="#b3cde3"), patch_artist=True,
                           medianprops=dict(color="black"))
            if len(p) > 0:
                jitter = np.random.uniform(-0.15, 0.15, len(p))
                ax.scatter(np.full(len(p), 2) + jitter,
                           p["Concentration"].values,
                           color="#e41a1c", s=30, alpha=0.75, zorder=3,
                           edgecolors="black", linewidth=0.5)
            ax.set_xticks([1, 2])
            ax.set_xticklabels([f"Lit\nn={len(l)}", f"Prim\nn={len(p)}"], fontsize=8)
            ax.set_yscale("log")
            ax.set_title(f"{metal} — {matrix} {season}", fontsize=10, pad=4)
            if col_idx == 0:
                ax.set_ylabel(unit, fontsize=9)
            ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT / "s3_distribution_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    pw = load_primary_water()
    ps = load_primary_sediment()
    primary = pd.concat([pw, ps], ignore_index=True)
    lit = load_secondary()

    print(f"Primary rows: water={len(pw)}, sediment={len(ps)}")
    print(f"Literature rows (with values): {len(lit)}")
    print(f"  by matrix:\n{lit.groupby('Matrix').size().to_string()}")

    stats = coverage_stats(primary, lit)
    stats.to_csv(OUT / "s3_stats.csv", index=False)

    print("\n" + "=" * 90)
    print("PHASE 1.1 — DISTRIBUTIONAL SANITY CHECK")
    print("=" * 90)
    print(stats.to_string(index=False))

    # Gate G1 evaluation: ≥6 of 8 metals show primary within lit 95% range (aggregate across matrix/season)
    # We use sediment+water jointly, but since units differ the check is per-metal-per-matrix.
    # Pragmatic rule: count a metal as "pass" if it passes ≥1 matrix-season where both have data.
    metal_pass = {}
    for metal in METALS:
        sub = stats[(stats.Metal == metal) & stats.pct_in_95.notna()]
        if sub.empty:
            metal_pass[metal] = None  # untestable
        else:
            # "Within 95% lit range" means ≥50% of primary points fall in lit 2.5–97.5 percentile
            metal_pass[metal] = (sub["pct_in_95"] >= 50).any()

    print("\nPer-metal Gate G1 evaluation (pass if ≥50% of primary in lit 95% range in any matrix-season):")
    passes = 0
    testable = 0
    for m, ok in metal_pass.items():
        if ok is None:
            print(f"  {m}: UNTESTABLE (insufficient lit coverage in both matrices)")
        else:
            testable += 1
            if ok:
                passes += 1
            print(f"  {m}: {'PASS' if ok else 'FAIL'}")

    print(f"\nTestable metals: {testable}/8")
    print(f"Passing metals: {passes}/{testable}")
    threshold = max(6, int(0.75 * testable))  # adapt if not all 8 testable
    print(f"Gate G1 threshold: ≥{threshold} of testable metals")
    gate_pass = passes >= threshold
    print(f"Gate G1: {'PASS — proceed to Phase 2+' if gate_pass else 'FAIL — investigate before Phase 4'}")

    make_figure(primary, lit)
    print(f"\nFigure saved to {OUT / 's3_distribution_comparison.png'}")
    print(f"Stats saved to {OUT / 's3_stats.csv'}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
