"""Compile head-to-head per-metal-season LOOCV comparison and pick winners.

Combines v2 baseline (s4_v2_loocv.csv) with new pooled+log models
(s8_model_comparison.csv). Reports per-metal-season:
- Each model's R^2
- Winner (highest R^2)
- Delta vs v2

Output: analysis_results/enrichment/s8_winners.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = RES / "s8_winners.csv"

METALS = ["Cd", "Cr", "As", "Pb", "Cu", "Ni", "Fe"]
SEASONS = ["Rainy", "Winter"]


def main():
    v2 = pd.read_csv(RES / "s4_v2_loocv.csv")
    new = pd.read_csv(RES / "s8_model_comparison.csv")

    v2["v2_best"] = v2[["Best_LOOCV_R2", "XGBoost_LOOCV_R2",
                        "RF_LOOCV_R2", "SVR_LOOCV_R2"]].max(axis=1)
    v2_lookup = {(r.Metal, r.Season): r.v2_best for _, r in v2.iterrows()}

    new_by_model = {}
    for model in new["Model"].unique():
        sub = new[new["Model"] == model]
        new_by_model[model] = {(r.Metal, r.Season): r.R2 for _, r in sub.iterrows()}

    rows = []
    for metal in METALS:
        for season in SEASONS:
            r2_v2 = v2_lookup.get((metal, season), np.nan)
            cands = {"v2_per_season_best": r2_v2}
            for model_name, lookup in new_by_model.items():
                cands[model_name] = lookup.get((metal, season), np.nan)
            best_model = max(cands, key=lambda k: cands[k])
            best_r2 = cands[best_model]
            delta = best_r2 - r2_v2
            row = {"Metal": metal, "Season": season, **cands,
                   "Winner_Model": best_model,
                   "Winner_R2": round(best_r2, 3),
                   "Delta_vs_v2": round(delta, 3)}
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT, index=False)

    print("=" * 100)
    print(f"{'Metal':4s} {'Season':7s}   "
          f"{'v2':>7s} {'SVR_log':>9s} {'GP_log':>8s} {'MT_DL':>7s} "
          f"{'XGB_log':>9s} {'RF_log':>7s}   {'Winner':22s} "
          f"{'R2':>7s} {'Δ':>7s}")
    print("-" * 100)
    for _, r in out_df.iterrows():
        print(f"{r.Metal:4s} {r.Season:7s}   "
              f"{r.v2_per_season_best:+7.3f} {r.SVR_pooled_log:+9.3f} "
              f"{r.GP_pooled_log:+8.3f} {r.MultiTask_DL_pooled_log:+7.3f} "
              f"{r.XGBoost_pooled_log:+9.3f} {r.RF_pooled_log:+7.3f}   "
              f"{r.Winner_Model:22s} {r.Winner_R2:+7.3f} {r.Delta_vs_v2:+7.3f}")

    print("=" * 100)
    print("\nMETALS WITH POSITIVE WINNER R^2 (substantively predictable):")
    pos = out_df[out_df["Winner_R2"] > 0].sort_values("Winner_R2", ascending=False)
    for _, r in pos.iterrows():
        flag = " *NEW*" if r.Delta_vs_v2 > 0.05 else ""
        print(f"  {r.Metal:3s} {r.Season:7s}  R^2={r.Winner_R2:+.3f}  "
              f"({r.Winner_Model})  Δvs_v2={r.Delta_vs_v2:+.3f}{flag}")

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
