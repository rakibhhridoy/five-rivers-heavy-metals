"""
Phase 4g — Honest ablation: proximity-only vs real-AE-only vs both.

For each metal × season, runs 5-fold CV on three feature sets:
  proximity (4 features: hydro_dist_brick, num_brick_field, hydro_dist_ind, num_industry)
  real AE   (64 features: AE_00 .. AE_63 from GEE V1)
  combined  (68 features)

Replaces the manuscript's "AlphaEarth captured 98% of combined model performance"
claim — that claim was tested on synthetic AE features.

Output: analysis_results/enrichment/s4_v2_ablation.csv
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT_CSV = ROOT / "results/s4_v2_ablation.csv"

PROX_COLS_RAINY  = ["hydro_dist_brick", "num_brick_field", "hydro_dist_ind", "num_industry"]
PROX_COLS_WINTER = ["hydrological_dist_to_nearest_BF", "num_upstream_BF",
                    "hydrological_dist_to_nearest_IND", "num_upstream_IND"]
AE_COLS = [f"AE_{i:02d}" for i in range(64)]

TARGETS = {
    "Cd": ("CdR", "CdW"), "Cr": ("CrR", "CrW"), "As": ("AsR", "AsW"),
    "Pb": ("PbR", "PbW"), "Cu": ("CuR", "CuW"), "Ni": ("NiR", "NiW"),
    "Fe": ("FeR", "FeW"),
}


def cv_r2(X, y, model_fn):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2s = []
    for tri, tei in kf.split(X):
        Xtr, Xte = X[tri], X[tei]; ytr, yte = y[tri], y[tei]
        sc = StandardScaler(); Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)
        m = model_fn(); m.fit(Xtr_s, ytr); yp = m.predict(Xte_s)
        r2s.append(r2_score(yte, yp))
    return float(np.mean(r2s)), float(np.std(r2s))


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rainy  = pd.read_csv(DATA / "sedimentRO_v2.csv")
    winter = pd.read_csv(DATA / "sedimentWO_v2.csv")

    # XGBoost as the workhorse for ablation (fast, decent baseline) — same model across feature sets.
    model_fn = lambda: GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                                  min_samples_leaf=3, random_state=42)

    rows = []
    for metal, (col_r, col_w) in TARGETS.items():
        for season, df, prim_col, prox_cols in [
            ("Rainy",  rainy,  col_r, PROX_COLS_RAINY),
            ("Winter", winter, col_w, PROX_COLS_WINTER),
        ]:
            y = df[prim_col].values.astype(np.float32)
            X_prox = df[prox_cols].values.astype(np.float32)
            X_ae   = df[AE_COLS].values.astype(np.float32)
            X_both = df[prox_cols + AE_COLS].values.astype(np.float32)

            r2_prox, _ = cv_r2(X_prox, y, model_fn)
            r2_ae,   _ = cv_r2(X_ae,   y, model_fn)
            r2_both, _ = cv_r2(X_both, y, model_fn)

            ae_share = (r2_ae / r2_both * 100) if r2_both > 0 else None
            prox_share = (r2_prox / r2_both * 100) if r2_both > 0 else None
            rows.append({
                "Metal": metal, "Season": season,
                "Prox_only_R2": round(r2_prox, 3),
                "AE_only_R2":   round(r2_ae, 3),
                "Both_R2":      round(r2_both, 3),
                "AE_share_pct":   round(ae_share, 1)   if ae_share is not None else None,
                "Prox_share_pct": round(prox_share, 1) if prox_share is not None else None,
                "AE_marginal":  round(r2_both - r2_prox, 3),
                "Prox_marginal":round(r2_both - r2_ae,   3),
            })
            print(f"  {metal:3s} {season:7s}  prox={r2_prox:>6.3f}  AE={r2_ae:>6.3f}  both={r2_both:>6.3f}  "
                  f"AE-share={ae_share}  AE-marginal={r2_both - r2_prox:+.3f}")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
