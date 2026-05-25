"""Add XGBoost + Random Forest pooled+log LOOCV for fair per-metal comparison.

Appends to analysis_results/enrichment/s8_model_comparison.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "results/s8_model_comparison.csv"

METALS = ["Cd", "Cr", "As", "Pb", "Cu", "Ni", "Fe"]
PROX = ["dist_brick", "num_brick", "dist_ind", "num_ind"]
AE_COLS = [f"AE_{i:02d}" for i in range(64)]
FEATURES = PROX + AE_COLS + ["is_winter"]


def loocv(X, y_log, y_raw, station_ids, model_factory):
    preds_raw = np.zeros_like(y_raw)
    for stn in np.unique(station_ids):
        te = station_ids == stn
        tr = ~te
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(X[tr])
        Xte_s = sc.transform(X[te])
        m = model_factory()
        m.fit(Xtr_s, y_log[tr])
        preds_raw[te] = np.expm1(m.predict(Xte_s))
    return preds_raw


def main():
    df = pd.read_csv(DATA / "sediment_pooled_v2.csv")
    X = df[FEATURES].values.astype(np.float32)
    station_ids = df["station_id"].values
    is_winter = df["is_winter"].values

    factories = {
        "XGBoost_pooled_log": lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            min_samples_leaf=3, random_state=42),
        "RF_pooled_log": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=3,
            random_state=42, n_jobs=-1),
    }

    new_rows = []
    for model_name, factory in factories.items():
        print(f"\n[{model_name}]")
        for metal in METALS:
            y_raw = df[metal].values
            y_log = df[f"log_{metal}"].values
            preds = loocv(X, y_log, y_raw, station_ids, factory)
            for season_label, mask in [("Rainy", is_winter == 0),
                                        ("Winter", is_winter == 1)]:
                r2 = r2_score(y_raw[mask], preds[mask])
                rmse = float(np.sqrt(mean_squared_error(y_raw[mask], preds[mask])))
                new_rows.append({
                    "Model": model_name, "Metal": metal,
                    "Season": season_label, "R2": round(r2, 3),
                    "RMSE": round(rmse, 2),
                })
                print(f"  {metal:3s} {season_label:6s}  R2={r2:+.3f}")

    existing = pd.read_csv(OUT)
    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined.to_csv(OUT, index=False)
    print(f"\nAppended {len(new_rows)} rows. Total: {len(combined)}")


if __name__ == "__main__":
    main()
