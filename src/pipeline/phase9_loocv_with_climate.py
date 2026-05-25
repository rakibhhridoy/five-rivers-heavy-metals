"""Re-run GP_pooled_log and SVR_pooled_log with climate covariates added.

Features: 4 PROX + 64 AE + 1 is_winter + 4 climate = 73
Climate: rain_30d_mm, rain_7d_mm, temp_30d_C, temp_7d_C

Output: appends model 'GP_pooled_log_climate' and 'SVR_pooled_log_climate'
to analysis_results/enrichment/s8_model_comparison.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, WhiteKernel, ConstantKernel,
)
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "results/s8_model_comparison.csv"

METALS = ["Cd", "Cr", "As", "Pb", "Cu", "Ni", "Fe"]
PROX = ["dist_brick", "num_brick", "dist_ind", "num_ind"]
AE_COLS = [f"AE_{i:02d}" for i in range(64)]
CLIMATE = ["rain_30d_mm", "rain_7d_mm", "temp_30d_C", "temp_7d_C"]
FEATURES = PROX + AE_COLS + ["is_winter"] + CLIMATE


def loocv(X, y_log, y_raw, station_ids, model_factory):
    preds = np.zeros_like(y_raw)
    for stn in np.unique(station_ids):
        te = station_ids == stn
        tr = ~te
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(X[tr])
        Xte_s = sc.transform(X[te])
        m = model_factory()
        m.fit(Xtr_s, y_log[tr])
        preds[te] = np.expm1(m.predict(Xte_s))
    return preds


def make_gp():
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0,
                                                   length_scale_bounds=(1e-1, 1e2),
                                                   nu=2.5)
        + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-3, 1e1))
    )
    return GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                    normalize_y=True, n_restarts_optimizer=3,
                                    random_state=42)


def make_svr():
    return SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")


def main():
    pooled = pd.read_csv(DATA / "sediment_pooled_v2.csv")
    climate = pd.read_csv(DATA / "climate_covariates_v2.csv")
    df = pooled.merge(climate[["station_id", "is_winter"] + CLIMATE],
                      on=["station_id", "is_winter"], how="left")
    print(f"Merged shape: {df.shape}")
    print(f"Climate features non-null: {df[CLIMATE].notna().all(axis=1).sum()}/{len(df)}")

    X = df[FEATURES].values.astype(np.float32)
    station_ids = df["station_id"].values
    is_winter = df["is_winter"].values

    new_rows = []
    for model_name, factory in [
        ("GP_pooled_log_climate", make_gp),
        ("SVR_pooled_log_climate", make_svr),
    ]:
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
