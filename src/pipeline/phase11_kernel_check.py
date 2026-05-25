"""GP kernel sensitivity: RBF vs Matérn 1.5 vs Matérn 2.5 on pooled+log LOOCV.

Reports per-metal-season LOOCV R² for the three kernel choices on Cd and Pb,
the two metals where GP is the winning model. Used to justify Matérn 2.5
choice in the manuscript.

Output: analysis_results/enrichment/s11_kernel_sensitivity.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, WhiteKernel, ConstantKernel,
)
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "results/s11_kernel_sensitivity.csv"

PROX = ["dist_brick", "num_brick", "dist_ind", "num_ind"]
AE_COLS = [f"AE_{i:02d}" for i in range(64)]
CLIMATE = ["rain_30d_mm", "rain_7d_mm", "temp_30d_C", "temp_7d_C"]


def make_gp(kernel_name):
    if kernel_name == "RBF":
        kernel = (ConstantKernel(1.0, (1e-2, 1e2))
                  * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e2))
                  + WhiteKernel(noise_level=0.5,
                                noise_level_bounds=(1e-3, 1e1)))
    elif kernel_name == "Matern_1.5":
        kernel = (ConstantKernel(1.0, (1e-2, 1e2))
                  * Matern(length_scale=1.0,
                           length_scale_bounds=(1e-1, 1e2), nu=1.5)
                  + WhiteKernel(noise_level=0.5,
                                noise_level_bounds=(1e-3, 1e1)))
    elif kernel_name == "Matern_2.5":
        kernel = (ConstantKernel(1.0, (1e-2, 1e2))
                  * Matern(length_scale=1.0,
                           length_scale_bounds=(1e-1, 1e2), nu=2.5)
                  + WhiteKernel(noise_level=0.5,
                                noise_level_bounds=(1e-3, 1e1)))
    return GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                    normalize_y=True, n_restarts_optimizer=3,
                                    random_state=42)


def loocv_log(X, y_log, y_raw, station_ids, kernel_name):
    preds_raw = np.zeros_like(y_raw)
    for stn in np.unique(station_ids):
        te = station_ids == stn
        tr = ~te
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(X[tr])
        Xte_s = sc.transform(X[te])
        m = make_gp(kernel_name)
        m.fit(Xtr_s, y_log[tr])
        preds_raw[te] = np.expm1(m.predict(Xte_s))
    return preds_raw


def main():
    pooled = pd.read_csv(DATA / "sediment_pooled_v2.csv")
    climate = pd.read_csv(DATA / "climate_covariates_v2.csv")
    df = pooled.merge(climate[["station_id", "is_winter"] + CLIMATE],
                      on=["station_id", "is_winter"], how="left")

    # Cd uses pooled+log (no climate). Pb uses pooled+log+climate.
    targets = [
        ("Cd", PROX + AE_COLS + ["is_winter"]),
        ("Pb", PROX + AE_COLS + ["is_winter"] + CLIMATE),
    ]

    rows = []
    for metal, feat_cols in targets:
        X = df[feat_cols].values.astype(np.float32)
        y_raw = df[metal].values.astype(float)
        y_log = df[f"log_{metal}"].values.astype(float)
        sids = df["station_id"].values
        is_winter = df["is_winter"].values

        for kernel in ["RBF", "Matern_1.5", "Matern_2.5"]:
            preds = loocv_log(X, y_log, y_raw, sids, kernel)
            for season_label, mask in [("Rainy", is_winter == 0),
                                        ("Winter", is_winter == 1)]:
                r2 = r2_score(y_raw[mask], preds[mask])
                rows.append({"Metal": metal, "Season": season_label,
                             "Kernel": kernel, "R2": round(r2, 3)})
                print(f"  {metal} {season_label:6s} {kernel:10s}  R2={r2:+.3f}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT, index=False)

    print("\nSummary (R^2 by kernel, per metal-season):")
    pivot = out_df.pivot_table(index=["Metal", "Season"],
                                columns="Kernel", values="R2")
    print(pivot)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
