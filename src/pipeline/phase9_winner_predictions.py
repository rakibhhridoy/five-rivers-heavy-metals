"""Capture per-station LOOCV predictions and 117-station spatial predictions
under the per-metal-season winning model. Powers Fig 7 (scatter) and Fig 8.

Winners (from s8_winners.csv):
  Cd Rainy/Winter:        GP_pooled_log
  Pb Rainy/Winter:        GP_pooled_log_climate
  Cu Rainy:               GP_pooled_log_climate
  + (Ni Rainy / Mn Rainy / Cu Winter winners are v2 per-season models;
     existing s4_v2_loocv_preds.csv already covers those for completeness.)

Outputs:
  analysis_results/enrichment/s9_winner_loocv_preds.csv
  analysis_results/enrichment/s9_winner_spatial_preds.csv  (117-station grid)
  analysis_results/enrichment/s9_cd_pb_bootstrap.csv      (95% CI for Cd, Pb)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RES = ROOT / "results"

PROX = ["dist_brick", "num_brick", "dist_ind", "num_ind"]
AE_COLS = [f"AE_{i:02d}" for i in range(64)]
CLIMATE = ["rain_30d_mm", "rain_7d_mm", "temp_30d_C", "temp_7d_C"]
N_REAL = 17
N_BOOT = 300


def make_gp(n_restarts=3):
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0,
                                                   length_scale_bounds=(1e-1, 1e2),
                                                   nu=2.5)
        + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-3, 1e1))
    )
    return GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                    normalize_y=True,
                                    n_restarts_optimizer=n_restarts,
                                    random_state=42)


def loocv_preds(X, y_log, station_ids):
    """Returns log-space predictions, station-LOOCV."""
    preds_log = np.zeros_like(y_log)
    for stn in np.unique(station_ids):
        te = station_ids == stn
        tr = ~te
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(X[tr])
        Xte_s = sc.transform(X[te])
        m = make_gp()
        m.fit(Xtr_s, y_log[tr])
        preds_log[te] = m.predict(Xte_s)
    return preds_log


def fit_predict_full(X_train, y_log_train, X_full):
    """Train GP on all real stations, predict at all 117 locations (log space)."""
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(X_train)
    X_full_s = sc.transform(X_full)
    m = make_gp()
    m.fit(Xtr_s, y_log_train)
    return m.predict(X_full_s), m


def build_pooled_features(use_climate):
    pooled = pd.read_csv(DATA / "sediment_pooled_v2.csv")
    if use_climate:
        climate = pd.read_csv(DATA / "climate_covariates_v2.csv")
        pooled = pooled.merge(climate[["station_id", "is_winter"] + CLIMATE],
                              on=["station_id", "is_winter"], how="left")
        feat_cols = PROX + AE_COLS + ["is_winter"] + CLIMATE
    else:
        feat_cols = PROX + AE_COLS + ["is_winter"]
    return pooled, feat_cols


def build_117_grid(metal, season_is_winter, use_climate):
    """Build 117-station feature matrix matching pooled training schema."""
    src = "sedimentRO_v2.csv" if not season_is_winter else "sedimentWO_v2.csv"
    df = pd.read_csv(DATA / src)
    rename = {
        "hydro_dist_brick": "dist_brick", "num_brick_field": "num_brick",
        "hydro_dist_ind": "dist_ind",     "num_industry": "num_ind",
        "hydrological_dist_to_nearest_BF":  "dist_brick",
        "num_upstream_BF":  "num_brick",
        "hydrological_dist_to_nearest_IND": "dist_ind",
        "num_upstream_IND": "num_ind",
    }
    df = df.rename(columns=rename)
    df["is_winter"] = int(season_is_winter)
    if use_climate:
        # Use the climate values from the closest real station (by index match)
        # For IDW points, repeat the median real-station climate for that season
        climate = pd.read_csv(DATA / "climate_covariates_v2.csv")
        med = climate[climate["is_winter"] == int(season_is_winter)][CLIMATE].median()
        for c in CLIMATE:
            df[c] = med[c]
        feat_cols = PROX + AE_COLS + ["is_winter"] + CLIMATE
    else:
        feat_cols = PROX + AE_COLS + ["is_winter"]
    return df, feat_cols


def main():
    # Bootstrap CIs reported only for Cd and Pb (the >R^2=0.5 metals).
    # Cu rainy R^2=0.34 reported as point estimate without CI to keep runtime tractable.
    targets = [
        ("Cd", False, True),   # GP_pooled_log,         bootstrap=True
        ("Pb", True,  True),   # GP_pooled_log_climate, bootstrap=True
        ("Cu", True,  False),  # GP_pooled_log_climate, bootstrap=False
    ]

    loocv_rows = []
    spatial_rows = []
    bootstrap_rows = []

    for metal, use_climate, do_bootstrap in targets:
        print(f"\n=== {metal} (climate={use_climate}, bootstrap={do_bootstrap}) ===",
              flush=True)
        pooled, feat_cols = build_pooled_features(use_climate)
        X = pooled[feat_cols].values.astype(np.float32)
        y_raw = pooled[metal].values.astype(float)
        y_log = pooled[f"log_{metal}"].values.astype(float)
        station_ids = pooled["station_id"].values
        is_winter = pooled["is_winter"].values

        # 1) LOOCV predictions
        preds_log = loocv_preds(X, y_log, station_ids)
        preds_raw = np.expm1(preds_log)

        for season_label, mask in [("Rainy", is_winter == 0),
                                    ("Winter", is_winter == 1)]:
            r2 = r2_score(y_raw[mask], preds_raw[mask])
            rmse = float(np.sqrt(mean_squared_error(y_raw[mask], preds_raw[mask])))
            print(f"  {metal} {season_label}: LOOCV R^2 = {r2:+.3f}, RMSE = {rmse:.2f}")
            sub = pooled[mask].reset_index(drop=True)
            for i in range(len(sub)):
                loocv_rows.append({
                    "Metal": metal, "Season": season_label,
                    "Model": "GP_pooled_log_climate" if use_climate else "GP_pooled_log",
                    "Station": sub["station_id"].iloc[i],
                    "River": sub["River"].iloc[i],
                    "Lat": sub["Lat"].iloc[i], "Long": sub["Long"].iloc[i],
                    "Observed": float(y_raw[mask][i]),
                    "Predicted": float(preds_raw[mask][i]),
                    "LOOCV_R2": round(r2, 3),
                    "LOOCV_RMSE": round(rmse, 2),
                })

        # 2) Bootstrap CI on LOOCV per season (at station level)
        if not do_bootstrap:
            continue
        for season_label, mask in [("Rainy", is_winter == 0),
                                    ("Winter", is_winter == 1)]:
            X_s = X[mask]
            y_log_s = y_log[mask]
            y_raw_s = y_raw[mask]
            stn_s = station_ids[mask]
            point_preds_log = np.zeros_like(y_log_s)
            for stn in np.unique(stn_s):
                te = stn_s == stn
                tr = ~te
                sc = StandardScaler()
                Xtr_s = sc.fit_transform(X_s[tr])
                Xte_s = sc.transform(X_s[te])
                m = make_gp(); m.fit(Xtr_s, y_log_s[tr])
                point_preds_log[te] = m.predict(Xte_s)
            point_r2 = r2_score(y_raw_s, np.expm1(point_preds_log))

            rng = np.random.default_rng(42)
            boot = []
            for b in range(N_BOOT):
                idx = rng.integers(0, len(y_raw_s), len(y_raw_s))
                if len(np.unique(idx)) < 5:
                    continue
                Xb = X_s[idx]; yb_log = y_log_s[idx]; yb_raw = y_raw_s[idx]
                stn_b = stn_s[idx]
                preds_b = np.zeros_like(yb_log)
                try:
                    for stn in np.unique(stn_b):
                        te = stn_b == stn
                        tr = ~te
                        sc = StandardScaler()
                        Xtr_s = sc.fit_transform(Xb[tr])
                        Xte_s = sc.transform(Xb[te])
                        m = make_gp(n_restarts=0); m.fit(Xtr_s, yb_log[tr])
                        preds_b[te] = m.predict(Xte_s)
                    boot.append(r2_score(yb_raw, np.expm1(preds_b)))
                except Exception:
                    continue
            boot = np.array(boot)
            ci_lo = float(np.percentile(boot, 2.5))
            ci_hi = float(np.percentile(boot, 97.5))
            med = float(np.median(boot))
            bootstrap_rows.append({
                "Metal": metal, "Season": season_label,
                "Model": "GP_pooled_log_climate" if use_climate else "GP_pooled_log",
                "Point_LOOCV_R2": round(point_r2, 3),
                "Bootstrap_median_R2": round(med, 3),
                "CI95_low": round(ci_lo, 3),
                "CI95_high": round(ci_hi, 3),
                "n_boot_used": int(len(boot)),
            })
            print(f"  {metal} {season_label}: bootstrap median = {med:+.3f}, "
                  f"95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")

        # 3) 117-station spatial predictions per season
        for season_is_winter in [False, True]:
            season_label = "Winter" if season_is_winter else "Rainy"
            df_117, feat_cols_117 = build_117_grid(metal, season_is_winter, use_climate)
            X_117 = df_117[feat_cols_117].values.astype(np.float32)
            mask_train = (is_winter == int(season_is_winter))
            preds_log, _ = fit_predict_full(X[mask_train], y_log[mask_train], X_117)
            preds_raw = np.expm1(preds_log)
            for i in range(len(df_117)):
                spatial_rows.append({
                    "Metal": metal, "Season": season_label,
                    "Model": "GP_pooled_log_climate" if use_climate else "GP_pooled_log",
                    "Station": df_117["Stations"].iloc[i],
                    "River": df_117["River"].iloc[i],
                    "Lat": df_117["Lat"].iloc[i], "Long": df_117["Long"].iloc[i],
                    "is_real_station": int(i < N_REAL),
                    "Predicted": float(preds_raw[i]),
                })

    pd.DataFrame(loocv_rows).to_csv(RES / "s9_winner_loocv_preds.csv", index=False)
    pd.DataFrame(spatial_rows).to_csv(RES / "s9_winner_spatial_preds.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(RES / "s9_cd_pb_bootstrap.csv", index=False)
    print(f"\nWrote 3 CSVs to {RES}")


if __name__ == "__main__":
    main()
