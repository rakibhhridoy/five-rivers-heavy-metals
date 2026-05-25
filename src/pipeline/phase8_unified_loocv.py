"""Unified LOOCV comparison: pooled + log + multi-task + GP.

Compares four model variants under station-leave-one-out CV (17 iterations,
both seasons of held-out station withheld together):

(A) v2 baseline: per-season SVR on raw concentrations (current manuscript)
(B) Pooled SVR + log: per-metal SVR on log-targets, pooled rainy+winter, +is_winter
(C) Multi-task DL: shared 128->64 trunk, 8 metal output heads, log-targets, pooled
(D) Gaussian Process: per-metal GP with composite kernel, log-targets, pooled

Outputs per-metal-season LOOCV R² and RMSE for each model in long format.

Output: analysis_results/enrichment/s8_model_comparison.csv
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ConstantKernel, Matern,
)
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "results/s8_model_comparison.csv"

N_REAL = 17
METALS = ["Cd", "Cr", "As", "Pb", "Cu", "Ni", "Fe"]
PROX = ["dist_brick", "num_brick", "dist_ind", "num_ind"]
AE_COLS = [f"AE_{i:02d}" for i in range(64)]
FEATURES_POOLED = PROX + AE_COLS + ["is_winter"]


def loocv_svr_per_metal(X, y_log, y_raw, station_ids, is_winter):
    """Pooled SVR per metal on log targets, station-LOOCV."""
    preds_raw = np.zeros_like(y_raw)
    unique_stations = np.unique(station_ids)
    for stn in unique_stations:
        te_mask = station_ids == stn
        tr_mask = ~te_mask
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(X[tr_mask])
        Xte_s = sc.transform(X[te_mask])
        m = SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")
        m.fit(Xtr_s, y_log[tr_mask])
        log_pred = m.predict(Xte_s)
        preds_raw[te_mask] = np.expm1(log_pred)
    return preds_raw


def loocv_gp_per_metal(X, y_log, y_raw, station_ids):
    """Pooled GP per metal on log targets, composite kernel."""
    n_prox, n_ae = len(PROX), len(AE_COLS)
    preds_raw = np.zeros_like(y_raw)
    unique_stations = np.unique(station_ids)
    for stn in unique_stations:
        te_mask = station_ids == stn
        tr_mask = ~te_mask
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(X[tr_mask])
        Xte_s = sc.transform(X[te_mask])
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0,
                                                       length_scale_bounds=(1e-1, 1e2),
                                                       nu=2.5)
            + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-3, 1e1))
        )
        m = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                     normalize_y=True, n_restarts_optimizer=3,
                                     random_state=42)
        m.fit(Xtr_s, y_log[tr_mask])
        log_pred = m.predict(Xte_s)
        preds_raw[te_mask] = np.expm1(log_pred)
    return preds_raw


def build_multitask(n_features, n_targets):
    inp = Input(shape=(n_features,))
    h = Dense(128, activation="relu")(inp)
    h = Dropout(0.3)(h)
    h = Dense(64, activation="relu")(h)
    h = Dropout(0.3)(h)
    outs = [Dense(1, name=f"out_{m}")(h) for m in METALS]
    m = Model(inp, outs)
    m.compile(optimizer=Adam(0.005), loss="mse")
    return m


def loocv_multitask(X, Y_log, Y_raw, station_ids):
    """Multi-task DL with 8 metal heads, station-LOOCV."""
    preds_raw = np.zeros_like(Y_raw)
    unique_stations = np.unique(station_ids)
    for stn in unique_stations:
        te_mask = station_ids == stn
        tr_mask = ~te_mask
        sc_X = StandardScaler()
        Xtr_s = sc_X.fit_transform(X[tr_mask])
        Xte_s = sc_X.transform(X[te_mask])

        target_means = Y_log[tr_mask].mean(axis=0)
        target_stds = Y_log[tr_mask].std(axis=0) + 1e-9
        Ytr_s = (Y_log[tr_mask] - target_means) / target_stds

        tf.random.set_seed(42)
        model = build_multitask(Xtr_s.shape[1], len(METALS))
        model.fit(
            Xtr_s, [Ytr_s[:, i] for i in range(len(METALS))],
            epochs=200, batch_size=8, verbose=0,
            validation_split=0.15,
            callbacks=[EarlyStopping(monitor="val_loss", patience=20,
                                     restore_best_weights=True)],
        )
        preds_s = np.column_stack(
            [p.flatten() for p in model.predict(Xte_s, verbose=0)]
        )
        preds_log = preds_s * target_stds + target_means
        preds_raw[te_mask] = np.expm1(preds_log)
    return preds_raw


def main():
    df = pd.read_csv(DATA / "sediment_pooled_v2.csv")
    print(f"Loaded pooled dataset: {df.shape}")

    X = df[FEATURES_POOLED].values.astype(np.float32)
    station_ids = df["station_id"].values
    is_winter = df["is_winter"].values

    Y_log = np.column_stack([df[f"log_{m}"].values for m in METALS])
    Y_raw = np.column_stack([df[m].values for m in METALS])

    rows = []

    # (B) Per-metal SVR, pooled+log
    print("\n[B] Per-metal SVR pooled+log")
    for i, metal in enumerate(METALS):
        preds = loocv_svr_per_metal(X, Y_log[:, i], Y_raw[:, i],
                                    station_ids, is_winter)
        for season_label, mask in [("Rainy", is_winter == 0),
                                    ("Winter", is_winter == 1)]:
            r2 = r2_score(Y_raw[mask, i], preds[mask])
            rmse = float(np.sqrt(mean_squared_error(Y_raw[mask, i], preds[mask])))
            rows.append({
                "Model": "SVR_pooled_log", "Metal": metal,
                "Season": season_label, "R2": round(r2, 3), "RMSE": round(rmse, 2),
            })
            print(f"  {metal:3s} {season_label:6s}  R2={r2:+.3f}  RMSE={rmse:.2f}")

    # (D) Per-metal GP, pooled+log
    print("\n[D] Per-metal GP pooled+log")
    for i, metal in enumerate(METALS):
        preds = loocv_gp_per_metal(X, Y_log[:, i], Y_raw[:, i], station_ids)
        for season_label, mask in [("Rainy", is_winter == 0),
                                    ("Winter", is_winter == 1)]:
            r2 = r2_score(Y_raw[mask, i], preds[mask])
            rmse = float(np.sqrt(mean_squared_error(Y_raw[mask, i], preds[mask])))
            rows.append({
                "Model": "GP_pooled_log", "Metal": metal,
                "Season": season_label, "R2": round(r2, 3), "RMSE": round(rmse, 2),
            })
            print(f"  {metal:3s} {season_label:6s}  R2={r2:+.3f}  RMSE={rmse:.2f}")

    # (C) Multi-task DL
    print("\n[C] Multi-task DL pooled+log (slow, 17 LOOCV iters)")
    preds_mt = loocv_multitask(X, Y_log, Y_raw, station_ids)
    for i, metal in enumerate(METALS):
        for season_label, mask in [("Rainy", is_winter == 0),
                                    ("Winter", is_winter == 1)]:
            r2 = r2_score(Y_raw[mask, i], preds_mt[mask, i])
            rmse = float(np.sqrt(mean_squared_error(Y_raw[mask, i], preds_mt[mask, i])))
            rows.append({
                "Model": "MultiTask_DL_pooled_log", "Metal": metal,
                "Season": season_label, "R2": round(r2, 3), "RMSE": round(rmse, 2),
            })
            print(f"  {metal:3s} {season_label:6s}  R2={r2:+.3f}  RMSE={rmse:.2f}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
