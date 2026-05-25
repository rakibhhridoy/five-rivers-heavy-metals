"""
Phase 4f (focused) — LOOCV on 17 real primary stations with real AE features.
Uses the manuscript's reported best architecture per metal, plus 3 ML baselines
for comparison. Skips bootstrap CI (would require 50× compute on synthetic data
that's no longer relevant).

Output: analysis_results/enrichment/s4_v2_loocv.csv
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, Concatenate, Dropout,
    MultiHeadAttention, Reshape, LayerNormalization, Multiply, Add, Lambda,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "results/s4_v2_loocv.csv"

EPOCHS = 100
BATCH_SIZE = 8
PATIENCE = 15
N_REAL = 17

METADATA = ["Stations", "River", "Lat", "Long", "geometry"]
RAINY_CHEM  = ["CrR", "NiR", "CuR", "AsR", "CdR", "PbR", "MR", "SandR", "SiltR", "ClayR", "FeR", "RI"]
WINTER_CHEM = ["CdW", "CrW", "AsW", "PbW", "CuW", "NiW", "FeW", "MW", "SandW", "SiltW", "ClayW", "RI"]

TARGETS = {
    "Cd": ("CdR", "CdW"), "Cr": ("CrR", "CrW"), "As": ("AsR", "AsW"),
    "Pb": ("PbR", "PbW"), "Cu": ("CuR", "CuW"), "Ni": ("NiR", "NiW"),
    "Fe": ("FeR", "FeW"),
}

BEST_MODEL = {
    "Cd": "Dual Attention",
    "Cr": "Transformer CNN", "As": "Transformer CNN", "Pb": "Transformer CNN",
    "Cu": "Transformer CNN", "Ni": "Transformer CNN",
    "Fe": "XGBoost",
}


def build_transformer_cnn(nf):
    inp = Input(shape=(nf,))
    x_mlp = Dense(64, activation="relu")(inp); x_mlp = Dropout(0.3)(x_mlp); x_mlp = Dense(32, activation="relu")(x_mlp)
    x_cnn = Reshape((nf, 1))(inp)
    x_cnn = Conv1D(32, 3, activation="relu", padding="same")(x_cnn)
    x_cnn = MaxPooling1D(2)(x_cnn)
    x_cnn = Conv1D(64, 3, activation="relu", padding="same")(x_cnn)
    x_cnn = Flatten()(x_cnn); x_cnn = Dense(32, activation="relu")(x_cnn)
    t_mlp = Reshape((1, 32))(x_mlp); t_cnn = Reshape((1, 32))(x_cnn)
    tokens = Concatenate(axis=1)([t_mlp, t_cnn])
    attn = MultiHeadAttention(num_heads=4, key_dim=8)(tokens, tokens)
    attn = LayerNormalization()(Add()([tokens, attn])); attn = Flatten()(attn)
    x = Dense(64, activation="relu")(attn); x = Dropout(0.3)(x)
    out = Dense(1)(x)
    m = Model(inp, out); m.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
    return m


def build_dual_attn(nf):
    inp = Input(shape=(nf,))
    x_ch = Dense(32, activation="relu")(inp)
    ch_a = Dense(32, activation="sigmoid")(x_ch); x_ch = Multiply()([x_ch, ch_a])
    x_sp = Reshape((nf, 1))(inp)
    x_sp = Conv1D(32, 3, activation="relu", padding="same")(x_sp)
    sp_a = Conv1D(1, 1, activation="sigmoid")(x_sp); x_sp = Multiply()([x_sp, sp_a])
    x_sp = Flatten()(x_sp); x_sp = Dense(32, activation="relu")(x_sp)
    gi = Concatenate()([x_ch, x_sp]); gw = Dense(1, activation="sigmoid")(gi)
    fused = Add()([Multiply()([x_ch, gw]),
                   Multiply()([x_sp, Lambda(lambda g: 1.0 - g)(gw)])])
    x = Dense(32, activation="relu")(fused); x = Dropout(0.3)(x)
    out = Dense(1)(x)
    m = Model(inp, out); m.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
    return m


DL = {"Transformer CNN": build_transformer_cnn, "Dual Attention": build_dual_attn}


def get_ml(name):
    if name == "XGBoost":
        return GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                         min_samples_leaf=3, random_state=42)
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=3,
                                     random_state=42, n_jobs=-1)
    if name == "SVR":
        return SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")
    raise ValueError(name)


def loocv(X17, y17, model_name):
    """LOOCV on 17 real stations only."""
    preds, trues = [], []
    for tri, tei in LeaveOneOut().split(X17):
        Xtr, Xte = X17[tri], X17[tei]; ytr, yte = y17[tri], y17[tei]
        sc = StandardScaler(); Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)
        if model_name in DL:
            tf.random.set_seed(42)
            m = DL[model_name](Xtr_s.shape[1])
            m.fit(Xtr_s, ytr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
                  validation_split=0.15,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)])
            yp = m.predict(Xte_s, verbose=0).flatten()
        else:
            m = get_ml(model_name); m.fit(Xtr_s, ytr); yp = m.predict(Xte_s)
        preds.append(float(yp[0])); trues.append(float(yte[0]))
    t = np.array(trues); p = np.array(preds)
    return {"R2": float(r2_score(t, p)),
            "RMSE": float(np.sqrt(mean_squared_error(t, p))),
            "MAE": float(mean_absolute_error(t, p))}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rainy = pd.read_csv(DATA / "sedimentRO_v2.csv")
    winter = pd.read_csv(DATA / "sedimentWO_v2.csv")
    feat_r = [c for c in rainy.columns if c not in METADATA + RAINY_CHEM]
    feat_w = [c for c in winter.columns if c not in METADATA + WINTER_CHEM]
    X_r = rainy[feat_r].fillna(0).values.astype(np.float32)
    X_w = winter[feat_w].fillna(0).values.astype(np.float32)

    rows = []
    for metal, (col_r, col_w) in TARGETS.items():
        model_name = BEST_MODEL[metal]
        for season, X, prim_col in [("Rainy", X_r, col_r), ("Winter", X_w, col_w)]:
            df = rainy if season == "Rainy" else winter
            y = df[prim_col].values.astype(np.float32)
            X17, y17 = X[:N_REAL], y[:N_REAL]
            print(f"  {metal:3s} {season:7s}  model={model_name}", flush=True)
            res_best = loocv(X17, y17, model_name)
            res_xgb  = loocv(X17, y17, "XGBoost")
            res_rf   = loocv(X17, y17, "Random Forest")
            res_svr  = loocv(X17, y17, "SVR")

            rows.append({
                "Metal": metal, "Season": season,
                "Best_Model": model_name,
                "Best_LOOCV_R2": round(res_best["R2"], 3),
                "Best_LOOCV_RMSE": round(res_best["RMSE"], 2),
                "XGBoost_LOOCV_R2": round(res_xgb["R2"], 3),
                "RF_LOOCV_R2":      round(res_rf["R2"], 3),
                "SVR_LOOCV_R2":     round(res_svr["R2"], 3),
            })
            print(f"    best={res_best['R2']:.3f}  XGB={res_xgb['R2']:.3f}  "
                  f"RF={res_rf['R2']:.3f}  SVR={res_svr['R2']:.3f}", flush=True)

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
