"""
Phase 4f — Comprehensive evaluation grid on real AlphaEarth (v2 features).

8 metals × 2 seasons × 8 models × {5-fold CV on n=117, LOOCV on n=17 real}.
Saves per-metal best model and full grid to:
  analysis_results/enrichment/s4_v2_full_grid.csv
  ReProcessV1/v2_results.json
"""

from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, LeaveOneOut
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
    MultiHeadAttention, Reshape, LayerNormalization, GlobalAveragePooling1D,
    Multiply, Add, Lambda,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT_CSV = ROOT / "results/s4_v2_full_grid.csv"
OUT_JSON = ROOT / "ReProcessV1/v2_results.json"

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


def build_cnn_gnn_mlp(nf):
    inp = Input(shape=(nf,))
    x_cnn = Reshape((nf, 1))(inp)
    x_cnn = Conv1D(32, 3, activation="relu", padding="same")(x_cnn)
    x_cnn = MaxPooling1D(2)(x_cnn)
    x_cnn = Conv1D(64, 3, activation="relu", padding="same")(x_cnn)
    x_cnn = Flatten()(x_cnn); x_cnn = Dense(64, activation="relu")(x_cnn)
    x_mlp = Dense(64, activation="relu")(inp); x_mlp = Dropout(0.3)(x_mlp); x_mlp = Dense(32, activation="relu")(x_mlp)
    x = Concatenate()([x_cnn, x_mlp]); x = Dense(64, activation="relu")(x); x = Dropout(0.3)(x)
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


def build_ae_mlp(nf):
    inp = Input(shape=(nf,))
    x = Dense(64, activation="relu")(inp); x = Dropout(0.3)(x); x = Dense(32, activation="relu")(x)
    bn = Dense(16, activation="relu", name="bottleneck")(x)
    dec = Dense(32, activation="relu")(bn); dec = Dense(64, activation="relu")(dec)
    recon = Dense(nf, name="reconstruction")(dec)
    reg = Dense(32, activation="relu")(bn); reg = Dropout(0.2)(reg)
    pred = Dense(1, name="prediction")(reg)
    m = Model(inp, [pred, recon])
    m.compile(optimizer=Adam(0.001), loss={"prediction": "mse", "reconstruction": "mse"},
              loss_weights={"prediction": 1.0, "reconstruction": 0.1},
              metrics={"prediction": "mae"})
    return m


def build_moe(nf):
    inp = Input(shape=(nf,))
    e1 = Dense(64, activation="relu")(inp); e1 = Dense(32, activation="relu")(e1); e1o = Dense(1)(e1)
    xc = Reshape((nf, 1))(inp); xc = Conv1D(32, 3, activation="relu", padding="same")(xc)
    xc = GlobalAveragePooling1D()(xc); e2o = Dense(1)(xc)
    e3 = Dense(128, activation="relu")(inp); e3 = Dropout(0.3)(e3); e3 = Dense(64, activation="relu")(e3); e3 = Dense(32, activation="relu")(e3)
    e3o = Dense(1)(e3)
    gate = Dense(32, activation="relu")(inp); gate = Dense(3, activation="softmax")(gate)
    experts = Concatenate()([e1o, e2o, e3o]); weighted = Multiply()([experts, gate])
    out = Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True))(weighted)
    m = Model(inp, out); m.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
    return m


DL = {
    "Transformer CNN GNN MLP": build_transformer_cnn,
    "CNN GNN MLP":              build_cnn_gnn_mlp,
    "Dual Attention":           build_dual_attn,
    "GNN MLP AE":               build_ae_mlp,
    "Mixture of Experts":       build_moe,
}


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


ML_NAMES = ["Random Forest", "XGBoost", "SVR"]
ALL_MODELS = list(DL.keys()) + ML_NAMES


def train_and_predict(model_name, Xtr, ytr, Xte, yte=None, seed_offset=0):
    if model_name in DL:
        tf.random.set_seed(42 + seed_offset)
        m = DL[model_name](Xtr.shape[1])
        if model_name == "GNN MLP AE":
            m.fit(Xtr, [ytr, Xtr], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
                  validation_split=0.1,
                  callbacks=[EarlyStopping(monitor="val_prediction_loss", patience=PATIENCE,
                                           restore_best_weights=True, mode="min")])
            return m.predict(Xte, verbose=0)[0].flatten()
        m.fit(Xtr, ytr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
              validation_split=0.1,
              callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)])
        return m.predict(Xte, verbose=0).flatten()
    m = get_ml(model_name); m.fit(Xtr, ytr); return m.predict(Xte)


def kfold_one_metal(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    out = {n: [] for n in ALL_MODELS}
    for fold, (tri, tei) in enumerate(kf.split(X)):
        Xtr, Xte = X[tri], X[tei]; ytr, yte = y[tri], y[tei]
        sc = StandardScaler(); Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)
        for name in ALL_MODELS:
            yp = train_and_predict(name, Xtr_s, ytr, Xte_s, yte, seed_offset=fold)
            out[name].append({"R2": r2_score(yte, yp),
                              "RMSE": np.sqrt(mean_squared_error(yte, yp)),
                              "MAE": mean_absolute_error(yte, yp)})
    return out


def loocv_one_metal(X, y):
    """LOOCV on the first 17 (real) stations only."""
    Xr, yr = X[:N_REAL], y[:N_REAL]
    out = {n: {"trues": [], "preds": []} for n in ALL_MODELS}
    for tri, tei in LeaveOneOut().split(Xr):
        Xtr, Xte = Xr[tri], Xr[tei]; ytr, yte = yr[tri], yr[tei]
        sc = StandardScaler(); Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)
        for name in ALL_MODELS:
            yp = train_and_predict(name, Xtr_s, ytr, Xte_s, yte)
            out[name]["preds"].append(float(yp[0])); out[name]["trues"].append(float(yte[0]))
    res = {}
    for name in ALL_MODELS:
        t = np.array(out[name]["trues"]); p = np.array(out[name]["preds"])
        res[name] = {"R2": float(r2_score(t, p)),
                     "RMSE": float(np.sqrt(mean_squared_error(t, p))),
                     "MAE": float(mean_absolute_error(t, p))}
    return res


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    rainy = pd.read_csv(DATA / "sedimentRO_v2.csv")
    winter = pd.read_csv(DATA / "sedimentWO_v2.csv")
    feat_r = [c for c in rainy.columns if c not in METADATA + RAINY_CHEM]
    feat_w = [c for c in winter.columns if c not in METADATA + WINTER_CHEM]
    X_r = rainy[feat_r].fillna(0).values.astype(np.float32)
    X_w = winter[feat_w].fillna(0).values.astype(np.float32)
    print(f"Rainy features: {len(feat_r)}, samples: {len(X_r)}")
    print(f"Winter features: {len(feat_w)}, samples: {len(X_w)}")

    results = {}
    rows = []
    for metal, (col_r, col_w) in TARGETS.items():
        results[metal] = {}
        for season, X, prim_col in [("Rainy", X_r, col_r), ("Winter", X_w, col_w)]:
            df = rainy if season == "Rainy" else winter
            y = df[prim_col].values.astype(np.float32)
            print(f"\n=== {metal} {season} ===")
            kf_res  = kfold_one_metal(X, y)
            loo_res = loocv_one_metal(X, y)

            best_kf = max(kf_res.items(), key=lambda kv: np.mean([f["R2"] for f in kv[1]]))
            best_loo = max(loo_res.items(), key=lambda kv: kv[1]["R2"])

            results[metal][season] = {
                "kfold": {n: {
                    "R2_mean":   float(np.mean([f["R2"] for f in folds])),
                    "R2_std":    float(np.std([f["R2"] for f in folds])),
                    "RMSE_mean": float(np.mean([f["RMSE"] for f in folds])),
                    "MAE_mean":  float(np.mean([f["MAE"] for f in folds])),
                } for n, folds in kf_res.items()},
                "loocv": loo_res,
                "best_kfold_model":  best_kf[0],
                "best_kfold_R2":     float(np.mean([f["R2"] for f in best_kf[1]])),
                "best_loocv_model":  best_loo[0],
                "best_loocv_R2":     best_loo[1]["R2"],
            }

            for name in ALL_MODELS:
                kf_r2 = float(np.mean([f["R2"] for f in kf_res[name]]))
                kf_std = float(np.std([f["R2"] for f in kf_res[name]]))
                kf_rmse = float(np.mean([f["RMSE"] for f in kf_res[name]]))
                rows.append({
                    "Metal": metal, "Season": season, "Model": name,
                    "5fold_R2_mean":  round(kf_r2, 3),
                    "5fold_R2_std":   round(kf_std, 3),
                    "5fold_RMSE":     round(kf_rmse, 2),
                    "LOOCV_R2":       round(loo_res[name]["R2"], 3),
                    "LOOCV_RMSE":     round(loo_res[name]["RMSE"], 2),
                })
            print(f"  best 5-fold: {best_kf[0]:25s} R²={results[metal][season]['best_kfold_R2']:.3f}")
            print(f"  best LOOCV:  {best_loo[0]:25s} R²={results[metal][season]['best_loocv_R2']:.3f}")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_JSON.name} and {OUT_CSV.name}")


if __name__ == "__main__":
    main()
