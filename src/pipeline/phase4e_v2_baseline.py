"""
Phase 4e — 5-fold CV baseline on v2 (real AlphaEarth) primary data.

For each metal × season, trains the manuscript's best architecture on the
v2 primary data and reports 5-fold CV R². Side-by-side comparison vs the
manuscript's published numbers (from synthetic AE) is included for transparency.

Output: analysis_results/enrichment/s4_v2_baseline.csv
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
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
OUT_CSV = ROOT / "results/s4_v2_baseline.csv"

EPOCHS = 100
BATCH_SIZE = 8
PATIENCE = 15

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

MANU_R2 = {
    "Cd": (0.944, 0.937), "Cr": (0.932, 0.916), "As": (0.899, 0.954),
    "Pb": (0.940, 0.929), "Cu": (0.910, 0.902), "Ni": (0.909, 0.919),
    "Fe": (0.803, 0.714),
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


def cv_r2(X, y, model_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2s = []
    for fold, (tri, tei) in enumerate(kf.split(X)):
        Xtr, Xte = X[tri], X[tei]
        ytr, yte = y[tri], y[tei]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

        if model_name in DL:
            tf.random.set_seed(42 + fold)
            m = DL[model_name](Xtr_s.shape[1])
            m.fit(Xtr_s, ytr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
                  validation_data=(Xte_s, yte),
                  callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)])
            yp = m.predict(Xte_s, verbose=0).flatten()
        else:
            m = get_ml(model_name); m.fit(Xtr_s, ytr); yp = m.predict(Xte_s)

        r2s.append(r2_score(yte, yp))
    return float(np.mean(r2s)), float(np.std(r2s))


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rainy = pd.read_csv(DATA / "sedimentRO_v2.csv")
    winter = pd.read_csv(DATA / "sedimentWO_v2.csv")

    feat_r = [c for c in rainy.columns if c not in METADATA + RAINY_CHEM]
    feat_w = [c for c in winter.columns if c not in METADATA + WINTER_CHEM]
    X_r = rainy[feat_r].fillna(0).values.astype(np.float32)
    X_w = winter[feat_w].fillna(0).values.astype(np.float32)
    print(f"Rainy features: {len(feat_r)}, samples: {len(X_r)}")
    print(f"Winter features: {len(feat_w)}, samples: {len(X_w)}")

    rows = []
    for metal, (col_r, col_w) in TARGETS.items():
        model_name = BEST_MODEL[metal]
        manu_r, manu_w = MANU_R2[metal]
        for season, X, prim_col, manu in [("Rainy", X_r, col_r, manu_r), ("Winter", X_w, col_w, manu_w)]:
            df = rainy if season == "Rainy" else winter
            y = df[prim_col].values.astype(np.float32)
            r2_mean, r2_std = cv_r2(X, y, model_name)
            delta = r2_mean - manu
            rows.append({
                "Metal": metal, "Season": season, "Model": model_name,
                "v2_5fold_R2": round(r2_mean, 3),
                "v2_5fold_R2_std": round(r2_std, 3),
                "Manuscript_R2_synthetic_AE": manu,
                "Delta_v2_vs_manuscript": round(delta, 3),
            })
            print(f"  {metal:3s} {season:7s} {model_name:18s}  v2_5fold={r2_mean:.3f}±{r2_std:.3f}  "
                  f"manuscript={manu:.3f}  Δ={delta:+.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")
    print(f"\nSummary:")
    print(f"  Improved over manuscript number: {sum(1 for r in rows if r['Delta_v2_vs_manuscript'] > 0)}/{len(rows)}")
    print(f"  Median Δ: {np.median([r['Delta_v2_vs_manuscript'] for r in rows]):+.3f}")


if __name__ == "__main__":
    main()
