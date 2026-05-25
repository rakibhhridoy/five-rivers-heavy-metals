"""
Phase 5 — Augmented retrain. Train each metal's best architecture on
(primary 117 + secondary 56) with sample weights primary=1.0, secondary=0.5.

Validation: 5-fold CV on primary stations only (LOOCV is skipped per user
direction). Reports per-metal R² before vs after augmentation, plus a delta
table.

Output: ReProcessV1/augmented_results.json
        analysis_results/enrichment/s5_augmentation_delta.csv
"""

from pathlib import Path
import json
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
SECONDARY = ROOT / "data/secondary"
OUT_JSON = ROOT / "ReProcessV1/augmented_results.json"
OUT_CSV  = ROOT / "results/s5_augmentation_delta.csv"

EPOCHS = 100
BATCH_SIZE = 8
PATIENCE = 15

METADATA_COLS = ["Stations", "River", "Lat", "Long", "geometry"]
RAINY_CHEM  = ["CrR", "NiR", "CuR", "AsR", "CdR", "PbR", "MR", "SandR", "SiltR", "ClayR", "FeR", "RI"]
WINTER_CHEM = ["CdW", "CrW", "AsW", "PbW", "CuW", "NiW", "FeW", "MW", "SandW", "SiltW", "ClayW", "RI"]

TARGETS = {
    "Cd": ("CdR", "CdW"), "Cr": ("CrR", "CrW"), "As": ("AsR", "AsW"),
    "Pb": ("PbR", "PbW"), "Cu": ("CuR", "CuW"), "Ni": ("NiR", "NiW"),
    "Fe": ("FeR", "FeW"),
}

BEST_MODEL = {
    "Cd": "Dual Attention", "Cr": "Transformer CNN", "As": "Transformer CNN",
    "Pb": "Transformer CNN", "Cu": "Transformer CNN", "Ni": "Transformer CNN",
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


DL_BUILDERS = {"Transformer CNN": build_transformer_cnn, "Dual Attention": build_dual_attn}


def get_ml_model(name):
    if name == "XGBoost":
        return GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                          min_samples_leaf=3, random_state=42)
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=3,
                                      random_state=42, n_jobs=-1)
    if name == "SVR":
        return SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")
    raise ValueError(name)


def calc_metrics(yt, yp):
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) < 2:
        return {"R2": float("nan"), "RMSE": float("nan"), "MAE": float("nan")}
    return {
        "R2":   float(r2_score(yt, yp)),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE":  float(mean_absolute_error(yt, yp)),
    }


def cv_train(X_train, y_train, X_eval, y_eval, model_name, weights=None):
    """Train on (X_train, y_train) with optional sample weights, predict on X_eval."""
    sc = StandardScaler(); Xtr = sc.fit_transform(X_train); Xev = sc.transform(X_eval)
    if model_name in DL_BUILDERS:
        tf.random.set_seed(42)
        m = DL_BUILDERS[model_name](Xtr.shape[1])
        # split off small val for early-stopping
        n_val = max(2, int(0.1 * len(Xtr))); rng = np.random.RandomState(42); idx = rng.permutation(len(Xtr))
        vi, ti = idx[:n_val], idx[n_val:]
        sw = None if weights is None else weights[ti]
        m.fit(Xtr[ti], y_train[ti], sample_weight=sw, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
              validation_data=(Xtr[vi], y_train[vi]),
              callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)])
        return m.predict(Xev, verbose=0).flatten()
    m = get_ml_model(model_name)
    if weights is not None:
        m.fit(Xtr, y_train, sample_weight=weights)
    else:
        m.fit(Xtr, y_train)
    return m.predict(Xev)


def build_features(df, chem_cols):
    exclude = METADATA_COLS + chem_cols
    feat_cols = [c for c in df.columns if c not in exclude]
    return feat_cols, df[feat_cols].fillna(0).values.astype(np.float32)


def build_secondary_X(sec_merged, feat_cols):
    cols = []
    for c in feat_cols:
        if c in sec_merged.columns:
            cols.append(sec_merged[c].values)
        elif c == "hydrological_dist_to_nearest_BF":
            cols.append(sec_merged["hydro_dist_brick"].values)
        elif c == "num_upstream_BF":
            cols.append(sec_merged["num_brick_field"].values)
        elif c == "hydrological_dist_to_nearest_IND":
            cols.append(sec_merged["hydro_dist_ind"].values)
        elif c == "num_upstream_IND":
            cols.append(sec_merged["num_industry"].values)
        else:
            raise KeyError(c)
    return np.column_stack(cols).astype(np.float32)


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    rainy  = pd.read_csv(DATA / "sedimentRO_v2.csv")
    winter = pd.read_csv(DATA / "sedimentWO_v2.csv")
    sec_prox = pd.read_csv(SECONDARY / "secondary_proximity_features.csv")
    sec_ae   = pd.read_csv(SECONDARY / "secondary_ae_features.csv")
    sec_chem = pd.read_csv(SECONDARY / "harmonized_sediment.csv")

    rainy_feat_cols,  X_rainy  = build_features(rainy,  RAINY_CHEM)
    winter_feat_cols, X_winter = build_features(winter, WINTER_CHEM)
    sec_merged = sec_prox.merge(sec_ae[["Station_ID"] + [f"AE_{i:02d}" for i in range(64)]], on="Station_ID")
    X_sec_rainy  = build_secondary_X(sec_merged, rainy_feat_cols)
    X_sec_winter = build_secondary_X(sec_merged, winter_feat_cols)

    rows = []
    out_json = {}
    for metal, (col_r, col_w) in TARGETS.items():
        model_name = BEST_MODEL[metal]
        out_json[metal] = {}
        for season, X_prim, prim_col, X_sec in [
            ("Rainy",  X_rainy,  col_r, X_sec_rainy),
            ("Winter", X_winter, col_w, X_sec_winter),
        ]:
            df_prim = rainy if season == "Rainy" else winter
            y_prim = df_prim[prim_col].values.astype(np.float32)

            # Secondary observations for this metal & season
            sec_subset = sec_chem[(sec_chem["Metal"] == metal) & (sec_chem["Season_Norm"] == season)]
            sec_obs_map = sec_subset.set_index("Station_ID")["Concentration"]
            station_idx = {sid: i for i, sid in enumerate(sec_merged["Station_ID"].values)}
            sec_X_rows = []; sec_y_rows = []
            for sid, val in sec_obs_map.items():
                if sid in station_idx and np.isfinite(val):
                    sec_X_rows.append(X_sec[station_idx[sid]]); sec_y_rows.append(val)
            if not sec_X_rows:
                continue
            X_sec_metal = np.array(sec_X_rows, dtype=np.float32)
            y_sec_metal = np.array(sec_y_rows, dtype=np.float32)

            # 5-fold CV on primary stations
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            base_r2s = []; aug_r2s = []
            for fold, (tri, tei) in enumerate(kf.split(X_prim)):
                Xtr_p, Xev_p = X_prim[tri], X_prim[tei]
                ytr_p, yev_p = y_prim[tri], y_prim[tei]

                # Baseline: train on primary only
                yp_base = cv_train(Xtr_p, ytr_p, Xev_p, yev_p, model_name)
                base_r2s.append(r2_score(yev_p, yp_base))

                # Augmented: append secondary, weighted 0.5
                X_aug = np.vstack([Xtr_p, X_sec_metal])
                y_aug = np.concatenate([ytr_p, y_sec_metal])
                w_aug = np.concatenate([np.ones(len(Xtr_p)), 0.5 * np.ones(len(X_sec_metal))]).astype(np.float32)
                yp_aug = cv_train(X_aug, y_aug, Xev_p, yev_p, model_name, weights=w_aug)
                aug_r2s.append(r2_score(yev_p, yp_aug))

            base_mean = float(np.mean(base_r2s)); aug_mean = float(np.mean(aug_r2s))
            delta = aug_mean - base_mean
            out_json[metal][season] = {
                "model": model_name,
                "n_primary": int(len(X_prim)),
                "n_secondary": int(len(X_sec_metal)),
                "baseline_5fold_R2": base_mean,
                "augmented_5fold_R2": aug_mean,
                "delta_R2": delta,
                "delta_pct_relative": (delta / abs(base_mean) * 100.0) if base_mean != 0 else None,
            }
            rows.append({
                "Metal": metal, "Season": season, "Model": model_name,
                "n_secondary": len(X_sec_metal),
                "Baseline_R2": round(base_mean, 3),
                "Augmented_R2": round(aug_mean, 3),
                "Delta_R2":     round(delta, 3),
                "Improved":     bool(delta > 0),
            })
            print(f"  {metal:3s} {season:7s} {model_name:18s}  "
                  f"baseline={base_mean:.3f}  augmented={aug_mean:.3f}  Δ={delta:+.3f}  "
                  f"n_sec={len(X_sec_metal)}")

    OUT_JSON.write_text(json.dumps(out_json, indent=2))
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    n_improved = sum(r["Improved"] for r in rows)
    n_improved_05 = sum(1 for r in rows if r["Delta_R2"] >= 0.05)
    print(f"\n=== Phase 5 summary ===")
    print(f"  Improved (Δ>0):     {n_improved}/{len(rows)}")
    print(f"  Materially improved (Δ≥0.05): {n_improved_05}/{len(rows)}")
    print(f"  Wrote {OUT_JSON.name} and {OUT_CSV.name}")


if __name__ == "__main__":
    main()
