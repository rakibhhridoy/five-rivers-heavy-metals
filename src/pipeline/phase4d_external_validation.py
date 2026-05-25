"""
Phase 4d — External validation: train on primary v2 sediment data, predict
on geocoded secondary stations, compute external R² per metal.

For each metal × season:
  1. Train on full primary v2 (117 stations) using the manuscript's reported
     best architecture.
  2. Predict on the secondary stations that have a measured value for that
     metal in that season.
  3. Compute external R², RMSE, MAE.
  4. Compare to the manuscript's 5-fold CV R².
  5. Apply Gate G3: |External R² − CV R²| < 0.15 for ≥5 of 8 metals → augmentation justified.

Output: analysis_results/enrichment/s2_external_validation.csv
"""

from pathlib import Path
import json
import os
import sys
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
    MultiHeadAttention, Reshape, LayerNormalization, GlobalAveragePooling1D,
    Multiply, Add, Lambda,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SECONDARY = ROOT / "data/secondary"
OUT_CSV = ROOT / "results/s2_external_validation.csv"

EPOCHS = 100
BATCH_SIZE = 8
PATIENCE = 15

METADATA_COLS = ["Stations", "River", "Lat", "Long", "geometry"]
RAINY_CHEM = ["CrR", "NiR", "CuR", "AsR", "CdR", "PbR", "MR", "SandR", "SiltR", "ClayR", "FeR", "RI"]
WINTER_CHEM = ["CdW", "CrW", "AsW", "PbW", "CuW", "NiW", "FeW", "MW", "SandW", "SiltW", "ClayW", "RI"]

# Primary target columns per metal × season
TARGETS = {
    "Cd": ("CdR", "CdW"), "Cr": ("CrR", "CrW"), "As": ("AsR", "AsW"),
    "Pb": ("PbR", "PbW"), "Cu": ("CuR", "CuW"), "Ni": ("NiR", "NiW"),
    "Fe": ("FeR", "FeW"),
}

# Best architecture per metal from the manuscript Table 4
BEST_MODEL = {
    "Cd": "Dual Attention",
    "Cr": "Transformer CNN",
    "As": "Transformer CNN",
    "Pb": "Transformer CNN",
    "Cu": "Transformer CNN",
    "Ni": "Transformer CNN",   # rainy; winter best is CNN-GNN-MLP, but TC is close — use one for simplicity
    "Fe": "XGBoost",
}

# Manuscript's reported 5-fold R² per metal (for delta comparison)
MANU_5FOLD_R2 = {
    "Cd": (0.944, 0.937), "Cr": (0.932, 0.916), "As": (0.899, 0.954),
    "Pb": (0.940, 0.929), "Cu": (0.910, 0.902), "Ni": (0.909, 0.919),
    "Fe": (0.803, 0.714),
}


# ===== Model builders =====
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
        return {"R2": float("nan"), "RMSE": float("nan"), "MAE": float("nan"), "n": int(len(yt))}
    return {
        "R2":   float(r2_score(yt, yp)),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE":  float(mean_absolute_error(yt, yp)),
        "n":    int(len(yt)),
    }


def cv_r2(X, y, model_name):
    """5-fold CV R² mean, on the v2 features."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2s = []
    for fold, (tri, tei) in enumerate(kf.split(X)):
        Xtr, Xte, ytr, yte = X[tri], X[tei], y[tri], y[tei]
        sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)

        if model_name in DL_BUILDERS:
            tf.random.set_seed(42 + fold)
            m = DL_BUILDERS[model_name](Xtr.shape[1])
            m.fit(Xtr, ytr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
                  validation_data=(Xte, yte),
                  callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)])
            yp = m.predict(Xte, verbose=0).flatten()
        else:
            m = get_ml_model(model_name); m.fit(Xtr, ytr); yp = m.predict(Xte)

        r2s.append(r2_score(yte, yp))
    return float(np.mean(r2s)), float(np.std(r2s))


def fit_full_and_predict(X, y, model_name, X_ext):
    sc = StandardScaler(); X_s = sc.fit_transform(X); X_ext_s = sc.transform(X_ext)
    if model_name in DL_BUILDERS:
        tf.random.set_seed(42)
        m = DL_BUILDERS[model_name](X_s.shape[1])
        # use 10% of training for early-stopping val
        n_val = max(2, int(0.1 * len(X_s)))
        rng = np.random.RandomState(42); idx = rng.permutation(len(X_s))
        vi, ti = idx[:n_val], idx[n_val:]
        m.fit(X_s[ti], y[ti], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
              validation_data=(X_s[vi], y[vi]),
              callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)])
        return m.predict(X_ext_s, verbose=0).flatten()
    m = get_ml_model(model_name); m.fit(X_s, y); return m.predict(X_ext_s)


def build_features(df, chem_cols):
    exclude = METADATA_COLS + chem_cols
    feat_cols = [c for c in df.columns if c not in exclude]
    return feat_cols, df[feat_cols].fillna(0).values.astype(np.float32)


def secondary_external_X(secondary_features):
    """The secondary AE+proximity columns must align with primary feature order."""
    return secondary_features


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rainy = pd.read_csv(DATA / "sedimentRO_v2.csv")
    winter = pd.read_csv(DATA / "sedimentWO_v2.csv")
    sec_prox = pd.read_csv(SECONDARY / "secondary_proximity_features.csv")
    sec_ae   = pd.read_csv(SECONDARY / "secondary_ae_features.csv")
    sec_chem = pd.read_csv(SECONDARY / "harmonized_sediment.csv")

    # Build secondary feature matrix in the same column order as primary (rainy)
    rainy_feat_cols, X_rainy = build_features(rainy, RAINY_CHEM)
    winter_feat_cols, X_winter = build_features(winter, WINTER_CHEM)

    # Map primary winter column names to rainy-style for aligned join with secondary
    # Primary winter uses "hydrological_dist_to_nearest_BF" etc. We need a single source of truth.
    # For external validation, build secondary-side features that match each season's column order.

    # Secondary stations have rainy-style proximity columns: hydro_dist_brick, num_brick_field, hydro_dist_ind, num_industry
    sec_merged = sec_prox.merge(sec_ae[["Station_ID"] + [f"AE_{i:02d}" for i in range(64)]], on="Station_ID")

    # For each season, build X_secondary in the exact column order primary uses.
    def build_secondary_X(feat_cols):
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
                raise KeyError(f"No secondary column for primary col '{c}'")
        return np.column_stack(cols).astype(np.float32)

    X_sec_rainy  = build_secondary_X(rainy_feat_cols)
    X_sec_winter = build_secondary_X(winter_feat_cols)

    print(f"Primary rainy features: {len(rainy_feat_cols)} ({X_rainy.shape})")
    print(f"Primary winter features: {len(winter_feat_cols)} ({X_winter.shape})")
    print(f"Secondary stations:     {len(sec_merged)}")

    # For each metal × season, train on primary then predict on subset of secondary that has the metal.
    rows = []
    for metal in TARGETS:
        col_r, col_w = TARGETS[metal]
        model_name = BEST_MODEL[metal]
        cv_r2_manu_r, cv_r2_manu_w = MANU_5FOLD_R2[metal]

        for season, X_prim, prim_chem_col, X_sec, manu_r2 in [
            ("Rainy",  X_rainy,  col_r, X_sec_rainy,  cv_r2_manu_r),
            ("Winter", X_winter, col_w, X_sec_winter, cv_r2_manu_w),
        ]:
            df_prim = rainy if season == "Rainy" else winter
            y_prim  = df_prim[prim_chem_col].values.astype(np.float32)

            # 5-fold CV on v2 primary features (so we can compare apples to apples to external)
            cv_mean, cv_std = cv_r2(X_prim, y_prim, model_name)

            # Train on full primary, predict on secondary
            preds = fit_full_and_predict(X_prim, y_prim, model_name, X_sec)

            # Match predictions to secondary stations that have a value for this metal & season
            sec_subset = sec_chem[(sec_chem["Metal"] == metal) & (sec_chem["Season_Norm"] == season)]
            sec_obs = sec_subset.set_index("Station_ID")["Concentration"]

            # Align secondary stations by Station_ID
            station_idx = {sid: i for i, sid in enumerate(sec_merged["Station_ID"].values)}
            ext_pairs = [(preds[station_idx[sid]], obs)
                         for sid, obs in sec_obs.items() if sid in station_idx]
            if not ext_pairs:
                ext = {"R2": float("nan"), "RMSE": float("nan"), "MAE": float("nan"), "n": 0}
            else:
                yp_arr = np.array([p[0] for p in ext_pairs])
                yt_arr = np.array([p[1] for p in ext_pairs], dtype=np.float32)
                ext = calc_metrics(yt_arr, yp_arr)

            delta_vs_manu = (ext["R2"] - manu_r2) if not np.isnan(ext["R2"]) else float("nan")
            delta_vs_v2cv = (ext["R2"] - cv_mean) if not np.isnan(ext["R2"]) else float("nan")
            gate_pass = abs(delta_vs_v2cv) < 0.15 if not np.isnan(delta_vs_v2cv) else False

            rows.append({
                "Metal": metal, "Season": season, "Model": model_name,
                "CV_R2_v2": round(cv_mean, 3), "CV_R2_v2_std": round(cv_std, 3),
                "Manu_5fold_R2": manu_r2,
                "External_R2": round(ext["R2"], 3) if not np.isnan(ext["R2"]) else None,
                "External_RMSE": round(ext["RMSE"], 3) if not np.isnan(ext["RMSE"]) else None,
                "External_MAE": round(ext["MAE"], 3) if not np.isnan(ext["MAE"]) else None,
                "External_n": ext["n"],
                "Delta_ExtCV_v2": round(delta_vs_v2cv, 3) if not np.isnan(delta_vs_v2cv) else None,
                "Delta_Ext_vs_Manu": round(delta_vs_manu, 3) if not np.isnan(delta_vs_manu) else None,
                "Gate_G3_pass": gate_pass,
            })
            print(f"  {metal:3s} {season:7s} model={model_name:18s}  "
                  f"CV_v2={cv_mean:.3f}  Manu={manu_r2:.3f}  "
                  f"Ext={ext['R2']:.3f} (n={ext['n']})  "
                  f"Δ(Ext-CVv2)={delta_vs_v2cv:+.3f}  G3={'PASS' if gate_pass else 'FAIL'}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    # Gate G3 summary
    n_pass = out["Gate_G3_pass"].sum()
    print(f"\n=== Gate G3 ===")
    print(f"  {n_pass}/16 metal×season combinations have |Ext R² − CV R²| < 0.15")
    print(f"  Required for augmentation: ≥10/16 (≥5 of 8 metals in both seasons)")
    metals_both_pass = sum(
        1 for m in TARGETS
        if all(out[(out["Metal"] == m) & (out["Season"] == s)]["Gate_G3_pass"].iloc[0] for s in ["Rainy", "Winter"])
    )
    print(f"  Metals passing in BOTH seasons: {metals_both_pass}/8")
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
