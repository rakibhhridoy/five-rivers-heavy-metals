"""LOOCV per-station predictions (sklearn-only) for Fig 7 regeneration.

Reproduces the ML LOOCV columns of analysis_results/enrichment/s4_v2_loocv.csv
and emits per-station predicted/observed pairs for plotting.

Output: analysis_results/enrichment/s4_v2_loocv_preds.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "results/s4_v2_loocv_preds.csv"

N_REAL = 17
METADATA = ["Stations", "River", "Lat", "Long", "geometry"]
RAINY_CHEM  = ["CrR", "NiR", "CuR", "AsR", "CdR", "PbR", "MR",
               "SandR", "SiltR", "ClayR", "FeR", "RI"]
WINTER_CHEM = ["CdW", "CrW", "AsW", "PbW", "CuW", "NiW", "FeW", "MW",
               "SandW", "SiltW", "ClayW", "RI"]

TARGETS = {
    "Cd": ("CdR", "CdW"), "Cr": ("CrR", "CrW"), "As": ("AsR", "AsW"),
    "Pb": ("PbR", "PbW"), "Cu": ("CuR", "CuW"), "Ni": ("NiR", "NiW"),
    "Fe": ("FeR", "FeW"),
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


def loocv(X17, y17, model_name):
    preds = np.zeros_like(y17, dtype=float)
    for tri, tei in LeaveOneOut().split(X17):
        Xtr, Xte = X17[tri], X17[tei]
        ytr = y17[tri]
        sc = StandardScaler()
        Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)
        m = get_ml(model_name)
        m.fit(Xtr_s, ytr)
        preds[tei] = m.predict(Xte_s)
    return preds


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
        for season, X_full, prim_col, source in [
            ("Rainy", X_r, col_r, rainy), ("Winter", X_w, col_w, winter)
        ]:
            y = source[prim_col].values.astype(np.float32)
            stations = source["Stations"].values[:N_REAL]
            rivers = source["River"].values[:N_REAL]
            X17, y17 = X_full[:N_REAL], y[:N_REAL]
            best_r2 = -np.inf
            best_name = None
            best_preds = None
            for name in ["XGBoost", "Random Forest", "SVR"]:
                preds = loocv(X17, y17, name)
                r2 = r2_score(y17, preds)
                if r2 > best_r2:
                    best_r2, best_name, best_preds = r2, name, preds
            print(f"  {metal:3s} {season:7s}  best={best_name:14s}  R2={best_r2:+.3f}")
            for i in range(N_REAL):
                rows.append({
                    "Metal": metal, "Season": season,
                    "Station": stations[i], "River": rivers[i],
                    "Observed": float(y17[i]),
                    "Predicted": float(best_preds[i]),
                    "Best_ML": best_name,
                    "LOOCV_R2": round(best_r2, 3),
                })

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
