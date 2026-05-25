"""SVR_pooled_log bootstrap CI for Pb (matches Cd methodology in task #10).

GP point estimates are reported in the manuscript; SVR bootstrap is reported
as a model-agnostic sensitivity check, consistent with the existing Cd bootstrap.

Output: analysis_results/enrichment/s9_pb_bootstrap.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.svm import SVR
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "results/s9_pb_bootstrap.csv"

PROX = ["dist_brick", "num_brick", "dist_ind", "num_ind"]
AE_COLS = [f"AE_{i:02d}" for i in range(64)]
CLIMATE = ["rain_30d_mm", "rain_7d_mm", "temp_30d_C", "temp_7d_C"]
FEATURES = PROX + AE_COLS + ["is_winter"] + CLIMATE
N_BOOT = 1000
RNG = np.random.default_rng(42)


def loocv_r2(X, y_log, y_raw):
    preds_log = np.zeros_like(y_log)
    for tri, tei in LeaveOneOut().split(X):
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(X[tri])
        Xte_s = sc.transform(X[tei])
        m = SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")
        m.fit(Xtr_s, y_log[tri])
        preds_log[tei] = m.predict(Xte_s)
    return r2_score(y_raw, np.expm1(preds_log))


def main():
    pooled = pd.read_csv(DATA / "sediment_pooled_v2.csv")
    climate = pd.read_csv(DATA / "climate_covariates_v2.csv")
    df = pooled.merge(climate[["station_id", "is_winter"] + CLIMATE],
                      on=["station_id", "is_winter"], how="left")

    rows = []
    for season_label, sw in [("Rainy", 0), ("Winter", 1)]:
        sub = df[df["is_winter"] == sw].reset_index(drop=True)
        X = sub[FEATURES].values.astype(np.float32)
        y_log = sub["log_Pb"].values.astype(float)
        y_raw = sub["Pb"].values.astype(float)
        n = len(y_raw)

        point_r2 = loocv_r2(X, y_log, y_raw)

        boot = []
        for b in range(N_BOOT):
            idx = RNG.integers(0, n, n)
            if len(np.unique(idx)) < 5:
                continue
            try:
                boot.append(loocv_r2(X[idx], y_log[idx], y_raw[idx]))
            except Exception:
                continue
        boot = np.array(boot)
        rows.append({
            "Metal": "Pb", "Season": season_label,
            "Model": "SVR_pooled_log_climate",
            "Point_LOOCV_R2_SVR": round(point_r2, 3),
            "Bootstrap_median": round(float(np.median(boot)), 3),
            "CI95_low": round(float(np.percentile(boot, 2.5)), 3),
            "CI95_high": round(float(np.percentile(boot, 97.5)), 3),
            "n_boot": int(len(boot)),
        })
        print(f"  Pb {season_label}: SVR point = {point_r2:+.3f}, "
              f"bootstrap median = {np.median(boot):+.3f}, "
              f"95% CI [{np.percentile(boot, 2.5):+.3f}, "
              f"{np.percentile(boot, 97.5):+.3f}]")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
