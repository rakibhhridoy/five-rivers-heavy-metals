"""Bootstrap CI on Cd LOOCV R^2 (n=17, SVR, real GEE V1 AE).

Resamples the 17 stations with replacement, recomputes LOOCV R^2 each replicate,
reports 95% CI. Cd is the only metal where bootstrap CI is meaningful; for
other metals LOOCV R^2 is non-positive and CI would be uninformative.

Output: analysis_results/enrichment/s4_v2_cd_bootstrap.csv (one row per season)
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
OUT = ROOT / "results/s4_v2_cd_bootstrap.csv"

N_REAL = 17
N_BOOT = 1000
RNG = np.random.default_rng(42)

METADATA = ["Stations", "River", "Lat", "Long", "geometry"]
RAINY_CHEM  = ["CrR", "NiR", "CuR", "AsR", "CdR", "PbR", "MR",
               "SandR", "SiltR", "ClayR", "FeR", "RI"]
WINTER_CHEM = ["CdW", "CrW", "AsW", "PbW", "CuW", "NiW", "FeW", "MW",
               "SandW", "SiltW", "ClayW", "RI"]


def loocv_r2(X, y):
    preds = np.zeros_like(y, dtype=float)
    for tri, tei in LeaveOneOut().split(X):
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(X[tri])
        Xte_s = sc.transform(X[tei])
        m = SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")
        m.fit(Xtr_s, y[tri])
        preds[tei] = m.predict(Xte_s)
    return r2_score(y, preds), preds


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rainy = pd.read_csv(DATA / "sedimentRO_v2.csv")
    winter = pd.read_csv(DATA / "sedimentWO_v2.csv")
    feat_r = [c for c in rainy.columns if c not in METADATA + RAINY_CHEM]
    feat_w = [c for c in winter.columns if c not in METADATA + WINTER_CHEM]
    X_r = rainy[feat_r].fillna(0).values.astype(np.float32)[:N_REAL]
    X_w = winter[feat_w].fillna(0).values.astype(np.float32)[:N_REAL]
    y_r = rainy["CdR"].values.astype(np.float32)[:N_REAL]
    y_w = winter["CdW"].values.astype(np.float32)[:N_REAL]

    rows = []
    for season, X, y in [("Rainy", X_r, y_r), ("Winter", X_w, y_w)]:
        point_r2, _ = loocv_r2(X, y)
        boot_r2 = []
        for b in range(N_BOOT):
            idx = RNG.integers(0, N_REAL, N_REAL)
            if len(np.unique(idx)) < 5:
                continue
            try:
                r2, _ = loocv_r2(X[idx], y[idx])
                boot_r2.append(r2)
            except Exception:
                continue
        boot_r2 = np.array(boot_r2)
        ci_lo = float(np.percentile(boot_r2, 2.5))
        ci_hi = float(np.percentile(boot_r2, 97.5))
        median = float(np.median(boot_r2))
        rows.append({
            "Metal": "Cd", "Season": season, "Model": "SVR",
            "Point_LOOCV_R2": round(point_r2, 3),
            "Bootstrap_median_R2": round(median, 3),
            "CI95_low": round(ci_lo, 3),
            "CI95_high": round(ci_hi, 3),
            "n_boot_used": int(len(boot_r2)),
        })
        print(f"  Cd {season}: point R2={point_r2:+.3f}  "
              f"boot median={median:+.3f}  95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]  "
              f"(n_boot={len(boot_r2)})")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
