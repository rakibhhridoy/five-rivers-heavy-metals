"""PCA with varimax rotation on primary n=34 sediment matrix (§3.6, Supplementary S6).

Reproduces the manuscript's three rotated components:
  RC1: Pb 0.95, Ni 0.91   — vehicular / metalworking
  RC2: Cr 0.95, Cu 0.72   — tannery + textile
  RC3: As 0.98            — geogenic / pesticide

Inputs
------
data/sedimentRO_v2.csv, data/sedimentWO_v2.csv  — primary 17 stations × 2 seasons

Output
------
results/s_pca_loadings.csv      — varimax-rotated loadings (metal × RC1/RC2/RC3)
results/s_pca_explained_var.csv — variance explained by each rotated component
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT_LOAD = ROOT / "results" / "s_pca_loadings.csv"
OUT_VAR = ROOT / "results" / "s_pca_explained_var.csv"

METALS = ["Cd", "Cr", "As", "Pb", "Cu", "Ni", "Fe"]
METAL_COLS_R = {"Cd": "CdR", "Cr": "CrR", "As": "AsR", "Pb": "PbR",
                "Cu": "CuR", "Ni": "NiR", "Fe": "FeR"}
METAL_COLS_W = {"Cd": "CdW", "Cr": "CrW", "As": "AsW", "Pb": "PbW",
                "Cu": "CuW", "Ni": "NiW", "Fe": "FeW"}


def varimax(loadings: np.ndarray, gamma: float = 1.0, max_iter: int = 100,
            tol: float = 1e-6) -> np.ndarray:
    """Kaiser's varimax rotation. Returns rotated loadings."""
    p, k = loadings.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        d_old = d
        L = loadings @ R
        u, s, vh = np.linalg.svd(
            loadings.T @ (L ** 3 - (gamma / p) * L @ np.diag(np.diag(L.T @ L)))
        )
        R = u @ vh
        d = s.sum()
        if d_old != 0 and abs(d - d_old) / d_old < tol:
            break
    return loadings @ R


def main() -> None:
    sed_r = pd.read_csv(DATA / "sedimentRO_v2.csv").iloc[:17]
    sed_w = pd.read_csv(DATA / "sedimentWO_v2.csv").iloc[:17]

    rainy = sed_r[[METAL_COLS_R[m] for m in METALS]].copy()
    rainy.columns = METALS
    winter = sed_w[[METAL_COLS_W[m] for m in METALS]].copy()
    winter.columns = METALS
    X = pd.concat([rainy, winter], ignore_index=True)  # n=34

    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3, random_state=42)
    pca.fit(Xs)

    # Unrotated loadings: components_.T scaled by sqrt(eigenvalue)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    rotated = varimax(loadings)

    # Sign-align so the dominant element of each component is positive
    for k in range(rotated.shape[1]):
        if rotated[np.argmax(np.abs(rotated[:, k])), k] < 0:
            rotated[:, k] *= -1

    # Order components by total absolute loading (largest first = "RC1")
    order = np.argsort(-(rotated ** 2).sum(axis=0))
    rotated = rotated[:, order]

    df = pd.DataFrame(rotated, index=METALS, columns=["RC1", "RC2", "RC3"])
    df.index.name = "Metal"
    df.to_csv(OUT_LOAD)

    # Rotated variance explained = column-sum-of-squares / total variance
    rot_var = (rotated ** 2).sum(axis=0) / Xs.shape[1]
    var_df = pd.DataFrame({
        "Component": ["RC1", "RC2", "RC3"],
        "Var_explained": rot_var,
        "Cumulative": np.cumsum(rot_var),
    })
    var_df.to_csv(OUT_VAR, index=False)

    print(f"Wrote {OUT_LOAD.name} and {OUT_VAR.name}")
    print(f"\nVarimax-rotated loadings (n=34, {len(METALS)} metals):")
    print(df.round(3).to_string())
    print(f"\nCumulative variance: {var_df['Cumulative'].iloc[-1]:.3f}  "
          f"(manuscript reports 0.886)")


if __name__ == "__main__":
    main()
