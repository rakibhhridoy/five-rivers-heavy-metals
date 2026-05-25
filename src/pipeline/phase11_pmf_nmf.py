"""PMF (NMF as proxy) sensitivity check on pooled n=71 sediment dataset.

Compares 3-component non-negative matrix factorization on the pooled
primary-plus-literature sediment dataset (n_pooled = 71 station-seasons,
8 metals) against the manuscript's primary-only PCA result (n=34, 3 components).

Reports per-metal loadings on each NMF component and labels each component by
its dominant metals.

Output: analysis_results/enrichment/s11_nmf_pooled.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SECDATA = ROOT / "data/secondary"
ENRICH = ROOT / "results"
OUT = ENRICH / "s11_nmf_pooled.csv"

METALS = ["Cd", "Cr", "As", "Pb", "Cu", "Ni", "Fe"]


def main():
    pooled = pd.read_csv(ENRICH / "s4_harmonized_indices.csv")
    print(f"Pooled dataset shape: {pooled.shape}")
    print(f"Columns: {list(pooled.columns)[:15]}")

    if "Concentration" in pooled.columns and "Metal" in pooled.columns:
        wide = pooled.pivot_table(
            index=["Source", "Station_ID", "Season_Norm"],
            columns="Metal", values="Concentration", aggfunc="median"
        ).reset_index()
        print(f"After pivot to wide: {wide.shape}")
        wide = wide.dropna(subset=METALS)
    else:
        wide = pooled

    available_metals = [m for m in METALS if m in wide.columns]
    print(f"Available metals after pivot: {available_metals}")

    if len(available_metals) < 6:
        print(f"\n<6 metals available; falling back to direct construction.")
        rainy = pd.read_csv(DATA / "sedimentRO_v2.csv").iloc[:17]
        winter = pd.read_csv(DATA / "sedimentWO_v2.csv").iloc[:17]
        prim_r = pd.DataFrame({m: rainy[c].values for m, c in
                               [("Cd", "CdR"), ("Cr", "CrR"), ("As", "AsR"),
                                ("Pb", "PbR"), ("Cu", "CuR"), ("Ni", "NiR"),
                                ("Fe", "FeR")]})
        prim_r["Source"] = "primary"; prim_r["Season"] = "Rainy"
        prim_w = pd.DataFrame({m: winter[c].values for m, c in
                               [("Cd", "CdW"), ("Cr", "CrW"), ("As", "AsW"),
                                ("Pb", "PbW"), ("Cu", "CuW"), ("Ni", "NiW"),
                                ("Fe", "FeW")]})
        prim_w["Source"] = "primary"; prim_w["Season"] = "Winter"

        sec = pd.read_csv(SECDATA / "harmonized_sediment.csv")
        sec_wide = sec.pivot_table(
            index=["Paper_ID", "Station_ID", "Season_Norm"],
            columns="Metal", values="Concentration", aggfunc="median"
        ).reset_index()
        sec_wide = sec_wide.rename(columns={"Season_Norm": "Season"})
        sec_wide["Source"] = "literature"
        keep_cols = ["Source", "Season"] + METALS
        sec_keep = [c for c in keep_cols if c in sec_wide.columns]
        sec_wide = sec_wide[sec_keep].dropna(subset=METALS)

        wide = pd.concat([prim_r[keep_cols], prim_w[keep_cols], sec_wide],
                         ignore_index=True)
        available_metals = METALS

    X = wide[available_metals].values.astype(float)
    X = np.where(X > 0, X, 1e-6)

    log_X = np.log(X)

    sc = StandardScaler()
    X_std = sc.fit_transform(log_X)
    X_nmf = X_std - X_std.min() + 0.1

    nmf = NMF(n_components=3, init="nndsvd", random_state=42, max_iter=1000)
    W = nmf.fit_transform(X_nmf)
    H = nmf.components_

    print(f"\nNMF on pooled dataset (n={len(X)}, {len(available_metals)} metals)")
    print(f"Reconstruction error: {nmf.reconstruction_err_:.3f}")

    H_norm = H / H.max(axis=0, keepdims=True)
    print("\nComponent loadings (column-normalized to max=1):")
    for k in range(3):
        sorted_metals = sorted(zip(available_metals, H_norm[k, :]),
                                key=lambda t: -t[1])
        top = ", ".join(f"{m}={v:.2f}" for m, v in sorted_metals[:3])
        print(f"  Component {k+1}: top loadings = {top}")

    rows = []
    for k in range(3):
        for i, m in enumerate(available_metals):
            rows.append({"Component": k + 1, "Metal": m,
                         "Loading_norm_max": round(float(H_norm[k, i]), 3),
                         "Loading_raw": round(float(H[k, i]), 3)})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")

    pivot = out_df.pivot(index="Metal", columns="Component",
                         values="Loading_norm_max").round(2)
    pivot.columns = [f"C{i}" for i in pivot.columns]
    print("\nLoading matrix (rows=metals, columns=components):")
    print(pivot.reindex(available_metals))


if __name__ == "__main__":
    main()
