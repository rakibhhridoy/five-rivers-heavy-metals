"""LIME local explanations for Fig 9 (Supplementary §S3.8).

Closes Gap 3 of docs/AUDIT_COVERAGE.md. The full LIME analyzer lives at the project
root in scripts/phase1_feature_importance_extraction.py; this is a self-contained
version that trains the CNN-GNN-MLP model on the 117 IDW-augmented rainy-season
stations and produces local explanations for three metals (As, Pb, Cd) at three
stations representing high / medium / low contamination — matching the Fig 9 caption.

Requires: pip install lime  (already in requirements.txt)

Inputs
------
data/sedimentRO_v2.csv  — rainy-season sediment + 68 satellite features

Output
------
results/s_lime_explanations.csv  — per (Metal, Station, Feature): LIME weight
figures/Fig9_lime_explanations.png — regenerated panel
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import lime
import lime.lime_tabular  # 0.2.x exposes the explainer here

from extra_dl_architectures import build_cnn_gnn_mlp

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT_CSV = ROOT / "results" / "s_lime_explanations.csv"
OUT_FIG = ROOT / "figures" / "Fig9_lime_explanations.png"

SEED = 42
METALS = ["As", "Pb", "Cd"]            # Fig 9 panel metals
METAL_COL = {"As": "AsR", "Pb": "PbR", "Cd": "CdR"}

PROX = ["hydro_dist_brick", "num_brick_field", "hydro_dist_ind", "num_industry"]
AE = [f"AE_{i:02d}" for i in range(64)]
FEATURES = PROX + AE
# Friendlier names for the proximity features in the plot
PRETTY = {"hydro_dist_brick": "dist_brick", "num_brick_field": "num_brick",
          "hydro_dist_ind": "dist_industry", "num_industry": "num_industry"}


def pick_stations(y: np.ndarray) -> dict:
    """Indices of high / medium / low contamination stations by target value."""
    order = np.argsort(y)
    return {"low": int(order[0]),
            "medium": int(order[len(order) // 2]),
            "high": int(order[-1])}


def main() -> None:
    np.random.seed(SEED)
    import tensorflow as tf
    tf.random.set_seed(SEED)

    df = pd.read_csv(DATA / "sedimentRO_v2.csv")  # 117 rows (17 real + 100 IDW)
    X = df[FEATURES].values.astype(float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    rows = []
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))

    for r, metal in enumerate(METALS):
        y = df[METAL_COL[metal]].values.astype(float)
        model = build_cnn_gnn_mlp(len(FEATURES))
        model.fit(Xs, y, epochs=150, batch_size=8, verbose=0)

        explainer = lime.lime_tabular.LimeTabularExplainer(
            Xs, feature_names=FEATURES, mode="regression",
            random_state=SEED, verbose=False,
        )

        def predict_fn(z):
            return model.predict(z, verbose=0).ravel()

        stations = pick_stations(y)
        for c, (level, idx) in enumerate(stations.items()):
            exp = explainer.explain_instance(
                Xs[idx], predict_fn, num_features=10, num_samples=1000,
            )
            pairs = exp.as_list()  # [(feature_desc, weight), ...]
            # Map back to clean feature names
            cleaned = []
            for desc, w in pairs:
                feat = next((f for f in FEATURES if f in desc), desc)
                cleaned.append((PRETTY.get(feat, feat), w))
                rows.append(dict(Metal=metal, Station=df["Stations"].iloc[idx],
                                 Level=level, Feature=PRETTY.get(feat, feat),
                                 LIME_weight=w))
            cleaned = sorted(cleaned, key=lambda kv: abs(kv[1]))[-10:]
            names = [c[0] for c in cleaned]
            weights = [c[1] for c in cleaned]
            colors = ["#2ca02c" if w > 0 else "#d62728" for w in weights]

            ax = axes[r, c]
            ax.barh(range(len(names)), weights, color=colors, edgecolor="black",
                    linewidth=0.5)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(f"{metal} — {level} (station {df['Stations'].iloc[idx]})",
                         fontsize=10)
            ax.grid(axis="x", linestyle=":", alpha=0.4)

    fig.suptitle("LIME local explanations (rainy season, CNN-GNN-MLP). "
                 "Green increases the prediction; red decreases it.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_FIG.name} and {OUT_CSV.name} ({len(rows)} weight rows)")


if __name__ == "__main__":
    main()
