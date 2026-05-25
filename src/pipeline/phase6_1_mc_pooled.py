"""
Phase 6.1 — Monte Carlo with literature-pooled priors (Strategy S6).

Reruns the ecological Risk Index (RI) Monte Carlo simulation under three prior
scenarios — primary-only (n=34), literature-only (n=37), pooled (n=71) — to test
whether pooling the secondary data changes the manuscript headline:
    "Pb dominates at 60.68% of total RI."

RI formula (Hakanson 1980): RI = Σ_i Er_i = Σ_i Tr_i × CF_i
Toxic-response factors (Tr_i): Cd=30, As=10, Cu=Pb=Ni=5, Cr=2.

Inputs
------
analysis_results/enrichment/s4_harmonized_indices.csv
    71 station-seasons with CF per metal on Turekian & Wedepohl 1961 background.

Outputs
-------
analysis_results/enrichment/s6_mc_pooled.csv
    Per-metal contribution %, mean/median/p95 Er, for each scenario × distribution.
analysis_results/enrichment/s6_mc_summary.csv
    RI summary stats (mean, median, p5, p95, max) for each scenario × distribution.
analysis_results/enrichment/s6_mc_pb_share.png
    Pb-share-of-RI distribution, primary-only vs pooled (lognormal).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
HARMONIZED = ROOT / "results/s4_harmonized_indices.csv"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260424)
N_ITER = 10_000

# Hakanson (1980) toxic-response factors. Zn/Hg absent from our metal panel.
TR = {"Cd": 30, "As": 10, "Cu": 5, "Pb": 5, "Ni": 5, "Cr": 2}
METALS = list(TR.keys())


def fit_lognormal(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x) & (x > 0)]
    logs = np.log(x)
    return float(logs.mean()), float(logs.std(ddof=1))


def fit_normal(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    return float(x.mean()), float(x.std(ddof=1))


def simulate(cf_by_metal: dict[str, np.ndarray], dist: str) -> dict:
    """Run N_ITER MC draws; return per-iteration Er matrix + metal contributions."""
    er = np.zeros((N_ITER, len(METALS)))
    for j, m in enumerate(METALS):
        cf = cf_by_metal[m]
        if len(cf) < 2:
            er[:, j] = np.nan
            continue
        if dist == "lognormal":
            mu, sigma = fit_lognormal(cf)
            draws = RNG.lognormal(mean=mu, sigma=sigma, size=N_ITER)
        else:  # normal, truncated at 0
            mu, sigma = fit_normal(cf)
            draws = np.clip(RNG.normal(loc=mu, scale=sigma, size=N_ITER), 0, None)
        er[:, j] = TR[m] * draws
    ri = er.sum(axis=1)
    contrib = er / np.where(ri[:, None] > 0, ri[:, None], np.nan)
    return {
        "ri": ri,
        "er": er,
        "contrib_mean_pct": np.nanmean(contrib, axis=0) * 100,
        "er_mean": np.nanmean(er, axis=0),
    }


def station_ri_mc(ri_values: np.ndarray, dist: str) -> dict:
    """Fit distribution to per-station RI values; draw N_ITER samples."""
    ri_values = ri_values[np.isfinite(ri_values) & (ri_values > 0)]
    if dist == "lognormal":
        mu, sigma = fit_lognormal(ri_values)
        draws = RNG.lognormal(mean=mu, sigma=sigma, size=N_ITER)
    else:
        mu, sigma = fit_normal(ri_values)
        draws = np.clip(RNG.normal(loc=mu, scale=sigma, size=N_ITER), 0, None)
    return {
        "RI_mean": float(np.mean(draws)),
        "RI_median": float(np.median(draws)),
        "RI_p5": float(np.percentile(draws, 5)),
        "RI_p95": float(np.percentile(draws, 95)),
        "RI_max": float(np.max(draws)),
    }


def main() -> None:
    df = pd.read_csv(HARMONIZED)
    print(f"Loaded harmonized indices: {df.shape}, sources={df['Source'].value_counts().to_dict()}")

    # Per-station RI under harmonized T&W 1961 background
    df = df.copy()
    df["RI_harmonized"] = sum(TR[m] * df[f"CF_{m}"] for m in METALS)

    scenarios = {
        "primary_only": df[df["Source"] == "Primary"],
        "literature_only": df[df["Source"] == "Literature"],
        "pooled": df,
    }

    contrib_rows, summary_rows, station_rows = [], [], []

    for scen_name, scen_df in scenarios.items():
        cf_by_metal = {m: scen_df[m].to_numpy() for m in METALS if m in scen_df.columns}
        n = len(scen_df)
        for dist in ("lognormal", "normal"):
            # Per-metal MC → ΣTr*CF
            res = simulate(cf_by_metal, dist)
            ri = res["ri"]
            summary_rows.append(
                {
                    "scenario": scen_name,
                    "distribution": dist,
                    "method": "per_metal_MC",
                    "n_stations": n,
                    "RI_mean": float(np.nanmean(ri)),
                    "RI_median": float(np.nanmedian(ri)),
                    "RI_p5": float(np.nanpercentile(ri, 5)),
                    "RI_p95": float(np.nanpercentile(ri, 95)),
                    "RI_max": float(np.nanmax(ri)),
                }
            )
            for j, m in enumerate(METALS):
                contrib_rows.append(
                    {
                        "scenario": scen_name,
                        "distribution": dist,
                        "metal": m,
                        "Tr": TR[m],
                        "Er_mean": float(res["er_mean"][j]),
                        "contrib_pct_of_RI": float(res["contrib_mean_pct"][j]),
                    }
                )
            # Per-station RI MC (matches manuscript methodology)
            stats = station_ri_mc(scen_df["RI_harmonized"].to_numpy(), dist)
            station_rows.append(
                {
                    "scenario": scen_name,
                    "distribution": dist,
                    "method": "per_station_RI_MC",
                    "n_stations": n,
                    **stats,
                }
            )

    contrib_df = pd.DataFrame(contrib_rows)
    summary_df = pd.concat([pd.DataFrame(summary_rows), pd.DataFrame(station_rows)], ignore_index=True)

    contrib_df.to_csv(OUT_DIR / "s6_mc_pooled.csv", index=False)
    summary_df.to_csv(OUT_DIR / "s6_mc_summary.csv", index=False)

    print("\n=== RI summary (per_station_RI_MC is directly comparable to manuscript) ===")
    print(summary_df.to_string(index=False))
    print("\n=== Per-metal contribution % (lognormal) ===")
    piv = (
        contrib_df[contrib_df["distribution"] == "lognormal"]
        .pivot(index="metal", columns="scenario", values="contrib_pct_of_RI")
        .reindex(METALS)
    )
    print(piv.to_string())

    # Pb-share distribution: primary-only vs pooled (lognormal)
    fig, ax = plt.subplots(figsize=(8, 5))
    for scen_name, color in [("primary_only", "C0"), ("pooled", "C3")]:
        scen_df = scenarios[scen_name]
        cf_by_metal = {m: scen_df[m].to_numpy() for m in METALS}
        res = simulate(cf_by_metal, "lognormal")
        pb_idx = METALS.index("Pb")
        pb_share = res["er"][:, pb_idx] / np.where(res["ri"] > 0, res["ri"], np.nan)
        ax.hist(
            pb_share * 100,
            bins=60,
            alpha=0.55,
            label=f"{scen_name} (n={len(scen_df)})",
            color=color,
        )
    ax.axvline(60.68, color="k", linestyle="--", label="Manuscript headline (60.68%)")
    ax.set_xlabel("Pb share of total RI (%)")
    ax.set_ylabel("MC iterations")
    ax.set_title("Pb contribution to ecological RI — primary-only vs pooled priors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "s6_mc_pb_share.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote:\n  {OUT_DIR/'s6_mc_pooled.csv'}\n  {OUT_DIR/'s6_mc_summary.csv'}\n  {OUT_DIR/'s6_mc_pb_share.png'}")


if __name__ == "__main__":
    main()
