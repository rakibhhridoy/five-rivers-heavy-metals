"""Original Monte Carlo Risk Index on primary-only data + source-paper background.

Reproduces the §3.7 headline numbers:
  mean RI = 254.78 (lognormal); Pb contributes 60.68%, Cu 17.17%, Cr 9.04%.

phase6_1_mc_pooled.py runs the sensitivity scenarios (primary-only, lit-only, pooled)
on the *harmonised* Turekian–Wedepohl background with standard Håkanson Tr factors;
this script reproduces the original headline scenario using the **source-paper
background and Tr values** recovered from the originating notebook
(Python/sample.ipynb cells 6 and 220):

  Bm  (mg/kg) — used in CF = C_sample / Bm
    Cr=90, Ni=68, Cu=45, As=13, Cd=0.3, Pb=20, Fe=30890
  Tr — toxic-response factors used in Er = Tr × CF
    Cr=26, Ni=16, Cu=16, As=6, Cd=0.6, Pb=31

These Tr values diverge from standard Håkanson 1980 (Cd=30, Pb=5). The
§3.7 robustness paragraph (phase6_1_mc_pooled.py, on Turekian–Wedepohl shale
+ standard Håkanson Tr) is the more defensible analysis; this script exists to
make the headline 60.68% / 254.78 / 17.17% / 9.04% numbers reproducible from
data + code rather than appearing only in the manuscript prose.

Inputs
------
data/sedimentRO_v2.csv, data/sedimentWO_v2.csv

Output
------
results/s_mc_primary_headline.csv  — per-metal contribution % and Er summary
results/s_mc_primary_headline_summary.csv  — RI distribution stats
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT_DETAIL = ROOT / "results" / "s_mc_primary_headline.csv"
OUT_SUMMARY = ROOT / "results" / "s_mc_primary_headline_summary.csv"

# Source-paper Tr values (Python/sample.ipynb cell 220). NB: these differ from
# Håkanson 1980 (Cd=30, Pb=5) and are responsible for Pb-dominance in §3.7.
TR = {"Cr": 26, "Ni": 16, "Cu": 16, "As": 6, "Cd": 0.6, "Pb": 31}

# Source-paper backgrounds (Python/sample.ipynb cell 6). Fe = 30890 mg/kg.
BG = {"Cr": 90.0, "Ni": 68.0, "Cu": 45.0, "As": 13.0,
      "Cd": 0.3,  "Pb": 20.0, "Fe": 30890.0}

METALS = ["Cd", "Cr", "As", "Pb", "Cu", "Ni"]  # Fe excluded from RI
METAL_COLS_R = {m: f"{m}R" for m in METALS}
METAL_COLS_W = {m: f"{m}W" for m in METALS}

N_ITER = 10_000
SEED = 42


def simulate(values_by_metal: dict, distribution: str, rng: np.random.Generator) -> np.ndarray:
    """Return Er samples (N_ITER × n_metals) under the chosen distribution."""
    samples = {}
    for m, v in values_by_metal.items():
        v = np.asarray(v)
        if distribution == "normal":
            mu, sd = float(np.mean(v)), float(np.std(v, ddof=1))
            draws = rng.normal(mu, sd, size=N_ITER)
        elif distribution == "lognormal":
            lv = np.log(np.clip(v, 1e-9, None))
            mu, sd = float(np.mean(lv)), float(np.std(lv, ddof=1))
            draws = rng.lognormal(mu, sd, size=N_ITER)
        else:
            raise ValueError(distribution)
        samples[m] = np.clip(draws, 0, None)  # concentrations cannot be negative
    # Er_m = Tr_m × CF_m = Tr_m × C_m / Bg_m
    er = np.column_stack([TR[m] * samples[m] / BG[m] for m in METALS])
    return er  # shape (N_ITER, n_metals)


def run_scenario(seasonal_values: dict, distribution: str) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED)
    er = simulate(seasonal_values, distribution, rng)
    ri = er.sum(axis=1)
    contrib_pct = (er.mean(axis=0) / ri.mean()) * 100  # % of mean RI
    detail = pd.DataFrame({
        "Metal": METALS,
        "Tr": [TR[m] for m in METALS],
        "Background": [BG[m] for m in METALS],
        "Er_mean": er.mean(axis=0),
        "Er_median": np.median(er, axis=0),
        "Er_p95": np.percentile(er, 95, axis=0),
        "Contribution_pct": contrib_pct,
    })
    summary = dict(
        RI_mean=float(ri.mean()), RI_median=float(np.median(ri)),
        RI_p5=float(np.percentile(ri, 5)), RI_p95=float(np.percentile(ri, 95)),
        RI_max=float(ri.max()),
    )
    return detail, summary


def main() -> None:
    sed_r = pd.read_csv(DATA / "sedimentRO_v2.csv").iloc[:17]
    sed_w = pd.read_csv(DATA / "sedimentWO_v2.csv").iloc[:17]
    # Stack rainy + winter for primary-only n=34
    pooled = {m: np.concatenate([sed_r[METAL_COLS_R[m]].values,
                                 sed_w[METAL_COLS_W[m]].values])
              for m in METALS}

    detail_dfs, summary_rows = [], []
    for dist in ("normal", "lognormal"):
        det, summ = run_scenario(pooled, dist)
        det["Distribution"] = dist
        detail_dfs.append(det)
        summary_rows.append({"Distribution": dist, **summ})

    detail = pd.concat(detail_dfs, ignore_index=True)
    detail.to_csv(OUT_DETAIL, index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"Wrote {OUT_DETAIL.name} and {OUT_SUMMARY.name}")
    print("\n§3.7 headline reproduction (primary-only n=34, USEPA regional shale bg):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nPer-metal contribution % (lognormal — manuscript headline distribution):")
    ln = detail[detail["Distribution"] == "lognormal"].sort_values(
        "Contribution_pct", ascending=False)
    print(ln[["Metal", "Er_mean", "Contribution_pct"]].to_string(
        index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nManuscript expects: Pb 60.68%, Cu 17.17%, Cr 9.04%.")


if __name__ == "__main__":
    main()
