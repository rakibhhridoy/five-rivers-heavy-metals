"""HQ and ILCR computation for sediment-bound and dissolved metals (§3.7, S8).

Implements USEPA RAGS (1989) and Exposure Factors Handbook (2011) formulas for:
  • Hazard Quotient (HQ = ADD / RfD) — non-carcinogenic
  • Incremental Lifetime Cancer Risk (ILCR = ADD × CSF) — carcinogenic
  • Three sediment pathways: incidental ingestion, dermal contact, dust inhalation
  • Two receptors: adult, child

Reports both EPA-default and Bangladesh-specific (Habib 2020, Proshad 2018,
Islam 2019, Rahman 2022) sensitivity (S8.3).

Inputs
------
data/sedimentRO_v2.csv, data/sedimentWO_v2.csv  — sediment metals (mg/kg)
data/waterR_original_17.csv, data/waterW_original_17.csv  — dissolved metals (mg/L)

Output
------
results/s_hq_ilcr_sediment.csv  — per (Station, Season, Metal, Receptor, Pathway): ADD, HQ, ILCR
results/s_hq_ilcr_water.csv      — per (Station, Season, Metal, Receptor): ADD, HQ, ILCR
results/s_hq_ilcr_sensitivity.csv — EPA vs Bangladesh-specific summary deltas
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT_SED = ROOT / "results" / "s_hq_ilcr_sediment.csv"
OUT_WAT = ROOT / "results" / "s_hq_ilcr_water.csv"
OUT_SENS = ROOT / "results" / "s_hq_ilcr_sensitivity.csv"

METALS = ["Cr", "Ni", "Cu", "As", "Cd", "Pb", "Fe"]

# Reference doses (RfD, mg/kg/day) — USEPA IRIS / RAGS
RFD_ING = {"Cr": 3e-3, "Ni": 2e-2, "Cu": 4e-2, "As": 3e-4,
           "Cd": 1e-3, "Pb": 3.6e-3, "Fe": 7e-1}
RFD_DER = {"Cr": 6e-5, "Ni": 5.4e-3, "Cu": 1.2e-2, "As": 1.23e-4,
           "Cd": 1e-5, "Pb": 5.25e-4, "Fe": 1.4e-1}
RFD_INH = {"Cr": 2.86e-5, "Ni": 2.06e-2, "Cu": 4e-2, "As": 3.01e-4,
           "Cd": 1e-3, "Pb": 3.52e-3, "Fe": 7e-1}

# Cancer slope factors (CSF, per mg/kg/day) — USEPA IRIS
CSF_ING = {"Cr": 0.5, "Ni": 0.84, "As": 1.5, "Cd": 6.3, "Pb": 8.5e-3}
CSF_DER = {"Cr": 20.0, "Ni": 42.0, "As": 3.66, "Cd": 6.3, "Pb": 8.5e-3}
CSF_INH = {"Cr": 42.0, "Ni": 0.84, "As": 15.1, "Cd": 6.3, "Pb": 4.2e-2}

# Exposure parameter sets
EPA = dict(
    IngR_a=100, IngR_c=200,
    ExpF=350, ExpD_a=24, ExpD_c=6,
    BW_a=70, BW_c=15,
    SA_a=5700, SA_c=2800,
    AF_a=0.07, AF_c=0.20,
    InhR_a=20, InhR_c=10,
    PEF=1.36e9,  # particulate emission factor m³/kg
    ABS=0.001,   # dermal absorption fraction
    AT_nc=365 * 24, AT_ca=365 * 70,  # averaging time non-carcin / carcin (days)
)
BANGLADESH = dict(
    IngR_a=80, IngR_c=180,
    ExpF=365, ExpD_a=30, ExpD_c=6,
    BW_a=60, BW_c=13,
    SA_a=5500, SA_c=2700,
    AF_a=0.20, AF_c=0.30,
    InhR_a=20, InhR_c=10,
    PEF=1.36e9, ABS=0.001,
    AT_nc=365 * 30, AT_ca=365 * 70,
)


def add_ingestion(C, p, receptor):
    """C in mg/kg, returns ADD in mg/kg/day."""
    ingr = p["IngR_a"] if receptor == "adult" else p["IngR_c"]
    expd = p["ExpD_a"] if receptor == "adult" else p["ExpD_c"]
    bw = p["BW_a"] if receptor == "adult" else p["BW_c"]
    return C * ingr * 1e-6 * p["ExpF"] * expd / (bw * p["AT_nc"])


def add_dermal(C, p, receptor):
    sa = p["SA_a"] if receptor == "adult" else p["SA_c"]
    af = p["AF_a"] if receptor == "adult" else p["AF_c"]
    expd = p["ExpD_a"] if receptor == "adult" else p["ExpD_c"]
    bw = p["BW_a"] if receptor == "adult" else p["BW_c"]
    return C * sa * af * p["ABS"] * 1e-6 * p["ExpF"] * expd / (bw * p["AT_nc"])


def add_inhalation(C, p, receptor):
    inhr = p["InhR_a"] if receptor == "adult" else p["InhR_c"]
    expd = p["ExpD_a"] if receptor == "adult" else p["ExpD_c"]
    bw = p["BW_a"] if receptor == "adult" else p["BW_c"]
    return C * inhr * p["ExpF"] * expd / (p["PEF"] * bw * p["AT_nc"])


def sediment_hq_ilcr(C, p, receptor, metal):
    add_i = add_ingestion(C, p, receptor)
    add_d = add_dermal(C, p, receptor)
    add_h = add_inhalation(C, p, receptor)
    hq_i = add_i / RFD_ING.get(metal, np.nan) if metal in RFD_ING else np.nan
    hq_d = add_d / RFD_DER.get(metal, np.nan) if metal in RFD_DER else np.nan
    hq_h = add_h / RFD_INH.get(metal, np.nan) if metal in RFD_INH else np.nan
    ilcr_i = add_i * CSF_ING.get(metal, 0.0) if metal in CSF_ING else 0.0
    ilcr_d = add_d * CSF_DER.get(metal, 0.0) if metal in CSF_DER else 0.0
    ilcr_h = add_h * CSF_INH.get(metal, 0.0) if metal in CSF_INH else 0.0
    return dict(ADD_ing=add_i, ADD_derm=add_d, ADD_inh=add_h,
                HQ_ing=hq_i, HQ_derm=hq_d, HQ_inh=hq_h,
                ILCR_ing=ilcr_i, ILCR_derm=ilcr_d, ILCR_inh=ilcr_h)


def water_hq_ilcr(C_mgL, p, receptor, metal):
    """Daily water ingestion exposure. C in mg/L; assumes 2 L/day adult, 1 L/day child."""
    intake = 2.0 if receptor == "adult" else 1.0
    expd = p["ExpD_a"] if receptor == "adult" else p["ExpD_c"]
    bw = p["BW_a"] if receptor == "adult" else p["BW_c"]
    add = C_mgL * intake * p["ExpF"] * expd / (bw * p["AT_nc"])
    hq = add / RFD_ING.get(metal, np.nan) if metal in RFD_ING else np.nan
    ilcr = add * CSF_ING.get(metal, 0.0) if metal in CSF_ING else 0.0
    return dict(ADD=add, HQ=hq, ILCR=ilcr)


def process_sediment(df, season, param_set, param_name):
    rows = []
    for i in range(len(df)):
        st, river = df["Stations"].iloc[i], df["River"].iloc[i]
        for m in METALS:
            col = f"{m}{'R' if season == 'Rainy' else 'W'}"
            C = float(df[col].iloc[i])
            for receptor in ("adult", "child"):
                r = sediment_hq_ilcr(C, param_set, receptor, m)
                rows.append(dict(Station=st, River=river, Season=season, Metal=m,
                                 Receptor=receptor, ParamSet=param_name,
                                 Conc_mg_per_kg=C, **r,
                                 HQ_total=r["HQ_ing"] + r["HQ_derm"] + r["HQ_inh"],
                                 ILCR_total=r["ILCR_ing"] + r["ILCR_derm"] + r["ILCR_inh"]))
    return rows


def process_water(df, season, param_set, param_name):
    rows = []
    for i in range(len(df)):
        st, river = df["Stations"].iloc[i], df["River"].iloc[i]
        for m in METALS:
            col = f"{m}{'R' if season == 'Rainy' else 'W'}"
            C = float(df[col].iloc[i])
            for receptor in ("adult", "child"):
                r = water_hq_ilcr(C, param_set, receptor, m)
                rows.append(dict(Station=st, River=river, Season=season, Metal=m,
                                 Receptor=receptor, ParamSet=param_name,
                                 Conc_mg_per_L=C, **r))
    return rows


def main() -> None:
    sed_r = pd.read_csv(DATA / "sedimentRO_v2.csv").iloc[:17]
    sed_w = pd.read_csv(DATA / "sedimentWO_v2.csv").iloc[:17]
    wat_r = pd.read_csv(DATA / "waterR_original_17.csv")
    wat_w = pd.read_csv(DATA / "waterW_original_17.csv")

    sed_rows, wat_rows = [], []
    for params, name in [(EPA, "EPA"), (BANGLADESH, "Bangladesh")]:
        for df, season in [(sed_r, "Rainy"), (sed_w, "Winter")]:
            sed_rows.extend(process_sediment(df, season, params, name))
        for df, season in [(wat_r, "Rainy"), (wat_w, "Winter")]:
            wat_rows.extend(process_water(df, season, params, name))

    sed_df = pd.DataFrame(sed_rows)
    wat_df = pd.DataFrame(wat_rows)
    sed_df.to_csv(OUT_SED, index=False)
    wat_df.to_csv(OUT_WAT, index=False)

    # Sensitivity: Bangladesh vs EPA, per metal and pathway (adult-receptor summary)
    sens_rows = []
    for m in METALS:
        for pw in ("HQ_ing", "HQ_derm", "HQ_inh", "ILCR_ing", "ILCR_derm", "ILCR_inh",
                   "HQ_total", "ILCR_total"):
            epa = sed_df[(sed_df["Metal"] == m) & (sed_df["Receptor"] == "adult")
                         & (sed_df["ParamSet"] == "EPA")][pw].median()
            bd = sed_df[(sed_df["Metal"] == m) & (sed_df["Receptor"] == "adult")
                        & (sed_df["ParamSet"] == "Bangladesh")][pw].median()
            sens_rows.append(dict(Metal=m, Metric=pw, EPA_median=epa, BD_median=bd,
                                  Pct_change=(bd / epa - 1) * 100 if epa else np.nan))
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(OUT_SENS, index=False)

    print(f"Wrote {OUT_SED.name}, {OUT_WAT.name}, {OUT_SENS.name}")
    print("\n§3.7 reproduction — count of stations exceeding HQ>1 or ILCR>1e-4 "
          "(adult, EPA defaults):")
    epa_adult = sed_df[(sed_df["ParamSet"] == "EPA") & (sed_df["Receptor"] == "adult")]
    by_metal = (epa_adult.groupby("Metal")
                .agg(HQ_gt1=("HQ_total", lambda s: int((s > 1).sum())),
                     ILCR_gt_1e4=("ILCR_total", lambda s: int((s > 1e-4).sum())),
                     n_stations=("HQ_total", "count"))
                .reset_index())
    print(by_metal.to_string(index=False))
    print("\nSection §3.7 sensitivity — median % change Bangladesh vs EPA "
          "(adult, total HQ / total ILCR):")
    sel = sens_df[sens_df["Metric"].isin(["HQ_total", "ILCR_total"])]
    print(sel[["Metal", "Metric", "Pct_change"]].to_string(
        index=False, float_format=lambda x: f"{x:+.1f}%"))


if __name__ == "__main__":
    main()
