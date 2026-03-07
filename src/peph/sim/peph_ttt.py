from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from peph.sim.peph import simulate_event_time_piecewise_exp
from peph.sim.spatial import sample_leroux_u


def _make_ring_with_second_neighbors(zips: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a small connected adjacency for the ZIP universe.

    Graph:
      - ring edges to immediate neighbors
      - extra edges to second neighbors for smoother correlation

    Returns
    -------
    W : np.ndarray
        Symmetric 0/1 adjacency matrix.
    D : np.ndarray
        Diagonal degree matrix.
    """
    zips = list(map(str, zips))
    G = len(zips)
    if G < 3:
        raise ValueError("Need at least 3 ZIPs to build the graph")

    W = np.zeros((G, G), dtype=float)

    for i in range(G):
        nbrs = {
            (i - 1) % G,
            (i + 1) % G,
            (i - 2) % G,
            (i + 2) % G,
        }
        for j in nbrs:
            if i != j:
                W[i, j] = 1.0
                W[j, i] = 1.0

    np.fill_diagonal(W, 0.0)
    D = np.diag(W.sum(axis=1))
    return W, D


def _sample_crc_covariates(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Simulate SEER-like Medicare CRC baseline covariates.

    Columns
    -------
    id
    age_per10_centered
    cci
    tumor_size_log
    ses
    sex
    stage
    """
    df = pd.DataFrame({"id": np.arange(n, dtype=int)})

    # Roughly older Medicare-like ages, centered and scaled by 10 years.
    age_years = np.clip(rng.normal(loc=72.0, scale=9.0, size=n), 50.0, 97.0)
    df["age_per10_centered"] = (age_years - 70.0) / 10.0

    # CCI: overdispersed nonnegative integer, capped.
    cci = rng.poisson(lam=1.2, size=n)
    df["cci"] = np.clip(cci, 0, 7).astype(int)

    # Log tumor size, roughly centered around your existing example scale.
    df["tumor_size_log"] = rng.normal(loc=3.55, scale=0.45, size=n)

    # SES standard normal.
    df["ses"] = rng.normal(loc=0.0, scale=1.0, size=n)

    # Sex and stage.
    df["sex"] = rng.choice(["F", "M"], size=n, p=[0.52, 0.48])
    df["stage"] = rng.choice(
        ["I", "II", "III", "IV"],
        size=n,
        p=[0.24, 0.35, 0.29, 0.12],
    )

    return df


def simulate_peph_spatial_ttt_dataset(
    *,
    n: int = 10000,
    seed: int = 0,
    # survival outcome model
    surv_breaks: List[float] | None = None,
    surv_nu: np.ndarray | None = None,
    surv_beta: Dict[str, float] | None = None,
    # treatment-time model
    treat_breaks: List[float] | None = None,
    treat_nu: np.ndarray | None = None,
    treat_beta: Dict[str, float] | None = None,
    # spatial frailty
    leroux_tau: float = 4.0,
    leroux_rho: float = 0.90,
    # censoring
    admin_censor: float = 1825.0,
    random_censor_rate: float = 1.0 / 2500.0,
    # treatment observation window
    max_treatment_time: float = 365.0,
    # include debug columns
    include_latent_truth: bool = True,
) -> pd.DataFrame:
    """
    Simulate SEER-like Medicare colorectal cancer data with:
      - wide survival outcome columns
      - ZIP code
      - time-to-treatment (surgery) column

    Interpretation
    --------------
    treatment_time is days from diagnosis to first surgery.

    It is observed only if surgery occurs before the subject's observed follow-up time:
        treatment_time observed iff T_treat <= time

    Therefore treatment_time is NA when surgery is unobserved by the end of follow-up.
    In particular, if death occurs before surgery, treatment_time is NA.

    Returns
    -------
    pd.DataFrame
        Wide dataset with columns such as:
        id, age_per10_centered, cci, tumor_size_log, ses, sex, stage, zip,
        time, event, treatment_time, ...
    """
    rng = np.random.default_rng(seed)

    # ----------------------------
    # Defaults
    # ----------------------------
    if surv_breaks is None:
        surv_breaks = [0.0, 30.0, 90.0, 180.0, 365.0, 730.0, 1825.0]

    if surv_nu is None:
        surv_nu = np.array([0.0019, 0.0015, 0.0012, 0.00095, 0.00075, 0.00060], dtype=float)

    if surv_beta is None:
        surv_beta = {
            "age_per10_centered": 0.12,
            "cci": 0.16,
            "tumor_size_log": 0.28,
            "ses": -0.10,
            "sexM": 0.05,
            "stageII": 0.30,
            "stageIII": 0.65,
            "stageIV": 1.15,
        }

    # Faster surgery = larger treatment hazard.
    if treat_breaks is None:
        treat_breaks = [0.0, 14.0, 30.0, 60.0, 90.0, 180.0, 365.0]

    if treat_nu is None:
        treat_nu = np.array([0.012, 0.016, 0.012, 0.008, 0.0045, 0.0020], dtype=float)

    if treat_beta is None:
        treat_beta = {
            "age_per10_centered": -0.08,
            "cci": -0.10,
            "tumor_size_log": 0.06,
            "ses": 0.12,
            "sexM": -0.03,
            "stageII": 0.10,
            "stageIII": -0.08,
            "stageIV": -0.85,
        }

    if len(surv_breaks) - 1 != len(surv_nu):
        raise ValueError("len(surv_nu) must equal len(surv_breaks) - 1")
    if len(treat_breaks) - 1 != len(treat_nu):
        raise ValueError("len(treat_nu) must equal len(treat_breaks) - 1")
    if admin_censor <= 0:
        raise ValueError("admin_censor must be > 0")
    if max_treatment_time <= 0:
        raise ValueError("max_treatment_time must be > 0")

    # ----------------------------
    # Example Georgia ZIP universe
    # ----------------------------
    zips = [
        "30303", "30305", "30309", "30318", "30030",
        "30033", "30060", "30067", "30080", "30084",
        "30101", "30114", "30120", "30213", "30214",
        "30601", "30605", "30809", "30901", "30909",
    ]
    G = len(zips)

    # ----------------------------
    # Spatial graph + frailty
    # ----------------------------
    W, D = _make_ring_with_second_neighbors(zips)
    u_zip = sample_leroux_u(
        W=W,
        D=D,
        tau=leroux_tau,
        rho=leroux_rho,
        rng=rng,
        component_ids=np.zeros(G, dtype=int),
        weights=np.ones(G, dtype=float),
    )
    zip_to_u = {zips[i]: float(u_zip[i]) for i in range(G)}

    # Optional treatment-access spatial effect.
    # Negatively correlate with death frailty, plus independent noise.
    access_noise = rng.normal(loc=0.0, scale=0.15, size=G)
    v_zip = -0.35 * u_zip + access_noise
    zip_to_v = {zips[i]: float(v_zip[i]) for i in range(G)}

    # ----------------------------
    # Baseline covariates
    # ----------------------------
    df = _sample_crc_covariates(n=n, rng=rng)
    df["zip"] = rng.choice(zips, size=n, replace=True)

    # ----------------------------
    # Survival linear predictor
    # ----------------------------
    eta_surv = np.zeros(n, dtype=float)
    eta_surv += surv_beta.get("age_per10_centered", 0.0) * df["age_per10_centered"].to_numpy(dtype=float)
    eta_surv += surv_beta.get("cci", 0.0) * df["cci"].to_numpy(dtype=float)
    eta_surv += surv_beta.get("tumor_size_log", 0.0) * df["tumor_size_log"].to_numpy(dtype=float)
    eta_surv += surv_beta.get("ses", 0.0) * df["ses"].to_numpy(dtype=float)
    eta_surv += surv_beta.get("sexM", 0.0) * (df["sex"].to_numpy() == "M").astype(float)
    eta_surv += surv_beta.get("stageII", 0.0) * (df["stage"].to_numpy() == "II").astype(float)
    eta_surv += surv_beta.get("stageIII", 0.0) * (df["stage"].to_numpy() == "III").astype(float)
    eta_surv += surv_beta.get("stageIV", 0.0) * (df["stage"].to_numpy() == "IV").astype(float)
    eta_surv += np.array([zip_to_u[z] for z in df["zip"].astype(str)], dtype=float)

    # ----------------------------
    # Treatment-time linear predictor
    # ----------------------------
    eta_treat = np.zeros(n, dtype=float)
    eta_treat += treat_beta.get("age_per10_centered", 0.0) * df["age_per10_centered"].to_numpy(dtype=float)
    eta_treat += treat_beta.get("cci", 0.0) * df["cci"].to_numpy(dtype=float)
    eta_treat += treat_beta.get("tumor_size_log", 0.0) * df["tumor_size_log"].to_numpy(dtype=float)
    eta_treat += treat_beta.get("ses", 0.0) * df["ses"].to_numpy(dtype=float)
    eta_treat += treat_beta.get("sexM", 0.0) * (df["sex"].to_numpy() == "M").astype(float)
    eta_treat += treat_beta.get("stageII", 0.0) * (df["stage"].to_numpy() == "II").astype(float)
    eta_treat += treat_beta.get("stageIII", 0.0) * (df["stage"].to_numpy() == "III").astype(float)
    eta_treat += treat_beta.get("stageIV", 0.0) * (df["stage"].to_numpy() == "IV").astype(float)
    eta_treat += np.array([zip_to_v[z] for z in df["zip"].astype(str)], dtype=float)

    # ----------------------------
    # Latent death times
    # ----------------------------
    T_death = np.empty(n, dtype=float)
    for i in range(n):
        T_death[i] = simulate_event_time_piecewise_exp(
            breaks=surv_breaks,
            nu=surv_nu,
            eta=float(eta_surv[i]),
            rng=rng,
        )

    # ----------------------------
    # Censoring
    # ----------------------------
    C_admin = np.full(n, float(admin_censor), dtype=float)
    if random_censor_rate > 0:
        C_rand = rng.exponential(scale=1.0 / random_censor_rate, size=n)
    else:
        C_rand = np.full(n, np.inf, dtype=float)

    C = np.minimum(C_admin, C_rand)

    # Observed survival outcome
    time = np.minimum(T_death, C)
    event = (T_death <= C).astype(int)

    df["time"] = time
    df["event"] = event

    # ----------------------------
    # Latent surgery times
    # ----------------------------
    T_treat = np.empty(n, dtype=float)
    for i in range(n):
        T_treat[i] = simulate_event_time_piecewise_exp(
            breaks=treat_breaks,
            nu=treat_nu,
            eta=float(eta_treat[i]),
            rng=rng,
        )

    # Administrative limit for the surgery process itself
    T_treat = np.minimum(T_treat, float(max_treatment_time) + 1e6)

    # Surgery is observed only if it occurs before the observed follow-up end.
    treat_observed = T_treat <= time

    treatment_time = np.where(treat_observed, T_treat, np.nan)
    df["treatment_time"] = treatment_time

    # Optional debug columns for why treatment is missing
    # 0 = observed
    # 1 = death before treatment
    # 2 = censoring before treatment
    ttt_missing_code = np.zeros(n, dtype=int)
    miss = ~treat_observed
    ttt_missing_code[miss & (event == 1)] = 1
    ttt_missing_code[miss & (event == 0)] = 2

    code_to_reason = {
        0: "observed",
        1: "death_before_treatment",
        2: "censor_before_treatment",
    }
    df["ttt_missing_code"] = ttt_missing_code
    df["ttt_missing_reason"] = pd.Categorical(
        [code_to_reason[int(x)] for x in ttt_missing_code],
        categories=["observed", "death_before_treatment", "censor_before_treatment"],
        ordered=False,
    )

    if include_latent_truth:
        df["_T_true"] = T_death
        df["_C_true"] = C
        df["_eta_true"] = eta_surv
        df["_admin_cens"] = (C_admin <= C_rand).astype(int)
        # Make baseline hazard explicit by interval
        for k, val in enumerate(surv_nu):
            df[f"_nu_true_k{k}"] = float(val)

        df["_T_treat_true"] = T_treat
        df["_eta_treat_true"] = eta_treat
        df["_u_zip_true"] = np.array([zip_to_u[z] for z in df["zip"].astype(str)], dtype=float)
        df["_v_zip_true"] = np.array([zip_to_v[z] for z in df["zip"].astype(str)], dtype=float)

    return df


if __name__ == "__main__":
    df = simulate_peph_spatial_ttt_dataset(n=10000, seed=123)

    print(df.head())
    print()
    print(df[["time", "event", "treatment_time"]].describe())
    print()
    print(df["ttt_missing_reason"].value_counts(dropna=False))