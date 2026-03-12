from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from peph.sim.spatial import sample_leroux_u
from peph.sim.ttt_effect import (
    _linear_predictor_from_wide,
    _simulate_baseline_covariates,
    simulate_event_time_piecewise_exp_with_switch,
)
from peph.spatial.graph import build_graph_from_edge_list


def _load_zip_universe(zips_path: str) -> List[str]:
    zdf = pd.read_csv(zips_path)
    if "zip" in zdf.columns:
        return zdf["zip"].astype(str).tolist()
    if zdf.shape[1] == 1:
        return zdf.iloc[:, 0].astype(str).tolist()
    raise ValueError(
        f"ZIP universe file must contain a 'zip' column or be single-column: {zips_path}"
    )


def _simulate_treatment_times_lognormal_aft(
    df: pd.DataFrame,
    *,
    beta_treatment: Dict[str, float],
    sigma_treatment: float,
    treatment_intercept: float,
    rng: np.random.Generator,
    spatial_offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Simulate latent treatment times from a log-normal AFT model:

        log(T_treat) =
            treatment_intercept
          + x beta_treatment
          + spatial_offset
          + sigma_treatment * eps,
        eps ~ N(0, 1)

    Positive coefficients imply longer treatment times.
    """
    if sigma_treatment <= 0.0:
        raise ValueError("sigma_treatment must be > 0")

    mu = float(treatment_intercept) + _linear_predictor_from_wide(df, beta=beta_treatment)

    if spatial_offset is not None:
        spatial_offset = np.asarray(spatial_offset, dtype=float)
        if spatial_offset.shape != (len(df),):
            raise ValueError("spatial_offset must have shape (n_subjects,)")
        mu = mu + spatial_offset

    log_t = mu + float(sigma_treatment) * rng.normal(size=len(df))
    t = np.exp(log_t)
    return np.maximum(t, 1e-8)


def simulate_joint_ttt_survival_dataset(
    *,
    n: int,
    breaks: List[float],
    nu: np.ndarray,
    beta_survival: Dict[str, float],
    gamma_treated: float,
    beta_treatment: Dict[str, float],
    sigma_treatment: float,
    treatment_intercept: float,
    zips_path: str,
    edges_path: str,
    edges_u_col: str = "zip_u",
    edges_v_col: str = "zip_v",
    tau_true: float = 2.0,
    rho_true: float = 0.85,
    treatment_spatial_mode: str = "none",
    treatment_tau_true: float = 2.0,
    treatment_rho_true: float = 0.85,
    admin_censor: Optional[float] = None,
    random_censor_rate: float = 0.0006,
    seed: int = 0,
    return_latent_truth: bool = True,
) -> pd.DataFrame:
    """
    Simulate a joint wide-form dataset for:

      1) treatment-time modeling
         - nonspatial or spatial log-normal AFT treatment times
      2) survival modeling
         - piecewise exponential survival
         - true time-dependent treatment effect
         - Leroux spatial frailty over ZIPs

    Output columns support both models:

      Survival model:
        - time
        - event
        - treatment_time   (NaN if treatment not observed before death/censoring)

      Treatment model:
        - treatment_time_obs
        - treatment_event

    Parameters
    ----------
    treatment_spatial_mode
        One of:
          - "none": treatment time has no spatial effect
          - "leroux": treatment time includes its own Leroux ZIP effect

    Notes
    -----
    - Survival and treatment spatial fields are separate in this simulator.
    - treatment_time_true is always simulated, but treatment may be censored by
      death or administrative/random censoring before it is observed.
    - treatment_time_obs = min(treatment_time_true, treatment censoring time)
    - treatment_event = 1 iff treatment occurs before death/censoring
    """
    rng = np.random.default_rng(seed)

    if admin_censor is None:
        admin_censor = float(breaks[-1])

    if len(breaks) != len(nu) + 1:
        raise ValueError("len(breaks) must equal len(nu) + 1")

    if sigma_treatment <= 0.0:
        raise ValueError("sigma_treatment must be > 0")

    if treatment_spatial_mode not in {"none", "leroux"}:
        raise ValueError("treatment_spatial_mode must be one of {'none', 'leroux'}")

    # ----------------------------
    # Build graph and sample spatial frailty for survival
    # ----------------------------
    zips = _load_zip_universe(zips_path)
    edges_df = pd.read_csv(edges_path)

    graph = build_graph_from_edge_list(
        zips,
        edges_df,
        col_u=edges_u_col,
        col_v=edges_v_col,
    )

    W = graph.W().toarray().astype(float)
    D = np.diag(W.sum(axis=1))
    component_ids = graph.component_ids()

    u_surv = sample_leroux_u(
        W=W,
        D=D,
        tau=float(tau_true),
        rho=float(rho_true),
        rng=rng,
        q_jitter=1e-10,
        component_ids=component_ids,
        weights=None,
    )
    zip_to_u_surv = {str(z): float(u_surv[i]) for i, z in enumerate(graph.zips)}

    # ----------------------------
    # Optional treatment spatial field
    # ----------------------------
    if treatment_spatial_mode == "leroux":
        u_treat = sample_leroux_u(
            W=W,
            D=D,
            tau=float(treatment_tau_true),
            rho=float(treatment_rho_true),
            rng=rng,
            q_jitter=1e-10,
            component_ids=component_ids,
            weights=None,
        )
        zip_to_u_treat = {str(z): float(u_treat[i]) for i, z in enumerate(graph.zips)}
    else:
        u_treat = np.zeros(len(graph.zips), dtype=float)
        zip_to_u_treat = {str(z): 0.0 for z in graph.zips}

    # ----------------------------
    # Baseline subject data
    # ----------------------------
    df = _simulate_baseline_covariates(n=n, rng=rng)
    df["id"] = np.arange(1, n + 1)
    df["zip"] = rng.choice(np.asarray(graph.zips, dtype=object), size=n, replace=True)

    # ----------------------------
    # Latent treatment times
    # ----------------------------
    u_treat_subject = np.array([zip_to_u_treat[str(z)] for z in df["zip"].astype(str)], dtype=float)

    t_treat_latent = _simulate_treatment_times_lognormal_aft(
        df,
        beta_treatment=beta_treatment,
        sigma_treatment=float(sigma_treatment),
        treatment_intercept=float(treatment_intercept),
        rng=rng,
        spatial_offset=(u_treat_subject if treatment_spatial_mode == "leroux" else None),
    )

    # ----------------------------
    # Latent event times: spatial TD survival model
    # ----------------------------
    eta_base_survival = _linear_predictor_from_wide(df, beta=beta_survival)
    u_surv_subject = np.array([zip_to_u_surv[str(z)] for z in df["zip"].astype(str)], dtype=float)
    eta_spatial_survival = eta_base_survival + u_surv_subject

    t_event_latent = np.empty(n, dtype=float)
    for i in range(n):
        eta_pre = float(eta_spatial_survival[i])
        eta_post = float(eta_spatial_survival[i] + gamma_treated)
        t_event_latent[i] = simulate_event_time_piecewise_exp_with_switch(
            breaks=breaks,
            nu=nu,
            eta_pre=eta_pre,
            eta_post=eta_post,
            switch_time=float(t_treat_latent[i]),
            rng=rng,
        )

    # enforce finite latent survival times for downstream use
    t_event_latent = np.where(
        np.isfinite(t_event_latent),
        t_event_latent,
        float(breaks[-1]),
    )

    # ----------------------------
    # External censoring
    # ----------------------------
    if random_censor_rate > 0.0:
        c_rand = rng.exponential(scale=1.0 / random_censor_rate, size=n)
    else:
        c_rand = np.full(n, np.inf, dtype=float)

    c_admin = np.full(n, float(admin_censor), dtype=float)
    c = np.minimum(c_rand, c_admin)

    # ----------------------------
    # Observed survival outcome
    # ----------------------------
    time_obs = np.minimum(t_event_latent, c)
    event_obs = (t_event_latent <= c).astype(int)

    # ----------------------------
    # Observed treatment process
    # ----------------------------
    treatment_censor_time = np.minimum(t_event_latent, c)
    treatment_event_obs = (t_treat_latent <= treatment_censor_time).astype(int)
    treatment_time_obs = np.minimum(t_treat_latent, treatment_censor_time)

    treatment_time_for_survival = np.where(
        treatment_event_obs == 1,
        t_treat_latent,
        np.nan,
    )

    out = pd.DataFrame(
        {
            "id": df["id"].to_numpy(dtype=int),
            "zip": df["zip"].astype(str).to_numpy(),
            "age_per10_centered": df["age_per10_centered"].to_numpy(dtype=float),
            "cci": df["cci"].to_numpy(dtype=int),
            "tumor_size_log": df["tumor_size_log"].to_numpy(dtype=float),
            "ses": df["ses"].to_numpy(dtype=float),
            "sex": df["sex"].astype(str).to_numpy(),
            "stage": df["stage"].astype(str).to_numpy(),
            "treatment_time": treatment_time_for_survival.astype(float),
            "treatment_time_obs": np.maximum(treatment_time_obs, 1e-8).astype(float),
            "treatment_event": treatment_event_obs.astype(int),
            "time": np.maximum(time_obs, 1e-8).astype(float),
            "event": event_obs.astype(int),
        }
    )

    if return_latent_truth:
        out["eta_base_survival_true"] = eta_base_survival
        out["u_true"] = u_surv_subject
        out["eta_spatial_survival_true"] = eta_spatial_survival
        out["gamma_treated_true"] = float(gamma_treated)
        out["tau_true"] = float(tau_true)
        out["rho_true"] = float(rho_true)

        out["sigma_treatment_true"] = float(sigma_treatment)
        out["treatment_intercept_true"] = float(treatment_intercept)
        out["treatment_spatial_mode_true"] = str(treatment_spatial_mode)
        out["u_treatment_true"] = u_treat_subject
        out["treatment_tau_true"] = float(treatment_tau_true if treatment_spatial_mode == "leroux" else 0.0)
        out["treatment_rho_true"] = float(treatment_rho_true if treatment_spatial_mode == "leroux" else 0.0)

        out["treatment_time_true"] = t_treat_latent
        out["survival_time_true"] = t_event_latent
        out["censor_time"] = c
        out["treated_observed"] = treatment_event_obs.astype(int)

    return out