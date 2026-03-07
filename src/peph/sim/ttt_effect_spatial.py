from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from peph.sim.spatial import sample_leroux_u
from peph.sim.ttt_effect import (
    _linear_predictor_from_wide,
    _simulate_baseline_covariates,
    _simulate_treatment_times,
    simulate_event_time_piecewise_exp_with_switch,
)
from peph.spatial.graph import build_graph_from_edge_list


def _load_zip_universe(zips_path: str) -> List[str]:
    zdf = pd.read_csv(zips_path)
    if "zip" in zdf.columns:
        return zdf["zip"].astype(str).tolist()
    if zdf.shape[1] == 1:
        return zdf.iloc[:, 0].astype(str).tolist()
    raise ValueError(f"ZIP universe file must contain a 'zip' column or be single-column: {zips_path}")


def simulate_peph_spatial_ttt_effect_dataset(
    *,
    n: int,
    breaks: List[float],
    nu: np.ndarray,
    beta: Dict[str, float],
    gamma_treated: float,
    zips_path: str,
    edges_path: str,
    edges_u_col: str = "zip_u",
    edges_v_col: str = "zip_v",
    tau_true: float = 2.0,
    rho_true: float = 0.85,
    admin_censor: Optional[float] = None,
    random_censor_rate: float = 0.0006,
    max_treatment_time: float = 365.0,
    seed: int = 0,
    return_latent_truth: bool = True,
) -> pd.DataFrame:
    """
    Simulate wide survival data with:
      - a true time-dependent treatment effect gamma_treated
      - a Leroux spatial frailty over the supplied ZIP graph

    Survival hazard:
        h_i(t) = h0(t) * exp(eta_base_i + u_zip(i) + gamma_treated * treated_i(t))
    """
    rng = np.random.default_rng(seed)

    if admin_censor is None:
        admin_censor = float(breaks[-1])

    if len(breaks) != len(nu) + 1:
        raise ValueError("len(breaks) must equal len(nu) + 1")

    # ----------------------------
    # Build graph and sample frailty
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

    u_true = sample_leroux_u(
        W=W,
        D=D,
        tau=float(tau_true),
        rho=float(rho_true),
        rng=rng,
        q_jitter=1e-10,
        component_ids=component_ids,
        weights=None,
    )
    zip_to_u = {str(z): float(u_true[i]) for i, z in enumerate(graph.zips)}

    # ----------------------------
    # Baseline subject data
    # ----------------------------
    df = _simulate_baseline_covariates(n=n, rng=rng)
    df["zip"] = rng.choice(np.asarray(graph.zips, dtype=object), size=n, replace=True)

    eta_base = _linear_predictor_from_wide(df, beta=beta)
    u_subject = np.array([zip_to_u[str(z)] for z in df["zip"].astype(str)], dtype=float)
    eta_spatial = eta_base + u_subject

    # ----------------------------
    # Latent treatment times
    # ----------------------------
    t_treat_latent = _simulate_treatment_times(
        df,
        rng=rng,
        max_treatment_time=max_treatment_time,
    )

    # ----------------------------
    # Latent event times with treatment switch
    # ----------------------------
    t_event_latent = np.empty(n, dtype=float)
    for i in range(n):
        eta_pre = float(eta_spatial[i])
        eta_post = float(eta_spatial[i] + gamma_treated)
        t_event_latent[i] = simulate_event_time_piecewise_exp_with_switch(
            breaks=breaks,
            nu=nu,
            eta_pre=eta_pre,
            eta_post=eta_post,
            switch_time=float(t_treat_latent[i]),
            rng=rng,
        )

    # ----------------------------
    # Censoring
    # ----------------------------
    if random_censor_rate > 0.0:
        c_rand = rng.exponential(scale=1.0 / random_censor_rate, size=n)
    else:
        c_rand = np.full(n, np.inf, dtype=float)

    c_admin = np.full(n, float(admin_censor), dtype=float)
    c = np.minimum(c_rand, c_admin)

    time_obs = np.minimum(t_event_latent, c)
    event_obs = (t_event_latent <= c).astype(int)

    # treatment_time observed only if treatment occurs by end of observed follow-up
    treatment_time_obs = np.where(t_treat_latent <= time_obs, t_treat_latent, np.nan)

    df["time"] = time_obs.astype(float)
    df["event"] = event_obs.astype(int)
    df["treatment_time"] = treatment_time_obs.astype(float)

    if return_latent_truth:
        df["_eta_base_true"] = eta_base
        df["_u_true"] = u_subject
        df["_eta_spatial_true"] = eta_spatial
        df["_gamma_treated_true"] = float(gamma_treated)
        df["_tau_true"] = float(tau_true)
        df["_rho_true"] = float(rho_true)
        df["_treatment_time_latent"] = t_treat_latent
        df["_event_time_latent"] = t_event_latent
        df["_censor_time_true"] = c
        df["_treated_observed"] = np.isfinite(treatment_time_obs).astype(int)

    return df