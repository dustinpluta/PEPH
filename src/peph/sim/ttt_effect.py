from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def baseline_cumhaz_piecewise(
    t: float,
    *,
    breaks: List[float],
    nu: np.ndarray,
) -> float:
    """
    Baseline cumulative hazard H0(t) for a piecewise-exponential hazard.

    Intervals are [breaks[k], breaks[k+1]) with constant hazard nu[k].
    """
    if t <= 0.0:
        return 0.0

    b = np.asarray(breaks, dtype=float)
    nu = np.asarray(nu, dtype=float)

    if len(b) != len(nu) + 1:
        raise ValueError("len(breaks) must equal len(nu) + 1")

    t_eff = min(float(t), float(b[-1]))
    out = 0.0

    for k in range(len(nu)):
        a = float(b[k])
        c = float(b[k + 1])
        if t_eff <= a:
            break
        dt = min(t_eff, c) - a
        if dt > 0.0:
            out += float(nu[k]) * dt

    return float(out)


def invert_baseline_cumhaz_piecewise(
    target: float,
    *,
    breaks: List[float],
    nu: np.ndarray,
) -> float:
    """
    Solve H0(t) = target for t under a piecewise-exponential baseline hazard.

    Returns a time in [0, breaks[-1]] if the target is attainable within the
    administrative horizon, otherwise returns np.inf.
    """
    if target <= 0.0:
        return 0.0

    b = np.asarray(breaks, dtype=float)
    nu = np.asarray(nu, dtype=float)

    if len(b) != len(nu) + 1:
        raise ValueError("len(breaks) must equal len(nu) + 1")

    acc = 0.0
    for k in range(len(nu)):
        a = float(b[k])
        c = float(b[k + 1])
        width = c - a
        inc = float(nu[k]) * width

        if target <= acc + inc:
            rate = float(nu[k])
            if rate <= 0.0:
                return np.inf
            return float(a + (target - acc) / rate)

        acc += inc

    return float(np.inf)


def simulate_event_time_piecewise_exp_with_switch(
    *,
    breaks: List[float],
    nu: np.ndarray,
    eta_pre: float,
    eta_post: float,
    switch_time: float,
    rng: np.random.Generator,
) -> float:
    """
    Simulate a survival time when the log-hazard shifts at switch_time.

    Hazard:
        h(t) = h0(t) * exp(eta_pre),  for t < switch_time
             = h0(t) * exp(eta_post), for t >= switch_time

    Notes
    -----
    - switch_time may be np.inf, in which case the subject is never treated.
    - If the simulated event does not occur within breaks[-1], returns np.inf.
    """
    if switch_time < 0.0:
        raise ValueError("switch_time must be nonnegative")

    z = rng.exponential(scale=1.0)

    # Never treated within horizon
    if not np.isfinite(switch_time):
        target_H0 = z / np.exp(eta_pre)
        return invert_baseline_cumhaz_piecewise(target_H0, breaks=breaks, nu=nu)

    H0_switch = baseline_cumhaz_piecewise(switch_time, breaks=breaks, nu=nu)
    pre_contrib = np.exp(eta_pre) * H0_switch

    # Event occurs before treatment
    if z <= pre_contrib:
        target_H0 = z / np.exp(eta_pre)
        return invert_baseline_cumhaz_piecewise(target_H0, breaks=breaks, nu=nu)

    # Event occurs after treatment
    remaining = z - pre_contrib
    target_post_H0 = H0_switch + remaining / np.exp(eta_post)
    return invert_baseline_cumhaz_piecewise(target_post_H0, breaks=breaks, nu=nu)


def _simulate_baseline_covariates(
    *,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate SEER-like CRC baseline covariates.
    """
    age_years = np.clip(rng.normal(loc=72.0, scale=8.0, size=n), 50.0, 95.0)
    age_per10_centered = (age_years - 70.0) / 10.0

    cci = np.clip(rng.poisson(lam=1.3, size=n), 0, 6)
    tumor_size_log = rng.normal(loc=3.4, scale=0.45, size=n)
    ses = rng.normal(loc=0.0, scale=1.0, size=n)

    sex = rng.choice(["F", "M"], size=n, p=[0.48, 0.52])
    stage = rng.choice(["I", "II", "III"], size=n, p=[0.30, 0.40, 0.30])

    return pd.DataFrame(
        {
            "id": np.arange(1, n + 1, dtype=int),
            "age_per10_centered": age_per10_centered,
            "cci": cci.astype(int),
            "tumor_size_log": tumor_size_log,
            "ses": ses,
            "sex": sex,
            "stage": stage,
        }
    )


def _linear_predictor_from_wide(
    df: pd.DataFrame,
    *,
    beta: Dict[str, float],
) -> np.ndarray:
    """
    Build the baseline survival linear predictor from wide data.
    """
    eta = np.zeros(len(df), dtype=float)

    if "age_per10_centered" in df.columns:
        eta += float(beta.get("age_per10_centered", 0.0)) * df["age_per10_centered"].to_numpy(dtype=float)
    if "cci" in df.columns:
        eta += float(beta.get("cci", 0.0)) * df["cci"].to_numpy(dtype=float)
    if "tumor_size_log" in df.columns:
        eta += float(beta.get("tumor_size_log", 0.0)) * df["tumor_size_log"].to_numpy(dtype=float)
    if "ses" in df.columns:
        eta += float(beta.get("ses", 0.0)) * df["ses"].to_numpy(dtype=float)

    if "sex" in df.columns:
        eta += float(beta.get("sexM", 0.0)) * (df["sex"].astype(str).to_numpy() == "M").astype(float)

    if "stage" in df.columns:
        stage_vals = df["stage"].astype(str).to_numpy()
        eta += float(beta.get("stageII", 0.0)) * (stage_vals == "II").astype(float)
        eta += float(beta.get("stageIII", 0.0)) * (stage_vals == "III").astype(float)

    return eta


def _simulate_treatment_times(
    df: pd.DataFrame,
    *,
    rng: np.random.Generator,
    max_treatment_time: float,
) -> np.ndarray:
    """
    Simulate latent treatment times from a lognormal AFT-style model.

    This is deliberately simple for the first recovery study.
    """
    mu = (
        np.log(75.0)
        + 0.10 * df["age_per10_centered"].to_numpy(dtype=float)
        + 0.08 * df["cci"].to_numpy(dtype=float)
        - 0.12 * df["ses"].to_numpy(dtype=float)
        + 0.12 * (df["stage"].astype(str).to_numpy() == "III").astype(float)
    )
    sigma = 0.45

    t = np.exp(mu + sigma * rng.normal(size=len(df)))
    t = np.maximum(t, 1.0)

    # Some subjects effectively never treated within the treatment window
    never_treated = rng.uniform(size=len(df)) < 0.08
    t[never_treated] = np.inf

    # If treatment occurs beyond this window, treat as unobserved in practice
    t = np.where(t <= max_treatment_time, t, np.inf)
    return t.astype(float)


def simulate_peph_ttt_effect_dataset(
    *,
    n: int,
    breaks: List[float],
    nu: np.ndarray,
    beta: Dict[str, float],
    gamma_treated: float,
    admin_censor: Optional[float] = None,
    random_censor_rate: float = 0.0006,
    max_treatment_time: float = 365.0,
    seed: int = 0,
    return_latent_truth: bool = True,
) -> pd.DataFrame:
    """
    Simulate wide survival data with a true time-dependent treatment effect.

    Survival hazard:
        h(t) = h0(t) * exp(eta + gamma_treated * treated(t))

    where treated(t) switches from 0 to 1 at the latent treatment time.

    Output columns
    --------------
    id, time, event, treatment_time,
    age_per10_centered, cci, tumor_size_log, ses, sex, stage,
    and optional latent-truth columns prefixed with "_".
    """
    rng = np.random.default_rng(seed)

    if admin_censor is None:
        admin_censor = float(breaks[-1])

    if len(breaks) != len(nu) + 1:
        raise ValueError("len(breaks) must equal len(nu) + 1")

    df = _simulate_baseline_covariates(n=n, rng=rng)
    eta_base = _linear_predictor_from_wide(df, beta=beta)

    t_treat_latent = _simulate_treatment_times(
        df,
        rng=rng,
        max_treatment_time=max_treatment_time,
    )

    t_event_latent = np.empty(n, dtype=float)
    for i in range(n):
        eta_pre = float(eta_base[i])
        eta_post = float(eta_base[i] + gamma_treated)
        t_event_latent[i] = simulate_event_time_piecewise_exp_with_switch(
            breaks=breaks,
            nu=nu,
            eta_pre=eta_pre,
            eta_post=eta_post,
            switch_time=float(t_treat_latent[i]),
            rng=rng,
        )

    if random_censor_rate > 0.0:
        c_rand = rng.exponential(scale=1.0 / random_censor_rate, size=n)
    else:
        c_rand = np.full(n, np.inf, dtype=float)

    c_admin = np.full(n, float(admin_censor), dtype=float)
    c = np.minimum(c_rand, c_admin)

    time_obs = np.minimum(t_event_latent, c)
    event_obs = (t_event_latent <= c).astype(int)

    # treatment_time is observed only if treatment occurs by end of observed follow-up
    treatment_time_obs = np.where(t_treat_latent <= time_obs, t_treat_latent, np.nan)

    df["time"] = time_obs.astype(float)
    df["event"] = event_obs.astype(int)
    df["treatment_time"] = treatment_time_obs.astype(float)

    if return_latent_truth:
        df["_eta_base_true"] = eta_base
        df["_gamma_treated_true"] = float(gamma_treated)
        df["_treatment_time_latent"] = t_treat_latent
        df["_event_time_latent"] = t_event_latent
        df["_censor_time_true"] = c
        df["_treated_observed"] = np.isfinite(treatment_time_obs).astype(int)

    return df