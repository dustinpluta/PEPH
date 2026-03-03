from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PHSimSpec:
    """
    Simulation spec for piecewise exponential PH.

    Baseline is piecewise-constant hazard nu[k] on [breaks[k], breaks[k+1]).
    Covariates are time-constant; hazard multiplier is exp(eta).

    Notes
    -----
    beta is specified on the *expanded* feature names used by the model:
      - numeric covariates by their column name (e.g. "age_per10_centered")
      - categorical one-hot names following PR1 convention:
          sexM, stageII, stageIII, stageIV
    """
    breaks: List[float]                 # e.g. [0, 30, 90, 180, 365, 730, 1825]
    nu: List[float]                     # length K = len(breaks)-1, hazards per day
    beta: Dict[str, float]              # expanded feature coefficients
    seed: int = 1

    # admin censoring at max follow-up by default
    admin_censor_days: Optional[float] = None

    # random censoring: exponential(rate) if enabled
    censoring_enabled: bool = True
    censoring_rate: float = 0.0008

    # Covariate generation parameters (defaults are reasonable for demos)
    # Numeric:
    age_mean: float = 0.0
    age_sd: float = 1.0
    cci_mean: float = 1.5
    tumor_mean: float = 0.0
    tumor_sd: float = 0.5
    ses_mean: float = 0.0
    ses_sd: float = 1.0

    # Categorical:
    p_male: float = 0.48
    stage_probs: Tuple[float, float, float, float] = (0.35, 0.30, 0.20, 0.15)  # I, II, III, IV


def _validate_spec(spec: PHSimSpec) -> None:
    b = np.asarray(spec.breaks, dtype=float)
    nu = np.asarray(spec.nu, dtype=float)
    if b.ndim != 1 or b.size < 2 or b[0] != 0 or np.any(np.diff(b) <= 0):
        raise ValueError("breaks must be strictly increasing and start at 0")
    if nu.size != b.size - 1:
        raise ValueError("nu must have length len(breaks)-1")
    if np.any(nu <= 0):
        raise ValueError("nu must be positive")
    if spec.admin_censor_days is not None and spec.admin_censor_days <= 0:
        raise ValueError("admin_censor_days must be >0")


def _draw_event_time_piecewise(breaks: np.ndarray, nu: np.ndarray, rate_mult: float, rng: np.random.Generator) -> float:
    """
    Draw T from piecewise exponential hazard:
      lambda(t in k) = nu[k] * rate_mult, constant within each interval.

    Returns event time T (may exceed breaks[-1] if hazard is low and we continue past).
    For our use we will truncate with admin censoring, but T is sampled without truncation.
    """
    # inverse CDF by accumulating piecewise cumulative hazard until crossing -log(U)
    u = rng.uniform()
    target = -np.log(u)  # Exp(1) draw

    H = 0.0
    for k in range(len(nu)):
        t0 = breaks[k]
        t1 = breaks[k + 1]
        dt = t1 - t0
        lam = nu[k] * rate_mult
        incr = lam * dt
        if H + incr >= target:
            # event occurs within this interval
            remaining = target - H
            return float(t0 + remaining / lam)
        H += incr

    # If we didn't hit within the break window, extend with last hazard beyond breaks[-1]
    # (this is mostly irrelevant once admin censoring applied, but keeps distribution well-defined)
    lam_last = nu[-1] * rate_mult
    extra = (target - H) / lam_last
    return float(breaks[-1] + extra)


def _expanded_linear_predictor(row: dict, beta: Dict[str, float]) -> float:
    """
    Compute eta using expanded features consistent with PR1 naming.
    Expected row keys:
      age_per10_centered, cci, tumor_size_log, ses, sex, stage
    """
    eta = 0.0

    # numeric (if present in beta)
    for k in ["age_per10_centered", "cci", "tumor_size_log", "ses"]:
        if k in beta:
            eta += float(beta[k]) * float(row[k])

    # sex one-hot: reference is F => sexM
    if "sexM" in beta:
        eta += float(beta["sexM"]) * (1.0 if row["sex"] == "M" else 0.0)

    # stage one-hot: reference is I => stageII, stageIII, stageIV
    for lvl in ["II", "III", "IV"]:
        name = f"stage{lvl}"
        if name in beta:
            eta += float(beta[name]) * (1.0 if row["stage"] == lvl else 0.0)

    return float(eta)


def simulate_ph_wide(
    n: int,
    spec: PHSimSpec,
    *,
    include_debug_cols: bool = True,
) -> pd.DataFrame:
    """
    Simulate PH survival data in canonical wide format:
      id, time (days), event, plus covariates:
        age_per10_centered, cci, tumor_size_log, ses, sex, stage

    Debug columns (optional, prefixed with _):
      _T_true, _C_true, _eta_true, _admin_cens, _nu_true
    """
    _validate_spec(spec)
    rng = np.random.default_rng(spec.seed)

    breaks = np.asarray(spec.breaks, dtype=float)
    nu = np.asarray(spec.nu, dtype=float)
    t_admin = float(spec.admin_censor_days) if spec.admin_censor_days is not None else float(breaks[-1])

    # covariates
    age = rng.normal(spec.age_mean, spec.age_sd, size=n)
    # keep cci >=0 and modest; use Poisson with mean ~ cci_mean
    cci = rng.poisson(lam=max(spec.cci_mean, 0.01), size=n).astype(float)
    tumor = rng.normal(spec.tumor_mean, spec.tumor_sd, size=n)
    ses = rng.normal(spec.ses_mean, spec.ses_sd, size=n)

    sex = np.where(rng.uniform(size=n) < spec.p_male, "M", "F")

    stage_levels = np.array(["I", "II", "III", "IV"], dtype=object)
    probs = np.asarray(spec.stage_probs, dtype=float)
    probs = probs / probs.sum()
    stage = rng.choice(stage_levels, size=n, replace=True, p=probs)

    rows = []
    T_true = np.zeros(n, dtype=float)
    C_true = np.zeros(n, dtype=float)
    eta_true = np.zeros(n, dtype=float)

    for i in range(n):
        row = {
            "age_per10_centered": float(age[i]),
            "cci": float(cci[i]),
            "tumor_size_log": float(tumor[i]),
            "ses": float(ses[i]),
            "sex": str(sex[i]),
            "stage": str(stage[i]),
        }
        eta = _expanded_linear_predictor(row, spec.beta)
        eta_true[i] = eta
        rate_mult = float(np.exp(eta))

        T = _draw_event_time_piecewise(breaks, nu, rate_mult, rng)
        T_true[i] = T

        if spec.censoring_enabled:
            # exponential with rate censoring_rate
            C = rng.exponential(scale=1.0 / spec.censoring_rate)
        else:
            C = np.inf
        C_true[i] = C

        obs = min(T, C, t_admin)
        event = 1 if (T <= C and T <= t_admin) else 0

        rows.append(
            {
                "id": i + 1,
                "time": float(obs),
                "event": int(event),
                **row,
            }
        )

    df = pd.DataFrame(rows)

    if include_debug_cols:
        df["_T_true"] = T_true
        df["_C_true"] = C_true
        df["_eta_true"] = eta_true
        df["_admin_cens"] = float(t_admin)
        df["_nu_true"] = ",".join([f"{x:.8g}" for x in nu.tolist()])

    return df