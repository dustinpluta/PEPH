from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PEParams:
    breaks: List[float]        # length K+1
    nu: np.ndarray            # length K, baseline hazards per interval (per day)


def baseline_cumhaz(breaks: Sequence[float], nu: np.ndarray, t: float) -> float:
    """
    H0(t) = sum_k nu_k * exposure_k(t), with [a,b) convention.
    """
    K = len(breaks) - 1
    H = 0.0
    for k in range(K):
        a = float(breaks[k])
        b = float(breaks[k + 1])
        dt = min(max(t, a), b) - a
        if dt > 0:
            H += float(nu[k]) * float(dt)
    return float(H)


def invert_baseline_cumhaz(breaks: Sequence[float], nu: np.ndarray, H_target: float) -> float:
    """
    Find t such that H0(t) = H_target, by walking intervals.
    Assumes nu_k > 0.
    """
    if H_target <= 0:
        return 0.0

    K = len(breaks) - 1
    H = 0.0
    for k in range(K):
        a = float(breaks[k])
        b = float(breaks[k + 1])
        rate = float(nu[k])
        if rate <= 0:
            raise ValueError("nu must be > 0 for inversion")
        width = b - a
        inc = rate * width
        if H + inc >= H_target:
            dt = (H_target - H) / rate
            return a + dt
        H += inc
    # If beyond last break, extrapolate using last interval hazard
    last_rate = float(nu[-1])
    extra = (H_target - H) / last_rate
    return float(breaks[-1] + extra)


def simulate_event_time_piecewise_exp(
    *,
    breaks: Sequence[float],
    nu: np.ndarray,
    eta: float,
    rng: np.random.Generator,
) -> float:
    """
    T = H0^{-1}(E / exp(eta)), with E ~ Exp(1).
    """
    E = rng.exponential(scale=1.0)
    H_target = float(E / np.exp(float(eta)))
    return invert_baseline_cumhaz(breaks, nu, H_target)


def simulate_peph_spatial_dataset(
    *,
    n: int,
    breaks: List[float],
    nu: np.ndarray,
    beta: Dict[str, float],
    # covariate generation
    x_numeric: List[str],
    x_categorical: List[str],
    cat_levels: Dict[str, List[str]],
    cat_ref: Dict[str, str],
    # spatial
    zips: List[str],
    zip_to_u: Dict[str, float],
    # censoring
    admin_censor: float = 1825.0,
    random_censor_rate: float = 0.0,  # 0 => none
    seed: int = 0,
) -> pd.DataFrame:
    """
    Simulate wide-form dataset with columns:
      id, time, event, x_numeric..., x_categorical..., zip

    Event times follow PE-PH with linear predictor:
      eta_i = sum_j beta_j x_ij + u_zip[i]
    Right censoring only:
      time = min(T, C_admin, C_rand)
      event = 1[T <= censor]
    """
    rng = np.random.default_rng(seed)
    if len(breaks) < 2:
        raise ValueError("breaks must have length >= 2")
    if nu.shape[0] != len(breaks) - 1:
        raise ValueError("nu length must be len(breaks)-1")
    if admin_censor <= 0:
        raise ValueError("admin_censor must be > 0")

    # --- generate covariates ---
    df = pd.DataFrame({"id": np.arange(n, dtype=int)})

    for col in x_numeric:
        # standard-ish numeric covariates
        df[col] = rng.normal(size=n)

    for col in x_categorical:
        levels = cat_levels[col]
        df[col] = rng.choice(levels, size=n)

    df["zip"] = rng.choice(zips, size=n)

    # --- build design for eta ---
    eta = np.zeros(n, dtype=float)

    # numeric effects
    for col in x_numeric:
        eta += float(beta.get(col, 0.0)) * df[col].to_numpy(dtype=float)

    # categorical (reference coding)
    # For each non-ref level, add beta[col + level] if provided, else 0.
    for col in x_categorical:
        ref = cat_ref[col]
        vals = df[col].astype(str).to_numpy()
        for lvl in cat_levels[col]:
            if lvl == ref:
                continue
            name = f"{col}{lvl}"
            b = float(beta.get(name, 0.0))
            eta += b * (vals == lvl).astype(float)

    # spatial u
    u = np.array([float(zip_to_u[str(z)]) for z in df["zip"].astype(str).to_numpy()], dtype=float)
    eta += u

    # --- simulate event times ---
    T = np.empty(n, dtype=float)
    for i in range(n):
        T[i] = simulate_event_time_piecewise_exp(breaks=breaks, nu=nu, eta=float(eta[i]), rng=rng)

    # censoring
    C_admin = float(admin_censor) * np.ones(n, dtype=float)
    if random_censor_rate > 0:
        C_rand = rng.exponential(scale=1.0 / random_censor_rate, size=n)
    else:
        C_rand = np.full(n, np.inf, dtype=float)

    C = np.minimum(C_admin, C_rand)
    time = np.minimum(T, C)
    event = (T <= C).astype(int)

    df["time"] = time
    df["event"] = event

    return df