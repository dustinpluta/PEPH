from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from peph.model.design import build_x_wide_for_prediction
from peph.model.result import FittedPEPHModel


def _cumhaz_single_time(breaks: np.ndarray, nu: np.ndarray, rate_mult: float, t: float) -> float:
    """
    Compute H(t) for piecewise-constant baseline hazard nu_k on intervals [b_k, b_{k+1}),
    scaled by exp(eta)=rate_mult.
    """
    if t <= 0:
        return 0.0
    tmax = breaks[-1]
    t_obs = min(float(t), float(tmax))

    H = 0.0
    K = len(nu)
    for k in range(K):
        t0 = breaks[k]
        t1 = breaks[k + 1]
        if t_obs <= t0:
            break
        dt = min(t_obs, t1) - t0
        if dt > 0:
            H += nu[k] * rate_mult * dt
        if t_obs < t1:
            break
    return float(H)


def predict_cumhaz(
    model: FittedPEPHModel,
    wide_df: pd.DataFrame,
    times: Iterable[float],
) -> np.ndarray:
    breaks = np.asarray(model.breaks, dtype=float)
    nu = np.asarray(model.nu, dtype=float)
    K = len(nu)
    params = model.params_array()
    beta = params[K:]

    X = build_x_wide_for_prediction(
        wide_df,
        x_numeric=model.encoding.x_numeric,
        x_categorical=model.encoding.x_categorical,
        categorical_reference_levels=model.encoding.categorical_reference_levels,
        categorical_levels_seen=model.encoding.categorical_levels_seen,
        x_col_names=model.x_col_names,
        hard_fail=True,
    )
    eta = X @ beta
    rate_mult = np.exp(eta)

    times = list(times)
    out = np.zeros((len(wide_df), len(times)), dtype=float)
    for j, t in enumerate(times):
        for i in range(len(wide_df)):
            out[i, j] = _cumhaz_single_time(breaks, nu, float(rate_mult[i]), float(t))
    return out


def predict_survival(model: FittedPEPHModel, wide_df: pd.DataFrame, times: Iterable[float]) -> np.ndarray:
    H = predict_cumhaz(model, wide_df, times)
    return np.exp(-H)


def predict_risk(model: FittedPEPHModel, wide_df: pd.DataFrame, horizons: Iterable[float]) -> np.ndarray:
    S = predict_survival(model, wide_df, horizons)
    return 1.0 - S


def predict_linear_predictor(model: FittedPEPHModel, wide_df: pd.DataFrame) -> np.ndarray:
    K = model.K
    params = model.params_array()
    beta = params[K:]

    X = build_x_wide_for_prediction(
        wide_df,
        x_numeric=model.encoding.x_numeric,
        x_categorical=model.encoding.x_categorical,
        categorical_reference_levels=model.encoding.categorical_reference_levels,
        categorical_levels_seen=model.encoding.categorical_levels_seen,
        x_col_names=model.x_col_names,
        hard_fail=True,
    )
    return X @ beta


def make_test_prediction_frame(
    model: FittedPEPHModel,
    test_wide: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    event_col: str,
    horizons: List[float],
) -> pd.DataFrame:
    eta = predict_linear_predictor(model, test_wide)
    risk = predict_risk(model, test_wide, horizons=horizons)

    out = pd.DataFrame(
        {
            id_col: test_wide[id_col].values,
            time_col: test_wide[time_col].values,
            event_col: test_wide[event_col].values,
            "eta": eta,
        }
    )
    for j, t in enumerate(horizons):
        out[f"risk_t{int(t)}"] = risk[:, j]
    return out