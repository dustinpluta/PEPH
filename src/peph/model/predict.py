from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from peph.model.design import build_x_wide_for_prediction
from peph.model.frailty import FrailtyMode, get_frailty_vector_for_wide
from peph.model.result import FittedPEPHModel


def _coerce_args_model_wide(
    a: object, b: object
) -> tuple[pd.DataFrame, FittedPEPHModel]:
    """
    Backward-compatible argument coercion.

    Accept either:
      - (wide_df, model)  [new]
      - (model, wide_df)  [old]
    """
    if isinstance(a, pd.DataFrame) and isinstance(b, FittedPEPHModel):
        return a, b
    if isinstance(a, FittedPEPHModel) and isinstance(b, pd.DataFrame):
        return b, a
    raise TypeError(
        "Expected arguments (wide_df: DataFrame, model: FittedPEPHModel) "
        "or (model: FittedPEPHModel, wide_df: DataFrame)."
    )


def _baseline_cumhaz_at_times(breaks: list[float], nu: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Baseline cumulative hazard H0(t) for a piecewise-constant hazard:
      H0(t) = sum_k nu_k * exposure_k(t),
    where exposure_k(t) is time spent in interval k up to time t.

    breaks define [b_k, b_{k+1}) and times are in same units (days).
    """
    K = len(breaks) - 1
    if nu.shape[0] != K:
        raise ValueError("nu length does not match breaks")

    t = np.asarray(times, dtype=float)
    if t.ndim != 1:
        raise ValueError("times must be 1D")
    if np.any(t < 0):
        raise ValueError("times must be nonnegative")

    H0 = np.zeros_like(t, dtype=float)
    for k in range(K):
        a = float(breaks[k])
        b = float(breaks[k + 1])
        dt = np.clip(t, a, b) - a
        dt = np.maximum(dt, 0.0)
        H0 += float(nu[k]) * dt
    return H0


def predict_linear_predictor(
    a: object,
    b: object,
    *,
    frailty_mode: FrailtyMode = "auto",
    allow_unseen_area: Optional[bool] = None,
    hard_fail: bool = True,
) -> np.ndarray:
    """
    Return eta_i = x_i^T beta (+ u_zip if enabled).

    Backward compatible:
      - predict_linear_predictor(wide_df, model, ...)
      - predict_linear_predictor(model, wide_df, ...)
    """
    wide_df, model = _coerce_args_model_wide(a, b)

    enc = model.encoding
    K = len(model.baseline_col_names)
    params = np.asarray(model.params, dtype=float)
    beta = params[K:]

    X, _ = build_x_wide_for_prediction(
        wide_df,
        x_numeric=enc.x_numeric,
        x_categorical=enc.x_categorical,
        categorical_reference_levels=enc.categorical_reference_levels,
        categorical_levels_seen=enc.categorical_levels_seen,
        x_col_names=enc.x_expanded_cols,
        hard_fail=hard_fail,
    )
    eta = X @ beta

    u = get_frailty_vector_for_wide(
        wide_df,
        model,
        mode=frailty_mode,
        allow_unseen_area=allow_unseen_area,
    )
    return eta + u


def predict_cumhaz(
    a: object,
    b: object,
    *,
    times: Iterable[float],
    frailty_mode: FrailtyMode = "auto",
    allow_unseen_area: Optional[bool] = None,
    hard_fail: bool = True,
) -> np.ndarray:
    """
    Predict cumulative hazard at given times.

    Backward compatible:
      - predict_cumhaz(wide_df, model, ...)
      - predict_cumhaz(model, wide_df, ...)
    """
    wide_df, model = _coerce_args_model_wide(a, b)

    t = np.asarray(list(times), dtype=float)
    eta = predict_linear_predictor(
        wide_df,
        model,
        frailty_mode=frailty_mode,
        allow_unseen_area=allow_unseen_area,
        hard_fail=hard_fail,
    )

    breaks = list(map(float, model.breaks))
    nu = np.asarray(model.nu, dtype=float)
    H0 = _baseline_cumhaz_at_times(breaks, nu, t)  # (m,)

    return np.exp(eta)[:, None] * H0[None, :]


def predict_survival(
    a: object,
    b: object,
    *,
    times: Iterable[float],
    frailty_mode: FrailtyMode = "auto",
    allow_unseen_area: Optional[bool] = None,
    hard_fail: bool = True,
) -> np.ndarray:
    """
    Survival S(t) = exp(-cumhaz(t)).

    Backward compatible:
      - predict_survival(wide_df, model, ...)
      - predict_survival(model, wide_df, ...)
    """
    wide_df, model = _coerce_args_model_wide(a, b)

    ch = predict_cumhaz(
        wide_df,
        model,
        times=times,
        frailty_mode=frailty_mode,
        allow_unseen_area=allow_unseen_area,
        hard_fail=hard_fail,
    )
    return np.exp(-ch)


def predict_risk(
    a: object,
    b: object,
    *,
    times: Iterable[float],
    frailty_mode: FrailtyMode = "auto",
    allow_unseen_area: Optional[bool] = None,
    hard_fail: bool = True,
) -> np.ndarray:
    """
    Risk(t) = 1 - S(t).

    Backward compatible:
      - predict_risk(wide_df, model, ...)
      - predict_risk(model, wide_df, ...)
    """
    wide_df, model = _coerce_args_model_wide(a, b)

    S = predict_survival(
        wide_df,
        model,
        times=times,
        frailty_mode=frailty_mode,
        allow_unseen_area=allow_unseen_area,
        hard_fail=hard_fail,
    )
    return 1.0 - S