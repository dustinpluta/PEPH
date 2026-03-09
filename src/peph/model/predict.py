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


def _has_supported_treated_td(model: FittedPEPHModel, treated_td_col: str) -> bool:
    """
    Return True iff the fitted model contains the supported TD covariate
    `treated_td_col`.

    PR-C supports only a single treated_td-style switch covariate.
    """
    enc = model.encoding
    x_td_numeric = list(getattr(enc, "x_td_numeric", []) or [])
    return treated_td_col in x_td_numeric


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

    For models with treated_td in x_td_numeric, this returns the baseline
    linear predictor excluding the time-dependent treatment effect only if the
    design matrix itself contains no TD columns. Otherwise, standard wide
    prediction is not well-defined and this function raises.

    Backward compatible:
      - predict_linear_predictor(wide_df, model, ...)
      - predict_linear_predictor(model, wide_df, ...)
    """
    wide_df, model = _coerce_args_model_wide(a, b)

    enc = model.encoding
    x_td_numeric = list(getattr(enc, "x_td_numeric", []) or [])
    if x_td_numeric:
        raise NotImplementedError(
            "predict_linear_predictor is not defined for models with time-dependent "
            f"covariates x_td_numeric={x_td_numeric}. Use predict_cumhaz / "
            "predict_survival / predict_risk with treatment_time_col for treated_td models."
        )

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
    treatment_time_col: Optional[str] = None,
    treated_td_col: str = "treated_td",
    counterfactual_mode: str = "observed",
    fixed_treatment_time: Optional[float] = None,
    delay_days: Optional[float] = None,
) -> np.ndarray:
    """
    Predict cumulative hazard at given times.

    For models with treated_td in x_td_numeric, dispatches to the TD-aware
    prediction path.

    Backward compatible:
      - predict_cumhaz(wide_df, model, ...)
      - predict_cumhaz(model, wide_df, ...)
    """
    wide_df, model = _coerce_args_model_wide(a, b)

    if _has_supported_treated_td(model, treated_td_col):
        from peph.model.predict_td import predict_cumhaz_treated_td

        tt_col = treatment_time_col or "treatment_time"
        return predict_cumhaz_treated_td(
            wide_df,
            model,
            times=times,
            treatment_time_col=tt_col,
            treated_td_col=treated_td_col,
            frailty_mode=frailty_mode,
            allow_unseen_area=allow_unseen_area,
            hard_fail=hard_fail,
            counterfactual_mode=counterfactual_mode,
            fixed_treatment_time=fixed_treatment_time,
            delay_days=delay_days,
        )

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
    treatment_time_col: Optional[str] = None,
    treated_td_col: str = "treated_td",
    counterfactual_mode: str = "observed",
    fixed_treatment_time: Optional[float] = None,
    delay_days: Optional[float] = None,
) -> np.ndarray:
    """
    Survival S(t) = exp(-cumhaz(t)).

    For models with treated_td in x_td_numeric, dispatches to the TD-aware
    prediction path.

    Backward compatible:
      - predict_survival(wide_df, model, ...)
      - predict_survival(model, wide_df, ...)
    """
    wide_df, model = _coerce_args_model_wide(a, b)

    if _has_supported_treated_td(model, treated_td_col):
        from peph.model.predict_td import predict_survival_treated_td

        tt_col = treatment_time_col or "treatment_time"
        return predict_survival_treated_td(
            wide_df,
            model,
            times=times,
            treatment_time_col=tt_col,
            treated_td_col=treated_td_col,
            frailty_mode=frailty_mode,
            allow_unseen_area=allow_unseen_area,
            hard_fail=hard_fail,
            counterfactual_mode=counterfactual_mode,
            fixed_treatment_time=fixed_treatment_time,
            delay_days=delay_days,
        )

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
    treatment_time_col: Optional[str] = None,
    treated_td_col: str = "treated_td",
    counterfactual_mode: str = "observed",
    fixed_treatment_time: Optional[float] = None,
    delay_days: Optional[float] = None,
) -> np.ndarray:
    """
    Risk(t) = 1 - S(t).

    For models with treated_td in x_td_numeric, dispatches to the TD-aware
    prediction path.

    Backward compatible:
      - predict_risk(wide_df, model, ...)
      - predict_risk(model, wide_df, ...)
    """
    wide_df, model = _coerce_args_model_wide(a, b)

    if _has_supported_treated_td(model, treated_td_col):
        from peph.model.predict_td import predict_risk_treated_td

        tt_col = treatment_time_col or "treatment_time"
        return predict_risk_treated_td(
            wide_df,
            model,
            times=times,
            treatment_time_col=tt_col,
            treated_td_col=treated_td_col,
            frailty_mode=frailty_mode,
            allow_unseen_area=allow_unseen_area,
            hard_fail=hard_fail,
            counterfactual_mode=counterfactual_mode,
            fixed_treatment_time=fixed_treatment_time,
            delay_days=delay_days,
        )

    S = predict_survival(
        wide_df,
        model,
        times=times,
        frailty_mode=frailty_mode,
        allow_unseen_area=allow_unseen_area,
        hard_fail=hard_fail,
    )
    return 1.0 - S