from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from peph.model.design import build_x_wide_for_prediction
from peph.model.frailty import FrailtyMode, get_frailty_vector_for_wide
from peph.model.predict import _baseline_cumhaz_at_times, _coerce_args_model_wide
from peph.model.result import FittedPEPHModel


def _extract_observed_treatment_times(
    wide_df: pd.DataFrame,
    *,
    treatment_time_col: str,
) -> np.ndarray:
    if treatment_time_col not in wide_df.columns:
        raise ValueError(
            f"treatment_time_col='{treatment_time_col}' not found in wide_df"
        )

    tt = pd.to_numeric(
        wide_df[treatment_time_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    finite = np.isfinite(tt)
    if np.any(tt[finite] < 0.0):
        raise ValueError("Observed treatment_time values must be nonnegative")

    return np.where(finite, tt, np.inf).astype(float)


def _resolve_counterfactual_treatment_times(
    wide_df: pd.DataFrame,
    *,
    treatment_time_col: str,
    counterfactual_mode: str = "observed",
    fixed_treatment_time: Optional[float] = None,
    delay_days: Optional[float] = None,
) -> np.ndarray:
    """
    Resolve subject-specific treatment times for counterfactual prediction.

    Modes
    -----
    observed
        Use observed treatment_time_col; missing => never treated (inf).
    never
        Force never treated for all subjects.
    fixed
        Force treatment at fixed_treatment_time for all subjects.
    delay_observed
        Use observed treatment times shifted later by delay_days.
        Subjects with missing observed treatment remain never treated.
    advance_observed
        Use observed treatment times shifted earlier by delay_days, floored at 0.
        Subjects with missing observed treatment remain never treated.
    """
    mode = str(counterfactual_mode)

    if mode == "observed":
        return _extract_observed_treatment_times(
            wide_df,
            treatment_time_col=treatment_time_col,
        )

    n = len(wide_df)

    if mode == "never":
        return np.full(n, np.inf, dtype=float)

    if mode == "fixed":
        if fixed_treatment_time is None:
            raise ValueError("fixed_treatment_time is required when counterfactual_mode='fixed'")
        ft = float(fixed_treatment_time)
        if ft < 0.0:
            raise ValueError("fixed_treatment_time must be nonnegative")
        return np.full(n, ft, dtype=float)

    obs = _extract_observed_treatment_times(
        wide_df,
        treatment_time_col=treatment_time_col,
    )

    if mode == "delay_observed":
        if delay_days is None:
            raise ValueError("delay_days is required when counterfactual_mode='delay_observed'")
        dd = float(delay_days)
        if dd < 0.0:
            raise ValueError("delay_days must be nonnegative")
        out = obs.copy()
        finite = np.isfinite(out)
        out[finite] = out[finite] + dd
        return out

    if mode == "advance_observed":
        if delay_days is None:
            raise ValueError("delay_days is required when counterfactual_mode='advance_observed'")
        dd = float(delay_days)
        if dd < 0.0:
            raise ValueError("delay_days must be nonnegative")
        out = obs.copy()
        finite = np.isfinite(out)
        out[finite] = np.maximum(0.0, out[finite] - dd)
        return out

    raise ValueError(
        "Unknown counterfactual_mode. "
        "Expected one of {'observed', 'never', 'fixed', 'delay_observed', 'advance_observed'}."
    )


def _extract_base_eta_and_gamma(
    wide_df: pd.DataFrame,
    model: FittedPEPHModel,
    *,
    treated_td_col: str,
    frailty_mode: FrailtyMode,
    allow_unseen_area: Optional[bool],
    hard_fail: bool,
) -> tuple[np.ndarray, float]:
    enc = model.encoding

    x_td_numeric = list(getattr(enc, "x_td_numeric", []) or [])
    if treated_td_col not in x_td_numeric:
        raise ValueError(
            f"treated_td_col='{treated_td_col}' not found in model.encoding.x_td_numeric"
        )

    x_names = list(model.x_col_names)
    if treated_td_col not in x_names:
        raise ValueError(
            f"treated_td_col='{treated_td_col}' not found in model.x_col_names"
        )

    params = np.asarray(model.params, dtype=float)
    K = len(model.baseline_col_names)
    beta = params[K:]

    j = x_names.index(treated_td_col)
    gamma = float(beta[j])

    beta_base = np.delete(beta, j)
    x_names_base = [c for c in x_names if c != treated_td_col]

    X_base, _ = build_x_wide_for_prediction(
        wide_df,
        x_numeric=enc.x_numeric,
        x_categorical=enc.x_categorical,
        categorical_reference_levels=enc.categorical_reference_levels,
        categorical_levels_seen=enc.categorical_levels_seen,
        x_col_names=x_names_base,
        hard_fail=hard_fail,
    )

    eta0 = X_base @ beta_base

    u = get_frailty_vector_for_wide(
        wide_df,
        model,
        mode=frailty_mode,
        allow_unseen_area=allow_unseen_area,
    )
    eta0 = eta0 + u

    return eta0.astype(float), gamma


def predict_cumhaz_treated_td(
    a: object,
    b: object,
    *,
    times: Iterable[float],
    treatment_time_col: str,
    treated_td_col: str = "treated_td",
    frailty_mode: FrailtyMode = "auto",
    allow_unseen_area: Optional[bool] = None,
    hard_fail: bool = True,
    counterfactual_mode: str = "observed",
    fixed_treatment_time: Optional[float] = None,
    delay_days: Optional[float] = None,
) -> np.ndarray:
    wide_df, model = _coerce_args_model_wide(a, b)

    t = np.asarray(list(times), dtype=float)
    if t.ndim != 1:
        raise ValueError("times must be 1D")
    if np.any(t < 0.0):
        raise ValueError("times must be nonnegative")

    breaks = list(map(float, model.breaks))
    nu = np.asarray(model.nu, dtype=float)

    switch_time = _resolve_counterfactual_treatment_times(
        wide_df,
        treatment_time_col=treatment_time_col,
        counterfactual_mode=counterfactual_mode,
        fixed_treatment_time=fixed_treatment_time,
        delay_days=delay_days,
    )

    eta0, gamma = _extract_base_eta_and_gamma(
        wide_df,
        model,
        treated_td_col=treated_td_col,
        frailty_mode=frailty_mode,
        allow_unseen_area=allow_unseen_area,
        hard_fail=hard_fail,
    )

    H0_t = _baseline_cumhaz_at_times(breaks, nu, t)

    H0_switch = np.full(wide_df.shape[0], np.inf, dtype=float)
    finite_switch = np.isfinite(switch_time)
    if np.any(finite_switch):
        H0_switch[finite_switch] = _baseline_cumhaz_at_times(
            breaks,
            nu,
            switch_time[finite_switch],
        )

    H0_t_mat = H0_t[None, :]
    H0_s_mat = H0_switch[:, None]

    H_pre = np.minimum(H0_t_mat, H0_s_mat)
    H_post = np.maximum(H0_t_mat - H0_s_mat, 0.0)

    ch = np.exp(eta0)[:, None] * H_pre + np.exp(eta0 + gamma)[:, None] * H_post
    return ch.astype(float)


def predict_survival_treated_td(
    a: object,
    b: object,
    *,
    times: Iterable[float],
    treatment_time_col: str,
    treated_td_col: str = "treated_td",
    frailty_mode: FrailtyMode = "auto",
    allow_unseen_area: Optional[bool] = None,
    hard_fail: bool = True,
    counterfactual_mode: str = "observed",
    fixed_treatment_time: Optional[float] = None,
    delay_days: Optional[float] = None,
) -> np.ndarray:
    ch = predict_cumhaz_treated_td(
        a,
        b,
        times=times,
        treatment_time_col=treatment_time_col,
        treated_td_col=treated_td_col,
        frailty_mode=frailty_mode,
        allow_unseen_area=allow_unseen_area,
        hard_fail=hard_fail,
        counterfactual_mode=counterfactual_mode,
        fixed_treatment_time=fixed_treatment_time,
        delay_days=delay_days,
    )
    return np.exp(-ch)


def predict_risk_treated_td(
    a: object,
    b: object,
    *,
    times: Iterable[float],
    treatment_time_col: str,
    treated_td_col: str = "treated_td",
    frailty_mode: FrailtyMode = "auto",
    allow_unseen_area: Optional[bool] = None,
    hard_fail: bool = True,
    counterfactual_mode: str = "observed",
    fixed_treatment_time: Optional[float] = None,
    delay_days: Optional[float] = None,
) -> np.ndarray:
    S = predict_survival_treated_td(
        a,
        b,
        times=times,
        treatment_time_col=treatment_time_col,
        treated_td_col=treated_td_col,
        frailty_mode=frailty_mode,
        allow_unseen_area=allow_unseen_area,
        hard_fail=hard_fail,
        counterfactual_mode=counterfactual_mode,
        fixed_treatment_time=fixed_treatment_time,
        delay_days=delay_days,
    )
    return 1.0 - S