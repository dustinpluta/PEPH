from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from peph.model.design import build_x_wide_for_prediction
from peph.model.frailty import FrailtyMode, get_frailty_vector_for_wide
from peph.model.predict import _baseline_cumhaz_at_times, _coerce_args_model_wide
from peph.model.result import FittedPEPHModel


def _extract_treatment_switch_times(
    wide_df: pd.DataFrame,
    *,
    treatment_time_col: str,
) -> np.ndarray:
    """
    Extract subject-specific treatment switch times from wide data.

    Missing values indicate treatment never observed during follow-up and are
    represented as +inf so the subject remains untreated for all prediction times.
    """
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


def _extract_base_eta_and_gamma(
    wide_df: pd.DataFrame,
    model: FittedPEPHModel,
    *,
    treated_td_col: str,
    frailty_mode: FrailtyMode,
    allow_unseen_area: Optional[bool],
    hard_fail: bool,
) -> tuple[np.ndarray, float]:
    """
    Build the baseline linear predictor eta0 (excluding treated_td) and extract
    the treated_td coefficient gamma from the fitted model.

    Returns
    -------
    eta0 : np.ndarray, shape (n_subjects,)
        Baseline subject-specific linear predictor excluding treated_td.
    gamma : float
        Coefficient on treated_td.
    """
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
) -> np.ndarray:
    """
    Predict cumulative hazard when a single time-dependent treatment indicator
    switches from 0 to 1 at treatment_time.

    Model:
        h(t) = h0(t) * exp(eta0 + gamma * 1[t >= treatment_time])

    Backward compatible:
      - predict_cumhaz_treated_td(wide_df, model, ...)
      - predict_cumhaz_treated_td(model, wide_df, ...)
    """
    wide_df, model = _coerce_args_model_wide(a, b)

    t = np.asarray(list(times), dtype=float)
    if t.ndim != 1:
        raise ValueError("times must be 1D")
    if np.any(t < 0.0):
        raise ValueError("times must be nonnegative")

    breaks = list(map(float, model.breaks))
    nu = np.asarray(model.nu, dtype=float)

    switch_time = _extract_treatment_switch_times(
        wide_df,
        treatment_time_col=treatment_time_col,
    )

    eta0, gamma = _extract_base_eta_and_gamma(
        wide_df,
        model,
        treated_td_col=treated_td_col,
        frailty_mode=frailty_mode,
        allow_unseen_area=allow_unseen_area,
        hard_fail=hard_fail,
    )

    # Baseline cumulative hazard at prediction horizons
    H0_t = _baseline_cumhaz_at_times(breaks, nu, t)  # (m,)

    # Baseline cumulative hazard at subject-specific treatment times
    H0_switch = np.full(wide_df.shape[0], np.inf, dtype=float)
    finite_switch = np.isfinite(switch_time)
    if np.any(finite_switch):
        H0_switch[finite_switch] = _baseline_cumhaz_at_times(
            breaks,
            nu,
            switch_time[finite_switch],
        )

    H0_t_mat = H0_t[None, :]       # (n, m)
    H0_s_mat = H0_switch[:, None]  # (n, 1)

    # Untreated contribution up to min(t, s)
    H_pre = np.minimum(H0_t_mat, H0_s_mat)

    # Treated contribution after s
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
) -> np.ndarray:
    """
    Survival under a single treatment switch:
        S(t) = exp(-H(t))

    Backward compatible:
      - predict_survival_treated_td(wide_df, model, ...)
      - predict_survival_treated_td(model, wide_df, ...)
    """
    ch = predict_cumhaz_treated_td(
        a,
        b,
        times=times,
        treatment_time_col=treatment_time_col,
        treated_td_col=treated_td_col,
        frailty_mode=frailty_mode,
        allow_unseen_area=allow_unseen_area,
        hard_fail=hard_fail,
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
) -> np.ndarray:
    """
    Risk under a single treatment switch:
        Risk(t) = 1 - S(t)

    Backward compatible:
      - predict_risk_treated_td(wide_df, model, ...)
      - predict_risk_treated_td(model, wide_df, ...)
    """
    S = predict_survival_treated_td(
        a,
        b,
        times=times,
        treatment_time_col=treatment_time_col,
        treated_td_col=treated_td_col,
        frailty_mode=frailty_mode,
        allow_unseen_area=allow_unseen_area,
        hard_fail=hard_fail,
    )
    return 1.0 - S