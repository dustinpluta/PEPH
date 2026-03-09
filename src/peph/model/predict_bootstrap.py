from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd

from peph.model.predict import predict_risk, predict_survival
from peph.model.result import FittedPEPHModel


def _draw_bootstrap_models(
    model: FittedPEPHModel,
    *,
    n_boot: int,
    seed: int = 123,
) -> list[FittedPEPHModel]:
    """
    Parametric bootstrap over the fitted parameter vector using the fitted
    covariance matrix.

    This treats the fitted parameter vector as approximately multivariate normal:
        theta* ~ N(theta_hat, Cov(theta_hat))

    The baseline hazards are then reconstructed from the bootstrapped
    log-baseline coefficients.
    """
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")

    theta_hat = np.asarray(model.params, dtype=float)
    cov = np.asarray(model.cov, dtype=float)

    if cov.shape != (len(theta_hat), len(theta_hat)):
        raise ValueError("Covariance matrix shape does not match parameter vector length")

    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mean=theta_hat, cov=cov, size=n_boot)

    K = len(model.baseline_col_names)
    out: list[FittedPEPHModel] = []

    for th in draws:
        alpha = np.asarray(th[:K], dtype=float)
        nu = np.exp(alpha)

        out.append(
            replace(
                model,
                params=np.asarray(th, dtype=float).tolist(),
                nu=np.asarray(nu, dtype=float).tolist(),
            )
        )

    return out


def _summarize_bootstrap_matrix(
    mat: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, np.ndarray]:
    """
    Summarize bootstrap predictions.

    Parameters
    ----------
    mat
        Array of shape (n_boot, n_subjects, n_times)
    """
    if mat.ndim != 3:
        raise ValueError("Bootstrap matrix must have shape (n_boot, n_subjects, n_times)")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    lo = alpha / 2.0
    hi = 1.0 - alpha / 2.0

    return {
        "mean": np.mean(mat, axis=0),
        "sd": np.std(mat, axis=0, ddof=1),
        "lower": np.quantile(mat, lo, axis=0),
        "upper": np.quantile(mat, hi, axis=0),
    }


def predict_risk_bootstrap(
    wide_df: pd.DataFrame,
    model: FittedPEPHModel,
    *,
    times: list[float],
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 123,
    treatment_time_col: Optional[str] = None,
    treated_td_col: str = "treated_td",
    counterfactual_mode: str = "observed",
    fixed_treatment_time: Optional[float] = None,
    delay_days: Optional[float] = None,
    frailty_mode: str = "auto",
    allow_unseen_area: Optional[bool] = None,
    hard_fail: bool = True,
) -> dict[str, np.ndarray]:
    """
    Bootstrap confidence bands for predicted risk.

    Returns a dict with keys:
      - point
      - mean
      - sd
      - lower
      - upper

    All arrays have shape (n_subjects, n_times).
    """
    point = predict_risk(
        wide_df,
        model,
        times=times,
        treatment_time_col=treatment_time_col,
        treated_td_col=treated_td_col,
        counterfactual_mode=counterfactual_mode,
        fixed_treatment_time=fixed_treatment_time,
        delay_days=delay_days,
        frailty_mode=frailty_mode,
        allow_unseen_area=allow_unseen_area,
        hard_fail=hard_fail,
    )

    boot_models = _draw_bootstrap_models(model, n_boot=n_boot, seed=seed)

    mats = []
    for m in boot_models:
        mats.append(
            predict_risk(
                wide_df,
                m,
                times=times,
                treatment_time_col=treatment_time_col,
                treated_td_col=treated_td_col,
                counterfactual_mode=counterfactual_mode,
                fixed_treatment_time=fixed_treatment_time,
                delay_days=delay_days,
                frailty_mode=frailty_mode,
                allow_unseen_area=allow_unseen_area,
                hard_fail=hard_fail,
            )
        )

    mat = np.stack(mats, axis=0)
    out = _summarize_bootstrap_matrix(mat, alpha=alpha)
    out["point"] = point
    return out


def predict_survival_bootstrap(
    wide_df: pd.DataFrame,
    model: FittedPEPHModel,
    *,
    times: list[float],
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 123,
    treatment_time_col: Optional[str] = None,
    treated_td_col: str = "treated_td",
    counterfactual_mode: str = "observed",
    fixed_treatment_time: Optional[float] = None,
    delay_days: Optional[float] = None,
    frailty_mode: str = "auto",
    allow_unseen_area: Optional[bool] = None,
    hard_fail: bool = True,
) -> dict[str, np.ndarray]:
    """
    Bootstrap confidence bands for predicted survival.

    Returns a dict with keys:
      - point
      - mean
      - sd
      - lower
      - upper

    All arrays have shape (n_subjects, n_times).
    """
    point = predict_survival(
        wide_df,
        model,
        times=times,
        treatment_time_col=treatment_time_col,
        treated_td_col=treated_td_col,
        counterfactual_mode=counterfactual_mode,
        fixed_treatment_time=fixed_treatment_time,
        delay_days=delay_days,
        frailty_mode=frailty_mode,
        allow_unseen_area=allow_unseen_area,
        hard_fail=hard_fail,
    )

    boot_models = _draw_bootstrap_models(model, n_boot=n_boot, seed=seed)

    mats = []
    for m in boot_models:
        mats.append(
            predict_survival(
                wide_df,
                m,
                times=times,
                treatment_time_col=treatment_time_col,
                treated_td_col=treated_td_col,
                counterfactual_mode=counterfactual_mode,
                fixed_treatment_time=fixed_treatment_time,
                delay_days=delay_days,
                frailty_mode=frailty_mode,
                allow_unseen_area=allow_unseen_area,
                hard_fail=hard_fail,
            )
        )

    mat = np.stack(mats, axis=0)
    out = _summarize_bootstrap_matrix(mat, alpha=alpha)
    out["point"] = point
    return out


def predict_risk_contrast_bootstrap(
    wide_df: pd.DataFrame,
    model: FittedPEPHModel,
    *,
    times: list[float],
    scenario_a: dict,
    scenario_b: dict,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 123,
    treatment_time_col: Optional[str] = None,
    treated_td_col: str = "treated_td",
    frailty_mode: str = "auto",
    allow_unseen_area: Optional[bool] = None,
    hard_fail: bool = True,
) -> dict[str, np.ndarray]:
    """
    Bootstrap uncertainty for the risk contrast:

        contrast = risk_b - risk_a

    Parameters
    ----------
    scenario_a, scenario_b
        Dictionaries of keyword arguments forwarded to predict_risk, e.g.
        {"counterfactual_mode": "fixed", "fixed_treatment_time": 30.0}

    Returns
    -------
    dict
        Keys:
          - point
          - mean
          - sd
          - lower
          - upper

        All arrays have shape (n_subjects, n_times).
    """
    point_a = predict_risk(
        wide_df,
        model,
        times=times,
        treatment_time_col=treatment_time_col,
        treated_td_col=treated_td_col,
        frailty_mode=frailty_mode,
        allow_unseen_area=allow_unseen_area,
        hard_fail=hard_fail,
        **scenario_a,
    )
    point_b = predict_risk(
        wide_df,
        model,
        times=times,
        treatment_time_col=treatment_time_col,
        treated_td_col=treated_td_col,
        frailty_mode=frailty_mode,
        allow_unseen_area=allow_unseen_area,
        hard_fail=hard_fail,
        **scenario_b,
    )
    point = point_b - point_a

    boot_models = _draw_bootstrap_models(model, n_boot=n_boot, seed=seed)

    mats = []
    for m in boot_models:
        ra = predict_risk(
            wide_df,
            m,
            times=times,
            treatment_time_col=treatment_time_col,
            treated_td_col=treated_td_col,
            frailty_mode=frailty_mode,
            allow_unseen_area=allow_unseen_area,
            hard_fail=hard_fail,
            **scenario_a,
        )
        rb = predict_risk(
            wide_df,
            m,
            times=times,
            treatment_time_col=treatment_time_col,
            treated_td_col=treated_td_col,
            frailty_mode=frailty_mode,
            allow_unseen_area=allow_unseen_area,
            hard_fail=hard_fail,
            **scenario_b,
        )
        mats.append(rb - ra)

    mat = np.stack(mats, axis=0)
    out = _summarize_bootstrap_matrix(mat, alpha=alpha)
    out["point"] = point
    return out