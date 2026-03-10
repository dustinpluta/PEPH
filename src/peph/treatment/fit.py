from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import erf

from peph.treatment.design import build_x_treatment_fit
from peph.treatment.result import FittedTreatmentAFTModel


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def _lognormal_aft_negloglik_and_grad(
    theta: np.ndarray,
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Negative log-likelihood and gradient for log-normal AFT with right censoring.

    Model
    -----
    log(T_i) = x_i' beta + sigma * eps_i,  eps_i ~ N(0, 1)

    event_i = 1  => observed treatment time
    event_i = 0  => right-censored before treatment
    """
    p = X.shape[1]

    beta = theta[:p]
    log_sigma = float(theta[p])
    sigma = float(np.exp(log_sigma))

    y = np.log(time)
    mu = X @ beta
    z = (y - mu) / sigma

    # observed contribution: log f(t)
    # censored contribution: log S(t) = log(1 - Phi(z))
    obs = event == 1
    cen = ~obs

    # numerical stabilizers
    Phi = _norm_cdf(z)
    Phi = np.clip(Phi, 1e-12, 1.0 - 1e-12)

    log_pdf = (
        -np.log(time)
        - np.log(sigma)
        - 0.5 * np.log(2.0 * np.pi)
        - 0.5 * z**2
    )
    log_surv = np.log1p(-Phi)

    nll = -float(np.sum(log_pdf[obs]) + np.sum(log_surv[cen]))

    # gradient
    # standard normal pdf
    phi = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)

    grad_beta = np.zeros(p, dtype=float)
    grad_log_sigma = 0.0

    if np.any(obs):
        z_obs = z[obs]
        X_obs = X[obs]

        # d log f / d mu = z / sigma
        grad_beta += -np.sum((z_obs[:, None] / sigma) * X_obs, axis=0)

        # d log f / d log_sigma = -1 + z^2
        grad_log_sigma += -np.sum(-1.0 + z_obs**2)

    if np.any(cen):
        z_cen = z[cen]
        X_cen = X[cen]
        phi_cen = phi[cen]
        surv_cen = 1.0 - Phi[cen]
        surv_cen = np.clip(surv_cen, 1e-12, None)

        mills = phi_cen / surv_cen

        # d log S / d mu = phi(z) / (sigma * S(z))
        grad_beta += -np.sum((mills[:, None] / sigma) * X_cen, axis=0)

        # d log S / d log_sigma = z * phi(z) / S(z)
        grad_log_sigma += -np.sum(z_cen * mills)

    grad = np.concatenate([grad_beta, np.array([grad_log_sigma])])
    return nll, grad


def _initial_theta(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> np.ndarray:
    """
    Build a stable initializer from uncensored log-times if available,
    otherwise from all observed times.
    """
    p = X.shape[1]
    y = np.log(time)

    use = event == 1
    if np.sum(use) < max(5, p + 1):
        use = np.ones_like(event, dtype=bool)

    X_use = X[use]
    y_use = y[use]

    # OLS initializer
    beta0, *_ = np.linalg.lstsq(X_use, y_use, rcond=None)
    resid = y_use - X_use @ beta0
    sigma0 = float(np.std(resid, ddof=min(1, len(resid) - 1)))
    sigma0 = max(sigma0, 0.25)

    return np.concatenate([beta0, np.array([np.log(sigma0)])])


def fit_treatment_lognormal_aft(
    wide_train: pd.DataFrame,
    *,
    treatment_time_col: str,
    treatment_event_col: str,
    x_numeric: list[str],
    x_categorical: list[str],
    categorical_reference_levels: dict[str, str],
    max_iter: int = 500,
    tol: float = 1e-8,
    optimizer_method: str = "L-BFGS-B",
) -> FittedTreatmentAFTModel:
    """
    Fit a log-normal AFT model for time to treatment with right censoring.

    Parameters
    ----------
    wide_train
        One row per subject.
    treatment_time_col
        Time from diagnosis to treatment or censoring.
    treatment_event_col
        1 if treatment observed, 0 if censored before treatment.
    x_numeric, x_categorical, categorical_reference_levels
        Covariate specification.
    max_iter, tol
        Optimizer controls.
    optimizer_method
        Passed to scipy.optimize.minimize. L-BFGS-B is recommended for stability.

    Returns
    -------
    FittedTreatmentAFTModel
    """
    required = [treatment_time_col, treatment_event_col] + list(x_numeric) + list(x_categorical)
    missing = [c for c in required if c not in wide_train.columns]
    if missing:
        raise ValueError(f"Missing required columns for treatment model fit: {missing}")

    time = pd.to_numeric(wide_train[treatment_time_col], errors="raise").to_numpy(dtype=float)
    event = pd.to_numeric(wide_train[treatment_event_col], errors="raise").to_numpy(dtype=int)

    if np.any(~np.isfinite(time)):
        raise ValueError("Treatment times must be finite")
    if np.any(time <= 0.0):
        raise ValueError("Treatment times must be strictly positive for log-normal AFT")
    if np.any(~np.isin(event, [0, 1])):
        raise ValueError("treatment_event_col must contain only 0/1")

    X, encoding = build_x_treatment_fit(
        wide_train,
        x_numeric=x_numeric,
        x_categorical=x_categorical,
        categorical_reference_levels=categorical_reference_levels,
    )

    theta0 = _initial_theta(X, time, event)

    def obj(theta: np.ndarray) -> float:
        nll, _ = _lognormal_aft_negloglik_and_grad(theta, X, time, event)
        return nll

    def grad(theta: np.ndarray) -> np.ndarray:
        _, g = _lognormal_aft_negloglik_and_grad(theta, X, time, event)
        return g

    res = minimize(
        obj,
        theta0,
        jac=grad,
        method=optimizer_method,
        options={"maxiter": int(max_iter), "ftol": float(tol)},
    )

    theta_hat = np.asarray(res.x, dtype=float)
    p = X.shape[1]

    beta_hat = theta_hat[:p]
    log_sigma_hat = float(theta_hat[p])
    sigma_hat = float(np.exp(log_sigma_hat))

    # covariance from optimizer inverse Hessian approximation if available
    cov = np.full((len(theta_hat), len(theta_hat)), np.nan, dtype=float)
    hess_inv = getattr(res, "hess_inv", None)

    if hess_inv is not None:
        try:
            if hasattr(hess_inv, "todense"):
                cov = np.asarray(hess_inv.todense(), dtype=float)
            else:
                cov = np.asarray(hess_inv, dtype=float)
        except Exception:
            pass

    param_names = list(encoding.x_expanded_cols) + ["log_sigma"]
    params = np.concatenate([beta_hat, np.array([log_sigma_hat])])

    loglik = -float(res.fun)
    k_params = len(params)
    aic = 2.0 * k_params - 2.0 * loglik

    return FittedTreatmentAFTModel(
        encoding=encoding,
        x_col_names=list(encoding.x_expanded_cols),
        param_names=param_names,
        params=params.tolist(),
        cov=cov.tolist(),
        beta=beta_hat.tolist(),
        log_sigma=log_sigma_hat,
        sigma=sigma_hat,
        fit_backend="lognormal_aft_mle",
        n_train_subjects=int(len(wide_train)),
        converged=bool(res.success),
        loglik=loglik,
        aic=float(aic),
    )