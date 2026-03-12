from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import erf, expit

from peph.spatial.graph import build_graph_from_edge_list
from peph.treatment.design import build_x_treatment_fit
from peph.treatment.result import FittedTreatmentAFTModel, TreatmentSpatialFit


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def _lognormal_aft_negloglik_and_grad(
    theta: np.ndarray,
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Negative log-likelihood and gradient for a log-normal AFT model with
    right censoring.

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

    obs = event == 1
    cen = ~obs

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

    phi = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)

    grad_beta = np.zeros(p, dtype=float)
    grad_log_sigma = 0.0

    if np.any(obs):
        z_obs = z[obs]
        X_obs = X[obs]

        grad_beta += -np.sum((z_obs[:, None] / sigma) * X_obs, axis=0)
        grad_log_sigma += -np.sum(-1.0 + z_obs**2)

    if np.any(cen):
        z_cen = z[cen]
        X_cen = X[cen]
        phi_cen = phi[cen]
        surv_cen = 1.0 - Phi[cen]
        surv_cen = np.clip(surv_cen, 1e-12, None)

        mills = phi_cen / surv_cen

        grad_beta += -np.sum((mills[:, None] / sigma) * X_cen, axis=0)
        grad_log_sigma += -np.sum(z_cen * mills)

    grad = np.concatenate([grad_beta, np.array([grad_log_sigma])])
    return nll, grad


def _lognormal_aft_negloglik_only(
    *,
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    beta: np.ndarray,
    log_sigma: float,
    offset: Optional[np.ndarray] = None,
) -> float:
    """
    Negative log-likelihood only for a log-normal AFT model with optional offset.
    """
    sigma = float(np.exp(log_sigma))
    y = np.log(time)

    mu = X @ beta
    if offset is not None:
        mu = mu + np.asarray(offset, dtype=float)

    z = (y - mu) / sigma
    obs = event == 1
    cen = ~obs

    Phi = _norm_cdf(z)
    Phi = np.clip(Phi, 1e-12, 1.0 - 1e-12)

    log_pdf = (
        -np.log(time)
        - np.log(sigma)
        - 0.5 * np.log(2.0 * np.pi)
        - 0.5 * z**2
    )
    log_surv = np.log1p(-Phi)

    return -float(np.sum(log_pdf[obs]) + np.sum(log_surv[cen]))


def _initial_theta(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> np.ndarray:
    """
    Construct a stable initializer from uncensored log-times if possible,
    otherwise from all observed times.
    """
    p = X.shape[1]
    y = np.log(time)

    use = event == 1
    if np.sum(use) < max(5, p + 1):
        use = np.ones_like(event, dtype=bool)

    X_use = X[use]
    y_use = y[use]

    beta0, *_ = np.linalg.lstsq(X_use, y_use, rcond=None)
    resid = y_use - X_use @ beta0

    if len(resid) >= 2:
        sigma0 = float(np.std(resid, ddof=1))
    else:
        sigma0 = float(np.std(resid))

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
    Fit a nonspatial log-normal AFT model for time to treatment with right censoring.
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
        spatial=None,
    )


def _build_area_index(
    wide_train: pd.DataFrame,
    *,
    area_col: str,
    zips: list[str],
) -> np.ndarray:
    """
    Map each row of wide_train to a graph index.
    """
    zip_to_idx = {str(z): i for i, z in enumerate(zips)}
    area_vals = wide_train[area_col].astype(str).to_numpy()

    missing = sorted(set(area_vals) - set(zip_to_idx))
    if missing:
        raise ValueError(
            f"Found areas in training data not present in treatment spatial ZIP universe: "
            f"{missing[:10]}"
        )

    return np.array([zip_to_idx[str(z)] for z in area_vals], dtype=int)


def _leroux_Q_base(
    W: np.ndarray,
    D: np.ndarray,
    *,
    rho: float,
) -> np.ndarray:
    """
    Base Leroux precision matrix without tau scaling:
        Q0(rho) = rho (D - W) + (1 - rho) I
    """
    G = W.shape[0]
    return float(rho) * (D - W) + (1.0 - float(rho)) * np.eye(G, dtype=float)


def fit_treatment_lognormal_aft_map_leroux(
    wide_train: pd.DataFrame,
    *,
    treatment_time_col: str,
    treatment_event_col: str,
    x_numeric: list[str],
    x_categorical: list[str],
    categorical_reference_levels: dict[str, str],
    area_col: str,
    zips_path: str,
    edges_path: str,
    edges_u_col: str = "zip_u",
    edges_v_col: str = "zip_v",
    max_iter: int = 300,
    tol: float = 1e-8,
    optimizer_method: str = "L-BFGS-B",
    q_jitter: float = 1e-8,
    rho_clip: float = 1e-6,
    prior_logtau_sd: float = 10.0,
    prior_rho_a: float = 1.0,
    prior_rho_b: float = 1.0,
) -> FittedTreatmentAFTModel:
    """
    Fit a spatial log-normal AFT treatment model with a Leroux ZIP effect.

    Model
    -----
    log(T_i) = x_i' beta + u_zip(i) + sigma * eps_i,  eps_i ~ N(0, 1)

    u ~ N(0, Q^{-1})
    Q = tau * [rho (D - W) + (1 - rho) I] + q_jitter * I

    Estimation
    ----------
    MAP via penalized likelihood optimization over:
      - beta
      - log_sigma
      - u
      - eta_rho   where rho = logistic(eta_rho) clipped to (rho_clip, 1-rho_clip)
      - log_tau   where tau = exp(log_tau)
    """
    required = [treatment_time_col, treatment_event_col, area_col] + list(x_numeric) + list(x_categorical)
    missing = [c for c in required if c not in wide_train.columns]
    if missing:
        raise ValueError(f"Missing required columns for spatial treatment model fit: {missing}")

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

    zdf = pd.read_csv(zips_path)
    if "zip" in zdf.columns:
        zips = zdf["zip"].astype(str).tolist()
    elif zdf.shape[1] == 1:
        zips = zdf.iloc[:, 0].astype(str).tolist()
    else:
        raise ValueError("ZIP universe file must have a 'zip' column or be single-column")

    edges_df = pd.read_csv(edges_path)
    graph = build_graph_from_edge_list(
        zips=zips,
        edges_df=edges_df,
        col_u=edges_u_col,
        col_v=edges_v_col,
    )

    zips_graph = [str(z) for z in graph.zips]
    area_idx = _build_area_index(wide_train, area_col=area_col, zips=zips_graph)

    W = graph.W().toarray().astype(float)
    deg = W.sum(axis=1)
    D = np.diag(deg)
    G = W.shape[0]
    p = X.shape[1]

    theta_nonspatial = _initial_theta(X, time, event)
    beta0 = theta_nonspatial[:p]
    log_sigma0 = float(theta_nonspatial[p])

    u0 = np.zeros(G, dtype=float)
    eta_rho0 = 0.0   # rho ≈ 0.5
    log_tau0 = 0.0   # tau = 1

    theta0 = np.concatenate(
        [
            beta0,
            np.array([log_sigma0]),
            u0,
            np.array([eta_rho0, log_tau0]),
        ]
    )

    def obj(theta: np.ndarray) -> float:
        beta = np.asarray(theta[:p], dtype=float)
        log_sigma = float(theta[p])

        u = np.asarray(theta[p + 1 : p + 1 + G], dtype=float)
        eta_rho = float(theta[p + 1 + G])
        log_tau = float(theta[p + 2 + G])

        rho = float(expit(eta_rho))
        rho = min(max(rho, rho_clip), 1.0 - rho_clip)
        tau = float(np.exp(log_tau))

        offset = u[area_idx]
        nll = _lognormal_aft_negloglik_only(
            X=X,
            time=time,
            event=event,
            beta=beta,
            log_sigma=log_sigma,
            offset=offset,
        )

        Q0 = _leroux_Q_base(W, D, rho=rho)
        Q = tau * Q0 + float(q_jitter) * np.eye(G, dtype=float)

        sign, logdetQ = np.linalg.slogdet(Q)
        if sign <= 0 or not np.isfinite(logdetQ):
            return 1e30

        quad = float(u @ (Q @ u))
        nlp_u = 0.5 * quad - 0.5 * logdetQ

        # weak prior on log_tau
        nlp_logtau = 0.5 * (log_tau / float(prior_logtau_sd)) ** 2

        # beta prior on rho (up to additive constant)
        nlp_rho = 0.0
        if prior_rho_a != 1.0:
            nlp_rho -= (float(prior_rho_a) - 1.0) * np.log(rho)
        if prior_rho_b != 1.0:
            nlp_rho -= (float(prior_rho_b) - 1.0) * np.log(1.0 - rho)

        return float(nll + nlp_u + nlp_logtau + nlp_rho)

    res = minimize(
        obj,
        theta0,
        method=optimizer_method,
        options={"maxiter": int(max_iter), "ftol": float(tol)},
    )

    theta_hat = np.asarray(res.x, dtype=float)

    beta_hat = theta_hat[:p]
    log_sigma_hat = float(theta_hat[p])
    sigma_hat = float(np.exp(log_sigma_hat))

    u_hat = np.asarray(theta_hat[p + 1 : p + 1 + G], dtype=float)
    eta_rho_hat = float(theta_hat[p + 1 + G])
    log_tau_hat = float(theta_hat[p + 2 + G])

    rho_hat = float(expit(eta_rho_hat))
    rho_hat = min(max(rho_hat, rho_clip), 1.0 - rho_clip)
    tau_hat = float(np.exp(log_tau_hat))

    # Extract approximate covariance for [beta, log_sigma, rho, log_tau]
    infer_param_names = list(encoding.x_expanded_cols) + ["log_sigma", "rho", "log_tau"]
    infer_params = np.concatenate(
        [
            beta_hat,
            np.array([log_sigma_hat, rho_hat, log_tau_hat]),
        ]
    )

    cov_small = np.full((len(infer_params), len(infer_params)), np.nan, dtype=float)

    hess_inv = getattr(res, "hess_inv", None)
    if hess_inv is not None:
        try:
            if hasattr(hess_inv, "todense"):
                full_cov = np.asarray(hess_inv.todense(), dtype=float)
            else:
                full_cov = np.asarray(hess_inv, dtype=float)

            idx_keep = list(range(p + 1)) + [p + 1 + G, p + 2 + G]
            cov_small = full_cov[np.ix_(idx_keep, idx_keep)]
        except Exception:
            pass

    loglik = -float(
        _lognormal_aft_negloglik_only(
            X=X,
            time=time,
            event=event,
            beta=beta_hat,
            log_sigma=log_sigma_hat,
            offset=u_hat[area_idx],
        )
    )
    k_params = len(infer_params)
    aic = 2.0 * k_params - 2.0 * loglik

    spatial = TreatmentSpatialFit(
        type="leroux",
        area_col=str(area_col),
        zips=zips_graph,
        u=u_hat.tolist(),
        tau=float(tau_hat),
        rho=float(rho_hat),
        optimizer={
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "nfev": getattr(res, "nfev", None),
            "nit": getattr(res, "nit", None),
            "fun": float(res.fun),
        },
    )

    return FittedTreatmentAFTModel(
        encoding=encoding,
        x_col_names=list(encoding.x_expanded_cols),
        param_names=infer_param_names,
        params=infer_params.tolist(),
        cov=cov_small.tolist(),
        beta=beta_hat.tolist(),
        log_sigma=float(log_sigma_hat),
        sigma=float(sigma_hat),
        fit_backend="lognormal_aft_map_leroux",
        n_train_subjects=int(len(wide_train)),
        converged=bool(res.success),
        loglik=loglik,
        aic=float(aic),
        spatial=spatial,
    )