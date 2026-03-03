from __future__ import annotations

import numpy as np


def ph_loglik_poisson_trick(
    *,
    alpha: np.ndarray,         # shape (K,)
    beta: np.ndarray,          # shape (p,)
    y: np.ndarray,             # shape (n,)
    exposure: np.ndarray,      # shape (n,)
    k: np.ndarray,             # shape (n,) in 0..K-1
    X: np.ndarray,             # shape (n,p)
) -> float:
    """
    Poisson trick log-likelihood for PE-PH (ignoring log(y!) constant).
    Uses mu = exposure * exp(alpha[k] + X beta).

    ll = sum( y * log(mu) - mu )

    with log(mu) = log(exposure) + alpha[k] + X beta
    """
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    y = np.asarray(y, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    k = np.asarray(k, dtype=int)
    X = np.asarray(X, dtype=float)

    if exposure.ndim != 1 or np.any(exposure <= 0):
        raise ValueError("exposure must be positive and 1D")
    if y.shape[0] != exposure.shape[0] or y.shape[0] != k.shape[0] or y.shape[0] != X.shape[0]:
        raise ValueError("Input arrays have inconsistent lengths")

    eta = alpha[k] + X @ beta
    log_mu = np.log(exposure) + eta
    mu = np.exp(log_mu)
    return float(np.sum(y * log_mu - mu))