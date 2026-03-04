from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class LerouxParams:
    tau: float  # precision scaling
    rho: float  # spatial dependence in (0,1)


def leroux_precision(D: np.ndarray, W: np.ndarray, rho: float) -> np.ndarray:
    """
    Leroux precision (unscaled):
      Q(rho) = rho * (D - W) + (1-rho) * I
    where D is diagonal degree matrix, W adjacency (0/1), symmetric.
    """
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must be in (0,1)")
    G = D.shape[0]
    I = np.eye(G, dtype=float)
    return rho * (D - W) + (1.0 - rho) * I


def sample_leroux_u(
    *,
    W: np.ndarray,
    D: np.ndarray,
    tau: float,
    rho: float,
    rng: np.random.Generator,
    q_jitter: float = 1e-10,
    component_ids: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample u ~ N(0, (tau * Q(rho))^{-1}) for small G using dense linear algebra.

    If component_ids is provided, applies component-wise weighted centering:
      sum_{i in comp} w_i u_i = 0

    weights defaults to ones.
    """
    if tau <= 0:
        raise ValueError("tau must be > 0")
    G = W.shape[0]
    Q = leroux_precision(D, W, rho)
    Q = tau * Q + q_jitter * np.eye(G, dtype=float)

    # Sample using covariance = inv(Q); for small graphs this is fine.
    # Draw z ~ N(0,I), set u = L z where cov = L L^T.
    cov = np.linalg.inv(Q)
    L = np.linalg.cholesky(cov)
    z = rng.normal(size=G)
    u = L @ z

    if component_ids is not None:
        comp = np.asarray(component_ids, dtype=int)
        if weights is None:
            weights = np.ones(G, dtype=float)
        w = np.asarray(weights, dtype=float)
        for c in range(comp.max() + 1):
            idx = np.where(comp == c)[0]
            if idx.size == 0:
                continue
            denom = float(np.sum(w[idx]))
            if denom <= 0:
                m = float(np.mean(u[idx]))
            else:
                m = float(np.sum(w[idx] * u[idx]) / denom)
            u[idx] = u[idx] - m

    return u