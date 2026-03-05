from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp

from peph.spatial.graph import SpatialGraph


@dataclass(frozen=True)
class MoransIResult:
    I: float
    expected: float
    variance: float
    z: float


def morans_I(x: np.ndarray, W: sp.csr_matrix) -> MoransIResult:
    """
    Moran's I with row/column symmetric adjacency W (0/1), zero diagonal.
    Uses standard normal approximation with variance under randomization.
    Assumes W is symmetric and nonnegative.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 3:
        raise ValueError("Need at least 3 values for Moran's I.")

    # Center
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 0:
        return MoransIResult(I=0.0, expected=-1.0/(n-1), variance=np.nan, z=np.nan)

    W = W.tocsr()
    W_sum = float(W.sum())
    if W_sum <= 0:
        return MoransIResult(I=0.0, expected=-1.0/(n-1), variance=np.nan, z=np.nan)

    num = float(xc @ (W @ xc))
    I = (n / W_sum) * (num / denom)

    # Expected under randomization
    EI = -1.0 / (n - 1)

    # Variance under randomization (Cliff & Ord style)
    # S0 = sum_ij w_ij
    S0 = W_sum

    # S1 = 1/2 sum_ij (w_ij + w_ji)^2 = sum_ij w_ij^2 for symmetric W
    # For 0/1 adjacency symmetric: w_ij^2 = w_ij
    # Keep generic:
    W_plus = W + W.T
    S1 = 0.5 * float((W_plus.multiply(W_plus)).sum())

    # S2 = sum_i (w_i+ + w_+i)^2 ; for symmetric, row sums == col sums
    rs = np.asarray(W.sum(axis=1)).ravel()
    cs = np.asarray(W.sum(axis=0)).ravel()
    S2 = float(np.sum((rs + cs) ** 2))

    # Moments of x
    x2 = xc**2
    x4 = xc**4
    b2 = float(n * np.sum(x4) / (np.sum(x2) ** 2)) if np.sum(x2) > 0 else 0.0

    # Variance under randomization:
    # Var(I) = [n*((n^2-3n+3)S1 - nS2 + 3S0^2) - b2*((n^2-n)S1 - 2nS2 + 6S0^2)] /
    #          [(n-1)(n-2)(n-3)S0^2] - EI^2
    if n <= 3:
        varI = np.nan
        z = np.nan
    else:
        num_var = (
            n * ((n**2 - 3*n + 3) * S1 - n * S2 + 3 * (S0**2))
            - b2 * ((n**2 - n) * S1 - 2*n * S2 + 6 * (S0**2))
        )
        den_var = (n - 1) * (n - 2) * (n - 3) * (S0**2)
        varI = float(num_var / den_var) - EI**2
        z = float((I - EI) / np.sqrt(varI)) if varI > 0 else np.nan

    return MoransIResult(I=float(I), expected=float(EI), variance=float(varI), z=float(z))


def graph_adjacency(graph: SpatialGraph) -> sp.csr_matrix:
    """Convenience wrapper."""
    return graph.W()