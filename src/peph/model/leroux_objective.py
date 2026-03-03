from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import expit
from scipy.stats import norm, beta as beta_dist

from peph.spatial.graph import SpatialGraph


@dataclass(frozen=True)
class LerouxHyperPriors:
    prior_logtau_sd: float = 10.0  # log(tau) ~ Normal(0, sd^2)
    prior_rho_a: float = 1.0       # rho ~ Beta(a,b)
    prior_rho_b: float = 1.0


def pack_theta(alpha: np.ndarray, beta: np.ndarray, u: np.ndarray, eta_tau: float, eta_rho: float) -> np.ndarray:
    return np.concatenate([alpha, beta, u, np.array([eta_tau, eta_rho], dtype=float)])


def unpack_theta(theta: np.ndarray, K: int, p: int, G: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    theta = np.asarray(theta, dtype=float)
    if theta.size != K + p + G + 2:
        raise ValueError("theta has wrong length")
    alpha = theta[:K]
    beta = theta[K : K + p]
    u = theta[K + p : K + p + G]
    eta_tau = float(theta[-2])
    eta_rho = float(theta[-1])
    return alpha, beta, u, eta_tau, eta_rho


def tau_from_eta(eta_tau: float) -> float:
    return float(np.exp(eta_tau))


def rho_from_eta(eta_rho: float, *, clip: float) -> float:
    rho = float(expit(eta_rho))
    rho = float(np.clip(rho, clip, 1.0 - clip))
    return rho


def project_center_by_component(u: np.ndarray, components: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Component-wise weighted centering:
      for each component c: sum_{g in c} w_g u_g = 0
    """
    u = np.asarray(u, dtype=float).copy()
    comp = np.asarray(components, dtype=int)
    w = np.asarray(weights, dtype=float)

    n_comp = int(comp.max() + 1) if comp.size else 0
    for c in range(n_comp):
        idx = np.where(comp == c)[0]
        if idx.size == 0:
            continue
        wc = w[idx]
        denom = float(np.sum(wc))
        if denom <= 0:
            # if no weighted mass in this component, fall back to unweighted mean
            u[idx] -= float(np.mean(u[idx]))
        else:
            mean_c = float(np.sum(wc * u[idx]) / denom)
            u[idx] -= mean_c
    return u


def _poisson_loglik_with_u(
    *,
    alpha: np.ndarray,
    beta: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    exposure: np.ndarray,
    k: np.ndarray,
    X: np.ndarray,
    area_idx: np.ndarray,
) -> float:
    """
    ll = sum(y * log_mu - mu), where
      log_mu = log(exposure) + alpha[k] + X beta + u[area_idx]
    """
    eta = alpha[k] + X @ beta + u[area_idx]
    log_mu = np.log(exposure) + eta
    mu = np.exp(log_mu)
    return float(np.sum(y * log_mu - mu))


def _sparse_logdet_spd(Q: sp.csr_matrix) -> float:
    """
    Compute log(det(Q)) using sparse LU factorization.
    Works for SPD Q; uses log|diag(U)| from LU.
    """
    lu = spla.splu(Q.tocsc())
    diagU = lu.U.diagonal()
    # For SPD, diagU should be positive; take abs for safety
    return float(np.sum(np.log(np.abs(diagU))))


def leroux_neg_log_posterior(
    theta: np.ndarray,
    *,
    K: int,
    p: int,
    graph: SpatialGraph,
    y: np.ndarray,
    exposure: np.ndarray,
    k: np.ndarray,
    X: np.ndarray,
    area_idx: np.ndarray,
    weights: np.ndarray,
    rho_clip: float,
    q_jitter: float,
    priors: LerouxHyperPriors,
) -> float:
    """
    Negative log posterior (up to additive constant) for Leroux spatial frailty MAP.

    Objective:
      - ll(alpha,beta,u) + (tau/2) u' Q(rho) u - 0.5 log|tau Q(rho)|
      + priors on eta_tau and rho (optional weak stabilizers)

    We enforce identifiability by projecting u to component-wise weighted mean zero.
    """
    G = graph.G
    alpha, beta, u_raw, eta_tau, eta_rho = unpack_theta(theta, K, p, G)

    tau = tau_from_eta(eta_tau)
    rho = rho_from_eta(eta_rho, clip=rho_clip)

    # Project u for identifiability
    u = project_center_by_component(u_raw, graph.component_ids(), weights)

    # Likelihood
    ll = _poisson_loglik_with_u(
        alpha=alpha, beta=beta, u=u, y=y, exposure=exposure, k=k, X=X, area_idx=area_idx
    )

    # Prior precision Q(rho) (with jitter)
    Q = graph.leroux_Q(rho)
    if q_jitter > 0:
        Q = Q + (q_jitter * sp.identity(G, format="csr"))

    quad = float(u @ (Q @ u))
    logdetQ = _sparse_logdet_spd(Q)

    # -log prior for u given tau,rho:
    #  (tau/2) u'Q u - 0.5 log|tau Q|
    # log|tau Q| = G log(tau) + log|Q|
    nlp_u = 0.5 * tau * quad - 0.5 * (G * np.log(tau) + logdetQ)

    # Weak hyperpriors (stabilize; keep very weak)
    # log tau ~ N(0, sd^2)
    sd = float(priors.prior_logtau_sd)
    nlp_tau = -float(norm.logpdf(eta_tau, loc=0.0, scale=sd))

    # rho ~ Beta(a,b) on (0,1); add -logpdf
    a = float(priors.prior_rho_a)
    b = float(priors.prior_rho_b)
    # Beta logpdf in terms of rho; add Jacobian from eta_rho -> rho:
    # p(eta) = p(rho) * rho(1-rho)
    logp_rho = float(beta_dist.logpdf(rho, a=a, b=b))
    log_jac = float(np.log(rho) + np.log(1.0 - rho))
    nlp_rho = -(logp_rho + log_jac)

    # Total negative log posterior (ignore constants)
    return float(-ll + nlp_u + nlp_tau + nlp_rho)