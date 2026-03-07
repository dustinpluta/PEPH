from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.optimize as opt
import statsmodels.api as sm

from peph.model.design import build_design_long_train
from peph.model.fit import fit_peph
from peph.model.leroux_objective import (
    LerouxHyperPriors,
    leroux_neg_log_posterior,
    pack_theta,
    project_center_by_component,
    rho_from_eta,
    tau_from_eta,
    unpack_theta,
)
from peph.model.result import FittedPEPHModel
from peph.spatial.graph import SpatialGraph, build_graph_from_edge_list
from peph.spatial.weights import zip_weights_from_train_wide


def _load_zip_universe(zips_path: str) -> List[str]:
    zdf = pd.read_csv(zips_path)
    if "zip" in zdf.columns:
        return zdf["zip"].astype(str).tolist()
    if zdf.shape[1] == 1:
        return zdf.iloc[:, 0].astype(str).tolist()
    raise ValueError(f"ZIP universe file must contain a 'zip' column or be single-column: {zips_path}")


def _build_area_idx(
    *,
    long_train: pd.DataFrame,
    area_col: str,
    zip_to_index: Dict[str, int],
    allow_unseen_area: bool,
) -> np.ndarray:
    if area_col not in long_train.columns:
        raise ValueError(f"area_col='{area_col}' not found in long_train")

    vals = long_train[area_col].astype(str).to_numpy()
    out = np.empty(len(vals), dtype=int)

    unseen = []
    for i, z in enumerate(vals):
        idx = zip_to_index.get(z)
        if idx is None:
            unseen.append(z)
            out[i] = -1
        else:
            out[i] = int(idx)

    if unseen and not allow_unseen_area:
        unseen_show = sorted(set(unseen))[:10]
        raise ValueError(
            f"Found unseen area values in long_train[{area_col!r}] not present in graph. "
            f"Examples: {unseen_show}"
        )

    if np.any(out < 0):
        raise ValueError(
            "allow_unseen_area=True is not supported for Leroux fitting because every row "
            "must map to a valid spatial frailty index."
        )

    return out


def fit_peph_leroux(
    long_train: pd.DataFrame,
    train_wide: pd.DataFrame,
    *,
    breaks: List[float],
    x_numeric: List[str],
    x_categorical: List[str],
    categorical_reference_levels: Dict[str, str],
    area_col: str,
    graph: SpatialGraph,
    allow_unseen_area: bool,
    n_train_subjects: int,
    x_td_numeric: Optional[List[str]] = None,
    max_iter: int = 200,
    ftol: float = 1e-7,
    rho_clip: float = 1e-6,
    q_jitter: float = 1e-8,
    priors: Optional[LerouxHyperPriors] = None,
) -> FittedPEPHModel:
    """
    PH initialization then Leroux MAP refinement.

    Returns a FittedPEPHModel with (alpha,beta) params/cov and attaches spatial extras.
    Inference covariance here is conditional (alpha,beta) given u via a GLM refit.
    """
    if x_td_numeric is None:
        x_td_numeric = []

    if priors is None:
        priors = LerouxHyperPriors()

    K = len(breaks) - 1
    if K <= 0:
        raise ValueError("breaks must define at least one interval")

    # 1) PH initializer
    ph = fit_peph(
        long_train=long_train,
        breaks=breaks,
        x_numeric=x_numeric,
        x_td_numeric=x_td_numeric,
        x_categorical=x_categorical,
        categorical_reference_levels=categorical_reference_levels,
        fit_backend="statsmodels_glm_poisson",
        max_iter=200,
        tol=1e-8,
        n_train_subjects=n_train_subjects,
        covariance="classical",
        cluster_col="id",
    )

    alpha0 = ph.params_array()[:K]
    beta0 = ph.params_array()[K:]
    p = beta0.size

    # 2) Long-form design / components
    y_glm, X_full, offset, dinfo = build_design_long_train(
        long_train,
        x_numeric=x_numeric,
        x_td_numeric=x_td_numeric,
        x_categorical=x_categorical,
        categorical_reference_levels=categorical_reference_levels,
        K=K,
        eps_offset=1e-12,
    )

    y = np.asarray(y_glm, dtype=float)
    exposure = long_train["exposure"].to_numpy(dtype=float)
    k_idx = long_train["k"].to_numpy(dtype=int)
    X = X_full[:, K:]  # fixed-effects only; alpha handled separately
    area_idx = _build_area_idx(
        long_train=long_train,
        area_col=area_col,
        zip_to_index=graph.zip_to_index,
        allow_unseen_area=allow_unseen_area,
    )

    # 3) weights per ZIP from training wide
    w_zip = zip_weights_from_train_wide(
        train_wide,
        area_col=area_col,
        zip_to_index=graph.zip_to_index,
        allow_unseen_area=allow_unseen_area,
    )

    G = graph.G
    u0 = np.zeros(G, dtype=float)
    eta_tau0 = 0.0
    eta_rho0 = 0.0

    theta0 = pack_theta(alpha0, beta0, u0, eta_tau0, eta_rho0)

    # 4) Optimize MAP
    pri = priors

    def obj(th: np.ndarray) -> float:
        return leroux_neg_log_posterior(
            th,
            K=K,
            p=p,
            graph=graph,
            y=y,
            exposure=exposure,
            k=k_idx,
            X=X,
            area_idx=area_idx,
            weights=w_zip,
            rho_clip=rho_clip,
            q_jitter=q_jitter,
            priors=pri,
        )

    res = opt.minimize(
        obj,
        theta0,
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": float(ftol)},
    )

    alpha_hat, beta_hat, u_raw, eta_tau_hat, eta_rho_hat = unpack_theta(res.x, K, p, G)
    tau_hat = tau_from_eta(eta_tau_hat)
    rho_hat = rho_from_eta(eta_rho_hat, clip=rho_clip)

    # enforce identifiability projection on final u
    u_hat = project_center_by_component(u_raw, graph.component_ids(), w_zip)

    # 5) Conditional refit for covariance of (alpha,beta) given u_hat
    offset2 = offset + u_hat[area_idx]

    glm = sm.GLM(y_glm, X_full, family=sm.families.Poisson(), offset=offset2)
    glm_res = glm.fit(maxiter=200, tol=1e-8)

    params_ab = np.asarray(glm_res.params, dtype=float)
    cov_ab = np.asarray(glm_res.cov_params(), dtype=float)

    alpha_final = params_ab[:K]
    nu_final = np.exp(alpha_final)

    # 6) Build fitted model object
    fitted = FittedPEPHModel(
        breaks=list(map(float, breaks)),
        interval_convention="[a,b)",
        encoding=ph.encoding,
        baseline_col_names=dinfo.baseline_col_names,
        x_col_names=dinfo.x_col_names,
        param_names=dinfo.param_names,
        params=params_ab.tolist(),
        cov=cov_ab.tolist(),
        nu=nu_final.tolist(),
        fit_backend="map_leroux",
        n_train_subjects=int(n_train_subjects),
        n_train_long_rows=int(len(long_train)),
        converged=bool(getattr(glm_res, "converged", True)),
        aic=float(getattr(glm_res, "aic", np.nan)) if getattr(glm_res, "aic", None) is not None else None,
        deviance=float(getattr(glm_res, "deviance", np.nan)) if getattr(glm_res, "deviance", None) is not None else None,
        llf=float(getattr(glm_res, "llf", np.nan)) if getattr(glm_res, "llf", None) is not None else None,
    )

    fitted.__dict__["spatial"] = {
        "type": "leroux",
        "area_col": area_col,
        "zips": graph.zips,
        "u": u_hat.tolist(),
        "tau": float(tau_hat),
        "rho": float(rho_hat),
        "optimizer": {
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "n_iter": int(getattr(res, "nit", -1)),
            "fun": float(res.fun),
        },
        "graph": {
            "G": int(graph.G),
            "n_components": int(graph.n_components()),
        },
    }

    return fitted


def fit_pe_leroux_map(
    *,
    long_train: pd.DataFrame,
    train_wide: pd.DataFrame,
    breaks: List[float],
    x_numeric: List[str],
    x_categorical: List[str],
    categorical_reference_levels: Dict[str, str],
    area_col: str,
    zips_path: str,
    edges_path: str,
    edges_u_col: str = "zip_u",
    edges_v_col: str = "zip_v",
    allow_unseen_area: bool = False,
    n_train_subjects: int,
    x_td_numeric: Optional[List[str]] = None,
    max_iter: int = 200,
    ftol: float = 1e-7,
    rho_clip: float = 1e-6,
    q_jitter: float = 1e-8,
    prior_logtau_sd: float = 10.0,
    prior_rho_a: float = 1.0,
    prior_rho_b: float = 1.0,
) -> FittedPEPHModel:
    """
    Convenience wrapper that builds the spatial graph from CSV artifacts, then fits
    the Leroux MAP model.
    """
    zips = _load_zip_universe(zips_path)
    edges_df = pd.read_csv(edges_path)

    graph = build_graph_from_edge_list(
        zips,
        edges_df,
        col_u=edges_u_col,
        col_v=edges_v_col,
    )

    priors = LerouxHyperPriors(
        prior_logtau_sd=float(prior_logtau_sd),
        prior_rho_a=float(prior_rho_a),
        prior_rho_b=float(prior_rho_b),
    )

    return fit_peph_leroux(
        long_train=long_train,
        train_wide=train_wide,
        breaks=breaks,
        x_numeric=x_numeric,
        x_td_numeric=x_td_numeric,
        x_categorical=x_categorical,
        categorical_reference_levels=categorical_reference_levels,
        area_col=area_col,
        graph=graph,
        allow_unseen_area=allow_unseen_area,
        n_train_subjects=n_train_subjects,
        max_iter=max_iter,
        ftol=ftol,
        rho_clip=rho_clip,
        q_jitter=q_jitter,
        priors=priors,
    )