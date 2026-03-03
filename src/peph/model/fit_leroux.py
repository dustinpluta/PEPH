from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.optimize as opt
import statsmodels.api as sm

from peph.model.components import build_long_components
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
from peph.spatial.graph import SpatialGraph
from peph.spatial.weights import zip_weights_from_train_wide


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
    # optimizer controls
    max_iter: int = 200,
    ftol: float = 1e-7,
    rho_clip: float = 1e-6,
    q_jitter: float = 1e-8,
    priors: Optional[LerouxHyperPriors] = None,
) -> FittedPEPHModel:
    """
    PH initialization then Leroux MAP refinement.

    Returns a FittedPEPHModel with (alpha,beta) params/cov and attaches spatial extras
    as additional JSON fields via FittedPEPHModel (see note below).

    Inference covariance here is conditional (alpha,beta) given u via a GLM refit.
    """
    if priors is None:
        priors = LerouxHyperPriors()

    K = len(breaks) - 1

    # 1) PH initializer (existing production code)
    ph = fit_peph(
        long_train,
        breaks=breaks,
        x_numeric=x_numeric,
        x_categorical=x_categorical,
        categorical_reference_levels=categorical_reference_levels,
        max_iter=200,
        tol=1e-8,
        n_train_subjects=n_train_subjects,
        covariance="classical",
        cluster_col="id",
    )

    alpha0 = ph.params_array()[:K]
    beta0 = ph.params_array()[K:]
    p = beta0.size

    # 2) Build components including area_idx
    y, exposure, k_idx, X, area_idx, info = build_long_components(
        long_train,
        K=K,
        area_col=area_col,
        x_numeric=x_numeric,
        x_categorical=x_categorical,
        categorical_reference_levels=categorical_reference_levels,
        zip_to_index=graph.zip_to_index,
        allow_unseen_area=allow_unseen_area,
    )
    if area_idx is None:
        raise ValueError("area_idx is None; area_col must be provided for Leroux")

    # 3) weights per ZIP from training wide (subject counts)
    w_zip = zip_weights_from_train_wide(
        train_wide,
        area_col=area_col,
        zip_to_index=graph.zip_to_index,
        allow_unseen_area=allow_unseen_area,
    )

    G = graph.G
    u0 = np.zeros(G, dtype=float)
    eta_tau0 = 0.0  # log(1)
    eta_rho0 = 0.0  # logit(0.5)

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
    # Build full GLM design (baseline dummies + fixed) and use offset' = log(exposure) + u_hat[area]
    y_glm, X_full, offset, dinfo = build_design_long_train(
        long_train,
        x_numeric=x_numeric,
        x_categorical=x_categorical,
        categorical_reference_levels=categorical_reference_levels,
        K=K,
        eps_offset=1e-12,
    )
    offset2 = offset + u_hat[area_idx]  # add frailty into offset

    glm = sm.GLM(y_glm, X_full, family=sm.families.Poisson(), offset=offset2)
    glm_res = glm.fit(maxiter=200, tol=1e-8)

    params_ab = np.asarray(glm_res.params, dtype=float)
    cov_ab = np.asarray(glm_res.cov_params(), dtype=float)

    alpha_final = params_ab[:K]
    beta_final = params_ab[K:]
    nu_final = np.exp(alpha_final)

    # 6) Build fitted model object (params/cov correspond to alpha+beta only, as before)
    fitted = FittedPEPHModel(
        breaks=list(map(float, breaks)),
        interval_convention="[a,b)",
        encoding=ph.encoding,  # fixed-effect encoding unchanged
        baseline_col_names=dinfo.baseline_col_names,
        x_col_names=dinfo.x_col_names,
        param_names=dinfo.param_names,
        params=params_ab.tolist(),
        cov=cov_ab.tolist(),
        nu=nu_final.tolist(),
        fit_backend=f"map_leroux",
        n_train_subjects=int(n_train_subjects),
        n_train_long_rows=int(len(long_train)),
        converged=bool(getattr(glm_res, "converged", True)),
        aic=float(getattr(glm_res, "aic", np.nan)) if getattr(glm_res, "aic", None) is not None else None,
        deviance=float(getattr(glm_res, "deviance", np.nan)) if getattr(glm_res, "deviance", None) is not None else None,
        llf=float(getattr(glm_res, "llf", np.nan)) if getattr(glm_res, "llf", None) is not None else None,
    )

    # Attach spatial extras in a JSON-safe way by using a public attribute pattern
    # (If you prefer, we can add these as optional fields on the dataclass in a follow-up.)
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