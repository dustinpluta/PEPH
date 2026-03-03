# src/peph/model/fit.py
from __future__ import annotations

from typing import List, Literal

import numpy as np
import statsmodels.api as sm

from peph.model.design import build_design_long_train
from peph.model.result import FeatureEncoding, FittedPEPHModel


def fit_peph(
    long_train,
    *,
    breaks: List[float],
    x_numeric: List[str],
    x_categorical: List[str],
    categorical_reference_levels: dict,
    fit_backend: str = "statsmodels_glm_poisson",
    max_iter: int = 200,
    tol: float = 1e-8,
    eps_offset: float = 1e-12,
    n_train_subjects: int,
    covariance: Literal["classical", "cluster_id"] = "classical",
    cluster_col: str = "id",
) -> FittedPEPHModel:
    """
    Fit piecewise exponential proportional hazards via Poisson GLM with offset.

    covariance:
      - "classical": model-based covariance (default)
      - "cluster_id": sandwich covariance clustered by subject id on long-form rows
    """
    K = len(breaks) - 1
    y, X, offset, info = build_design_long_train(
        long_train,
        x_numeric=x_numeric,
        x_categorical=x_categorical,
        categorical_reference_levels=categorical_reference_levels,
        K=K,
        eps_offset=eps_offset,
    )

    model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)

    if covariance == "classical":
        res = model.fit(maxiter=max_iter, tol=tol)
    elif covariance == "cluster_id":
        if cluster_col not in long_train.columns:
            raise ValueError(f"cluster_col='{cluster_col}' not found in long_train")
        groups = long_train[cluster_col].to_numpy()
        res = model.fit(
            maxiter=max_iter,
            tol=tol,
            cov_type="cluster",
            cov_kwds={"groups": groups},
        )
    else:
        raise ValueError(f"Unknown covariance option: {covariance}")

    params = np.asarray(res.params, dtype=float)
    cov = np.asarray(res.cov_params(), dtype=float)

    alpha = params[:K]
    nu = np.exp(alpha)

    enc = FeatureEncoding(
        x_numeric=list(x_numeric),
        x_categorical=list(x_categorical),
        categorical_reference_levels=dict(categorical_reference_levels),
        categorical_levels_seen=info.categorical_levels_seen,
        x_expanded_cols=info.x_col_names,
    )

    llf = float(getattr(res, "llf", np.nan))

    fitted = FittedPEPHModel(
        breaks=list(map(float, breaks)),
        interval_convention="[a,b)",
        encoding=enc,
        baseline_col_names=info.baseline_col_names,
        x_col_names=info.x_col_names,
        param_names=info.param_names,
        params=params.tolist(),
        cov=cov.tolist(),
        nu=nu.tolist(),
        fit_backend=fit_backend + f"::{covariance}",
        n_train_subjects=int(n_train_subjects),
        n_train_long_rows=int(len(long_train)),
        converged=getattr(res, "converged", None),
        aic=float(getattr(res, "aic", np.nan)) if getattr(res, "aic", None) is not None else None,
        deviance=float(getattr(res, "deviance", np.nan)) if getattr(res, "deviance", None) is not None else None,
        llf=None if np.isnan(llf) else llf,
    )
    return fitted