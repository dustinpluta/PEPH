from __future__ import annotations

from typing import List

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
) -> FittedPEPHModel:
    """
    Fit piecewise exponential proportional hazards via Poisson GLM with offset.
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
    res = model.fit(maxiter=max_iter, tol=tol)

    params = np.asarray(res.params, dtype=float)
    cov = np.asarray(res.cov_params(), dtype=float)

    # baseline interval log hazards are first K params
    alpha = params[:K]
    nu = np.exp(alpha)

    enc = FeatureEncoding(
        x_numeric=list(x_numeric),
        x_categorical=list(x_categorical),
        categorical_reference_levels=dict(categorical_reference_levels),
        categorical_levels_seen=info.categorical_levels_seen,
        x_expanded_cols=info.x_col_names,
    )

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
        fit_backend=fit_backend,
        n_train_subjects=int(n_train_subjects),
        n_train_long_rows=int(len(long_train)),
        converged=getattr(res, "converged", None),
        aic=float(getattr(res, "aic", np.nan)) if getattr(res, "aic", None) is not None else None,
        deviance=float(getattr(res, "deviance", np.nan)) if getattr(res, "deviance", None) is not None else None,
    )
    return fitted