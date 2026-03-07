from __future__ import annotations

from typing import List, Dict, Optional, Any

import pandas as pd

from peph.model.fit import fit_peph
from peph.model.fit_leroux import fit_pe_leroux_map


def fit_model_dispatch(
    *,
    backend: str,
    long_train: pd.DataFrame,
    train_wide: pd.DataFrame,
    breaks: List[float],
    x_numeric: List[str],
    x_categorical: List[str],
    categorical_reference_levels: Dict[str, str],
    n_train_subjects: int,
    covariance: str = "classical",
    x_td_numeric: Optional[List[str]] = None,
    spatial_area_col: Optional[str] = None,
    spatial_zips_path: Optional[str] = None,
    spatial_edges_path: Optional[str] = None,
    spatial_edges_u_col: str = "zip_u",
    spatial_edges_v_col: str = "zip_v",
    allow_unseen_area: bool = False,
    leroux_max_iter: int = 200,
    leroux_ftol: float = 1e-7,
    rho_clip: float = 1e-6,
    q_jitter: float = 1e-8,
    prior_logtau_sd: float = 10.0,
    prior_rho_a: float = 1.0,
    prior_rho_b: float = 1.0,
) -> Any:
    """
    Dispatch fitting to the selected backend.

    backend
      - "statsmodels_glm_poisson"
      - "map_leroux"
    """
    if x_td_numeric is None:
        x_td_numeric = []

    backend = str(backend)

    if backend == "statsmodels_glm_poisson":
        return fit_peph(
            long_train=long_train,
            breaks=breaks,
            x_numeric=x_numeric,
            x_td_numeric=x_td_numeric,
            x_categorical=x_categorical,
            categorical_reference_levels=categorical_reference_levels,
            fit_backend="statsmodels_glm_poisson",
            n_train_subjects=n_train_subjects,
            covariance=covariance,
        )

    if backend == "map_leroux":
        if spatial_area_col is None:
            raise ValueError("map_leroux backend requires spatial_area_col")
        if spatial_zips_path is None:
            raise ValueError("map_leroux backend requires spatial_zips_path")
        if spatial_edges_path is None:
            raise ValueError("map_leroux backend requires spatial_edges_path")

        return fit_pe_leroux_map(
            long_train=long_train,
            train_wide=train_wide,
            breaks=breaks,
            x_numeric=x_numeric,
            x_td_numeric=x_td_numeric,
            x_categorical=x_categorical,
            categorical_reference_levels=categorical_reference_levels,
            area_col=spatial_area_col,
            zips_path=spatial_zips_path,
            edges_path=spatial_edges_path,
            edges_u_col=spatial_edges_u_col,
            edges_v_col=spatial_edges_v_col,
            allow_unseen_area=allow_unseen_area,
            n_train_subjects=n_train_subjects,
            max_iter=leroux_max_iter,
            ftol=leroux_ftol,
            rho_clip=rho_clip,
            q_jitter=q_jitter,
            prior_logtau_sd=prior_logtau_sd,
            prior_rho_a=prior_rho_a,
            prior_rho_b=prior_rho_b,
        )

    raise ValueError(
        f"Unknown backend '{backend}'. "
        "Expected one of {'statsmodels_glm_poisson', 'map_leroux'}."
    )