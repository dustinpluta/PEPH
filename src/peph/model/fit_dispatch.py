from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from peph.model.fit import fit_peph


def fit_model_dispatch(
    *,
    backend: str,
    long_train,
    train_wide,
    breaks: List[float],
    x_numeric: List[str],
    x_categorical: List[str],
    categorical_reference_levels: Dict[str, str],
    n_train_subjects: int,
    # PH options
    covariance: str = "classical",
    # Spatial options (Leroux)
    spatial_area_col: Optional[str] = None,
    spatial_zips_path: Optional[str] = None,
    spatial_edges_path: Optional[str] = None,
    spatial_edges_u_col: str = "zip_u",
    spatial_edges_v_col: str = "zip_v",
    allow_unseen_area: bool = False,
    # Leroux options
    leroux_max_iter: int = 200,
    leroux_ftol: float = 1e-7,
    rho_clip: float = 1e-6,
    q_jitter: float = 1e-8,
    prior_logtau_sd: float = 10.0,
    prior_rho_a: float = 1.0,
    prior_rho_b: float = 1.0,
):
    if backend == "statsmodels_glm_poisson":
        return fit_peph(
            long_train,
            breaks=breaks,
            x_numeric=x_numeric,
            x_categorical=x_categorical,
            categorical_reference_levels=categorical_reference_levels,
            n_train_subjects=n_train_subjects,
            covariance=covariance,
            cluster_col="id",
        )

    if backend == "map_leroux":
        if spatial_area_col is None or spatial_zips_path is None or spatial_edges_path is None:
            raise ValueError("Leroux backend requires spatial_area_col, spatial_zips_path, spatial_edges_path")

        # local imports to avoid import cycles at module import time
        from peph.model.fit_leroux import fit_peph_leroux
        from peph.model.leroux_objective import LerouxHyperPriors
        from peph.spatial.graph import build_graph_from_edge_list

        zips = pd.read_csv(spatial_zips_path)["zip"].astype(str).tolist()
        edges = pd.read_csv(spatial_edges_path)

        graph = build_graph_from_edge_list(
            zips=zips,
            edges_df=edges,
            col_u=spatial_edges_u_col,
            col_v=spatial_edges_v_col,
        )

        priors = LerouxHyperPriors(
            prior_logtau_sd=prior_logtau_sd,
            prior_rho_a=prior_rho_a,
            prior_rho_b=prior_rho_b,
        )

        return fit_peph_leroux(
            long_train=long_train,
            train_wide=train_wide,
            breaks=breaks,
            x_numeric=x_numeric,
            x_categorical=x_categorical,
            categorical_reference_levels=categorical_reference_levels,
            area_col=spatial_area_col,
            graph=graph,
            allow_unseen_area=allow_unseen_area,
            n_train_subjects=n_train_subjects,
            max_iter=leroux_max_iter,
            ftol=leroux_ftol,
            rho_clip=rho_clip,
            q_jitter=q_jitter,
            priors=priors,
        )

    raise ValueError(f"Unknown backend: {backend}")