# src/peph/config/schema.py
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel


class SchemaConfig(BaseModel):
    """
    Defines the canonical column names and covariate lists for a dataset.
    """
    id_col: str
    time_col: str
    event_col: str

    # Explicit X-matrix columns
    x_numeric: List[str]
    x_categorical: List[str]

    # Reference levels for each categorical column (must be provided for all categoricals)
    categorical_reference_levels: Dict[str, str]


class TimeConfig(BaseModel):
    """
    Time scale is days. Breaks define a left-closed, right-open convention [a,b).
    """
    breaks: List[float]


class SpatialConfig(BaseModel):
    """
    Spatial frailty configuration (ZIP-level in your case).
    Graph is provided as a ZIP universe + undirected edge list.
    """
    # Column in wide (and propagated into long) holding area label (e.g., zip code)
    area_col: str = "zip"

    # Graph assets
    zips_path: str
    edges_path: str

    # Edge list column names
    edges_u_col: str = "zip_u"
    edges_v_col: str = "zip_v"

    # If False, unseen ZIPs (not in graph universe) hard-fail at fit/predict time
    allow_unseen_area: bool = False


class FitConfig(BaseModel):
    """
    Fitting configuration.

    backend:
      - statsmodels_glm_poisson: standard PE-PH via Poisson trick (current default)
      - map_leroux: PH init then MAP refinement with Leroux CAR frailty
    """
    backend: Literal["statsmodels_glm_poisson", "map_leroux"] = "statsmodels_glm_poisson"

    # statsmodels fit controls (used for PH init and for pure PH backend)
    max_iter: int = 200
    tol: float = 1e-8

    # covariance for PH backend (and PH init stage): classical or cluster-robust by subject id
    covariance: Literal["classical", "cluster_id"] = "classical"

    # ---- Leroux MAP controls (used only when backend == "map_leroux") ----
    leroux_max_iter: int = 200
    leroux_ftol: float = 1e-7

    # Keep rho away from 0/1 for numerical stability
    rho_clip: float = 1e-6

    # Jitter added to Q(rho) diagonal to stabilize sparse factorization
    q_jitter: float = 1e-8

    # Weak stabilizing hyperpriors (MAP)
    # log(tau) ~ Normal(0, prior_logtau_sd^2)
    prior_logtau_sd: float = 10.0

    # rho ~ Beta(prior_rho_a, prior_rho_b)
    prior_rho_a: float = 1.0
    prior_rho_b: float = 1.0


class RunConfig(BaseModel):
    """
    Top-level run configuration.
    """
    input_csv: str
    out_dir: str

    schema: SchemaConfig
    time: TimeConfig
    fit: FitConfig

    # Only required when fit.backend == "map_leroux"
    spatial: Optional[SpatialConfig] = None