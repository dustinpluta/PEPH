# src/peph/config/schema.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel


class SchemaConfig(BaseModel):
    """
    Canonical column names + covariate lists.
    """
    id_col: str
    time_col: str
    event_col: str

    x_numeric: List[str]
    x_categorical: List[str]

    categorical_reference_levels: Dict[str, str]


class TimeConfig(BaseModel):
    """
    Time scale: days. Breaks define [a,b) intervals.
    """
    breaks: List[float]


class SpatialConfig(BaseModel):
    """
    Spatial frailty configuration (ZIP-level).
    """
    area_col: str = "zip"

    zips_path: str
    edges_path: str

    edges_u_col: str = "zip_u"
    edges_v_col: str = "zip_v"

    allow_unseen_area: bool = False


class FitConfig(BaseModel):
    """
    Fit configuration.
    """
    backend: Literal["statsmodels_glm_poisson", "map_leroux"] = "statsmodels_glm_poisson"

    # statsmodels fit controls (PH init and/or PH-only)
    max_iter: int = 200
    tol: float = 1e-8

    # covariance for PH fit stage
    covariance: Literal["classical", "cluster_id"] = "classical"

    # Leroux MAP controls (only used for backend == map_leroux)
    leroux_max_iter: int = 200
    leroux_ftol: float = 1e-7
    rho_clip: float = 1e-6
    q_jitter: float = 1e-8

    prior_logtau_sd: float = 10.0
    prior_rho_a: float = 1.0
    prior_rho_b: float = 1.0


class PredictConfig(BaseModel):
    """
    Prediction settings. Keep minimal; your pipeline can interpret these.
    """
    horizons_days: List[float] = [365.0, 730.0, 1825.0]
    frailty_mode: Literal["auto", "none", "conditional", "marginal"] = "auto"


class SplitConfig(BaseModel):
    test_size: float = 0.25
    seed: int = 0


class OutputConfig(BaseModel):
    root_dir: str = "models"


class MetricsConfig(BaseModel):
    discrimination: Dict[str, bool] = {"c_index": True, "time_dependent_auc": True}
    calibration: Dict[str, bool] = {
        "brier_score": True,
        "calibration_in_the_large": True,
        "calibration_slope": True,
    }
    residuals: Dict[str, bool] = {"cox_snell": True}


class PlotsConfig(BaseModel):
    cox_snell: bool = True
    calibration_risk: bool = True


class DataConfig(BaseModel):
    path: str
    format: Literal["csv", "parquet"] = "csv"


class RunConfig(BaseModel):
    """
    Top-level run config used by the pipeline.
    NOTE: field name 'schema' triggers a pydantic warning (shadows BaseModel.schema).
    If you want to eliminate that warning later, rename to 'data_schema' and update YAML + code.
    """
    run_name: str = "run"

    data: DataConfig
    output: OutputConfig = OutputConfig()

    schema: SchemaConfig
    time: TimeConfig
    fit: FitConfig

    # Only required when fit.backend == "map_leroux"
    spatial: Optional[SpatialConfig] = None

    split: SplitConfig = SplitConfig()
    predict: PredictConfig = PredictConfig()
    metrics: MetricsConfig = MetricsConfig()
    plots: PlotsConfig = PlotsConfig()


def _apply_overrides(cfg: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Shallow merge overrides into cfg.
    If you want dotted-key overrides later (e.g. fit.backend=map_leroux), implement here.
    """
    if not overrides:
        return cfg
    out = dict(cfg)
    for k, v in overrides.items():
        out[k] = v
    return out


def load_run_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> RunConfig:
    """
    Load YAML and parse into RunConfig.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"Empty config: {p}")

    raw = _apply_overrides(raw, overrides)
    return RunConfig(**raw)