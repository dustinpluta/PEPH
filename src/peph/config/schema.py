from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class DataConfig(BaseModel):
    path: str
    format: Literal["csv", "parquet"] = "csv"


class SchemaConfig(BaseModel):
    id_col: str
    time_col: str
    event_col: str

    x_numeric: List[str] = Field(default_factory=list)
    x_categorical: List[str] = Field(default_factory=list)
    categorical_reference_levels: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_no_overlap(self) -> "SchemaConfig":
        overlap = set(self.x_numeric).intersection(self.x_categorical)
        if overlap:
            raise ValueError(f"x_numeric and x_categorical overlap: {sorted(overlap)}")
        for c in self.x_categorical:
            if c not in self.categorical_reference_levels:
                raise ValueError(
                    f"Missing categorical reference level for '{c}'. "
                    f"Provide schema.categorical_reference_levels.{c}"
                )
        return self


class TimeConfig(BaseModel):
    scale: Literal["days"] = "days"
    breaks: List[float]
    interval_closed: Literal["left"] = "left"
    interval_open: Literal["right"] = "right"

    @field_validator("breaks")
    @classmethod
    def _validate_breaks(cls, v: List[float]) -> List[float]:
        if len(v) < 2:
            raise ValueError("breaks must have at least two entries")
        if v[0] != 0:
            raise ValueError("breaks[0] must be 0 for this package")
        # strictly increasing
        for i in range(len(v) - 1):
            if not (v[i + 1] > v[i]):
                raise ValueError("breaks must be strictly increasing")
        return v


class SplitConfig(BaseModel):
    strategy: Literal["subject_random"] = "subject_random"
    test_size: float = 0.25
    seed: int = 123

    @field_validator("test_size")
    @classmethod
    def _validate_test_size(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("test_size must be in (0,1)")
        return v


class FitConfig(BaseModel):
    backend: Literal["statsmodels_glm_poisson"] = "statsmodels_glm_poisson"
    max_iter: int = 200
    tol: float = 1e-8

    # NEW
    covariance: Literal["classical", "cluster_id"] = "classical"


class PredictConfig(BaseModel):
    horizons_days: List[float] = Field(default_factory=list)
    grid: Dict[str, Any] = Field(default_factory=dict)


class MetricsConfig(BaseModel):
    compute_on: Literal["test", "train"] = "test"
    discrimination: Dict[str, bool] = Field(default_factory=dict)
    calibration: Dict[str, bool] = Field(default_factory=dict)
    residuals: Dict[str, bool] = Field(default_factory=dict)


class PlotsConfig(BaseModel):
    calibration_risk: bool = True
    cox_snell: bool = True


class OutputConfig(BaseModel):
    root_dir: str = "models"


class RunConfig(BaseModel):
    run_name: str = "peph_run"
    data: DataConfig
    schema: SchemaConfig
    time: TimeConfig
    split: SplitConfig
    fit: FitConfig = FitConfig()
    predict: PredictConfig = PredictConfig()
    metrics: MetricsConfig = MetricsConfig()
    plots: PlotsConfig = PlotsConfig()
    output: OutputConfig = OutputConfig()

    @model_validator(mode="after")
    def _validate_horizons(self) -> "RunConfig":
        if self.predict.horizons_days:
            max_b = self.time.breaks[-1]
            bad = [t for t in self.predict.horizons_days if t <= 0 or t > max_b]
            if bad:
                raise ValueError(
                    f"predict.horizons_days must be in (0, {max_b}] but got {bad}"
                )
        return self


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping at top-level")
    return data


def _deep_set(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def apply_overrides(cfg_dict: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """
    overrides: list of strings like ["split.seed=999", "fit.max_iter=300"]
    Values are YAML-parsed, so "true"/"3.2"/"[1,2]" work.
    """
    out = dict(cfg_dict)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected key=value")
        k, v_str = item.split("=", 1)
        # parse scalar/list/dict using YAML parser
        v = yaml.safe_load(v_str)
        _deep_set(out, k, v)
    return out


def load_run_config(path: str | Path, overrides: Optional[List[str]] = None) -> RunConfig:
    d = load_yaml(path)
    if overrides:
        d = apply_overrides(d, overrides)
    return RunConfig.model_validate(d)