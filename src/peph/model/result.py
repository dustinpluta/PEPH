from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from peph.utils.json import read_json, write_json


@dataclass(frozen=True)
class FeatureEncoding:
    x_numeric: List[str]
    x_categorical: List[str]
    categorical_reference_levels: Dict[str, str]
    categorical_levels_seen: Dict[str, List[str]]
    x_expanded_cols: List[str]
    x_td_numeric: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class FittedPEPHModel:
    breaks: List[float]
    interval_convention: str
    encoding: FeatureEncoding
    baseline_col_names: List[str]
    x_col_names: List[str]
    param_names: List[str]
    params: List[float]
    cov: List[List[float]]
    nu: List[float]
    fit_backend: str
    n_train_subjects: int
    n_train_long_rows: int
    converged: Optional[bool] = None
    aic: Optional[float] = None
    deviance: Optional[float] = None
    llf: Optional[float] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: Dict[str, Any]) -> "FittedPEPHModel":
        d2 = dict(d)
        enc_dict = dict(d2["encoding"])
        enc_dict.setdefault("x_td_numeric", [])
        d2["encoding"] = FeatureEncoding(**enc_dict)
        d2.setdefault("llf", None)
        return cls(**d2)

    def save(self, path: str) -> None:
        write_json(path, self.to_json_dict())

    @classmethod
    def load(cls, path: str) -> "FittedPEPHModel":
        return cls.from_json_dict(read_json(path))

    def params_array(self) -> np.ndarray:
        return np.asarray(self.params, dtype=float)

    def cov_array(self) -> np.ndarray:
        return np.asarray(self.cov, dtype=float)

    @property
    def K(self) -> int:
        return len(self.baseline_col_names)

    @property
    def p(self) -> int:
        return len(self.x_col_names)