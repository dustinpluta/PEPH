from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from peph.utils.json import read_json, write_json


@dataclass(frozen=True)
class FeatureEncoding:
    x_numeric: List[str]
    x_categorical: List[str]
    categorical_reference_levels: Dict[str, str]
    categorical_levels_seen: Dict[str, List[str]]  # sorted unique levels observed in training
    x_expanded_cols: List[str]  # expanded feature names (numeric + one-hot), excluding baseline interval cols


@dataclass(frozen=True)
class FittedPEPHModel:
    # time
    breaks: List[float]
    interval_convention: str  # "[a,b)"
    # covariates
    encoding: FeatureEncoding

    # parameters
    baseline_col_names: List[str]           # length K
    x_col_names: List[str]                  # expanded X cols (numeric + one-hot)
    param_names: List[str]                  # baseline + X (full)
    params: List[float]                     # length K + p
    cov: List[List[float]]                  # (K+p) x (K+p)

    # derived baseline hazards per interval
    nu: List[float]                         # exp(alpha_k)

    # fit metadata
    fit_backend: str
    n_train_subjects: int
    n_train_long_rows: int
    converged: Optional[bool] = None
    aic: Optional[float] = None
    deviance: Optional[float] = None

    def to_json_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_json_dict(cls, d: Dict[str, Any]) -> "FittedPEPHModel":
        enc = FeatureEncoding(**d["encoding"])
        d2 = dict(d)
        d2["encoding"] = enc
        return cls(**d2)

    def save(self, path: str) -> None:
        write_json(path, self.to_json_dict())

    @classmethod
    def load(cls, path: str) -> "FittedPEPHModel":
        return cls.from_json_dict(read_json(path))

    # convenient numpy views
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