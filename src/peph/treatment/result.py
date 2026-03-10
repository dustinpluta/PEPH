from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Optional


@dataclass
class TreatmentFeatureEncoding:
    """
    Encoding metadata for the treatment-time model design matrix.

    Fields
    ------
    x_numeric
        Numeric covariates included as-is.
    x_categorical
        Categorical covariates expanded to dummy columns.
    categorical_reference_levels
        Mapping from categorical column name to chosen reference level.
    categorical_levels_seen
        Levels observed during fitting, in deterministic order.
    x_expanded_cols
        Final expanded fixed-effect column names used in fitting/prediction.
    """

    x_numeric: list[str]
    x_categorical: list[str]
    categorical_reference_levels: dict[str, str]
    categorical_levels_seen: dict[str, list[str]]
    x_expanded_cols: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "TreatmentFeatureEncoding":
        return cls(
            x_numeric=list(obj["x_numeric"]),
            x_categorical=list(obj["x_categorical"]),
            categorical_reference_levels=dict(obj["categorical_reference_levels"]),
            categorical_levels_seen={
                str(k): list(v) for k, v in obj["categorical_levels_seen"].items()
            },
            x_expanded_cols=list(obj["x_expanded_cols"]),
        )


@dataclass
class FittedTreatmentAFTModel:
    """
    Fitted log-normal AFT treatment-time model.

    Model
    -----
    log(T_i) = x_i' beta + sigma * eps_i,   eps_i ~ N(0, 1)

    Parameters
    ----------
    encoding
        Design-matrix encoding metadata.
    x_col_names
        Expanded fixed-effect column names.
    param_names
        Full parameter names, typically x_col_names plus 'log_sigma'.
    params
        Full fitted parameter vector in the same order as param_names.
    cov
        Estimated covariance matrix for params.
    beta
        Fixed-effect coefficients corresponding to x_col_names.
    log_sigma
        Log residual scale parameter.
    sigma
        Residual scale parameter on log-time scale.
    fit_backend
        Fitting backend identifier, e.g. 'lognormal_aft_mle'.
    n_train_subjects
        Number of wide-format subjects used in fitting.
    converged
        Whether the optimizer reported convergence.
    loglik
        Maximized log-likelihood.
    aic
        Akaike information criterion, if available.
    """

    encoding: TreatmentFeatureEncoding
    x_col_names: list[str]
    param_names: list[str]
    params: list[float]
    cov: list[list[float]]
    beta: list[float]
    log_sigma: float
    sigma: float
    fit_backend: str
    n_train_subjects: int
    converged: bool
    loglik: Optional[float] = None
    aic: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoding": self.encoding.to_dict(),
            "x_col_names": list(self.x_col_names),
            "param_names": list(self.param_names),
            "params": list(self.params),
            "cov": [list(row) for row in self.cov],
            "beta": list(self.beta),
            "log_sigma": float(self.log_sigma),
            "sigma": float(self.sigma),
            "fit_backend": str(self.fit_backend),
            "n_train_subjects": int(self.n_train_subjects),
            "converged": bool(self.converged),
            "loglik": None if self.loglik is None else float(self.loglik),
            "aic": None if self.aic is None else float(self.aic),
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "FittedTreatmentAFTModel":
        return cls(
            encoding=TreatmentFeatureEncoding.from_dict(obj["encoding"]),
            x_col_names=list(obj["x_col_names"]),
            param_names=list(obj["param_names"]),
            params=[float(x) for x in obj["params"]],
            cov=[[float(x) for x in row] for row in obj["cov"]],
            beta=[float(x) for x in obj["beta"]],
            log_sigma=float(obj["log_sigma"]),
            sigma=float(obj["sigma"]),
            fit_backend=str(obj["fit_backend"]),
            n_train_subjects=int(obj["n_train_subjects"]),
            converged=bool(obj["converged"]),
            loglik=None if obj.get("loglik") is None else float(obj["loglik"]),
            aic=None if obj.get("aic") is None else float(obj["aic"]),
        )

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "FittedTreatmentAFTModel":
        with Path(path).open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls.from_dict(obj)