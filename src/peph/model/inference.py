from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import norm

from peph.model.result import FittedPEPHModel


def coef_table(model: FittedPEPHModel, alpha: float = 0.05) -> pd.DataFrame:
    params = model.params_array()
    cov = model.cov_array()
    se = np.sqrt(np.diag(cov))

    z = params / se
    p = 2.0 * (1.0 - norm.cdf(np.abs(z)))
    zcrit = norm.ppf(1.0 - alpha / 2.0)

    lo = params - zcrit * se
    hi = params + zcrit * se

    return pd.DataFrame(
        {
            "term": model.param_names,
            "estimate": params,
            "std_error": se,
            "z": z,
            "p_value": p,
            "ci_lower": lo,
            "ci_upper": hi,
        }
    )


def baseline_table(model: FittedPEPHModel, alpha: float = 0.05) -> pd.DataFrame:
    K = model.K
    breaks = np.asarray(model.breaks, dtype=float)
    params = model.params_array()
    cov = model.cov_array()
    se = np.sqrt(np.diag(cov))[:K]

    log_nu = params[:K]
    zcrit = norm.ppf(1.0 - alpha / 2.0)
    lo_log = log_nu - zcrit * se
    hi_log = log_nu + zcrit * se

    nu = np.exp(log_nu)
    
    # float64 exp overflow around ~709.78
    EXP_MAX = 700.0
    EXP_MIN = -745.0  # underflow to 0 is fine, but keep finite logs

    hi_nu = np.exp(np.clip(hi_log, EXP_MIN, EXP_MAX))
    lo_nu = np.exp(np.clip(lo_log, EXP_MIN, EXP_MAX))
    
    return pd.DataFrame(
        {
            "k": np.arange(K, dtype=int),
            "t0": breaks[:-1],
            "t1": breaks[1:],
            "log_nu": log_nu,
            "nu": nu,
            "nu_ci_lower": lo_nu,
            "nu_ci_upper": hi_nu,
        }
    )


def inference_summary(
    model: FittedPEPHModel,
    *,
    train_wide_time: np.ndarray,
    train_wide_event: np.ndarray,
) -> Dict[str, float]:
    n = int(train_wide_time.size)
    n_events = int(np.sum(np.asarray(train_wide_event, dtype=int) == 1))
    n_censored = int(n - n_events)

    return {
        "n_train": float(n),
        "n_events_train": float(n_events),
        "n_censored_train": float(n_censored),
        "aic": float(model.aic) if model.aic is not None else float("nan"),
        "deviance": float(model.deviance) if model.deviance is not None else float("nan"),
        "llf": float(model.llf) if model.llf is not None else float("nan"),
        "covariance": model.fit_backend.split("::")[-1] if "::" in model.fit_backend else "classical",
    }