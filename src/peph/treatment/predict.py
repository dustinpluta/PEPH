from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.special import erf

from peph.treatment.design import build_x_treatment_prediction
from peph.treatment.result import FittedTreatmentAFTModel


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def _coerce_args_model_wide(
    a: object,
    b: object,
) -> tuple[pd.DataFrame, FittedTreatmentAFTModel]:
    """
    Backward-compatible coercion for:
      - predict_xxx(wide_df, model)
      - predict_xxx(model, wide_df)
    """
    if isinstance(a, pd.DataFrame) and isinstance(b, FittedTreatmentAFTModel):
        return a, b

    if isinstance(a, FittedTreatmentAFTModel) and isinstance(b, pd.DataFrame):
        return b, a

    raise TypeError(
        "Expected arguments (wide_df, model) or (model, wide_df), where "
        "wide_df is a pandas DataFrame and model is a FittedTreatmentAFTModel."
    )


def predict_treatment_linear_predictor(
    a: object,
    b: object,
    *,
    hard_fail: bool = True,
    return_unseen: bool = False,
) -> np.ndarray | tuple[np.ndarray, Optional[dict[str, list[str]]]]:
    """
    Predict the linear predictor on the log-time scale:

        mu_i = x_i' beta

    Parameters
    ----------
    hard_fail
        If True, unseen categorical levels raise.
        If False, unseen categorical levels are encoded as all-zero dummy columns
        for that categorical variable and returned if return_unseen=True.
    return_unseen
        If True, also return the unseen-level mapping produced by the design builder.

    Returns
    -------
    mu : np.ndarray
        Shape (n_subjects,)
    unseen : dict or None
        Only returned when return_unseen=True
    """
    wide_df, model = _coerce_args_model_wide(a, b)

    enc = model.encoding
    X, unseen = build_x_treatment_prediction(
        wide_df,
        x_numeric=enc.x_numeric,
        x_categorical=enc.x_categorical,
        categorical_reference_levels=enc.categorical_reference_levels,
        categorical_levels_seen=enc.categorical_levels_seen,
        x_col_names=model.x_col_names,
        hard_fail=hard_fail,
    )

    beta = np.asarray(model.beta, dtype=float)
    mu = X @ beta

    if return_unseen:
        return mu.astype(float), unseen
    return mu.astype(float)


def predict_treatment_logtime_mean(
    a: object,
    b: object,
    *,
    hard_fail: bool = True,
    return_unseen: bool = False,
) -> np.ndarray | tuple[np.ndarray, Optional[dict[str, list[str]]]]:
    """
    Predict E[log(T) | x] = mu.

    This is identical to the linear predictor for the log-normal AFT model.
    """
    return predict_treatment_linear_predictor(
        a,
        b,
        hard_fail=hard_fail,
        return_unseen=return_unseen,
    )


def predict_treatment_median(
    a: object,
    b: object,
    *,
    hard_fail: bool = True,
    return_unseen: bool = False,
) -> np.ndarray | tuple[np.ndarray, Optional[dict[str, list[str]]]]:
    """
    Predict the conditional median treatment time:

        median(T | x) = exp(mu)
    """
    out = predict_treatment_linear_predictor(
        a,
        b,
        hard_fail=hard_fail,
        return_unseen=return_unseen,
    )

    if return_unseen:
        mu, unseen = out
        return np.exp(mu).astype(float), unseen

    mu = out
    return np.exp(mu).astype(float)


def predict_treatment_mean(
    a: object,
    b: object,
    *,
    hard_fail: bool = True,
    return_unseen: bool = False,
) -> np.ndarray | tuple[np.ndarray, Optional[dict[str, list[str]]]]:
    """
    Predict the conditional mean treatment time under the log-normal AFT model:

        E[T | x] = exp(mu + sigma^2 / 2)
    """
    wide_df, model = _coerce_args_model_wide(a, b)
    out = predict_treatment_linear_predictor(
        wide_df,
        model,
        hard_fail=hard_fail,
        return_unseen=return_unseen,
    )

    sigma2_half = 0.5 * float(model.sigma) ** 2

    if return_unseen:
        mu, unseen = out
        return np.exp(mu + sigma2_half).astype(float), unseen

    mu = out
    return np.exp(mu + sigma2_half).astype(float)


def predict_treatment_quantile(
    a: object,
    b: object,
    *,
    p: float,
    hard_fail: bool = True,
    return_unseen: bool = False,
) -> np.ndarray | tuple[np.ndarray, Optional[dict[str, list[str]]]]:
    """
    Predict the p-th treatment-time quantile.

    For the log-normal AFT model:
        Q_p(T | x) = exp(mu + sigma * z_p)

    Notes
    -----
    This implementation uses numpy's inverse-error-function equivalent via
    interpolation from the standard normal quantile identity is not available
    directly in numpy. To avoid adding a scipy.stats dependency, this uses
    scipy.special.erf only elsewhere and imports scipy.optimize already in fit.py.
    Here we rely on scipy through numpy? No. For exactness, use scipy if present.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be strictly between 0 and 1")

    try:
        from scipy.stats import norm
    except Exception as e:  # pragma: no cover
        raise ImportError("predict_treatment_quantile requires scipy.stats.norm") from e

    z_p = float(norm.ppf(p))

    wide_df, model = _coerce_args_model_wide(a, b)
    out = predict_treatment_linear_predictor(
        wide_df,
        model,
        hard_fail=hard_fail,
        return_unseen=return_unseen,
    )

    if return_unseen:
        mu, unseen = out
        return np.exp(mu + float(model.sigma) * z_p).astype(float), unseen

    mu = out
    return np.exp(mu + float(model.sigma) * z_p).astype(float)


def predict_treatment_survival(
    a: object,
    b: object,
    *,
    times: Iterable[float],
    hard_fail: bool = True,
    return_unseen: bool = False,
) -> np.ndarray | tuple[np.ndarray, Optional[dict[str, list[str]]]]:
    """
    Predict the treatment-time survival function:

        S_T(t | x) = P(T > t | x)
                   = 1 - Phi((log t - mu) / sigma)

    Parameters
    ----------
    times
        Iterable of positive times.

    Returns
    -------
    S : np.ndarray
        Shape (n_subjects, n_times)
    unseen : dict or None
        Only returned when return_unseen=True
    """
    t = np.asarray(list(times), dtype=float)
    if t.ndim != 1:
        raise ValueError("times must be 1D")
    if np.any(~np.isfinite(t)):
        raise ValueError("times must be finite")
    if np.any(t <= 0.0):
        raise ValueError("times must be strictly positive for log-normal predictions")

    wide_df, model = _coerce_args_model_wide(a, b)
    out = predict_treatment_linear_predictor(
        wide_df,
        model,
        hard_fail=hard_fail,
        return_unseen=return_unseen,
    )

    if return_unseen:
        mu, unseen = out
    else:
        mu = out
        unseen = None

    z = (np.log(t)[None, :] - mu[:, None]) / float(model.sigma)
    S = 1.0 - _norm_cdf(z)
    S = np.clip(S, 0.0, 1.0)

    if return_unseen:
        return S.astype(float), unseen
    return S.astype(float)


def predict_treatment_cdf(
    a: object,
    b: object,
    *,
    times: Iterable[float],
    hard_fail: bool = True,
    return_unseen: bool = False,
) -> np.ndarray | tuple[np.ndarray, Optional[dict[str, list[str]]]]:
    """
    Predict the treatment-time CDF:

        F_T(t | x) = P(T <= t | x)
                   = Phi((log t - mu) / sigma)
    """
    out = predict_treatment_survival(
        a,
        b,
        times=times,
        hard_fail=hard_fail,
        return_unseen=return_unseen,
    )

    if return_unseen:
        S, unseen = out
        return (1.0 - S).astype(float), unseen

    S = out
    return (1.0 - S).astype(float)


def predict_treatment_probability_by_time(
    a: object,
    b: object,
    *,
    times: Iterable[float],
    hard_fail: bool = True,
    return_unseen: bool = False,
) -> np.ndarray | tuple[np.ndarray, Optional[dict[str, list[str]]]]:
    """
    Alias for predict_treatment_cdf(...), useful for applied reporting:

        P(T_treat <= t | x)
    """
    return predict_treatment_cdf(
        a,
        b,
        times=times,
        hard_fail=hard_fail,
        return_unseen=return_unseen,
    )