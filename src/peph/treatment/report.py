from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from peph.treatment.predict import (
    predict_treatment_cdf,
    predict_treatment_mean,
    predict_treatment_median,
    predict_treatment_quantile,
)
from peph.treatment.result import FittedTreatmentAFTModel


def summarize_treatment_coefficients(
    model: FittedTreatmentAFTModel,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Summarize fitted treatment-model coefficients.

    For the log-normal AFT model:
      - coefficient is on the log-time scale
      - exp(coef) is the time ratio

    Returns
    -------
    pd.DataFrame
        Columns:
          - term
          - coef
          - se
          - z
          - p_value
          - ci_lower
          - ci_upper
          - time_ratio
          - time_ratio_ci_lower
          - time_ratio_ci_upper
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    params = np.asarray(model.params, dtype=float)
    cov = np.asarray(model.cov, dtype=float)

    if cov.shape != (len(params), len(params)):
        raise ValueError("Covariance matrix shape does not match parameter vector length")

    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    z = np.divide(params, se, out=np.full_like(params, np.nan), where=se > 0.0)
    p_value = 2.0 * (1.0 - norm.cdf(np.abs(z)))

    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    ci_lower = params - z_alpha * se
    ci_upper = params + z_alpha * se

    time_ratio = np.exp(params)
    time_ratio_ci_lower = np.exp(ci_lower)
    time_ratio_ci_upper = np.exp(ci_upper)

    return pd.DataFrame(
        {
            "term": list(model.param_names),
            "coef": params,
            "se": se,
            "z": z,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "time_ratio": time_ratio,
            "time_ratio_ci_lower": time_ratio_ci_lower,
            "time_ratio_ci_upper": time_ratio_ci_upper,
        }
    )


def summarize_treatment_model(
    model: FittedTreatmentAFTModel,
) -> dict:
    """
    Return a compact summary dictionary for a fitted treatment model.
    """
    return {
        "fit_backend": str(model.fit_backend),
        "n_train_subjects": int(model.n_train_subjects),
        "converged": bool(model.converged),
        "loglik": None if model.loglik is None else float(model.loglik),
        "aic": None if model.aic is None else float(model.aic),
        "sigma": float(model.sigma),
        "log_sigma": float(model.log_sigma),
        "n_parameters": int(len(model.param_names)),
        "n_covariates": int(len(model.x_col_names)),
        "x_col_names": list(model.x_col_names),
        "param_names": list(model.param_names),
    }


def summarize_treatment_reference_predictions(
    wide_df: pd.DataFrame,
    model: FittedTreatmentAFTModel,
    *,
    horizons: Iterable[float],
    quantiles: Optional[Iterable[float]] = None,
    hard_fail: bool = True,
) -> pd.DataFrame:
    """
    Summarize treatment-time predictions for one or more reference rows.

    Returns one row per subject with:
      - median treatment time
      - mean treatment time
      - probability treated by each requested horizon
      - optional treatment-time quantiles

    Parameters
    ----------
    wide_df
        Reference patient rows.
    horizons
        Positive times t for P(T_treat <= t).
    quantiles
        Optional quantiles p in (0,1), e.g. [0.25, 0.75].
    """
    horizons = list(horizons)
    if len(horizons) == 0:
        raise ValueError("horizons must contain at least one value")
    if any(float(t) <= 0.0 for t in horizons):
        raise ValueError("All horizons must be strictly positive")

    quantiles = [] if quantiles is None else list(quantiles)
    if any((float(p) <= 0.0) or (float(p) >= 1.0) for p in quantiles):
        raise ValueError("All quantiles must be strictly between 0 and 1")

    out = wide_df.copy()

    med = predict_treatment_median(wide_df, model, hard_fail=hard_fail)
    mean = predict_treatment_mean(wide_df, model, hard_fail=hard_fail)
    cdf = predict_treatment_cdf(wide_df, model, times=horizons, hard_fail=hard_fail)

    out["pred_treatment_median"] = med
    out["pred_treatment_mean"] = mean

    for j, t in enumerate(horizons):
        out[f"pred_prob_treated_by_{int(float(t))}"] = cdf[:, j]

    for p in quantiles:
        q = predict_treatment_quantile(wide_df, model, p=float(p), hard_fail=hard_fail)
        q_label = str(round(float(p), 4)).replace(".", "p")
        out[f"pred_treatment_quantile_{q_label}"] = q

    return out


def summarize_treatment_probability_by_horizon(
    wide_df: pd.DataFrame,
    model: FittedTreatmentAFTModel,
    *,
    horizons: Iterable[float],
    hard_fail: bool = True,
) -> pd.DataFrame:
    """
    Return a long-format summary of treatment probabilities by horizon.

    Returns
    -------
    pd.DataFrame
        Columns:
          - row_index
          - horizon_days
          - pred_prob_treated_by_horizon
    """
    horizons = list(horizons)
    if len(horizons) == 0:
        raise ValueError("horizons must contain at least one value")
    if any(float(t) <= 0.0 for t in horizons):
        raise ValueError("All horizons must be strictly positive")

    cdf = predict_treatment_cdf(wide_df, model, times=horizons, hard_fail=hard_fail)

    rows = []
    for i in range(cdf.shape[0]):
        for j, t in enumerate(horizons):
            rows.append(
                {
                    "row_index": int(i),
                    "horizon_days": float(t),
                    "pred_prob_treated_by_horizon": float(cdf[i, j]),
                }
            )

    return pd.DataFrame(rows)


def summarize_treatment_reference_pair_difference(
    wide_df_a: pd.DataFrame,
    wide_df_b: pd.DataFrame,
    model: FittedTreatmentAFTModel,
    *,
    horizons: Iterable[float],
    hard_fail: bool = True,
) -> dict:
    """
    Compare two single-row reference profiles on treatment timing predictions.

    Returns differences in:
      - median treatment time
      - mean treatment time
      - probability treated by each horizon

    Assumes each input has exactly one row.
    """
    if len(wide_df_a) != 1 or len(wide_df_b) != 1:
        raise ValueError("wide_df_a and wide_df_b must each contain exactly one row")

    horizons = list(horizons)
    if len(horizons) == 0:
        raise ValueError("horizons must contain at least one value")

    med_a = float(predict_treatment_median(wide_df_a, model, hard_fail=hard_fail)[0])
    med_b = float(predict_treatment_median(wide_df_b, model, hard_fail=hard_fail)[0])

    mean_a = float(predict_treatment_mean(wide_df_a, model, hard_fail=hard_fail)[0])
    mean_b = float(predict_treatment_mean(wide_df_b, model, hard_fail=hard_fail)[0])

    cdf_a = predict_treatment_cdf(wide_df_a, model, times=horizons, hard_fail=hard_fail)[0]
    cdf_b = predict_treatment_cdf(wide_df_b, model, times=horizons, hard_fail=hard_fail)[0]

    out = {
        "median_a": med_a,
        "median_b": med_b,
        "median_diff_b_minus_a": med_b - med_a,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff_b_minus_a": mean_b - mean_a,
        "probability_differences_b_minus_a": {},
    }

    for j, t in enumerate(horizons):
        out["probability_differences_b_minus_a"][f"by_{int(float(t))}"] = float(cdf_b[j] - cdf_a[j])

    return out