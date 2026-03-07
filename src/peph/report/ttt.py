from __future__ import annotations

from math import erf, exp, sqrt
from statistics import NormalDist
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt


def _series_summary(x: pd.Series) -> Dict[str, Optional[float]]:
    """
    Numeric summary for a series, ignoring missing values.

    Returns None-valued fields if there are no non-missing observations.
    """
    x_nonmissing = pd.to_numeric(x, errors="coerce").dropna()

    if len(x_nonmissing) == 0:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
        }

    return {
        "n": int(len(x_nonmissing)),
        "mean": float(x_nonmissing.mean()),
        "sd": float(x_nonmissing.std(ddof=1)) if len(x_nonmissing) > 1 else 0.0,
        "min": float(x_nonmissing.min()),
        "q25": float(x_nonmissing.quantile(0.25)),
        "median": float(x_nonmissing.quantile(0.50)),
        "q75": float(x_nonmissing.quantile(0.75)),
        "max": float(x_nonmissing.max()),
    }


def summarize_treatment_wide(
    wide_df: pd.DataFrame,
    *,
    treatment_time_col: str,
    stage_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute wide-level treatment summaries from subject-level data.
    """
    if treatment_time_col not in wide_df.columns:
        raise ValueError(f"treatment_time_col='{treatment_time_col}' not found in wide_df")

    n_subjects = int(len(wide_df))
    tt = pd.to_numeric(wide_df[treatment_time_col], errors="coerce")
    treated_observed = tt.notna()

    out: Dict[str, Any] = {
        "n_subjects": n_subjects,
        "n_treated_observed": int(treated_observed.sum()),
        "n_treatment_unobserved": int((~treated_observed).sum()),
        "prop_treated_observed": float(treated_observed.mean()) if n_subjects > 0 else None,
        "prop_treatment_unobserved": float((~treated_observed).mean()) if n_subjects > 0 else None,
        "treatment_time_summary": _series_summary(tt),
    }

    if stage_col is not None:
        if stage_col not in wide_df.columns:
            raise ValueError(f"stage_col='{stage_col}' not found in wide_df")

        rows = []
        stage_vals = wide_df[stage_col].astype(str)

        for stage in sorted(pd.unique(stage_vals)):
            mask = stage_vals == stage
            tt_stage = tt.loc[mask]
            treated_stage = tt_stage.notna()

            row = {
                "stage": stage,
                "n_subjects": int(mask.sum()),
                "n_treated_observed": int(treated_stage.sum()),
                "n_treatment_unobserved": int((~treated_stage).sum()),
                "prop_treated_observed": float(treated_stage.mean()) if mask.sum() > 0 else None,
                "prop_treatment_unobserved": float((~treated_stage).mean()) if mask.sum() > 0 else None,
            }
            row.update(
                {
                    f"treatment_time_{k}": v
                    for k, v in _series_summary(tt_stage).items()
                }
            )
            rows.append(row)

        out["by_stage"] = rows

    return out


def summarize_treatment_long(
    long_df: pd.DataFrame,
    *,
    treated_td_col: str = "treated_td",
    exposure_col: str = "exposure",
    event_col: str = "event",
    stage_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute long-form treatment summaries from counting-process style data.
    """
    required = [treated_td_col, exposure_col, event_col]
    missing = [c for c in required if c not in long_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in long_df: {missing}")

    treated = pd.to_numeric(long_df[treated_td_col], errors="raise").astype(int)
    exposure = pd.to_numeric(long_df[exposure_col], errors="raise").astype(float)
    event = pd.to_numeric(long_df[event_col], errors="raise").astype(int)

    invalid_treated = sorted(set(treated.unique()) - {0, 1})
    if invalid_treated:
        raise ValueError(f"{treated_td_col} must contain only 0/1 values, got {invalid_treated}")

    if np.any(exposure < 0.0):
        raise ValueError(f"{exposure_col} must be nonnegative")

    if np.any((event != 0) & (event != 1)):
        raise ValueError(f"{event_col} must contain only 0/1 values")

    def _one_group(mask: np.ndarray) -> Dict[str, Any]:
        n_rows = int(mask.sum())
        exp_sum = float(exposure.loc[mask].sum())
        ev_sum = int(event.loc[mask].sum())
        rate = float(ev_sum / exp_sum) if exp_sum > 0.0 else None
        return {
            "n_rows": n_rows,
            "person_time": exp_sum,
            "events": ev_sum,
            "event_rate_per_time": rate,
        }

    untreated_mask = treated.to_numpy(dtype=int) == 0
    treated_mask = treated.to_numpy(dtype=int) == 1

    untreated = _one_group(untreated_mask)
    treated_grp = _one_group(treated_mask)

    total_rows = int(len(long_df))
    total_person_time = float(exposure.sum())
    total_events = int(event.sum())

    out: Dict[str, Any] = {
        "n_rows": total_rows,
        "person_time_total": total_person_time,
        "events_total": total_events,
        "n_rows_untreated": untreated["n_rows"],
        "n_rows_treated": treated_grp["n_rows"],
        "prop_rows_untreated": float(untreated["n_rows"] / total_rows) if total_rows > 0 else None,
        "prop_rows_treated": float(treated_grp["n_rows"] / total_rows) if total_rows > 0 else None,
        "person_time_untreated": untreated["person_time"],
        "person_time_treated": treated_grp["person_time"],
        "prop_person_time_untreated": (
            float(untreated["person_time"] / total_person_time) if total_person_time > 0.0 else None
        ),
        "prop_person_time_treated": (
            float(treated_grp["person_time"] / total_person_time) if total_person_time > 0.0 else None
        ),
        "events_untreated": untreated["events"],
        "events_treated": treated_grp["events"],
        "prop_events_untreated": float(untreated["events"] / total_events) if total_events > 0 else None,
        "prop_events_treated": float(treated_grp["events"] / total_events) if total_events > 0 else None,
        "event_rate_per_time_untreated": untreated["event_rate_per_time"],
        "event_rate_per_time_treated": treated_grp["event_rate_per_time"],
    }

    if stage_col is not None:
        if stage_col not in long_df.columns:
            raise ValueError(f"stage_col='{stage_col}' not found in long_df")

        rows = []
        stage_vals = long_df[stage_col].astype(str)

        for stage in sorted(pd.unique(stage_vals)):
            mask_stage = stage_vals == stage

            untreated_stage = _one_group(mask_stage.to_numpy() & untreated_mask)
            treated_stage = _one_group(mask_stage.to_numpy() & treated_mask)

            stage_rows = int(mask_stage.sum())
            stage_pt = float(exposure.loc[mask_stage].sum())
            stage_events = int(event.loc[mask_stage].sum())

            rows.append(
                {
                    "stage": stage,
                    "n_rows": stage_rows,
                    "person_time_total": stage_pt,
                    "events_total": stage_events,
                    "n_rows_untreated": untreated_stage["n_rows"],
                    "n_rows_treated": treated_stage["n_rows"],
                    "person_time_untreated": untreated_stage["person_time"],
                    "person_time_treated": treated_stage["person_time"],
                    "events_untreated": untreated_stage["events"],
                    "events_treated": treated_stage["events"],
                    "event_rate_per_time_untreated": untreated_stage["event_rate_per_time"],
                    "event_rate_per_time_treated": treated_stage["event_rate_per_time"],
                }
            )

        out["by_stage"] = rows

    return out


def summarize_treated_td_effect(
    fitted,
    *,
    treated_td_col: str = "treated_td",
    alpha: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """
    Extract the treated_td coefficient summary from a fitted model.

    Parameters
    ----------
    fitted
        FittedPEPHModel-like object with:
          - param_names
          - params
          - cov
    treated_td_col
        Name of the time-dependent treatment coefficient to extract.
    alpha
        Two-sided significance level for Wald confidence intervals.

    Returns
    -------
    dict or None
        Returns None if treated_td_col is not present in fitted.param_names.

        Otherwise returns a JSON-serializable dict with:
          - coefficient
          - hazard_ratio
          - se
          - z
          - p_value
          - ci_lower / ci_upper
          - hr_ci_lower / hr_ci_upper
          - alpha
    """
    param_names = list(getattr(fitted, "param_names", []))
    if treated_td_col not in param_names:
        return None

    params = np.asarray(getattr(fitted, "params"), dtype=float)
    cov = np.asarray(getattr(fitted, "cov"), dtype=float)

    j = param_names.index(treated_td_col)
    coef = float(params[j])

    if cov.ndim != 2 or cov.shape[0] <= j or cov.shape[1] <= j:
        raise ValueError("Fitted covariance matrix has incompatible shape")

    var = float(cov[j, j])
    if var < 0.0:
        raise ValueError(f"Negative variance encountered for '{treated_td_col}': {var}")

    se = float(np.sqrt(var))
    hr = float(np.exp(coef))

    if se == 0.0:
        z_stat = None
        p_value = None
        ci_lower = coef
        ci_upper = coef
    else:
        z_stat = float(coef / se)
        # two-sided normal-approx p-value
        p_value = float(2.0 * (1.0 - NormalDist().cdf(abs(z_stat))))
        z_crit = float(NormalDist().inv_cdf(1.0 - alpha / 2.0))
        ci_lower = float(coef - z_crit * se)
        ci_upper = float(coef + z_crit * se)

    return {
        "term": treated_td_col,
        "coefficient": coef,
        "hazard_ratio": hr,
        "se": se,
        "z": z_stat,
        "p_value": p_value,
        "alpha": float(alpha),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "hr_ci_lower": float(np.exp(ci_lower)),
        "hr_ci_upper": float(np.exp(ci_upper)),
    }

def summarize_treatment_time_distribution(
    wide_df: pd.DataFrame,
    *,
    treatment_time_col: str,
    bins: Optional[list[float]] = None,
    include_overflow_bin: bool = True,
) -> pd.DataFrame:
    """
    Build a treatment-time distribution table from wide subject-level data.

    Parameters
    ----------
    wide_df
        One row per subject.
    treatment_time_col
        Column containing observed treatment time; missing means treatment not
        observed during follow-up.
    bins
        Increasing bin edges in days. Default:
            [0, 30, 90, 180, 365]
        producing bins:
            [0,30), [30,90), [90,180), [180,365)
        and optionally an overflow bin [365, inf).
    include_overflow_bin
        If True, append a final [last_bin, inf) bin.

    Returns
    -------
    pd.DataFrame
        Columns:
          - bin
          - bin_left
          - bin_right
          - n_subjects
          - prop_subjects

    Notes
    -----
    - Only subjects with observed treatment_time are included in the distribution.
    - prop_subjects is relative to the number of observed treatment times, not all subjects.
    """
    if treatment_time_col not in wide_df.columns:
        raise ValueError(f"treatment_time_col='{treatment_time_col}' not found in wide_df")

    tt = pd.to_numeric(wide_df[treatment_time_col], errors="coerce").dropna()

    if bins is None:
        bins = [0.0, 30.0, 90.0, 180.0, 365.0]

    if len(bins) < 2:
        raise ValueError("bins must have length at least 2")
    if any(float(bins[i + 1]) <= float(bins[i]) for i in range(len(bins) - 1)):
        raise ValueError("bins must be strictly increasing")

    edges = list(map(float, bins))
    if include_overflow_bin:
        edges = edges + [np.inf]

    rows = []
    n_obs = int(len(tt))

    for i in range(len(edges) - 1):
        left = float(edges[i])
        right = float(edges[i + 1])

        if np.isinf(right):
            mask = tt >= left
            label = f"[{int(left)}, inf)"
        else:
            mask = (tt >= left) & (tt < right)
            label = f"[{int(left)}, {int(right)})"

        n_bin = int(mask.sum())
        rows.append(
            {
                "bin": label,
                "bin_left": left,
                "bin_right": None if np.isinf(right) else right,
                "n_subjects": n_bin,
                "prop_subjects": (float(n_bin / n_obs) if n_obs > 0 else None),
            }
        )

    return pd.DataFrame(rows)

def plot_treatment_time_histogram(
    wide_df: pd.DataFrame,
    *,
    treatment_time_col: str,
    out_path: Path,
    bins: Optional[list[float]] = None,
) -> None:
    """
    Plot histogram of observed treatment times.

    Parameters
    ----------
    wide_df
        One row per subject.
    treatment_time_col
        Column containing observed treatment time (NaN = not treated).
    out_path
        Output PNG path.
    bins
        Histogram bin edges in days. Default:
        [0, 30, 90, 180, 365, 730]
    """

    if treatment_time_col not in wide_df.columns:
        raise ValueError(
            f"treatment_time_col='{treatment_time_col}' not found in wide_df"
        )

    tt = pd.to_numeric(wide_df[treatment_time_col], errors="coerce").dropna()

    if bins is None:
        bins = [0, 30, 90, 180, 365, 730]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(tt, bins=bins)

    ax.set_xlabel("Time to treatment (days)")
    ax.set_ylabel("Number of subjects")
    ax.set_title("Distribution of treatment timing")

    ax.set_xlim(left=0)

    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)