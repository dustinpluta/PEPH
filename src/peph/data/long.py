from __future__ import annotations

from math import isclose
from typing import List

import numpy as np
import pandas as pd


def _interval_index_left_closed_right_open(breaks: np.ndarray, t: float) -> int:
    """
    Return k such that t in [breaks[k], breaks[k+1]) under left-closed right-open convention.

    Notes
    -----
    - Uses searchsorted(side="right") so that t exactly equal to a break is assigned
      to the interval that starts at that break.
    - If t == breaks[-1], there is no valid interval in [a,b) (outside the window).
    """
    k = int(np.searchsorted(breaks, t, side="right") - 1)
    return k


def _coerce_subject_cut_times(value: object) -> List[float]:
    """
    Normalize a row-level cut-time field into a list[float].

    Allowed forms
    -------------
    - scalar number
    - list / tuple / np.ndarray / pd.Series of numbers
    - missing / None / NaN -> []

    Strings are intentionally not parsed.
    """
    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        out: List[float] = []
        for v in value:
            if v is None:
                continue
            try:
                if pd.isna(v):
                    continue
            except Exception:
                pass
            out.append(float(v))
        return out

    if np.isscalar(value):
        return [float(value)]

    raise ValueError(
        "cut_times_col values must be numeric, list-like of numerics, or missing"
    )


def _merged_subject_breaks(
    *,
    global_breaks: np.ndarray,
    t_obs: float,
    subject_cut_times: List[float],
) -> np.ndarray:
    """
    Merge global PE breaks with subject-specific cut times.

    Keep only cut times strictly inside (0, t_obs).
    """
    extras: List[float] = []
    for c in subject_cut_times:
        if c < 0:
            raise ValueError(f"Subject cut time must be nonnegative, got {c}")
        if 0.0 < c < t_obs:
            extras.append(float(c))

    merged = np.concatenate([global_breaks, np.asarray(extras, dtype=float)])
    merged = np.unique(merged)
    merged.sort()
    return merged


def _treated_td_value(
    *,
    t0: float,
    treatment_time: float | None,
) -> int:
    """
    Time-dependent treatment indicator for a long row.

    Convention
    ----------
    - treated_td = 1 iff treatment has occurred by the start of the interval/subinterval
    - i.e. treated_td = 1[t0 >= treatment_time]
    - missing treatment_time => 0
    """
    if treatment_time is None:
        return 0
    return int(t0 >= treatment_time)


def expand_long(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    event_col: str,
    x_cols: List[str],
    breaks: List[float],
    cut_times_col: str | None = None,
    td_treatment_col: str | None = None,
    treated_td_col: str = "treated_td",
) -> pd.DataFrame:
    """
    Expand wide right-censored survival data into piecewise exponential long form.

    Conventions
    -----------
    - breaks define intervals [breaks[k], breaks[k+1]) in days.
    - event time exactly equal to a break is assigned to the interval that starts at that break.
    - follow-up is truncated at breaks[-1] (administrative censoring window).
    - If an event occurs exactly at an interval start (a break or subject cut), we emit a row
      in that interval with exposure=0 and event=1. This preserves the boundary convention.
    - If cut_times_col is provided, each subject's intervals are additionally split at the
      subject-specific cut time(s).
    - k is always the GLOBAL PE interval index, not a subject-specific subinterval index.
    - If td_treatment_col is provided, a long-form time-dependent treatment indicator is added:
          treated_td = 1[t0 >= treatment_time]
      with missing treatment_time treated as untreated throughout observed follow-up.

    Parameters
    ----------
    df : pd.DataFrame
        Wide survival dataset (one row per subject) with right censoring.
    id_col, time_col, event_col : str
        Column names for subject id, observed time (days), and event indicator (0/1).
    x_cols : list[str]
        Covariate columns to carry through.
    breaks : list[float]
        Increasing breakpoints in days, starting at 0. Intervals are [a,b).
    cut_times_col : str | None
        Optional column whose row-level value is either:
        - a scalar cut time
        - a list-like of cut times
        - missing / null
        Cut times are used only to split intervals.
    td_treatment_col : str | None
        Optional wide-data column containing time to treatment. If provided, add a
        long-form treatment indicator named treated_td_col.
    treated_td_col : str
        Output long-form column name for the treatment indicator.

    Returns
    -------
    pd.DataFrame
        Long-form dataset with columns:
        id, k, t0, t1, exposure, event, plus x_cols, and optionally treated_td_col.
    """
    b = np.asarray(breaks, dtype=float)
    if b.ndim != 1 or b.size < 2:
        raise ValueError("breaks must be a 1D array with length>=2")
    if b[0] != 0 or np.any(np.diff(b) <= 0):
        raise ValueError("breaks must be strictly increasing and start at 0")

    required = [id_col, time_col, event_col] + list(x_cols)
    if cut_times_col is not None:
        required.append(cut_times_col)
    if td_treatment_col is not None and td_treatment_col not in required:
        required.append(td_treatment_col)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if td_treatment_col is not None and treated_td_col in df.columns and treated_td_col not in x_cols:
        raise ValueError(
            f"treated_td_col='{treated_td_col}' already exists in input df; choose a different output name"
        )

    out_rows = []
    tmax = float(b[-1])
    K = b.size - 1

    for row in df.itertuples(index=False):
        rid = getattr(row, id_col)
        t = float(getattr(row, time_col))
        e = int(getattr(row, event_col))

        if t < 0:
            raise ValueError(f"Negative time for id={rid}: {t}")
        if e not in (0, 1):
            raise ValueError(f"event must be 0/1 for id={rid}, got {e}")

        t_obs = min(t, tmax)
        e_obs = 1 if (e == 1 and t <= tmax) else 0

        if t_obs == 0:
            continue

        k_event = None
        if e_obs == 1:
            if t_obs == tmax:
                e_obs = 0
            else:
                k_event = _interval_index_left_closed_right_open(b, t_obs)
                if not (0 <= k_event < K):
                    raise RuntimeError(f"Event interval out of range for id={rid}, t={t_obs}")

        subject_cut_times: List[float] = []
        if cut_times_col is not None:
            subject_cut_times = _coerce_subject_cut_times(getattr(row, cut_times_col))

        treatment_time_value: float | None = None
        if td_treatment_col is not None:
            raw_tt = getattr(row, td_treatment_col)
            if raw_tt is None:
                treatment_time_value = None
            else:
                try:
                    if pd.isna(raw_tt):
                        treatment_time_value = None
                    else:
                        treatment_time_value = float(raw_tt)
                except Exception:
                    treatment_time_value = float(raw_tt)

                if treatment_time_value is not None and treatment_time_value < 0:
                    raise ValueError(
                        f"Treatment time must be nonnegative for id={rid}, got {treatment_time_value}"
                    )

        row_breaks = _merged_subject_breaks(
            global_breaks=b,
            t_obs=t_obs,
            subject_cut_times=subject_cut_times,
        )

        for j in range(len(row_breaks) - 1):
            t0 = float(row_breaks[j])
            t1_global = float(row_breaks[j + 1])

            if t_obs < t0:
                break

            k = _interval_index_left_closed_right_open(b, t0)
            if not (0 <= k < K):
                continue

            if e_obs == 1 and k_event is not None and k > k_event:
                break

            treated_td = None
            if td_treatment_col is not None:
                treated_td = _treated_td_value(
                    t0=t0,
                    treatment_time=treatment_time_value,
                )

            if e_obs == 1 and k_event is not None and k == k_event:
                if isclose(t_obs, t0):
                    rec = {
                        "id": rid,
                        "k": k,
                        "t0": t0,
                        "t1": t1_global,
                        "exposure": 0.0,
                        "event": 1,
                    }
                    for c in x_cols:
                        rec[c] = getattr(row, c)
                    if td_treatment_col is not None:
                        rec[treated_td_col] = int(treated_td)
                    out_rows.append(rec)
                    break

                if t0 < t_obs < t1_global:
                    rec = {
                        "id": rid,
                        "k": k,
                        "t0": t0,
                        "t1": t_obs,
                        "exposure": float(t_obs - t0),
                        "event": 1,
                    }
                    for c in x_cols:
                        rec[c] = getattr(row, c)
                    if td_treatment_col is not None:
                        rec[treated_td_col] = int(treated_td)
                    out_rows.append(rec)
                    break

            t1_eff = min(t1_global, t_obs)
            exposure = t1_eff - t0
            if exposure <= 0:
                continue

            rec = {
                "id": rid,
                "k": k,
                "t0": t0,
                "t1": t1_eff,
                "exposure": float(exposure),
                "event": 0,
            }
            for c in x_cols:
                rec[c] = getattr(row, c)
            if td_treatment_col is not None:
                rec[treated_td_col] = int(treated_td)
            out_rows.append(rec)

    out = pd.DataFrame(out_rows)
    if not out.empty:
        out["k"] = out["k"].astype(int)
        out["event"] = out["event"].astype(int)
        if td_treatment_col is not None:
            out[treated_td_col] = out[treated_td_col].astype(int)
    return out