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
      to the interval that *starts* at that break.
    - If t == breaks[-1], there is no valid interval in [a,b) (outside the window).
    """
    k = int(np.searchsorted(breaks, t, side="right") - 1)
    return k


def expand_long(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    event_col: str,
    x_cols: List[str],
    breaks: List[float],
) -> pd.DataFrame:
    """
    Expand wide right-censored survival data into piecewise exponential long form.

    Conventions
    -----------
    - breaks define intervals [breaks[k], breaks[k+1]) in days.
    - event time exactly equal to a break is assigned to the interval that starts at that break.
    - follow-up is truncated at breaks[-1] (administrative censoring window).
    - If an event occurs exactly at an interval start (a break), we emit a row in that interval
      with exposure=0 and event=1. This preserves the boundary convention in long form.

    Parameters
    ----------
    df : pd.DataFrame
        Wide survival dataset (one row per subject) with right censoring.
    id_col, time_col, event_col : str
        Column names for subject id, observed time (days), and event indicator (0/1).
    x_cols : list[str]
        Covariate columns to carry through (time-constant in v1).
    breaks : list[float]
        Increasing breakpoints in days, starting at 0. Intervals are [a,b).

    Returns
    -------
    pd.DataFrame
        Long-form dataset with columns:
        id, k, t0, t1, exposure, event, plus x_cols.
    """
    b = np.asarray(breaks, dtype=float)
    if b.ndim != 1 or b.size < 2:
        raise ValueError("breaks must be a 1D array with length>=2")
    if b[0] != 0 or np.any(np.diff(b) <= 0):
        raise ValueError("breaks must be strictly increasing and start at 0")

    required = [id_col, time_col, event_col] + list(x_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

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

        # truncate at administrative window (breaks[-1])
        t_obs = min(t, tmax)
        e_obs = 1 if (e == 1 and t <= tmax) else 0

        if t_obs == 0:
            # no exposure => contributes nothing; produce zero rows
            continue

        # Determine event interval index if event observed within window
        k_event = None
        if e_obs == 1:
            if t_obs == tmax:
                # t == breaks[-1] is outside [a,b) window -> treat as censored at tmax
                e_obs = 0
            else:
                k_event = _interval_index_left_closed_right_open(b, t_obs)
                if not (0 <= k_event < K):
                    raise RuntimeError(f"Event interval out of range for id={rid}, t={t_obs}")

        # Iterate intervals, accumulate exposure
        for k in range(K):
            t0 = float(b[k])
            t1 = float(b[k + 1])

            # Stop once subject time is strictly before interval start.
            # NOTE: we use < (not <=) so that t_obs == t0 still allows the
            # event-at-boundary special case below.
            if t_obs < t0:
                break

            exposure = min(t_obs, t1) - t0

            # Special case: event exactly at interval start belongs to this interval (exposure=0).
            if (
                e_obs == 1
                and k_event is not None
                and k == k_event
                and isclose(t_obs, t0)
            ):
                rec = {
                    "id": rid,
                    "k": k,
                    "t0": t0,
                    "t1": t1,
                    "exposure": 0.0,
                    "event": 1,
                }
                for c in x_cols:
                    rec[c] = getattr(row, c)
                out_rows.append(rec)
                break

            if exposure <= 0:
                continue

            is_event = 1 if (e_obs == 1 and k_event is not None and k == k_event and t_obs < t1) else 0

            rec = {
                "id": rid,
                "k": k,
                "t0": t0,
                "t1": t1,
                "exposure": float(exposure),
                "event": int(is_event),
            }
            for c in x_cols:
                rec[c] = getattr(row, c)
            out_rows.append(rec)

            if is_event == 1:
                break  # no further intervals after event

    out = pd.DataFrame(out_rows)
    if not out.empty:
        out["k"] = out["k"].astype(int)
        out["event"] = out["event"].astype(int)
    return out