from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from peph.report.discover import RunArtifacts


def load_predictions_table(art: RunArtifacts) -> pd.DataFrame:
    if art.predictions_dir is None:
        raise FileNotFoundError(f"predictions/ directory not found under: {art.run_dir}")

    p = art.predictions_dir / "test_predictions.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Prediction file not found: {p}")

    return pd.read_parquet(p)


def prediction_horizons_from_df(df: pd.DataFrame) -> List[int]:
    out: List[int] = []
    for c in df.columns:
        if c.startswith("risk_t"):
            out.append(int(c.split("t", 1)[1]))
    return sorted(set(out))


def prediction_summary_table(
    df: pd.DataFrame,
    horizons: Sequence[int] | None = None,
) -> pd.DataFrame:
    if horizons is None:
        horizons = prediction_horizons_from_df(df)

    rows: List[Dict[str, float]] = []
    for h in horizons:
        rc = f"risk_t{int(h)}"
        sc = f"surv_t{int(h)}"
        cc = f"cumhaz_t{int(h)}"

        if rc not in df.columns:
            continue

        risk = df[rc].to_numpy(dtype=float)
        row: Dict[str, float] = {
            "horizon_days": int(h),
            "n": int(len(risk)),
            "risk_mean": float(np.mean(risk)),
            "risk_sd": float(np.std(risk, ddof=1)) if len(risk) > 1 else 0.0,
            "risk_p10": float(np.quantile(risk, 0.10)),
            "risk_p25": float(np.quantile(risk, 0.25)),
            "risk_p50": float(np.quantile(risk, 0.50)),
            "risk_p75": float(np.quantile(risk, 0.75)),
            "risk_p90": float(np.quantile(risk, 0.90)),
            "risk_min": float(np.min(risk)),
            "risk_max": float(np.max(risk)),
        }

        if sc in df.columns:
            surv = df[sc].to_numpy(dtype=float)
            row["surv_mean"] = float(np.mean(surv))
            row["surv_p50"] = float(np.quantile(surv, 0.50))

        if cc in df.columns:
            ch = df[cc].to_numpy(dtype=float)
            row["cumhaz_mean"] = float(np.mean(ch))
            row["cumhaz_p50"] = float(np.quantile(ch, 0.50))

        rows.append(row)

    return pd.DataFrame(rows)


def top_predicted_risk_table(
    df: pd.DataFrame,
    *,
    horizon: int,
    id_col: str = "id",
    time_col: str = "time",
    event_col: str = "event",
    top: int = 20,
) -> pd.DataFrame:
    rc = f"risk_t{int(horizon)}"
    cols = [c for c in [id_col, time_col, event_col, "eta", rc] if c in df.columns]
    out = df[cols].copy().sort_values(rc, ascending=False).head(int(top)).reset_index(drop=True)
    return out


def risk_group_table(
    df: pd.DataFrame,
    *,
    horizon: int,
    n_groups: int = 10,
    time_col: str = "time",
    event_col: str = "event",
) -> pd.DataFrame:
    """
    Deciles (or n_groups) of predicted risk with observed event rate on known-status subset.

    Known status at horizon:
      - time > horizon
      - or event==1 and time<=horizon
    """
    rc = f"risk_t{int(horizon)}"
    if rc not in df.columns:
        raise ValueError(f"Missing prediction column: {rc}")
    if time_col not in df.columns or event_col not in df.columns:
        raise ValueError(f"Predictions table must contain '{time_col}' and '{event_col}'")

    d = df.copy()
    time = d[time_col].to_numpy(dtype=float)
    event = d[event_col].to_numpy(dtype=int)

    known = (time > float(horizon)) | ((event == 1) & (time <= float(horizon)))
    d = d.loc[known].copy()

    y = (
        (d[event_col].to_numpy(dtype=int) == 1)
        & (d[time_col].to_numpy(dtype=float) <= float(horizon))
    ).astype(float)

    d["_y"] = y
    d["_group"] = pd.qcut(d[rc], q=n_groups, labels=False, duplicates="drop")

    out = (
        d.groupby("_group", as_index=False)
        .agg(
            n=("_y", "size"),
            mean_pred=(rc, "mean"),
            obs_rate=("_y", "mean"),
            risk_min=(rc, "min"),
            risk_max=(rc, "max"),
        )
        .rename(columns={"_group": "group"})
    )
    out["calibration_gap"] = out["obs_rate"] - out["mean_pred"]
    return out.sort_values("group").reset_index(drop=True)


def prediction_diagnostics_table(
    df: pd.DataFrame,
    horizons: Sequence[int] | None = None,
) -> pd.DataFrame:
    if horizons is None:
        horizons = prediction_horizons_from_df(df)

    rows: List[Dict[str, object]] = []

    # bounds
    risk_ok = True
    surv_ok = True
    ch_ok = True
    mono_risk_ok = True
    mono_surv_ok = True
    mono_ch_ok = True

    risk_cols = [f"risk_t{int(h)}" for h in horizons if f"risk_t{int(h)}" in df.columns]
    surv_cols = [f"surv_t{int(h)}" for h in horizons if f"surv_t{int(h)}" in df.columns]
    ch_cols = [f"cumhaz_t{int(h)}" for h in horizons if f"cumhaz_t{int(h)}" in df.columns]

    for c in risk_cols:
        x = df[c].to_numpy(dtype=float)
        if not (np.all(np.isfinite(x)) and np.min(x) >= -1e-10 and np.max(x) <= 1.0 + 1e-10):
            risk_ok = False

    for c in surv_cols:
        x = df[c].to_numpy(dtype=float)
        if not (np.all(np.isfinite(x)) and np.min(x) >= -1e-10 and np.max(x) <= 1.0 + 1e-10):
            surv_ok = False

    for c in ch_cols:
        x = df[c].to_numpy(dtype=float)
        if not (np.all(np.isfinite(x)) and np.min(x) >= -1e-10):
            ch_ok = False

    for c1, c2 in zip(risk_cols[:-1], risk_cols[1:]):
        if not np.all(df[c2].to_numpy(dtype=float) >= df[c1].to_numpy(dtype=float) - 1e-8):
            mono_risk_ok = False

    for c1, c2 in zip(surv_cols[:-1], surv_cols[1:]):
        if not np.all(df[c2].to_numpy(dtype=float) <= df[c1].to_numpy(dtype=float) + 1e-8):
            mono_surv_ok = False

    for c1, c2 in zip(ch_cols[:-1], ch_cols[1:]):
        if not np.all(df[c2].to_numpy(dtype=float) >= df[c1].to_numpy(dtype=float) - 1e-8):
            mono_ch_ok = False

    rows.extend(
        [
            {"check": "risk_bounds", "pass": risk_ok},
            {"check": "survival_bounds", "pass": surv_ok},
            {"check": "cumhaz_nonnegative", "pass": ch_ok},
            {"check": "risk_monotone_in_horizon", "pass": mono_risk_ok},
            {"check": "survival_monotone_in_horizon", "pass": mono_surv_ok},
            {"check": "cumhaz_monotone_in_horizon", "pass": mono_ch_ok},
        ]
    )
    return pd.DataFrame(rows)