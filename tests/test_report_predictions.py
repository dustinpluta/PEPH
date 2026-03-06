from __future__ import annotations

from pathlib import Path

import pandas as pd

from peph.report.discover import discover_run_artifacts
from peph.report.predictions import (
    load_predictions_table,
    prediction_diagnostics_table,
    prediction_horizons_from_df,
    prediction_summary_table,
    risk_group_table,
    top_predicted_risk_table,
)


def _make_fake_run_dir_with_predictions(tmp_path: Path) -> Path:
    rd = tmp_path / "run"
    (rd / "predictions").mkdir(parents=True)

    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "time": [100, 500, 900, 2000],
            "event": [1, 0, 1, 0],
            "eta": [0.1, -0.2, 0.4, 0.0],
            "surv_t365": [0.8, 0.9, 0.7, 0.85],
            "risk_t365": [0.2, 0.1, 0.3, 0.15],
            "cumhaz_t365": [0.22, 0.10, 0.36, 0.16],
            "surv_t730": [0.7, 0.85, 0.55, 0.75],
            "risk_t730": [0.3, 0.15, 0.45, 0.25],
            "cumhaz_t730": [0.36, 0.16, 0.60, 0.29],
            "surv_t1825": [0.5, 0.7, 0.3, 0.55],
            "risk_t1825": [0.5, 0.3, 0.7, 0.45],
            "cumhaz_t1825": [0.69, 0.36, 1.20, 0.60],
        }
    )
    df.to_parquet(rd / "predictions" / "test_predictions.parquet", index=False)
    return rd


def test_prediction_summary_and_horizons(tmp_path: Path) -> None:
    rd = _make_fake_run_dir_with_predictions(tmp_path)
    art = discover_run_artifacts(rd)

    df = load_predictions_table(art)
    horizons = prediction_horizons_from_df(df)
    assert horizons == [365, 730, 1825]

    summary = prediction_summary_table(df, horizons=horizons)
    assert len(summary) == 3
    assert "risk_mean" in summary.columns
    assert "risk_p90" in summary.columns


def test_top_predicted_risk_table(tmp_path: Path) -> None:
    rd = _make_fake_run_dir_with_predictions(tmp_path)
    art = discover_run_artifacts(rd)
    df = load_predictions_table(art)

    top = top_predicted_risk_table(df, horizon=1825, top=2)
    assert len(top) == 2
    assert top.iloc[0]["risk_t1825"] >= top.iloc[1]["risk_t1825"]


def test_risk_group_table(tmp_path: Path) -> None:
    rd = _make_fake_run_dir_with_predictions(tmp_path)
    art = discover_run_artifacts(rd)
    df = load_predictions_table(art)

    out = risk_group_table(df, horizon=365, n_groups=2, time_col="time", event_col="event")
    assert "group" in out.columns
    assert "mean_pred" in out.columns
    assert "obs_rate" in out.columns
    assert out["n"].sum() >= 1


def test_prediction_diagnostics_table(tmp_path: Path) -> None:
    rd = _make_fake_run_dir_with_predictions(tmp_path)
    art = discover_run_artifacts(rd)
    df = load_predictions_table(art)

    out = prediction_diagnostics_table(df)
    assert set(out["check"]) == {
        "risk_bounds",
        "survival_bounds",
        "cumhaz_nonnegative",
        "risk_monotone_in_horizon",
        "survival_monotone_in_horizon",
        "cumhaz_monotone_in_horizon",
    }
    assert out["pass"].all()