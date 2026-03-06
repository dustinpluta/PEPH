from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from peph.report.discover import discover_run_artifacts
from peph.report.tables import load_coef_table, load_baseline_table, load_metrics, coef_with_hr, top_terms
from peph.utils.json import write_json


def _make_fake_run_dir(tmp_path: Path) -> Path:
    rd = tmp_path / "run"
    rd.mkdir()

    # minimal coef table
    coef = pd.DataFrame(
        {
            "term": ["x1", "x2"],
            "beta": [0.1, -0.2],
            "se": [0.05, 0.07],
            "z": [2.0, -2.857],
            "p": [0.045, 0.0043],
            "ci_lo": [0.002, -0.337],
            "ci_hi": [0.198, -0.063],
        }
    )
    coef.to_parquet(rd / "coef_table.parquet", index=False)

    # minimal baseline table
    base = pd.DataFrame({"t0": [0, 30], "t1": [30, 90], "nu_hat": [0.004, 0.003]})
    base.to_parquet(rd / "baseline_table.parquet", index=False)

    # minimal metrics
    write_json(str(rd / "metrics.json"), {"c_index": 0.62, "auc_t365": 0.70})

    # dummy dirs
    (rd / "plots").mkdir()
    (rd / "predictions").mkdir()
    (rd / "tables").mkdir()

    return rd


def test_discover_and_load_tables(tmp_path: Path) -> None:
    rd = _make_fake_run_dir(tmp_path)
    art = discover_run_artifacts(rd)

    df_coef = load_coef_table(art)
    assert list(df_coef["term"]) == ["x1", "x2"]

    df_base = load_baseline_table(art)
    assert "nu_hat" in df_base.columns

    m = load_metrics(art)
    assert m["c_index"] == 0.62


def test_coef_with_hr_and_top_terms(tmp_path: Path) -> None:
    rd = _make_fake_run_dir(tmp_path)
    art = discover_run_artifacts(rd)
    df = load_coef_table(art)

    df2 = coef_with_hr(df)
    assert "hr" in df2.columns
    assert "hr_lo" in df2.columns
    assert "hr_hi" in df2.columns

    top = top_terms(df2, top=1, sort="abs_z")
    assert len(top) == 1