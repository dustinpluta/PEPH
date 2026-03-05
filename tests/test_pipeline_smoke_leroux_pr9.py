from __future__ import annotations

from pathlib import Path

import pytest

from peph.config.schema import load_run_config
from peph.pipeline.run import run_pipeline


@pytest.mark.slow
def test_pipeline_smoke_leroux_pr9_outputs(tmp_path: Path) -> None:
    """
    Smoke test for Leroux pipeline with PR9 spatial artifacts.

    Uses tests/fixtures/run_leroux_small.yml (map_leroux backend, horizons 365/730/1825).
    Writes outputs under a tmp root_dir via overrides.
    """
    cfg_path = "tests/fixtures/run_leroux_small.yml"

    # Your load_run_config expects overrides as a dict (calls .items()).
    cfg = load_run_config(cfg_path, overrides={"output.root_dir": tmp_path.as_posix()})

    out_dir = run_pipeline(cfg)
    assert out_dir.exists()

    # --- Core pipeline artifacts ---
    assert (out_dir / "config_resolved.yml").exists()
    assert (out_dir / "split_ids.json").exists()

    assert (out_dir / "train_wide.parquet").exists()
    assert (out_dir / "test_wide.parquet").exists()
    assert (out_dir / "long_train.parquet").exists()
    assert (out_dir / "long_test.parquet").exists()

    assert (out_dir / "model.json").exists()
    assert (out_dir / "coef_table.parquet").exists()
    assert (out_dir / "baseline_table.parquet").exists()
    assert (out_dir / "inference.json").exists()

    assert (out_dir / "predictions" / "test_predictions.parquet").exists()
    assert (out_dir / "metrics.json").exists()

    # --- PR9 spatial artifacts ---
    assert (out_dir / "frailty_table.parquet").exists()
    assert (out_dir / "frailty_summary.json").exists()
    assert (out_dir / "spatial_autocorr.json").exists()

    # Plots
    assert (out_dir / "plots" / "frailty_caterpillar.png").exists()
    assert (out_dir / "plots" / "morans_scatter_u.png").exists()

    # --- PR9 grouped calibration by frailty decile at each horizon ---
    for h in (365, 730, 1825):
        assert (out_dir / "tables" / f"calibration_by_frailty_decile_t{h}.parquet").exists()
        assert (out_dir / "plots" / f"calibration_by_frailty_decile_t{h}.png").exists()