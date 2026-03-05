from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peph.config.schema import load_run_config
from peph.pipeline.run import run_pipeline
from peph.utils.json import read_json

from peph.sim.ph import PHSimSpec, simulate_ph_wide  # simulator attached


@pytest.mark.slow
def test_pipeline_simulation_ph_end_to_end(tmp_path: Path) -> None:
    """
    End-to-end PH simulation verification:
      simulate -> CSV -> pipeline -> outputs/metrics/predictions invariants.

    This is intended as a non-flaky integration test; thresholds are conservative.
    """
    # --- 1) simulate a dataset ---
    breaks = [0, 30, 90, 180, 365, 730, 1825]
    nu = [0.0040, 0.0030, 0.0024, 0.0018, 0.0013, 0.0010]  # per-day hazards

    beta_true = {
        "age_per10_centered": 0.10,
        "cci": 0.18,
        "tumor_size_log": 0.25,
        "ses": -0.08,
        "sexM": 0.05,
        "stageII": 0.35,
        "stageIII": 0.55,
        "stageIV": 0.80,
    }

    spec = PHSimSpec(
        breaks=breaks,
        nu=nu,
        beta=beta_true,
        seed=1,
        admin_censor_days=1825.0,
        censoring_enabled=True,
        censoring_rate=0.0008,
    )

    n = 4000  # big enough for stable metrics, still reasonable for a slow test
    df = simulate_ph_wide(n=n, spec=spec, include_debug_cols=False)

    sim_csv = tmp_path / "sim_ph.csv"
    df.to_csv(sim_csv, index=False)

    # --- 2) load config + override to use temp CSV and temp output dir ---
    cfg_path = "tests/fixtures/run_sim_ph_small.yml"

    # Shallow override: replace entire "data" block + output.root_dir.
    cfg = load_run_config(
        cfg_path,
        overrides={
            "data": {"path": sim_csv.as_posix(), "format": "csv"},
            "output": {"root_dir": tmp_path.as_posix()},
        },
    )

    # --- 3) run pipeline ---
    out_dir = run_pipeline(cfg)
    assert out_dir.exists()

    # --- 4) artifact existence ---
    assert (out_dir / "model.json").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "coef_table.parquet").exists()
    assert (out_dir / "baseline_table.parquet").exists()
    assert (out_dir / "inference.json").exists()

    pred_path = out_dir / "predictions" / "test_predictions.parquet"
    assert pred_path.exists()

    # --- 5) prediction invariants ---
    pred = pd.read_parquet(pred_path)

    risk_cols = sorted([c for c in pred.columns if c.startswith("risk_t")], key=lambda s: int(s.split("t")[1]))
    surv_cols = sorted([c for c in pred.columns if c.startswith("surv_t")], key=lambda s: int(s.split("t")[1]))
    ch_cols = sorted([c for c in pred.columns if c.startswith("cumhaz_t")], key=lambda s: int(s.split("t")[1]))

    assert len(risk_cols) == len(surv_cols) == len(ch_cols) >= 1

    for c in risk_cols:
        x = pred[c].to_numpy(float)
        assert np.all(np.isfinite(x))
        assert x.min() >= -1e-10
        assert x.max() <= 1.0 + 1e-10

    for c in surv_cols:
        x = pred[c].to_numpy(float)
        assert np.all(np.isfinite(x))
        assert x.min() >= -1e-10
        assert x.max() <= 1.0 + 1e-10

    for c in ch_cols:
        x = pred[c].to_numpy(float)
        assert np.all(np.isfinite(x))
        assert x.min() >= -1e-10

    # monotonicity across horizons (increasing time)
    for c1, c2 in zip(risk_cols[:-1], risk_cols[1:]):
        assert np.all(pred[c2].to_numpy(float) >= pred[c1].to_numpy(float) - 1e-8)

    for c1, c2 in zip(surv_cols[:-1], surv_cols[1:]):
        assert np.all(pred[c2].to_numpy(float) <= pred[c1].to_numpy(float) + 1e-8)

    for c1, c2 in zip(ch_cols[:-1], ch_cols[1:]):
        assert np.all(pred[c2].to_numpy(float) >= pred[c1].to_numpy(float) - 1e-8)

    # --- 6) metric sanity ---
    metrics = read_json(str(out_dir / "metrics.json"))

    # Your metrics.json already has c_index (per earlier pipeline checks).
    c_index = float(metrics["c_index"])
    assert c_index > 0.60