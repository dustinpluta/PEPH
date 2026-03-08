from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from peph.config.schema import load_run_config
from peph.pipeline.run import run_pipeline
from peph.sim.ttt_effect_spatial import simulate_peph_spatial_ttt_effect_dataset


def _spatial_ttt_prediction_config(csv_path: Path, out_root: Path) -> dict:
    return {
        "run_name": "ttt_prediction_spatial_smoke",
        "data": {
            "path": str(csv_path),
            "format": "csv",
        },
        "output": {
            "root_dir": str(out_root),
        },
        "data_schema": {
            "id_col": "id",
            "time_col": "time",
            "event_col": "event",
            "x_numeric": ["age_per10_centered", "cci", "tumor_size_log", "ses"],
            "x_categorical": ["sex", "stage"],
            "x_td_numeric": ["treated_td"],
            "categorical_reference_levels": {
                "sex": "F",
                "stage": "I",
            },
        },
        "time": {
            "breaks": [0.0, 30.0, 90.0, 180.0, 365.0, 730.0, 1825.0],
        },
        "fit": {
            "backend": "map_leroux",
            "max_iter": 200,
            "tol": 1e-8,
            "covariance": "classical",
            "leroux_max_iter": 100,
            "leroux_ftol": 1e-6,
            "rho_clip": 1e-6,
            "q_jitter": 1e-8,
            "prior_logtau_sd": 10.0,
            "prior_rho_a": 1.0,
            "prior_rho_b": 1.0,
        },
        "ttt": {
            "enabled": True,
            "treatment_time_col": "treatment_time",
            "treated_td_col": "treated_td",
        },
        "spatial": {
            "area_col": "zip",
            "zips_path": "data/zips.csv",
            "edges_path": "data/zip_adjacency.csv",
            "edges_u_col": "zip_i",
            "edges_v_col": "zip_j",
            "allow_unseen_area": False,
        },
        "split": {
            "test_size": 0.25,
            "seed": 1,
        },
        "predict": {
            "horizons_days": [365.0, 730.0],
            "frailty_mode": "auto",
        },
        # For TD models, scalar eta-based discrimination is not currently used.
        "metrics": {
            "discrimination": {"c_index": False, "time_dependent_auc": False},
            "calibration": {
                "brier_score": True,
                "calibration_in_the_large": True,
                "calibration_slope": True,
            },
            "residuals": {"cox_snell": False},
        },
        "plots": {
            "cox_snell": False,
            "calibration_risk": False,
        },
    }


def test_pipeline_ttt_prediction_spatial_smoke(tmp_path: Path) -> None:
    breaks = [0.0, 30.0, 90.0, 180.0, 365.0, 730.0, 1825.0]
    nu_true = np.array([0.0040, 0.0030, 0.0024, 0.0018, 0.0013, 0.0010], dtype=float)

    beta_true = {
        "age_per10_centered": 0.10,
        "cci": 0.16,
        "tumor_size_log": 0.22,
        "ses": -0.08,
        "sexM": 0.05,
        "stageII": 0.30,
        "stageIII": 0.55,
    }

    wide = simulate_peph_spatial_ttt_effect_dataset(
        n=2000,
        breaks=breaks,
        nu=nu_true,
        beta=beta_true,
        gamma_treated=-0.60,
        zips_path="data/zips.csv",
        edges_path="data/zip_adjacency.csv",
        edges_u_col="zip_i",
        edges_v_col="zip_j",
        tau_true=2.0,
        rho_true=0.85,
        admin_censor=float(breaks[-1]),
        random_censor_rate=0.0007,
        max_treatment_time=365.0,
        seed=456,
        return_latent_truth=False,
    )

    csv_path = tmp_path / "ttt_prediction_spatial_input.csv"
    wide.to_csv(csv_path, index=False)

    out_root = tmp_path / "out_ttt_prediction_spatial"
    cfg_dict = _spatial_ttt_prediction_config(csv_path, out_root)

    cfg_path = tmp_path / "run_ttt_prediction_spatial.yml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")

    cfg = load_run_config(cfg_path)
    out_dir = run_pipeline(cfg)

    pred_path = out_dir / "predictions" / "test_predictions.parquet"
    metrics_path = out_dir / "metrics.json"
    frailty_table_path = out_dir / "frailty_table.parquet"
    frailty_summary_path = out_dir / "frailty_summary.json"
    spatial_autocorr_path = out_dir / "spatial_autocorr.json"
    ttt_summary_path = out_dir / "ttt_summary.json"

    assert pred_path.exists()
    assert metrics_path.exists()
    assert frailty_table_path.exists()
    assert frailty_summary_path.exists()
    assert spatial_autocorr_path.exists()
    assert ttt_summary_path.exists()

    pred = pd.read_parquet(pred_path)

    required_cols = {
        "id",
        "time",
        "event",
        "surv_t365",
        "risk_t365",
        "cumhaz_t365",
        "surv_t730",
        "risk_t730",
        "cumhaz_t730",
    }
    missing = required_cols - set(pred.columns)
    assert not missing, f"Missing prediction columns: {sorted(missing)}"

    # TTT prediction path should not produce scalar eta.
    assert "eta" not in pred.columns

    for col in ["surv_t365", "surv_t730"]:
        vals = pred[col].to_numpy(dtype=float)
        assert np.all(np.isfinite(vals))
        assert np.all((vals >= 0.0) & (vals <= 1.0))

    for col in ["risk_t365", "risk_t730"]:
        vals = pred[col].to_numpy(dtype=float)
        assert np.all(np.isfinite(vals))
        assert np.all((vals >= 0.0) & (vals <= 1.0))

    for col in ["cumhaz_t365", "cumhaz_t730"]:
        vals = pred[col].to_numpy(dtype=float)
        assert np.all(np.isfinite(vals))
        assert np.all(vals >= 0.0)

    # Horizon monotonicity
    assert np.all(pred["risk_t730"].to_numpy(dtype=float) >= pred["risk_t365"].to_numpy(dtype=float))
    assert np.all(pred["cumhaz_t730"].to_numpy(dtype=float) >= pred["cumhaz_t365"].to_numpy(dtype=float))
    assert np.all(pred["surv_t730"].to_numpy(dtype=float) <= pred["surv_t365"].to_numpy(dtype=float))