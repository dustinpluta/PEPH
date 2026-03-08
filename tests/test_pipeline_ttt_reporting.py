from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from peph.config.schema import load_run_config
from peph.pipeline.run import run_pipeline
from peph.sim.ttt_effect import simulate_peph_ttt_effect_dataset


def _write_ttt_input_csv(tmp_path: Path) -> Path:
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

    wide = simulate_peph_ttt_effect_dataset(
        n=1500,
        breaks=breaks,
        nu=nu_true,
        beta=beta_true,
        gamma_treated=-0.60,
        admin_censor=float(breaks[-1]),
        random_censor_rate=0.0007,
        max_treatment_time=365.0,
        seed=123,
        return_latent_truth=False,
    )

    csv_path = tmp_path / "ttt_input.csv"
    wide.to_csv(csv_path, index=False)
    return csv_path


def _base_ttt_config(csv_path: Path, out_root: Path) -> dict:
    return {
        "run_name": "ttt_reporting_test",
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
            "backend": "statsmodels_glm_poisson",
            "max_iter": 200,
            "tol": 1e-8,
            "covariance": "classical",
            "leroux_max_iter": 25,
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
        "split": {
            "test_size": 0.25,
            "seed": 1,
        },
        # prediction for x_td_numeric is not implemented yet
        "predict": {
            "horizons_days": [],
            "frailty_mode": "auto",
        },
        "metrics": {
            "discrimination": {"c_index": False, "time_dependent_auc": False},
            "calibration": {
                "brier_score": False,
                "calibration_in_the_large": False,
                "calibration_slope": False,
            },
            "residuals": {"cox_snell": False},
        },
        "plots": {
            "cox_snell": False,
            "calibration_risk": False,
        },
    }


def test_pipeline_writes_ttt_reporting_artifacts_when_enabled(tmp_path: Path) -> None:
    csv_path = _write_ttt_input_csv(tmp_path)
    out_root = tmp_path / "out_enabled"

    cfg_dict = _base_ttt_config(csv_path, out_root)
    cfg_path = tmp_path / "run_ttt_enabled.yml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")

    cfg = load_run_config(cfg_path)
    out_dir = run_pipeline(cfg)

    ttt_summary_path = out_dir / "ttt_summary.json"
    ttt_wide_stage_path = out_dir / "ttt_wide_by_stage.parquet"
    ttt_long_stage_path = out_dir / "ttt_long_by_stage.parquet"
    ttt_distribution_path = out_dir / "treatment_time_distribution.parquet"
    assert ttt_summary_path.exists()
    assert ttt_wide_stage_path.exists()
    assert ttt_long_stage_path.exists()
    assert (ttt_distribution_path).exists()
    assert (out_dir / "plots" / "treatment_time_histogram.png").exists()

    with open(ttt_summary_path, "r", encoding="utf-8") as f:
        ttt_summary = json.load(f)

    assert "treatment_process" in ttt_summary
    assert "long_risk_time" in ttt_summary
    assert "treatment_effect" in ttt_summary

    tp = ttt_summary["treatment_process"]
    lr = ttt_summary["long_risk_time"]
    te = ttt_summary["treatment_effect"]

    assert tp["n_subjects"] > 0
    assert 0.0 <= tp["prop_treated_observed"] <= 1.0
    assert "treatment_time_summary" in tp

    assert lr["person_time_total"] > 0.0
    assert lr["n_rows"] > 0
    assert "person_time_treated" in lr
    assert "person_time_untreated" in lr

    assert te is not None
    assert te["term"] == "treated_td"
    assert "coefficient" in te
    assert "hazard_ratio" in te
    assert "se" in te
    assert "ci_lower" in te
    assert "ci_upper" in te


def test_pipeline_skips_ttt_reporting_artifacts_when_disabled(tmp_path: Path) -> None:
    csv_path = _write_ttt_input_csv(tmp_path)
    out_root = tmp_path / "out_disabled"

    cfg_dict = _base_ttt_config(csv_path, out_root)

    # disable TTT and remove TD covariates from schema
    cfg_dict["ttt"] = None
    cfg_dict["data_schema"]["x_td_numeric"] = []

    cfg_path = tmp_path / "run_ttt_disabled.yml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")

    cfg = load_run_config(cfg_path)
    out_dir = run_pipeline(cfg)

    assert not (out_dir / "ttt_summary.json").exists()
    assert not (out_dir / "ttt_wide_by_stage.parquet").exists()
    assert not (out_dir / "ttt_long_by_stage.parquet").exists()
    assert not (out_dir / "treatment_time_distribution.parquet").exists()
    assert not (out_dir / "plots" / "treatment_time_histogram.png").exists()