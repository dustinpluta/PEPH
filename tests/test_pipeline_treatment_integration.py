from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from peph.config.schema import load_run_config
from peph.pipeline.run import run_pipeline
from peph.sim.joint_ttt_survival import simulate_joint_ttt_survival_dataset


def _write_joint_input_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "joint_ttt_survival_dataset.csv"

    df = simulate_joint_ttt_survival_dataset(
        n=2000,
        breaks=[0.0, 30.0, 90.0, 180.0, 365.0, 730.0, 1825.0],
        nu=[0.0018, 0.0014, 0.0011, 0.0009, 0.0007, 0.0005],
        beta_survival={
            "age_per10_centered": 0.06,
            "cci": 0.10,
            "tumor_size_log": 0.14,
            "ses": -0.05,
            "sexM": 0.03,
            "stageII": 0.18,
            "stageIII": 0.30,
        },
        gamma_treated=-0.30,
        beta_treatment={
            "age_per10_centered": 0.04,
            "cci": 0.08,
            "tumor_size_log": 0.06,
            "ses": -0.04,
            "sexM": 0.02,
            "stageII": 0.10,
            "stageIII": 0.22,
        },
        sigma_treatment=0.45,
        treatment_intercept=float(np.log(120.0)),
        zips_path="data/zips.csv",
        edges_path="data/zip_adjacency.csv",
        edges_u_col="zip_i",
        edges_v_col="zip_j",
        tau_true=1.0,
        rho_true=0.85,
        admin_censor=1825.0,
        random_censor_rate=0.001,
        seed=777,
        return_latent_truth=False,
    )
    df.to_csv(csv_path, index=False)
    return csv_path


def _pipeline_config(csv_path: Path, out_root: Path) -> dict:
    return {
        "run_name": "test_joint_pipeline_with_treatment",
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
            "x_numeric": [
                "age_per10_centered",
                "cci",
                "tumor_size_log",
                "ses",
            ],
            "x_categorical": [
                "sex",
                "stage",
            ],
            "x_td_numeric": [
                "treated_td",
            ],
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
            "covariance": "classical",
            "max_iter": 200,
            "tol": 1e-8,
            "leroux_max_iter": 50,
            "leroux_ftol": 1e-7,
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
        "treatment": {
            "enabled": True,
            "time_col": "treatment_time_obs",
            "event_col": "treatment_event",
            "x_numeric": [
                "age_per10_centered",
                "cci",
                "ses",
            ],
            "x_categorical": [
                "sex",
                "stage",
            ],
            "categorical_reference_levels": {
                "sex": "F",
                "stage": "I",
            },
            "max_iter": 300,
            "tol": 1e-8,
            "optimizer_method": "L-BFGS-B",
            "write_reference_predictions": True,
            "reference_n": 4,
            "reference_horizons": [30.0, 60.0, 90.0, 180.0],
            "reference_quantiles": [0.25, 0.75],
        },
        "split": {
            "test_size": 0.25,
            "seed": 123,
        },
        "predict": {
            "horizons_days": [365.0, 730.0],
            "frailty_mode": "auto",
        },
        "metrics": {
            "discrimination": {
                "c_index": True,
                "time_dependent_auc": True,
            },
            "calibration": {
                "brier_score": True,
                "calibration_in_the_large": True,
                "calibration_slope": True,
            },
            "residuals": {
                "cox_snell": True,
            },
        },
        "plots": {
            "cox_snell": False,
            "calibration_risk": False,
        },
    }


def test_pipeline_writes_treatment_artifacts_when_enabled(tmp_path: Path) -> None:
    csv_path = _write_joint_input_csv(tmp_path)
    out_root = tmp_path / "pipeline_out"

    cfg_dict = _pipeline_config(csv_path, out_root)
    cfg_path = tmp_path / "run_joint_pipeline_with_treatment.yml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")

    cfg = load_run_config(cfg_path)
    out_dir = run_pipeline(cfg)

    assert out_dir.exists()

    treatment_dir = out_dir / "treatment"
    assert treatment_dir.exists()

    model_path = treatment_dir / "treatment_model.json"
    coef_csv = treatment_dir / "treatment_coefficients.csv"
    coef_parquet = treatment_dir / "treatment_coefficients.parquet"
    summary_json = treatment_dir / "treatment_summary.json"
    ref_csv = treatment_dir / "treatment_reference_predictions.csv"
    ref_parquet = treatment_dir / "treatment_reference_predictions.parquet"

    assert model_path.exists()
    assert coef_csv.exists()
    assert coef_parquet.exists()
    assert summary_json.exists()
    assert ref_csv.exists()
    assert ref_parquet.exists()

    coef_df = pd.read_csv(coef_csv)
    assert not coef_df.empty
    assert {
        "term",
        "coef",
        "se",
        "z",
        "p_value",
        "ci_lower",
        "ci_upper",
        "time_ratio",
        "time_ratio_ci_lower",
        "time_ratio_ci_upper",
    }.issubset(coef_df.columns)

    ref_df = pd.read_csv(ref_csv)
    assert len(ref_df) == 4
    assert {
        "pred_treatment_median",
        "pred_treatment_mean",
        "pred_prob_treated_by_30",
        "pred_prob_treated_by_60",
        "pred_prob_treated_by_90",
        "pred_prob_treated_by_180",
        "pred_treatment_quantile_0p25",
        "pred_treatment_quantile_0p75",
    }.issubset(ref_df.columns)

    assert (ref_df["pred_treatment_mean"] >= ref_df["pred_treatment_median"]).all()
    assert (ref_df["pred_prob_treated_by_30"] <= ref_df["pred_prob_treated_by_60"]).all()
    assert (ref_df["pred_prob_treated_by_60"] <= ref_df["pred_prob_treated_by_90"]).all()
    assert (ref_df["pred_prob_treated_by_90"] <= ref_df["pred_prob_treated_by_180"]).all()

    # Survival side should still run
    assert (out_dir / "model.json").exists()
    assert (out_dir / "predictions" / "test_predictions.parquet").exists()
    assert (out_dir / "metrics.json").exists()
