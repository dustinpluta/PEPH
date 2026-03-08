from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
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
        seed=321,
        return_latent_truth=False,
    )

    csv_path = tmp_path / "ttt_prediction_input.csv"
    wide.to_csv(csv_path, index=False)
    return csv_path


def _ttt_prediction_config(csv_path: Path, out_root: Path) -> dict:
    return {
        "run_name": "ttt_prediction_test",
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
        "predict": {
            "horizons_days": [365.0, 730.0],
            "frailty_mode": "auto",
        },
        # Skip discrimination metrics for TD prediction until a scalar score is defined.
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
            "calibration_risk": True,
        },
    }


def test_pipeline_ttt_prediction_writes_prediction_artifacts(tmp_path: Path) -> None:
    csv_path = _write_ttt_input_csv(tmp_path)
    out_root = tmp_path / "out_ttt_prediction"

    cfg_dict = _ttt_prediction_config(csv_path, out_root)
    cfg_path = tmp_path / "run_ttt_prediction.yml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")

    cfg = load_run_config(cfg_path)
    out_dir = run_pipeline(cfg)

    pred_path = out_dir / "predictions" / "test_predictions.parquet"
    metrics_path = out_dir / "metrics.json"
    plot_365 = out_dir / "plots" / "calibration_risk_t365.png"
    plot_730 = out_dir / "plots" / "calibration_risk_t730.png"

    assert pred_path.exists()
    assert metrics_path.exists()
    assert plot_365.exists()
    assert plot_730.exists()

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

    # TTT models do not currently produce a scalar eta in the pipeline.
    assert "eta" not in pred.columns

    # Basic range / monotonicity checks
    for col in ["surv_t365", "surv_t730"]:
        assert np.all(np.isfinite(pred[col].to_numpy(dtype=float)))
        assert np.all((pred[col].to_numpy(dtype=float) >= 0.0) & (pred[col].to_numpy(dtype=float) <= 1.0))

    for col in ["risk_t365", "risk_t730"]:
        assert np.all(np.isfinite(pred[col].to_numpy(dtype=float)))
        assert np.all((pred[col].to_numpy(dtype=float) >= 0.0) & (pred[col].to_numpy(dtype=float) <= 1.0))

    for col in ["cumhaz_t365", "cumhaz_t730"]:
        assert np.all(np.isfinite(pred[col].to_numpy(dtype=float)))
        assert np.all(pred[col].to_numpy(dtype=float) >= 0.0)

    # Longer horizon should not decrease risk / cumhaz or increase survival
    assert np.all(pred["risk_t730"].to_numpy(dtype=float) >= pred["risk_t365"].to_numpy(dtype=float))
    assert np.all(pred["cumhaz_t730"].to_numpy(dtype=float) >= pred["cumhaz_t365"].to_numpy(dtype=float))
    assert np.all(pred["surv_t730"].to_numpy(dtype=float) <= pred["surv_t365"].to_numpy(dtype=float))


def test_pipeline_ttt_prediction_differs_for_earlier_vs_never_treated(tmp_path: Path) -> None:
    csv_path = _write_ttt_input_csv(tmp_path)
    out_root = tmp_path / "out_ttt_prediction_compare"

    cfg_dict = _ttt_prediction_config(csv_path, out_root)
    cfg_path = tmp_path / "run_ttt_prediction_compare.yml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")

    cfg = load_run_config(cfg_path)
    out_dir = run_pipeline(cfg)

    pred = pd.read_parquet(out_dir / "predictions" / "test_predictions.parquet")
    test_wide = pd.read_parquet(out_dir / "test_wide.parquet")

    merged = test_wide.merge(pred, on=["id", "time", "event"], how="inner", validate="one_to_one")

    early_treated = merged.loc[merged["treatment_time"].notna() & (merged["treatment_time"] <= 90.0)]
    never_treated = merged.loc[merged["treatment_time"].isna()]

    # Need nonempty groups for a meaningful comparison
    assert len(early_treated) > 0
    assert len(never_treated) > 0

    # Since gamma_true < 0 in the simulator, earlier-treated subjects should
    # tend to have lower predicted risk on average.
    mean_risk_365_early = float(early_treated["risk_t365"].mean())
    mean_risk_365_never = float(never_treated["risk_t365"].mean())

    assert mean_risk_365_early < mean_risk_365_never, (
        f"Expected earlier-treated group to have lower mean predicted risk at 365 days, "
        f"got early={mean_risk_365_early:.4f}, never={mean_risk_365_never:.4f}"
    )