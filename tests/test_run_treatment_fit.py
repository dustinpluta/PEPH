from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from peph.sim.joint_ttt_survival import simulate_joint_ttt_survival_dataset


def test_run_treatment_fit_script_end_to_end(tmp_path: Path) -> None:
    data_path = tmp_path / "joint_ttt_survival_dataset.csv"
    out_dir = tmp_path / "treatment_fit_out"

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
        treatment_intercept=float(__import__("numpy").log(120.0)),
        zips_path="data/zips.csv",
        edges_path="data/zip_adjacency.csv",
        edges_u_col="zip_i",
        edges_v_col="zip_j",
        tau_true=1.0,
        rho_true=0.85,
        admin_censor=1825.0,
        random_censor_rate=0.001,
        seed=321,
        return_latent_truth=False,
    )
    df.to_csv(data_path, index=False)

    cmd = [
        sys.executable,
        "scripts/run_treatment_fit.py",
        "--data",
        str(data_path),
        "--out-dir",
        str(out_dir),
        "--time-col",
        "treatment_time_obs",
        "--event-col",
        "treatment_event",
        "--x-numeric",
        "age_per10_centered,cci,ses",
        "--x-categorical",
        "sex,stage",
        "--reference-levels",
        "sex=F,stage=I",
        "--write-reference-predictions",
        "--reference-n",
        "4",
        "--reference-horizons",
        "30,60,90,180",
        "--reference-quantiles",
        "0.25,0.75",
    ]

    completed = subprocess.run(cmd, capture_output=True, text=True)

    assert completed.returncode == 0, (
        f"Script failed.\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
    )

    model_path = out_dir / "treatment_model.json"
    coef_path = out_dir / "treatment_coefficients.csv"
    summary_path = out_dir / "treatment_summary.json"
    ref_path = out_dir / "treatment_reference_predictions.csv"

    assert model_path.exists()
    assert coef_path.exists()
    assert summary_path.exists()
    assert ref_path.exists()

    coef_df = pd.read_csv(coef_path)
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

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["fit_backend"] == "lognormal_aft_mle"
    assert summary["n_train_subjects"] == len(df)
    assert "sigma" in summary
    assert "param_names" in summary

    ref_df = pd.read_csv(ref_path)
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