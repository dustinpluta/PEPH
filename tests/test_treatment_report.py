from __future__ import annotations

import numpy as np
import pandas as pd

from peph.treatment.report import (
    summarize_treatment_coefficients,
    summarize_treatment_model,
    summarize_treatment_probability_by_horizon,
    summarize_treatment_reference_pair_difference,
    summarize_treatment_reference_predictions,
)
from peph.treatment.result import FittedTreatmentAFTModel, TreatmentFeatureEncoding


def _make_treatment_test_model() -> FittedTreatmentAFTModel:
    enc = TreatmentFeatureEncoding(
        x_numeric=["age_per10_centered", "cci"],
        x_categorical=["sex", "stage"],
        categorical_reference_levels={"sex": "F", "stage": "I"},
        categorical_levels_seen={
            "sex": ["F", "M"],
            "stage": ["I", "II", "III"],
        },
        x_expanded_cols=[
            "Intercept",
            "age_per10_centered",
            "cci",
            "sexM",
            "stageII",
            "stageIII",
        ],
    )

    beta = np.array(
        [
            np.log(120.0),  # Intercept
            0.10,           # age_per10_centered
            0.20,           # cci
            0.05,           # sexM
            0.30,           # stageII
            0.60,           # stageIII
        ],
        dtype=float,
    )
    log_sigma = np.log(0.50)
    sigma = float(np.exp(log_sigma))

    params = np.concatenate([beta, np.array([log_sigma])])
    cov = np.eye(len(params), dtype=float) * 0.01

    return FittedTreatmentAFTModel(
        encoding=enc,
        x_col_names=list(enc.x_expanded_cols),
        param_names=list(enc.x_expanded_cols) + ["log_sigma"],
        params=params.tolist(),
        cov=cov.tolist(),
        beta=beta.tolist(),
        log_sigma=float(log_sigma),
        sigma=sigma,
        fit_backend="lognormal_aft_mle",
        n_train_subjects=100,
        converged=True,
        loglik=-123.4,
        aic=260.8,
    )


def _make_prediction_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2],
            "age_per10_centered": [0.0, 1.0],
            "cci": [0, 2],
            "sex": ["F", "M"],
            "stage": ["I", "III"],
        }
    )


def test_summarize_treatment_coefficients_structure_and_values() -> None:
    model = _make_treatment_test_model()

    out = summarize_treatment_coefficients(model, alpha=0.05)

    expected_cols = {
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
    }
    assert expected_cols.issubset(out.columns)
    assert len(out) == len(model.param_names)
    assert out["term"].tolist() == model.param_names

    # Check a couple of exact relationships
    intercept_row = out.loc[out["term"] == "Intercept"].iloc[0]
    cci_row = out.loc[out["term"] == "cci"].iloc[0]
    log_sigma_row = out.loc[out["term"] == "log_sigma"].iloc[0]

    assert np.isclose(intercept_row["coef"], np.log(120.0))
    assert np.isclose(cci_row["coef"], 0.20)
    assert np.isclose(log_sigma_row["coef"], np.log(0.50))

    assert np.isclose(intercept_row["time_ratio"], np.exp(intercept_row["coef"]))
    assert np.isclose(cci_row["time_ratio"], np.exp(0.20))
    assert np.all(out["time_ratio_ci_lower"] <= out["time_ratio"])
    assert np.all(out["time_ratio"] <= out["time_ratio_ci_upper"])


def test_summarize_treatment_model_returns_expected_summary() -> None:
    model = _make_treatment_test_model()

    out = summarize_treatment_model(model)

    assert out["fit_backend"] == "lognormal_aft_mle"
    assert out["n_train_subjects"] == 100
    assert out["converged"] is True
    assert np.isclose(out["loglik"], -123.4)
    assert np.isclose(out["aic"], 260.8)
    assert np.isclose(out["sigma"], 0.50)
    assert np.isclose(out["log_sigma"], np.log(0.50))
    assert out["n_parameters"] == len(model.param_names)
    assert out["n_covariates"] == len(model.x_col_names)
    assert out["x_col_names"] == model.x_col_names
    assert out["param_names"] == model.param_names


def test_summarize_treatment_reference_predictions_columns_and_monotonicity() -> None:
    model = _make_treatment_test_model()
    wide = _make_prediction_df()

    out = summarize_treatment_reference_predictions(
        wide,
        model,
        horizons=[60.0, 120.0, 240.0],
        quantiles=[0.25, 0.75],
        hard_fail=True,
    )

    expected_cols = {
        "pred_treatment_median",
        "pred_treatment_mean",
        "pred_prob_treated_by_60",
        "pred_prob_treated_by_120",
        "pred_prob_treated_by_240",
        "pred_treatment_quantile_0p25",
        "pred_treatment_quantile_0p75",
    }
    assert expected_cols.issubset(out.columns)
    assert len(out) == len(wide)

    # Probabilities should be increasing with horizon
    assert np.all(out["pred_prob_treated_by_60"] <= out["pred_prob_treated_by_120"])
    assert np.all(out["pred_prob_treated_by_120"] <= out["pred_prob_treated_by_240"])

    # Mean >= median for log-normal
    assert np.all(out["pred_treatment_mean"] >= out["pred_treatment_median"])

    # q25 <= median <= q75
    assert np.all(out["pred_treatment_quantile_0p25"] <= out["pred_treatment_median"])
    assert np.all(out["pred_treatment_median"] <= out["pred_treatment_quantile_0p75"])


def test_summarize_treatment_probability_by_horizon_long_format() -> None:
    model = _make_treatment_test_model()
    wide = _make_prediction_df()

    out = summarize_treatment_probability_by_horizon(
        wide,
        model,
        horizons=[30.0, 90.0, 180.0],
        hard_fail=True,
    )

    assert set(out.columns) == {
        "row_index",
        "horizon_days",
        "pred_prob_treated_by_horizon",
    }
    assert len(out) == len(wide) * 3

    # Each row index should appear once per horizon
    counts = out.groupby("row_index").size()
    assert counts.tolist() == [3, 3]

    # Probabilities are bounded
    assert np.all(out["pred_prob_treated_by_horizon"] >= 0.0)
    assert np.all(out["pred_prob_treated_by_horizon"] <= 1.0)


def test_summarize_treatment_reference_pair_difference_signs() -> None:
    model = _make_treatment_test_model()

    # A: lower-risk / faster-treatment profile
    wide_a = pd.DataFrame(
        {
            "id": [1],
            "age_per10_centered": [0.0],
            "cci": [0],
            "sex": ["F"],
            "stage": ["I"],
        }
    )

    # B: higher-risk / slower-treatment profile
    wide_b = pd.DataFrame(
        {
            "id": [2],
            "age_per10_centered": [1.0],
            "cci": [2],
            "sex": ["M"],
            "stage": ["III"],
        }
    )

    out = summarize_treatment_reference_pair_difference(
        wide_a,
        wide_b,
        model,
        horizons=[60.0, 120.0, 240.0],
        hard_fail=True,
    )

    expected_keys = {
        "median_a",
        "median_b",
        "median_diff_b_minus_a",
        "mean_a",
        "mean_b",
        "mean_diff_b_minus_a",
        "probability_differences_b_minus_a",
    }
    assert expected_keys.issubset(out.keys())

    # B has systematically larger mu, so longer treatment times
    assert out["median_b"] > out["median_a"]
    assert out["median_diff_b_minus_a"] > 0.0
    assert out["mean_b"] > out["mean_a"]
    assert out["mean_diff_b_minus_a"] > 0.0

    # Larger treatment time means lower chance of treatment by fixed horizon
    prob_diffs = out["probability_differences_b_minus_a"]
    assert prob_diffs["by_60"] < 0.0
    assert prob_diffs["by_120"] < 0.0
    assert prob_diffs["by_240"] < 0.0


def test_summarize_treatment_reference_pair_difference_requires_single_rows() -> None:
    model = _make_treatment_test_model()
    wide = _make_prediction_df()

    try:
        summarize_treatment_reference_pair_difference(
            wide,
            wide.iloc[[0]],
            model,
            horizons=[60.0, 120.0],
            hard_fail=True,
        )
    except ValueError as e:
        assert "exactly one row" in str(e)
    else:
        raise AssertionError("Expected ValueError for multi-row reference input")


def test_summarize_treatment_reference_predictions_invalid_inputs_raise() -> None:
    model = _make_treatment_test_model()
    wide = _make_prediction_df()

    try:
        summarize_treatment_reference_predictions(
            wide,
            model,
            horizons=[],
            hard_fail=True,
        )
    except ValueError as e:
        assert "horizons must contain at least one value" in str(e)
    else:
        raise AssertionError("Expected ValueError for empty horizons")

    try:
        summarize_treatment_reference_predictions(
            wide,
            model,
            horizons=[60.0],
            quantiles=[0.0],
            hard_fail=True,
        )
    except ValueError as e:
        assert "strictly between 0 and 1" in str(e)
    else:
        raise AssertionError("Expected ValueError for invalid quantile")