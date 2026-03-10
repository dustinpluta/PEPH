from __future__ import annotations

import numpy as np
import pandas as pd

from peph.report.ttt import summarize_treatment_time_distribution
from peph.model.result import FeatureEncoding, FittedPEPHModel
from peph.report.ttt import (
    summarize_treated_td_effect,
    summarize_treatment_long,
    summarize_treatment_wide,
)


def test_summarize_treatment_wide_overall_and_by_stage() -> None:
    wide_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "stage": ["I", "I", "II", "II", "III", "III"],
            "treatment_time": [30.0, np.nan, 45.0, 60.0, np.nan, 90.0],
        }
    )

    out = summarize_treatment_wide(
        wide_df,
        treatment_time_col="treatment_time",
        stage_col="stage",
    )

    assert out["n_subjects"] == 6
    assert out["n_treated_observed"] == 4
    assert out["n_treatment_unobserved"] == 2
    assert np.isclose(out["prop_treated_observed"], 4 / 6)

    tts = out["treatment_time_summary"]
    assert tts["n"] == 4
    assert np.isclose(tts["mean"], (30.0 + 45.0 + 60.0 + 90.0) / 4.0)
    assert np.isclose(tts["median"], 52.5)

    by_stage = {row["stage"]: row for row in out["by_stage"]}

    assert by_stage["I"]["n_subjects"] == 2
    assert by_stage["I"]["n_treated_observed"] == 1
    assert np.isclose(by_stage["I"]["prop_treated_observed"], 0.5)

    assert by_stage["II"]["n_subjects"] == 2
    assert by_stage["II"]["n_treated_observed"] == 2
    assert np.isclose(by_stage["II"]["prop_treated_observed"], 1.0)

    assert by_stage["III"]["n_subjects"] == 2
    assert by_stage["III"]["n_treated_observed"] == 1
    assert np.isclose(by_stage["III"]["prop_treated_observed"], 0.5)


def test_summarize_treatment_wide_all_missing_times() -> None:
    wide_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "treatment_time": [np.nan, np.nan, np.nan],
        }
    )

    out = summarize_treatment_wide(
        wide_df,
        treatment_time_col="treatment_time",
    )

    assert out["n_subjects"] == 3
    assert out["n_treated_observed"] == 0
    assert out["n_treatment_unobserved"] == 3
    assert np.isclose(out["prop_treated_observed"], 0.0)

    tts = out["treatment_time_summary"]
    assert tts["n"] == 0
    assert tts["mean"] is None
    assert tts["median"] is None


def test_summarize_treatment_long_overall_and_by_stage() -> None:
    long_df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "stage": ["II", "II", "II", "II", "III", "III"],
            "treated_td": [0, 1, 0, 0, 1, 1],
            "exposure": [30.0, 20.0, 25.0, 15.0, 10.0, 40.0],
            "event": [0, 1, 0, 0, 0, 1],
        }
    )

    out = summarize_treatment_long(
        long_df,
        treated_td_col="treated_td",
        exposure_col="exposure",
        event_col="event",
        stage_col="stage",
    )

    assert out["n_rows"] == 6
    assert np.isclose(out["person_time_total"], 140.0)
    assert out["events_total"] == 2

    assert out["n_rows_untreated"] == 3
    assert out["n_rows_treated"] == 3
    assert np.isclose(out["person_time_untreated"], 70.0)
    assert np.isclose(out["person_time_treated"], 70.0)
    assert out["events_untreated"] == 0
    assert out["events_treated"] == 2
    assert np.isclose(out["prop_person_time_treated"], 0.5)
    assert np.isclose(out["event_rate_per_time_treated"], 2.0 / 70.0)
    assert np.isclose(out["event_rate_per_time_untreated"], 0.0)

    by_stage = {row["stage"]: row for row in out["by_stage"]}

    assert by_stage["II"]["n_rows"] == 4
    assert np.isclose(by_stage["II"]["person_time_total"], 90.0)
    assert by_stage["II"]["events_total"] == 1
    assert np.isclose(by_stage["II"]["person_time_untreated"], 70.0)
    assert np.isclose(by_stage["II"]["person_time_treated"], 20.0)

    assert by_stage["III"]["n_rows"] == 2
    assert np.isclose(by_stage["III"]["person_time_total"], 50.0)
    assert by_stage["III"]["events_total"] == 1
    assert np.isclose(by_stage["III"]["person_time_untreated"], 0.0)
    assert np.isclose(by_stage["III"]["person_time_treated"], 50.0)


def test_summarize_treatment_long_all_untreated() -> None:
    long_df = pd.DataFrame(
        {
            "treated_td": [0, 0, 0],
            "exposure": [10.0, 15.0, 5.0],
            "event": [0, 1, 0],
        }
    )

    out = summarize_treatment_long(long_df)

    assert out["n_rows_treated"] == 0
    assert np.isclose(out["person_time_treated"], 0.0)
    assert out["events_treated"] == 0
    assert out["event_rate_per_time_treated"] is None
    assert np.isclose(out["person_time_untreated"], 30.0)
    assert out["events_untreated"] == 1


def test_summarize_treated_td_effect_extracts_coef_hr_se_and_interval() -> None:
    enc = FeatureEncoding(
        x_numeric=["x1"],
        x_categorical=["sex"],
        categorical_reference_levels={"sex": "F"},
        categorical_levels_seen={"sex": ["F", "M"]},
        x_expanded_cols=["x1", "treated_td", "sexM"],
        x_td_numeric=["treated_td"],
    )

    fitted = FittedPEPHModel(
        breaks=[0.0, 30.0, 90.0],
        interval_convention="[a,b)",
        encoding=enc,
        baseline_col_names=["log_nu[0]", "log_nu[1]"],
        x_col_names=["x1", "treated_td", "sexM"],
        param_names=["log_nu[0]", "log_nu[1]", "x1", "treated_td", "sexM"],
        params=[-5.5, -6.0, 0.1, -0.4, 0.2],
        cov=[
            [0.04, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.05, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.01, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.09, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.04],
        ],
        nu=[np.exp(-5.5), np.exp(-6.0)],
        fit_backend="statsmodels_glm_poisson::classical",
        n_train_subjects=100,
        n_train_long_rows=250,
    )

    out = summarize_treated_td_effect(fitted, treated_td_col="treated_td", alpha=0.05)

    assert out is not None
    assert out["term"] == "treated_td"
    assert np.isclose(out["coefficient"], -0.4)
    assert np.isclose(out["se"], 0.3)
    assert np.isclose(out["hazard_ratio"], np.exp(-0.4))
    assert out["ci_lower"] < out["coefficient"] < out["ci_upper"]
    assert out["hr_ci_lower"] < out["hazard_ratio"] < out["hr_ci_upper"]
    assert out["p_value"] is not None


def test_summarize_treated_td_effect_returns_none_if_missing() -> None:
    enc = FeatureEncoding(
        x_numeric=["x1"],
        x_categorical=[],
        categorical_reference_levels={},
        categorical_levels_seen={},
        x_expanded_cols=["x1"],
        x_td_numeric=[],
    )

    fitted = FittedPEPHModel(
        breaks=[0.0, 30.0, 90.0],
        interval_convention="[a,b)",
        encoding=enc,
        baseline_col_names=["log_nu[0]", "log_nu[1]"],
        x_col_names=["x1"],
        param_names=["log_nu[0]", "log_nu[1]", "x1"],
        params=[-5.5, -6.0, 0.1],
        cov=[
            [0.04, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.0, 0.0, 0.01],
        ],
        nu=[np.exp(-5.5), np.exp(-6.0)],
        fit_backend="statsmodels_glm_poisson::classical",
        n_train_subjects=100,
        n_train_long_rows=250,
    )

    out = summarize_treated_td_effect(fitted, treated_td_col="treated_td")
    assert out is None

def test_summarize_treatment_time_distribution_default_bins() -> None:
    wide_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7],
            "treatment_time": [10.0, 35.0, 80.0, 120.0, 250.0, 500.0, np.nan],
        }
    )

    out = summarize_treatment_time_distribution(
        wide_df,
        treatment_time_col="treatment_time",
    )

    assert out["n_subjects"].sum() == 6
    assert np.isclose(out["prop_subjects"].sum(), 1.0)

    counts = {row["bin"]: row["n_subjects"] for _, row in out.iterrows()}
    assert counts["[0, 30)"] == 1
    assert counts["[30, 90)"] == 2
    assert counts["[90, 180)"] == 1
    assert counts["[180, 365)"] == 1
    assert counts["[365, inf)"] == 1


def test_summarize_treatment_time_distribution_no_overflow() -> None:
    wide_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "treatment_time": [10.0, 35.0, 80.0, 120.0],
        }
    )

    out = summarize_treatment_time_distribution(
        wide_df,
        treatment_time_col="treatment_time",
        bins=[0.0, 30.0, 90.0, 180.0],
        include_overflow_bin=False,
    )

    assert list(out["bin"]) == ["[0, 30)", "[30, 90)", "[90, 180)"]
    assert out["n_subjects"].sum() == 4
    assert np.isclose(out["prop_subjects"].sum(), 1.0)


def test_summarize_treatment_time_distribution_all_missing() -> None:
    wide_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "treatment_time": [np.nan, np.nan, np.nan],
        }
    )

    out = summarize_treatment_time_distribution(
        wide_df,
        treatment_time_col="treatment_time",
    )

    assert out["n_subjects"].sum() == 0
    assert out["prop_subjects"].isna().all() or (out["prop_subjects"].fillna(0.0) == 0.0).all()

def test_summarize_treated_td_effect_extracts_coef_hr_se_and_interval() -> None:
    from peph.model.result import FeatureEncoding, FittedPEPHModel
    from peph.report.ttt import summarize_treated_td_effect

    enc = FeatureEncoding(
        x_numeric=["x1"],
        x_categorical=["sex"],
        categorical_reference_levels={"sex": "F"},
        categorical_levels_seen={"sex": ["F", "M"]},
        x_expanded_cols=["x1", "treated_td", "sexM"],
        x_td_numeric=["treated_td"],
    )

    fitted = FittedPEPHModel(
        breaks=[0.0, 30.0, 90.0],
        interval_convention="[a,b)",
        encoding=enc,
        baseline_col_names=["log_nu[0]", "log_nu[1]"],
        x_col_names=["x1", "treated_td", "sexM"],
        param_names=["log_nu[0]", "log_nu[1]", "x1", "treated_td", "sexM"],
        params=[-5.5, -6.0, 0.1, -0.4, 0.2],
        cov=[
            [0.04, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.05, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.01, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.09, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.04],
        ],
        nu=[float(np.exp(-5.5)), float(np.exp(-6.0))],
        fit_backend="statsmodels_glm_poisson::classical",
        n_train_subjects=100,
        n_train_long_rows=250,
    )

    out = summarize_treated_td_effect(
        fitted,
        treated_td_col="treated_td",
        alpha=0.05,
    )

    assert out is not None
    assert out["term"] == "treated_td"

    assert np.isclose(out["coefficient"], -0.4)
    assert np.isclose(out["se"], 0.3)
    assert np.isclose(out["hazard_ratio"], np.exp(-0.4))

    assert out["z"] is not None
    assert out["p_value"] is not None
    assert 0.0 <= out["p_value"] <= 1.0

    assert out["ci_lower"] < out["coefficient"] < out["ci_upper"]
    assert out["hr_ci_lower"] < out["hazard_ratio"] < out["hr_ci_upper"]


def test_summarize_treated_td_effect_returns_none_if_missing() -> None:
    from peph.model.result import FeatureEncoding, FittedPEPHModel
    from peph.report.ttt import summarize_treated_td_effect

    enc = FeatureEncoding(
        x_numeric=["x1"],
        x_categorical=[],
        categorical_reference_levels={},
        categorical_levels_seen={},
        x_expanded_cols=["x1"],
        x_td_numeric=[],
    )

    fitted = FittedPEPHModel(
        breaks=[0.0, 30.0, 90.0],
        interval_convention="[a,b)",
        encoding=enc,
        baseline_col_names=["log_nu[0]", "log_nu[1]"],
        x_col_names=["x1"],
        param_names=["log_nu[0]", "log_nu[1]", "x1"],
        params=[-5.5, -6.0, 0.1],
        cov=[
            [0.04, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.0, 0.0, 0.01],
        ],
        nu=[float(np.exp(-5.5)), float(np.exp(-6.0))],
        fit_backend="statsmodels_glm_poisson::classical",
        n_train_subjects=100,
        n_train_long_rows=250,
    )

    out = summarize_treated_td_effect(
        fitted,
        treated_td_col="treated_td",
    )

    assert out is None