from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peph.treatment.design import (
    build_x_treatment_fit,
    build_x_treatment_prediction,
)


def _make_wide_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "age_per10_centered": [-0.5, 0.0, 0.5, 1.0],
            "cci": [0, 1, 2, 1],
            "sex": ["F", "M", "F", "M"],
            "stage": ["I", "II", "III", "II"],
        }
    )


def test_build_x_treatment_fit_expands_columns_and_encoding() -> None:
    wide_df = _make_wide_df()

    X, enc = build_x_treatment_fit(
        wide_df,
        x_numeric=["age_per10_centered", "cci"],
        x_categorical=["sex", "stage"],
        categorical_reference_levels={"sex": "F", "stage": "I"},
    )

    assert X.shape == (4, 5)

    assert enc.x_numeric == ["age_per10_centered", "cci"]
    assert enc.x_categorical == ["sex", "stage"]
    assert enc.categorical_reference_levels == {"sex": "F", "stage": "I"}
    assert enc.categorical_levels_seen == {
        "sex": ["F", "M"],
        "stage": ["I", "II", "III"],
    }
    assert enc.x_expanded_cols == [
        "age_per10_centered",
        "cci",
        "sexM",
        "stageII",
        "stageIII",
    ]

    expected = np.array(
        [
            [-0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.5, 2.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    assert np.allclose(X, expected)


def test_build_x_treatment_fit_missing_required_column_raises() -> None:
    wide_df = _make_wide_df().drop(columns=["cci"])

    with pytest.raises(ValueError, match="Missing required treatment covariate columns"):
        build_x_treatment_fit(
            wide_df,
            x_numeric=["age_per10_centered", "cci"],
            x_categorical=["sex"],
            categorical_reference_levels={"sex": "F"},
        )


def test_build_x_treatment_fit_missing_reference_level_spec_raises() -> None:
    wide_df = _make_wide_df()

    with pytest.raises(ValueError, match="Missing categorical reference levels"):
        build_x_treatment_fit(
            wide_df,
            x_numeric=["age_per10_centered"],
            x_categorical=["sex", "stage"],
            categorical_reference_levels={"sex": "F"},
        )


def test_build_x_treatment_fit_reference_level_not_observed_raises() -> None:
    wide_df = _make_wide_df()

    with pytest.raises(ValueError, match="Reference level 'X' for categorical column 'stage'"):
        build_x_treatment_fit(
            wide_df,
            x_numeric=["age_per10_centered"],
            x_categorical=["stage"],
            categorical_reference_levels={"stage": "X"},
        )


def test_build_x_treatment_prediction_matches_fit_on_training_data() -> None:
    wide_df = _make_wide_df()

    X_fit, enc = build_x_treatment_fit(
        wide_df,
        x_numeric=["age_per10_centered", "cci"],
        x_categorical=["sex", "stage"],
        categorical_reference_levels={"sex": "F", "stage": "I"},
    )

    X_pred, unseen = build_x_treatment_prediction(
        wide_df,
        x_numeric=enc.x_numeric,
        x_categorical=enc.x_categorical,
        categorical_reference_levels=enc.categorical_reference_levels,
        categorical_levels_seen=enc.categorical_levels_seen,
        x_col_names=enc.x_expanded_cols,
        hard_fail=True,
    )

    assert unseen is None
    assert X_pred.shape == X_fit.shape
    assert np.allclose(X_pred, X_fit)


def test_build_x_treatment_prediction_unseen_category_hard_fail() -> None:
    fit_df = _make_wide_df()

    _, enc = build_x_treatment_fit(
        fit_df,
        x_numeric=["age_per10_centered", "cci"],
        x_categorical=["sex", "stage"],
        categorical_reference_levels={"sex": "F", "stage": "I"},
    )

    pred_df = pd.DataFrame(
        {
            "age_per10_centered": [0.2],
            "cci": [1],
            "sex": ["F"],
            "stage": ["IV"],
        }
    )

    with pytest.raises(ValueError, match="Unseen categorical levels for 'stage'"):
        build_x_treatment_prediction(
            pred_df,
            x_numeric=enc.x_numeric,
            x_categorical=enc.x_categorical,
            categorical_reference_levels=enc.categorical_reference_levels,
            categorical_levels_seen=enc.categorical_levels_seen,
            x_col_names=enc.x_expanded_cols,
            hard_fail=True,
        )


def test_build_x_treatment_prediction_unseen_category_soft_fail_zero_codes() -> None:
    fit_df = _make_wide_df()

    _, enc = build_x_treatment_fit(
        fit_df,
        x_numeric=["age_per10_centered", "cci"],
        x_categorical=["sex", "stage"],
        categorical_reference_levels={"sex": "F", "stage": "I"},
    )

    pred_df = pd.DataFrame(
        {
            "age_per10_centered": [0.2],
            "cci": [1],
            "sex": ["M"],
            "stage": ["IV"],
        }
    )

    X_pred, unseen = build_x_treatment_prediction(
        pred_df,
        x_numeric=enc.x_numeric,
        x_categorical=enc.x_categorical,
        categorical_reference_levels=enc.categorical_reference_levels,
        categorical_levels_seen=enc.categorical_levels_seen,
        x_col_names=enc.x_expanded_cols,
        hard_fail=False,
    )

    assert unseen == {"stage": ["IV"]}
    assert X_pred.shape == (1, 5)

    expected = np.array([[0.2, 1.0, 1.0, 0.0, 0.0]], dtype=float)
    assert np.allclose(X_pred, expected)


def test_build_x_treatment_prediction_missing_required_column_raises() -> None:
    fit_df = _make_wide_df()

    _, enc = build_x_treatment_fit(
        fit_df,
        x_numeric=["age_per10_centered", "cci"],
        x_categorical=["sex", "stage"],
        categorical_reference_levels={"sex": "F", "stage": "I"},
    )

    pred_df = pd.DataFrame(
        {
            "age_per10_centered": [0.2],
            "sex": ["F"],
            "stage": ["II"],
        }
    )

    with pytest.raises(ValueError, match="Missing required treatment covariate columns"):
        build_x_treatment_prediction(
            pred_df,
            x_numeric=enc.x_numeric,
            x_categorical=enc.x_categorical,
            categorical_reference_levels=enc.categorical_reference_levels,
            categorical_levels_seen=enc.categorical_levels_seen,
            x_col_names=enc.x_expanded_cols,
            hard_fail=True,
        )