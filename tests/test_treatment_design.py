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
            "age_per10_centered": [0.0, 1.0, -0.5, 0.3],
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

    assert X.shape == (4, 6)
    assert enc.x_numeric == ["age_per10_centered", "cci"]
    assert enc.x_categorical == ["sex", "stage"]
    assert enc.categorical_reference_levels == {"sex": "F", "stage": "I"}
    assert enc.categorical_levels_seen == {
        "sex": ["F", "M"],
        "stage": ["I", "II", "III"],
    }
    assert enc.x_expanded_cols == [
        "Intercept",
        "age_per10_centered",
        "cci",
        "sexM",
        "stageII",
        "stageIII",
    ]

    X_df = pd.DataFrame(X, columns=enc.x_expanded_cols)

    assert np.allclose(X_df["Intercept"].to_numpy(dtype=float), 1.0)
    assert np.allclose(
        X_df["age_per10_centered"].to_numpy(dtype=float),
        wide_df["age_per10_centered"].to_numpy(dtype=float),
    )
    assert np.allclose(
        X_df["cci"].to_numpy(dtype=float),
        wide_df["cci"].to_numpy(dtype=float),
    )

    assert X_df["sexM"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert X_df["stageII"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert X_df["stageIII"].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_build_x_treatment_fit_missing_covariate_raises() -> None:
    wide_df = _make_wide_df().drop(columns=["cci"])

    with pytest.raises(ValueError, match="Missing required treatment covariate columns"):
        build_x_treatment_fit(
            wide_df,
            x_numeric=["age_per10_centered", "cci"],
            x_categorical=["sex", "stage"],
            categorical_reference_levels={"sex": "F", "stage": "I"},
        )


def test_build_x_treatment_fit_missing_reference_level_raises() -> None:
    wide_df = _make_wide_df()

    with pytest.raises(ValueError, match="Missing categorical reference levels"):
        build_x_treatment_fit(
            wide_df,
            x_numeric=["age_per10_centered", "cci"],
            x_categorical=["sex", "stage"],
            categorical_reference_levels={"sex": "F"},
        )


def test_build_x_treatment_fit_reference_not_observed_raises() -> None:
    wide_df = _make_wide_df()

    with pytest.raises(ValueError, match="Reference level 'IV'"):
        build_x_treatment_fit(
            wide_df,
            x_numeric=["age_per10_centered", "cci"],
            x_categorical=["sex", "stage"],
            categorical_reference_levels={"sex": "F", "stage": "IV"},
        )


def test_build_x_treatment_prediction_matches_fit_encoding() -> None:
    fit_df = _make_wide_df()

    _, enc = build_x_treatment_fit(
        fit_df,
        x_numeric=["age_per10_centered", "cci"],
        x_categorical=["sex", "stage"],
        categorical_reference_levels={"sex": "F", "stage": "I"},
    )

    pred_df = pd.DataFrame(
        {
            "age_per10_centered": [0.2, -0.1],
            "cci": [1, 0],
            "sex": ["M", "F"],
            "stage": ["III", "I"],
        }
    )

    X_pred, unseen = build_x_treatment_prediction(
        pred_df,
        x_numeric=enc.x_numeric,
        x_categorical=enc.x_categorical,
        categorical_reference_levels=enc.categorical_reference_levels,
        categorical_levels_seen=enc.categorical_levels_seen,
        x_col_names=enc.x_expanded_cols,
        hard_fail=True,
    )

    assert unseen is None
    assert X_pred.shape == (2, 6)

    X_df = pd.DataFrame(X_pred, columns=enc.x_expanded_cols)

    assert X_df["Intercept"].tolist() == [1.0, 1.0]
    assert X_df["age_per10_centered"].tolist() == [0.2, -0.1]
    assert X_df["cci"].tolist() == [1.0, 0.0]
    assert X_df["sexM"].tolist() == [1.0, 0.0]
    assert X_df["stageII"].tolist() == [0.0, 0.0]
    assert X_df["stageIII"].tolist() == [1.0, 0.0]


def test_build_x_treatment_prediction_unseen_category_hard_fail_raises() -> None:
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
    assert X_pred.shape == (1, 6)

    X_df = pd.DataFrame(X_pred, columns=enc.x_expanded_cols)
    assert X_df.loc[0, "Intercept"] == 1.0
    assert X_df.loc[0, "sexM"] == 1.0
    assert X_df.loc[0, "stageII"] == 0.0
    assert X_df.loc[0, "stageIII"] == 0.0