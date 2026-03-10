import numpy as np
import pandas as pd
import pytest

from peph.model.design import build_design_long_train


def test_build_design_long_train_includes_treated_td_column() -> None:
    long_df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "k": [0, 1, 0, 1],
            "event": [0, 1, 0, 0],
            "exposure": [30.0, 20.0, 30.0, 30.0],
            "age_per10_centered": [0.2, 0.2, -0.1, -0.1],
            "treated_td": [0, 1, 0, 0],
            "sex": ["F", "F", "M", "M"],
            "stage": ["II", "II", "III", "III"],
        }
    )

    y, X, offset, info = build_design_long_train(
        long_df,
        x_numeric=["age_per10_centered"],
        x_td_numeric=["treated_td"],
        x_categorical=["sex", "stage"],
        categorical_reference_levels={"sex": "F", "stage": "II"},
        K=2,
    )

    assert y.tolist() == [0.0, 1.0, 0.0, 0.0]
    assert np.allclose(offset, np.log([30.0, 20.0, 30.0, 30.0]))
    assert info.baseline_col_names == ["log_nu[0]", "log_nu[1]"]
    assert info.x_col_names == [
        "age_per10_centered",
        "treated_td",
        "sexM",
        "stageIII",
    ]
    assert info.param_names == [
        "log_nu[0]",
        "log_nu[1]",
        "age_per10_centered",
        "treated_td",
        "sexM",
        "stageIII",
    ]
    assert X.shape == (4, 6)

    treated_idx = info.param_names.index("treated_td")
    assert np.allclose(X[:, treated_idx], long_df["treated_td"].to_numpy(dtype=float))


def test_build_design_long_train_all_zero_treated_td_is_allowed() -> None:
    long_df = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "k": [0, 1, 0],
            "event": [0, 0, 1],
            "exposure": [30.0, 40.0, 25.0],
            "x": [1.0, 1.0, -0.5],
            "treated_td": [0, 0, 0],
        }
    )

    y, X, offset, info = build_design_long_train(
        long_df,
        x_numeric=["x"],
        x_td_numeric=["treated_td"],
        x_categorical=[],
        categorical_reference_levels={},
        K=2,
    )

    assert info.x_col_names == ["x", "treated_td"]
    treated_idx = info.param_names.index("treated_td")
    assert np.allclose(X[:, treated_idx], 0.0)
    assert np.allclose(offset, np.log([30.0, 40.0, 25.0]))


def test_build_design_long_train_without_x_td_numeric_is_backward_compatible() -> None:
    long_df = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "k": [0, 1, 0],
            "event": [0, 1, 0],
            "exposure": [30.0, 10.0, 25.0],
            "x": [0.5, 0.5, -0.25],
            "sex": ["F", "F", "M"],
        }
    )

    y, X, offset, info = build_design_long_train(
        long_df,
        x_numeric=["x"],
        x_categorical=["sex"],
        categorical_reference_levels={"sex": "F"},
        K=2,
    )

    assert info.x_col_names == ["x", "sexM"]
    assert info.param_names == ["log_nu[0]", "log_nu[1]", "x", "sexM"]
    assert X.shape == (3, 4)
    assert np.allclose(offset, np.log([30.0, 10.0, 25.0]))


def test_build_design_long_train_raises_if_treated_td_missing() -> None:
    long_df = pd.DataFrame(
        {
            "id": [1, 1],
            "k": [0, 1],
            "event": [0, 1],
            "exposure": [30.0, 20.0],
            "x": [0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="Time-dependent numeric covariate column not found"):
        build_design_long_train(
            long_df,
            x_numeric=["x"],
            x_td_numeric=["treated_td"],
            x_categorical=[],
            categorical_reference_levels={},
            K=2,
        )


def test_build_design_long_train_raises_on_duplicate_covariate_names() -> None:
    long_df = pd.DataFrame(
        {
            "id": [1, 1],
            "k": [0, 1],
            "event": [0, 1],
            "exposure": [30.0, 20.0],
            "treated_td": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="Covariate names must be unique"):
        build_design_long_train(
            long_df,
            x_numeric=["treated_td"],
            x_td_numeric=["treated_td"],
            x_categorical=[],
            categorical_reference_levels={},
            K=2,
        )