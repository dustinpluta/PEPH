from __future__ import annotations

import numpy as np
import pandas as pd

from peph.model.predict_bootstrap import (
    predict_risk_bootstrap,
    predict_risk_contrast_bootstrap,
    predict_survival_bootstrap,
)
from peph.model.result import FeatureEncoding, FittedPEPHModel


def _make_ttt_test_model(gamma_treated: float = -0.5) -> FittedPEPHModel:
    enc = FeatureEncoding(
        x_numeric=["x1"],
        x_categorical=["sex"],
        categorical_reference_levels={"sex": "F"},
        categorical_levels_seen={"sex": ["F", "M"]},
        x_expanded_cols=["x1", "treated_td", "sexM"],
        x_td_numeric=["treated_td"],
    )

    alpha = np.log(np.array([0.01, 0.02], dtype=float))
    beta_x1 = 0.3
    beta_sexM = 0.2

    params = [
        float(alpha[0]),
        float(alpha[1]),
        float(beta_x1),
        float(gamma_treated),
        float(beta_sexM),
    ]

    cov = np.eye(len(params), dtype=float) * 0.01

    return FittedPEPHModel(
        breaks=[0.0, 30.0, 90.0],
        interval_convention="[a,b)",
        encoding=enc,
        baseline_col_names=["log_nu[0]", "log_nu[1]"],
        x_col_names=["x1", "treated_td", "sexM"],
        param_names=["log_nu[0]", "log_nu[1]", "x1", "treated_td", "sexM"],
        params=params,
        cov=cov.tolist(),
        nu=[0.01, 0.02],
        fit_backend="statsmodels_glm_poisson::classical",
        n_train_subjects=100,
        n_train_long_rows=200,
    )


def test_predict_risk_bootstrap_returns_intervals() -> None:
    model = _make_ttt_test_model(gamma_treated=-0.5)

    wide = pd.DataFrame(
        {
            "id": [1],
            "x1": [0.2],
            "sex": ["F"],
            "treatment_time": [30.0],
        }
    )

    out = predict_risk_bootstrap(
        wide,
        model,
        times=[45.0, 80.0],
        treatment_time_col="treatment_time",
        counterfactual_mode="observed",
        n_boot=50,
        alpha=0.05,
        seed=123,
        hard_fail=True,
    )

    assert set(out.keys()) == {"point", "mean", "sd", "lower", "upper"}

    for key in ["point", "mean", "sd", "lower", "upper"]:
        assert out[key].shape == (1, 2)

    assert np.all(out["lower"] <= out["point"])
    assert np.all(out["point"] <= out["upper"])
    assert np.all(out["sd"] >= 0.0)


def test_predict_survival_bootstrap_returns_intervals() -> None:
    model = _make_ttt_test_model(gamma_treated=-0.5)

    wide = pd.DataFrame(
        {
            "id": [1],
            "x1": [0.2],
            "sex": ["F"],
            "treatment_time": [30.0],
        }
    )

    out = predict_survival_bootstrap(
        wide,
        model,
        times=[45.0, 80.0],
        treatment_time_col="treatment_time",
        counterfactual_mode="observed",
        n_boot=50,
        alpha=0.05,
        seed=123,
        hard_fail=True,
    )

    assert set(out.keys()) == {"point", "mean", "sd", "lower", "upper"}

    for key in ["point", "mean", "sd", "lower", "upper"]:
        assert out[key].shape == (1, 2)

    assert np.all(out["lower"] <= out["point"])
    assert np.all(out["point"] <= out["upper"])
    assert np.all(out["point"] >= 0.0)
    assert np.all(out["point"] <= 1.0)


def test_predict_risk_contrast_bootstrap_returns_intervals() -> None:
    model = _make_ttt_test_model(gamma_treated=-0.6)

    wide = pd.DataFrame(
        {
            "id": [1],
            "x1": [0.0],
            "sex": ["F"],
            "treatment_time": [np.nan],
        }
    )

    out = predict_risk_contrast_bootstrap(
        wide,
        model,
        times=[365.0],
        scenario_a={
            "counterfactual_mode": "fixed",
            "fixed_treatment_time": 30.0,
        },
        scenario_b={
            "counterfactual_mode": "fixed",
            "fixed_treatment_time": 90.0,
        },
        n_boot=50,
        alpha=0.05,
        seed=123,
        treatment_time_col="treatment_time",
        hard_fail=True,
    )

    assert set(out.keys()) == {"point", "mean", "sd", "lower", "upper"}

    for key in ["point", "mean", "sd", "lower", "upper"]:
        assert out[key].shape == (1, 1)

    assert np.all(out["lower"] <= out["point"])
    assert np.all(out["point"] <= out["upper"])

    # With gamma < 0, delaying treatment should increase risk
    assert out["point"][0, 0] > 0.0