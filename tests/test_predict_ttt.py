from __future__ import annotations

import numpy as np
import pandas as pd

from peph.model.predict import predict_cumhaz, predict_risk, predict_survival
from peph.model.predict_td import (
    predict_cumhaz_treated_td,
    predict_risk_treated_td,
    predict_survival_treated_td,
)
from peph.model.result import FeatureEncoding, FittedPEPHModel


def _make_ttt_test_model(gamma_treated: float = -0.5) -> FittedPEPHModel:
    """
    Build a tiny fitted model with:
      - 2 PE intervals: [0, 30), [30, 90)
      - one baseline numeric covariate x1
      - one TD covariate treated_td
      - one categorical sex with reference F
    """
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


def test_predict_cumhaz_treated_td_matches_standard_when_never_treated() -> None:
    model = _make_ttt_test_model(gamma_treated=-0.5)

    wide = pd.DataFrame(
        {
            "id": [1, 2],
            "x1": [0.0, 1.0],
            "sex": ["F", "M"],
            "treatment_time": [np.nan, np.nan],
        }
    )
    times = [15.0, 45.0, 80.0]

    ch_td = predict_cumhaz_treated_td(
        wide,
        model,
        times=times,
        treatment_time_col="treatment_time",
        treated_td_col="treated_td",
        hard_fail=True,
    )

    ch_std = predict_cumhaz(
        wide,
        model,
        times=times,
        hard_fail=True,
    )

    assert np.allclose(ch_td, ch_std)


def test_predict_cumhaz_treated_td_matches_standard_when_treated_after_horizon() -> None:
    model = _make_ttt_test_model(gamma_treated=-0.5)

    wide = pd.DataFrame(
        {
            "id": [1, 2],
            "x1": [0.0, 1.0],
            "sex": ["F", "M"],
            "treatment_time": [120.0, 999.0],
        }
    )
    times = [15.0, 45.0, 80.0]

    ch_td = predict_cumhaz_treated_td(
        wide,
        model,
        times=times,
        treatment_time_col="treatment_time",
        treated_td_col="treated_td",
        hard_fail=True,
    )
    ch_std = predict_cumhaz(
        wide,
        model,
        times=times,
        hard_fail=True,
    )

    assert np.allclose(ch_td, ch_std)


def test_predict_cumhaz_treated_td_reduces_hazard_after_treatment_when_gamma_negative() -> None:
    model = _make_ttt_test_model(gamma_treated=-0.7)

    wide = pd.DataFrame(
        {
            "id": [1, 2],
            "x1": [0.5, 0.5],
            "sex": ["F", "F"],
            "treatment_time": [20.0, np.nan],
        }
    )
    times = [10.0, 25.0, 60.0]

    ch_td = predict_cumhaz_treated_td(
        wide,
        model,
        times=times,
        treatment_time_col="treatment_time",
        treated_td_col="treated_td",
        hard_fail=True,
    )

    assert np.isclose(ch_td[0, 0], ch_td[1, 0])
    assert ch_td[0, 1] < ch_td[1, 1]
    assert ch_td[0, 2] < ch_td[1, 2]


def test_predict_survival_and_risk_treated_td_are_consistent() -> None:
    model = _make_ttt_test_model(gamma_treated=-0.5)

    wide = pd.DataFrame(
        {
            "id": [1, 2],
            "x1": [0.2, -0.3],
            "sex": ["F", "M"],
            "treatment_time": [20.0, np.nan],
        }
    )
    times = [15.0, 45.0, 80.0]

    ch = predict_cumhaz_treated_td(
        wide,
        model,
        times=times,
        treatment_time_col="treatment_time",
        treated_td_col="treated_td",
        hard_fail=True,
    )
    surv = predict_survival_treated_td(
        wide,
        model,
        times=times,
        treatment_time_col="treatment_time",
        treated_td_col="treated_td",
        hard_fail=True,
    )
    risk = predict_risk_treated_td(
        wide,
        model,
        times=times,
        treatment_time_col="treatment_time",
        treated_td_col="treated_td",
        hard_fail=True,
    )

    assert np.allclose(surv, np.exp(-ch))
    assert np.allclose(risk, 1.0 - surv)
    assert np.all((surv >= 0.0) & (surv <= 1.0))
    assert np.all((risk >= 0.0) & (risk <= 1.0))


def test_predict_treated_td_backward_argument_order() -> None:
    model = _make_ttt_test_model(gamma_treated=-0.5)

    wide = pd.DataFrame(
        {
            "id": [1],
            "x1": [0.2],
            "sex": ["F"],
            "treatment_time": [20.0],
        }
    )
    times = [15.0, 45.0]

    ch1 = predict_cumhaz_treated_td(
        wide,
        model,
        times=times,
        treatment_time_col="treatment_time",
        treated_td_col="treated_td",
        hard_fail=True,
    )
    ch2 = predict_cumhaz_treated_td(
        model,
        wide,
        times=times,
        treatment_time_col="treatment_time",
        treated_td_col="treated_td",
        hard_fail=True,
    )

    assert np.allclose(ch1, ch2)


def test_predict_treated_td_missing_column_raises() -> None:
    model = _make_ttt_test_model(gamma_treated=-0.5)

    wide = pd.DataFrame(
        {
            "id": [1],
            "x1": [0.2],
            "sex": ["F"],
        }
    )

    try:
        predict_cumhaz_treated_td(
            wide,
            model,
            times=[30.0],
            treatment_time_col="treatment_time",
            treated_td_col="treated_td",
            hard_fail=True,
        )
    except ValueError as e:
        assert "treatment_time_col='treatment_time' not found" in str(e)
    else:
        raise AssertionError("Expected ValueError for missing treatment_time column")


def test_predict_treated_td_missing_in_model_returns_error() -> None:
    enc = FeatureEncoding(
        x_numeric=["x1"],
        x_categorical=["sex"],
        categorical_reference_levels={"sex": "F"},
        categorical_levels_seen={"sex": ["F", "M"]},
        x_expanded_cols=["x1", "sexM"],
        x_td_numeric=[],
    )

    model = FittedPEPHModel(
        breaks=[0.0, 30.0, 90.0],
        interval_convention="[a,b)",
        encoding=enc,
        baseline_col_names=["log_nu[0]", "log_nu[1]"],
        x_col_names=["x1", "sexM"],
        param_names=["log_nu[0]", "log_nu[1]", "x1", "sexM"],
        params=[np.log(0.01), np.log(0.02), 0.3, 0.2],
        cov=(np.eye(4) * 0.01).tolist(),
        nu=[0.01, 0.02],
        fit_backend="statsmodels_glm_poisson::classical",
        n_train_subjects=100,
        n_train_long_rows=200,
    )

    wide = pd.DataFrame(
        {
            "id": [1],
            "x1": [0.2],
            "sex": ["F"],
            "treatment_time": [20.0],
        }
    )

    try:
        predict_cumhaz_treated_td(
            wide,
            model,
            times=[30.0],
            treatment_time_col="treatment_time",
            treated_td_col="treated_td",
            hard_fail=True,
        )
    except ValueError as e:
        assert "treated_td_col='treated_td' not found" in str(e)
    else:
        raise AssertionError("Expected ValueError when treated_td is not in fitted model")


def test_predict_cumhaz_treated_td_matches_manual_integration() -> None:
    """
    Exact manual check for a single subject with one treatment switch.

    Setup:
      - breaks: [0, 30), [30, 90)
      - baseline hazards: nu = [0.01, 0.02]
      - x1 = 0, sex = F  => baseline linear predictor eta0 = 0
      - gamma_treated = -0.5
      - treatment at day 20
      - predict at day 45

    Manual cumulative hazard:
      H(45)
        = exp(eta0) * H0(20)
        + exp(eta0 + gamma) * (H0(45) - H0(20))

    Here:
      H0(20) = 0.01 * 20 = 0.2
      H0(45) = 0.01 * 30 + 0.02 * 15 = 0.6
      H(45)  = 1.0 * 0.2 + exp(-0.5) * 0.4
    """
    gamma = -0.5
    model = _make_ttt_test_model(gamma_treated=gamma)

    wide = pd.DataFrame(
        {
            "id": [1],
            "x1": [0.0],
            "sex": ["F"],
            "treatment_time": [20.0],
        }
    )

    ch = predict_cumhaz_treated_td(
        wide,
        model,
        times=[45.0],
        treatment_time_col="treatment_time",
        treated_td_col="treated_td",
        hard_fail=True,
    )

    h0_20 = 0.01 * 20.0
    h0_45 = 0.01 * 30.0 + 0.02 * 15.0
    manual = h0_20 + np.exp(gamma) * (h0_45 - h0_20)

    assert ch.shape == (1, 1)
    assert np.isclose(ch[0, 0], manual)

    surv = predict_survival_treated_td(
        wide,
        model,
        times=[45.0],
        treatment_time_col="treatment_time",
        treated_td_col="treated_td",
        hard_fail=True,
    )
    risk = predict_risk_treated_td(
        wide,
        model,
        times=[45.0],
        treatment_time_col="treatment_time",
        treated_td_col="treated_td",
        hard_fail=True,
    )

    assert np.isclose(surv[0, 0], np.exp(-manual))
    assert np.isclose(risk[0, 0], 1.0 - np.exp(-manual))


def test_predict_risk_treated_td_never_has_higher_risk_than_early_treatment_when_gamma_negative() -> None:
    model = _make_ttt_test_model(gamma_treated=-0.7)

    wide = pd.DataFrame(
        {
            "id": [1],
            "x1": [0.3],
            "sex": ["F"],
            "treatment_time": [20.0],
        }
    )

    risk_observed = predict_risk_treated_td(
        wide,
        model,
        times=[60.0],
        treatment_time_col="treatment_time",
        treated_td_col="treated_td",
        counterfactual_mode="observed",
        hard_fail=True,
    )

    risk_never = predict_risk_treated_td(
        wide,
        model,
        times=[60.0],
        treatment_time_col="treatment_time",
        treated_td_col="treated_td",
        counterfactual_mode="never",
        hard_fail=True,
    )

    assert risk_never.shape == (1, 1)
    assert risk_observed.shape == (1, 1)
    assert risk_never[0, 0] > risk_observed[0, 0]


def test_predict_risk_treated_td_fixed_earlier_has_lower_risk_than_fixed_later() -> None:
    model = _make_ttt_test_model(gamma_treated=-0.6)

    wide = pd.DataFrame(
        {
            "id": [1],
            "x1": [0.0],
            "sex": ["F"],
            "treatment_time": [np.nan],
        }
    )

    risk_t30 = predict_risk_treated_td(
        wide,
        model,
        times=[365.0],
        treatment_time_col="treatment_time",
        treated_td_col="treated_td",
        counterfactual_mode="fixed",
        fixed_treatment_time=30.0,
        hard_fail=True,
    )

    risk_t90 = predict_risk_treated_td(
        wide,
        model,
        times=[365.0],
        treatment_time_col="treatment_time",
        treated_td_col="treated_td",
        counterfactual_mode="fixed",
        fixed_treatment_time=90.0,
        hard_fail=True,
    )

    assert risk_t30.shape == (1, 1)
    assert risk_t90.shape == (1, 1)
    assert risk_t30[0, 0] < risk_t90[0, 0]


def test_predict_risk_treated_td_delay_observed_increases_risk() -> None:
    model = _make_ttt_test_model(gamma_treated=-0.5)

    wide = pd.DataFrame(
        {
            "id": [1, 2],
            "x1": [0.2, -0.1],
            "sex": ["F", "M"],
            "treatment_time": [40.0, 70.0],
        }
    )

    risk_observed = predict_risk_treated_td(
        wide,
        model,
        times=[365.0],
        treatment_time_col="treatment_time",
        treated_td_col="treated_td",
        counterfactual_mode="observed",
        hard_fail=True,
    )

    risk_delayed = predict_risk_treated_td(
        wide,
        model,
        times=[365.0],
        treatment_time_col="treatment_time",
        treated_td_col="treated_td",
        counterfactual_mode="delay_observed",
        delay_days=30.0,
        hard_fail=True,
    )

    assert np.all(risk_delayed >= risk_observed)
    assert np.any(risk_delayed > risk_observed)


def test_predict_risk_treated_td_advance_observed_decreases_risk() -> None:
    model = _make_ttt_test_model(gamma_treated=-0.5)

    wide = pd.DataFrame(
        {
            "id": [1, 2],
            "x1": [0.2, -0.1],
            "sex": ["F", "M"],
            "treatment_time": [40.0, 70.0],
        }
    )

    risk_observed = predict_risk_treated_td(
        wide,
        model,
        times=[365.0],
        treatment_time_col="treatment_time",
        treated_td_col="treated_td",
        counterfactual_mode="observed",
        hard_fail=True,
    )

    risk_advanced = predict_risk_treated_td(
        wide,
        model,
        times=[365.0],
        treatment_time_col="treatment_time",
        treated_td_col="treated_td",
        counterfactual_mode="advance_observed",
        delay_days=30.0,
        hard_fail=True,
    )

    assert np.all(risk_advanced <= risk_observed)
    assert np.any(risk_advanced < risk_observed)


def test_predict_treated_td_fixed_requires_fixed_treatment_time() -> None:
    model = _make_ttt_test_model(gamma_treated=-0.5)

    wide = pd.DataFrame(
        {
            "id": [1],
            "x1": [0.0],
            "sex": ["F"],
            "treatment_time": [20.0],
        }
    )

    try:
        predict_risk_treated_td(
            wide,
            model,
            times=[365.0],
            treatment_time_col="treatment_time",
            treated_td_col="treated_td",
            counterfactual_mode="fixed",
            hard_fail=True,
        )
    except ValueError as e:
        assert "fixed_treatment_time is required" in str(e)
    else:
        raise AssertionError("Expected ValueError when fixed_treatment_time is missing")


def test_predict_treated_td_delay_requires_delay_days() -> None:
    model = _make_ttt_test_model(gamma_treated=-0.5)

    wide = pd.DataFrame(
        {
            "id": [1],
            "x1": [0.0],
            "sex": ["F"],
            "treatment_time": [20.0],
        }
    )

    try:
        predict_risk_treated_td(
            wide,
            model,
            times=[365.0],
            treatment_time_col="treatment_time",
            treated_td_col="treated_td",
            counterfactual_mode="delay_observed",
            hard_fail=True,
        )
    except ValueError as e:
        assert "delay_days is required" in str(e)
    else:
        raise AssertionError("Expected ValueError when delay_days is missing")


def test_predict_treated_td_public_api_counterfactual_dispatch() -> None:
    model = _make_ttt_test_model(gamma_treated=-0.6)

    wide = pd.DataFrame(
        {
            "id": [1],
            "x1": [0.0],
            "sex": ["F"],
            "treatment_time": [60.0],
        }
    )

    risk_public = predict_risk(
        wide,
        model,
        times=[365.0],
        treatment_time_col="treatment_time",
        treated_td_col="treated_td",
        counterfactual_mode="fixed",
        fixed_treatment_time=30.0,
        hard_fail=True,
    )

    risk_td = predict_risk_treated_td(
        wide,
        model,
        times=[365.0],
        treatment_time_col="treatment_time",
        treated_td_col="treated_td",
        counterfactual_mode="fixed",
        fixed_treatment_time=30.0,
        hard_fail=True,
    )

    assert np.allclose(risk_public, risk_td)