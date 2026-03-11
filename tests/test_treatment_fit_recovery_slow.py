from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peph.treatment.fit import fit_treatment_lognormal_aft

def _simulate_lognormal_aft_data(
    *,
    n: int,
    beta: dict[str, float],
    sigma: float,
    treatment_intercept: float,
    seed: int = 123,
    censor_rate: float | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.normal(0.0, 1.0, n)
    cci = rng.poisson(1.2, n)
    ses = rng.normal(0.0, 1.0, n)
    sex = rng.choice(["F", "M"], size=n, p=[0.55, 0.45])
    stage = rng.choice(["I", "II", "III"], size=n, p=[0.45, 0.35, 0.20])

    mu = (
        float(treatment_intercept)
        + beta["age_per10_centered"] * age
        + beta["cci"] * cci
        + beta["ses"] * ses
        + beta["sexM"] * (sex == "M").astype(float)
        + beta["stageII"] * (stage == "II").astype(float)
        + beta["stageIII"] * (stage == "III").astype(float)
    )

    log_t = mu + float(sigma) * rng.normal(size=n)
    t_true = np.exp(log_t)

    if censor_rate is None:
        t_obs = t_true
        event = np.ones(n, dtype=int)
    else:
        c = rng.exponential(scale=1.0 / censor_rate, size=n)
        t_obs = np.minimum(t_true, c)
        event = (t_true <= c).astype(int)

    t_obs = np.maximum(t_obs, 1e-6)

    return pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "treatment_time_obs": t_obs,
            "treatment_event": event,
            "age_per10_centered": age,
            "cci": cci,
            "ses": ses,
            "sex": sex,
            "stage": stage,
        }
    )


@pytest.mark.slow
def test_fit_treatment_lognormal_aft_parameter_recovery_slow() -> None:
    beta_true = {
        "age_per10_centered": 0.12,
        "cci": 0.18,
        "ses": -0.10,
        "sexM": 0.06,
        "stageII": 0.25,
        "stageIII": 0.50,
    }
    sigma_true = 0.35
    treatment_intercept_true = float(np.log(120.0))

    df = _simulate_lognormal_aft_data(
        n=12000,
        beta=beta_true,
        sigma=sigma_true,
        treatment_intercept=treatment_intercept_true,
        seed=2026,
        censor_rate=None,
    )

    fitted = fit_treatment_lognormal_aft(
        df,
        treatment_time_col="treatment_time_obs",
        treatment_event_col="treatment_event",
        x_numeric=["age_per10_centered", "cci", "ses"],
        x_categorical=["sex", "stage"],
        categorical_reference_levels={"sex": "F", "stage": "I"},
    )

    beta_hat = dict(zip(fitted.x_col_names, fitted.beta))

    assert fitted.converged
    assert abs(beta_hat["Intercept"] - treatment_intercept_true) < 0.04
    assert abs(beta_hat["age_per10_centered"] - beta_true["age_per10_centered"]) < 0.025
    assert abs(beta_hat["cci"] - beta_true["cci"]) < 0.025
    assert abs(beta_hat["ses"] - beta_true["ses"]) < 0.025
    assert abs(beta_hat["sexM"] - beta_true["sexM"]) < 0.035
    assert abs(beta_hat["stageII"] - beta_true["stageII"]) < 0.035
    assert abs(beta_hat["stageIII"] - beta_true["stageIII"]) < 0.04
    assert abs(fitted.sigma - sigma_true) < 0.025


@pytest.mark.slow
def test_fit_treatment_lognormal_aft_parameter_recovery_slow_with_moderate_censoring() -> None:
    beta_true = {
        "age_per10_centered": 0.10,
        "cci": 0.14,
        "ses": -0.08,
        "sexM": 0.05,
        "stageII": 0.20,
        "stageIII": 0.40,
    }
    sigma_true = 0.45
    treatment_intercept_true = float(np.log(120.0))

    df = _simulate_lognormal_aft_data(
        n=15000,
        beta=beta_true,
        sigma=sigma_true,
        treatment_intercept=treatment_intercept_true,
        seed=2027,
        censor_rate=0.003,
    )

    prop_event = float(df["treatment_event"].mean())
    assert 0.35 < prop_event < 0.95

    fitted = fit_treatment_lognormal_aft(
        df,
        treatment_time_col="treatment_time_obs",
        treatment_event_col="treatment_event",
        x_numeric=["age_per10_centered", "cci", "ses"],
        x_categorical=["sex", "stage"],
        categorical_reference_levels={"sex": "F", "stage": "I"},
    )

    beta_hat = dict(zip(fitted.x_col_names, fitted.beta))

    assert fitted.converged
    assert abs(beta_hat["Intercept"] - treatment_intercept_true) < 0.08
    assert abs(beta_hat["age_per10_centered"] - beta_true["age_per10_centered"]) < 0.04
    assert abs(beta_hat["cci"] - beta_true["cci"]) < 0.04
    assert abs(beta_hat["ses"] - beta_true["ses"]) < 0.04
    assert abs(beta_hat["sexM"] - beta_true["sexM"]) < 0.05
    assert abs(beta_hat["stageII"] - beta_true["stageII"]) < 0.06
    assert abs(beta_hat["stageIII"] - beta_true["stageIII"]) < 0.07
    assert abs(fitted.sigma - sigma_true) < 0.04