from __future__ import annotations

import numpy as np
import pandas as pd

from peph.treatment.fit import fit_treatment_lognormal_aft
from peph.treatment.result import FittedTreatmentAFTModel


def _simulate_lognormal_aft_data(
    *,
    n: int,
    beta: dict[str, float],
    sigma: float,
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
        beta["age_per10_centered"] * age
        + beta["cci"] * cci
        + beta["ses"] * ses
        + beta["sexM"] * (sex == "M").astype(float)
        + beta["stageII"] * (stage == "II").astype(float)
        + beta["stageIII"] * (stage == "III").astype(float)
    )

    log_t = mu + sigma * rng.normal(size=n)
    t_true = np.exp(log_t)

    if censor_rate is None:
        t_obs = t_true
        event = np.ones(n, dtype=int)
    else:
        c = rng.exponential(scale=1.0 / censor_rate, size=n)
        t_obs = np.minimum(t_true, c)
        event = (t_true <= c).astype(int)

    # strictly positive times for log-normal AFT
    t_obs = np.maximum(t_obs, 1e-6)

    return pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "treatment_time": t_obs,
            "treatment_event": event,
            "age_per10_centered": age,
            "cci": cci,
            "ses": ses,
            "sex": sex,
            "stage": stage,
        }
    )


def test_fit_treatment_lognormal_aft_smoke() -> None:
    beta_true = {
        "age_per10_centered": 0.10,
        "cci": 0.15,
        "ses": -0.08,
        "sexM": 0.05,
        "stageII": 0.20,
        "stageIII": 0.45,
    }

    df = _simulate_lognormal_aft_data(
        n=400,
        beta=beta_true,
        sigma=0.45,
        seed=1,
        censor_rate=0.10,
    )

    fitted = fit_treatment_lognormal_aft(
        df,
        treatment_time_col="treatment_time",
        treatment_event_col="treatment_event",
        x_numeric=["age_per10_centered", "cci", "ses"],
        x_categorical=["sex", "stage"],
        categorical_reference_levels={"sex": "F", "stage": "I"},
    )

    assert isinstance(fitted, FittedTreatmentAFTModel)
    assert fitted.fit_backend == "lognormal_aft_mle"
    assert fitted.n_train_subjects == len(df)
    assert fitted.sigma > 0.0
    assert len(fitted.x_col_names) == len(fitted.beta)
    assert fitted.param_names[-1] == "log_sigma"
    assert len(fitted.params) == len(fitted.param_names)


def test_fit_treatment_lognormal_aft_parameter_recovery_no_censoring() -> None:
    beta_true = {
        "age_per10_centered": 0.12,
        "cci": 0.18,
        "ses": -0.10,
        "sexM": 0.06,
        "stageII": 0.25,
        "stageIII": 0.50,
    }
    sigma_true = 0.35

    df = _simulate_lognormal_aft_data(
        n=4000,
        beta=beta_true,
        sigma=sigma_true,
        seed=42,
        censor_rate=None,
    )

    fitted = fit_treatment_lognormal_aft(
        df,
        treatment_time_col="treatment_time",
        treatment_event_col="treatment_event",
        x_numeric=["age_per10_centered", "cci", "ses"],
        x_categorical=["sex", "stage"],
        categorical_reference_levels={"sex": "F", "stage": "I"},
    )

    beta_hat = dict(zip(fitted.x_col_names, fitted.beta))

    # moderate tolerances for stable unit testing
    assert abs(beta_hat["age_per10_centered"] - beta_true["age_per10_centered"]) < 0.05
    assert abs(beta_hat["cci"] - beta_true["cci"]) < 0.05
    assert abs(beta_hat["ses"] - beta_true["ses"]) < 0.05
    assert abs(beta_hat["sexM"] - beta_true["sexM"]) < 0.06
    assert abs(beta_hat["stageII"] - beta_true["stageII"]) < 0.06
    assert abs(beta_hat["stageIII"] - beta_true["stageIII"]) < 0.07

    assert abs(fitted.sigma - sigma_true) < 0.05


def test_fit_treatment_lognormal_aft_handles_right_censoring() -> None:
    beta_true = {
        "age_per10_centered": 0.08,
        "cci": 0.12,
        "ses": -0.05,
        "sexM": 0.04,
        "stageII": 0.18,
        "stageIII": 0.35,
    }

    df = _simulate_lognormal_aft_data(
        n=1200,
        beta=beta_true,
        sigma=0.50,
        seed=123,
        censor_rate=0.15,
    )

    # ensure some censoring is actually present
    prop_event = float(df["treatment_event"].mean())
    assert 0.2 < prop_event < 0.95

    fitted = fit_treatment_lognormal_aft(
        df,
        treatment_time_col="treatment_time",
        treatment_event_col="treatment_event",
        x_numeric=["age_per10_centered", "cci", "ses"],
        x_categorical=["sex", "stage"],
        categorical_reference_levels={"sex": "F", "stage": "I"},
    )

    assert isinstance(fitted, FittedTreatmentAFTModel)
    assert fitted.sigma > 0.0
    assert np.isfinite(fitted.log_sigma)
    assert fitted.converged in {True, False}

    beta_hat = dict(zip(fitted.x_col_names, fitted.beta))

    # basic directional sanity checks under censoring
    assert beta_hat["age_per10_centered"] > 0.0
    assert beta_hat["cci"] > 0.0
    assert beta_hat["ses"] < 0.0
    assert beta_hat["stageIII"] > beta_hat["stageII"] > 0.0