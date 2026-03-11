from __future__ import annotations

import numpy as np
import pandas as pd

from peph.treatment.fit import fit_treatment_lognormal_aft
from peph.treatment.result import FittedTreatmentAFTModel


def _simulate_small_dataset(n: int = 400, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.normal(0.0, 1.0, n)
    cci = rng.poisson(1.2, n)
    ses = rng.normal(0.0, 1.0, n)
    sex = rng.choice(["F", "M"], size=n, p=[0.55, 0.45])
    stage = rng.choice(["I", "II", "III"], size=n, p=[0.45, 0.35, 0.20])

    beta = {
        "age_per10_centered": 0.10,
        "cci": 0.15,
        "ses": -0.08,
        "sexM": 0.05,
        "stageII": 0.20,
        "stageIII": 0.40,
    }

    treatment_intercept = float(np.log(120.0))
    sigma = 0.40

    mu = (
        treatment_intercept
        + beta["age_per10_centered"] * age
        + beta["cci"] * cci
        + beta["ses"] * ses
        + beta["sexM"] * (sex == "M")
        + beta["stageII"] * (stage == "II")
        + beta["stageIII"] * (stage == "III")
    )

    log_t = mu + sigma * rng.normal(size=n)
    t = np.exp(log_t)

    return pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "treatment_time_obs": t,
            "treatment_event": np.ones(n, dtype=int),
            "age_per10_centered": age,
            "cci": cci,
            "ses": ses,
            "sex": sex,
            "stage": stage,
        }
    )


def test_treatment_model_json_roundtrip(tmp_path) -> None:
    df = _simulate_small_dataset()

    fitted = fit_treatment_lognormal_aft(
        df,
        treatment_time_col="treatment_time_obs",
        treatment_event_col="treatment_event",
        x_numeric=["age_per10_centered", "cci", "ses"],
        x_categorical=["sex", "stage"],
        categorical_reference_levels={"sex": "F", "stage": "I"},
    )

    assert isinstance(fitted, FittedTreatmentAFTModel)

    model_path = tmp_path / "treatment_model.json"

    fitted.save(model_path)

    loaded = FittedTreatmentAFTModel.load(model_path)

    assert isinstance(loaded, FittedTreatmentAFTModel)

    # core parameter checks
    assert np.allclose(fitted.beta, loaded.beta)
    assert fitted.log_sigma == loaded.log_sigma
    assert fitted.sigma == loaded.sigma
    assert np.allclose(fitted.params, loaded.params)

    # encoding checks
    assert fitted.x_col_names == loaded.x_col_names
    assert fitted.param_names == loaded.param_names
    assert fitted.encoding.x_numeric == loaded.encoding.x_numeric
    assert fitted.encoding.x_categorical == loaded.encoding.x_categorical
    assert (
        fitted.encoding.categorical_reference_levels
        == loaded.encoding.categorical_reference_levels
    )
    assert (
        fitted.encoding.categorical_levels_seen
        == loaded.encoding.categorical_levels_seen
    )
    assert fitted.encoding.x_expanded_cols == loaded.encoding.x_expanded_cols

    # metadata checks
    assert fitted.fit_backend == loaded.fit_backend
    assert fitted.n_train_subjects == loaded.n_train_subjects
    assert fitted.converged == loaded.converged
    assert fitted.loglik == loaded.loglik
    assert fitted.aic == loaded.aic

    # covariance
    assert np.allclose(np.asarray(fitted.cov, dtype=float), np.asarray(loaded.cov, dtype=float))