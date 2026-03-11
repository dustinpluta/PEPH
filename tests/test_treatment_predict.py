from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from peph.treatment.predict import (
    predict_treatment_cdf,
    predict_treatment_linear_predictor,
    predict_treatment_logtime_mean,
    predict_treatment_mean,
    predict_treatment_median,
    predict_treatment_probability_by_time,
    predict_treatment_quantile,
    predict_treatment_survival,
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


def test_predict_treatment_linear_predictor_matches_manual() -> None:
    model = _make_treatment_test_model()
    wide = _make_prediction_df()

    mu = predict_treatment_linear_predictor(wide, model)

    mu_manual_0 = np.log(120.0)
    mu_manual_1 = np.log(120.0) + 0.10 * 1.0 + 0.20 * 2.0 + 0.05 + 0.60

    assert mu.shape == (2,)
    assert np.allclose(mu, np.array([mu_manual_0, mu_manual_1], dtype=float))


def test_predict_treatment_logtime_mean_equals_linear_predictor() -> None:
    model = _make_treatment_test_model()
    wide = _make_prediction_df()

    mu = predict_treatment_linear_predictor(wide, model)
    logmean = predict_treatment_logtime_mean(wide, model)

    assert np.allclose(mu, logmean)


def test_predict_treatment_median_matches_exp_mu() -> None:
    model = _make_treatment_test_model()
    wide = _make_prediction_df()

    mu = predict_treatment_linear_predictor(wide, model)
    med = predict_treatment_median(wide, model)

    assert med.shape == (2,)
    assert np.allclose(med, np.exp(mu))


def test_predict_treatment_mean_matches_lognormal_formula() -> None:
    model = _make_treatment_test_model()
    wide = _make_prediction_df()

    mu = predict_treatment_linear_predictor(wide, model)
    mean_pred = predict_treatment_mean(wide, model)

    manual = np.exp(mu + 0.5 * model.sigma**2)

    assert mean_pred.shape == (2,)
    assert np.allclose(mean_pred, manual)


def test_predict_treatment_quantile_matches_formula() -> None:
    model = _make_treatment_test_model()
    wide = _make_prediction_df()

    p = 0.75
    mu = predict_treatment_linear_predictor(wide, model)
    q = predict_treatment_quantile(wide, model, p=p)

    manual = np.exp(mu + model.sigma * norm.ppf(p))

    assert q.shape == (2,)
    assert np.allclose(q, manual)


def test_predict_treatment_survival_matches_formula() -> None:
    model = _make_treatment_test_model()
    wide = _make_prediction_df()
    times = [60.0, 120.0, 240.0]

    surv = predict_treatment_survival(wide, model, times=times)

    mu = predict_treatment_linear_predictor(wide, model)
    z = (np.log(np.asarray(times))[None, :] - mu[:, None]) / model.sigma
    manual = 1.0 - norm.cdf(z)

    assert surv.shape == (2, 3)
    assert np.allclose(surv, manual)
    assert np.all((surv >= 0.0) & (surv <= 1.0))


def test_predict_treatment_cdf_is_one_minus_survival() -> None:
    model = _make_treatment_test_model()
    wide = _make_prediction_df()
    times = [60.0, 120.0, 240.0]

    surv = predict_treatment_survival(wide, model, times=times)
    cdf = predict_treatment_cdf(wide, model, times=times)
    prob = predict_treatment_probability_by_time(wide, model, times=times)

    assert np.allclose(cdf, 1.0 - surv)
    assert np.allclose(prob, cdf)
    assert np.all((cdf >= 0.0) & (cdf <= 1.0))


def test_predict_treatment_backward_argument_order() -> None:
    model = _make_treatment_test_model()
    wide = _make_prediction_df()

    mu1 = predict_treatment_linear_predictor(wide, model)
    mu2 = predict_treatment_linear_predictor(model, wide)

    med1 = predict_treatment_median(wide, model)
    med2 = predict_treatment_median(model, wide)

    surv1 = predict_treatment_survival(wide, model, times=[120.0, 240.0])
    surv2 = predict_treatment_survival(model, wide, times=[120.0, 240.0])

    assert np.allclose(mu1, mu2)
    assert np.allclose(med1, med2)
    assert np.allclose(surv1, surv2)


def test_predict_treatment_unseen_category_hard_fail() -> None:
    model = _make_treatment_test_model()

    wide = pd.DataFrame(
        {
            "id": [1],
            "age_per10_centered": [0.0],
            "cci": [1],
            "sex": ["F"],
            "stage": ["IV"],
        }
    )

    with pytest.raises(ValueError, match="Unseen categorical levels for 'stage'"):
        predict_treatment_linear_predictor(wide, model, hard_fail=True)


def test_predict_treatment_unseen_category_soft_fail() -> None:
    model = _make_treatment_test_model()

    wide = pd.DataFrame(
        {
            "id": [1],
            "age_per10_centered": [0.0],
            "cci": [1],
            "sex": ["M"],
            "stage": ["IV"],
        }
    )

    mu, unseen = predict_treatment_linear_predictor(
        wide,
        model,
        hard_fail=False,
        return_unseen=True,
    )

    manual = np.log(120.0) + 0.20 * 1.0 + 0.05

    assert unseen == {"stage": ["IV"]}
    assert mu.shape == (1,)
    assert np.allclose(mu[0], manual)


def test_predict_treatment_survival_requires_positive_times() -> None:
    model = _make_treatment_test_model()
    wide = _make_prediction_df()

    with pytest.raises(ValueError, match="strictly positive"):
        predict_treatment_survival(wide, model, times=[0.0, 10.0])


def test_predict_treatment_quantile_requires_valid_p() -> None:
    model = _make_treatment_test_model()
    wide = _make_prediction_df()

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        predict_treatment_quantile(wide, model, p=0.0)

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        predict_treatment_quantile(wide, model, p=1.0)


def test_predict_treatment_shapes_for_single_subject() -> None:
    model = _make_treatment_test_model()
    wide = _make_prediction_df().iloc[[0]].copy()

    mu = predict_treatment_linear_predictor(wide, model)
    med = predict_treatment_median(wide, model)
    mean = predict_treatment_mean(wide, model)
    surv = predict_treatment_survival(wide, model, times=[30.0, 120.0])

    assert mu.shape == (1,)
    assert med.shape == (1,)
    assert mean.shape == (1,)
    assert surv.shape == (1, 2)