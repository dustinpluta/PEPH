import numpy as np
import pandas as pd
import pytest

from peph.model.predict import predict_linear_predictor, predict_risk
from peph.model.result import FeatureEncoding, FittedPEPHModel


def _make_tiny_model_with_spatial() -> FittedPEPHModel:
    # Minimal model: K=1 baseline + 1 numeric covariate
    enc = FeatureEncoding(
        x_numeric=["x"],
        x_categorical=[],
        categorical_reference_levels={},
        categorical_levels_seen={},
        x_expanded_cols=["x"],
    )

    m = FittedPEPHModel(
        breaks=[0.0, 10.0],
        interval_convention="[a,b)",
        encoding=enc,
        baseline_col_names=["log_nu[0]"],
        x_col_names=["x"],
        param_names=["log_nu[0]", "x"],
        params=[np.log(0.1), 1.0],   # nu=0.1, beta=1.0
        cov=[[1.0, 0.0], [0.0, 1.0]],
        nu=[0.1],
        fit_backend="map_leroux",
        n_train_subjects=2,
        n_train_long_rows=2,
        converged=True,
        aic=None,
        deviance=None,
        llf=None,
    )

    # Attach spatial info like PR5
    m.__dict__["spatial"] = {
        "type": "leroux",
        "area_col": "zip",
        "zips": ["A", "B"],
        "u": [0.5, -0.5],
        "tau": 1.0,
        "rho": 0.5,
        "optimizer": {"success": True},
        "graph": {"G": 2, "n_components": 1},
    }
    return m


def test_predict_linear_predictor_includes_frailty_conditional() -> None:
    model = _make_tiny_model_with_spatial()
    df = pd.DataFrame({"x": [0.0, 0.0], "zip": ["A", "B"]})

    eta = predict_linear_predictor(df, model, frailty_mode="conditional")
    assert np.allclose(eta, np.array([0.5, -0.5]))


def test_predict_unseen_zip_hard_fails() -> None:
    model = _make_tiny_model_with_spatial()
    df = pd.DataFrame({"x": [0.0], "zip": ["C"]})

    with pytest.raises(ValueError, match="Unseen area values"):
        _ = predict_linear_predictor(df, model, frailty_mode="conditional", allow_unseen_area=False)


def test_predict_risk_differs_by_zip() -> None:
    model = _make_tiny_model_with_spatial()
    df = pd.DataFrame({"x": [0.0, 0.0], "zip": ["A", "B"]})

    risk = predict_risk(df, model, times=[10.0], frailty_mode="conditional").ravel()
    assert risk[0] != risk[1]