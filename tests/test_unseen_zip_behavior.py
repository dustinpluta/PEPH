import numpy as np
import pandas as pd
import pytest

from peph.model.predict import predict_linear_predictor
from peph.model.result import FeatureEncoding, FittedPEPHModel


def _model_with_spatial() -> FittedPEPHModel:
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
        params=[float(np.log(0.1)), 0.0],
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


def test_unseen_zip_hard_fail() -> None:
    m = _model_with_spatial()
    df = pd.DataFrame({"x": [0.0], "zip": ["C"]})

    with pytest.raises(ValueError, match="Unseen area values"):
        _ = predict_linear_predictor(df, m, frailty_mode="conditional", allow_unseen_area=False)


def test_unseen_zip_allow_override_sets_zero() -> None:
    m = _model_with_spatial()
    df = pd.DataFrame({"x": [0.0], "zip": ["C"]})

    eta = predict_linear_predictor(df, m, frailty_mode="conditional", allow_unseen_area=True)
    assert np.isfinite(eta).all()
    # with beta=0 and u default 0 for unseen, eta should be 0
    assert float(eta[0]) == 0.0