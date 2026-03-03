import pandas as pd
import pytest

from peph.model.result import FeatureEncoding, FittedPEPHModel
from peph.model.predict import predict_linear_predictor


def test_predict_hard_fail_unseen_category() -> None:
    enc = FeatureEncoding(
        x_numeric=["x1"],
        x_categorical=["sex"],
        categorical_reference_levels={"sex": "F"},
        categorical_levels_seen={"sex": ["F", "M"]},
        x_expanded_cols=["x1", "sexM"],
    )
    model = FittedPEPHModel(
        breaks=[0.0, 30.0, 90.0],
        interval_convention="[a,b)",
        encoding=enc,
        baseline_col_names=["log_nu[0]", "log_nu[1]"],
        x_col_names=["x1", "sexM"],
        param_names=["log_nu[0]", "log_nu[1]", "x1", "sexM"],
        params=[-5.0, -5.0, 0.1, 0.2],
        cov=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        nu=[0.01, 0.01],
        fit_backend="statsmodels_glm_poisson",
        n_train_subjects=1,
        n_train_long_rows=1,
    )

    df = pd.DataFrame({"x1": [0.0], "sex": ["X"]})
    with pytest.raises(ValueError):
        _ = predict_linear_predictor(model, df)