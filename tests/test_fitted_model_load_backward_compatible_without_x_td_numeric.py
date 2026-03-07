from peph.model.result import FittedPEPHModel


def test_fitted_model_load_backward_compatible_without_x_td_numeric() -> None:
    d = {
        "breaks": [0.0, 30.0, 90.0],
        "interval_convention": "[a,b)",
        "encoding": {
            "x_numeric": ["x"],
            "x_categorical": ["sex"],
            "categorical_reference_levels": {"sex": "F"},
            "categorical_levels_seen": {"sex": ["F", "M"]},
            "x_expanded_cols": ["x", "sexM"],
        },
        "baseline_col_names": ["log_nu[0]", "log_nu[1]"],
        "x_col_names": ["x", "sexM"],
        "param_names": ["log_nu[0]", "log_nu[1]", "x", "sexM"],
        "params": [0.1, -0.2, 0.3, 0.4],
        "cov": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "nu": [1.105170185, 0.818730753],
        "fit_backend": "statsmodels_glm_poisson::classical",
        "n_train_subjects": 10,
        "n_train_long_rows": 25,
        "converged": True,
        "aic": 12.3,
        "deviance": 5.6,
    }

    model = FittedPEPHModel.from_json_dict(d)
    assert model.encoding.x_td_numeric == []