import numpy as np

from peph.model.result import FeatureEncoding, FittedPEPHModel


def test_model_json_roundtrip(tmp_path) -> None:
    enc = FeatureEncoding(
        x_numeric=["x1"],
        x_categorical=["c1"],
        categorical_reference_levels={"c1": "A"},
        categorical_levels_seen={"c1": ["A", "B"]},
        x_expanded_cols=["x1", "c1B"],
    )
    m = FittedPEPHModel(
        breaks=[0.0, 1.0, 2.0],
        interval_convention="[a,b)",
        encoding=enc,
        baseline_col_names=["log_nu[0]", "log_nu[1]"],
        x_col_names=["x1", "c1B"],
        param_names=["log_nu[0]", "log_nu[1]", "x1", "c1B"],
        params=[0.0, -0.1, 0.2, 0.3],
        cov=np.eye(4).tolist(),
        nu=[1.0, 0.9],
        fit_backend="statsmodels_glm_poisson",
        n_train_subjects=10,
        n_train_long_rows=20,
        converged=True,
        aic=1.0,
        deviance=2.0,
    )

    p = tmp_path / "model.json"
    m.save(str(p))
    m2 = FittedPEPHModel.load(str(p))

    assert m2.breaks == m.breaks
    assert m2.encoding.categorical_levels_seen == m.encoding.categorical_levels_seen
    assert np.allclose(m2.params_array(), m.params_array())
    assert np.allclose(m2.cov_array(), m.cov_array())