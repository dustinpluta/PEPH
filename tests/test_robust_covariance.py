import numpy as np
import pandas as pd

from peph.data.long import expand_long
from peph.model.fit import fit_peph


def test_cluster_robust_covariance_runs() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "time": [10, 40, 100, 200, 500, 900],
            "event": [1, 0, 1, 0, 1, 0],
            "age_per10_centered": [0.0, 0.2, -0.1, 0.4, -0.3, 0.1],
            "cci": [0, 1, 2, 1, 3, 0],
            "tumor_size_log": [0.1, 0.2, 0.0, -0.2, 0.3, 0.1],
            "ses": [0.0, 1.0, -1.0, 0.5, -0.2, 0.3],
            "sex": ["F", "M", "F", "M", "F", "M"],
            "stage": ["I", "II", "III", "IV", "II", "I"],
        }
    )

    breaks = [0, 30, 90, 180, 365, 730, 1825]
    x_numeric = ["age_per10_centered", "cci", "tumor_size_log", "ses"]
    x_categorical = ["sex", "stage"]

    long_df = expand_long(
        df,
        id_col="id",
        time_col="time",
        event_col="event",
        x_cols=x_numeric + x_categorical,
        breaks=breaks,
    )

    fitted = fit_peph(
        long_df,
        breaks=breaks,
        x_numeric=x_numeric,
        x_categorical=x_categorical,
        categorical_reference_levels={"sex": "F", "stage": "I"},
        n_train_subjects=int(df["id"].nunique()),
        covariance="cluster_id",
        cluster_col="id",
    )

    cov = fitted.cov_array()
    assert np.isfinite(cov).all()
    assert cov.shape[0] == cov.shape[1]
    assert np.min(np.diag(cov)) > -1e-10