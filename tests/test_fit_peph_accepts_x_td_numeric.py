from __future__ import annotations

import pandas as pd

from peph.model.fit import fit_peph


def test_fit_peph_accepts_x_td_numeric() -> None:
    long_df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "k":  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "event":    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            "exposure": [30.0, 20.0, 30.0, 25.0, 30.0, 15.0, 30.0, 30.0, 30.0, 18.0, 30.0, 30.0],
            "age_per10_centered": [0.2, 0.2, -0.1, -0.1, 0.4, 0.4, -0.3, -0.3, 0.1, 0.1, -0.2, -0.2],
            "treated_td": [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
            "sex": ["F", "F", "M", "M", "F", "F", "M", "M", "F", "F", "M", "M"],
            "stage": ["II", "II", "III", "III", "II", "II", "III", "III", "II", "II", "III", "III"],
        }
    )

    fitted = fit_peph(
        long_train=long_df,
        breaks=[0.0, 30.0, 90.0],
        x_numeric=["age_per10_centered"],
        x_td_numeric=["treated_td"],
        x_categorical=["sex", "stage"],
        categorical_reference_levels={"sex": "F", "stage": "II"},
        n_train_subjects=6,
        covariance="classical",
    )

    assert "treated_td" in fitted.param_names
    assert fitted.encoding.x_td_numeric == ["treated_td"]