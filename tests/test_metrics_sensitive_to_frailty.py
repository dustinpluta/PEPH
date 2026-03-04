import numpy as np
import pandas as pd

from peph.model.predict import predict_risk
from peph.model.result import FeatureEncoding, FittedPEPHModel


def _brier_score_binary(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((y - p) ** 2))


def test_metrics_change_when_frailty_included() -> None:
    # Model: baseline + x, plus spatial frailty makes two ZIP groups differ
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
        params=[float(np.log(0.1)), 0.0],  # beta=0 so only frailty drives differences
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
        "u": [1.0, -1.0],  # strong separation
        "tau": 1.0,
        "rho": 0.5,
        "optimizer": {"success": True},
        "graph": {"G": 2, "n_components": 1},
    }

    # Create pseudo test set: same x, different ZIPs, and events more common in A than B
    n = 200
    df = pd.DataFrame(
        {
            "x": np.zeros(n),
            "zip": np.array(["A"] * (n // 2) + ["B"] * (n // 2)),
        }
    )
    # Binary label: higher event probability in A
    y = np.array([1] * (n // 2) + [0] * (n // 2), dtype=float)

    # Frailty OFF (marginal/none): both groups identical predictions
    p_none = predict_risk(df, m, times=[10.0], frailty_mode="none").ravel()
    # Frailty ON (conditional): groups differ
    p_cond = predict_risk(df, m, times=[10.0], frailty_mode="conditional").ravel()

    b_none = _brier_score_binary(y, p_none)
    b_cond = _brier_score_binary(y, p_cond)

    # With correct directionality, conditional should be better (lower Brier) here
    assert b_cond < b_none