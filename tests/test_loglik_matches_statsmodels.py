import numpy as np
import pandas as pd
import statsmodels.api as sm

from peph.data.long import expand_long
from peph.model.design import build_design_long_train
from peph.model.loglik import ph_loglik_poisson_trick


def test_ph_loglik_matches_statsmodels_on_long_data() -> None:
    breaks = [0, 30, 90, 180, 365, 730, 1825]
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

    y, X_full, offset, info = build_design_long_train(
        long_df,
        x_numeric=x_numeric,
        x_categorical=x_categorical,
        categorical_reference_levels={"sex": "F", "stage": "I"},
        K=len(breaks) - 1,
        eps_offset=1e-12,
    )

    # Fit statsmodels to get params
    model = sm.GLM(y, X_full, family=sm.families.Poisson(), offset=offset)
    res = model.fit(maxiter=200, tol=1e-8)
    params = np.asarray(res.params, dtype=float)

    K = len(breaks) - 1
    alpha = params[:K]
    beta = params[K:]

    # Pull out components needed by internal loglik:
    # - k and exposure are in long_df
    k = long_df["k"].to_numpy(dtype=int)
    exposure = long_df["exposure"].to_numpy(dtype=float)

    # X_fixed should match the fixed-effect block in X_full
    # (design.py should already track this ordering via info.x_col_names)
    # Here we take last p columns as fixed block.
    p = len(info.x_col_names)
    X_fixed = np.asarray(X_full[:, -p:], dtype=float)

    ll_internal = ph_loglik_poisson_trick(
        alpha=alpha, beta=beta, y=y, exposure=exposure, k=k, X=X_fixed
    )

    # statsmodels llf includes same constants for Poisson with y in {0,1} (log(y!)=0)
    ll_sm = float(res.llf)

    assert abs(ll_internal - ll_sm) < 1e-6