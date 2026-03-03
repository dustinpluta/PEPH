import numpy as np
import pytest

from peph.data.long import expand_long
from peph.data.split import apply_split, train_test_split_subject
from peph.model.fit import fit_peph
from peph.sim import PHSimSpec, simulate_ph_wide


@pytest.mark.slow
def test_fit_peph_parameter_recovery() -> None:
    breaks = [0, 30, 90, 180, 365, 730, 1825]
    nu_true = np.array([0.0040, 0.0030, 0.0024, 0.0018, 0.0013, 0.0010], dtype=float)
    beta_true = {
        "age_per10_centered": 0.10,
        "cci": 0.18,
        "tumor_size_log": 0.25,
        "ses": -0.08,
        "sexM": 0.05,
        "stageII": 0.35,
        "stageIII": 0.70,
        "stageIV": 1.20,
    }

    spec = PHSimSpec(
        breaks=breaks,
        nu=nu_true.tolist(),
        beta=beta_true,
        seed=123,
        censoring_enabled=True,
        censoring_rate=0.0008,
    )

    df = simulate_ph_wide(20000, spec, include_debug_cols=False)

    # split
    split = train_test_split_subject(df, id_col="id", test_size=0.25, seed=999)
    train_wide, _ = apply_split(df, id_col="id", split=split)

    # long expand
    x_numeric = ["age_per10_centered", "cci", "tumor_size_log", "ses"]
    x_categorical = ["sex", "stage"]
    x_cols_all = x_numeric + x_categorical

    long_train = expand_long(
        train_wide,
        id_col="id",
        time_col="time",
        event_col="event",
        x_cols=x_cols_all,
        breaks=breaks,
    )

    # fit
    fitted = fit_peph(
        long_train,
        breaks=breaks,
        x_numeric=x_numeric,
        x_categorical=x_categorical,
        categorical_reference_levels={"sex": "F", "stage": "I"},
        max_iter=200,
        tol=1e-8,
        n_train_subjects=int(train_wide["id"].nunique()),
    )

    # Extract estimates
    K = len(breaks) - 1
    params = fitted.params_array()
    alpha_hat = params[:K]
    beta_hat = params[K:]

    # Map beta_hat to names
    name_to_est = dict(zip(fitted.x_col_names, beta_hat.tolist()))

    # Baseline hazard recovery: compare log hazards to reduce scale issues
    # (nu are small; log-scale is more stable)
    nu_hat = np.asarray(fitted.nu, dtype=float)
    assert nu_hat.shape == nu_true.shape

    # tolerances: these are practical for n~15k train; can tighten later
    # baseline: relative tolerance 20%
    rel_err = np.abs(nu_hat - nu_true) / nu_true
    assert float(np.max(rel_err)) < 0.20

    # betas: absolute tolerance 0.05–0.10 depending on coefficient size
    for nm, bt in beta_true.items():
        est = name_to_est.get(nm)
        assert est is not None, f"Missing estimated beta for '{nm}'"
        tol = 0.06 if abs(bt) <= 0.2 else 0.08
        assert abs(est - bt) < tol, f"{nm}: est={est:.4f}, true={bt:.4f}"