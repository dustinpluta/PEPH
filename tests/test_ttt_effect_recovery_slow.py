from __future__ import annotations

import numpy as np
import pytest

from peph.data.long import expand_long
from peph.model.fit import fit_peph
from peph.sim.ttt_effect import simulate_peph_ttt_effect_dataset


@pytest.mark.slow
def test_ttt_effect_recovery_nonspatial_ph() -> None:
    """
    Recovery test for the first TTT model:

        h(t) = h0(t) * exp(x beta + gamma * treated_td(t))

    We simulate data from that model, then fit the PH backend using treated_td
    as a long-form time-dependent covariate.

    The tolerance is intentionally moderate for stability.
    """
    breaks = [0.0, 30.0, 90.0, 180.0, 365.0, 730.0, 1825.0]
    nu_true = np.array([0.0040, 0.0030, 0.0024, 0.0018, 0.0013, 0.0010], dtype=float)

    beta_true = {
        "age_per10_centered": 0.10,
        "cci": 0.16,
        "tumor_size_log": 0.22,
        "ses": -0.08,
        "sexM": 0.05,
        "stageII": 0.30,
        "stageIII": 0.55,
    }
    gamma_true = -0.60

    wide = simulate_peph_ttt_effect_dataset(
        n=8000,
        breaks=breaks,
        nu=nu_true,
        beta=beta_true,
        gamma_treated=gamma_true,
        admin_censor=float(breaks[-1]),
        random_censor_rate=0.0007,
        max_treatment_time=365.0,
        seed=123,
        return_latent_truth=True,
    )

    long_df = expand_long(
        wide,
        id_col="id",
        time_col="time",
        event_col="event",
        x_cols=[
            "age_per10_centered",
            "cci",
            "tumor_size_log",
            "ses",
            "sex",
            "stage",
        ],
        breaks=breaks,
        cut_times_col="treatment_time",
        td_treatment_col="treatment_time",
        treated_td_col="treated_td",
    )

    fitted = fit_peph(
        long_train=long_df,
        breaks=breaks,
        x_numeric=["age_per10_centered", "cci", "tumor_size_log", "ses"],
        x_td_numeric=["treated_td"],
        x_categorical=["sex", "stage"],
        categorical_reference_levels={"sex": "F", "stage": "I"},
        n_train_subjects=int(wide["id"].nunique()),
        covariance="classical",
    )

    param_names = list(fitted.param_names)
    params = np.asarray(fitted.params, dtype=float)

    assert "treated_td" in param_names, "treated_td coefficient missing from fitted model"

    gamma_hat = float(params[param_names.index("treated_td")])

    # basic signal checks
    assert gamma_hat < 0.0, f"Expected negative treatment effect, got {gamma_hat:.4f}"

    # recovery tolerance: moderate but meaningful
    assert abs(gamma_hat - gamma_true) < 0.20, (
        f"treated_td estimate not close enough: "
        f"gamma_hat={gamma_hat:.4f}, gamma_true={gamma_true:.4f}"
    )

    # sanity checks on observed treatment process
    prop_treated_observed = float(np.mean(wide["treatment_time"].notna()))
    assert 0.2 < prop_treated_observed < 0.95

    # ensure treated person-time exists in long form
    prop_treated_rows = float(np.mean(long_df["treated_td"].to_numpy(dtype=float)))
    assert prop_treated_rows > 0.05