from __future__ import annotations

import numpy as np
import pytest

from peph.data.long import expand_long
from peph.model.fit import fit_peph
from peph.sim.ttt_effect import simulate_peph_ttt_effect_dataset


@pytest.mark.slow
def test_ttt_effect_recovery_nonspatial_ph_multiseed() -> None:
    """
    Multi-seed recovery test for the first TTT PH model:

        h(t) = h0(t) * exp(x beta + gamma * treated_td(t))

    This is stronger than the single-seed test because it checks recovery
    stability over repeated simulations.

    We assess:
      - correct sign across all seeds
      - small mean bias
      - reasonable average absolute error
      - at least 80% of seeds within 0.15 of gamma_true
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

    seeds = list(range(10))
    gamma_hats = []
    treated_observed_props = []
    treated_row_props = []
    event_rates = []

    for seed in seeds:
        wide = simulate_peph_ttt_effect_dataset(
            n=8000,
            breaks=breaks,
            nu=nu_true,
            beta=beta_true,
            gamma_treated=gamma_true,
            admin_censor=float(breaks[-1]),
            random_censor_rate=0.0007,
            max_treatment_time=365.0,
            seed=seed,
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
        gamma_hats.append(gamma_hat)

        treated_observed_props.append(float(wide["treatment_time"].notna().mean()))
        treated_row_props.append(float(long_df["treated_td"].mean()))
        event_rates.append(float(wide["event"].mean()))

    gamma_hats = np.asarray(gamma_hats, dtype=float)
    abs_errors = np.abs(gamma_hats - gamma_true)
    bias = gamma_hats - gamma_true

    # core recovery checks
    assert np.all(gamma_hats < 0.0), f"Expected all gamma_hat values to be negative, got {gamma_hats.tolist()}"

    assert abs(float(np.mean(bias))) < 0.08, (
        f"Mean bias too large: mean(gamma_hat - gamma_true) = {float(np.mean(bias)):.4f}"
    )

    assert float(np.mean(abs_errors)) < 0.12, (
        f"Mean absolute error too large: {float(np.mean(abs_errors)):.4f}"
    )

    assert float(np.mean(abs_errors < 0.15)) >= 0.80, (
        f"Fewer than 80% of seeds were within 0.15 of gamma_true. "
        f"Abs errors: {abs_errors.tolist()}"
    )

    # sanity checks on treatment/event process across seeds
    treated_observed_props = np.asarray(treated_observed_props, dtype=float)
    treated_row_props = np.asarray(treated_row_props, dtype=float)
    event_rates = np.asarray(event_rates, dtype=float)

    assert np.all((treated_observed_props > 0.20) & (treated_observed_props < 0.95)), (
        f"Observed treatment proportions out of range: {treated_observed_props.tolist()}"
    )
    assert np.all(treated_row_props > 0.05), (
        f"Treated row proportions too small: {treated_row_props.tolist()}"
    )
    assert np.all(event_rates > 0.0), f"Event rates should be positive, got {event_rates.tolist()}"