from __future__ import annotations

import numpy as np
import pytest

from peph.data.long import expand_long
from peph.model.fit_dispatch import fit_model_dispatch
from peph.sim.ttt_effect_spatial import simulate_peph_spatial_ttt_effect_dataset


@pytest.mark.slow
def test_ttt_effect_recovery_spatial_leroux_multiseed() -> None:
    """
    Multiseed recovery test for treated_td under a Leroux spatial frailty model.

    Data-generating model:
        h_i(t) = h0(t) * exp(x beta + u_zip(i) + gamma * treated_td(t))

    Fit model:
        backend = map_leroux

    We check:
      - treated_td estimate has correct sign for all seeds
      - mean bias is reasonably small
      - average absolute error is controlled
      - at least 70% of seeds are within 0.20 of gamma_true

    Thresholds are a bit looser than the nonspatial test because spatial recovery
    adds another layer of estimation noise.
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

    zips_path = "data/zips.csv"
    edges_path = "data/zip_adjacency.csv"

    seeds = list(range(10))
    gamma_hats = []
    treated_observed_props = []
    treated_row_props = []
    event_rates = []

    for seed in seeds:
        wide = simulate_peph_spatial_ttt_effect_dataset(
            n=8000,
            breaks=breaks,
            nu=nu_true,
            beta=beta_true,
            gamma_treated=gamma_true,
            zips_path=zips_path,
            edges_path=edges_path,
            edges_u_col="zip_i",
            edges_v_col="zip_j",
            tau_true=2.0,
            rho_true=0.85,
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
                "zip",
            ],
            breaks=breaks,
            cut_times_col="treatment_time",
            td_treatment_col="treatment_time",
            treated_td_col="treated_td",
        )

        fitted = fit_model_dispatch(
            backend="map_leroux",
            long_train=long_df,
            train_wide=wide,
            breaks=breaks,
            x_numeric=["age_per10_centered", "cci", "tumor_size_log", "ses"],
            x_td_numeric=["treated_td"],
            x_categorical=["sex", "stage"],
            categorical_reference_levels={"sex": "F", "stage": "I"},
            n_train_subjects=int(wide["id"].nunique()),
            covariance="classical",
            spatial_area_col="zip",
            spatial_zips_path=zips_path,
            spatial_edges_path=edges_path,
            spatial_edges_u_col="zip_i",
            spatial_edges_v_col="zip_j",
            allow_unseen_area=False,
            leroux_max_iter=200,
            leroux_ftol=1e-7,
            rho_clip=1e-6,
            q_jitter=1e-8,
            prior_logtau_sd=10.0,
            prior_rho_a=1.0,
            prior_rho_b=1.0,
        )

        param_names = list(fitted.param_names)
        params = np.asarray(fitted.params, dtype=float)

        assert "treated_td" in param_names, "treated_td coefficient missing from fitted Leroux model"

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

    assert abs(float(np.mean(bias))) < 0.10, (
        f"Mean bias too large under spatial Leroux fit: {float(np.mean(bias)):.4f}"
    )

    assert float(np.mean(abs_errors)) < 0.16, (
        f"Mean absolute error too large under spatial Leroux fit: {float(np.mean(abs_errors)):.4f}"
    )

    assert float(np.mean(abs_errors < 0.20)) >= 0.70, (
        f"Fewer than 70% of seeds were within 0.20 of gamma_true. "
        f"Abs errors: {abs_errors.tolist()}"
    )

    # sanity checks on treatment / event process
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