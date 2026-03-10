from __future__ import annotations

import numpy as np
import pandas as pd

from peph.sim.joint_ttt_survival import simulate_joint_ttt_survival_dataset


def test_simulate_joint_ttt_survival_dataset_invariants() -> None:
    df = simulate_joint_ttt_survival_dataset(
        n=1000,
        breaks=[0.0, 30.0, 90.0, 180.0, 365.0, 730.0, 1825.0],
        nu=[0.0018, 0.0014, 0.0011, 0.0009, 0.0007, 0.0005],
        beta_survival={
            "age_per10_centered": 0.06,
            "cci": 0.10,
            "tumor_size_log": 0.14,
            "ses": -0.05,
            "sexM": 0.03,
            "stageII": 0.18,
            "stageIII": 0.30,
        },
        gamma_treated=-0.30,
        beta_treatment={
            "age_per10_centered": 0.04,
            "cci": 0.08,
            "tumor_size_log": 0.06,
            "ses": -0.04,
            "sexM": 0.02,
            "stageII": 0.10,
            "stageIII": 0.22,
        },
        sigma_treatment=0.45,
        treatment_intercept=float(np.log(120.0)),
        zips_path="data/zips.csv",
        edges_path="data/zip_adjacency.csv",
        edges_u_col="zip_i",
        edges_v_col="zip_j",
        tau_true=1.0,
        rho_true=0.85,
        admin_censor=1825.0,
        random_censor_rate=0.001,
        seed=123,
        return_latent_truth=False,
    )

    required_cols = {
        "id",
        "zip",
        "age_per10_centered",
        "cci",
        "tumor_size_log",
        "ses",
        "sex",
        "stage",
        "treatment_time",
        "treatment_time_obs",
        "treatment_event",
        "time",
        "event",
    }
    assert required_cols.issubset(df.columns)

    assert len(df) == 1000
    assert df["id"].nunique() == 1000

    assert np.all(np.isfinite(df["time"].to_numpy(dtype=float)))
    assert np.all(df["time"].to_numpy(dtype=float) > 0.0)
    assert np.all(np.isfinite(df["treatment_time_obs"].to_numpy(dtype=float)))
    assert np.all(df["treatment_time_obs"].to_numpy(dtype=float) > 0.0)

    assert set(df["event"].unique()).issubset({0, 1})
    assert set(df["treatment_event"].unique()).issubset({0, 1})

    treated = df["treatment_event"] == 1
    censored_before_treatment = df["treatment_event"] == 0

    # If treatment observed, survival-model treatment_time should be present
    assert df.loc[treated, "treatment_time"].notna().all()

    # If treatment censored before occurrence, survival-model treatment_time should be missing
    assert df.loc[censored_before_treatment, "treatment_time"].isna().all()

    # When treatment observed, treatment_time_obs should match treatment_time
    assert np.allclose(
        df.loc[treated, "treatment_time_obs"].to_numpy(dtype=float),
        df.loc[treated, "treatment_time"].to_numpy(dtype=float),
    )

    # Treatment censoring time cannot exceed observed survival/censoring time
    assert np.all(
        df["treatment_time_obs"].to_numpy(dtype=float)
        <= df["time"].to_numpy(dtype=float) + 1e-10
    )

    # If death observed before treatment, then treatment must be censored for treatment model
    death_before_treatment = (
        (df["event"] == 1)
        & df["treatment_time"].isna()
    )
    assert (df.loc[death_before_treatment, "treatment_event"] == 0).all()

    # Basic category sanity checks
    assert set(df["sex"].astype(str).unique()).issubset({"F", "M"})
    assert set(df["stage"].astype(str).unique()).issubset({"I", "II", "III"})
    assert df["zip"].astype(str).nunique() > 1


def test_simulate_joint_ttt_survival_dataset_latent_truth_columns() -> None:
    df = simulate_joint_ttt_survival_dataset(
        n=200,
        breaks=[0.0, 30.0, 90.0, 180.0, 365.0, 730.0, 1825.0],
        nu=[0.0018, 0.0014, 0.0011, 0.0009, 0.0007, 0.0005],
        beta_survival={
            "age_per10_centered": 0.06,
            "cci": 0.10,
            "tumor_size_log": 0.14,
            "ses": -0.05,
            "sexM": 0.03,
            "stageII": 0.18,
            "stageIII": 0.30,
        },
        gamma_treated=-0.30,
        beta_treatment={
            "age_per10_centered": 0.04,
            "cci": 0.08,
            "tumor_size_log": 0.06,
            "ses": -0.04,
            "sexM": 0.02,
            "stageII": 0.10,
            "stageIII": 0.22,
        },
        sigma_treatment=0.45,
        treatment_intercept=float(np.log(120.0)),
        zips_path="data/zips.csv",
        edges_path="data/zip_adjacency.csv",
        edges_u_col="zip_i",
        edges_v_col="zip_j",
        tau_true=1.0,
        rho_true=0.85,
        admin_censor=1825.0,
        random_censor_rate=0.001,
        seed=456,
        return_latent_truth=True,
    )

    latent_cols = {
        "treatment_time_true",
        "survival_time_true",
        "censor_time",
        "u_true",
    }
    assert latent_cols.issubset(df.columns)

    assert np.all(np.isfinite(df["treatment_time_true"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(df["survival_time_true"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(df["censor_time"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(df["u_true"].to_numpy(dtype=float)))

    assert np.all(df["treatment_time_true"].to_numpy(dtype=float) > 0.0)
    assert np.all(df["survival_time_true"].to_numpy(dtype=float) > 0.0)
    assert np.all(df["censor_time"].to_numpy(dtype=float) > 0.0)

    treated = df["treatment_event"] == 1
    untreated = df["treatment_event"] == 0

    # If treatment observed, true treatment time must be no later than treatment censoring time
    assert np.all(
        df.loc[treated, "treatment_time_true"].to_numpy(dtype=float)
        <= df.loc[treated, "treatment_time_obs"].to_numpy(dtype=float) + 1e-10
    )

    # If treatment censored, observed treatment-model time is censoring before treatment
    assert np.all(
        df.loc[untreated, "treatment_time_true"].to_numpy(dtype=float)
        >= df.loc[untreated, "treatment_time_obs"].to_numpy(dtype=float) - 1e-10
    )


def test_simulate_joint_ttt_survival_dataset_reproducible_with_seed() -> None:
    kwargs = dict(
        n=300,
        breaks=[0.0, 30.0, 90.0, 180.0, 365.0, 730.0, 1825.0],
        nu=[0.0018, 0.0014, 0.0011, 0.0009, 0.0007, 0.0005],
        beta_survival={
            "age_per10_centered": 0.06,
            "cci": 0.10,
            "tumor_size_log": 0.14,
            "ses": -0.05,
            "sexM": 0.03,
            "stageII": 0.18,
            "stageIII": 0.30,
        },
        gamma_treated=-0.30,
        beta_treatment={
            "age_per10_centered": 0.04,
            "cci": 0.08,
            "tumor_size_log": 0.06,
            "ses": -0.04,
            "sexM": 0.02,
            "stageII": 0.10,
            "stageIII": 0.22,
        },
        sigma_treatment=0.45,
        treatment_intercept=float(np.log(120.0)),
        zips_path="data/zips.csv",
        edges_path="data/zip_adjacency.csv",
        edges_u_col="zip_i",
        edges_v_col="zip_j",
        tau_true=1.0,
        rho_true=0.85,
        admin_censor=1825.0,
        random_censor_rate=0.001,
        seed=789,
        return_latent_truth=True,
    )

    df1 = simulate_joint_ttt_survival_dataset(**kwargs)
    df2 = simulate_joint_ttt_survival_dataset(**kwargs)

    pd.testing.assert_frame_equal(df1, df2)