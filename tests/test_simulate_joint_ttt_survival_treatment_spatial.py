from __future__ import annotations

import numpy as np
import pandas as pd

from peph.sim.joint_ttt_survival import simulate_joint_ttt_survival_dataset


def test_simulate_joint_ttt_survival_dataset_treatment_spatial_mode_none_has_zero_treatment_field() -> None:
    df = simulate_joint_ttt_survival_dataset(
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
        treatment_spatial_mode="none",
        treatment_tau_true=1.5,
        treatment_rho_true=0.75,
        admin_censor=1825.0,
        random_censor_rate=0.001,
        seed=111,
        return_latent_truth=True,
    )

    assert "u_treatment_true" in df.columns
    assert "treatment_spatial_mode_true" in df.columns
    assert "treatment_tau_true" in df.columns
    assert "treatment_rho_true" in df.columns

    assert (df["treatment_spatial_mode_true"] == "none").all()
    assert np.allclose(df["u_treatment_true"].to_numpy(dtype=float), 0.0)
    assert np.allclose(df["treatment_tau_true"].to_numpy(dtype=float), 0.0)
    assert np.allclose(df["treatment_rho_true"].to_numpy(dtype=float), 0.0)


def test_simulate_joint_ttt_survival_dataset_treatment_spatial_mode_leroux_has_nonzero_treatment_field() -> None:
    df = simulate_joint_ttt_survival_dataset(
        n=600,
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
        treatment_spatial_mode="leroux",
        treatment_tau_true=1.5,
        treatment_rho_true=0.75,
        admin_censor=1825.0,
        random_censor_rate=0.001,
        seed=222,
        return_latent_truth=True,
    )

    assert (df["treatment_spatial_mode_true"] == "leroux").all()

    u_treat = df["u_treatment_true"].to_numpy(dtype=float)
    assert np.all(np.isfinite(u_treat))
    assert not np.allclose(u_treat, 0.0)

    # Should be constant within ZIP because this is a ZIP-level field
    zip_u = df.groupby("zip")["u_treatment_true"].nunique()
    assert (zip_u == 1).all()

    # Survival and treatment spatial fields are separate in this simulator
    u_surv = df["u_true"].to_numpy(dtype=float)
    assert np.all(np.isfinite(u_surv))
    assert not np.allclose(u_surv, u_treat)

    # Metadata should reflect requested treatment spatial parameters
    assert np.allclose(df["treatment_tau_true"].to_numpy(dtype=float), 1.5)
    assert np.allclose(df["treatment_rho_true"].to_numpy(dtype=float), 0.75)


def test_simulate_joint_ttt_survival_dataset_treatment_spatial_mode_changes_treatment_times() -> None:
    common_kwargs = dict(
        n=800,
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
        treatment_tau_true=1.5,
        treatment_rho_true=0.75,
        admin_censor=1825.0,
        random_censor_rate=0.001,
        seed=333,
        return_latent_truth=True,
    )

    df_none = simulate_joint_ttt_survival_dataset(
        treatment_spatial_mode="none",
        **common_kwargs,
    )
    df_leroux = simulate_joint_ttt_survival_dataset(
        treatment_spatial_mode="leroux",
        **common_kwargs,
    )

    # Treatment-side spatial field should differ across modes
    assert np.allclose(df_none["u_treatment_true"].to_numpy(dtype=float), 0.0)
    assert not np.allclose(df_leroux["u_treatment_true"].to_numpy(dtype=float), 0.0)

    # Treatment times should change when treatment spatial field is introduced
    t_none = df_none["treatment_time_true"].to_numpy(dtype=float)
    t_leroux = df_leroux["treatment_time_true"].to_numpy(dtype=float)
    assert not np.allclose(t_none, t_leroux)

    # ZIP-level treatment field should be constant within ZIP in leroux mode
    zip_u = df_leroux.groupby("zip")["u_treatment_true"].nunique()
    assert (zip_u == 1).all()

    # Basic outputs should remain valid
    assert np.all(df_leroux["treatment_time_obs"].to_numpy(dtype=float) > 0.0)
    assert set(df_leroux["treatment_event"].unique()).issubset({0, 1})
    assert np.all(df_leroux["time"].to_numpy(dtype=float) > 0.0)
    assert set(df_leroux["event"].unique()).issubset({0, 1})

def test_simulate_joint_ttt_survival_dataset_invalid_treatment_spatial_mode_raises() -> None:
    try:
        simulate_joint_ttt_survival_dataset(
            n=100,
            breaks=[0.0, 30.0, 90.0],
            nu=[0.0018, 0.0014],
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
            treatment_spatial_mode="bad_mode",
            admin_censor=365.0,
            random_censor_rate=0.001,
            seed=444,
            return_latent_truth=False,
        )
    except ValueError as e:
        assert "treatment_spatial_mode" in str(e)
    else:
        raise AssertionError("Expected ValueError for invalid treatment_spatial_mode")