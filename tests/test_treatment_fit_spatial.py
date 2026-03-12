from __future__ import annotations

import numpy as np
import pandas as pd

from peph.treatment.fit import fit_treatment_lognormal_aft_map_leroux
from peph.treatment.result import FittedTreatmentAFTModel


def _simulate_spatial_aft_smoke_data(
    *,
    n: int = 800,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Simple smoke-test generator for the spatial treatment AFT fitter.

    This is not a recovery test. It just creates plausible treatment-time data
    with ZIP-level heterogeneity so we can verify the fitter runs and returns
    a spatial model object.
    """
    rng = np.random.default_rng(seed)

    age = rng.normal(0.0, 1.0, n)
    cci = rng.poisson(1.2, n)
    ses = rng.normal(0.0, 1.0, n)
    sex = rng.choice(["F", "M"], size=n, p=[0.55, 0.45])
    stage = rng.choice(["I", "II", "III"], size=n, p=[0.45, 0.35, 0.20])

    # Use ZIPs from the repo dataset
    zips_df = pd.read_csv("data/zips.csv")
    if "zip" in zips_df.columns:
        zips = zips_df["zip"].astype(str).tolist()
    else:
        zips = zips_df.iloc[:, 0].astype(str).tolist()

    # Use a subset for faster fitting
    zips = zips[:40]
    zip_vals = rng.choice(zips, size=n, replace=True)

    # simple synthetic ZIP effect, not necessarily graph-generated
    zip_unique = sorted(set(zips))
    zip_to_u = {z: float(rng.normal(0.0, 0.25)) for z in zip_unique}

    intercept = float(np.log(120.0))
    sigma = 0.45

    mu = (
        intercept
        + 0.08 * age
        + 0.10 * cci
        - 0.05 * ses
        + 0.04 * (sex == "M").astype(float)
        + 0.18 * (stage == "II").astype(float)
        + 0.35 * (stage == "III").astype(float)
        + np.array([zip_to_u[z] for z in zip_vals], dtype=float)
    )

    log_t = mu + sigma * rng.normal(size=n)
    t_true = np.exp(log_t)

    c = rng.exponential(scale=1.0 / 0.004, size=n)
    t_obs = np.minimum(t_true, c)
    event = (t_true <= c).astype(int)
    t_obs = np.maximum(t_obs, 1e-6)

    return pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "zip": zip_vals,
            "treatment_time_obs": t_obs,
            "treatment_event": event,
            "age_per10_centered": age,
            "cci": cci,
            "ses": ses,
            "sex": sex,
            "stage": stage,
        }
    )


def test_fit_treatment_lognormal_aft_map_leroux_smoke() -> None:
    df = _simulate_spatial_aft_smoke_data(n=800, seed=2026)

    fitted = fit_treatment_lognormal_aft_map_leroux(
        df,
        treatment_time_col="treatment_time_obs",
        treatment_event_col="treatment_event",
        x_numeric=["age_per10_centered", "cci", "ses"],
        x_categorical=["sex", "stage"],
        categorical_reference_levels={"sex": "F", "stage": "I"},
        area_col="zip",
        zips_path="data/zips.csv",
        edges_path="data/zip_adjacency.csv",
        edges_u_col="zip_i",
        edges_v_col="zip_j",
        max_iter=60,
        tol=1e-6,
        optimizer_method="L-BFGS-B",
        q_jitter=1e-8,
        rho_clip=1e-6,
        prior_logtau_sd=10.0,
        prior_rho_a=1.0,
        prior_rho_b=1.0,
    )

    assert isinstance(fitted, FittedTreatmentAFTModel)
    assert fitted.fit_backend == "lognormal_aft_map_leroux"
    assert fitted.n_train_subjects == len(df)
    assert fitted.sigma > 0.0
    assert np.isfinite(fitted.log_sigma)
    assert fitted.spatial is not None

    sp = fitted.spatial
    assert sp.type == "leroux"
    assert sp.area_col == "zip"
    assert len(sp.zips) > 0
    assert len(sp.u) == len(sp.zips)
    assert np.all(np.isfinite(np.asarray(sp.u, dtype=float)))
    assert np.isfinite(sp.rho)
    assert np.isfinite(sp.tau)
    assert 0.0 < sp.rho < 1.0
    assert sp.tau > 0.0

    # params/cov are for inferential parameters only, not full ZIP vector
    assert "rho" in fitted.param_names
    assert "log_tau" in fitted.param_names
    assert len(fitted.params) == len(fitted.param_names)

    cov = np.asarray(fitted.cov, dtype=float)
    assert cov.shape == (len(fitted.param_names), len(fitted.param_names))

    # optimizer metadata should exist
    assert sp.optimizer is not None
    assert "success" in sp.optimizer
    assert "fun" in sp.optimizer