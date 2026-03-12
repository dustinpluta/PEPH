from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peph.sim.spatial import sample_leroux_u
from peph.spatial.graph import build_graph_from_edge_list
from peph.treatment.fit import fit_treatment_lognormal_aft_map_leroux


def _load_zip_subset_and_graph(
    *,
    zips_path: str,
    edges_path: str,
    edges_u_col: str,
    edges_v_col: str,
    n_zips: int,
):
    zdf = pd.read_csv(zips_path)
    if "zip" in zdf.columns:
        zips_all = zdf["zip"].astype(str).tolist()
    else:
        zips_all = zdf.iloc[:, 0].astype(str).tolist()

    zips_subset = zips_all[:n_zips]
    edges_df = pd.read_csv(edges_path)

    graph = build_graph_from_edge_list(
        zips=zips_subset,
        edges_df=edges_df,
        col_u=edges_u_col,
        col_v=edges_v_col,
    )
    return [str(z) for z in graph.zips], graph


def _simulate_spatial_treatment_aft_data(
    *,
    n: int,
    zips_path: str,
    edges_path: str,
    edges_u_col: str = "zip_i",
    edges_v_col: str = "zip_j",
    n_zips: int = 40,
    beta_true: dict[str, float],
    treatment_intercept_true: float,
    sigma_true: float,
    rho_true: float,
    tau_true: float,
    seed: int = 123,
    censor_rate: float | None = None,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """
    Simulate wide treatment data under a spatial log-normal AFT model:

        log(T_i) = intercept + x_i' beta + u_zip(i) + sigma * eps_i
    """
    rng = np.random.default_rng(seed)

    graph_zips, graph = _load_zip_subset_and_graph(
        zips_path=zips_path,
        edges_path=edges_path,
        edges_u_col=edges_u_col,
        edges_v_col=edges_v_col,
        n_zips=n_zips,
    )

    W = graph.W().toarray().astype(float)
    D = np.diag(W.sum(axis=1))

    u_true = sample_leroux_u(
        W=W,
        D=D,
        tau=float(tau_true),
        rho=float(rho_true),
        rng=rng,
        q_jitter=1e-10,
        component_ids=graph.component_ids(),
        weights=None,
    )
    zip_to_u = {str(z): float(u_true[i]) for i, z in enumerate(graph_zips)}

    age = rng.normal(0.0, 1.0, n)
    cci = rng.poisson(1.2, n)
    ses = rng.normal(0.0, 1.0, n)
    sex = rng.choice(["F", "M"], size=n, p=[0.55, 0.45])
    stage = rng.choice(["I", "II", "III"], size=n, p=[0.45, 0.35, 0.20])
    zip_vals = rng.choice(graph_zips, size=n, replace=True)

    mu = (
        float(treatment_intercept_true)
        + beta_true["age_per10_centered"] * age
        + beta_true["cci"] * cci
        + beta_true["ses"] * ses
        + beta_true["sexM"] * (sex == "M").astype(float)
        + beta_true["stageII"] * (stage == "II").astype(float)
        + beta_true["stageIII"] * (stage == "III").astype(float)
        + np.array([zip_to_u[z] for z in zip_vals], dtype=float)
    )

    log_t = mu + float(sigma_true) * rng.normal(size=n)
    t_true = np.exp(log_t)

    if censor_rate is None:
        t_obs = t_true
        event = np.ones(n, dtype=int)
    else:
        c = rng.exponential(scale=1.0 / censor_rate, size=n)
        t_obs = np.minimum(t_true, c)
        event = (t_true <= c).astype(int)

    t_obs = np.maximum(t_obs, 1e-6)

    df = pd.DataFrame(
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
    return df, graph_zips, np.asarray(u_true, dtype=float)


def _align_u_hat(fitted_zips: list[str], u_hat: np.ndarray, target_zips: list[str]) -> np.ndarray:
    idx = {str(z): i for i, z in enumerate(fitted_zips)}
    return np.array([u_hat[idx[z]] for z in target_zips], dtype=float)


@pytest.mark.slow
def test_fit_treatment_lognormal_aft_map_leroux_recovery_multiseed() -> None:
    beta_true = {
        "age_per10_centered": 0.10,
        "cci": 0.12,
        "ses": -0.06,
        "sexM": 0.04,
        "stageII": 0.18,
        "stageIII": 0.32,
    }
    treatment_intercept_true = float(np.log(120.0))
    sigma_true = 0.45
    rho_true = 0.80
    tau_true = 2.0

    seeds = [2027, 2028, 2029, 2030, 2031]

    rows: list[dict[str, float]] = []

    for seed in seeds:
        df, graph_zips, u_true = _simulate_spatial_treatment_aft_data(
            n=3500,
            zips_path="data/zips.csv",
            edges_path="data/zip_adjacency.csv",
            edges_u_col="zip_i",
            edges_v_col="zip_j",
            n_zips=40,
            beta_true=beta_true,
            treatment_intercept_true=treatment_intercept_true,
            sigma_true=sigma_true,
            rho_true=rho_true,
            tau_true=tau_true,
            seed=seed,
            censor_rate=None,
        )

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
            max_iter=220,
            tol=1e-6,
            optimizer_method="L-BFGS-B",
            q_jitter=1e-8,
            rho_clip=1e-6,
            prior_logtau_sd=10.0,
            prior_rho_a=1.0,
            prior_rho_b=1.0,
        )

        assert fitted.converged
        assert fitted.spatial is not None

        beta_hat = dict(zip(fitted.x_col_names, fitted.beta))
        u_hat = np.asarray(fitted.spatial.u, dtype=float)
        u_hat_aligned = _align_u_hat(
            [str(z) for z in fitted.spatial.zips],
            u_hat,
            graph_zips,
        )
        corr = float(np.corrcoef(u_true, u_hat_aligned)[0, 1])

        rows.append(
            {
                "seed": float(seed),
                "intercept_hat": float(beta_hat["Intercept"]),
                "age_hat": float(beta_hat["age_per10_centered"]),
                "cci_hat": float(beta_hat["cci"]),
                "ses_hat": float(beta_hat["ses"]),
                "sexM_hat": float(beta_hat["sexM"]),
                "stageII_hat": float(beta_hat["stageII"]),
                "stageIII_hat": float(beta_hat["stageIII"]),
                "sigma_hat": float(fitted.sigma),
                "rho_hat": float(fitted.spatial.rho),
                "tau_hat": float(fitted.spatial.tau),
                "u_corr": corr,
            }
        )

    res = pd.DataFrame(rows)

    # Per-seed sanity
    assert (res["rho_hat"] > 0.0).all()
    assert (res["rho_hat"] < 1.0).all()
    assert (res["tau_hat"] > 0.0).all()
    assert np.isfinite(res["u_corr"]).all()

    # Median recovery across seeds for fixed effects
    assert abs(float(res["intercept_hat"].median()) - treatment_intercept_true) < 0.20
    assert abs(float(res["age_hat"].median()) - beta_true["age_per10_centered"]) < 0.06
    assert abs(float(res["cci_hat"].median()) - beta_true["cci"]) < 0.06
    assert abs(float(res["ses_hat"].median()) - beta_true["ses"]) < 0.06
    assert abs(float(res["sexM_hat"].median()) - beta_true["sexM"]) < 0.07
    assert abs(float(res["stageII_hat"].median()) - beta_true["stageII"]) < 0.09
    assert abs(float(res["stageIII_hat"].median()) - beta_true["stageIII"]) < 0.10

    # Scale recovery across seeds
    assert abs(float(res["sigma_hat"].median()) - sigma_true) < 0.08

    # Spatial field recovery across seeds
    assert float(res["u_corr"].median()) > 0.50
    assert float(res["u_corr"].min()) > 0.30

    # Spatial parameter recovery: broad tolerances
    assert abs(float(res["rho_hat"].median()) - rho_true) < 0.3

    tau_ratio_med = float(res["tau_hat"].median() / tau_true)
    assert 0.35 < tau_ratio_med < 3.0