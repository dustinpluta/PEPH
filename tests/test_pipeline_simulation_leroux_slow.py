from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peph.config.schema import load_run_config
from peph.pipeline.run import run_pipeline
from peph.utils.json import read_json

from peph.spatial.graph import build_graph_from_edge_list
from peph.sim.spatial import sample_leroux_u
from peph.sim.peph import simulate_peph_spatial_dataset


@pytest.mark.slow
def test_pipeline_simulation_leroux_end_to_end_recovery(tmp_path: Path) -> None:
    """
    End-to-end Leroux simulation verification:

      1) Build SpatialGraph from the same zips/edges used by run_leroux_small.yml.
      2) Sample u_true ~ Leroux(tau_true, rho_true).
      3) Simulate a wide dataset with eta = X beta + u_true[zip].
      4) Run the full pipeline with backend=map_leroux.
      5) Assert PR9 spatial artifacts exist and u_hat correlates with u_true.

    Thresholds are conservative to reduce flakiness.
    """
    fixture_cfg_path = "tests/fixtures/run_leroux_small.yml"
    cfg0 = load_run_config(fixture_cfg_path)

    if cfg0.spatial is None:
        raise AssertionError("Fixture run_leroux_small.yml must include a spatial block")

    # --- graph inputs from fixture ---
    zips_path = cfg0.spatial.zips_path
    edges_path = cfg0.spatial.edges_path
    col_u = cfg0.spatial.edges_u_col
    col_v = cfg0.spatial.edges_v_col

    # Load zip universe file (supports either a 'zip' column or single-column format)
    zdf = pd.read_csv(zips_path)
    if "zip" in zdf.columns:
        zips = zdf["zip"].astype(str).tolist()
    elif zdf.shape[1] == 1:
        zips = zdf.iloc[:, 0].astype(str).tolist()
    else:
        raise AssertionError("ZIP universe file must have a 'zip' column or be single-column")

    edges_df = pd.read_csv(edges_path)
    graph = build_graph_from_edge_list(zips, edges_df, col_u=col_u, col_v=col_v)

    # Dense W/D for sampling u_true (OK for this *small* fixture graph)
    W = graph.W().toarray().astype(float)
    deg = W.sum(axis=1)
    D = np.diag(deg)

    rng = np.random.default_rng(123)

    # --- sample spatial surface u_true ---
    tau_true = 2.0
    rho_true = 0.85

    u_true = sample_leroux_u(
        W=W,
        D=D,
        tau=tau_true,
        rho=rho_true,
        rng=rng,
        q_jitter=1e-10,
        component_ids=graph.component_ids(),
        weights=None,
    )
    zip_to_u = {str(z): float(u_true[i]) for i, z in enumerate(graph.zips)}

    # --- simulate wide dataset (match fixture schema) ---
    breaks = list(map(float, cfg0.time.breaks))
    K = len(breaks) - 1

    # Baseline hazards (per day)
    if K != 6:
        nu = np.full(K, 0.002, dtype=float)
    else:
        nu = np.array([0.0040, 0.0030, 0.0024, 0.0018, 0.0013, 0.0010], dtype=float)

    x_numeric = list(cfg0.data_schema.x_numeric)
    x_categorical = list(cfg0.data_schema.x_categorical)
    cat_ref = dict(cfg0.data_schema.categorical_reference_levels)

    # IMPORTANT: match observed fixture category levels (leroux_small.csv has stage in {I, II, III})
    cat_levels = {
        "sex": ["F", "M"],
        "stage": ["I", "II", "III"],
    }
    for c in x_categorical:
        if c not in cat_levels:
            raise AssertionError(f"Simulation test needs cat_levels for categorical '{c}'.")
        if c not in cat_ref:
            raise AssertionError(f"Simulation test needs categorical_reference_levels for '{c}' in config.")

    # Effects: numeric as-is; categoricals as f"{col}{lvl}" excluding reference
    beta_true = {
        "age_per10_centered": 0.10,
        "cci": 0.18,
        "tumor_size_log": 0.25,
        "ses": -0.08,
        "sexM": 0.05,
        "stageII": 0.35,
        "stageIII": 0.55,
    }

    n = 6000  # slow integration test size

    df_wide = simulate_peph_spatial_dataset(
        n=n,
        breaks=breaks,
        nu=nu,
        beta=beta_true,
        x_numeric=x_numeric,
        x_categorical=x_categorical,
        cat_levels=cat_levels,
        cat_ref=cat_ref,
        zips=[str(z) for z in graph.zips],
        zip_to_u=zip_to_u,
        admin_censor=float(breaks[-1]),
        random_censor_rate=0.0008,
        seed=1,
    )

    # Ensure simulated area column matches config expectation
    area_col = cfg0.spatial.area_col
    if area_col != "zip":
        df_wide = df_wide.rename(columns={"zip": area_col})

    sim_csv = tmp_path / "sim_leroux.csv"
    df_wide.to_csv(sim_csv, index=False)

    # --- run pipeline on simulated CSV ---
    cfg = load_run_config(
        fixture_cfg_path,
        overrides={
            "data": {"path": sim_csv.as_posix(), "format": "csv"},
            "output": {"root_dir": tmp_path.as_posix()},
        },
    )

    out_dir = run_pipeline(cfg)
    assert out_dir.exists()

    # --- PR9 spatial artifacts exist ---
    assert (out_dir / "frailty_table.parquet").exists()
    assert (out_dir / "frailty_summary.json").exists()
    assert (out_dir / "spatial_autocorr.json").exists()
    assert (out_dir / "plots" / "frailty_caterpillar.png").exists()
    assert (out_dir / "plots" / "morans_scatter_u.png").exists()

    # --- u recovery ---
    frailty = pd.read_parquet(out_dir / "frailty_table.parquet")
    u_hat_map = dict(zip(frailty["zip"].astype(str), frailty["u_hat"].astype(float)))

    z_graph = [str(z) for z in graph.zips]
    missing = [z for z in z_graph if z not in u_hat_map]
    assert not missing, f"Missing zips in frailty_table: {missing[:5]}"

    u_hat = np.array([u_hat_map[z] for z in z_graph], dtype=float)

    corr = float(np.corrcoef(u_true, u_hat)[0, 1])
    assert corr > 0.35, f"u recovery correlation too low: corr={corr:.3f}"

    # --- rho_hat sanity (very conservative) ---
    fs = read_json(str(out_dir / "frailty_summary.json"))
    rho_hat = float(fs["rho_hat"])
    assert rho_hat > 0.20, f"rho_hat unexpectedly low: {rho_hat:.3f}"