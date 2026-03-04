import numpy as np
import pandas as pd
import pytest

from peph.data.long import expand_long
from peph.model.fit import fit_peph
from peph.model.fit_leroux import fit_peph_leroux
from peph.model.predict import predict_risk
from peph.spatial.graph import build_graph_from_edge_list
from peph.spatial.weights import zip_weights_from_train_wide
from peph.sim.spatial import sample_leroux_u
from peph.sim.peph import simulate_peph_spatial_dataset


@pytest.mark.slow
def test_spatial_simulation_recovery_leroux_vs_ph() -> None:
    # --- small chain graph: 6 ZIPs ---
    zips = [f"Z{i}" for i in range(6)]
    edges = pd.DataFrame({"zip_u": zips[:-1], "zip_v": zips[1:]})
    graph = build_graph_from_edge_list(zips, edges)

    # Use SpatialGraph API
    W_csr = graph.W()
    G = graph.G
    W = W_csr.toarray().astype(float)  # small graph; dense OK for simulation
    D = np.diag(graph.degree().astype(float))

    rng = np.random.default_rng(123)

    # --- true Leroux hyperparameters ---
    rho_true = 0.75
    tau_true = 3.0

    comp = graph.component_ids()
    u_true = sample_leroux_u(
        W=W,
        D=D,
        tau=tau_true,
        rho=rho_true,
        rng=rng,
        component_ids=comp,
        weights=np.ones(G, dtype=float),
    )
    zip_to_u = {z: float(u_true[graph.zip_to_index[z]]) for z in zips}

    # --- PE baseline ---
    breaks = [0, 30, 90, 180, 365, 730, 1825]
    nu = np.array([0.0040, 0.0030, 0.0024, 0.0018, 0.0013, 0.0010], dtype=float)

    # --- covariates consistent with CRC setup ---
    x_numeric = ["age_per10_centered", "cci", "tumor_size_log", "ses"]
    x_categorical = ["sex", "stage"]
    cat_levels = {"sex": ["F", "M"], "stage": ["I", "II", "III"]}
    cat_ref = {"sex": "F", "stage": "I"}

    beta_true = {
        "age_per10_centered": 0.10,
        "cci": 0.18,
        "tumor_size_log": 0.25,
        "ses": -0.08,
        "sexM": 0.05,
        "stageII": 0.35,
        "stageIII": 0.60,
    }

    # --- simulate dataset ---
    df = simulate_peph_spatial_dataset(
        n=2500,
        breaks=breaks,
        nu=nu,
        beta=beta_true,
        x_numeric=x_numeric,
        x_categorical=x_categorical,
        cat_levels=cat_levels,
        cat_ref=cat_ref,
        zips=zips,
        zip_to_u=zip_to_u,
        admin_censor=1825.0,
        random_censor_rate=0.0,
        seed=777,
    )

    # subject split
    rng2 = np.random.default_rng(7777)
    ids = df["id"].to_numpy()
    rng2.shuffle(ids)
    n_train = int(0.7 * len(ids))
    train_ids = set(ids[:n_train].tolist())
    train = df[df["id"].isin(train_ids)].copy()
    test = df[~df["id"].isin(train_ids)].copy()

    # expand to long (include zip)
    x_cols_long = x_numeric + x_categorical + ["zip"]
    long_train = expand_long(
        train,
        id_col="id",
        time_col="time",
        event_col="event",
        x_cols=x_cols_long,
        breaks=breaks,
    )

    # --- fit PH (ignores frailty) ---
    ph = fit_peph(
        long_train,
        breaks=breaks,
        x_numeric=x_numeric,
        x_categorical=x_categorical,
        categorical_reference_levels=cat_ref,
        n_train_subjects=int(train["id"].nunique()),
        covariance="classical",
        cluster_col="id",
    )

    # --- fit Leroux ---
    _ = zip_weights_from_train_wide(  # ensures mapping works; centering uses this in your code
        train,
        area_col="zip",
        zip_to_index=graph.zip_to_index,
        allow_unseen_area=False,
    )

    leroux = fit_peph_leroux(
        long_train=long_train,
        train_wide=train,
        breaks=breaks,
        x_numeric=x_numeric,
        x_categorical=x_categorical,
        categorical_reference_levels=cat_ref,
        area_col="zip",
        graph=graph,
        allow_unseen_area=False,
        n_train_subjects=int(train["id"].nunique()),
        max_iter=80,
        ftol=1e-6,
        rho_clip=1e-6,
        q_jitter=1e-8,
    )

    spatial = leroux.__dict__.get("spatial", None)
    assert spatial is not None
    rho_hat = float(spatial["rho"])
    tau_hat = float(spatial["tau"])
    assert 0.0 < rho_hat < 1.0
    assert tau_hat > 0.0

    # With spatial signal present, rho_hat shouldn't collapse to ~0
    assert rho_hat > 0.15

    # --- beta recovery ---
    def _coef(model, name: str) -> float:
        names = list(model.param_names)
        idx = names.index(name)
        return float(model.params[idx])

    for key in ["age_per10_centered", "cci", "tumor_size_log", "ses"]:
        ph_est = _coef(ph, key)
        le_est = _coef(leroux, key)
        true = float(beta_true[key])

        assert abs(le_est - true) < 0.08
        assert abs(ph_est - true) < 0.15

    # --- simple Brier at horizon on known-status subset ---
    horizon = 365.0
    mask_known = (test["time"] > horizon) | ((test["event"] == 1) & (test["time"] <= horizon))
    test_k = test.loc[mask_known].copy()

    y = ((test_k["event"] == 1) & (test_k["time"] <= horizon)).astype(float).to_numpy()

    p_ph = predict_risk(test_k, ph, times=[horizon], frailty_mode="none").ravel()
    p_le = predict_risk(test_k, leroux, times=[horizon], frailty_mode="auto").ravel()

    b_ph = float(np.mean((y - p_ph) ** 2))
    b_le = float(np.mean((y - p_le) ** 2))

    assert b_le <= b_ph + 1e-3