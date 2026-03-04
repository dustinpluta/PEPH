import numpy as np
import pandas as pd
import pytest

from peph.data.long import expand_long
from peph.model.fit_leroux import fit_peph_leroux
from peph.spatial.graph import build_graph_from_edge_list
from peph.spatial.weights import zip_weights_from_train_wide


def _component_weighted_means(u: np.ndarray, comp: np.ndarray, w: np.ndarray) -> dict[int, float]:
    out: dict[int, float] = {}
    n_comp = int(comp.max() + 1) if comp.size else 0
    for c in range(n_comp):
        idx = np.where(comp == c)[0]
        if idx.size == 0:
            continue
        denom = float(np.sum(w[idx]))
        if denom <= 0:
            out[c] = float(np.mean(u[idx]))
        else:
            out[c] = float(np.sum(w[idx] * u[idx]) / denom)
    return out


def test_map_leroux_integration_tiny() -> None:
    # --- tiny spatial graph: chain 3 ZIPs (connected) ---
    zips = ["29001", "29002", "29003"]
    edges = pd.DataFrame({"zip_u": ["29001", "29002"], "zip_v": ["29002", "29003"]})
    graph = build_graph_from_edge_list(zips, edges)

    # --- tiny wide dataset ---
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame(
        {
            "id": np.arange(n, dtype=int),
            "time": rng.integers(low=5, high=400, size=n).astype(float),
            "event": rng.binomial(1, 0.6, size=n).astype(int),
            "age_per10_centered": rng.normal(size=n),
            "cci": rng.integers(0, 4, size=n).astype(float),
            "tumor_size_log": rng.normal(size=n),
            "ses": rng.normal(size=n),
            "sex": rng.choice(["F", "M"], size=n),
            "stage": rng.choice(["I", "II", "III"], size=n),
            "zip": rng.choice(zips, size=n),
        }
    )

    breaks = [0, 30, 90, 180, 365, 730, 1825]
    x_numeric = ["age_per10_centered", "cci", "tumor_size_log", "ses"]
    x_categorical = ["sex", "stage"]
    cat_ref = {"sex": "F", "stage": "I"}

    # --- expand to long form (training == all rows here) ---
    long_df = expand_long(
        df,
        id_col="id",
        time_col="time",
        event_col="event",
        x_cols=x_numeric + x_categorical + ["zip"],
        breaks=breaks,
    )

    # --- fit leroux with PH init; keep iterations small for "fast integration" ---
    fitted = fit_peph_leroux(
        long_train=long_df,
        train_wide=df,
        breaks=breaks,
        x_numeric=x_numeric,
        x_categorical=x_categorical,
        categorical_reference_levels=cat_ref,
        area_col="zip",
        graph=graph,
        allow_unseen_area=False,
        n_train_subjects=int(df["id"].nunique()),
        max_iter=40,
        ftol=1e-6,
        rho_clip=1e-6,
        q_jitter=1e-8,
    )

    # spatial payload is attached via __dict__ in PR5
    spatial = fitted.__dict__.get("spatial", None)
    assert spatial is not None
    assert spatial["type"] == "leroux"
    assert spatial["area_col"] == "zip"

    rho = float(spatial["rho"])
    tau = float(spatial["tau"])
    u = np.asarray(spatial["u"], dtype=float)

    assert 0.0 < rho < 1.0
    assert tau > 0.0
    assert np.isfinite(u).all()
    assert u.shape[0] == graph.G

    # check weighted centering by component using subject counts
    w = zip_weights_from_train_wide(df, area_col="zip", zip_to_index=graph.zip_to_index, allow_unseen_area=False)
    comp = graph.component_ids()
    means = _component_weighted_means(u, comp, w)
    for c, m in means.items():
        assert abs(m) < 1e-8