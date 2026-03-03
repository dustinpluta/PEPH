import numpy as np
import pandas as pd

from peph.model.leroux_objective import LerouxHyperPriors, leroux_neg_log_posterior, pack_theta
from peph.spatial.graph import build_graph_from_edge_list


def test_leroux_objective_finite_at_init() -> None:
    # tiny graph
    zips = ["1", "2", "3"]
    edges = pd.DataFrame({"zip_u": ["1", "2"], "zip_v": ["2", "3"]})
    graph = build_graph_from_edge_list(zips, edges)

    # fake long components
    K, p, G = 2, 2, graph.G
    n = 5
    y = np.array([0, 1, 0, 0, 1], dtype=float)
    exposure = np.ones(n, dtype=float)
    k = np.array([0, 0, 1, 1, 1], dtype=int)
    X = np.random.default_rng(0).normal(size=(n, p))
    area = np.array([0, 1, 1, 2, 0], dtype=int)
    weights = np.array([2.0, 2.0, 1.0], dtype=float)

    alpha0 = np.zeros(K)
    beta0 = np.zeros(p)
    u0 = np.zeros(G)
    theta0 = pack_theta(alpha0, beta0, u0, 0.0, 0.0)

    val = leroux_neg_log_posterior(
        theta0,
        K=K,
        p=p,
        graph=graph,
        y=y,
        exposure=exposure,
        k=k,
        X=X,
        area_idx=area,
        weights=weights,
        rho_clip=1e-6,
        q_jitter=1e-8,
        priors=LerouxHyperPriors(),
    )
    assert np.isfinite(val)