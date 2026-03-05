from __future__ import annotations

import numpy as np
import pandas as pd

from peph.pipeline.run import _morans_I
from peph.spatial.graph import build_graph_from_edge_list


def test_morans_I_sanity_trivial_chain_graph() -> None:
    # 5-node chain: 0-1-2-3-4
    zips = ["z0", "z1", "z2", "z3", "z4"]
    edges = pd.DataFrame(
        {
            "zip_u": ["z0", "z1", "z2", "z3"],
            "zip_v": ["z1", "z2", "z3", "z4"],
        }
    )
    graph = build_graph_from_edge_list(zips, edges, col_u="zip_u", col_v="zip_v")
    W = graph.W()

    # A simple structured signal (non-constant)
    u = np.array([0.0, 1.0, 2.0, 1.0, 0.0], dtype=float)

    res = _morans_I(u, W)

    assert np.isfinite(res["I"])
    assert np.isfinite(res["expected"])
    # variance may be small/negative depending on approximation; require it not nan
    assert not np.isnan(res["variance"])

    n = len(u)
    assert np.isclose(res["expected"], -1.0 / (n - 1))

    # For this simple case, Moran's I should be bounded reasonably.
    assert -1.0 <= res["I"] <= 1.0