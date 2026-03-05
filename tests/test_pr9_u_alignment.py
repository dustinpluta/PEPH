from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peph.spatial.graph import build_graph_from_edge_list


def _align_u_to_graph_zips(graph_zips: list[str], sp_zips: list[str], u: np.ndarray) -> np.ndarray:
    """
    Reference implementation of the alignment logic used in pipeline PR9:
      - sp_zips gives the order of u
      - graph_zips is the desired order
    """
    u = np.asarray(u, dtype=float).ravel()
    sp_zips = [str(z) for z in sp_zips]
    graph_zips = [str(z) for z in graph_zips]

    if len(sp_zips) != u.size:
        raise ValueError(f"Mismatch: len(sp_zips)={len(sp_zips)} vs u.size={u.size}")

    u_map = {z: u[i] for i, z in enumerate(sp_zips)}
    missing = [z for z in graph_zips if z not in u_map]
    if missing:
        raise ValueError(
            f"Graph contains ZIPs not present in fitted spatial.zips (n_missing={len(missing)}). "
            f"Example: {missing[:5]}"
        )
    return np.array([u_map[z] for z in graph_zips], dtype=float)


def test_pr9_u_alignment_permutation_roundtrip() -> None:
    # Graph ZIP order
    zips_graph = ["10001", "10002", "10003", "10004", "10005"]

    # Build a simple connected graph on these zips (chain)
    edges = pd.DataFrame(
        {
            "zip_u": ["10001", "10002", "10003", "10004"],
            "zip_v": ["10002", "10003", "10004", "10005"],
        }
    )
    graph = build_graph_from_edge_list(zips_graph, edges, col_u="zip_u", col_v="zip_v")

    # Fitted spatial provides u in a different (permuted) ZIP order
    sp_zips = ["10003", "10001", "10005", "10002", "10004"]
    # Define u in sp_zips order so we can check exact mapping
    u_sp = np.array([3.0, 1.0, 5.0, 2.0, 4.0], dtype=float)

    u_aligned = _align_u_to_graph_zips(graph.zips, sp_zips, u_sp)

    # Expect u in graph order: 10001->1, 10002->2, 10003->3, 10004->4, 10005->5
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    assert np.allclose(u_aligned, expected)


def test_pr9_u_alignment_raises_on_missing_zip() -> None:
    # Graph ZIP order includes one extra zip not present in sp_zips
    zips_graph = ["10001", "10002", "10003", "99999"]

    # Simple edges (still valid wrt zips_graph)
    edges = pd.DataFrame({"zip_u": ["10001", "10002"], "zip_v": ["10002", "10003"]})
    graph = build_graph_from_edge_list(zips_graph, edges, col_u="zip_u", col_v="zip_v")

    sp_zips = ["10001", "10002", "10003"]
    u_sp = np.array([1.0, 2.0, 3.0], dtype=float)

    with pytest.raises(ValueError, match=r"Graph contains ZIPs not present"):
        _align_u_to_graph_zips(graph.zips, sp_zips, u_sp)