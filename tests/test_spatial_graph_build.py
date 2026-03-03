import pandas as pd

from peph.spatial.graph import build_graph_from_edge_list


def test_build_graph_components() -> None:
    zips = ["29001", "29002", "29003", "99999"]  # last is isolated
    edges = pd.DataFrame({"zip_u": ["29001", "29002"], "zip_v": ["29002", "29003"]})

    g = build_graph_from_edge_list(zips, edges)
    assert g.G == 4
    assert g.n_components() == 2  # chain component + isolated