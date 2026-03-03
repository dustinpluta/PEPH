from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from peph.utils.json import read_json, write_json


@dataclass(frozen=True)
class SpatialGraph:
    """
    Spatial graph for areal units (ZIPs).

    - zips: ordered list of node labels (strings)
    - edges: undirected edge list in node-index space (i,j) with i<j
    - W: symmetric adjacency matrix (sparse CSR), zero diagonal
    - components: component id per node, computed from W (undirected)
    """
    zips: List[str]
    edges: List[Tuple[int, int]]
    W_csr_data: Dict[str, object]  # serialized sparse (data/indices/indptr/shape)
    components: List[int]

    @property
    def G(self) -> int:
        return len(self.zips)

    @property
    def zip_to_index(self) -> Dict[str, int]:
        return {z: i for i, z in enumerate(self.zips)}

    def W(self) -> sp.csr_matrix:
        d = self.W_csr_data
        return sp.csr_matrix(
            (np.asarray(d["data"]), np.asarray(d["indices"]), np.asarray(d["indptr"])),
            shape=tuple(d["shape"]),
        )

    def degree(self) -> np.ndarray:
        W = self.W()
        return np.asarray(W.sum(axis=1)).ravel()

    def leroux_Q(self, rho: float) -> sp.csr_matrix:
        """
        Leroux precision (up to scale):
          Q(rho) = (1-rho) I + rho (D - W)

        Proper for 0 <= rho < 1.
        """
        if not (0.0 <= rho < 1.0):
            raise ValueError("rho must satisfy 0 <= rho < 1")

        G = self.G
        W = self.W()
        deg = self.degree()
        D = sp.diags(deg, format="csr")
        I = sp.identity(G, format="csr")
        Q = (1.0 - rho) * I + rho * (D - W)
        return Q.tocsr()

    def component_ids(self) -> np.ndarray:
        return np.asarray(self.components, dtype=int)

    def n_components(self) -> int:
        return int(np.max(self.component_ids()) + 1) if self.G > 0 else 0

    def save(self, path: str | Path) -> None:
        write_json(str(path), asdict(self))

    @classmethod
    def load(cls, path: str | Path) -> "SpatialGraph":
        d = read_json(str(path))
        return cls(**d)


def _serialize_csr(W: sp.csr_matrix) -> Dict[str, object]:
    W = W.tocsr()
    return {
        "data": W.data.astype(float).tolist(),
        "indices": W.indices.astype(int).tolist(),
        "indptr": W.indptr.astype(int).tolist(),
        "shape": list(W.shape),
    }


def build_graph_from_edge_list(
    zips: List[str],
    edges_df: pd.DataFrame,
    *,
    col_u: str = "zip_u",
    col_v: str = "zip_v",
) -> SpatialGraph:
    """
    Build an undirected graph from a ZIP universe + edge list.

    edges_df must contain columns [col_u, col_v] with ZIP labels.
    Self-loops are rejected.
    """
    if len(set(zips)) != len(zips):
        raise ValueError("ZIP universe contains duplicates")
    zip_to_idx = {z: i for i, z in enumerate(zips)}

    if col_u not in edges_df.columns or col_v not in edges_df.columns:
        raise ValueError(f"edges_df must contain columns '{col_u}' and '{col_v}'")

    pairs: List[Tuple[int, int]] = []
    for u, v in zip(edges_df[col_u].astype(str), edges_df[col_v].astype(str)):
        if u not in zip_to_idx or v not in zip_to_idx:
            raise ValueError(f"Edge contains ZIP not in universe: ({u},{v})")
        i = zip_to_idx[u]
        j = zip_to_idx[v]
        if i == j:
            raise ValueError(f"Self-loop edge not allowed: {u}")
        if i > j:
            i, j = j, i
        pairs.append((i, j))

    # de-duplicate
    pairs = sorted(set(pairs))

    # build symmetric adjacency
    G = len(zips)
    row = np.array([i for (i, j) in pairs] + [j for (i, j) in pairs], dtype=int)
    col = np.array([j for (i, j) in pairs] + [i for (i, j) in pairs], dtype=int)
    data = np.ones_like(row, dtype=float)

    W = sp.csr_matrix((data, (row, col)), shape=(G, G))
    W.setdiag(0.0)
    W.eliminate_zeros()

    # components
    n_comp, labels = connected_components(W, directed=False, return_labels=True)

    return SpatialGraph(
        zips=list(map(str, zips)),
        edges=pairs,
        W_csr_data=_serialize_csr(W),
        components=labels.astype(int).tolist(),
    )