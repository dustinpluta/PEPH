from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from peph.sim.peph_ttt import simulate_peph_spatial_ttt_dataset


def _make_ring_with_second_neighbors(zips):
    """
    Same adjacency used in the simulator.

    Ring + second-neighbor edges to ensure good spatial connectivity.
    """
    zips = list(zips)
    G = len(zips)

    W = np.zeros((G, G), dtype=int)

    for i in range(G):
        neighbors = {
            (i - 1) % G,
            (i + 1) % G,
            (i - 2) % G,
            (i + 2) % G,
        }

        for j in neighbors:
            if i != j:
                W[i, j] = 1
                W[j, i] = 1

    np.fill_diagonal(W, 0)
    return W


def _adjacency_to_edge_list(W, zips):
    """
    Convert adjacency matrix to edge list (undirected, unique).
    """
    edges = []

    G = len(zips)

    for i in range(G):
        for j in range(i + 1, G):
            if W[i, j] == 1:
                edges.append((zips[i], zips[j]))

    return pd.DataFrame(edges, columns=["zip_i", "zip_j"])


def main():

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    # ----------------------------
    # Simulate dataset
    # ----------------------------

    df = simulate_peph_spatial_ttt_dataset(
        n=10000,
        seed=123,
    )

    data_path = out_dir / "simulated_seer_crc_ttt.csv"
    df.to_csv(data_path, index=False)

    print(f"dataset written → {data_path}")

    # ----------------------------
    # ZIP universe
    # ----------------------------

    zips = sorted(df["zip"].unique())

    zip_df = pd.DataFrame(
        {
            "zip": zips,
            "state": ["GA"] * len(zips),
        }
    )

    zips_path = out_dir / "zips.csv"
    zip_df.to_csv(zips_path, index=False)

    print(f"zip universe written → {zips_path}")

    # ----------------------------
    # Adjacency edges
    # ----------------------------

    W = _make_ring_with_second_neighbors(zips)

    edges = _adjacency_to_edge_list(W, zips)

    edges_path = out_dir / "zip_adjacency.csv"
    edges.to_csv(edges_path, index=False)

    print(f"adjacency written → {edges_path}")

    # ----------------------------
    # Quick sanity check
    # ----------------------------

    print("\nsummary:")
    print(df[["time", "event", "treatment_time"]].describe())

    print("\nmissing treatment breakdown:")
    print(df["ttt_missing_reason"].value_counts(dropna=False))


if __name__ == "__main__":
    main()