from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_zips(zips_path: Path):
    zdf = pd.read_csv(zips_path)

    if "zip" in zdf.columns:
        zips = zdf["zip"].astype(str).tolist()
    else:
        zips = zdf.iloc[:, 0].astype(str).tolist()

    return zips


def load_edges(edges_path: Path):
    edges_df = pd.read_csv(edges_path)

    if {"zip_i", "zip_j"}.issubset(edges_df.columns):
        col_u, col_v = "zip_i", "zip_j"
    elif {"zip_u", "zip_v"}.issubset(edges_df.columns):
        col_u, col_v = "zip_u", "zip_v"
    else:
        col_u, col_v = edges_df.columns[:2]

    edges_df[col_u] = edges_df[col_u].astype(str)
    edges_df[col_v] = edges_df[col_v].astype(str)

    return edges_df, col_u, col_v


def build_W(zips, edges_df, col_u, col_v):

    n = len(zips)
    zip_index = {z: i for i, z in enumerate(zips)}

    W = np.zeros((n, n))

    for _, r in edges_df.iterrows():

        if r[col_u] not in zip_index:
            continue

        if r[col_v] not in zip_index:
            continue

        i = zip_index[r[col_u]]
        j = zip_index[r[col_v]]

        W[i, j] = 1
        W[j, i] = 1

    return W


def morans_I(u, W):

    n = len(u)

    u = np.asarray(u)
    u_bar = np.mean(u)

    u_center = u - u_bar

    S0 = W.sum()

    numerator = np.sum(W * np.outer(u_center, u_center))
    denominator = np.sum(u_center ** 2)

    I = (n / S0) * (numerator / denominator)

    return float(I)


def permutation_test(u, W, n_perm=1000):

    rng = np.random.default_rng(123)

    I_obs = morans_I(u, W)

    perm_vals = []

    for _ in range(n_perm):
        u_perm = rng.permutation(u)
        perm_vals.append(morans_I(u_perm, W))

    perm_vals = np.array(perm_vals)

    p_val = np.mean(perm_vals >= I_obs)

    return I_obs, perm_vals, p_val


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--zips", required=True)
    parser.add_argument("--edges", required=True)

    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    frailty_path = run_dir / "frailty_table.parquet"

    if not frailty_path.exists():
        raise FileNotFoundError("frailty_table.parquet not found")

    frailty_df = pd.read_parquet(frailty_path)

    zips = load_zips(Path(args.zips))
    edges_df, col_u, col_v = load_edges(Path(args.edges))

    W = build_W(zips, edges_df, col_u, col_v)

    zip_to_u = dict(zip(frailty_df["zip"].astype(str), frailty_df["u_hat"]))

    u = np.array([zip_to_u[z] for z in zips])

    I_obs, perm_vals, p_val = permutation_test(u, W)

    print("\n==============================")
    print("MORAN'S I SPATIAL DIAGNOSTIC")
    print("==============================")

    print(f"\nObserved Moran's I: {I_obs:.4f}")

    print("\nPermutation test:")
    print(f"p-value: {p_val:.4f}")

    print("\nNull distribution summary:")
    print(f"mean: {perm_vals.mean():.4f}")
    print(f"std : {perm_vals.std():.4f}")

    print("\nInterpretation:")

    if I_obs > 0 and p_val < 0.05:
        print("Positive spatial autocorrelation detected (expected).")

    elif I_obs > 0:
        print("Positive autocorrelation but weak evidence.")

    else:
        print("No positive spatial structure detected.")

    edges = pd.read_csv("data/zip_adjacency.csv")

    # Ensure ZIP codes are strings
    edges["zip_i"] = edges["zip_i"].astype(str)
    edges["zip_j"] = edges["zip_j"].astype(str)

    frailty_df["zip"] = frailty_df["zip"].astype(str)

    merged = edges.merge(
        frailty_df[["zip", "u_hat"]],
        left_on="zip_i",
        right_on="zip",
        how="left"
    ).rename(columns={"u_hat": "u_i"}).drop(columns="zip")

    merged = merged.merge(
        frailty_df[["zip", "u_hat"]],
        left_on="zip_j",
        right_on="zip",
        how="left"
    ).rename(columns={"u_hat": "u_j"}).drop(columns="zip")

    neighbor_corr = np.corrcoef(merged["u_i"], merged["u_j"])[0, 1]

    print("\nNeighbor frailty correlation:", round(neighbor_corr, 4))

if __name__ == "__main__":
    main()