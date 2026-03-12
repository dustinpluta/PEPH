from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from peph.sim.ttt_effect_spatial import simulate_peph_spatial_ttt_effect_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    breaks = [0.0, 30.0, 90.0, 180.0, 365.0, 730.0, 1825.0]

    nu = np.array(
        [0.0018, 0.0014, 0.0011, 0.0009, 0.0007, 0.0005],
        dtype=float,
    )

    beta = {
        "age_per10_centered": 0.06,
        "cci": 0.10,
        "tumor_size_log": 0.14,
        "ses": -0.05,
        "sexM": 0.03,
        "stageII": 0.18,
        "stageIII": 0.30,
    }

    gamma_treated = -0.30

    df = simulate_peph_spatial_ttt_effect_dataset(
        n=20000,
        breaks=breaks,
        nu=nu,
        beta=beta,
        gamma_treated=gamma_treated,
        zips_path=str(DATA_DIR / "zips.csv"),
        edges_path=str(DATA_DIR / "zip_adjacency.csv"),
        edges_u_col="zip_i",
        edges_v_col="zip_j",
        tau_true=1.0,
        rho_true=0.85,
        admin_censor=float(breaks[-1]),
        random_censor_rate=0.00015,
        max_treatment_time=90.0,
        seed=123,
        return_latent_truth=False,
    )

    data_path = DATA_DIR / "ga_spatial_ttt_dataset.csv"
    meta_path = DATA_DIR / "ga_spatial_ttt_metadata.json"

    df.to_csv(data_path, index=False)

    metadata = {
        "n_subjects": int(len(df)),
        "event_rate": float(df["event"].mean()),
        "treated_fraction": float(df["treatment_time"].notna().mean()),
        "breaks": breaks,
        "nu": nu.tolist(),
        "beta": beta,
        "gamma_treated": gamma_treated,
        "tau_true": 1.0,
        "rho_true": 0.85,
        "random_censor_rate": 0.00015,
        "seed": 123,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote dataset: {data_path}")
    print(f"Wrote metadata: {meta_path}")
    print(f"Event rate: {metadata['event_rate']:.3f}")
    print(f"Treated fraction: {metadata['treated_fraction']:.3f}")


if __name__ == "__main__":
    main()