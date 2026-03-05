from __future__ import annotations

from pathlib import Path

from peph.config.schema import load_run_config
from peph.pipeline.run import run_pipeline


def test_pr9_spatial_artifacts_skipped_for_ph_backend(tmp_path: Path) -> None:
    """
    Ensure PR9 spatial diagnostics are skipped for non-spatial PH runs.

    Uses the leroux small fixture but overrides the entire 'fit' block to force
    statsmodels_glm_poisson and sets spatial=None.
    """
    cfg_path = "tests/fixtures/run_leroux_small.yml"

    cfg = load_run_config(
        cfg_path,
        overrides={
            "output.root_dir": tmp_path.as_posix(),
            # Shallow override: replace whole fit block with schema-compliant values
            "fit": {
                "backend": "statsmodels_glm_poisson",
                "covariance": "classical",
                # The remaining fields may be required by the FitConfig schema even if unused.
                "leroux_max_iter": 5,
                "leroux_ftol": 1e-6,
                "rho_clip": 0.999,      # schema expects float
                "q_jitter": 1e-6,
                "prior_logtau_sd": 1.0,
                "prior_rho_a": 1.5,
                "prior_rho_b": 1.5,
            },
            "spatial": None,
        },
    )

    out_dir = run_pipeline(cfg)
    assert out_dir.exists()

    # PR9 spatial artifacts should NOT exist in PH-only run
    assert not (out_dir / "frailty_table.parquet").exists()
    assert not (out_dir / "frailty_summary.json").exists()
    assert not (out_dir / "spatial_autocorr.json").exists()

    assert not (out_dir / "plots" / "frailty_caterpillar.png").exists()
    assert not (out_dir / "plots" / "morans_scatter_u.png").exists()

    tables_dir = out_dir / "tables"
    if tables_dir.exists():
        assert len(list(tables_dir.glob("calibration_by_frailty_decile_t*.parquet"))) == 0

    plots_dir = out_dir / "plots"
    if plots_dir.exists():
        assert len(list(plots_dir.glob("calibration_by_frailty_decile_t*.png"))) == 0