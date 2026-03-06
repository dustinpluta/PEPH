from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    model_json: Optional[Path] = None
    metrics_json: Optional[Path] = None
    coef_table: Optional[Path] = None
    baseline_table: Optional[Path] = None
    frailty_table: Optional[Path] = None
    frailty_summary: Optional[Path] = None
    spatial_autocorr: Optional[Path] = None
    inference_json: Optional[Path] = None

    predictions_dir: Optional[Path] = None
    plots_dir: Optional[Path] = None
    tables_dir: Optional[Path] = None


def discover_run_artifacts(run_dir: str | Path) -> RunArtifacts:
    rd = Path(run_dir)
    if not rd.exists() or not rd.is_dir():
        raise FileNotFoundError(f"Run directory not found: {rd}")

    def f(name: str) -> Optional[Path]:
        p = rd / name
        return p if p.exists() else None

    preds = rd / "predictions"
    plots = rd / "plots"
    tables = rd / "tables"

    return RunArtifacts(
        run_dir=rd,
        model_json=f("model.json"),
        metrics_json=f("metrics.json"),
        coef_table=f("coef_table.parquet"),
        baseline_table=f("baseline_table.parquet"),
        frailty_table=f("frailty_table.parquet"),
        frailty_summary=f("frailty_summary.json"),
        spatial_autocorr=f("spatial_autocorr.json"),
        inference_json=f("inference.json"),
        predictions_dir=preds if preds.exists() else None,
        plots_dir=plots if plots.exists() else None,
        tables_dir=tables if tables.exists() else None,
    )