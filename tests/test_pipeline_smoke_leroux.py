import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.slow
def test_pipeline_smoke_leroux_subprocess() -> None:
    cfg = Path("tests/fixtures/run_leroux_small.yml")
    assert cfg.exists(), "Missing fixture config tests/fixtures/run_leroux_small.yml"

    out_dir = Path("data/dev/test_smoke_leroux")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # Try likely runnable modules (update/extend based on your repo)
    candidate_modules = [
        "peph.cli",
        "peph.cli.main",
        "peph.pipeline.run_pipeline",
        "peph.scripts.run_pipeline",
        "peph.scripts.run",
    ]

    last_err = None
    for mod in candidate_modules:
        cmd = ["python", "-m", mod, "run", "--config", str(cfg)]
        completed = subprocess.run(cmd, capture_output=True, text=True)

        if completed.returncode == 0:
            # Success
            assert out_dir.exists(), "Expected out_dir to be created"
            jsons = list(out_dir.glob("**/*.json"))
            assert jsons, "Expected at least one JSON artifact in out_dir"
            return

        # If module doesn't exist, python returns a specific message; keep trying
        if "No module named" in completed.stderr:
            last_err = completed.stderr
            continue

        # Module exists but pipeline failed: hard fail and show stderr
        raise AssertionError(
            f"Pipeline failed when running module '{mod}'.\n"
            f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
        )

    raise AssertionError(
        "Could not find a runnable pipeline module among candidates.\n"
        f"Last error:\n{last_err}"
    )