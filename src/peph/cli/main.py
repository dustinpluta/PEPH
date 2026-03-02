from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from peph.config.schema import load_run_config


def _write_yaml(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(prog="peph")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run the PE-PH pipeline from a YAML config")
    run.add_argument("--config", required=True, help="Path to run.yml")
    run.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values, e.g. --override split.seed=999",
    )

    args = parser.parse_args()

    if args.cmd == "run":
        cfg = load_run_config(args.config, overrides=args.override)

        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = Path(cfg.output.root_dir) / f"{cfg.run_name}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save resolved config (as dict)
        _write_yaml(out_dir / "config_resolved.yml", cfg.model_dump())

        # Pipeline will be wired in next PR
        print(f"[peph] Wrote resolved config to: {out_dir / 'config_resolved.yml'}")
        print("[peph] Pipeline execution not yet implemented (PR1 will add split/long/fit/metrics).")