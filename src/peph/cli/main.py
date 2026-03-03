from __future__ import annotations

import argparse

from peph.config.schema import load_run_config
from peph.pipeline.run import run_pipeline


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
        out_dir = run_pipeline(cfg)
        print(f"[peph] Run complete. Artifacts at: {out_dir}")


if __name__ == "__main__":
    main()