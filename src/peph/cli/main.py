from __future__ import annotations

import argparse

from peph.config.schema import load_run_config
from peph.pipeline.run import run_pipeline
from peph.report.cli import build_report_parser


def main() -> None:
    parser = argparse.ArgumentParser(prog="peph")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = subparsers.add_parser("run", help="Run the PE-PH pipeline from a YAML config.")
    p_run.add_argument("--config", required=True, help="Path to YAML config file.")
    p_run.add_argument("--override", action="append", default=None, help="Optional overrides (repeatable).")
    p_run.set_defaults(_cmd="run")

    # --- report ---
    build_report_parser(subparsers)

    args = parser.parse_args()

    if getattr(args, "_cmd", None) == "run":
        cfg = load_run_config(args.config, overrides=args.override)
        out_dir = run_pipeline(cfg)
        print(out_dir)
        return

    if hasattr(args, "func"):
        args.func(args)
        return

    raise RuntimeError("No command handler found.")


if __name__ == "__main__":
    main()