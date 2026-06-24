"""Command-line entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_run_config
from .runner import run_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QMM nuclear-matter workflow runner.")
    parser.add_argument("config", help="Path to a JSON configuration file.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_run_config(args.config)
    summary = run_workflow(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
