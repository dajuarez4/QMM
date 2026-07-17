from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_command(args: list[str]) -> None:
    completed = subprocess.run(args, cwd=ROOT, text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    python = str(ROOT / ".venv" / "bin" / "python")
    run_command(
        [
            python,
            "scripts/run_asymmetric_ev_dense_10point_suite.py",
            "--models",
            "clausius",
            "clausius_cs",
            "clausius_tvm",
            "--manifest-only",
        ]
    )
    run_command(
        [
            python,
            "scripts/plot_asymmetric_ev_k0_comparison.py",
            "--manifest",
            "results/reports/asymmetric_ev_dense_10point_suite/run_manifest.csv",
            "--figure-dir",
            "results/reports/asymmetric_ev_dense_10point_suite/figures_all_models",
            "--all-k0",
            "--window",
            "29",
            "--degree",
            "3",
        ]
    )
    print("done")
    print("manifest=results/reports/asymmetric_ev_dense_10point_suite/run_manifest.csv")
    print("figures=results/reports/asymmetric_ev_dense_10point_suite/figures_all_models")


if __name__ == "__main__":
    main()
