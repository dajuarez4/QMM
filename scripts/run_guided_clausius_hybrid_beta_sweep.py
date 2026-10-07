"""Run the Clausius mean-field + CS/TVM excluded-volume beta sweep.

The asymmetric EoS output is beta equilibrium only.  The quantum critical
point is a separate finite-temperature symmetric-matter diagnostic.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qmm.qmm_json_helper import build_configs, default_request, save_configs  # noqa: E402


MODELS = ("clausius_cs", "clausius_tvm")
K0_VALUES = tuple(range(250, 316, 5))
BRANCHES = ("target_l", "equal_b")
SUITE = "guided_clausius_hybrid_beta_lambda200"


def make_configs(python_bin: str) -> list[dict]:
    configs: list[dict] = []
    for model in MODELS:
        excluded_volume = model.removeprefix("clausius_")
        for k0 in K0_VALUES:
            request = default_request()
            request.update({
                "run_name": f"{SUITE}_{model}_k0_{k0}",
                "output_subdir": f"{SUITE}/{model}/K0_{k0}",
                "model_name": model,
                "mean_field": "clausius",
                "excluded_volume": excluded_volume,
                "preset": "asymmetric_beta",
                "parameter_mode": "search",
                "target_k0": float(k0),
                "quick_mode": False,
                "momentum_mode": "quarkyonic",
                "lambda_momentum_mev": 200.0,
                "branch_mode": "both",
                "beta_equilibrium": True,
                "proton_fraction_values": [],
                "quarkyonic_n_min_ratio": 1.0e-6,
                "quarkyonic_n_max_ratio": 5.0,
                "quarkyonic_n_points": 60,
                "quarkyonic_fq_scan_points": 61,
                "quarkyonic_shell_integral_points": 100,
                "quarkyonic_quark_integral_points": 100,
                "write_json": True,
                "write_csv": True,
                "write_plots": False,
                "plot_formats": [],
            })
            built = build_configs(ROOT, request, python_bin)
            for config in built:
                config["model"]["parameter_search"]["scan_steps"] = 121
                config["model"]["parameter_search"]["tol"] = 1.0e-8
                config["hadronic_eos"].update({
                    "n_min_ratio": 0.2,
                    "n_max_ratio": 5.0,
                    "n_points": 160,
                    "smoothing_window": 9,
                    "smoothing_degree": 3,
                    "derivative_floor": 1.0e-10,
                })
                config["quarkyonic"]["smoothing_window"] = 9
                config["quarkyonic"]["smoothing_degree"] = 3
            configs.extend(built)
    return configs


def summary_path(config: dict) -> Path:
    output = Path(config["output"]["directory"])
    if not output.is_absolute():
        output = ROOT / output
    return output / f"{config['run_name']}_summary.json"


def run_one(config_path: Path, config: dict, python_bin: str, force: bool) -> dict:
    summary = summary_path(config)
    if summary.exists() and not force:
        return {"run_name": config["run_name"], "status": "cached", "summary": str(summary)}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    command = [python_bin, "-m", "qmm", str(config_path.relative_to(ROOT))]
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            f"{config['run_name']} failed ({completed.returncode})\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return {"run_name": config["run_name"], "status": "computed", "summary": str(summary)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=str(ROOT / ".venv" / "bin" / "python"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    configs = make_configs(args.python)
    paths = save_configs(ROOT, configs)
    manifest = ROOT / "results" / "generated" / SUITE / "config_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps([str(path.relative_to(ROOT)) for path in paths], indent=2) + "\n")
    print(f"Prepared {len(configs)} configurations; manifest: {manifest.relative_to(ROOT)}", flush=True)
    if args.prepare_only:
        return

    completed_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one, path, config, args.python, args.force): config
            for path, config in zip(paths, configs)
        }
        for future in as_completed(futures):
            row = future.result()
            completed_rows.append(row)
            print(f"{row['status']:>8}  {row['run_name']}", flush=True)
            checkpoint = manifest.with_name("run_status.partial.json")
            checkpoint.write_text(json.dumps(completed_rows, indent=2) + "\n")
    manifest.with_name("run_status.json").write_text(json.dumps(completed_rows, indent=2) + "\n")
    print(f"Completed {len(completed_rows)} beta-equilibrium configurations.", flush=True)


if __name__ == "__main__":
    main()
