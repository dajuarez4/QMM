from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_GRID = ROOT / "results" / "reports" / "asymmetric_fit_dense_sweep" / "k0_grid_50_values.csv"
REPORT_ROOT = ROOT / "results" / "reports" / "asymmetric_ev_dense_10point_suite"
CONFIG_ROOT = REPORT_ROOT / "configs"
OUTPUT_ROOT = ROOT / "results" / "generated" / "asymmetric_ev_dense_10point_suite"

MODELS: tuple[tuple[str, str, float], ...] = (
    ("clausius", "clausius", 4.74),
    ("clausius_cs", "clausius_cs", 6.0),
    ("clausius_tvm", "clausius_tvm", 6.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a denser 10-point asymmetric beta-equilibrium EOS suite.")
    parser.add_argument("--count", type=int, default=10, help="Number of K0 values selected from the 50-point grid.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[model_name for model_name, _, _ in MODELS],
        help="Subset of model names to run. Choices: clausius, clausius_cs, clausius_tvm.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for the qmm runs.")
    parser.add_argument("--generate-only", action="store_true", help="Write the JSON configs and exit.")
    parser.add_argument("--manifest-only", action="store_true", help="Build the manifest from existing summary JSON files.")
    return parser.parse_args()


def load_source_grid() -> list[float]:
    with SOURCE_GRID.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [float(row["target_k0_mev"]) for row in rows]


def select_k0_values(values: list[float], count: int) -> list[float]:
    if count < 2:
        raise ValueError("count must be at least 2.")
    if count > len(values):
        raise ValueError("count cannot exceed the source grid length.")
    last = len(values) - 1
    indices = [int(index * last / (count - 1)) for index in range(count)]
    selected: list[float] = []
    seen: set[float] = set()
    for index in indices:
        value = values[index]
        if value not in seen:
            selected.append(value)
            seen.add(value)
    if len(selected) != count:
        raise ValueError("Selected K0 values were not unique.")
    return selected


def k0_tag(value: float) -> str:
    rounded = round(float(value), 3)
    if abs(rounded - round(rounded)) < 1.0e-9:
        return str(int(round(rounded)))
    return f"{rounded:.3f}".replace(".", "p")


def build_config(model_name: str, run_label: str, parameter_max: float, target_k0: float) -> dict[str, object]:
    tag = k0_tag(target_k0)
    run_name = f"{run_label}_dense_beta_k0_{tag}"
    output_dir = f"results/generated/asymmetric_ev_dense_10point_suite/{run_label}/K0_{tag}"
    return {
        "run_name": run_name,
        "model": {
            "name": model_name,
            "parameter_search": {
                "enabled": True,
                "target_k0": float(target_k0),
                "parameter_min": 0.0,
                "parameter_max": parameter_max,
                "scan_steps": 121,
                "tol": 1.0e-8,
            },
        },
        "workflows": {
            "ground_state": True,
            "quantum_critical": False,
            "hadronic_eos_table": False,
            "symmetric_quarkyonic": False,
            "asymmetric_fit": True,
            "asymmetric_quarkyonic": True,
            "neutron_star": False,
        },
        "physical": {
            "n0": 0.16,
            "binding_energy": -16.0,
            "symmetry_energy_j": 32.5,
            "symmetry_slope_l": 58.9,
        },
        "ground_state": {
            "b_min": 1.0e-6,
            "b_max": 6.0,
            "b_scan_steps": 1600,
            "n_integral_points": 1500,
            "derivative_step": 1.0e-6,
            "saturation_left": 0.05,
            "saturation_right": 0.30,
        },
        "quarkyonic": {
            "quick": False,
            "momentum_mode": "quarkyonic",
            "lambda_momentum_mev": 200.0,
            "n_min_ratio": 0.05,
            "n_max_ratio": 5.0,
            "n_points": 80,
            "fq_scan_points": 41,
            "fq_min_shift": 1.0e-5,
            "refine_tol": 1.0e-6,
            "refine_max_iter": 400,
            "shell_integral_points": 160,
            "quark_integral_points": 160,
            "smoothing_window": 9,
            "smoothing_degree": 3,
            "derivative_floor": 1.0e-10,
            "beta_y_scan_points": 48,
        },
        "asymmetric": {
            "enabled": True,
            "branch_mode": "target_l",
            "target_j": 32.5,
            "target_l": 58.9,
            "target_k0": float(target_k0),
            "delta_scan_steps": 401,
            "derivative_step_l": 1.0e-4,
            "proton_fraction_values": [],
            "beta_equilibrium": True,
        },
        "output": {
            "directory": output_dir,
            "write_json": True,
            "write_csv": True,
            "write_plots": False,
        },
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_config(config_path: Path) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-m", "qmm", str(config_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def summary_path_from_config(config: dict[str, object]) -> Path:
    output_dir = ROOT / str(dict(config["output"])["directory"])
    return output_dir / f"{config['run_name']}_summary.json"


def manifest_row(summary: dict[str, object], config_path: Path) -> dict[str, object]:
    outputs = dict(summary.get("outputs", {}))
    asymmetric_fit = dict(summary.get("asymmetric_fit", {}))
    return {
        "model": summary.get("model", {}).get("name", ""),
        "run_name": summary.get("run_name", ""),
        "target_k0": asymmetric_fit.get("target_k0", ""),
        "fit_k0": asymmetric_fit.get("K0", ""),
        "branch_mode": asymmetric_fit.get("branch_mode", ""),
        "parameter_name": asymmetric_fit.get("parameter_name", ""),
        "parameter_value": asymmetric_fit.get("parameter_value", ""),
        "config_path": str(config_path.relative_to(ROOT)),
        "summary_json": outputs.get("summary_json", ""),
        "ground_state_csv": outputs.get("ground_state_csv", ""),
        "hadronic_eos_csv": outputs.get("hadronic_eos_csv", ""),
        "asymmetric_fit_csv": outputs.get("asymmetric_fit_csv", ""),
        "asymmetric_beta_csv": outputs.get("asymmetric_beta_csv", ""),
    }


def write_manifest(rows: list[dict[str, object]]) -> Path:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = REPORT_ROOT / "run_manifest.csv"
    fieldnames = list(rows[0].keys())
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def write_selected_k0_table(values: list[float]) -> Path:
    path = REPORT_ROOT / "selected_k0_values.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "target_k0_mev", "k0_tag"])
        writer.writeheader()
        for index, value in enumerate(values, start=1):
            writer.writerow({"index": index, "target_k0_mev": f"{value:.12f}", "k0_tag": k0_tag(value)})
    return path


def build_all_configs(selected_k0_values: list[float], selected_models: set[str]) -> list[tuple[Path, dict[str, object]]]:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    configs: list[tuple[Path, dict[str, object]]] = []
    for run_label, model_name, parameter_max in MODELS:
        if model_name not in selected_models:
            continue
        for target_k0 in selected_k0_values:
            tag = k0_tag(target_k0)
            config = build_config(model_name, run_label, parameter_max, target_k0)
            config_path = CONFIG_ROOT / f"{run_label}_k0_{tag}.json"
            write_json(config_path, config)
            configs.append((config_path, config))
    return configs


def main() -> None:
    args = parse_args()
    if args.generate_only and args.manifest_only:
        raise ValueError("Choose only one of --generate-only or --manifest-only.")

    allowed_models = {model_name for _, model_name, _ in MODELS}
    selected_models = set(args.models)
    unknown = selected_models - allowed_models
    if unknown:
        raise ValueError(f"Unknown model names: {', '.join(sorted(unknown))}")

    selected_k0_values = select_k0_values(load_source_grid(), args.count)
    selected_path = write_selected_k0_table(selected_k0_values)
    configs = build_all_configs(selected_k0_values, selected_models)

    if args.generate_only:
        print(
            json.dumps(
                {
                    "config_count": len(configs),
                    "selected_k0_csv": str(selected_path.relative_to(ROOT)),
                    "config_root": str(CONFIG_ROOT.relative_to(ROOT)),
                },
                indent=2,
            )
        )
        return

    rows: list[dict[str, object]] = []
    if args.manifest_only:
        for config_path, config in configs:
            summary_path = summary_path_from_config(config)
            if not summary_path.exists():
                raise FileNotFoundError(f"Missing summary JSON: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(manifest_row(summary, config_path))
    elif args.workers <= 1:
        for config_path, config in configs:
            print(f"Running {config_path.name}...", flush=True)
            summary = run_config(config_path)
            print(f"Finished {config_path.name}", flush=True)
            rows.append(manifest_row(summary, config_path))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {}
            for config_path, config in configs:
                print(f"Queued {config_path.name}...", flush=True)
                future = executor.submit(run_config, config_path)
                future_map[future] = (config_path, config)
            for future in as_completed(future_map):
                config_path, config = future_map[future]
                summary = future.result()
                print(f"Finished {config_path.name}", flush=True)
                rows.append(manifest_row(summary, config_path))

    manifest_path = write_manifest(rows)
    print(
        json.dumps(
            {
                "selected_k0_csv": str(selected_path.relative_to(ROOT)),
                "manifest_csv": str(manifest_path.relative_to(ROOT)),
                "runs": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
