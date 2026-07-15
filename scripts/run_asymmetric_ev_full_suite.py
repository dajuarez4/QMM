from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "results" / "reports" / "asymmetric_ev_full_suite" / "configs"
REPORT_ROOT = ROOT / "results" / "reports" / "asymmetric_ev_full_suite"
OUTPUT_ROOT = ROOT / "results" / "generated" / "asymmetric_ev_full_suite"

MODELS: tuple[tuple[str, str, float], ...] = (
    ("clausius", "clausius", 4.74),
    ("clausius_cs", "clausius_cs", 6.0),
    ("clausius_tvm", "clausius_tvm", 6.0),
)
K0_VALUES: tuple[int, ...] = (250, 350, 450, 550)
PROTON_FRACTIONS: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5]


def build_config(model_name: str, run_label: str, parameter_max: float, target_k0: int) -> dict[str, object]:
    run_name = f"{run_label}_asym_beta_k0_{target_k0}"
    output_dir = f"results/generated/asymmetric_ev_full_suite/{run_label}/K0_{target_k0}"
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
            "hadronic_eos_table": True,
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
        "hadronic_eos": {
            "n_min_ratio": 0.2,
            "n_max_ratio": 5.0,
            "n_points": 80,
            "smoothing_window": 9,
            "smoothing_degree": 3,
            "derivative_floor": 1.0e-10,
        },
        "quarkyonic": {
            "quick": True,
            "momentum_mode": "quarkyonic",
            "lambda_momentum_mev": 200.0,
            "n_min_ratio": 0.05,
            "n_max_ratio": 5.0,
            "fq_min_shift": 1.0e-5,
            "refine_tol": 1.0e-6,
            "refine_max_iter": 400,
            "smoothing_window": 9,
            "smoothing_degree": 3,
            "derivative_floor": 1.0e-10,
        },
        "asymmetric": {
            "enabled": True,
            "branch_mode": "target_l",
            "target_j": 32.5,
            "target_l": 58.9,
            "target_k0": float(target_k0),
            "delta_scan_steps": 401,
            "derivative_step_l": 1.0e-4,
            "proton_fraction_values": PROTON_FRACTIONS,
            "beta_equilibrium": True,
        },
        "output": {
            "directory": output_dir,
            "write_json": True,
            "write_csv": True,
            "write_plots": True,
            "plot_formats": ["png"],
        },
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def summary_path_from_config(config: dict[str, object]) -> Path:
    output = dict(config["output"])
    output_dir = ROOT / str(output["directory"])
    return output_dir / f"{config['run_name']}_summary.json"


def run_config(config_path: Path) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-m", "qmm", str(config_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def manifest_row(summary: dict[str, object], config_path: Path) -> dict[str, object]:
    outputs = dict(summary.get("outputs", {}))
    asymmetric_fit = dict(summary.get("asymmetric_fit", {}))
    row: dict[str, object] = {
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
        "asymmetric_quarkyonic_png": "",
        "beta_equilibrium_observables_png": "",
        "ground_state_eos_png": "",
    }
    for y_value in PROTON_FRACTIONS:
        row[f"asymmetric_y_{y_value:0.3f}_csv"] = outputs.get(f"asymmetric_y_{y_value:0.3f}_csv", "")
    for key, value in outputs.items():
        if key.endswith("_asymmetric_quarkyonic_png"):
            row["asymmetric_quarkyonic_png"] = value
        elif key.endswith("_beta_equilibrium_observables_png"):
            row["beta_equilibrium_observables_png"] = value
        elif key.endswith("_ground_state_eos_png"):
            row["ground_state_eos_png"] = value
    return row


def write_manifest(rows: list[dict[str, object]]) -> Path:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = REPORT_ROOT / "run_manifest.csv"
    if not rows:
        raise ValueError("No run rows were produced.")
    fieldnames = list(rows[0].keys())
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def build_all_configs() -> list[tuple[Path, dict[str, object]]]:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    configs: list[tuple[Path, dict[str, object]]] = []
    for run_label, model_name, parameter_max in MODELS:
        for target_k0 in K0_VALUES:
            config = build_config(model_name, run_label, parameter_max, target_k0)
            config_path = CONFIG_ROOT / f"{run_label}_k0_{target_k0}.json"
            write_json(config_path, config)
            configs.append((config_path, config))
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and run the asymmetric excluded-volume suite.")
    parser.add_argument("--generate-only", action="store_true", help="Write the JSON configs and exit.")
    parser.add_argument("--manifest-only", action="store_true", help="Build the manifest from existing summary JSON files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = build_all_configs()

    if args.generate_only and args.manifest_only:
        raise ValueError("Choose only one of --generate-only or --manifest-only.")

    if args.generate_only:
        print(json.dumps({"config_count": len(configs), "config_root": str(CONFIG_ROOT.relative_to(ROOT))}, indent=2))
        return

    rows: list[dict[str, object]] = []
    for config_path, config in configs:
        if args.manifest_only:
            summary_path = summary_path_from_config(config)
            if not summary_path.exists():
                raise FileNotFoundError(f"Missing summary JSON: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            print(f"Running {config_path.name}...", flush=True)
            summary = run_config(config_path)
            print(f"Finished {config_path.name}", flush=True)
        rows.append(manifest_row(summary, config_path))

    manifest_path = write_manifest(rows)
    print(json.dumps({"manifest_csv": str(manifest_path.relative_to(ROOT)), "runs": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
