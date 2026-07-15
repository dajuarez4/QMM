#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qmm.asymmetry import compute_asymmetric_fit
from qmm.constants import AsymmetricFitSettings, GroundStateSettings, ParameterSearchSettings


REPORT_ROOT = ROOT / "results" / "reports" / "asymmetric_fit_dense_sweep"

MODEL_SPECS: tuple[tuple[str, str, float, str], ...] = (
    ("clausius", "Clausius-VDW", 4.74, "#143d59"),
    ("clausius_cs", "Clausius-CS", 6.0, "#2a9d8f"),
    ("clausius_tvm", "Clausius-TVM", 6.0, "#c46c2b"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep asymmetric-fit parameters versus target K0.")
    parser.add_argument("--count", type=int, default=50, help="Number of K0 values in the sweep.")
    parser.add_argument("--k0-min", type=float, default=250.0, help="Minimum target K0 in MeV.")
    parser.add_argument("--k0-max", type=float, default=550.0, help="Maximum target K0 in MeV.")
    parser.add_argument(
        "--branch-mode",
        default="target_l",
        choices=("target_l", "equal_b"),
        help="Asymmetric branch to solve.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPORT_ROOT.relative_to(ROOT)),
        help="Output directory relative to the QMM root.",
    )
    return parser.parse_args()


def build_k0_grid(k0_min: float, k0_max: float, count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("count must be at least 2.")
    if k0_max <= k0_min:
        raise ValueError("k0-max must be larger than k0-min.")
    return np.linspace(k0_min, k0_max, count, dtype=float)


def run_fit(model_name: str, parameter_max: float, target_k0: float, branch_mode: str) -> dict[str, float | str]:
    fit = compute_asymmetric_fit(
        model_name,
        parameter_search=ParameterSearchSettings(
            enabled=True,
            target_k0=float(target_k0),
            parameter_min=0.0,
            parameter_max=parameter_max,
            scan_steps=121,
            tol=1.0e-8,
        ),
        fit_settings=AsymmetricFitSettings(
            enabled=True,
            branch_mode=branch_mode,
            target_j=32.5,
            target_l=58.9,
            target_k0=float(target_k0),
            beta_equilibrium=True,
        ),
        ground_state_settings=GroundStateSettings(
            b_min=1.0e-6,
            b_max=6.0,
            b_scan_steps=1600,
            n_integral_points=1500,
            derivative_step=1.0e-6,
            saturation_left=0.05,
            saturation_right=0.30,
        ),
    )
    return {
        "model": fit.model,
        "branch_mode": fit.branch_mode,
        "parameter_name": fit.parameter_name or "",
        "parameter_value": float(fit.parameter_value) if fit.parameter_value is not None else math.nan,
        "target_k0_mev": float(target_k0),
        "fit_k0_mev": float(fit.K0),
        "a_avg_mev_fm3": float(fit.a_avg),
        "b_avg_fm3": float(fit.b_avg),
        "a_n_mev_fm3": float(fit.a_n),
        "a_pn_mev_fm3": float(fit.a_pn),
        "b_n_fm3": float(fit.b_n),
        "b_pn_fm3": float(fit.b_pn),
        "ratio_a_pn_over_a_n": float(fit.ratio_a_pn_over_a_n),
        "ratio_b_pn_over_b_n": float(fit.ratio_b_pn_over_b_n),
        "J_mev": float(fit.J),
        "L_mev": float(fit.L),
        "binding_mev": float(fit.binding),
        "n_sat_fm3": float(fit.n_sat),
        "p_sat_mev_fm3": float(fit.p_sat) if fit.p_sat is not None else math.nan,
        "j_residual_mev": float(fit.j_residual),
        "l_residual_mev": float(fit.l_residual),
        "k0_residual_mev": float(fit.k0_residual),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 240,
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": False,
        }
    )


def model_rows(rows: list[dict[str, object]], model_name: str) -> list[dict[str, object]]:
    return sorted(
        (row for row in rows if row["model"] == model_name),
        key=lambda row: float(row["target_k0_mev"]),
    )


def plot_parameter_sweep(rows: list[dict[str, object]], output_dir: Path, branch_mode: str, count: int) -> None:
    setup_matplotlib()
    figure, axes = plt.subplots(3, 2, figsize=(11.0, 11.5), constrained_layout=True)
    axes_flat = axes.flatten()

    panels = (
        ("parameter_value", r"$c\;[\mathrm{fm}^3]$"),
        ("a_n_mev_fm3", r"$a_n\;[\mathrm{MeV\,fm}^3]$"),
        ("a_pn_mev_fm3", r"$a_{pn}\;[\mathrm{MeV\,fm}^3]$"),
        ("b_n_fm3", r"$b_n\;[\mathrm{fm}^3]$"),
        ("b_pn_fm3", r"$b_{pn}\;[\mathrm{fm}^3]$"),
    )

    for axis, (column, ylabel) in zip(axes_flat[:5], panels):
        for model_name, label, _, color in MODEL_SPECS:
            selected = model_rows(rows, model_name)
            x_values = np.array([float(row["target_k0_mev"]) for row in selected], dtype=float)
            y_values = np.array([float(row[column]) for row in selected], dtype=float)
            axis.plot(x_values, y_values, color=color, label=label)
        axis.set_xlabel(r"$K_0\;[\mathrm{MeV}]$")
        axis.set_ylabel(ylabel)

    ratio_axis = axes_flat[5]
    for model_name, label, _, color in MODEL_SPECS:
        selected = model_rows(rows, model_name)
        x_values = np.array([float(row["target_k0_mev"]) for row in selected], dtype=float)
        ratio_a = np.array([float(row["ratio_a_pn_over_a_n"]) for row in selected], dtype=float)
        ratio_b = np.array([float(row["ratio_b_pn_over_b_n"]) for row in selected], dtype=float)
        ratio_axis.plot(x_values, ratio_a, color=color, label=f"{label}: $a_{{pn}}/a_n$")
        ratio_axis.plot(x_values, ratio_b, color=color, linestyle="--", label=f"{label}: $b_{{pn}}/b_n$")
    ratio_axis.set_xlabel(r"$K_0\;[\mathrm{MeV}]$")
    ratio_axis.set_ylabel("ratio")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    ratio_axis.legend(loc="best", fontsize=9)

    branch_tag = "target_l" if branch_mode == "target_l" else "equal_b"
    base_name = f"asymmetric_fit_{branch_tag}_{count}point_parameter_sweep"
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"{base_name}.{suffix}", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    k0_grid = build_k0_grid(args.k0_min, args.k0_max, args.count)
    grid_rows = [
        {"index": index + 1, "target_k0_mev": f"{value:.12f}"}
        for index, value in enumerate(k0_grid)
    ]
    write_csv(output_dir / f"k0_grid_{args.count}_values.csv", grid_rows, ["index", "target_k0_mev"])

    rows: list[dict[str, object]] = []
    for model_name, label, parameter_max, _ in MODEL_SPECS:
        print(f"Running {label}...", flush=True)
        for target_k0 in k0_grid:
            row = run_fit(model_name, parameter_max, float(target_k0), args.branch_mode)
            row["model_label"] = label
            rows.append(row)

    fieldnames = [
        "model",
        "model_label",
        "branch_mode",
        "parameter_name",
        "parameter_value",
        "target_k0_mev",
        "fit_k0_mev",
        "a_avg_mev_fm3",
        "b_avg_fm3",
        "a_n_mev_fm3",
        "a_pn_mev_fm3",
        "b_n_fm3",
        "b_pn_fm3",
        "ratio_a_pn_over_a_n",
        "ratio_b_pn_over_b_n",
        "J_mev",
        "L_mev",
        "binding_mev",
        "n_sat_fm3",
        "p_sat_mev_fm3",
        "j_residual_mev",
        "l_residual_mev",
        "k0_residual_mev",
    ]
    branch_tag = "target_l" if args.branch_mode == "target_l" else "equal_b"
    write_csv(output_dir / f"asymmetric_fit_{branch_tag}_{args.count}point_sweep.csv", rows, fieldnames)

    plot_parameter_sweep(rows, output_dir, args.branch_mode, args.count)
    print(f"Wrote sweep data and plots to {output_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
