"""Extended-density Clausius asymmetric beta-equilibrium TOV calculation.

This reuses the fitted branches from ``guided_clausius_beta_branch_recompute``
and extends its selected K0 beta profiles from 5 n0 to 15 n0 before solving
the TOV equations.  Only beta-equilibrium rows are recomputed; the expensive
fixed-y and parameter-search workflows are not repeated.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
TOV_ROOT = SCRIPT_DIR.parent
QMM_ROOT = TOV_ROOT.parent
QMM_SRC = QMM_ROOT / "src"
SOURCE_ROOT = QMM_ROOT / "results" / "generated" / "guided_clausius_beta_branch_recompute"
SOURCE_SUITE = "guided_clausius_beta_branch_recompute"

for path in (QMM_SRC, TOV_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from qmm.asymmetry import AsymmetricFitResult  # noqa: E402
from qmm.constants import DEFAULT_PHYSICAL_CONSTANTS, DEFAULT_QUARKYONIC_SETTINGS  # noqa: E402
from qmm.quarkyonic import AsymmetricQuarkyonicEOS  # noqa: E402
from TOV_solver_code import radius_at_mass, solve_sequence, stitch_crust  # noqa: E402


ALL_K0 = tuple(range(250, 316, 5))
SELECTED_K0 = ALL_K0
BRANCHES = ("target_l", "equal_b")
LINESTYLES = {"target_l": "-", "equal_b": "--"}
BRANCH_LABELS = {"target_l": r"$b_{pn}\ne b_n$", "equal_b": r"$b_{pn}=b_n$"}


def color_for_k0(k0: int) -> tuple[float, float, float, float]:
    fraction = (k0 - min(ALL_K0)) / (max(ALL_K0) - min(ALL_K0))
    return plt.get_cmap("turbo")(fraction)


def source_paths(k0: int, branch: str) -> tuple[Path, Path]:
    stem = f"{SOURCE_SUITE}_k0_{k0}_{branch}"
    directory = SOURCE_ROOT / f"K0_{k0}_{branch}"
    return directory / f"{stem}_summary.json", directory / f"{stem}_asymmetric_beta.csv"


def load_existing_rows(path: Path) -> list[dict[str, float]]:
    fields = (
        "n_b", "n_over_n0", "y", "quark_fraction", "k_bu", "k_fp", "k_fn",
        "n_p", "n_n", "energy_density", "mu_e", "eps_e", "eps_mu",
    )
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            row = {field: float(source[field]) for field in fields}
            if all(np.isfinite(list(row.values()))):
                rows.append(row)
    return sorted(rows, key=lambda row: row["n_b"])


def load_extended_rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [
            {key: float(value) for key, value in source.items()}
            for source in csv.DictReader(handle)
        ]
    return sorted(rows, key=lambda row: row["n_b"])


def load_cached_model(k0: int, branch: str, path: Path) -> dict:
    summary_path, _ = source_paths(k0, branch)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fit = AsymmetricFitResult(**summary["asymmetric_fit"])
    return {
        "k0": k0,
        "branch": branch,
        "parameter_value": fit.parameter_value,
        "fit": summary["asymmetric_fit"],
        "rows": load_extended_rows(path),
    }


def local_thermodynamics(rows: list[dict[str, float]], window: int = 9, degree: int = 3) -> None:
    """Reconstruct P=n*d(epsilon)/dn-epsilon and vs2 on a nonuniform grid."""
    n = np.asarray([row["n_b"] for row in rows], dtype=float)
    energy = np.asarray([row["energy_density"] for row in rows], dtype=float)
    count = len(rows)
    half = window // 2
    pressure = np.empty(count)
    mu_b = np.empty(count)
    vs2 = np.empty(count)
    for index in range(count):
        left = max(0, index - half)
        right = min(count, index + half + 1)
        if right - left < degree + 1:
            if left == 0:
                right = degree + 1
            else:
                left = count - degree - 1
        x = n[left:right] - n[index]
        design = np.vstack([x**power for power in range(degree + 1)]).T
        coeffs, *_ = np.linalg.lstsq(design, energy[left:right], rcond=None)
        derivative = coeffs[1]
        second = 2.0 * coeffs[2]
        mu_b[index] = derivative
        pressure[index] = n[index] * derivative - coeffs[0]
        vs2[index] = n[index] * second / derivative if derivative > 0.0 else np.nan
    for index, row in enumerate(rows):
        row["mu_b"] = float(mu_b[index])
        row["pressure"] = float(pressure[index])
        row["vs2"] = float(vs2[index])


def compute_extended_model(task: tuple[int, str, float, int, float, int, int, int]) -> dict:
    (k0, branch, max_ratio, extension_points, lambda_momentum_mev,
     fq_scan_points, shell_integral_points, quark_integral_points) = task
    summary_path, beta_path = source_paths(k0, branch)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fit = AsymmetricFitResult(**summary["asymmetric_fit"])
    settings = replace(
        DEFAULT_QUARKYONIC_SETTINGS,
        n_min_ratio=5.0,
        n_max_ratio=max_ratio,
        n_points=extension_points,
        fq_scan_points=fq_scan_points,
        shell_integral_points=shell_integral_points,
        quark_integral_points=quark_integral_points,
        smoothing_window=9,
        smoothing_degree=3,
        lambda_momentum_mev=lambda_momentum_mev,
    )
    eos = AsymmetricQuarkyonicEOS(fit, DEFAULT_PHYSICAL_CONSTANTS, settings)
    rows = load_existing_rows(beta_path)
    # The 5 n0 endpoint already exists in the notebook table.
    for ratio in np.linspace(5.0, max_ratio, extension_points)[1:]:
        built = eos.build_beta_row(float(ratio * DEFAULT_PHYSICAL_CONSTANTS.n0))
        if built is None:
            continue
        rows.append({
            "n_b": built.n_b,
            "n_over_n0": built.n_over_n0,
            "y": built.y,
            "quark_fraction": built.quark_fraction,
            "k_bu": built.k_bu,
            "k_fp": built.k_fp,
            "k_fn": built.k_fn,
            "n_p": built.n_p,
            "n_n": built.n_n,
            "energy_density": built.energy_density,
            "mu_e": float(built.mu_e),
            "eps_e": built.eps_e,
            "eps_mu": built.eps_mu,
        })
    rows.sort(key=lambda row: row["n_b"])
    local_thermodynamics(rows)
    return {
        "k0": k0,
        "branch": branch,
        "parameter_value": fit.parameter_value,
        "fit": summary["asymmetric_fit"],
        "rows": rows,
    }


def write_extended_eos(path: Path, result: dict) -> None:
    fields = list(result["rows"][0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["rows"])


def write_mr(path: Path, result: dict) -> None:
    rows = result["mr_rows"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_mass_bands(axis: plt.Axes) -> None:
    bands = (
        (2.50, 2.67, "0.78", "GW 190814", 16.0),
        (2.18, 2.52, "#efb2b2", "PSR J0952-0607", 14.0),
        (2.01, 2.15, "0.78", "PSR J0740-6620", 15.0),
        (1.18, 1.59, "0.78", "PSR J0030-0451", 12.0),
    )
    for low, high, color, label, x in bands:
        axis.axhspan(low, high, color=color, alpha=0.58, zorder=0)
        axis.text(x, 0.5 * (low + high), label, fontsize=7, va="center", zorder=1)


def plot_results(results: list[dict], output_dir: Path) -> None:
    ordered = sorted(results, key=lambda item: (item["k0"], item["branch"]))

    fig, (ax_eos, ax_mr) = plt.subplots(1, 2, figsize=(11.4, 5.0))
    draw_mass_bands(ax_mr)
    for result in ordered:
        k0, branch = result["k0"], result["branch"]
        rows, mr = result["rows"], result["mr_rows"]
        label = rf"$K_0={k0}$ MeV, {BRANCH_LABELS[branch]}"
        ax_eos.plot(
            np.asarray([row["energy_density"] for row in rows]) / 1000.0,
            np.asarray([row["pressure"] for row in rows]) / 1000.0,
            color=color_for_k0(k0), ls=LINESTYLES[branch], lw=1.25, label=label,
        )
        ax_mr.plot(
            [row["radius_km"] for row in mr], [row["mass_msun"] for row in mr],
            color=color_for_k0(k0), ls=LINESTYLES[branch], lw=1.25, zorder=3,
        )
    ax_eos.set_xlim(0.0, 4.5)
    ax_eos.set_ylim(0.0, 1.3)
    ax_eos.set_xlabel(r"$\epsilon$ [GeV fm$^{-3}$]")
    ax_eos.set_ylabel(r"$p$ [GeV fm$^{-3}$]")
    ax_eos.text(0.025, 0.96, "(a)", transform=ax_eos.transAxes, va="top")
    ax_eos.legend(frameon=False, fontsize=5.8, loc="upper left", ncol=2)
    ax_mr.set_xlim(8.0, 18.0)
    ax_mr.set_ylim(0.0, 3.2)
    ax_mr.set_xlabel(r"$R$ [km]")
    ax_mr.set_ylabel(r"$M$ [$M_\odot$]")
    ax_mr.text(0.025, 0.96, "(b)", transform=ax_mr.transAxes, va="top")
    for axis in (ax_eos, ax_mr):
        axis.tick_params(direction="in", top=True, right=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.97, wspace=0.22)
    fig.savefig(output_dir / "clausius_asymmetric_figure4_extended.png", dpi=300)
    fig.savefig(output_dir / "clausius_asymmetric_figure4_extended.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharex=True, sharey=True)
    for axis, branch in zip(axes, BRANCHES):
        for result in ordered:
            if result["branch"] != branch:
                continue
            mr = result["mr_rows"]
            axis.plot([row["radius_km"] for row in mr], [row["mass_msun"] for row in mr],
                      color=color_for_k0(result["k0"]), lw=1.35,
                      label=rf"$K_0={result['k0']}$ MeV")
        axis.set_title(BRANCH_LABELS[branch])
        axis.set_xlabel(r"$R$ [km]")
        axis.set_xlim(8.0, 18.0)
        axis.set_ylim(0.0, 3.2)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=7.0, ncol=2)
    axes[0].set_ylabel(r"$M$ [$M_\odot$]")
    fig.tight_layout()
    fig.savefig(output_dir / "clausius_asymmetric_mass_radius_branches.png", dpi=260)
    fig.savefig(output_dir / "clausius_asymmetric_mass_radius_branches.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-density-ratio", type=float, default=15.0)
    parser.add_argument("--extension-points", type=int, default=31)
    parser.add_argument("--tov-points", type=int, default=260)
    parser.add_argument("--lambda-momentum-mev", type=float, default=200.0)
    parser.add_argument("--fq-scan-points", type=int, default=61)
    parser.add_argument("--shell-integral-points", type=int, default=100)
    parser.add_argument("--quark-integral-points", type=int, default=100)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--k0-values", default=",".join(str(value) for value in ALL_K0),
        help="Comma-separated K0 values in MeV (default: every fitted value 250..315)",
    )
    parser.add_argument("--force-eos", action="store_true",
                        help="Recompute extended EoS tables even when cached CSV files exist")
    parser.add_argument("--output-dir", type=Path,
                        default=SCRIPT_DIR / "results_clausius_asymmetric")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--source-suite", default=SOURCE_SUITE)
    return parser.parse_args()


def main() -> None:
    global SOURCE_ROOT, SOURCE_SUITE
    args = parse_args()
    SOURCE_ROOT = args.source_root.resolve()
    SOURCE_SUITE = args.source_suite
    if args.max_density_ratio <= 5.0 or args.extension_points < 5:
        raise ValueError("The extension must end above 5 n0 and contain at least five points")
    output_dir = args.output_dir.resolve()
    eos_dir, mr_dir = output_dir / "extended_eos", output_dir / "curves"
    eos_dir.mkdir(parents=True, exist_ok=True)
    mr_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "source_root": str(SOURCE_ROOT),
        "source_suite": SOURCE_SUITE,
        "lambda_momentum_mev": args.lambda_momentum_mev,
        "maximum_density_ratio": args.max_density_ratio,
        "extension_points_including_5n0": args.extension_points,
        "fq_scan_points": args.fq_scan_points,
        "shell_integral_points": args.shell_integral_points,
        "quark_integral_points": args.quark_integral_points,
        "tov_points_requested": args.tov_points,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    selected_k0 = tuple(int(value.strip()) for value in args.k0_values.split(",") if value.strip())
    invalid = sorted(set(selected_k0) - set(ALL_K0))
    if not selected_k0 or invalid:
        raise ValueError(f"K0 values must be selected from {ALL_K0}; invalid={invalid}")
    tasks = []
    results: list[dict] = []
    for k0 in selected_k0:
        for branch in BRANCHES:
            key = f"K0_{k0}_{branch}"
            cached_path = eos_dir / f"{key}_extended_beta_eos.csv"
            if cached_path.exists() and not args.force_eos:
                result = load_cached_model(k0, branch, cached_path)
                print(f"Loaded cached K0={k0} MeV, {branch}: {len(result['rows'])} EoS points", flush=True)
                results.append(result)
            else:
                tasks.append((k0, branch, args.max_density_ratio, args.extension_points,
                              args.lambda_momentum_mev, args.fq_scan_points,
                              args.shell_integral_points, args.quark_integral_points))
    # Threads avoid macOS semaphore restrictions in managed environments. The
    # numerical kernels spend most of their time in NumPy/SciPy calls.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(compute_extended_model, task): task for task in tasks}
        for future in as_completed(futures):
            k0, branch, *_ = futures[future]
            result = future.result()
            print(f"Extended K0={k0} MeV, {branch}: {len(result['rows'])} EoS points", flush=True)
            # Persist each expensive microscopic extension immediately.  A
            # later interruption can then resume from completed cases.
            key = f"K0_{k0}_{branch}"
            write_extended_eos(eos_dir / f"{key}_extended_beta_eos.csv", result)
            results.append(result)

    summaries = []
    for result in sorted(results, key=lambda item: (item["k0"], item["branch"])):
        key = f"K0_{result['k0']}_{result['branch']}"
        write_extended_eos(eos_dir / f"{key}_extended_beta_eos.csv", result)
        energy = np.asarray([row["energy_density"] for row in result["rows"]], dtype=float)
        pressure = np.asarray([row["pressure"] for row in result["rows"]], dtype=float)
        valid = np.isfinite(energy) & np.isfinite(pressure) & (energy > 0.0) & (pressure > 0.0)
        energy, pressure = energy[valid], pressure[valid]
        order = np.argsort(energy)
        energy, pressure = energy[order], pressure[order]
        # Retain a globally increasing pressure envelope. Comparing only with
        # the immediate predecessor can preserve a point below the last point
        # already retained after an oscillatory transition region.
        monotonic = np.zeros(len(pressure), dtype=bool)
        last_pressure = -np.inf
        for index, value in enumerate(pressure):
            if value > last_pressure:
                monotonic[index] = True
                last_pressure = value
        energy, pressure = energy[monotonic], pressure[monotonic]
        stitched_e, stitched_p, transition_e, transition_p = stitch_crust(energy, pressure)
        mr_rows = solve_sequence(stitched_e, stitched_p, transition_e, args.tov_points, 50.0, 250.0)
        result["mr_rows"] = mr_rows
        write_mr(mr_dir / f"{key}_mass_radius.csv", result)
        masses = np.asarray([row["mass_msun"] for row in mr_rows])
        peak_index = int(np.argmax(masses))
        peak = mr_rows[peak_index]
        last = mr_rows[-1]
        summary = {
            "K0_MeV": result["k0"],
            "branch_mode": result["branch"],
            "lambda_momentum_mev": args.lambda_momentum_mev,
            "fq_scan_points": args.fq_scan_points,
            "shell_integral_points": args.shell_integral_points,
            "quark_integral_points": args.quark_integral_points,
            "maximum_density_ratio": args.max_density_ratio,
            "maximum_energy_density_mev_fm3": float(energy[-1]),
            "maximum_mass_msun": peak["mass_msun"],
            "radius_at_maximum_mass_km": peak["radius_km"],
            "central_energy_at_maximum_mass_mev_fm3": peak["central_energy_density_mev_fm3"],
            "radius_at_1p4_msun_km": radius_at_mass(mr_rows, 1.4),
            "final_unstable_mass_msun": last["mass_msun"],
            "final_unstable_radius_km": last["radius_km"],
            "maximum_at_eos_boundary": peak_index == len(mr_rows) - 1,
            "crust_transition_energy_mev_fm3": transition_e,
            "crust_transition_pressure_mev_fm3": transition_p,
        }
        summaries.append(summary)
        print(f"TOV {key}: Mmax={peak['mass_msun']:.4f} Msun, "
              f"Rmax={peak['radius_km']:.3f} km, tail R={last['radius_km']:.3f} km", flush=True)

    plot_results(results, output_dir)
    (output_dir / "clausius_asymmetric_tov_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved Clausius extended EoS, TOV curves, and figures under {output_dir}")


if __name__ == "__main__":
    main()
