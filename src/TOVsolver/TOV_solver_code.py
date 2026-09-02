"""Compute neutron-star mass-radius sequences for the guided beta EoS models.

The input tables are produced by
``good_repo/QMM/notebooks/guided_beta_cs_tvm_vdw_exc_vol.ipynb``. Their
``energy_density`` and ``pressure`` columns are already in MeV/fm^3, the units
expected by :class:`TOVsolver.tov.TOV`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq


SCRIPT_DIR = Path(__file__).resolve().parent
TOV_ROOT = SCRIPT_DIR.parent
QMM_ROOT = TOV_ROOT.parent
QMM_RESULTS = QMM_ROOT / "results" / "generated"

if str(TOV_ROOT) not in sys.path:
    sys.path.insert(0, str(TOV_ROOT))

from TOVsolver.constants import MeV_fm3_to_pa_cgs, c  # noqa: E402
from TOVsolver.tov import TOV  # noqa: E402


@dataclass(frozen=True)
class Model:
    key: str
    label: str
    subdirectory: str
    color: str
    linestyle: str | tuple

    @property
    def beta_csv(self) -> Path:
        stem = self.subdirectory
        return QMM_RESULTS / stem / f"{stem}_asymmetric_beta.csv"


MODELS = (
    Model("vdw_equal_b", r"van der Waals, $b_{pn}=b_n$",
          "guided_beta_model_replication_vdw_equal_b", "black", "-"),
    Model("cs_equal_b", r"Carnahan-Starling, $b_{pn}=b_n$",
          "guided_beta_model_replication_cs_equal_b", "#3B61FF", "--"),
    Model("tvm_equal_b", r"trivial model, $b_{pn}=b_n$",
          "guided_beta_model_replication_tvm_equal_b", "#E45756", "-."),
    Model("cs_target_l", r"Carnahan-Starling, $L=58.9$ MeV",
          "guided_beta_model_replication_cs_target_l", "#4DAA57", ":"),
    Model("tvm_target_l", r"trivial model, $L=58.9$ MeV",
          "guided_beta_model_replication_tvm_target_l", "#FF8C42",
          (0, (5, 1.4, 1.2, 1.4))),
)


def load_core_eos(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate a beta-equilibrium core EoS in MeV/fm^3."""
    if not path.exists():
        raise FileNotFoundError(f"Notebook EoS table not found: {path}")
    table = np.genfromtxt(path, delimiter=",", names=True)
    required = {"energy_density", "pressure"}
    missing = required.difference(table.dtype.names or ())
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    energy = np.asarray(table["energy_density"], dtype=float)
    pressure = np.asarray(table["pressure"], dtype=float)
    valid = np.isfinite(energy) & np.isfinite(pressure) & (energy > 0.0) & (pressure > 0.0)
    energy, pressure = energy[valid], pressure[valid]
    order = np.argsort(energy)
    energy, pressure = energy[order], pressure[order]
    if len(energy) < 4:
        raise ValueError(f"{path} has fewer than four usable EoS points")
    if np.any(np.diff(energy) <= 0.0) or np.any(np.diff(pressure) <= 0.0):
        raise ValueError(f"{path} must have strictly increasing energy density and pressure")
    return energy, pressure


def load_bps_crust() -> tuple[np.ndarray, np.ndarray]:
    """Return the bundled BPS crust table converted from cgs to MeV/fm^3."""
    path = TOV_ROOT / "TOVsolver" / "data" / "Baym_eos.dat"
    table = np.genfromtxt(path, dtype=float, skip_header=1,
                          names=["energy", "pressure", "n_b"])
    energy = table["energy"] / (MeV_fm3_to_pa_cgs / c**2)
    pressure = table["pressure"] / MeV_fm3_to_pa_cgs
    return np.asarray(energy), np.asarray(pressure)


def stitch_crust(
    core_energy: np.ndarray,
    core_pressure: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Join the core to BPS at the highest pressure crossing in their overlap.

    Some target-L tables cross the crust twice at low density. Selecting the
    highest crossing gives the physical transition from crust to uniform core
    and avoids the unbounded root search in the package's legacy stitcher.
    """
    crust_energy, crust_pressure = load_bps_crust()
    core_p = PchipInterpolator(core_energy, core_pressure)
    crust_p = PchipInterpolator(crust_energy, crust_pressure)
    lower = max(float(core_energy[0]), float(crust_energy[0]))
    upper = min(float(core_energy[-1]), float(crust_energy[-1]))
    grid = np.linspace(lower, upper, 10_000)
    difference = crust_p(grid) - core_p(grid)
    crossing_indices = np.flatnonzero(difference[:-1] * difference[1:] <= 0.0)
    if not len(crossing_indices):
        raise ValueError("Core EoS and BPS crust do not cross in their common density range")

    i = int(crossing_indices[-1])
    transition_energy = float(
        brentq(lambda value: float(crust_p(value) - core_p(value)), grid[i], grid[i + 1])
    )
    transition_pressure = float(core_p(transition_energy))
    crust_mask = crust_energy < transition_energy
    core_mask = core_energy > transition_energy
    energy = np.concatenate((crust_energy[crust_mask], [transition_energy], core_energy[core_mask]))
    pressure = np.concatenate((crust_pressure[crust_mask], [transition_pressure], core_pressure[core_mask]))
    if np.any(np.diff(energy) <= 0.0) or np.any(np.diff(pressure) <= 0.0):
        raise ValueError("Stitched EoS is not strictly monotonic")
    return energy, pressure, transition_energy, transition_pressure


def solve_sequence(
    energy: np.ndarray,
    pressure: np.ndarray,
    transition_energy: float,
    points: int,
    rmax_km: float,
    dr_cm: float,
) -> list[dict[str, float | bool]]:
    """Integrate a logarithmic grid of central energy densities."""
    # TOV.__init__ converts its inputs in place, hence the explicit copies.
    solver = TOV(energy.copy(), pressure.copy(), add_crust=False, plot_eos=False)
    central_energies = np.geomspace(
        transition_energy * 1.001,
        float(energy[-1]) * (1.0 - 1.0e-10),
        points,
    )
    rows: list[dict[str, float | bool]] = []
    for central_energy in central_energies:
        radius, mass, _ = solver.solve(float(central_energy),
                                       rmax=rmax_km * 1.0e5, dr=dr_cm)
        if not np.isfinite(radius) or not np.isfinite(mass):
            continue
        if radius >= 0.99 * rmax_km or radius <= 0.0 or mass <= 0.0:
            continue
        rows.append({
            "central_energy_density_mev_fm3": float(central_energy),
            "central_pressure_mev_fm3": float(np.interp(central_energy, energy, pressure)),
            "radius_km": float(radius),
            "mass_msun": float(mass),
        })
    if len(rows) < 4:
        raise RuntimeError("TOV integration produced fewer than four valid stars")

    maximum_index = int(np.argmax([float(row["mass_msun"]) for row in rows]))
    for index, row in enumerate(rows):
        row["stable_branch"] = index <= maximum_index
    return rows


def radius_at_mass(rows: list[dict[str, float | bool]], target_mass: float) -> float | None:
    stable = [row for row in rows if bool(row["stable_branch"])]
    masses = np.asarray([row["mass_msun"] for row in stable], dtype=float)
    radii = np.asarray([row["radius_km"] for row in stable], dtype=float)
    if target_mass < masses.min() or target_mass > masses.max():
        return None
    order = np.argsort(masses)
    masses, radii = masses[order], radii[order]
    masses, unique_indices = np.unique(masses, return_index=True)
    return float(np.interp(target_mass, masses, radii[unique_indices]))


def write_curve(path: Path, model: Model, rows: list[dict[str, float | bool]]) -> None:
    fieldnames = ["model", *rows[0].keys()]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"model": model.key, **row})


def plot_all_models(results: dict[str, dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    for model in MODELS:
        rows = results[model.key]["rows"]
        stable = [row for row in rows if row["stable_branch"]]
        unstable = [row for row in rows if not row["stable_branch"]]
        ax.plot([row["radius_km"] for row in stable],
                [row["mass_msun"] for row in stable],
                color=model.color, linestyle=model.linestyle, linewidth=2.0,
                label=model.label)
        if unstable:
            ax.plot([row["radius_km"] for row in unstable],
                    [row["mass_msun"] for row in unstable],
                    color=model.color, linestyle="--", linewidth=1.0, alpha=0.45)
    ax.set_xlabel(r"Radius $R$ [km]")
    ax.set_ylabel(r"Mass $M/M_\odot$")
    ax.set_title("Guided beta-equilibrium models: TOV mass-radius sequences")
    ax.set_xlim(11.5, 18.0)
    ax.set_ylim(0.5, 3.2)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(output_dir / "mass_radius_all_models.png", dpi=240)
    fig.savefig(output_dir / "mass_radius_all_models.pdf")
    plt.close(fig)


def plot_branch_panels(results: dict[str, dict], output_dir: Path) -> None:
    groups = (
        ("Equal excluded-volume branch", ("vdw_equal_b", "cs_equal_b", "tvm_equal_b")),
        (r"Empirical-slope branch ($L=58.9$ MeV)", ("cs_target_l", "tvm_target_l")),
    )
    by_key = {model.key: model for model in MODELS}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)
    for ax, (title, keys) in zip(axes, groups):
        for key in keys:
            model = by_key[key]
            rows = results[key]["rows"]
            ax.plot([row["radius_km"] for row in rows],
                    [row["mass_msun"] for row in rows],
                    color=model.color, linestyle=model.linestyle, linewidth=2.0,
                    label=model.label.split(",")[0])
        ax.set_xlabel(r"Radius $R$ [km]")
        ax.set_title(title)
        ax.set_xlim(11.5, 18.0)
        ax.set_ylim(0.5, 3.2)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel(r"Mass $M/M_\odot$")
    fig.tight_layout()
    fig.savefig(output_dir / "mass_radius_branch_panels.png", dpi=240)
    fig.savefig(output_dir / "mass_radius_branch_panels.pdf")
    plt.close(fig)


def plot_eos(results: dict[str, dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for model in MODELS:
        item = results[model.key]
        ax.loglog(item["energy"], item["pressure"], color=model.color,
                  linestyle=model.linestyle, linewidth=1.8, label=model.label)
    ax.set_xlabel(r"Energy density $\epsilon$ [MeV/fm$^3$]")
    ax.set_ylabel(r"Pressure $P$ [MeV/fm$^3$]")
    ax.set_title("BPS crust + guided beta-equilibrium equations of state")
    ax.grid(which="both", alpha=0.22)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(output_dir / "stitched_eos_all_models.png", dpi=240)
    fig.savefig(output_dir / "stitched_eos_all_models.pdf")
    plt.close(fig)


def plot_paper_figure4(results: dict[str, dict], output_dir: Path) -> None:
    """Render the EoS and M-R results in the layout used by paper Fig. 4."""
    fig, (ax_eos, ax_mr) = plt.subplots(1, 2, figsize=(11.2, 4.9))

    # Figure 4(a) uses the uniform-matter EoS (without the attached crust) and
    # expresses both axes in GeV/fm^3.
    for model in MODELS:
        item = results[model.key]
        ax_eos.plot(
            item["core_energy"] / 1000.0,
            item["core_pressure"] / 1000.0,
            color=model.color,
            linestyle=model.linestyle,
            linewidth=1.45,
        )
    ax_eos.set_xlim(0.0, 1.0)
    ax_eos.set_ylim(0.0, 0.32)
    ax_eos.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax_eos.set_yticks(np.arange(0.0, 0.31, 0.1))
    ax_eos.set_xlabel(r"$\epsilon$ [GeV fm$^{-3}$]")
    ax_eos.set_ylabel(r"$p$ [GeV fm$^{-3}$]")
    ax_eos.text(0.025, 0.96, "(a)", transform=ax_eos.transAxes, va="top")

    # Approximate published 1-sigma/credible mass intervals, drawn over the
    # same horizontal extents as the visual guides in Fig. 4(b).
    mass_bands = (
        (11.65, 16.80, 2.50, 2.67, "0.72", "GW 190814", 15.75),
        (11.85, 15.75, 2.18, 2.52, "#efb2b2", "PSR J0952-0607", 14.15),
        (13.25, 16.20, 2.01, 2.15, "0.72", "PSR J0740-6620", 14.85),
        (12.15, 14.55, 1.18, 1.59, "0.72", "PSR J0030-0451", 12.35),
    )
    for xmin, xmax, ymin, ymax, color, label, label_x in mass_bands:
        ax_mr.add_patch(
            Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                facecolor=color, edgecolor="none", alpha=0.72, zorder=0,
            )
        )
        ax_mr.text(label_x, (ymin + ymax) / 2.0, label, fontsize=7.0,
                   va="center", ha="left", zorder=1)

    # Keep the same style on both sides of Mmax, as in the publication.
    for model in MODELS:
        rows = results[model.key]["rows"]
        ax_mr.plot(
            [row["radius_km"] for row in rows],
            [row["mass_msun"] for row in rows],
            color=model.color,
            linestyle=model.linestyle,
            linewidth=1.45,
            zorder=3,
        )
    ax_mr.set_xlim(11.5, 18.0)
    ax_mr.set_ylim(0.0, 3.2)
    ax_mr.set_xticks(np.arange(12.0, 18.1, 1.0))
    ax_mr.set_yticks(np.arange(0.0, 3.1, 0.5))
    ax_mr.set_xlabel(r"$R$ [km]")
    ax_mr.set_ylabel(r"$M$ [$M_\odot$]")
    ax_mr.text(0.025, 0.96, "(b)", transform=ax_mr.transAxes, va="top")

    for axis in (ax_eos, ax_mr):
        axis.tick_params(direction="in", top=True, right=True)
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.15, top=0.97, wspace=0.24)
    fig.savefig(output_dir / "figure4_paper_style.png", dpi=300)
    fig.savefig(output_dir / "figure4_paper_style.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=180,
                        help="central-density samples per model")
    parser.add_argument("--rmax-km", type=float, default=50.0,
                        help="radial integration limit")
    parser.add_argument("--dr-cm", type=float, default=250.0,
                        help="radial output step")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.points < 4 or args.rmax_km <= 0.0 or args.dr_cm <= 0.0:
        raise ValueError("--points must be >= 4 and radial settings must be positive")
    output_dir = args.output_dir.resolve()
    curve_dir = output_dir / "curves"
    eos_dir = output_dir / "stitched_eos"
    curve_dir.mkdir(parents=True, exist_ok=True)
    eos_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    summaries: list[dict] = []
    for model in MODELS:
        print(f"Integrating {model.label} from {model.beta_csv.relative_to(WORKSPACE_ROOT)}")
        core_energy, core_pressure = load_core_eos(model.beta_csv)
        energy, pressure, transition_energy, transition_pressure = stitch_crust(
            core_energy, core_pressure
        )
        rows = solve_sequence(energy, pressure, transition_energy, args.points,
                              args.rmax_km, args.dr_cm)
        write_curve(curve_dir / f"{model.key}_mass_radius.csv", model, rows)
        np.savetxt(eos_dir / f"{model.key}_stitched_eos.csv",
                   np.column_stack((energy, pressure)), delimiter=",",
                   header="energy_density_mev_fm3,pressure_mev_fm3", comments="")

        masses = np.asarray([row["mass_msun"] for row in rows], dtype=float)
        maximum_index = int(np.argmax(masses))
        maximum = rows[maximum_index]
        limited_by_table = maximum_index == len(rows) - 1
        summary = {
            "model": model.key,
            "source_csv": str(model.beta_csv),
            "transition_energy_density_mev_fm3": transition_energy,
            "transition_pressure_mev_fm3": transition_pressure,
            "number_of_stars": len(rows),
            "maximum_mass_msun": maximum["mass_msun"],
            "radius_at_maximum_mass_km": maximum["radius_km"],
            "central_energy_at_maximum_mass_mev_fm3": maximum["central_energy_density_mev_fm3"],
            "radius_at_1p4_msun_km": radius_at_mass(rows, 1.4),
            "maximum_at_eos_table_boundary": limited_by_table,
        }
        summaries.append(summary)
        results[model.key] = {
            "core_energy": core_energy,
            "core_pressure": core_pressure,
            "energy": energy,
            "pressure": pressure,
            "rows": rows,
        }
        qualifier = " (lower bound; EoS table ends here)" if limited_by_table else ""
        r14 = summary["radius_at_1p4_msun_km"]
        r14_text = "not reached" if r14 is None else f"{r14:.3f} km"
        print(f"  Mmax={summary['maximum_mass_msun']:.4f} Msun at "
              f"R={summary['radius_at_maximum_mass_km']:.3f} km{qualifier}; "
              f"R1.4={r14_text}")

    plot_all_models(results, output_dir)
    plot_branch_panels(results, output_dir)
    plot_eos(results, output_dir)
    plot_paper_figure4(results, output_dir)
    summary_path = output_dir / "mass_radius_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + os.linesep, encoding="utf-8")
    print(f"Saved plots, curves, stitched EoS tables, and summary under {output_dir}")


if __name__ == "__main__":
    main()
