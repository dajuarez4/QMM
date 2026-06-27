"""Automatic plotting utilities for unified workflow outputs."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .constants import GroundStateSettings, PhysicalConstants, QuantumCriticalSettings
from .ground_state import GroundStateResult, energy_per_particle_minus_m, pressure_total
from .neutron_star import NeutronStarSequence
from .numerics import linspace
from .quantum import QuantumCriticalPoint
from .quarkyonic import AsymmetricQuarkyonicResult, SymmetricQuarkyonicCurve


def _load_matplotlib():
    mpl_cache_dir = Path(tempfile.gettempdir()) / "qmm_mplconfig"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _sanitize_suffix(value: str) -> str:
    cleaned = value.replace(" ", "_").replace("/", "_")
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in cleaned)


def _parameter_caption(parameter_name: str | None, parameter_value: float | None) -> str:
    if parameter_name is None or parameter_value is None:
        return ""
    return f", {parameter_name}={parameter_value:.6g}"


def _save_figure(fig, base_path: Path, formats: Iterable[str]) -> dict[str, str]:
    written: dict[str, str] = {}
    for fmt in formats:
        suffix = _sanitize_suffix(str(fmt).lower())
        output_path = base_path.with_suffix(f".{suffix}")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        written[f"{base_path.name}_{suffix}"] = str(output_path)
    plt = _load_matplotlib()
    plt.close(fig)
    return written


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_nan(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def _energy_per_baryon_minus_m(energy_density: float, density: float, physical: PhysicalConstants) -> float:
    if density <= 0.0:
        return math.nan
    return energy_density / density - physical.m_nucleon


def plot_ground_state_eos(
    output_dir: str | Path,
    run_name: str,
    result: GroundStateResult,
    physical: PhysicalConstants,
    settings: GroundStateSettings,
    formats: Iterable[str],
) -> dict[str, str]:
    """Plot the hadronic EOS used by the ground-state fit."""
    plt = _load_matplotlib()
    density_ratios = linspace(0.4, 2.0, 160)
    density_values = [ratio * physical.n0 for ratio in density_ratios]
    energy_values = [
        energy_per_particle_minus_m(
            result.model,
            result.a,
            result.b,
            density,
            result.parameter_value,
            physical,
            settings,
        )
        for density in density_values
    ]
    pressure_values = [
        (
            math.nan
            if (
                pressure := pressure_total(
                    result.model,
                    result.a,
                    result.b,
                    density,
                    result.parameter_value,
                    physical,
                    settings,
                )
            )
            is None
            else pressure
        )
        for density in density_values
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    label = f"{result.model} (K0={result.K0:.3f} MeV{_parameter_caption(result.parameter_name, result.parameter_value)})"

    axes[0].plot(density_ratios, energy_values, color="#0F4C81", lw=2.0)
    axes[0].axhline(physical.binding_energy, color="#9A031E", lw=1.2, ls="--")
    axes[0].axvline(1.0, color="#5F0F40", lw=1.2, ls=":")
    axes[0].set_title("Hadronic Binding Curve")
    axes[0].set_xlabel(r"$n / n_0$")
    axes[0].set_ylabel(r"$E/A - m_N$ [MeV]")
    axes[0].grid(alpha=0.25)
    axes[0].annotate(label, xy=(0.02, 0.03), xycoords="axes fraction", fontsize=9)

    axes[1].plot(density_ratios, pressure_values, color="#2A9D8F", lw=2.0)
    axes[1].axhline(0.0, color="#9A031E", lw=1.2, ls="--")
    axes[1].axvline(1.0, color="#5F0F40", lw=1.2, ls=":")
    axes[1].set_title("Hadronic Pressure")
    axes[1].set_xlabel(r"$n / n_0$")
    axes[1].set_ylabel(r"$P$ [MeV fm$^{-3}$]")
    axes[1].grid(alpha=0.25)

    base_path = Path(output_dir) / f"{run_name}_ground_state_eos"
    return _save_figure(fig, base_path, formats)


def plot_quantum_diagnostics(
    output_dir: str | Path,
    run_name: str,
    result: QuantumCriticalPoint,
    settings: QuantumCriticalSettings,
    formats: Iterable[str],
) -> dict[str, str]:
    """Plot convergence and critical-point diagnostics for one quantum result."""
    plt = _load_matplotlib()
    norm_dPdn = abs(result.dPdn) / max(settings.outer_tol_dPdn, 1.0e-30)
    norm_d2Pdn2 = abs(result.d2Pdn2) / max(settings.outer_tol_d2Pdn2, 1.0e-30)
    log10_score = math.log10(max(result.score, 1.0e-30))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)

    scatter_1 = axes[0, 0].scatter([result.nc], [result.Tc], c=[result.K0], cmap="viridis", s=180)
    axes[0, 0].set_title(r"$T_c$ vs $n_c$")
    axes[0, 0].set_xlabel(r"$n_c$ [fm$^{-3}$]")
    axes[0, 0].set_ylabel(r"$T_c$ [MeV]")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].annotate(result.model, (result.nc, result.Tc), xytext=(8, 8), textcoords="offset points")
    fig.colorbar(scatter_1, ax=axes[0, 0], label=r"$K_0$ [MeV]")

    scatter_2 = axes[0, 1].scatter([result.K0], [result.nc], c=[result.Tc], cmap="plasma", s=180)
    axes[0, 1].set_title(r"$K_0$ vs $n_c$")
    axes[0, 1].set_xlabel(r"$K_0$ [MeV]")
    axes[0, 1].set_ylabel(r"$n_c$ [fm$^{-3}$]")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].annotate(result.model, (result.K0, result.nc), xytext=(8, 8), textcoords="offset points")
    fig.colorbar(scatter_2, ax=axes[0, 1], label=r"$T_c$ [MeV]")

    axes[1, 0].bar(
        ["|dP/dn| / tol", r"|d$^2$P/dn$^2$| / tol"],
        [norm_dPdn, norm_d2Pdn2],
        color=["#F4A261", "#E76F51"],
    )
    axes[1, 0].axhline(1.0, color="black", lw=1.0, ls="--")
    axes[1, 0].set_title("Normalized Residuals")
    axes[1, 0].set_ylabel("Residual / tolerance")
    axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].bar(
        ["iterations", r"$\log_{10}(\mathrm{score})$"],
        [float(result.iterations), log10_score],
        color=["#264653", "#2A9D8F"],
    )
    axes[1, 1].set_title("Convergence Metrics")
    axes[1, 1].grid(axis="y", alpha=0.25)
    text = (
        f"model={result.model}\n"
        f"K0={result.K0:.6g} MeV\n"
        f"Tc={result.Tc:.6g} MeV\n"
        f"nc={result.nc:.6g} fm^-3\n"
        f"Pc={result.Pc:.6g} MeV fm^-3"
        f"{_parameter_caption(result.parameter_name, result.parameter_value)}"
    )
    axes[1, 1].text(1.05, 0.5, text, transform=axes[1, 1].transAxes, va="center", fontsize=9)

    base_path = Path(output_dir) / f"{run_name}_quantum_diagnostics"
    return _save_figure(fig, base_path, formats)


def plot_symmetric_quarkyonic_bundle(
    output_dir: str | Path,
    run_name: str,
    curve: SymmetricQuarkyonicCurve,
    physical: PhysicalConstants,
    formats: Iterable[str],
) -> dict[str, str]:
    """Plot the main symmetric quarkyonic observables."""
    plt = _load_matplotlib()
    mode_label = "baryquark" if curve.momentum_mode == "baryquark" else "quarkyonic"
    energy_per_baryon = [
        _energy_per_baryon_minus_m(energy_density, density, physical)
        for energy_density, density in zip(curve.eps, curve.n)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(
        f"{curve.model} {mode_label} curve (K0={curve.K0:.3f} MeV{_parameter_caption(curve.parameter_name, curve.parameter_value)})",
        fontsize=12,
    )

    axes[0, 0].plot(curve.n_over_n0, curve.vs2, color="#0F4C81", lw=2.0)
    axes[0, 0].axvline(curve.n_tr_over_n0, color="#9A031E", ls="--", lw=1.0)
    axes[0, 0].axhline(1.0 / 3.0, color="gray", ls=(0, (5, 5)), lw=1.0)
    axes[0, 0].set_title(r"$v_s^2$ vs $n/n_0$")
    axes[0, 0].set_xlabel(r"$n / n_0$")
    axes[0, 0].set_ylabel(r"$v_s^2$")
    axes[0, 0].set_ylim(-0.1, 1.0)
    axes[0, 0].set_xlim(0.0, 5.0)
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(curve.n_over_n0, curve.quark_fraction, color="#2A9D8F", lw=2.0)
    axes[0, 1].set_title("Quark Fraction")
    axes[0, 1].set_xlabel(r"$n / n_0$")
    axes[0, 1].set_ylabel(r"$f_Q$")
    axes[0, 1].set_xlim(0.0, 5.0)
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(curve.n_over_n0, curve.P, color="#E76F51", lw=2.0)
    axes[1, 0].axhline(0.0, color="black", ls="--", lw=1.0)
    axes[1, 0].set_title("Pressure")
    axes[1, 0].set_xlabel(r"$n / n_0$")
    axes[1, 0].set_ylabel(r"$P$ [MeV fm$^{-3}$]")
    axes[1, 0].set_xlim(0.0, 5.0)
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(curve.n_over_n0, energy_per_baryon, color="#6A4C93", lw=2.0)
    axes[1, 1].axhline(0.0, color="black", ls="--", lw=1.0)
    axes[1, 1].set_title(r"$E/A - m_N$")
    axes[1, 1].set_xlabel(r"$n / n_0$")
    axes[1, 1].set_ylabel(r"$E/A - m_N$ [MeV]")
    axes[1, 1].set_xlim(0.0, 5.0)
    axes[1, 1].grid(alpha=0.25)

    base_path = Path(output_dir) / f"{run_name}_symmetric_quarkyonic"
    return _save_figure(fig, base_path, formats)


def plot_asymmetric_profiles(
    output_dir: str | Path,
    run_name: str,
    result: AsymmetricQuarkyonicResult,
    physical: PhysicalConstants,
    formats: Iterable[str],
) -> dict[str, str]:
    """Plot fixed-`y` asymmetric quarkyonic profiles."""
    plt = _load_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(
        f"{result.model} asymmetric quarkyonic profiles{_parameter_caption(result.parameter_name, result.parameter_value)}",
        fontsize=12,
    )

    for label, rows in sorted(result.fixed_y_profiles.items()):
        if not rows:
            continue
        x_values = [row.n_over_n0 for row in rows]
        vs2_values = [math.nan if row.vs2 is None else row.vs2 for row in rows]
        fq_values = [row.quark_fraction for row in rows]
        pressure_values = [math.nan if row.pressure is None else row.pressure for row in rows]
        energy_values = [_energy_per_baryon_minus_m(row.energy_density, row.n_b, physical) for row in rows]
        legend_label = f"y={rows[0].y:.3f}"

        axes[0, 0].plot(x_values, vs2_values, lw=2.0, label=legend_label)
        axes[0, 1].plot(x_values, fq_values, lw=2.0, label=legend_label)
        axes[1, 0].plot(x_values, pressure_values, lw=2.0, label=legend_label)
        axes[1, 1].plot(x_values, energy_values, lw=2.0, label=legend_label)

    if result.beta_profile:
        x_values = [row.n_over_n0 for row in result.beta_profile]
        vs2_values = [math.nan if row.vs2 is None else row.vs2 for row in result.beta_profile]
        fq_values = [row.quark_fraction for row in result.beta_profile]
        pressure_values = [math.nan if row.pressure is None else row.pressure for row in result.beta_profile]
        energy_values = [_energy_per_baryon_minus_m(row.energy_density, row.n_b, physical) for row in result.beta_profile]

        axes[0, 0].plot(x_values, vs2_values, lw=2.2, ls="--", color="black", label=r"$\beta$ eq")
        axes[0, 1].plot(x_values, fq_values, lw=2.2, ls="--", color="black", label=r"$\beta$ eq")
        axes[1, 0].plot(x_values, pressure_values, lw=2.2, ls="--", color="black", label=r"$\beta$ eq")
        axes[1, 1].plot(x_values, energy_values, lw=2.2, ls="--", color="black", label=r"$\beta$ eq")

    axes[0, 0].set_title(r"$v_s^2$ vs $n/n_0$")
    axes[0, 0].set_xlabel(r"$n / n_0$")
    axes[0, 0].set_ylabel(r"$v_s^2$")
    axes[0, 0].axhline(1.0 / 3.0, color="gray", ls=(0, (5, 5)), lw=1.0)
    axes[0, 0].set_ylim(-0.1, 1.0)
    axes[0, 0].set_xlim(0.0, 5.0)
    axes[0, 1].set_title("Quark Fraction")
    axes[0, 1].set_xlabel(r"$n / n_0$")
    axes[0, 1].set_ylabel(r"$f_Q$")
    axes[0, 1].set_xlim(0.0, 5.0)
    axes[1, 0].set_title("Pressure")
    axes[1, 0].set_xlabel(r"$n / n_0$")
    axes[1, 0].set_ylabel(r"$P$ [MeV fm$^{-3}$]")
    axes[1, 0].set_xlim(0.0, 5.0)
    axes[1, 1].set_title(r"$E/A - m_N$")
    axes[1, 1].set_xlabel(r"$n / n_0$")
    axes[1, 1].set_ylabel(r"$E/A - m_N$ [MeV]")
    axes[1, 1].set_xlim(0.0, 5.0)

    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    base_path = Path(output_dir) / f"{run_name}_asymmetric_quarkyonic"
    return _save_figure(fig, base_path, formats)


def plot_beta_equilibrium_observables(
    output_dir: str | Path,
    run_name: str,
    result: AsymmetricQuarkyonicResult,
    physical: PhysicalConstants,
    formats: Iterable[str],
) -> dict[str, str]:
    """Plot the beta-equilibrium observables used for neutron-star matter."""
    if not result.beta_profile:
        return {}
    rows = [row for row in result.beta_profile if row.n_over_n0 >= 0.05]
    if len(rows) < 3:
        rows = result.beta_profile

    plt = _load_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(
        f"{result.model} beta-equilibrium quarkyonic matter{_parameter_caption(result.parameter_name, result.parameter_value)}",
        fontsize=12,
    )

    x_values = [row.n_over_n0 for row in rows]
    y_values = [row.y for row in rows]
    vs2_values = [math.nan if row.vs2 is None else row.vs2 for row in rows]
    energy_values = [
        _energy_per_baryon_minus_m(row.energy_density, row.n_b, physical)
        for row in rows
    ]
    fq_values = [row.quark_fraction for row in rows]

    axes[0, 0].plot(x_values, y_values, color="#BC4749", lw=2.2)
    axes[0, 0].set_title("Charge Fraction")
    axes[0, 0].set_xlabel(r"$n / n_0$")
    axes[0, 0].set_ylabel(r"$y$")
    axes[0, 0].set_xlim(0.0, 5.0)

    axes[0, 1].plot(x_values, vs2_values, color="#0F4C81", lw=2.2)
    axes[0, 1].axhline(1.0 / 3.0, color="gray", ls=(0, (5, 5)), lw=1.0)
    axes[0, 1].set_ylim(-0.1, 1.0)
    axes[0, 1].set_title(r"$v_s^2$")
    axes[0, 1].set_xlabel(r"$n / n_0$")
    axes[0, 1].set_ylabel(r"$v_s^2$")
    axes[0, 1].set_xlim(0.0, 5.0)

    axes[1, 0].plot(x_values, energy_values, color="#6A4C93", lw=2.2)
    axes[1, 0].axhline(0.0, color="black", ls="--", lw=1.0)
    axes[1, 0].set_title(r"$\varepsilon / n_B - m_N$")
    axes[1, 0].set_xlabel(r"$n / n_0$")
    axes[1, 0].set_ylabel(r"$\varepsilon / n_B - m_N$ [MeV]")
    axes[1, 0].set_xlim(0.0, 5.0)

    axes[1, 1].plot(x_values, fq_values, color="#2A9D8F", lw=2.2)
    axes[1, 1].set_title("Quark Fraction")
    axes[1, 1].set_xlabel(r"$n / n_0$")
    axes[1, 1].set_ylabel(r"$f_Q$")
    axes[1, 1].set_xlim(0.0, 5.0)

    for axis in axes.flat:
        axis.grid(alpha=0.25)

    base_path = Path(output_dir) / f"{run_name}_beta_equilibrium_observables"
    return _save_figure(fig, base_path, formats)


def plot_neutron_star_sequence(
    output_dir: str | Path,
    run_name: str,
    sequence: NeutronStarSequence,
    formats: Iterable[str],
) -> dict[str, str]:
    """Plot the beta-equilibrium EOS and the resulting TOV sequence."""
    plt = _load_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    eos = sequence.eos

    axes[0].plot(eos.energy_density_mev_fm3, eos.pressure_mev_fm3, color="#264653", lw=2.2)
    axes[0].set_title(r"Beta-Equilibrium EOS")
    axes[0].set_xlabel(r"$\varepsilon$ [MeV fm$^{-3}$]")
    axes[0].set_ylabel(r"$P$ [MeV fm$^{-3}$]")
    axes[0].grid(alpha=0.25)

    radii = [point.radius_km for point in sequence.points]
    masses = [point.mass_msun for point in sequence.points]
    axes[1].plot(radii, masses, color="#E76F51", lw=2.2)
    axes[1].scatter(
        [sequence.radius_at_max_mass_km],
        [sequence.max_mass_msun],
        color="black",
        s=35,
        zorder=3,
    )
    axes[1].annotate(
        rf"$M_{{\max}}={sequence.max_mass_msun:.3f}\,M_\odot$",
        (sequence.radius_at_max_mass_km, sequence.max_mass_msun),
        xytext=(8, -14),
        textcoords="offset points",
        fontsize=9,
    )
    axes[1].set_title("Mass-Radius Sequence")
    axes[1].set_xlabel(r"$R$ [km]")
    axes[1].set_ylabel(r"$M$ [$M_\odot$]")
    axes[1].grid(alpha=0.25)

    fig.suptitle(
        f"{eos.model} beta-equilibrium neutron-star sequence{_parameter_caption(eos.parameter_name, eos.parameter_value)}",
        fontsize=12,
    )

    base_path = Path(output_dir) / f"{run_name}_neutron_star_sequence"
    return _save_figure(fig, base_path, formats)


def plot_fixed_y_quantum_diagnostics(
    results_csv_path: str | Path,
    status_csv_path: str | Path | None,
    output_prefix: str | Path,
    settings: QuantumCriticalSettings,
    formats: Iterable[str],
) -> dict[str, str]:
    """Reproduce the residual/convergence heatmaps from the reference notebook."""
    results_rows = _read_csv_rows(results_csv_path)
    if not results_rows:
        return {}

    status_rows = _read_csv_rows(status_csv_path) if status_csv_path is not None else []
    status_map = {
        (row["target_k0"], row["y"]): 1.0 if row.get("status", "") == "converged" else 0.0
        for row in status_rows
        if "target_k0" in row and "y" in row
    }

    processed_rows: list[dict[str, float]] = []
    for row in results_rows:
        target_k0 = _float_or_nan(row, "target_k0")
        y_value = _float_or_nan(row, "y")
        dPdn = _float_or_nan(row, "dPdn")
        d2Pdn2 = _float_or_nan(row, "d2Pdn2")
        score = _float_or_nan(row, "score")
        iterations = _float_or_nan(row, "iterations")
        norm_residual = max(
            abs(dPdn) / max(settings.outer_tol_dPdn, 1.0e-30),
            abs(d2Pdn2) / max(settings.outer_tol_d2Pdn2, 1.0e-30),
        )
        log10_score = math.log10(max(score, 1.0e-30))
        status_value = status_map.get((row.get("target_k0", ""), row.get("y", "")), 1.0)
        processed_rows.append(
            {
                "target_k0": target_k0,
                "y": y_value,
                "iterations": iterations,
                "log10_score": log10_score,
                "norm_residual": norm_residual,
                "ok": status_value,
            }
        )

    unique_k0 = sorted({row["target_k0"] for row in processed_rows if math.isfinite(row["target_k0"])})
    unique_y = sorted({row["y"] for row in processed_rows if math.isfinite(row["y"])})
    if not unique_k0 or not unique_y:
        return {}

    def build_matrix(key: str) -> list[list[float]]:
        lookup = {(row["target_k0"], row["y"]): row[key] for row in processed_rows}
        matrix: list[list[float]] = []
        for target_k0 in unique_k0:
            matrix.append([lookup.get((target_k0, y_value), math.nan) for y_value in unique_y])
        return matrix

    plt = _load_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    maps = [
        ("iterations", "Iterations", axes[0, 0], "viridis"),
        ("log10_score", r"$\log_{10}(\mathrm{score})$", axes[0, 1], "magma_r"),
        ("norm_residual", "Normalized residual", axes[1, 0], "plasma"),
        ("ok", "Convergence mask", axes[1, 1], "Greens"),
    ]

    for key, title, axis, cmap in maps:
        matrix = build_matrix(key)
        image = axis.imshow(matrix, aspect="auto", origin="lower", cmap=cmap)
        axis.set_title(title)
        axis.set_xticks(range(len(unique_y)))
        axis.set_xticklabels([f"{value:.1f}" for value in unique_y])
        axis.set_yticks(range(len(unique_k0)))
        axis.set_yticklabels([f"{value:.0f}" for value in unique_k0])
        axis.set_xlabel("y")
        axis.set_ylabel(r"target $K_0$ [MeV]")
        fig.colorbar(image, ax=axis)

    return _save_figure(fig, Path(output_prefix), formats)


def quantum_collection_rows(search_root: str | Path, settings: QuantumCriticalSettings) -> list[dict[str, float | str]]:
    root = Path(search_root)
    rows: list[dict[str, float | str]] = []
    for path in sorted(root.rglob("*_quantum_critical.csv")):
        for row in _read_csv_rows(path):
            score = _float_or_nan(row, "score")
            dPdn = _float_or_nan(row, "dPdn")
            d2Pdn2 = _float_or_nan(row, "d2Pdn2")
            norm_residual = max(
                abs(dPdn) / max(settings.outer_tol_dPdn, 1.0e-30),
                abs(d2Pdn2) / max(settings.outer_tol_d2Pdn2, 1.0e-30),
            )
            rows.append(
                {
                    "run_name": path.stem.removesuffix("_quantum_critical"),
                    "model": row.get("model", ""),
                    "parameter_name": row.get("parameter_name", ""),
                    "parameter_value": _float_or_nan(row, "parameter_value"),
                    "K0": _float_or_nan(row, "K0"),
                    "Tc": _float_or_nan(row, "Tc"),
                    "nc": _float_or_nan(row, "nc"),
                    "Pc": _float_or_nan(row, "Pc"),
                    "iterations": _float_or_nan(row, "iterations"),
                    "score": score,
                    "log10_score": math.log10(max(score, 1.0e-30)),
                    "norm_residual": norm_residual,
                }
            )
    return rows


def write_quantum_collection_csv(
    rows: list[dict[str, float | str]],
    output_path: str | Path,
) -> str | None:
    """Write an aggregated quantum comparison table."""
    if not rows:
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(path)


def plot_quantum_collection(
    rows: list[dict[str, float | str]],
    output_prefix: str | Path,
    formats: Iterable[str],
) -> dict[str, str]:
    """Plot cross-run quantum summary scatters."""
    if not rows:
        return {}
    plt = _load_matplotlib()

    nc_values = [float(row["nc"]) for row in rows]
    tc_values = [float(row["Tc"]) for row in rows]
    k0_values = [float(row["K0"]) for row in rows]
    log10_values = [float(row["log10_score"]) for row in rows]
    residual_values = [float(row["norm_residual"]) for row in rows]
    iteration_values = [float(row["iterations"]) for row in rows]
    labels = [str(row["model"]) if not row["parameter_name"] else f"{row['model']} {row['parameter_name']}={row['parameter_value']:.4g}" for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    scatter = axes[0, 0].scatter(nc_values, tc_values, c=k0_values, cmap="viridis", s=110)
    axes[0, 0].set_title(r"$T_c$ vs $n_c$")
    axes[0, 0].set_xlabel(r"$n_c$ [fm$^{-3}$]")
    axes[0, 0].set_ylabel(r"$T_c$ [MeV]")
    axes[0, 0].grid(alpha=0.25)
    for x_value, y_value, label in zip(nc_values, tc_values, labels):
        axes[0, 0].annotate(label, (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
    fig.colorbar(scatter, ax=axes[0, 0], label=r"$K_0$ [MeV]")

    scatter = axes[0, 1].scatter(k0_values, nc_values, c=tc_values, cmap="plasma", s=110)
    axes[0, 1].set_title(r"$K_0$ vs $n_c$")
    axes[0, 1].set_xlabel(r"$K_0$ [MeV]")
    axes[0, 1].set_ylabel(r"$n_c$ [fm$^{-3}$]")
    axes[0, 1].grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes[0, 1], label=r"$T_c$ [MeV]")

    scatter = axes[1, 0].scatter(k0_values, tc_values, c=residual_values, cmap="magma_r", s=110)
    axes[1, 0].set_title(r"$T_c$ vs $K_0$")
    axes[1, 0].set_xlabel(r"$K_0$ [MeV]")
    axes[1, 0].set_ylabel(r"$T_c$ [MeV]")
    axes[1, 0].grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes[1, 0], label="normalized residual")

    scatter = axes[1, 1].scatter(iteration_values, residual_values, c=log10_values, cmap="cividis", s=110)
    axes[1, 1].set_title("Convergence Summary")
    axes[1, 1].set_xlabel("iterations")
    axes[1, 1].set_ylabel("normalized residual")
    axes[1, 1].grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes[1, 1], label=r"$\log_{10}(\mathrm{score})$")
    for x_value, y_value, label in zip(iteration_values, residual_values, labels):
        axes[1, 1].annotate(label, (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)

    return _save_figure(fig, Path(output_prefix), formats)


def plot_fixed_y_quantum_diagnostics_in_tree(
    search_root: str | Path,
    output_dir: str | Path,
    settings: QuantumCriticalSettings,
    formats: Iterable[str],
) -> dict[str, str]:
    """Generate fixed-`y` convergence heatmaps for every matching CSV pair in a tree."""
    written: dict[str, str] = {}
    root = Path(search_root)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    for results_csv in sorted(root.rglob("*fixed_y*tc_nc.csv")):
        status_csv = results_csv.with_name(results_csv.name.replace("tc_nc.csv", "branch_status.csv"))
        base_name = results_csv.stem.removesuffix("_tc_nc")
        output_prefix = destination / f"{base_name}_diagnostics"
        written.update(
            plot_fixed_y_quantum_diagnostics(
                results_csv,
                status_csv if status_csv.exists() else None,
                output_prefix,
                settings,
                formats,
            )
        )
    return written
