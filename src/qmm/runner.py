"""Top-level orchestration for workflow runs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .asymmetry import AsymmetricFitResult, compute_asymmetric_fit
from .config import RunConfig
from .hadronic_eos import HadronicEOSTable, compute_hadronic_eos_table
from .ground_state import GroundStateResult, compute_ground_state_point
from .plotting import (
    plot_beta_equilibrium_observables,
    plot_asymmetric_profiles,
    plot_fixed_y_quantum_diagnostics_in_tree,
    plot_ground_state_eos,
    plot_neutron_star_sequence,
    quantum_collection_rows,
    plot_quantum_collection,
    plot_quantum_diagnostics,
    plot_symmetric_quarkyonic_bundle,
    write_quantum_collection_csv,
)
from .neutron_star import NeutronStarSequence, solve_neutron_star_sequence
from .quantum import QuantumCriticalPoint, QuantumPressureEvaluator, solve_quantum_critical_point
from .quarkyonic import (
    AsymmetricQuarkyonicResult,
    SymmetricQuarkyonicCurve,
    compute_asymmetric_quarkyonic_profiles,
    compute_symmetric_quarkyonic_curve,
)
from .reporting import write_csv_rows, write_json


def _ground_state_row(result: GroundStateResult) -> dict[str, Any]:
    return asdict(result)


def _quantum_row(result: QuantumCriticalPoint) -> dict[str, Any]:
    return asdict(result)


def _hadronic_eos_rows(table: HadronicEOSTable) -> list[dict[str, Any]]:
    return [
        {
            "model": table.model,
            "parameter_name": table.parameter_name,
            "parameter_value": table.parameter_value,
            "a": table.a,
            "b": table.b,
            "K0": table.K0,
            "n": row.n,
            "n_over_n0": row.n_over_n0,
            "eps_raw": row.eps_raw,
            "eps": row.eps,
            "e_per_particle_minus_m": row.e_per_particle_minus_m,
            "mu_b": row.mu_b,
            "pressure": row.pressure,
            "dP_dn": row.dP_dn,
            "d2eps_dn2": row.d2eps_dn2,
            "vs2": row.vs2,
        }
        for row in table.rows
    ]


def _asymmetric_row(result: AsymmetricFitResult) -> dict[str, Any]:
    return asdict(result)


def _symmetric_curve_rows(curve: SymmetricQuarkyonicCurve) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(len(curve.n)):
        rows.append(
            {
                "model": curve.model,
                "momentum_mode": curve.momentum_mode,
                "parameter_name": curve.parameter_name,
                "parameter_value": curve.parameter_value,
                "a": curve.a,
                "b": curve.b,
                "K0": curve.K0,
                "n": curve.n[index],
                "n_over_n0": curve.n_over_n0[index],
                "eps_raw": curve.eps_raw[index],
                "eps": curve.eps[index],
                "mu_b": curve.mu_b[index],
                "pressure": curve.P[index],
                "dP_dn": curve.dP_dn[index],
                "d2eps_dn2": curve.d2eps_dn2[index],
                "vs2": curve.vs2[index],
                "quark_fraction": curve.quark_fraction[index],
                "n_q": curve.n_q[index],
                "n_n": curve.n_n[index],
                "k_bu": curve.k_bu[index],
                "k_f": curve.k_f[index],
            }
        )
    return rows


def _asymmetric_profile_rows(
    rows,
    model_name: str,
    momentum_mode: str,
    parameter_name: str | None,
    parameter_value: float | None,
) -> list[dict[str, Any]]:
    return [
        {
            "model": model_name,
            "momentum_mode": momentum_mode,
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "n_b": row.n_b,
            "n_over_n0": row.n_over_n0,
            "y": row.y,
            "quark_fraction": row.quark_fraction,
            "k_bu": row.k_bu,
            "k_fp": row.k_fp,
            "k_fn": row.k_fn,
            "n_p": row.n_p,
            "n_n": row.n_n,
            "energy_density": row.energy_density,
            "mu_e": row.mu_e,
            "eps_e": row.eps_e,
            "eps_mu": row.eps_mu,
            "mu_b": row.mu_b,
            "pressure": row.pressure,
            "vs2": row.vs2,
        }
        for row in rows
    ]


def _neutron_star_rows(sequence: NeutronStarSequence) -> list[dict[str, Any]]:
    return [
        {
            "model": sequence.eos.model,
            "parameter_name": sequence.eos.parameter_name,
            "parameter_value": sequence.eos.parameter_value,
            "central_pressure_mev_fm3": point.central_pressure_mev_fm3,
            "central_energy_density_mev_fm3": point.central_energy_density_mev_fm3,
            "mass_msun": point.mass_msun,
            "radius_km": point.radius_km,
        }
        for point in sequence.points
    ]


def run_workflow(config: RunConfig) -> dict[str, Any]:
    """Execute the selected workflows and write outputs."""
    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_formats = tuple(config.output.plot_formats)

    summary: dict[str, Any] = {
        "run_name": config.run_name,
        "model": {
            "name": config.model_name,
            "parameter_value": config.parameter_value,
            "parameter_search": asdict(config.parameter_search),
        },
        "workflows": asdict(config.workflows),
        "outputs": {},
    }

    ground_state: GroundStateResult | None = None
    quantum: QuantumCriticalPoint | None = None
    hadronic_eos_table: HadronicEOSTable | None = None
    symmetric_quarkyonic: SymmetricQuarkyonicCurve | None = None
    asymmetric_fit: AsymmetricFitResult | None = None
    asymmetric_quarkyonic: AsymmetricQuarkyonicResult | None = None
    neutron_star: NeutronStarSequence | None = None

    needs_ground_state = any(
        [
            config.workflows.ground_state,
            config.workflows.quantum_critical,
            config.workflows.hadronic_eos_table,
            config.workflows.symmetric_quarkyonic,
            config.workflows.asymmetric_fit,
            config.workflows.asymmetric_quarkyonic,
            config.workflows.neutron_star,
        ]
    )
    if needs_ground_state:
        ground_state = compute_ground_state_point(
            config.model_name,
            parameter_value=config.parameter_value,
            parameter_search=config.parameter_search,
            physical=config.physical,
            settings=config.ground_state,
        )
        summary["ground_state"] = asdict(ground_state)
        if config.output.write_csv:
            path = output_dir / f"{config.run_name}_ground_state.csv"
            write_csv_rows(path, [_ground_state_row(ground_state)])
            summary["outputs"]["ground_state_csv"] = str(path)

    if config.workflows.hadronic_eos_table:
        hadronic_eos_table = compute_hadronic_eos_table(
            ground_state,
            table_settings=config.hadronic_eos,
            physical=config.physical,
            ground_state_settings=config.ground_state,
        )
        summary["hadronic_eos_table"] = {
            "model": hadronic_eos_table.model,
            "parameter_name": hadronic_eos_table.parameter_name,
            "parameter_value": hadronic_eos_table.parameter_value,
            "a": hadronic_eos_table.a,
            "b": hadronic_eos_table.b,
            "K0": hadronic_eos_table.K0,
            "points": len(hadronic_eos_table.rows),
        }
        if config.output.write_csv:
            path = output_dir / f"{config.run_name}_hadronic_eos.csv"
            write_csv_rows(path, _hadronic_eos_rows(hadronic_eos_table))
            summary["outputs"]["hadronic_eos_csv"] = str(path)

    if config.workflows.quantum_critical:
        evaluator = QuantumPressureEvaluator(ground_state, config.physical, config.quantum)
        quantum = solve_quantum_critical_point(evaluator, config.quantum)
        if quantum is None:
            raise ValueError(f"Quantum critical-point solve failed for model '{config.model_name}'.")
        summary["quantum_critical"] = asdict(quantum)
        if config.output.write_csv:
            path = output_dir / f"{config.run_name}_quantum_critical.csv"
            write_csv_rows(path, [_quantum_row(quantum)])
            summary["outputs"]["quantum_critical_csv"] = str(path)

    if config.workflows.symmetric_quarkyonic:
        symmetric_quarkyonic = compute_symmetric_quarkyonic_curve(
            config.model_name,
            parameter_value=config.parameter_value,
            parameter_search=config.parameter_search,
            settings=config.quarkyonic,
            physical=config.physical,
            gs_settings=config.ground_state,
        )
        summary["symmetric_quarkyonic"] = {
            "model": symmetric_quarkyonic.model,
            "momentum_mode": symmetric_quarkyonic.momentum_mode,
            "parameter_name": symmetric_quarkyonic.parameter_name,
            "parameter_value": symmetric_quarkyonic.parameter_value,
            "a": symmetric_quarkyonic.a,
            "b": symmetric_quarkyonic.b,
            "K0": symmetric_quarkyonic.K0,
            "vs2_max": symmetric_quarkyonic.vs2_max,
            "n_tr": symmetric_quarkyonic.n_tr,
            "n_tr_over_n0": symmetric_quarkyonic.n_tr_over_n0,
        }
        if config.output.write_csv:
            path = output_dir / f"{config.run_name}_symmetric_quarkyonic_curve.csv"
            write_csv_rows(path, _symmetric_curve_rows(symmetric_quarkyonic))
            summary["outputs"]["symmetric_quarkyonic_csv"] = str(path)

    if config.workflows.asymmetric_fit or config.workflows.asymmetric_quarkyonic or config.workflows.neutron_star:
        asymmetric_fit = compute_asymmetric_fit(
            config.model_name,
            parameter_value=config.parameter_value,
            parameter_search=config.parameter_search,
            fit_settings=config.asymmetric,
            physical=config.physical,
            ground_state_settings=config.ground_state,
        )
        summary["asymmetric_fit"] = asdict(asymmetric_fit)
        if config.output.write_csv:
            path = output_dir / f"{config.run_name}_asymmetric_fit.csv"
            write_csv_rows(path, [_asymmetric_row(asymmetric_fit)])
            summary["outputs"]["asymmetric_fit_csv"] = str(path)

    if config.workflows.asymmetric_quarkyonic:
        asymmetric_quarkyonic = compute_asymmetric_quarkyonic_profiles(
            asymmetric_fit,
            proton_fraction_values=config.asymmetric.proton_fraction_values,
            include_beta_equilibrium=config.asymmetric.beta_equilibrium,
            settings=config.quarkyonic,
            physical=config.physical,
        )
        summary["asymmetric_quarkyonic"] = {
            "model": asymmetric_quarkyonic.model,
            "momentum_mode": asymmetric_quarkyonic.momentum_mode,
            "parameter_name": asymmetric_quarkyonic.parameter_name,
            "parameter_value": asymmetric_quarkyonic.parameter_value,
            "fixed_y_labels": sorted(asymmetric_quarkyonic.fixed_y_profiles.keys()),
            "beta_profile_points": len(asymmetric_quarkyonic.beta_profile),
        }
        if config.output.write_csv:
            for label, rows in asymmetric_quarkyonic.fixed_y_profiles.items():
                path = output_dir / f"{config.run_name}_asymmetric_{label}.csv"
                write_csv_rows(
                    path,
                    _asymmetric_profile_rows(
                        rows,
                        asymmetric_quarkyonic.model,
                        asymmetric_quarkyonic.momentum_mode,
                        asymmetric_quarkyonic.parameter_name,
                        asymmetric_quarkyonic.parameter_value,
                    ),
                )
                summary["outputs"][f"asymmetric_{label}_csv"] = str(path)
            if asymmetric_quarkyonic.beta_profile:
                path = output_dir / f"{config.run_name}_asymmetric_beta.csv"
                write_csv_rows(
                    path,
                    _asymmetric_profile_rows(
                        asymmetric_quarkyonic.beta_profile,
                        asymmetric_quarkyonic.model,
                        asymmetric_quarkyonic.momentum_mode,
                        asymmetric_quarkyonic.parameter_name,
                        asymmetric_quarkyonic.parameter_value,
                    ),
                )
                summary["outputs"]["asymmetric_beta_csv"] = str(path)

    if config.workflows.neutron_star:
        if asymmetric_quarkyonic is None:
            raise ValueError("The neutron-star workflow requires `asymmetric_quarkyonic` to be enabled.")
        if not asymmetric_quarkyonic.beta_profile:
            raise ValueError("The neutron-star workflow requires `asymmetric.beta_equilibrium = true`.")
        neutron_star = solve_neutron_star_sequence(
            asymmetric_quarkyonic,
            settings=config.neutron_star,
        )
        summary["neutron_star"] = {
            "model": neutron_star.eos.model,
            "parameter_name": neutron_star.eos.parameter_name,
            "parameter_value": neutron_star.eos.parameter_value,
            "points": len(neutron_star.points),
            "max_mass_msun": neutron_star.max_mass_msun,
            "radius_at_max_mass_km": neutron_star.radius_at_max_mass_km,
            "central_pressure_at_max_mass_mev_fm3": neutron_star.central_pressure_at_max_mass_mev_fm3,
            "central_energy_density_at_max_mass_mev_fm3": neutron_star.central_energy_density_at_max_mass_mev_fm3,
            "max_mass_at_upper_eos_boundary": neutron_star.max_mass_at_upper_eos_boundary,
        }
        if config.output.write_csv:
            path = output_dir / f"{config.run_name}_neutron_star_sequence.csv"
            write_csv_rows(path, _neutron_star_rows(neutron_star))
            summary["outputs"]["neutron_star_sequence_csv"] = str(path)

    if config.output.write_json:
        summary_path = output_dir / f"{config.run_name}_summary.json"
        summary["outputs"]["summary_json"] = str(summary_path)
        write_json(summary_path, summary)

    if config.output.write_plots:
        if ground_state is not None:
            summary["outputs"].update(
                plot_ground_state_eos(
                    output_dir,
                    config.run_name,
                    ground_state,
                    config.physical,
                    config.ground_state,
                    plot_formats,
                )
            )

        if quantum is not None:
            summary["outputs"].update(
                plot_quantum_diagnostics(
                    output_dir,
                    config.run_name,
                    quantum,
                    config.quantum,
                    plot_formats,
                )
            )

        if symmetric_quarkyonic is not None:
            summary["outputs"].update(
                plot_symmetric_quarkyonic_bundle(
                    output_dir,
                    config.run_name,
                    symmetric_quarkyonic,
                    config.physical,
                    plot_formats,
                )
            )

        if asymmetric_quarkyonic is not None:
            summary["outputs"].update(
                plot_asymmetric_profiles(
                    output_dir,
                    config.run_name,
                    asymmetric_quarkyonic,
                    config.physical,
                    plot_formats,
                )
            )
            summary["outputs"].update(
                plot_beta_equilibrium_observables(
                    output_dir,
                    config.run_name,
                    asymmetric_quarkyonic,
                    config.physical,
                    plot_formats,
                )
            )

        if neutron_star is not None:
            summary["outputs"].update(
                plot_neutron_star_sequence(
                    output_dir,
                    config.run_name,
                    neutron_star,
                    plot_formats,
                )
            )

        collection_search_root = (
            Path(config.output.collection_search_root)
            if config.output.collection_search_root is not None
            else output_dir.parent
        )
        collection_output_dir = (
            Path(config.output.collection_output_directory)
            if config.output.collection_output_directory is not None
            else collection_search_root
        )
        collection_output_dir.mkdir(parents=True, exist_ok=True)

        collection_rows = quantum_collection_rows(collection_search_root, config.quantum)
        collection_csv_path = collection_output_dir / "collection_quantum_summary.csv"
        collection_csv = write_quantum_collection_csv(collection_rows, collection_csv_path)
        if collection_csv is not None:
            summary["outputs"]["collection_quantum_summary_csv"] = collection_csv
        summary["outputs"].update(
            plot_quantum_collection(
                collection_rows,
                collection_output_dir / "collection_quantum_summary",
                plot_formats,
            )
        )
        summary["outputs"].update(
            plot_fixed_y_quantum_diagnostics_in_tree(
                collection_search_root,
                collection_output_dir,
                config.quantum,
                plot_formats,
            )
        )

        if config.output.write_json:
            summary_path = output_dir / f"{config.run_name}_summary.json"
            write_json(summary_path, summary)

    return summary
