"""Dense zero-temperature hadronic EOS tables."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import (
    DEFAULT_GROUND_STATE_SETTINGS,
    DEFAULT_HADRONIC_EOS_SETTINGS,
    DEFAULT_PHYSICAL_CONSTANTS,
    GroundStateSettings,
    HadronicEOSSettings,
    PhysicalConstants,
)
from .ground_state import GroundStateResult, energy_per_particle_minus_m, eps_total
from .ground_state import pressure_total as pressure_total_hadronic
from .numerics import linspace
from .sound_speed import reconstruct_sound_speed_curve


@dataclass(frozen=True)
class HadronicEOSRow:
    """One row of a dense hadronic EOS table."""

    n: float
    n_over_n0: float
    eps_raw: float
    eps: float
    e_per_particle_minus_m: float
    mu_b: float
    pressure: float
    dP_dn: float
    d2eps_dn2: float
    vs2: float


@dataclass(frozen=True)
class HadronicEOSTable:
    """Dense hadronic EOS and sound-speed reconstruction."""

    model: str
    parameter_name: str | None
    parameter_value: float | None
    a: float
    b: float
    K0: float
    rows: list[HadronicEOSRow]


def compute_hadronic_eos_table(
    ground_state: GroundStateResult,
    table_settings: HadronicEOSSettings = DEFAULT_HADRONIC_EOS_SETTINGS,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    ground_state_settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> HadronicEOSTable:
    """Build a dense hadronic EOS table from one calibrated ground-state point."""
    density_ratios = linspace(table_settings.n_min_ratio, table_settings.n_max_ratio, table_settings.n_points)
    n_values: list[float] = []
    eps_raw_values: list[float] = []

    for ratio in density_ratios:
        n_value = ratio * physical.n0
        eps_value = eps_total(
            ground_state.model,
            ground_state.a,
            ground_state.b,
            n_value,
            ground_state.parameter_value,
            physical,
            ground_state_settings,
        )
        if eps_value is None:
            continue
        n_values.append(n_value)
        eps_raw_values.append(eps_value)

    if len(n_values) < 5:
        raise ValueError(f"Could not build a hadronic EOS table for model '{ground_state.model}'.")

    sound_speed_curve = reconstruct_sound_speed_curve(
        n_values,
        eps_raw_values,
        table_settings.smoothing_window,
        table_settings.smoothing_degree,
        table_settings.derivative_floor,
    )

    rows: list[HadronicEOSRow] = []
    for index, n_value in enumerate(n_values):
        rows.append(
            HadronicEOSRow(
                n=n_value,
                n_over_n0=n_value / physical.n0,
                eps_raw=eps_raw_values[index],
                eps=sound_speed_curve.energy_density_smoothed[index],
                e_per_particle_minus_m=energy_per_particle_minus_m(
                    ground_state.model,
                    ground_state.a,
                    ground_state.b,
                    n_value,
                    ground_state.parameter_value,
                    physical,
                    ground_state_settings,
                ),
                mu_b=sound_speed_curve.chemical_potential[index],
                pressure=sound_speed_curve.pressure[index],
                dP_dn=sound_speed_curve.dP_dn[index],
                d2eps_dn2=sound_speed_curve.d2eps_dn2[index],
                vs2=sound_speed_curve.vs2[index],
            )
        )

    return HadronicEOSTable(
        model=ground_state.model,
        parameter_name=ground_state.parameter_name,
        parameter_value=ground_state.parameter_value,
        a=ground_state.a,
        b=ground_state.b,
        K0=ground_state.K0,
        rows=rows,
    )
