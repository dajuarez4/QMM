"""Beta-equilibrium EOS preparation and neutron-star structure solves."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass

from .constants import DEFAULT_NEUTRON_STAR_SETTINGS, NeutronStarSettings
from .quarkyonic import AsymmetricQuarkyonicResult, AsymmetricQuarkyonicRow

MEV_FM3_TO_DYN_CM2 = 1.602176634e33
C_CGS = 2.99792458e10
G_CGS = 6.67430e-8
MSUN_CGS = 1.98847e33
CM_TO_KM = 1.0e-5


@dataclass(frozen=True)
class BetaEquilibriumEOS:
    """Monotonic beta-equilibrium EOS ready for TOV integration."""

    model: str
    parameter_name: str | None
    parameter_value: float | None
    baryon_density_fm3: list[float]
    baryon_density_over_n0: list[float]
    energy_density_mev_fm3: list[float]
    pressure_mev_fm3: list[float]
    proton_fraction: list[float]
    quark_fraction: list[float]
    mu_e_mev: list[float]
    vs2: list[float]


@dataclass(frozen=True)
class NeutronStarPoint:
    """One point on a TOV mass-radius sequence."""

    central_pressure_mev_fm3: float
    central_energy_density_mev_fm3: float
    mass_msun: float
    radius_km: float


@dataclass(frozen=True)
class NeutronStarSequence:
    """A neutron-star mass-radius sequence for one EOS."""

    eos: BetaEquilibriumEOS
    points: list[NeutronStarPoint]
    max_mass_msun: float
    radius_at_max_mass_km: float
    central_pressure_at_max_mass_mev_fm3: float
    central_energy_density_at_max_mass_mev_fm3: float
    max_mass_at_upper_eos_boundary: bool


@dataclass(frozen=True)
class _PreparedToveos:
    pressure_cgs: list[float]
    mass_density_cgs: list[float]
    log_pressure: list[float]
    log_mass_density: list[float]
    pressure_min_cgs: float
    pressure_max_cgs: float


def _logspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [float(start)]
    if start <= 0.0 or stop <= 0.0:
        raise ValueError("Logarithmic grids require positive endpoints.")
    log_start = math.log(start)
    log_stop = math.log(stop)
    step = (log_stop - log_start) / float(count - 1)
    return [math.exp(log_start + step * float(index)) for index in range(count)]


def _interpolate_log(x_value: float, x_grid: list[float], y_grid: list[float], log_x: list[float], log_y: list[float]) -> float:
    if x_value <= x_grid[0]:
        return y_grid[0]
    if x_value >= x_grid[-1]:
        return y_grid[-1]

    index = bisect_left(x_grid, x_value)
    if index <= 0:
        return y_grid[0]
    if index >= len(x_grid):
        return y_grid[-1]

    x_left = x_grid[index - 1]
    x_right = x_grid[index]
    if x_right <= x_left:
        return y_grid[index]

    weight = (math.log(x_value) - log_x[index - 1]) / (log_x[index] - log_x[index - 1])
    return math.exp(log_y[index - 1] + weight * (log_y[index] - log_y[index - 1]))


def _prepare_beta_rows(result: AsymmetricQuarkyonicResult) -> list[AsymmetricQuarkyonicRow]:
    prepared: list[AsymmetricQuarkyonicRow] = []
    last_energy = -math.inf
    last_pressure = -math.inf

    for row in result.beta_profile:
        pressure = row.pressure
        mu_e = row.mu_e
        vs2 = row.vs2
        if pressure is None or mu_e is None or vs2 is None:
            continue
        if (
            not math.isfinite(row.n_b)
            or not math.isfinite(row.energy_density)
            or not math.isfinite(pressure)
            or not math.isfinite(mu_e)
            or row.n_b <= 0.0
            or row.energy_density <= 0.0
            or pressure <= 0.0
        ):
            continue
        if row.energy_density <= last_energy or pressure <= last_pressure:
            continue
        prepared.append(row)
        last_energy = row.energy_density
        last_pressure = pressure

    return prepared


def build_beta_equilibrium_eos(result: AsymmetricQuarkyonicResult) -> BetaEquilibriumEOS:
    """Extract a monotonic beta-equilibrium EOS from a workflow result."""
    rows = _prepare_beta_rows(result)
    if len(rows) < 8:
        raise ValueError(
            f"Need at least 8 monotonic beta-equilibrium EOS points for '{result.model}', found {len(rows)}."
        )

    return BetaEquilibriumEOS(
        model=result.model,
        parameter_name=result.parameter_name,
        parameter_value=result.parameter_value,
        baryon_density_fm3=[row.n_b for row in rows],
        baryon_density_over_n0=[row.n_over_n0 for row in rows],
        energy_density_mev_fm3=[row.energy_density for row in rows],
        pressure_mev_fm3=[float(row.pressure) for row in rows],
        proton_fraction=[row.y for row in rows],
        quark_fraction=[row.quark_fraction for row in rows],
        mu_e_mev=[float(row.mu_e) for row in rows],
        vs2=[float(row.vs2) for row in rows],
    )


def _prepare_tov_eos(eos: BetaEquilibriumEOS) -> _PreparedToveos:
    pressure_cgs = [value * MEV_FM3_TO_DYN_CM2 for value in eos.pressure_mev_fm3]
    mass_density_cgs = [value * MEV_FM3_TO_DYN_CM2 / (C_CGS * C_CGS) for value in eos.energy_density_mev_fm3]
    return _PreparedToveos(
        pressure_cgs=pressure_cgs,
        mass_density_cgs=mass_density_cgs,
        log_pressure=[math.log(value) for value in pressure_cgs],
        log_mass_density=[math.log(value) for value in mass_density_cgs],
        pressure_min_cgs=pressure_cgs[0],
        pressure_max_cgs=pressure_cgs[-1],
    )


def _mass_density_from_pressure(pressure_cgs: float, eos: _PreparedToveos) -> float:
    return _interpolate_log(
        pressure_cgs,
        eos.pressure_cgs,
        eos.mass_density_cgs,
        eos.log_pressure,
        eos.log_mass_density,
    )


def _energy_density_from_pressure_mev_fm3(pressure_mev_fm3: float, eos: BetaEquilibriumEOS) -> float:
    return _interpolate_log(
        pressure_mev_fm3,
        eos.pressure_mev_fm3,
        eos.energy_density_mev_fm3,
        [math.log(value) for value in eos.pressure_mev_fm3],
        [math.log(value) for value in eos.energy_density_mev_fm3],
    )


def _tov_rhs(radius_cm: float, pressure_cgs: float, mass_g: float, eos: _PreparedToveos) -> tuple[float, float] | None:
    if pressure_cgs <= eos.pressure_min_cgs:
        return 0.0, 0.0

    mass_density_cgs = _mass_density_from_pressure(pressure_cgs, eos)
    compactness_factor = radius_cm - 2.0 * G_CGS * mass_g / (C_CGS * C_CGS)
    if compactness_factor <= 0.0 or radius_cm <= 0.0:
        return None

    dPdr = -G_CGS * (mass_density_cgs + pressure_cgs / (C_CGS * C_CGS))
    dPdr *= mass_g + 4.0 * math.pi * radius_cm ** 3 * pressure_cgs / (C_CGS * C_CGS)
    dPdr /= radius_cm * compactness_factor

    dmdr = 4.0 * math.pi * radius_cm ** 2 * mass_density_cgs
    return dPdr, dmdr


def _rk4_step(radius_cm: float, pressure_cgs: float, mass_g: float, step_cm: float, eos: _PreparedToveos) -> tuple[float, float] | None:
    k1 = _tov_rhs(radius_cm, pressure_cgs, mass_g, eos)
    if k1 is None:
        return None

    k2 = _tov_rhs(
        radius_cm + 0.5 * step_cm,
        pressure_cgs + 0.5 * step_cm * k1[0],
        mass_g + 0.5 * step_cm * k1[1],
        eos,
    )
    if k2 is None:
        return None

    k3 = _tov_rhs(
        radius_cm + 0.5 * step_cm,
        pressure_cgs + 0.5 * step_cm * k2[0],
        mass_g + 0.5 * step_cm * k2[1],
        eos,
    )
    if k3 is None:
        return None

    k4 = _tov_rhs(
        radius_cm + step_cm,
        pressure_cgs + step_cm * k3[0],
        mass_g + step_cm * k3[1],
        eos,
    )
    if k4 is None:
        return None

    next_pressure = pressure_cgs + step_cm * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
    next_mass = mass_g + step_cm * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
    return next_pressure, next_mass


def _integrate_star(
    central_pressure_mev_fm3: float,
    eos: BetaEquilibriumEOS,
    prepared_eos: _PreparedToveos,
    settings: NeutronStarSettings,
) -> NeutronStarPoint | None:
    central_pressure_cgs = central_pressure_mev_fm3 * MEV_FM3_TO_DYN_CM2
    surface_pressure_mev_fm3 = settings.surface_pressure_mev_fm3
    if surface_pressure_mev_fm3 is None:
        surface_pressure_mev_fm3 = eos.pressure_mev_fm3[0]
    surface_pressure_cgs = max(surface_pressure_mev_fm3 * MEV_FM3_TO_DYN_CM2, prepared_eos.pressure_min_cgs)

    if central_pressure_cgs <= surface_pressure_cgs:
        return None

    central_energy_density_mev_fm3 = _energy_density_from_pressure_mev_fm3(central_pressure_mev_fm3, eos)
    central_mass_density_cgs = central_energy_density_mev_fm3 * MEV_FM3_TO_DYN_CM2 / (C_CGS * C_CGS)

    radius_cm = settings.start_radius_cm
    mass_g = 4.0 * math.pi * radius_cm ** 3 * central_mass_density_cgs / 3.0
    pressure_cgs = central_pressure_cgs

    previous_radius = radius_cm
    previous_mass = mass_g
    previous_pressure = pressure_cgs

    while radius_cm < settings.max_radius_cm and pressure_cgs > surface_pressure_cgs:
        stepped = _rk4_step(radius_cm, pressure_cgs, mass_g, settings.step_cm, prepared_eos)
        if stepped is None:
            return None

        next_pressure, next_mass = stepped
        next_radius = radius_cm + settings.step_cm
        if not math.isfinite(next_pressure) or not math.isfinite(next_mass) or next_mass <= 0.0:
            return None

        if next_pressure <= surface_pressure_cgs:
            delta_pressure = previous_pressure - next_pressure
            if delta_pressure <= 0.0:
                surface_radius = next_radius
                surface_mass = next_mass
            else:
                weight = (previous_pressure - surface_pressure_cgs) / delta_pressure
                surface_radius = previous_radius + weight * (next_radius - previous_radius)
                surface_mass = previous_mass + weight * (next_mass - previous_mass)
            return NeutronStarPoint(
                central_pressure_mev_fm3=central_pressure_mev_fm3,
                central_energy_density_mev_fm3=central_energy_density_mev_fm3,
                mass_msun=surface_mass / MSUN_CGS,
                radius_km=surface_radius * CM_TO_KM,
            )

        previous_radius = next_radius
        previous_mass = next_mass
        previous_pressure = next_pressure
        radius_cm = next_radius
        mass_g = next_mass
        pressure_cgs = next_pressure

    return None


def solve_neutron_star_sequence(
    result: AsymmetricQuarkyonicResult,
    settings: NeutronStarSettings = DEFAULT_NEUTRON_STAR_SETTINGS,
) -> NeutronStarSequence:
    """Solve the TOV sequence from the beta-equilibrium EOS in one workflow result."""
    eos = build_beta_equilibrium_eos(result)
    prepared_eos = _prepare_tov_eos(eos)

    surface_pressure_mev_fm3 = settings.surface_pressure_mev_fm3
    if surface_pressure_mev_fm3 is None:
        surface_pressure_mev_fm3 = eos.pressure_mev_fm3[0]
    min_pressure = max(
        settings.central_pressure_min_mev_fm3,
        eos.pressure_mev_fm3[0] * 1.05,
        surface_pressure_mev_fm3 * 1.05,
    )
    max_pressure = eos.pressure_mev_fm3[-1] * settings.central_pressure_max_fraction
    if min_pressure >= max_pressure:
        raise ValueError(
            f"Invalid neutron-star pressure window for '{result.model}': {min_pressure:.6g} >= {max_pressure:.6g}."
        )

    central_pressures = _logspace(min_pressure, max_pressure, settings.sequence_points)
    points: list[NeutronStarPoint] = []
    for central_pressure in central_pressures:
        point = _integrate_star(central_pressure, eos, prepared_eos, settings)
        if point is not None:
            points.append(point)

    if len(points) < 3:
        raise ValueError(f"TOV sequence for '{result.model}' produced too few valid stars: {len(points)}.")

    max_index, max_point = max(enumerate(points), key=lambda item: item[1].mass_msun)
    return NeutronStarSequence(
        eos=eos,
        points=points,
        max_mass_msun=max_point.mass_msun,
        radius_at_max_mass_km=max_point.radius_km,
        central_pressure_at_max_mass_mev_fm3=max_point.central_pressure_mev_fm3,
        central_energy_density_at_max_mass_mev_fm3=max_point.central_energy_density_mev_fm3,
        max_mass_at_upper_eos_boundary=max_index == len(points) - 1,
    )
