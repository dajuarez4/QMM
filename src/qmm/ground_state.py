"""Ground-state hadronic fits for the unified workflow."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import (
    DEFAULT_GROUND_STATE_SETTINGS,
    DEFAULT_PHYSICAL_CONSTANTS,
    GroundStateSettings,
    ParameterSearchSettings,
    PhysicalConstants,
)
from .fermi_gas import eps_id_from_nid, kf_from_nid, p_id_from_nid
from .models import get_model
from .numerics import bisection_root, golden_section_min, linspace, numerical_derivative


@dataclass(frozen=True)
class GroundStateResult:
    """Final ground-state output for one model."""

    model: str
    parameter_name: str | None
    parameter_value: float | None
    a: float
    b: float
    K0: float
    n_id: float
    kf: float
    p_id: float
    eps_id: float
    binding: float
    n_sat: float
    e_per_particle_minus_m_at_sat: float
    p_sat: float | None


def n_id_ground_from_b(
    model_name: str,
    b_value: float,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
) -> float | None:
    """Map the physical density `n0` to the ideal-gas density."""
    model = get_model(model_name)
    return model.nid_from_n(physical.n0, b_value)


def pressure_zero_a(
    model_name: str,
    b_value: float,
    parameter_value: float | None,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> float | None:
    """Solve `a` from the zero-pressure condition at saturation."""
    model = get_model(model_name)
    n_id = n_id_ground_from_b(model_name, b_value, physical)
    if n_id is None:
        return None

    p_id = p_id_from_nid(n_id, physical.degeneracy_symmetric, physical, settings)
    pressure_prefactor = model.pressure_prefactor(physical.n0, b_value)
    if pressure_prefactor is None:
        return None

    dU_value = model.dU(physical.n0, b_value, parameter_value)
    if dU_value is None:
        return None

    denom = physical.n0 * physical.n0 * dU_value
    if abs(denom) < 1.0e-14:
        return None
    return -(pressure_prefactor * p_id) / denom


def binding_energy_per_particle(
    model_name: str,
    a_value: float,
    b_value: float,
    parameter_value: float | None,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> float | None:
    """Return `E/A - m` at saturation."""
    model = get_model(model_name)
    n_id = n_id_ground_from_b(model_name, b_value, physical)
    if n_id is None:
        return None
    eps_id = eps_id_from_nid(n_id, physical.degeneracy_symmetric, physical, settings)
    U_value = model.U(physical.n0, b_value, parameter_value)
    if U_value is None:
        return None
    return eps_id / n_id + a_value * U_value - physical.m_nucleon


def binding_residual_for_b(
    model_name: str,
    b_value: float,
    parameter_value: float | None,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> float | None:
    """Return the binding-energy residual used to solve `b`."""
    a_value = pressure_zero_a(model_name, b_value, parameter_value, physical, settings)
    if a_value is None:
        return None
    binding = binding_energy_per_particle(model_name, a_value, b_value, parameter_value, physical, settings)
    if binding is None:
        return None
    return binding - physical.binding_energy


def find_bracket_for_b(
    model_name: str,
    parameter_value: float | None,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> tuple[float | None, float | None]:
    """Scan for a sign-change bracket in `b`."""
    previous_b: float | None = None
    previous_residual: float | None = None

    for b_value in linspace(settings.b_min, settings.b_max, settings.b_scan_steps):
        residual = binding_residual_for_b(model_name, b_value, parameter_value, physical, settings)
        if residual is None:
            previous_b = None
            previous_residual = None
            continue
        if previous_b is not None and previous_residual is not None:
            if residual == 0.0:
                return b_value, b_value
            if previous_residual * residual < 0.0:
                return previous_b, b_value
        previous_b = b_value
        previous_residual = residual
    return None, None


def solve_ab(
    model_name: str,
    parameter_value: float | None,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> tuple[float, float, float, float, float, float] | None:
    """Solve `a`, `b`, `n_id`, `k_F`, `p_id`, and `eps_id`."""
    b_left, b_right = find_bracket_for_b(model_name, parameter_value, physical, settings)
    if b_left is None or b_right is None:
        return None

    def residual(candidate_b: float) -> float | None:
        return binding_residual_for_b(model_name, candidate_b, parameter_value, physical, settings)

    b_value = bisection_root(residual, b_left, b_right)
    if b_value is None:
        return None

    a_value = pressure_zero_a(model_name, b_value, parameter_value, physical, settings)
    if a_value is None:
        return None

    n_id = n_id_ground_from_b(model_name, b_value, physical)
    if n_id is None:
        return None

    kf = kf_from_nid(n_id, physical.degeneracy_symmetric, physical)
    p_id = p_id_from_nid(n_id, physical.degeneracy_symmetric, physical, settings)
    eps_id = eps_id_from_nid(n_id, physical.degeneracy_symmetric, physical, settings)
    binding = binding_energy_per_particle(model_name, a_value, b_value, parameter_value, physical, settings)
    if binding is None:
        return None
    return a_value, b_value, n_id, kf, p_id, eps_id


def eps_total(
    model_name: str,
    a_value: float,
    b_value: float,
    n_value: float,
    parameter_value: float | None,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> float | None:
    """Return the total zero-temperature energy density."""
    model = get_model(model_name)
    if n_value <= 0.0:
        return None

    n_id = model.nid_from_n(n_value, b_value)
    if n_id is None:
        return None
    fraction = model.volume_fraction(n_value, b_value)
    if fraction is None:
        return None
    eps_id = eps_id_from_nid(n_id, physical.degeneracy_symmetric, physical, settings)
    U_value = model.U(n_value, b_value, parameter_value)
    if U_value is None:
        return None
    return fraction * eps_id + n_value * a_value * U_value


def pressure_total(
    model_name: str,
    a_value: float,
    b_value: float,
    n_value: float,
    parameter_value: float | None,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> float | None:
    """Return the total zero-temperature pressure."""
    model = get_model(model_name)
    if n_value <= 0.0:
        return None
    n_id = model.nid_from_n(n_value, b_value)
    if n_id is None:
        return None
    prefactor = model.pressure_prefactor(n_value, b_value)
    if prefactor is None:
        return None
    p_id = p_id_from_nid(n_id, physical.degeneracy_symmetric, physical, settings)
    dU_value = model.dU(n_value, b_value, parameter_value)
    if dU_value is None:
        return None
    return prefactor * p_id + n_value * n_value * a_value * dU_value


def energy_per_particle_minus_m(
    model_name: str,
    a_value: float,
    b_value: float,
    n_value: float,
    parameter_value: float | None,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> float:
    """Objective used to locate the saturation-density minimum."""
    eps_value = eps_total(model_name, a_value, b_value, n_value, parameter_value, physical, settings)
    if eps_value is None or n_value <= 0.0:
        return 1.0e99
    return eps_value / n_value - physical.m_nucleon


def find_saturation_density(
    model_name: str,
    a_value: float,
    b_value: float,
    parameter_value: float | None,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> tuple[float, float, float | None]:
    """Locate the minimum of `E/A - m`."""

    def objective(n_value: float) -> float:
        return energy_per_particle_minus_m(
            model_name,
            a_value,
            b_value,
            n_value,
            parameter_value,
            physical,
            settings,
        )

    n_sat, value = golden_section_min(objective, settings.saturation_left, settings.saturation_right)
    p_sat = pressure_total(model_name, a_value, b_value, n_sat, parameter_value, physical, settings)
    return n_sat, value, p_sat


def incompressibility_k0(
    model_name: str,
    a_value: float,
    b_value: float,
    parameter_value: float | None,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> float:
    """Return `K0 = 9 dP/dn |_(n0)`."""

    def pressure_at_density(n_value: float) -> float:
        value = pressure_total(model_name, a_value, b_value, n_value, parameter_value, physical, settings)
        return math.nan if value is None else value

    return 9.0 * numerical_derivative(pressure_at_density, physical.n0, settings.derivative_step)


def compute_ground_state_point_explicit(
    model_name: str,
    parameter_value: float | None,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> GroundStateResult | None:
    """Compute one ground-state point from an explicit model parameter."""
    model = get_model(model_name)
    if model.parameter_name is not None and parameter_value is None:
        return None

    solved = solve_ab(model_name, parameter_value, physical, settings)
    if solved is None:
        return None
    a_value, b_value, n_id, kf, p_id, eps_id = solved
    binding = binding_energy_per_particle(model_name, a_value, b_value, parameter_value, physical, settings)
    if binding is None:
        return None
    K0 = incompressibility_k0(model_name, a_value, b_value, parameter_value, physical, settings)
    n_sat, e_sat, p_sat = find_saturation_density(model_name, a_value, b_value, parameter_value, physical, settings)
    return GroundStateResult(
        model=model_name,
        parameter_name=model.parameter_name,
        parameter_value=parameter_value,
        a=a_value,
        b=b_value,
        K0=K0,
        n_id=n_id,
        kf=kf,
        p_id=p_id,
        eps_id=eps_id,
        binding=binding,
        n_sat=n_sat,
        e_per_particle_minus_m_at_sat=e_sat,
        p_sat=p_sat,
    )


def solve_parameter_from_target_k0(
    model_name: str,
    search: ParameterSearchSettings,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> float | None:
    """Solve the model parameter from a target incompressibility."""
    model = get_model(model_name)
    if model.parameter_name is None:
        return None
    if search.target_k0 is None:
        return None

    lower = search.parameter_min if search.parameter_min is not None else model.parameter_range[0]
    upper = search.parameter_max if search.parameter_max is not None else model.parameter_range[1]
    previous_parameter: float | None = None
    previous_residual: float | None = None

    for parameter_value in linspace(lower, upper, search.scan_steps):
        point = compute_ground_state_point_explicit(model_name, parameter_value, physical, settings)
        if point is None:
            previous_parameter = None
            previous_residual = None
            continue
        residual = point.K0 - search.target_k0
        if previous_parameter is not None and previous_residual is not None:
            if residual == 0.0:
                return parameter_value
            if previous_residual * residual < 0.0:
                return bisection_root(
                    lambda candidate: (
                        None
                        if (candidate_point := compute_ground_state_point_explicit(model_name, candidate, physical, settings)) is None
                        else candidate_point.K0 - search.target_k0
                    ),
                    previous_parameter,
                    parameter_value,
                    tol=search.tol,
                )
        previous_parameter = parameter_value
        previous_residual = residual
    return None


def resolve_parameter_value(
    model_name: str,
    parameter_value: float | None,
    parameter_search: ParameterSearchSettings | None = None,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> float | None:
    """Resolve the effective model parameter for the selected model."""
    model = get_model(model_name)
    if model.parameter_name is None:
        return None
    if parameter_value is not None:
        return parameter_value
    if parameter_search is None or not parameter_search.enabled:
        raise ValueError(
            f"Model '{model_name}' requires '{model.parameter_name}'. "
            "Provide `parameter_value` or enable `parameter_search`."
        )
    resolved = solve_parameter_from_target_k0(model_name, parameter_search, physical, settings)
    if resolved is None:
        raise ValueError(
            f"Could not solve {model.parameter_name} for model '{model_name}' "
            f"from target K0={parameter_search.target_k0} MeV."
        )
    return resolved


def compute_ground_state_point(
    model_name: str,
    parameter_value: float | None = None,
    parameter_search: ParameterSearchSettings | None = None,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> GroundStateResult:
    """Compute the ground-state point, resolving the model parameter if needed."""
    resolved_parameter = resolve_parameter_value(model_name, parameter_value, parameter_search, physical, settings)
    result = compute_ground_state_point_explicit(model_name, resolved_parameter, physical, settings)
    if result is None:
        raise ValueError(f"Ground-state solve failed for model '{model_name}'.")
    return result
