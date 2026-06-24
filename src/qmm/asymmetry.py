"""Asymmetric hadronic fits at zero temperature."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import DEFAULT_ASYMMETRIC_SETTINGS, DEFAULT_GROUND_STATE_SETTINGS, DEFAULT_PHYSICAL_CONSTANTS, AsymmetricFitSettings, GroundStateSettings, PhysicalConstants
from .fermi_gas import eps_id_from_nid
from .ground_state import GroundStateResult, compute_ground_state_point
from .models import get_model
from .numerics import bisection_root, linspace, numerical_derivative


@dataclass(frozen=True)
class AsymmetricFitResult:
    """Full asymmetric fit derived from one symmetric ground-state point."""

    model: str
    branch_mode: str
    parameter_name: str | None
    parameter_value: float | None
    target_k0: float | None
    a_avg: float
    b_avg: float
    a_n: float
    a_pn: float
    b_n: float
    b_pn: float
    ratio_a_pn_over_a_n: float
    ratio_b_pn_over_b_n: float
    J: float
    L: float
    K0: float
    binding: float
    n_sat: float
    p_sat: float | None
    j_residual: float
    l_residual: float
    k0_residual: float


def _ensure_asymmetric_support(model_name: str) -> None:
    model = get_model(model_name)
    if not model.supports_asymmetric_hadronic:
        raise ValueError(
            f"Model '{model_name}' does not have a defined asymmetric hadronic extension "
            "in the provided references. Supported asymmetric models are: "
            "vdw, clausius, clausius_cs, clausius_tvm, dieterici, cs, tvm."
        )


def _validate_branch_mode(branch_mode: str) -> str:
    normalized = str(branch_mode).strip().lower()
    if normalized in {"target_l", "unequal_b", "split_b"}:
        return "target_l"
    if normalized in {"equal_b", "eq", "delta_b_zero"}:
        return "equal_b"
    raise ValueError(
        "Unsupported asymmetric branch_mode. Use 'target_l' for the "
        "b_pn != b_n branch or 'equal_b' for the b_pn = b_n branch."
    )


def attractive_coefficient(y_value: float, a_n: float, a_pn: float) -> float:
    """Return the composition-dependent attractive coefficient."""
    return a_n * (y_value * y_value + (1.0 - y_value) ** 2) + 2.0 * a_pn * y_value * (1.0 - y_value)


def asymmetric_species_data(
    model_name: str,
    n_value: float,
    y_value: float,
    b_n: float,
    b_pn: float,
) -> tuple[float, float, float, float, float, float] | None:
    """Return species densities, fractions, and ideal densities."""
    model = get_model(model_name)
    if n_value <= 0.0 or y_value < 0.0 or y_value > 0.5:
        return None

    n_p = n_value * y_value
    n_n = n_value * (1.0 - y_value)
    x_p = b_n * n_p + b_pn * n_n
    x_n = b_pn * n_p + b_n * n_n
    f_p = model.species_volume_fraction(x_p)
    f_n = model.species_volume_fraction(x_n)
    if f_p is None or f_n is None:
        return None

    n_p_id = 0.0 if n_p <= 0.0 else n_p / f_p
    n_n_id = 0.0 if n_n <= 0.0 else n_n / f_n
    return n_p, n_n, f_p, f_n, n_p_id, n_n_id


def asymmetric_energy_density(
    model_name: str,
    n_value: float,
    y_value: float,
    a_n: float,
    a_pn: float,
    b_n: float,
    b_pn: float,
    parameter_value: float | None,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> float | None:
    """Return the asymmetric hadronic energy density."""
    _ensure_asymmetric_support(model_name)
    model = get_model(model_name)
    data = asymmetric_species_data(model_name, n_value, y_value, b_n, b_pn)
    if data is None:
        return None
    n_p, n_n, f_p, f_n, n_p_id, n_n_id = data

    eps_p = f_p * eps_id_from_nid(n_p_id, physical.degeneracy_species, physical, settings)
    eps_n = f_n * eps_id_from_nid(n_n_id, physical.degeneracy_species, physical, settings)

    U_value = model.U(n_value, 0.0, parameter_value)
    if U_value is None:
        return None

    a_eff = attractive_coefficient(y_value, a_n, a_pn)
    return eps_p + eps_n + n_value * a_eff * U_value


def asymmetric_energy_per_particle_minus_m(
    model_name: str,
    n_value: float,
    y_value: float,
    a_n: float,
    a_pn: float,
    b_n: float,
    b_pn: float,
    parameter_value: float | None,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> float | None:
    """Return `E/A - m` for asymmetric matter."""
    eps_value = asymmetric_energy_density(
        model_name,
        n_value,
        y_value,
        a_n,
        a_pn,
        b_n,
        b_pn,
        parameter_value,
        physical,
        settings,
    )
    if eps_value is None or n_value <= 0.0:
        return None
    return eps_value / n_value - physical.m_nucleon


def symmetry_energy_parabolic(
    model_name: str,
    n_value: float,
    a_n: float,
    a_pn: float,
    b_n: float,
    b_pn: float,
    parameter_value: float | None,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> float | None:
    """Return `S(n) = E_PNM(n) - E_SNM(n)`."""
    e_pnm = asymmetric_energy_per_particle_minus_m(
        model_name,
        n_value,
        0.0,
        a_n,
        a_pn,
        b_n,
        b_pn,
        parameter_value,
        physical,
        settings,
    )
    e_snm = asymmetric_energy_per_particle_minus_m(
        model_name,
        n_value,
        0.5,
        a_n,
        a_pn,
        b_n,
        b_pn,
        parameter_value,
        physical,
        settings,
    )
    if e_pnm is None or e_snm is None:
        return None
    return e_pnm - e_snm


def symmetry_slope_l(
    model_name: str,
    a_n: float,
    a_pn: float,
    b_n: float,
    b_pn: float,
    parameter_value: float | None,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: AsymmetricFitSettings = DEFAULT_ASYMMETRIC_SETTINGS,
    ground_state_settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> float:
    """Return `L = 3 n0 dS/dn |_(n0)`."""

    def symmetry_at_density(n_value: float) -> float:
        value = symmetry_energy_parabolic(
            model_name,
            n_value,
            a_n,
            a_pn,
            b_n,
            b_pn,
            parameter_value,
            physical,
            ground_state_settings,
        )
        return math.nan if value is None else value

    return 3.0 * physical.n0 * numerical_derivative(symmetry_at_density, physical.n0, settings.derivative_step_l)


def split_parameters_from_averages(a_avg: float, b_avg: float, delta_a: float, delta_b: float) -> dict[str, float]:
    """Map average and split variables to physical asymmetric parameters."""
    return {
        "a_n": a_avg - delta_a,
        "a_pn": a_avg + delta_a,
        "b_n": b_avg - delta_b,
        "b_pn": b_avg + delta_b,
    }


def solve_delta_a_for_j(
    model_name: str,
    ground_state: GroundStateResult,
    delta_b: float,
    fit_settings: AsymmetricFitSettings = DEFAULT_ASYMMETRIC_SETTINGS,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    ground_state_settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> float | None:
    """Solve `delta_a` from the symmetry energy constraint."""
    limit = 0.995 * ground_state.a
    previous_delta_a: float | None = None
    previous_residual: float | None = None

    for delta_a in linspace(-limit, limit, fit_settings.delta_scan_steps):
        params = split_parameters_from_averages(ground_state.a, ground_state.b, delta_a, delta_b)
        if min(params["a_n"], params["a_pn"], params["b_n"], params["b_pn"]) <= 0.0:
            previous_delta_a = None
            previous_residual = None
            continue
        j_value = symmetry_energy_parabolic(
            model_name,
            physical.n0,
            params["a_n"],
            params["a_pn"],
            params["b_n"],
            params["b_pn"],
            ground_state.parameter_value,
            physical,
            ground_state_settings,
        )
        if j_value is None:
            previous_delta_a = None
            previous_residual = None
            continue
        residual = j_value - fit_settings.target_j
        if previous_delta_a is not None and previous_residual is not None:
            if residual == 0.0:
                return delta_a
            if previous_residual * residual < 0.0:
                return bisection_root(
                    lambda candidate_delta_a: (
                        None
                        if min(
                            split_parameters_from_averages(
                                ground_state.a,
                                ground_state.b,
                                candidate_delta_a,
                                delta_b,
                            ).values()
                        )
                        <= 0.0
                        else symmetry_energy_parabolic(
                            model_name,
                            physical.n0,
                            split_parameters_from_averages(
                                ground_state.a,
                                ground_state.b,
                                candidate_delta_a,
                                delta_b,
                            )["a_n"],
                            split_parameters_from_averages(
                                ground_state.a,
                                ground_state.b,
                                candidate_delta_a,
                                delta_b,
                            )["a_pn"],
                            split_parameters_from_averages(
                                ground_state.a,
                                ground_state.b,
                                candidate_delta_a,
                                delta_b,
                            )["b_n"],
                            split_parameters_from_averages(
                                ground_state.a,
                                ground_state.b,
                                candidate_delta_a,
                                delta_b,
                            )["b_pn"],
                            ground_state.parameter_value,
                            physical,
                            ground_state_settings,
                        )
                        - fit_settings.target_j
                    ),
                    previous_delta_a,
                    delta_a,
                    tol=1.0e-8,
                )
        previous_delta_a = delta_a
        previous_residual = residual
    return None


def solve_delta_b_for_l(
    model_name: str,
    ground_state: GroundStateResult,
    fit_settings: AsymmetricFitSettings = DEFAULT_ASYMMETRIC_SETTINGS,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    ground_state_settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> tuple[float | None, float | None]:
    """Solve `(delta_a, delta_b)` from the `J` and `L` constraints."""
    limit = 0.995 * ground_state.b
    previous_delta_b: float | None = None
    previous_residual: float | None = None

    for delta_b in linspace(-limit, limit, fit_settings.delta_scan_steps):
        if ground_state.b - delta_b <= 0.0 or ground_state.b + delta_b <= 0.0:
            previous_delta_b = None
            previous_residual = None
            continue

        delta_a = solve_delta_a_for_j(model_name, ground_state, delta_b, fit_settings, physical, ground_state_settings)
        if delta_a is None:
            previous_delta_b = None
            previous_residual = None
            continue

        params = split_parameters_from_averages(ground_state.a, ground_state.b, delta_a, delta_b)
        l_value = symmetry_slope_l(
            model_name,
            params["a_n"],
            params["a_pn"],
            params["b_n"],
            params["b_pn"],
            ground_state.parameter_value,
            physical,
            fit_settings,
            ground_state_settings,
        )
        if math.isnan(l_value):
            previous_delta_b = None
            previous_residual = None
            continue

        residual = l_value - fit_settings.target_l
        if previous_delta_b is not None and previous_residual is not None:
            if residual == 0.0:
                return delta_a, delta_b
            if previous_residual * residual < 0.0:
                root_delta_b = bisection_root(
                    lambda candidate_delta_b: (
                        None
                        if ground_state.b - candidate_delta_b <= 0.0 or ground_state.b + candidate_delta_b <= 0.0
                        else (
                            None
                            if (candidate_delta_a := solve_delta_a_for_j(
                                model_name,
                                ground_state,
                                candidate_delta_b,
                                fit_settings,
                                physical,
                                ground_state_settings,
                            ))
                            is None
                            else symmetry_slope_l(
                                model_name,
                                split_parameters_from_averages(
                                    ground_state.a,
                                    ground_state.b,
                                    candidate_delta_a,
                                    candidate_delta_b,
                                )["a_n"],
                                split_parameters_from_averages(
                                    ground_state.a,
                                    ground_state.b,
                                    candidate_delta_a,
                                    candidate_delta_b,
                                )["a_pn"],
                                split_parameters_from_averages(
                                    ground_state.a,
                                    ground_state.b,
                                    candidate_delta_a,
                                    candidate_delta_b,
                                )["b_n"],
                                split_parameters_from_averages(
                                    ground_state.a,
                                    ground_state.b,
                                    candidate_delta_a,
                                    candidate_delta_b,
                                )["b_pn"],
                                ground_state.parameter_value,
                                physical,
                                fit_settings,
                                ground_state_settings,
                            )
                            - fit_settings.target_l
                        )
                    ),
                    previous_delta_b,
                    delta_b,
                    tol=1.0e-8,
                )
                if root_delta_b is None:
                    return None, None
                root_delta_a = solve_delta_a_for_j(
                    model_name,
                    ground_state,
                    root_delta_b,
                    fit_settings,
                    physical,
                    ground_state_settings,
                )
                return root_delta_a, root_delta_b

        previous_delta_b = delta_b
        previous_residual = residual
    return None, None


def compute_asymmetric_fit(
    model_name: str,
    parameter_value: float | None = None,
    parameter_search=None,
    fit_settings: AsymmetricFitSettings = DEFAULT_ASYMMETRIC_SETTINGS,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    ground_state_settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> AsymmetricFitResult:
    """Compute the asymmetric hadronic fit for one supported model."""
    _ensure_asymmetric_support(model_name)
    branch_mode = _validate_branch_mode(fit_settings.branch_mode)
    ground_state = compute_ground_state_point(
        model_name,
        parameter_value=parameter_value,
        parameter_search=parameter_search,
        physical=physical,
        settings=ground_state_settings,
    )

    if fit_settings.target_k0 is not None and get_model(model_name).parameter_name is None:
        if abs(ground_state.K0 - fit_settings.target_k0) > 1.0:
            raise ValueError(
                f"Model '{model_name}' has no extra parameter to tune K0. "
                f"Its fitted symmetric value is K0={ground_state.K0:.3f} MeV, "
                f"which does not match requested target_k0={fit_settings.target_k0:.3f} MeV."
            )

    if branch_mode == "equal_b":
        delta_b = 0.0
        delta_a = solve_delta_a_for_j(
            model_name,
            ground_state,
            delta_b,
            fit_settings,
            physical,
            ground_state_settings,
        )
    else:
        delta_a, delta_b = solve_delta_b_for_l(
            model_name,
            ground_state,
            fit_settings,
            physical,
            ground_state_settings,
        )
    if delta_a is None or delta_b is None:
        raise ValueError(f"Asymmetric fit failed for model '{model_name}' with branch_mode='{branch_mode}'.")

    params = split_parameters_from_averages(ground_state.a, ground_state.b, delta_a, delta_b)
    j_value = symmetry_energy_parabolic(
        model_name,
        physical.n0,
        params["a_n"],
        params["a_pn"],
        params["b_n"],
        params["b_pn"],
        ground_state.parameter_value,
        physical,
        ground_state_settings,
    )
    if j_value is None:
        raise ValueError(f"Asymmetric symmetry-energy evaluation failed for model '{model_name}'.")
    l_value = symmetry_slope_l(
        model_name,
        params["a_n"],
        params["a_pn"],
        params["b_n"],
        params["b_pn"],
        ground_state.parameter_value,
        physical,
        fit_settings,
        ground_state_settings,
    )

    return AsymmetricFitResult(
        model=model_name,
        branch_mode=branch_mode,
        parameter_name=ground_state.parameter_name,
        parameter_value=ground_state.parameter_value,
        target_k0=fit_settings.target_k0,
        a_avg=ground_state.a,
        b_avg=ground_state.b,
        a_n=params["a_n"],
        a_pn=params["a_pn"],
        b_n=params["b_n"],
        b_pn=params["b_pn"],
        ratio_a_pn_over_a_n=params["a_pn"] / params["a_n"],
        ratio_b_pn_over_b_n=params["b_pn"] / params["b_n"],
        J=j_value,
        L=l_value,
        K0=ground_state.K0,
        binding=ground_state.binding,
        n_sat=ground_state.n_sat,
        p_sat=ground_state.p_sat,
        j_residual=j_value - fit_settings.target_j,
        l_residual=l_value - fit_settings.target_l,
        k0_residual=0.0 if fit_settings.target_k0 is None else ground_state.K0 - fit_settings.target_k0,
    )
