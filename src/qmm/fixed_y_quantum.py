"""Finite-temperature fixed-composition critical points for asymmetric models."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable

from .asymmetry import AsymmetricFitResult, asymmetric_species_data, attractive_coefficient
from .constants import DEFAULT_PHYSICAL_CONSTANTS, DEFAULT_QUANTUM_SETTINGS, PhysicalConstants, QuantumCriticalSettings
from .models import get_model
from .numerics import linspace
from .quantum import QuantumFermiGas


FIXED_Y_MODELS = frozenset({"vdw", "clausius", "dieterici", "cs", "tvm"})


@dataclass(frozen=True)
class FixedYQuantumCriticalPoint:
    """Critical point for one fixed proton fraction `y`."""

    model: str
    parameter_name: str | None
    parameter_value: float | None
    target_k0: float | None
    y: float
    Tc: float
    nc: float
    Pc: float
    dPdn: float
    d2Pdn2: float
    score: float
    iterations: int
    mu_p_star: float
    mu_n_star: float
    a_avg: float
    b_avg: float
    a_n: float
    a_pn: float
    b_n: float
    b_pn: float


@dataclass(frozen=True)
class FixedYQuantumBranchStatus:
    """Solver status for one `(target_k0, y)` branch."""

    target_k0: float | None
    y: float
    status: str


def fixed_y_quantum_settings() -> QuantumCriticalSettings:
    """Return the broader search window used by the legacy fixed-`y` notebook."""
    return replace(
        QuantumCriticalSettings.quick(),
        t_min_cp=4.0,
        t_max_cp=22.0,
        n_min_cp=0.020,
        n_max_cp=0.075,
        coarse_t_count=15,
        coarse_n_count=15,
        outer_max_iter=30,
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _validate_model(model_name: str) -> None:
    if model_name not in FIXED_Y_MODELS:
        allowed = ", ".join(sorted(FIXED_Y_MODELS))
        raise ValueError(
            "The fixed-y finite-temperature solver is currently implemented for "
            f"the asymmetric hadronic models {allowed}. Got '{model_name}'."
        )


class FixedYEVPressureEvaluator:
    """Evaluate `P(T, n; y)` for asymmetric matter at fixed composition."""

    def __init__(
        self,
        fit: AsymmetricFitResult,
        y: float,
        physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
        settings: QuantumCriticalSettings = DEFAULT_QUANTUM_SETTINGS,
    ) -> None:
        _validate_model(fit.model)
        self.fit = fit
        self.y = float(y)
        self.physical = physical
        self.settings = settings
        self.model = get_model(fit.model)

        species_physical = replace(physical, degeneracy_symmetric=physical.degeneracy_species)
        self.proton_fermi = QuantumFermiGas(species_physical, settings)
        self.neutron_fermi = QuantumFermiGas(species_physical, settings)
        self._pressure_cache: dict[tuple[float, float], tuple[float | None, float | None, float | None]] = {}

    def _cache_key(self, temperature: float, density: float) -> tuple[float, float]:
        digits = self.settings.cache_round_digits
        return (round(float(temperature), digits), round(float(density), digits))

    def target_n_id(self, density: float) -> tuple[float, float] | None:
        data = asymmetric_species_data(
            self.fit.model,
            density,
            self.y,
            self.fit.b_n,
            self.fit.b_pn,
        )
        if data is None:
            return None
        return data[4], data[5]

    def _species_b_eff(self) -> tuple[float, float]:
        """Return the fixed-`y` effective repulsion coefficients for `p` and `n`."""
        proton_b = self.y * self.fit.b_n + (1.0 - self.y) * self.fit.b_pn
        neutron_b = self.y * self.fit.b_pn + (1.0 - self.y) * self.fit.b_n
        return proton_b, neutron_b

    def _solve_mu_star_species(
        self,
        fermi: QuantumFermiGas,
        temperature: float,
        n_id_target: float,
        mu0: float | None = None,
    ) -> float | None:
        if temperature <= 0.0 or n_id_target < 0.0:
            return None
        if n_id_target == 0.0:
            return self.physical.m_nucleon

        mu = fermi.mu_seed_from_n_id(n_id_target) if mu0 is None else mu0
        mu = _clamp(mu, self.settings.mu_min, self.settings.mu_max)

        for _ in range(self.settings.mu_scf_max_iter):
            n_id_value = fermi.number_density(temperature, mu)
            residual = n_id_value - n_id_target
            if abs(residual) <= self.settings.mu_scf_tol * max(1.0, n_id_target):
                return mu

            mu_plus = min(self.settings.mu_max, mu + self.settings.mu_delta)
            mu_minus = max(self.settings.mu_min, mu - self.settings.mu_delta)
            if mu_plus <= mu_minus:
                return None

            n_plus = fermi.number_density(temperature, mu_plus)
            n_minus = fermi.number_density(temperature, mu_minus)
            susceptibility = (n_plus - n_minus) / (mu_plus - mu_minus)
            if not math.isfinite(susceptibility) or abs(susceptibility) < 1.0e-12:
                return None

            update = residual / susceptibility
            mu_new = _clamp(
                mu - self.settings.mu_scf_damping * update,
                self.settings.mu_min,
                self.settings.mu_max,
            )
            if abs(mu_new - mu) <= self.settings.mu_scf_tol * max(1.0, abs(mu)):
                mu = mu_new
                if abs(residual) <= 10.0 * self.settings.mu_scf_tol * max(1.0, n_id_target):
                    return mu
            mu = mu_new
        return None

    def pressure(self, temperature: float, density: float) -> tuple[float | None, float | None, float | None]:
        """Return `(P, mu_p*, mu_n*)` for one fixed-`y` branch."""
        key = self._cache_key(temperature, density)
        if key in self._pressure_cache:
            return self._pressure_cache[key]

        targets = self.target_n_id(density)
        if targets is None:
            self._pressure_cache[key] = (None, None, None)
            return None, None, None
        n_p_id_target, n_n_id_target = targets

        mu_p = self._solve_mu_star_species(self.proton_fermi, temperature, n_p_id_target)
        mu_n = self._solve_mu_star_species(self.neutron_fermi, temperature, n_n_id_target)
        if mu_p is None or mu_n is None:
            self._pressure_cache[key] = (None, None, None)
            return None, None, None

        p_p = self.proton_fermi.pressure(temperature, mu_p)
        p_n = self.neutron_fermi.pressure(temperature, mu_n)
        b_eff_p, b_eff_n = self._species_b_eff()
        prefactor_p = self.model.pressure_prefactor(density, b_eff_p)
        prefactor_n = self.model.pressure_prefactor(density, b_eff_n)
        if prefactor_p is None or prefactor_n is None:
            self._pressure_cache[key] = (None, None, None)
            return None, None, None

        dU_value = self.model.dU(density, 0.0, self.fit.parameter_value)
        if dU_value is None:
            self._pressure_cache[key] = (None, None, None)
            return None, None, None

        coeff = attractive_coefficient(self.y, self.fit.a_n, self.fit.a_pn)
        result = (
            prefactor_p * p_p + prefactor_n * p_n + density * density * coeff * dU_value,
            mu_p,
            mu_n,
        )
        self._pressure_cache[key] = result
        return result

    def critical_equations(
        self,
        temperature: float,
        density: float,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """Return `(dP/dn, d2P/dn2, mu_p*, mu_n*)`."""
        h_n = self.settings.h_n
        density_minus = density - h_n
        density_plus = density + h_n
        if density_minus <= self.settings.n_min_cp or density_plus >= self.settings.n_max_cp:
            return None, None, None, None

        p_minus, _, _ = self.pressure(temperature, density_minus)
        p_zero, mu_p, mu_n = self.pressure(temperature, density)
        p_plus, _, _ = self.pressure(temperature, density_plus)
        if p_minus is None or p_zero is None or p_plus is None or mu_p is None or mu_n is None:
            return None, None, None, None

        dPdn = (p_plus - p_minus) / (2.0 * h_n)
        d2Pdn2 = (p_plus - 2.0 * p_zero + p_minus) / (h_n * h_n)
        return dPdn, d2Pdn2, mu_p, mu_n


def _build_seed_grid(settings: QuantumCriticalSettings) -> tuple[list[float], list[float]]:
    return (
        linspace(settings.t_min_cp, settings.t_max_cp, settings.coarse_t_count),
        linspace(settings.n_min_cp, settings.n_max_cp, settings.coarse_n_count),
    )


def find_seed_point_fixed_y(
    evaluator: FixedYEVPressureEvaluator,
    settings: QuantumCriticalSettings = DEFAULT_QUANTUM_SETTINGS,
    preferred_seed: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], float, float] | None:
    """Return a coarse critical seed and normalization scales."""
    t_grid, n_grid = _build_seed_grid(settings)
    candidates: list[tuple[float, float, float, float]] = []

    if preferred_seed is not None:
        temperature_seed, density_seed = preferred_seed
        values = evaluator.critical_equations(temperature_seed, density_seed)
        if values[0] is not None and values[1] is not None:
            candidates.append((temperature_seed, density_seed, float(values[0]), float(values[1])))

    for temperature in t_grid:
        for density in n_grid:
            dPdn, d2Pdn2, _, _ = evaluator.critical_equations(temperature, density)
            if dPdn is None or d2Pdn2 is None:
                continue
            if not math.isfinite(dPdn) or not math.isfinite(d2Pdn2):
                continue
            candidates.append((temperature, density, dPdn, d2Pdn2))

    if not candidates:
        return None

    scale_1 = max(abs(item[2]) for item in candidates)
    scale_2 = max(abs(item[3]) for item in candidates)
    scale_1 = max(scale_1, 1.0e-12)
    scale_2 = max(scale_2, 1.0e-12)

    best: tuple[float, float, float] | None = None
    for temperature, density, dPdn, d2Pdn2 in candidates:
        score = (dPdn / scale_1) ** 2 + (d2Pdn2 / scale_2) ** 2
        if best is None or score < best[0]:
            best = (score, temperature, density)

    if best is None:
        return None
    return (best[1], best[2]), scale_1, scale_2


def solve_fixed_y_critical_point(
    evaluator: FixedYEVPressureEvaluator,
    settings: QuantumCriticalSettings = DEFAULT_QUANTUM_SETTINGS,
    preferred_seed: tuple[float, float] | None = None,
) -> tuple[float, float, float, float, float, float, int, float, float] | None:
    """Solve the fixed-`y` critical-point equations with damped SCF updates."""
    seed_info = find_seed_point_fixed_y(evaluator, settings=settings, preferred_seed=preferred_seed)
    if seed_info is None:
        return None

    (temperature, density), scale_1, scale_2 = seed_info
    score = float("inf")

    for iteration in range(1, settings.outer_max_iter + 1):
        dPdn, d2Pdn2, mu_p, mu_n = evaluator.critical_equations(temperature, density)
        if dPdn is None or d2Pdn2 is None or mu_p is None or mu_n is None:
            return None

        score = (dPdn / scale_1) ** 2 + (d2Pdn2 / scale_2) ** 2
        if abs(dPdn) <= settings.outer_tol_dPdn and abs(d2Pdn2) <= settings.outer_tol_d2Pdn2:
            pressure_value, mu_p_final, mu_n_final = evaluator.pressure(temperature, density)
            if pressure_value is None or mu_p_final is None or mu_n_final is None:
                return None
            return (
                temperature,
                density,
                pressure_value,
                dPdn,
                d2Pdn2,
                score,
                iteration,
                mu_p_final,
                mu_n_final,
            )

        temperature_plus = min(settings.t_max_cp, temperature + settings.h_t)
        temperature_minus = max(settings.t_min_cp, temperature - settings.h_t)
        if temperature_plus <= temperature_minus:
            return None

        dPdn_plus, _, _, _ = evaluator.critical_equations(temperature_plus, density)
        dPdn_minus, _, _, _ = evaluator.critical_equations(temperature_minus, density)
        if dPdn_plus is None or dPdn_minus is None:
            return None
        slope_temperature = (dPdn_plus - dPdn_minus) / (temperature_plus - temperature_minus)

        density_plus = min(settings.n_max_cp, density + settings.h_n)
        density_minus = max(settings.n_min_cp, density - settings.h_n)
        if density_plus <= density_minus:
            return None

        _, d2Pdn2_plus, _, _ = evaluator.critical_equations(temperature, density_plus)
        _, d2Pdn2_minus, _, _ = evaluator.critical_equations(temperature, density_minus)
        if d2Pdn2_plus is None or d2Pdn2_minus is None:
            return None
        slope_density = (d2Pdn2_plus - d2Pdn2_minus) / (density_plus - density_minus)

        if abs(slope_temperature) < 1.0e-12 or abs(slope_density) < 1.0e-12:
            return None

        delta_temperature = _clamp(
            -settings.outer_damping_t * dPdn / slope_temperature,
            -settings.outer_max_step_t,
            settings.outer_max_step_t,
        )
        delta_density = _clamp(
            -settings.outer_damping_n * d2Pdn2 / slope_density,
            -settings.outer_max_step_n,
            settings.outer_max_step_n,
        )

        accepted = False
        lambda_value = 1.0
        while lambda_value >= settings.outer_min_lambda:
            temperature_try = _clamp(
                temperature + lambda_value * delta_temperature,
                settings.t_min_cp,
                settings.t_max_cp,
            )
            density_try = _clamp(
                density + lambda_value * delta_density,
                settings.n_min_cp,
                settings.n_max_cp,
            )

            dPdn_try, d2Pdn2_try, _, _ = evaluator.critical_equations(temperature_try, density_try)
            if dPdn_try is None or d2Pdn2_try is None:
                lambda_value *= 0.5
                continue

            score_try = (dPdn_try / scale_1) ** 2 + (d2Pdn2_try / scale_2) ** 2
            if score_try < score:
                temperature = temperature_try
                density = density_try
                accepted = True
                break
            lambda_value *= 0.5

        if not accepted:
            return None

    pressure_value, mu_p_final, mu_n_final = evaluator.pressure(temperature, density)
    if pressure_value is None or mu_p_final is None or mu_n_final is None:
        return None

    dPdn, d2Pdn2, _, _ = evaluator.critical_equations(temperature, density)
    if dPdn is None or d2Pdn2 is None:
        return None

    score = (dPdn / scale_1) ** 2 + (d2Pdn2 / scale_2) ** 2
    return (
        temperature,
        density,
        pressure_value,
        dPdn,
        d2Pdn2,
        score,
        settings.outer_max_iter,
        mu_p_final,
        mu_n_final,
    )


def compute_fixed_y_critical_point(
    fit: AsymmetricFitResult,
    y: float,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: QuantumCriticalSettings = DEFAULT_QUANTUM_SETTINGS,
    preferred_seed: tuple[float, float] | None = None,
) -> FixedYQuantumCriticalPoint | None:
    """Compute one fixed-`y` finite-temperature critical point from one asymmetric fit."""
    evaluator = FixedYEVPressureEvaluator(fit, float(y), physical, settings)
    solved = solve_fixed_y_critical_point(evaluator, settings=settings, preferred_seed=preferred_seed)
    if solved is None:
        return None

    Tc, nc, Pc, dPdn, d2Pdn2, score, iterations, mu_p_star, mu_n_star = solved
    return FixedYQuantumCriticalPoint(
        model=fit.model,
        parameter_name=fit.parameter_name,
        parameter_value=fit.parameter_value,
        target_k0=fit.target_k0,
        y=float(y),
        Tc=Tc,
        nc=nc,
        Pc=Pc,
        dPdn=dPdn,
        d2Pdn2=d2Pdn2,
        score=score,
        iterations=iterations,
        mu_p_star=mu_p_star,
        mu_n_star=mu_n_star,
        a_avg=fit.a_avg,
        b_avg=fit.b_avg,
        a_n=fit.a_n,
        a_pn=fit.a_pn,
        b_n=fit.b_n,
        b_pn=fit.b_pn,
    )


def compute_fixed_y_family(
    fits: Iterable[AsymmetricFitResult],
    y_values: Iterable[float],
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: QuantumCriticalSettings = DEFAULT_QUANTUM_SETTINGS,
) -> tuple[list[FixedYQuantumCriticalPoint], list[FixedYQuantumBranchStatus]]:
    """Compute a fixed-`y` family using continuation seeds along each fit branch."""
    results: list[FixedYQuantumCriticalPoint] = []
    statuses: list[FixedYQuantumBranchStatus] = []

    y_list = [float(y_value) for y_value in y_values]
    for fit in fits:
        previous_seed: tuple[float, float] | None = None
        for y_value in y_list:
            result = compute_fixed_y_critical_point(
                fit,
                y_value,
                physical=physical,
                settings=settings,
                preferred_seed=previous_seed,
            )
            if result is None:
                statuses.append(
                    FixedYQuantumBranchStatus(
                        target_k0=fit.target_k0,
                        y=y_value,
                        status="no_solution_in_window",
                    )
                )
                previous_seed = None
                continue

            results.append(result)
            statuses.append(
                FixedYQuantumBranchStatus(
                    target_k0=fit.target_k0,
                    y=y_value,
                    status="converged",
                )
            )
            previous_seed = (result.Tc, result.nc)

    return results, statuses
