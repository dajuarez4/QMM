"""Finite-temperature quantum critical-point solver."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import DEFAULT_PHYSICAL_CONSTANTS, DEFAULT_QUANTUM_SETTINGS, PhysicalConstants, QuantumCriticalSettings
from .ground_state import GroundStateResult, compute_ground_state_point
from .models import get_model
from .numerics import linspace, simpson_weights


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class QuantumCriticalPoint:
    """Quantum critical point from the SCF solver."""

    model: str
    parameter_name: str | None
    parameter_value: float | None
    a: float
    b: float
    K0: float
    Tc: float
    nc: float
    Pc: float
    dPdn: float
    d2Pdn2: float
    score: float
    iterations: int
    mu_star: float


class QuantumFermiGas:
    """Finite-temperature ideal-gas integrals."""

    def __init__(
        self,
        physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
        settings: QuantumCriticalSettings = DEFAULT_QUANTUM_SETTINGS,
    ) -> None:
        self.physical = physical
        self.settings = settings
        self.k_grid, self.weights = simpson_weights(0.0, settings.k_max_fd, settings.n_k_fd)
        self.e_grid = [
            math.sqrt((physical.hbarc * k_value) ** 2 + physical.m_nucleon ** 2)
            for k_value in self.k_grid
        ]
        self.k2_grid = [k_value * k_value for k_value in self.k_grid]
        self.pk_grid = [
            (physical.hbarc ** 2) * k_value ** 4 / energy
            for k_value, energy in zip(self.k_grid, self.e_grid)
        ]
        self.pref_n = physical.degeneracy_symmetric / (2.0 * math.pi ** 2)
        self.pref_p = physical.degeneracy_symmetric / (6.0 * math.pi ** 2)
        self._cache: dict[tuple[float, float], tuple[float, float]] = {}

    def _cache_key(self, temperature: float, mu: float) -> tuple[float, float]:
        digits = self.settings.cache_round_digits
        return (round(float(temperature), digits), round(float(mu), digits))

    @staticmethod
    def fermi_dirac(energy: float, mu: float, temperature: float) -> float:
        """Return the occupation factor with overflow protection."""
        arg = (energy - mu) / temperature
        if arg > 700.0:
            return 0.0
        if arg < -700.0:
            return 1.0
        return 1.0 / (math.exp(arg) + 1.0)

    def evaluate_number_and_pressure(self, temperature: float, mu: float) -> tuple[float, float]:
        """Return `(n_id, p_id)` at finite temperature."""
        key = self._cache_key(temperature, mu)
        if key in self._cache:
            return self._cache[key]

        sum_n = 0.0
        sum_p = 0.0
        for weight, energy, k2_value, pk_value in zip(self.weights, self.e_grid, self.k2_grid, self.pk_grid):
            occupation = self.fermi_dirac(energy, mu, temperature)
            sum_n += weight * k2_value * occupation
            sum_p += weight * pk_value * occupation

        result = (self.pref_n * sum_n, self.pref_p * sum_p)
        self._cache[key] = result
        return result

    def number_density(self, temperature: float, mu: float) -> float:
        return self.evaluate_number_and_pressure(temperature, mu)[0]

    def pressure(self, temperature: float, mu: float) -> float:
        return self.evaluate_number_and_pressure(temperature, mu)[1]

    def mu_seed_from_n_id(self, n_id_target: float) -> float:
        """Return a zero-temperature estimate for `mu*`."""
        if n_id_target <= 0.0:
            return self.physical.m_nucleon
        kf = (6.0 * math.pi ** 2 * n_id_target / self.physical.degeneracy_symmetric) ** (1.0 / 3.0)
        return math.sqrt((self.physical.hbarc * kf) ** 2 + self.physical.m_nucleon ** 2)


class QuantumPressureEvaluator:
    """Evaluate the quantum pressure `P(T, n)`."""

    def __init__(
        self,
        ground_state: GroundStateResult,
        physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
        settings: QuantumCriticalSettings = DEFAULT_QUANTUM_SETTINGS,
    ) -> None:
        self.ground_state = ground_state
        self.physical = physical
        self.settings = settings
        self.model = get_model(ground_state.model)
        self.fermi = QuantumFermiGas(physical, settings)
        self._pressure_cache: dict[tuple[float, float], tuple[float | None, float | None]] = {}

    def _cache_key(self, temperature: float, density: float) -> tuple[float, float]:
        digits = self.settings.cache_round_digits
        return (round(float(temperature), digits), round(float(density), digits))

    def target_n_id(self, density: float) -> float | None:
        return self.model.nid_from_n(density, self.ground_state.b)

    def solve_mu_star_scf(self, temperature: float, density: float, mu0: float | None = None) -> float | None:
        """Invert `n_id(T, mu*) = n_id_target` with damped Newton/SCF."""
        target_n_id = self.target_n_id(density)
        if target_n_id is None or target_n_id <= 0.0 or temperature <= 0.0:
            return None

        mu = self.fermi.mu_seed_from_n_id(target_n_id) if mu0 is None else mu0
        mu = _clamp(mu, self.settings.mu_min, self.settings.mu_max)

        for _ in range(self.settings.mu_scf_max_iter):
            n_id_value = self.fermi.number_density(temperature, mu)
            residual = n_id_value - target_n_id
            if abs(residual) <= self.settings.mu_scf_tol * max(1.0, target_n_id):
                return mu

            mu_plus = min(self.settings.mu_max, mu + self.settings.mu_delta)
            mu_minus = max(self.settings.mu_min, mu - self.settings.mu_delta)
            if mu_plus <= mu_minus:
                return None

            n_plus = self.fermi.number_density(temperature, mu_plus)
            n_minus = self.fermi.number_density(temperature, mu_minus)
            susceptibility = (n_plus - n_minus) / (mu_plus - mu_minus)
            if not math.isfinite(susceptibility) or abs(susceptibility) < 1.0e-12:
                return None

            update = residual / susceptibility
            mu_new = _clamp(mu - self.settings.mu_scf_damping * update, self.settings.mu_min, self.settings.mu_max)
            if abs(mu_new - mu) <= self.settings.mu_scf_tol * max(1.0, abs(mu)):
                mu = mu_new
                if abs(residual) <= 10.0 * self.settings.mu_scf_tol * max(1.0, target_n_id):
                    return mu
            mu = mu_new
        return None

    def pressure(self, temperature: float, density: float) -> tuple[float | None, float | None]:
        """Return `(P, mu*)` for one thermodynamic point."""
        key = self._cache_key(temperature, density)
        if key in self._pressure_cache:
            return self._pressure_cache[key]

        mu_star = self.solve_mu_star_scf(temperature, density)
        if mu_star is None:
            self._pressure_cache[key] = (None, None)
            return None, None

        _, p_id = self.fermi.evaluate_number_and_pressure(temperature, mu_star)
        prefactor = self.model.pressure_prefactor(density, self.ground_state.b)
        if prefactor is None:
            self._pressure_cache[key] = (None, None)
            return None, None

        dU_value = self.model.dU(density, self.ground_state.b, self.ground_state.parameter_value)
        if dU_value is None:
            self._pressure_cache[key] = (None, None)
            return None, None

        pressure_value = prefactor * p_id + density * density * self.ground_state.a * dU_value
        result = (pressure_value, mu_star)
        self._pressure_cache[key] = result
        return result

    def critical_equations(self, temperature: float, density: float) -> tuple[float | None, float | None, float | None]:
        """Return `(dP/dn, d2P/dn2, mu*)`."""
        h_n = self.settings.h_n
        density_minus = density - h_n
        density_plus = density + h_n
        if density_minus <= self.settings.n_min_cp or density_plus >= self.settings.n_max_cp:
            return None, None, None

        p_minus, _ = self.pressure(temperature, density_minus)
        p_zero, mu_star = self.pressure(temperature, density)
        p_plus, _ = self.pressure(temperature, density_plus)
        if p_minus is None or p_zero is None or p_plus is None:
            return None, None, None

        dPdn = (p_plus - p_minus) / (2.0 * h_n)
        d2Pdn2 = (p_plus - 2.0 * p_zero + p_minus) / (h_n * h_n)
        return dPdn, d2Pdn2, mu_star


def find_seed_point(
    evaluator: QuantumPressureEvaluator,
    settings: QuantumCriticalSettings = DEFAULT_QUANTUM_SETTINGS,
    preferred_seed: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], float, float] | None:
    """Return a coarse critical seed and normalization scales."""
    t_grid = linspace(settings.t_min_cp, settings.t_max_cp, settings.coarse_t_count)
    n_grid = linspace(settings.n_min_cp, settings.n_max_cp, settings.coarse_n_count)
    candidates: list[tuple[float, float, float, float]] = []

    if preferred_seed is not None:
        T_seed, n_seed = preferred_seed
        values = evaluator.critical_equations(T_seed, n_seed)
        if values[0] is not None and values[1] is not None:
            candidates.append((T_seed, n_seed, float(values[0]), float(values[1])))

    for temperature in t_grid:
        for density in n_grid:
            dPdn, d2Pdn2, _ = evaluator.critical_equations(temperature, density)
            if dPdn is None or d2Pdn2 is None:
                continue
            if not math.isfinite(dPdn) or not math.isfinite(d2Pdn2):
                continue
            candidates.append((temperature, density, dPdn, d2Pdn2))

    if not candidates:
        return None

    scale_1 = max(max(abs(item[2]) for item in candidates), 1.0e-12)
    scale_2 = max(max(abs(item[3]) for item in candidates), 1.0e-12)
    best = min(candidates, key=lambda item: (item[2] / scale_1) ** 2 + (item[3] / scale_2) ** 2)
    return (best[0], best[1]), scale_1, scale_2


def solve_quantum_critical_point(
    evaluator: QuantumPressureEvaluator,
    settings: QuantumCriticalSettings = DEFAULT_QUANTUM_SETTINGS,
    preferred_seed: tuple[float, float] | None = None,
) -> QuantumCriticalPoint | None:
    """Solve the critical-point equations with damped outer updates."""
    seed_info = find_seed_point(evaluator, settings, preferred_seed)
    if seed_info is None:
        return None
    (temperature, density), scale_1, scale_2 = seed_info
    score = float("inf")

    for iteration in range(1, settings.outer_max_iter + 1):
        dPdn, d2Pdn2, mu_star = evaluator.critical_equations(temperature, density)
        if dPdn is None or d2Pdn2 is None or mu_star is None:
            return None

        score = (dPdn / scale_1) ** 2 + (d2Pdn2 / scale_2) ** 2
        if abs(dPdn) <= settings.outer_tol_dPdn and abs(d2Pdn2) <= settings.outer_tol_d2Pdn2:
            pressure_value, mu_final = evaluator.pressure(temperature, density)
            if pressure_value is None or mu_final is None:
                return None
            gs = evaluator.ground_state
            return QuantumCriticalPoint(
                model=gs.model,
                parameter_name=gs.parameter_name,
                parameter_value=gs.parameter_value,
                a=gs.a,
                b=gs.b,
                K0=gs.K0,
                Tc=temperature,
                nc=density,
                Pc=pressure_value,
                dPdn=dPdn,
                d2Pdn2=d2Pdn2,
                score=score,
                iterations=iteration,
                mu_star=mu_final,
            )

        t_plus = min(settings.t_max_cp, temperature + settings.h_t)
        t_minus = max(settings.t_min_cp, temperature - settings.h_t)
        if t_plus <= t_minus:
            return None

        dPdn_plus, _, _ = evaluator.critical_equations(t_plus, density)
        dPdn_minus, _, _ = evaluator.critical_equations(t_minus, density)
        if dPdn_plus is None or dPdn_minus is None:
            return None
        slope_t = (dPdn_plus - dPdn_minus) / (t_plus - t_minus)

        n_plus = min(settings.n_max_cp, density + settings.h_n)
        n_minus = max(settings.n_min_cp, density - settings.h_n)
        if n_plus <= n_minus:
            return None

        _, d2Pdn2_plus, _ = evaluator.critical_equations(temperature, n_plus)
        _, d2Pdn2_minus, _ = evaluator.critical_equations(temperature, n_minus)
        if d2Pdn2_plus is None or d2Pdn2_minus is None:
            return None
        slope_n = (d2Pdn2_plus - d2Pdn2_minus) / (n_plus - n_minus)

        if abs(slope_t) < 1.0e-12 or abs(slope_n) < 1.0e-12:
            return None

        delta_t = _clamp(-settings.outer_damping_t * dPdn / slope_t, -settings.outer_max_step_t, settings.outer_max_step_t)
        delta_n = _clamp(-settings.outer_damping_n * d2Pdn2 / slope_n, -settings.outer_max_step_n, settings.outer_max_step_n)

        accepted = False
        lam = 1.0
        while lam >= settings.outer_min_lambda:
            t_try = _clamp(temperature + lam * delta_t, settings.t_min_cp, settings.t_max_cp)
            n_try = _clamp(density + lam * delta_n, settings.n_min_cp, settings.n_max_cp)
            dPdn_try, d2Pdn2_try, _ = evaluator.critical_equations(t_try, n_try)
            if dPdn_try is None or d2Pdn2_try is None:
                lam *= 0.5
                continue
            score_try = (dPdn_try / scale_1) ** 2 + (d2Pdn2_try / scale_2) ** 2
            if score_try < score:
                temperature = t_try
                density = n_try
                accepted = True
                break
            lam *= 0.5

        if not accepted:
            return None

    pressure_value, mu_final = evaluator.pressure(temperature, density)
    dPdn, d2Pdn2, _ = evaluator.critical_equations(temperature, density)
    if pressure_value is None or mu_final is None or dPdn is None or d2Pdn2 is None:
        return None

    gs = evaluator.ground_state
    score = (dPdn / scale_1) ** 2 + (d2Pdn2 / scale_2) ** 2
    return QuantumCriticalPoint(
        model=gs.model,
        parameter_name=gs.parameter_name,
        parameter_value=gs.parameter_value,
        a=gs.a,
        b=gs.b,
        K0=gs.K0,
        Tc=temperature,
        nc=density,
        Pc=pressure_value,
        dPdn=dPdn,
        d2Pdn2=d2Pdn2,
        score=score,
        iterations=settings.outer_max_iter,
        mu_star=mu_final,
    )


def compute_quantum_critical_point(
    model_name: str,
    parameter_value: float | None = None,
    parameter_search=None,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: QuantumCriticalSettings = DEFAULT_QUANTUM_SETTINGS,
    preferred_seed: tuple[float, float] | None = None,
) -> QuantumCriticalPoint:
    """Compute the quantum critical point for one model."""
    model = get_model(model_name)
    if not model.supports_quantum:
        raise ValueError(f"Model '{model_name}' does not support the quantum workflow.")
    ground_state = compute_ground_state_point(model_name, parameter_value, parameter_search, physical)
    evaluator = QuantumPressureEvaluator(ground_state, physical, settings)
    result = solve_quantum_critical_point(evaluator, settings, preferred_seed)
    if result is None:
        raise ValueError(f"Quantum SCF solve failed for model '{model_name}'.")
    return result
