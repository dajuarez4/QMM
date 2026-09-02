"""Zero-temperature quarkyonic and baryquark workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .asymmetry import AsymmetricFitResult, attractive_coefficient
from .constants import DEFAULT_GROUND_STATE_SETTINGS, DEFAULT_PHYSICAL_CONSTANTS, DEFAULT_QUARKYONIC_SETTINGS, GroundStateSettings, PhysicalConstants, QuarkyonicSettings
from .fermi_gas import eps_id_from_kf, eps_id_from_nid
from .ground_state import GroundStateResult, compute_ground_state_point
from .models import get_model
from .numerics import (
    golden_section_min,
    golden_section_min_safe,
    linspace,
    simpson_integral,
)
from .sound_speed import reconstruct_sound_speed_curve


@dataclass(frozen=True)
class QuarkyonicState:
    """Minimum-energy state at one baryon density."""

    n_b: float
    n_over_n0: float
    quark_fraction: float
    n_q: float
    n_n: float
    k_bu: float
    k_f: float
    energy_density: float


@dataclass(frozen=True)
class SymmetricQuarkyonicCurve:
    """One symmetric quarkyonic sound-speed curve."""

    model: str
    momentum_mode: str
    parameter_name: str | None
    parameter_value: float | None
    a: float
    b: float
    K0: float
    n: list[float]
    n_over_n0: list[float]
    eps_raw: list[float]
    eps: list[float]
    mu_b: list[float]
    P: list[float]
    dP_dn: list[float]
    d2eps_dn2: list[float]
    vs2: list[float]
    quark_fraction: list[float]
    n_q: list[float]
    n_n: list[float]
    k_bu: list[float]
    k_f: list[float]
    vs2_max: float
    n_tr: float
    n_tr_over_n0: float


@dataclass(frozen=True)
class AsymmetricQuarkyonicRow:
    """One row of an asymmetric quarkyonic profile."""

    n_b: float
    n_over_n0: float
    y: float
    quark_fraction: float
    k_bu: float
    k_fp: float
    k_fn: float
    n_p: float
    n_n: float
    energy_density: float
    mu_e: float | None
    eps_e: float
    eps_mu: float
    mu_b: float | None = None
    pressure: float | None = None
    vs2: float | None = None


@dataclass(frozen=True)
class AsymmetricQuarkyonicResult:
    """All asymmetric quarkyonic outputs for one model."""

    model: str
    momentum_mode: str
    parameter_name: str | None
    parameter_value: float | None
    fixed_y_profiles: dict[str, list[AsymmetricQuarkyonicRow]]
    beta_profile: list[AsymmetricQuarkyonicRow]


def _nucleon_energy_shell(
    k_bu: float,
    k_f: float,
    shell_points: int,
    physical: PhysicalConstants,
) -> float:
    del shell_points
    if k_f <= k_bu:
        return 0.0
    return eps_id_from_kf(k_f, physical.degeneracy_symmetric, physical) - eps_id_from_kf(
        k_bu,
        physical.degeneracy_symmetric,
        physical,
    )


def _validate_momentum_mode(settings: QuarkyonicSettings) -> str:
    mode = settings.momentum_mode.strip().lower()
    if mode not in {"quarkyonic", "baryquark"}:
        raise ValueError(
            f"Unsupported quarkyonic.momentum_mode '{settings.momentum_mode}'. "
            "Use 'quarkyonic' or 'baryquark'."
        )
    return mode


def _quark_density_from_kbu(k_bu: float, settings: QuarkyonicSettings, physical: PhysicalConstants) -> float:
    if k_bu <= 0.0:
        return 0.0

    upper = k_bu / float(settings.nc)
    kappa = settings.lambda_momentum_mev / physical.hbarc

    def integrand(q_value: float) -> float:
        return q_value * math.sqrt(kappa * kappa + q_value * q_value)

    integral = simpson_integral(integrand, 0.0, upper, settings.quark_integral_points)
    return settings.quark_degeneracy * integral / (2.0 * math.pi ** 2)


def _quark_energy_density_from_kbu(k_bu: float, settings: QuarkyonicSettings, physical: PhysicalConstants) -> float:
    if k_bu <= 0.0:
        return 0.0

    upper = k_bu / float(settings.nc)
    kappa = settings.lambda_momentum_mev / physical.hbarc
    m_q = physical.m_nucleon / float(settings.nc)

    def integrand(q_value: float) -> float:
        density_factor = q_value * math.sqrt(kappa * kappa + q_value * q_value)
        energy = math.sqrt((physical.hbarc * q_value) ** 2 + m_q * m_q)
        return density_factor * energy

    integral = simpson_integral(integrand, 0.0, upper, settings.quark_integral_points)
    prefactor = float(settings.nc) * settings.quark_degeneracy / (2.0 * math.pi ** 2)
    return prefactor * integral


def _quark_density_between_momenta(
    k_inner: float,
    k_outer: float,
    settings: QuarkyonicSettings,
    physical: PhysicalConstants,
) -> float:
    if k_outer <= k_inner:
        return 0.0
    return _quark_density_from_kbu(k_outer, settings, physical) - _quark_density_from_kbu(k_inner, settings, physical)


def _quark_energy_density_between_momenta(
    k_inner: float,
    k_outer: float,
    settings: QuarkyonicSettings,
    physical: PhysicalConstants,
) -> float:
    if k_outer <= k_inner:
        return 0.0
    return _quark_energy_density_from_kbu(k_outer, settings, physical) - _quark_energy_density_from_kbu(k_inner, settings, physical)


def _kbu_from_quark_density(n_q: float, settings: QuarkyonicSettings, physical: PhysicalConstants) -> float:
    if n_q <= 0.0:
        return 0.0
    kappa = settings.lambda_momentum_mev / physical.hbarc
    term = kappa ** 3 + (6.0 * math.pi ** 2 * n_q) / settings.quark_degeneracy
    value = term ** (2.0 / 3.0) - kappa * kappa
    return float(settings.nc) * math.sqrt(max(value, 0.0))


def _max_hadronic_density(model_name: str, b_value: float) -> float:
    if b_value <= 0.0:
        return math.inf
    normalized = model_name.strip().lower()
    if normalized == "cs" or normalized.endswith("_cs"):
        return 4.0 / b_value
    if normalized == "tvm" or normalized.endswith("_tvm"):
        return math.inf
    return 1.0 / b_value


def _lower_quark_fraction_bound(
    model_name: str,
    n_b: float,
    b_value: float,
    settings: QuarkyonicSettings,
) -> float:
    if n_b <= 0.0:
        return 0.0
    max_n_h = _max_hadronic_density(model_name, b_value)
    if not math.isfinite(max_n_h) or n_b <= max_n_h:
        return 0.0
    return max(0.0, 1.0 - max_n_h / n_b + settings.fq_min_shift)


def _state_at_fraction(
    ground_state: GroundStateResult,
    n_b: float,
    fq: float,
    settings: QuarkyonicSettings,
    physical: PhysicalConstants,
    gs_settings: GroundStateSettings,
) -> QuarkyonicState | None:
    mode = _validate_momentum_mode(settings)
    if mode == "baryquark":
        return _baryquark_state_at_fraction(ground_state, n_b, fq, settings, physical)
    return _quarkyonic_state_at_fraction(ground_state, n_b, fq, settings, physical)


def _quarkyonic_state_at_fraction(
    ground_state: GroundStateResult,
    n_b: float,
    fq: float,
    settings: QuarkyonicSettings,
    physical: PhysicalConstants,
) -> QuarkyonicState | None:
    model = get_model(ground_state.model)
    if fq < 0.0 or fq > 1.0:
        return None

    n_q = n_b * fq
    n_n = n_b - n_q
    if n_n < 0.0:
        return None

    k_bu = _kbu_from_quark_density(n_q, settings, physical)
    if n_n <= 1.0e-14:
        n_n_id = 0.0
        k_f = k_bu
        shell_energy = 0.0
        interaction_energy = 0.0
    else:
        n_n_id = model.nid_from_n(n_n, ground_state.b)
        if n_n_id is None:
            return None
        k_f = (k_bu ** 3 + (6.0 * math.pi ** 2 * n_n_id) / physical.degeneracy_symmetric) ** (1.0 / 3.0)
        shell_energy = _nucleon_energy_shell(k_bu, k_f, settings.shell_integral_points, physical)
        volume_fraction = n_n / n_n_id if n_n_id > 0.0 else 1.0
        u_value = model.U(n_n, ground_state.b, ground_state.parameter_value)
        if u_value is None:
            return None
        interaction_energy = n_n * ground_state.a * u_value
        shell_energy = volume_fraction * shell_energy

    quark_energy = _quark_energy_density_from_kbu(k_bu, settings, physical)
    total_energy = shell_energy + interaction_energy + quark_energy
    if not math.isfinite(total_energy):
        return None

    return QuarkyonicState(
        n_b=n_b,
        n_over_n0=n_b / physical.n0,
        quark_fraction=fq,
        n_q=n_q,
        n_n=n_n,
        k_bu=k_bu,
        k_f=k_f,
        energy_density=total_energy,
    )


def _baryquark_state_at_fraction(
    ground_state: GroundStateResult,
    n_b: float,
    fq: float,
    settings: QuarkyonicSettings,
    physical: PhysicalConstants,
) -> QuarkyonicState | None:
    model = get_model(ground_state.model)
    if fq < 0.0 or fq > 1.0:
        return None

    n_q = n_b * fq
    n_n = n_b - n_q
    if n_n < 0.0:
        return None

    if n_n <= 1.0e-14:
        n_n_id = 0.0
        k_f = 0.0
        baryon_energy = 0.0
        interaction_energy = 0.0
    else:
        n_n_id = model.nid_from_n(n_n, ground_state.b)
        if n_n_id is None:
            return None
        k_f = ((6.0 * math.pi ** 2 * n_n_id) / physical.degeneracy_symmetric) ** (1.0 / 3.0)
        volume_fraction = n_n / n_n_id if n_n_id > 0.0 else 1.0
        baryon_energy = volume_fraction * eps_id_from_kf(k_f, physical.degeneracy_symmetric, physical)
        u_value = model.U(n_n, ground_state.b, ground_state.parameter_value)
        if u_value is None:
            return None
        interaction_energy = n_n * ground_state.a * u_value

    inner_quark_density = _quark_density_from_kbu(k_f, settings, physical)
    k_bu = _kbu_from_quark_density(n_q + inner_quark_density, settings, physical)
    quark_energy = _quark_energy_density_between_momenta(k_f, k_bu, settings, physical)
    total_energy = baryon_energy + interaction_energy + quark_energy
    if not math.isfinite(total_energy):
        return None

    return QuarkyonicState(
        n_b=n_b,
        n_over_n0=n_b / physical.n0,
        quark_fraction=fq,
        n_q=n_q,
        n_n=n_n,
        k_bu=k_bu,
        k_f=k_f,
        energy_density=total_energy,
    )


def _minimize_state_at_density(
    ground_state: GroundStateResult,
    n_b: float,
    settings: QuarkyonicSettings,
    physical: PhysicalConstants,
    gs_settings: GroundStateSettings,
) -> QuarkyonicState | None:
    lower = _lower_quark_fraction_bound(ground_state.model, n_b, ground_state.b, settings)
    if lower >= 1.0:
        return None

    fq_grid = linspace(lower, 1.0, settings.fq_scan_points)
    trial_states = [
        _state_at_fraction(ground_state, n_b, fq, settings, physical, gs_settings)
        for fq in fq_grid
    ]
    finite_trials = [(index, state) for index, state in enumerate(trial_states) if state is not None]
    if not finite_trials:
        return None

    best_state = min((state for _, state in finite_trials), key=lambda item: item.energy_density)
    finite_energies = [state.energy_density if state is not None else float("inf") for state in trial_states]
    candidate_intervals: list[tuple[float, float]] = []

    if len(fq_grid) >= 2:
        if finite_energies[0] <= finite_energies[1]:
            candidate_intervals.append((fq_grid[0], fq_grid[1]))
        if finite_energies[-1] <= finite_energies[-2]:
            candidate_intervals.append((fq_grid[-2], fq_grid[-1]))

    for index in range(1, len(fq_grid) - 1):
        e_left = finite_energies[index - 1]
        e_mid = finite_energies[index]
        e_right = finite_energies[index + 1]
        if not math.isfinite(e_mid):
            continue
        if e_mid <= e_left and e_mid <= e_right:
            candidate_intervals.append((fq_grid[index - 1], fq_grid[index + 1]))

    if not candidate_intervals:
        return best_state

    def objective(fq_value: float) -> float:
        state = _state_at_fraction(ground_state, n_b, fq_value, settings, physical, gs_settings)
        if state is None:
            return float("inf")
        return state.energy_density

    for left, right in candidate_intervals:
        fq_star, _ = golden_section_min(objective, left, right, tol=settings.refine_tol, max_iter=settings.refine_max_iter)
        refined_state = _state_at_fraction(ground_state, n_b, fq_star, settings, physical, gs_settings)
        if refined_state is not None and refined_state.energy_density < best_state.energy_density:
            best_state = refined_state
    return best_state


def compute_symmetric_quarkyonic_curve(
    model_name: str,
    parameter_value: float | None = None,
    parameter_search=None,
    settings: QuarkyonicSettings = DEFAULT_QUARKYONIC_SETTINGS,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    gs_settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> SymmetricQuarkyonicCurve:
    """Compute the symmetric quarkyonic sound-speed curve."""
    model = get_model(model_name)
    if not model.supports_symmetric_quarkyonic:
        raise ValueError(f"Model '{model_name}' does not support the symmetric quarkyonic workflow.")
    momentum_mode = _validate_momentum_mode(settings)

    ground_state = compute_ground_state_point(model_name, parameter_value, parameter_search, physical, gs_settings)
    density_ratios = linspace(settings.n_min_ratio, settings.n_max_ratio, settings.n_points)
    states: list[QuarkyonicState] = []
    for ratio in density_ratios:
        n_b = ratio * physical.n0
        state = _minimize_state_at_density(ground_state, n_b, settings, physical, gs_settings)
        if state is not None:
            states.append(state)

    if len(states) < 5:
        raise ValueError(f"Symmetric quarkyonic solve failed for model '{model_name}'.")

    n_values = [state.n_b for state in states]
    n_over_n0 = [state.n_over_n0 for state in states]
    eps_raw = [state.energy_density for state in states]
    fq_values = [state.quark_fraction for state in states]
    n_q_values = [state.n_q for state in states]
    n_n_values = [state.n_n for state in states]
    k_bu_values = [state.k_bu for state in states]
    k_f_values = [state.k_f for state in states]

    sound_speed_curve = reconstruct_sound_speed_curve(
        n_values,
        eps_raw,
        settings.smoothing_window,
        settings.smoothing_degree,
        settings.derivative_floor,
    )
    eps_values = sound_speed_curve.energy_density_smoothed
    mu_b_values = sound_speed_curve.chemical_potential
    p_values = sound_speed_curve.pressure
    dP_dn_values = sound_speed_curve.dP_dn
    d2eps_values = sound_speed_curve.d2eps_dn2
    vs2_values = sound_speed_curve.vs2

    finite_pairs = [(index, value) for index, value in enumerate(vs2_values) if math.isfinite(value)]
    if not finite_pairs:
        raise ValueError(f"Symmetric quarkyonic derivative reconstruction failed for model '{model_name}'.")
    peak_index, peak_value = max(finite_pairs, key=lambda item: item[1])

    return SymmetricQuarkyonicCurve(
        model=ground_state.model,
        momentum_mode=momentum_mode,
        parameter_name=ground_state.parameter_name,
        parameter_value=ground_state.parameter_value,
        a=ground_state.a,
        b=ground_state.b,
        K0=ground_state.K0,
        n=n_values,
        n_over_n0=n_over_n0,
        eps_raw=eps_raw,
        eps=eps_values,
        mu_b=mu_b_values,
        P=p_values,
        dP_dn=dP_dn_values,
        d2eps_dn2=d2eps_values,
        vs2=vs2_values,
        quark_fraction=fq_values,
        n_q=n_q_values,
        n_n=n_n_values,
        k_bu=k_bu_values,
        k_f=k_f_values,
        vs2_max=peak_value,
        n_tr=n_values[peak_index],
        n_tr_over_n0=n_over_n0[peak_index],
    )


class AsymmetricQuarkyonicEOS:
    """Quarkyonic EOS for asymmetric hadronic parameters."""

    def __init__(
        self,
        fit_result: AsymmetricFitResult,
        physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
        settings: QuarkyonicSettings = DEFAULT_QUARKYONIC_SETTINGS,
    ) -> None:
        self.fit_result = fit_result
        self.physical = physical
        self.settings = settings
        self.model = get_model(fit_result.model)

    @property
    def kappa(self) -> float:
        return self.settings.lambda_momentum_mev / self.physical.hbarc

    @property
    def m_q(self) -> float:
        return self.physical.m_nucleon / float(self.settings.nc)

    def euid(self, k_bu: float, y_value: float) -> float:
        pref = self.physical.degeneracy_species / (2.0 * math.pi ** 2)
        upper = k_bu / float(self.settings.nc)

        def integrand(q_value: float) -> float:
            density_factor = q_value * math.sqrt(self.kappa * self.kappa + q_value * q_value)
            return density_factor * math.sqrt((self.physical.hbarc * q_value) ** 2 + self.m_q * self.m_q)

        integral = simpson_integral(integrand, 0.0, upper, self.settings.quark_integral_points)
        return float(self.settings.nc) * pref * (1.0 + y_value) / (2.0 - y_value) * integral

    def edid(self, k_bu: float) -> float:
        pref = self.physical.degeneracy_species / (2.0 * math.pi ** 2)
        upper = k_bu / float(self.settings.nc)

        def integrand(q_value: float) -> float:
            density_factor = q_value * math.sqrt(self.kappa * self.kappa + q_value * q_value)
            return density_factor * math.sqrt((self.physical.hbarc * q_value) ** 2 + self.m_q * self.m_q)

        integral = simpson_integral(integrand, 0.0, upper, self.settings.quark_integral_points)
        return float(self.settings.nc) * pref * integral

    def nuid(self, k_bu: float, y_value: float) -> float:
        pref = self.physical.degeneracy_species / (2.0 * math.pi ** 2)
        upper = k_bu / float(self.settings.nc)

        def integrand(q_value: float) -> float:
            return q_value * math.sqrt(self.kappa * self.kappa + q_value * q_value)

        integral = simpson_integral(integrand, 0.0, upper, self.settings.quark_integral_points)
        return float(self.settings.nc) * pref * (1.0 + y_value) / (2.0 - y_value) * integral

    def ndid(self, k_bu: float) -> float:
        pref = self.physical.degeneracy_species / (2.0 * math.pi ** 2)
        upper = k_bu / float(self.settings.nc)

        def integrand(q_value: float) -> float:
            return q_value * math.sqrt(self.kappa * self.kappa + q_value * q_value)

        integral = simpson_integral(integrand, 0.0, upper, self.settings.quark_integral_points)
        return float(self.settings.nc) * pref * integral

    def kbu_solver(self, n_b: float, fq: float, y_value: float) -> float:
        pref = self.physical.degeneracy_species / (2.0 * math.pi ** 2)
        w_value = (3.0 / (2.0 - y_value)) * pref
        n_q = n_b * fq
        inside = (3.0 * n_q / w_value + self.kappa ** 3) ** (2.0 / 3.0) - self.kappa ** 2
        return float(self.settings.nc) * math.sqrt(max(inside, 0.0))

    def hadronic_densities(self, n_b: float, fq: float, y_value: float) -> tuple[float, float]:
        n_h = n_b * (1.0 - fq)
        return n_h * y_value, n_h * (1.0 - y_value)

    def volume_fractions(self, n_b: float, fq: float, y_value: float) -> tuple[float | None, float | None]:
        n_p, n_n = self.hadronic_densities(n_b, fq, y_value)
        x_p = self.fit_result.b_n * n_p + self.fit_result.b_pn * n_n
        x_n = self.fit_result.b_pn * n_p + self.fit_result.b_n * n_n
        return self.model.species_volume_fraction(x_p), self.model.species_volume_fraction(x_n)

    def kfp_solver(self, n_b: float, fq: float, y_value: float, k_bu: float) -> float:
        n_p, _ = self.hadronic_densities(n_b, fq, y_value)
        if n_p <= 0.0:
            return k_bu
        f_p, _ = self.volume_fractions(n_b, fq, y_value)
        if f_p is None or f_p <= 0.0:
            return math.nan
        n_p_id = n_p / f_p
        value = k_bu ** 3 + (6.0 * math.pi ** 2 / self.physical.degeneracy_species) * n_p_id
        return value ** (1.0 / 3.0)

    def kfn_solver(self, n_b: float, fq: float, y_value: float, k_bu: float) -> float:
        _, n_n = self.hadronic_densities(n_b, fq, y_value)
        if n_n <= 0.0:
            return k_bu
        _, f_n = self.volume_fractions(n_b, fq, y_value)
        if f_n is None or f_n <= 0.0:
            return math.nan
        n_n_id = n_n / f_n
        value = k_bu ** 3 + (6.0 * math.pi ** 2 / self.physical.degeneracy_species) * n_n_id
        return value ** (1.0 / 3.0)

    def enid(self, k_bu: float, k_fn: float) -> float:
        if k_fn <= k_bu:
            return 0.0
        return eps_id_from_kf(k_fn, self.physical.degeneracy_species, self.physical) - eps_id_from_kf(
            k_bu,
            self.physical.degeneracy_species,
            self.physical,
        )

    def epid(self, k_bu: float, k_fp: float) -> float:
        if k_fp <= k_bu:
            return 0.0
        return eps_id_from_kf(k_fp, self.physical.degeneracy_species, self.physical) - eps_id_from_kf(
            k_bu,
            self.physical.degeneracy_species,
            self.physical,
        )

    def energy_hq_of_fq_y(self, fq: float, n_b: float, y_value: float) -> float:
        k_bu = self.kbu_solver(n_b, fq, y_value)
        k_fp = self.kfp_solver(n_b, fq, y_value, k_bu)
        k_fn = self.kfn_solver(n_b, fq, y_value, k_bu)
        if not math.isfinite(k_bu) or not math.isfinite(k_fp) or not math.isfinite(k_fn):
            return math.nan

        e_u = self.euid(k_bu, y_value)
        e_d = self.edid(k_bu)
        e_n = self.enid(k_bu, k_fn)
        e_p = self.epid(k_bu, k_fp)

        n_p, n_n = self.hadronic_densities(n_b, fq, y_value)
        n_h = n_p + n_n
        f_p, f_n = self.volume_fractions(n_b, fq, y_value)
        if n_p > 0.0 and (f_p is None or f_p <= 0.0):
            return math.nan
        if n_n > 0.0 and (f_n is None or f_n <= 0.0):
            return math.nan
        f_p = 1.0 if f_p is None else f_p
        f_n = 1.0 if f_n is None else f_n

        hadronic_shell = f_p * e_p + f_n * e_n
        U_value = self.model.U(n_h, 0.0, self.fit_result.parameter_value)
        if U_value is None:
            return math.nan
        a_eff = attractive_coefficient(y_value, self.fit_result.a_n, self.fit_result.a_pn)
        interaction = n_h * a_eff * U_value
        return e_u + e_d + hadronic_shell + interaction

    def lepton_kf_from_mu(self, mu_l: float, mass_l: float) -> float:
        return math.sqrt(max(mu_l * mu_l - mass_l * mass_l, 0.0)) / self.physical.hbarc

    def lepton_density_from_mu(self, mu_l: float, mass_l: float) -> float:
        if mu_l <= mass_l:
            return 0.0
        kf = self.lepton_kf_from_mu(mu_l, mass_l)
        return kf ** 3 / (3.0 * math.pi ** 2)

    def lepton_energy_density_from_mu(self, mu_l: float, mass_l: float) -> float:
        if mu_l <= mass_l:
            return 0.0
        kf = self.lepton_kf_from_mu(mu_l, mass_l)

        def integrand(k_value: float) -> float:
            return k_value * k_value * math.sqrt((self.physical.hbarc * k_value) ** 2 + mass_l * mass_l)

        return simpson_integral(integrand, 0.0, kf, self.settings.shell_integral_points) / (math.pi ** 2)

    def charge_density_hq(self, n_b: float, fq: float, y_value: float) -> float:
        k_bu = self.kbu_solver(n_b, fq, y_value)
        n_p, _ = self.hadronic_densities(n_b, fq, y_value)
        # `nuid()` and `ndid()` already return quark number densities.
        # Multiplying by `N_c` again overcounts the quark charge by a factor of 3.
        n_u = self.nuid(k_bu, y_value)
        n_d = self.ndid(k_bu)
        return n_p + (2.0 / 3.0) * n_u - (1.0 / 3.0) * n_d

    def mu_q_fd(self, n_b: float, fq: float, y_value: float, dy: float = 1.0e-5) -> float:
        y_1 = max(1.0e-8, y_value - dy)
        y_2 = min(0.5 - 1.0e-8, y_value + dy)
        e_1 = self.energy_hq_of_fq_y(fq, n_b, y_1)
        e_2 = self.energy_hq_of_fq_y(fq, n_b, y_2)
        if not math.isfinite(e_1) or not math.isfinite(e_2):
            return math.nan
        return (e_2 - e_1) / (n_b * (y_2 - y_1))

    def mu_e_from_y(self, n_b: float, fq: float, y_value: float) -> float:
        mu_q = self.mu_q_fd(n_b, fq, y_value)
        if not math.isfinite(mu_q):
            return math.nan
        return -mu_q

    def neutrality_residual(self, y_value: float, n_b: float, fq: float) -> float:
        mu_e = self.mu_e_from_y(n_b, fq, y_value)
        if not math.isfinite(mu_e):
            return math.nan
        rho_hq = self.charge_density_hq(n_b, fq, y_value)
        rho_e = self.lepton_density_from_mu(mu_e, 0.511)
        rho_mu = self.lepton_density_from_mu(mu_e, 105.66)
        return rho_hq - (rho_e + rho_mu)

    def solve_y_beta_equilibrium(
        self,
        n_b: float,
        fq: float,
        y_min: float = 1.0e-6,
        y_max: float = 0.5 - 1.0e-6,
        n_scan: int | None = None,
    ) -> tuple[float, float, float, float]:
        if n_scan is None:
            n_scan = self.settings.beta_y_scan_points
        y_grid = linspace(y_min, y_max, n_scan)
        values = [self.neutrality_residual(y_value, n_b, fq) for y_value in y_grid]
        y_star = math.nan

        for index in range(len(y_grid) - 1):
            left = y_grid[index]
            right = y_grid[index + 1]
            f_left = values[index]
            f_right = values[index + 1]
            if not math.isfinite(f_left) or not math.isfinite(f_right):
                continue
            if abs(f_left) < 1.0e-12:
                y_star = left
                break
            if f_left * f_right >= 0.0:
                continue

            for _ in range(200):
                mid = 0.5 * (left + right)
                f_mid = self.neutrality_residual(mid, n_b, fq)
                if not math.isfinite(f_mid):
                    break
                if abs(f_mid) < 1.0e-12:
                    left = mid
                    right = mid
                    break
                if f_left * f_mid <= 0.0:
                    right = mid
                    f_right = f_mid
                else:
                    left = mid
                    f_left = f_mid
            y_star = 0.5 * (left + right)
            break

        if not math.isfinite(y_star):
            return math.nan, math.nan, math.nan, math.nan

        mu_e = self.mu_e_from_y(n_b, fq, y_star)
        eps_e = self.lepton_energy_density_from_mu(mu_e, 0.511)
        eps_mu = self.lepton_energy_density_from_mu(mu_e, 105.66)
        return y_star, mu_e, eps_e, eps_mu

    def total_energy_beta(self, fq: float, n_b: float, y_scan: int | None = None) -> float:
        y_star, _, eps_e, eps_mu = self.solve_y_beta_equilibrium(n_b, fq, n_scan=y_scan)
        if not math.isfinite(y_star):
            return math.nan
        eps_hq = self.energy_hq_of_fq_y(fq, n_b, y_star)
        if not math.isfinite(eps_hq):
            return math.nan
        return eps_hq + eps_e + eps_mu

    def beta_lower_quark_fraction_bound(self, n_b: float) -> float:
        """Return a conservative lower bound that keeps the hadronic sector finite."""
        if n_b <= 0.0:
            return 0.0

        model_name = self.fit_result.model
        b_limit = max(self.fit_result.b_n, self.fit_result.b_pn)
        max_n_h = _max_hadronic_density(model_name, b_limit)
        if not math.isfinite(max_n_h) or n_b <= max_n_h:
            return 0.0
        return max(0.0, 1.0 - max_n_h / n_b + self.settings.fq_min_shift)

    def fixed_y_lower_quark_fraction_bound(self, n_b: float, y_value: float) -> float:
        """Return a conservative lower bound that keeps both hadronic species finite."""
        if n_b <= 0.0:
            return 0.0

        coeff_p = self.fit_result.b_n * y_value + self.fit_result.b_pn * (1.0 - y_value)
        coeff_n = self.fit_result.b_pn * y_value + self.fit_result.b_n * (1.0 - y_value)
        b_limit = max(coeff_p, coeff_n)
        if b_limit <= 0.0:
            return 0.0

        max_n_h = _max_hadronic_density(self.fit_result.model, b_limit)
        if not math.isfinite(max_n_h) or n_b <= max_n_h:
            return 0.0
        return max(0.0, 1.0 - max_n_h / n_b + self.settings.fq_min_shift)

    @staticmethod
    def _candidate_intervals_from_trials(
        fq_grid: list[float],
        trial_energies: list[float],
    ) -> list[tuple[float, float]]:
        candidate_intervals: list[tuple[float, float]] = []

        if len(fq_grid) >= 2:
            if math.isfinite(trial_energies[0]) and math.isfinite(trial_energies[1]) and trial_energies[0] <= trial_energies[1]:
                candidate_intervals.append((fq_grid[0], fq_grid[1]))
            if math.isfinite(trial_energies[-1]) and math.isfinite(trial_energies[-2]) and trial_energies[-1] <= trial_energies[-2]:
                candidate_intervals.append((fq_grid[-2], fq_grid[-1]))

        for index in range(1, len(fq_grid) - 1):
            e_left = trial_energies[index - 1]
            e_mid = trial_energies[index]
            e_right = trial_energies[index + 1]
            if not (math.isfinite(e_left) and math.isfinite(e_mid) and math.isfinite(e_right)):
                continue
            if e_mid <= e_left and e_mid <= e_right:
                candidate_intervals.append((fq_grid[index - 1], fq_grid[index + 1]))

        return candidate_intervals

    def _coarse_fixed_y_candidates(
        self,
        n_b: float,
        y_value: float,
        fq_min: float,
        fq_max: float,
    ) -> tuple[tuple[float, float], list[tuple[float, float]]]:
        """Return the best coarse fixed-`y` point and refinement intervals."""
        fq_grid = linspace(fq_min, fq_max, self.settings.fq_scan_points)
        trial_energies = [self.energy_hq_of_fq_y(fq, n_b, y_value) for fq in fq_grid]
        finite_trials = [
            (index, fq_value, energy)
            for index, (fq_value, energy) in enumerate(zip(fq_grid, trial_energies))
            if math.isfinite(energy)
        ]
        if not finite_trials:
            return (math.nan, math.nan), []

        _, best_fq, best_energy = min(finite_trials, key=lambda item: item[2])
        return (best_fq, best_energy), self._candidate_intervals_from_trials(fq_grid, trial_energies)

    def _coarse_beta_candidates(
        self,
        n_b: float,
        fq_min: float,
        fq_max: float,
        y_scan: int,
    ) -> tuple[tuple[float, float], list[tuple[float, float]]]:
        """Return the best coarse point and refinement intervals for beta equilibrium."""
        fq_grid = linspace(fq_min, fq_max, self.settings.fq_scan_points)
        trial_energies = [self.total_energy_beta(fq, n_b, y_scan=y_scan) for fq in fq_grid]
        finite_trials = [
            (index, fq_value, energy)
            for index, (fq_value, energy) in enumerate(zip(fq_grid, trial_energies))
            if math.isfinite(energy)
        ]
        if not finite_trials:
            return (math.nan, math.nan), []

        _, best_fq, best_energy = min(finite_trials, key=lambda item: item[2])
        return (best_fq, best_energy), self._candidate_intervals_from_trials(fq_grid, trial_energies)

    def solve_fq_fixed_y(self, n_b: float, y_value: float, fq_min: float = 0.0, fq_max: float = 1.0) -> tuple[float, float]:
        lower = max(fq_min, self.fixed_y_lower_quark_fraction_bound(n_b, y_value))
        if lower >= fq_max:
            return math.nan, math.nan

        objective = lambda fq: self.energy_hq_of_fq_y(fq, n_b, y_value)
        fq_star, energy_star = golden_section_min_safe(
            objective,
            lower,
            fq_max,
            tol=self.settings.refine_tol,
            max_iter=self.settings.refine_max_iter,
        )
        if math.isfinite(fq_star) and math.isfinite(energy_star):
            return fq_star, energy_star

        (best_fq, best_energy), candidate_intervals = self._coarse_fixed_y_candidates(n_b, y_value, lower, fq_max)
        if not math.isfinite(best_fq) or not math.isfinite(best_energy):
            return math.nan, math.nan

        for left, right in candidate_intervals:
            fq_star, energy_star = golden_section_min_safe(
                objective,
                left,
                right,
                tol=self.settings.refine_tol,
                max_iter=self.settings.refine_max_iter,
            )
            if math.isfinite(energy_star) and energy_star < best_energy:
                best_fq = fq_star
                best_energy = energy_star
        return best_fq, best_energy

    def solve_fq_beta(
        self,
        n_b: float,
        fq_min: float = 0.0,
        fq_max: float = 1.0,
        y_scan: int | None = None,
    ) -> tuple[float, float]:
        lower = max(fq_min, self.beta_lower_quark_fraction_bound(n_b))
        if lower >= fq_max:
            return math.nan, math.nan

        objective = lambda fq: self.total_energy_beta(fq, n_b, y_scan=y_scan)
        fq_star, energy_star = golden_section_min_safe(
            objective,
            lower,
            fq_max,
            tol=self.settings.refine_tol,
            max_iter=self.settings.refine_max_iter,
        )
        if math.isfinite(fq_star) and math.isfinite(energy_star):
            return fq_star, energy_star

        (best_fq, best_energy), candidate_intervals = self._coarse_beta_candidates(n_b, lower, fq_max, y_scan)
        if not math.isfinite(best_fq) or not math.isfinite(best_energy):
            return math.nan, math.nan

        for left, right in candidate_intervals:
            fq_star, energy_star = golden_section_min_safe(
                objective,
                left,
                right,
                tol=self.settings.refine_tol,
                max_iter=self.settings.refine_max_iter,
            )
            if math.isfinite(energy_star) and energy_star < best_energy:
                best_fq = fq_star
                best_energy = energy_star
        return best_fq, best_energy

    def build_fixed_y_row(self, n_b: float, y_value: float) -> AsymmetricQuarkyonicRow | None:
        fq_star, energy = self.solve_fq_fixed_y(n_b, y_value)
        if not math.isfinite(fq_star) or not math.isfinite(energy):
            return None
        k_bu = self.kbu_solver(n_b, fq_star, y_value)
        k_fp = self.kfp_solver(n_b, fq_star, y_value, k_bu)
        k_fn = self.kfn_solver(n_b, fq_star, y_value, k_bu)
        n_p, n_n = self.hadronic_densities(n_b, fq_star, y_value)
        return AsymmetricQuarkyonicRow(
            n_b=n_b,
            n_over_n0=n_b / self.physical.n0,
            y=y_value,
            quark_fraction=fq_star,
            k_bu=k_bu,
            k_fp=k_fp,
            k_fn=k_fn,
            n_p=n_p,
            n_n=n_n,
            energy_density=energy,
            mu_e=None,
            eps_e=0.0,
            eps_mu=0.0,
        )

    def build_beta_row(self, n_b: float) -> AsymmetricQuarkyonicRow | None:
        fq_star, energy = self.solve_fq_beta(n_b)
        if not math.isfinite(fq_star) or not math.isfinite(energy):
            return None
        y_star, mu_e, eps_e, eps_mu = self.solve_y_beta_equilibrium(n_b, fq_star)
        if not math.isfinite(y_star):
            return None
        k_bu = self.kbu_solver(n_b, fq_star, y_star)
        k_fp = self.kfp_solver(n_b, fq_star, y_star, k_bu)
        k_fn = self.kfn_solver(n_b, fq_star, y_star, k_bu)
        n_p, n_n = self.hadronic_densities(n_b, fq_star, y_star)
        return AsymmetricQuarkyonicRow(
            n_b=n_b,
            n_over_n0=n_b / self.physical.n0,
            y=y_star,
            quark_fraction=fq_star,
            k_bu=k_bu,
            k_fp=k_fp,
            k_fn=k_fn,
            n_p=n_p,
            n_n=n_n,
            energy_density=energy,
            mu_e=mu_e,
            eps_e=eps_e,
            eps_mu=eps_mu,
        )


def _finalize_profile(rows: list[AsymmetricQuarkyonicRow], settings: QuarkyonicSettings) -> list[AsymmetricQuarkyonicRow]:
    if len(rows) < 3:
        return rows
    densities = [row.n_b for row in rows]
    eps_values = [row.energy_density for row in rows]
    sound_speed_curve = reconstruct_sound_speed_curve(
        densities,
        eps_values,
        settings.smoothing_window,
        settings.smoothing_degree,
        settings.derivative_floor,
    )
    eps_smooth = sound_speed_curve.energy_density_smoothed
    mu_b_values = sound_speed_curve.chemical_potential
    pressure_values = sound_speed_curve.pressure
    dP_dn_values = sound_speed_curve.dP_dn
    d2eps_values = sound_speed_curve.d2eps_dn2

    finalized: list[AsymmetricQuarkyonicRow] = []
    for row, mu_b, pressure, vs2 in zip(rows, mu_b_values, pressure_values, sound_speed_curve.vs2):
        finalized.append(
            AsymmetricQuarkyonicRow(
                n_b=row.n_b,
                n_over_n0=row.n_over_n0,
                y=row.y,
                quark_fraction=row.quark_fraction,
                k_bu=row.k_bu,
                k_fp=row.k_fp,
                k_fn=row.k_fn,
                n_p=row.n_p,
                n_n=row.n_n,
                energy_density=row.energy_density,
                mu_e=row.mu_e,
                eps_e=row.eps_e,
                eps_mu=row.eps_mu,
                mu_b=mu_b,
                pressure=pressure,
                vs2=vs2,
            )
        )
    return finalized


def compute_asymmetric_quarkyonic_profiles(
    fit_result: AsymmetricFitResult,
    proton_fraction_values: tuple[float, ...] | list[float],
    include_beta_equilibrium: bool = True,
    settings: QuarkyonicSettings = DEFAULT_QUARKYONIC_SETTINGS,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
) -> AsymmetricQuarkyonicResult:
    """Compute fixed-y and beta-equilibrium asymmetric quarkyonic profiles."""
    model = get_model(fit_result.model)
    if not model.supports_asymmetric_quarkyonic:
        raise ValueError(f"Model '{fit_result.model}' does not support the asymmetric quarkyonic workflow.")
    momentum_mode = _validate_momentum_mode(settings)
    if momentum_mode == "baryquark":
        raise ValueError(
            "The current baryquark implementation is only available for the symmetric workflow. "
            "The asymmetric extension requires a dedicated prescription for proton/neutron and u/d shell boundaries."
        )

    eos = AsymmetricQuarkyonicEOS(fit_result, physical, settings)
    density_ratios = linspace(settings.n_min_ratio, settings.n_max_ratio, settings.n_points)
    density_grid = [ratio * physical.n0 for ratio in density_ratios]

    fixed_y_profiles: dict[str, list[AsymmetricQuarkyonicRow]] = {}
    for y_value in proton_fraction_values:
        rows: list[AsymmetricQuarkyonicRow] = []
        for density in density_grid:
            row = eos.build_fixed_y_row(density, y_value)
            if row is not None:
                rows.append(row)
        fixed_y_profiles[f"y_{y_value:.3f}"] = _finalize_profile(rows, settings)

    beta_profile: list[AsymmetricQuarkyonicRow] = []
    if include_beta_equilibrium:
        beta_rows: list[AsymmetricQuarkyonicRow] = []
        for density in density_grid:
            row = eos.build_beta_row(density)
            if row is not None:
                beta_rows.append(row)
        beta_profile = _finalize_profile(beta_rows, settings)

    return AsymmetricQuarkyonicResult(
        model=fit_result.model,
        momentum_mode=momentum_mode,
        parameter_name=fit_result.parameter_name,
        parameter_value=fit_result.parameter_value,
        fixed_y_profiles=fixed_y_profiles,
        beta_profile=beta_profile,
    )
