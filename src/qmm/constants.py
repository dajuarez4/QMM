"""Shared constants and solver settings."""

from __future__ import annotations

from dataclasses import dataclass, field


def make_grid(start: float, stop: float, count: int) -> tuple[float, ...]:
    """Return a simple evenly spaced tuple."""
    if count <= 0:
        return ()
    if count == 1:
        return (float(start),)
    step = (stop - start) / float(count - 1)
    return tuple(start + step * float(index) for index in range(count))


@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants in fm/MeV units unless noted otherwise."""

    hbarc: float = 197.3269804
    m_nucleon: float = 938.0
    degeneracy_symmetric: float = 4.0
    degeneracy_species: float = 2.0
    n0: float = 0.16
    binding_energy: float = -16.0
    symmetry_energy_j: float = 32.5
    symmetry_slope_l: float = 58.9


@dataclass(frozen=True)
class GroundStateSettings:
    """Numerical controls for the zero-temperature hadronic fit."""

    b_min: float = 1.0e-6
    b_max: float = 6.0
    b_scan_steps: int = 5000
    n_integral_points: int = 4000
    saturation_left: float = 0.05
    saturation_right: float = 0.30
    derivative_step: float = 1.0e-6


@dataclass(frozen=True)
class ParameterSearchSettings:
    """Controls for solving a model parameter from a target `K0`."""

    enabled: bool = False
    target_k0: float | None = None
    parameter_min: float | None = None
    parameter_max: float | None = None
    scan_steps: int = 121
    tol: float = 1.0e-8


@dataclass(frozen=True)
class QuantumCriticalSettings:
    """SCF settings for the finite-temperature critical point."""

    n_k_fd: int = 400
    k_max_fd: float = 20.0
    mu_min: float = -200.0
    mu_max: float = 1200.0
    mu_scf_tol: float = 1.0e-8
    mu_scf_max_iter: int = 80
    mu_scf_damping: float = 0.70
    mu_delta: float = 0.25
    t_min_cp: float = 10.0
    t_max_cp: float = 22.0
    n_min_cp: float = 0.045
    n_max_cp: float = 0.075
    coarse_t_count: int = 17
    coarse_n_count: int = 17
    h_n: float = 2.0e-4
    h_t: float = 2.0e-2
    outer_tol_dPdn: float = 1.0e-5
    outer_tol_d2Pdn2: float = 1.0e-4
    outer_max_iter: int = 35
    outer_damping_t: float = 0.70
    outer_damping_n: float = 0.70
    outer_max_step_t: float = 0.75
    outer_max_step_n: float = 0.004
    outer_min_lambda: float = 1.0e-3
    cache_round_digits: int = 8

    @staticmethod
    def quick() -> "QuantumCriticalSettings":
        return QuantumCriticalSettings(
            n_k_fd=220,
            coarse_t_count=11,
            coarse_n_count=11,
            outer_max_iter=24,
        )


@dataclass(frozen=True)
class HadronicEOSSettings:
    """Settings for a dense zero-temperature hadronic EOS table."""

    n_min_ratio: float = 0.2
    n_max_ratio: float = 5.0
    n_points: int = 160
    smoothing_window: int = 9
    smoothing_degree: int = 3
    derivative_floor: float = 1.0e-10


@dataclass(frozen=True)
class QuarkyonicSettings:
    """Settings for zero-temperature quarkyonic minimization.

    `momentum_mode` selects how baryons and quarks fill momentum space:

    - `quarkyonic`: quark Fermi sea plus baryonic shell
    - `baryquark`: baryonic Fermi sea plus quark shell
    """

    n_min_ratio: float = 1.0
    n_max_ratio: float = 8.0
    n_points: int = 120
    fq_scan_points: int = 121
    fq_min_shift: float = 1.0e-5
    refine_tol: float = 1.0e-6
    refine_max_iter: int = 400
    shell_integral_points: int = 400
    quark_integral_points: int = 400
    smoothing_window: int = 9
    smoothing_degree: int = 3
    derivative_floor: float = 1.0e-10
    beta_y_scan_points: int = 160
    nc: int = 3
    quark_degeneracy: float = 4.0
    lambda_momentum_mev: float = 300.0
    momentum_mode: str = "quarkyonic"

    @staticmethod
    def quick() -> "QuarkyonicSettings":
        return QuarkyonicSettings(
            n_points=60,
            fq_scan_points=61,
            shell_integral_points=200,
            quark_integral_points=200,
        )


@dataclass(frozen=True)
class AsymmetricFitSettings:
    """Controls for the asymmetric hadronic fit."""

    enabled: bool = False
    branch_mode: str = "target_l"
    target_j: float = 32.5
    target_l: float = 58.9
    target_k0: float | None = None
    delta_scan_steps: int = 401
    derivative_step_l: float = 1.0e-4
    proton_fraction_values: tuple[float, ...] = field(
        default_factory=lambda: (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    )
    beta_equilibrium: bool = True


@dataclass(frozen=True)
class NeutronStarSettings:
    """Controls for the beta-equilibrium neutron-star sequence."""

    sequence_points: int = 48
    central_pressure_min_mev_fm3: float = 0.10
    central_pressure_max_fraction: float = 0.995
    surface_pressure_mev_fm3: float | None = 0.05
    start_radius_cm: float = 1.0
    step_cm: float = 2.5e3
    max_radius_cm: float = 4.0e6


@dataclass(frozen=True)
class OutputSettings:
    """Output controls."""

    directory: str = "results/default_run"
    write_json: bool = True
    write_csv: bool = True
    write_plots: bool = True
    plot_formats: tuple[str, ...] = ("png", "pdf")
    collection_search_root: str | None = None
    collection_output_directory: str | None = None


DEFAULT_PHYSICAL_CONSTANTS = PhysicalConstants()
DEFAULT_GROUND_STATE_SETTINGS = GroundStateSettings()
DEFAULT_QUANTUM_SETTINGS = QuantumCriticalSettings()
DEFAULT_HADRONIC_EOS_SETTINGS = HadronicEOSSettings()
DEFAULT_QUARKYONIC_SETTINGS = QuarkyonicSettings()
DEFAULT_ASYMMETRIC_SETTINGS = AsymmetricFitSettings()
DEFAULT_NEUTRON_STAR_SETTINGS = NeutronStarSettings()

DEFAULT_CLAUSIUS_RANGE = (0.0, 4.74)
DEFAULT_CLAUSIUS_CS_RANGE = (0.0, 6.0)
DEFAULT_CLAUSIUS_TVM_RANGE = (0.0, 6.0)
DEFAULT_DIETERICI_RANGE = (5.0 / 3.0, 2.0)
