
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

GEV4_TO_MEV_FM3 = 1.30148926289e5


@dataclass(frozen=True)
class SoundSpeedCurve:
    """Thermodynamic quantities reconstructed from `epsilon(n_B)`."""

    energy_density_smoothed: list[float]
    chemical_potential: list[float]
    pressure: list[float]
    dP_dn: list[float]
    d2eps_dn2: list[float]
    vs2: list[float]


def reconstruct_sound_speed_curve(
    densities: list[float],
    energy_density: list[float],
    smoothing_window: int,
    smoothing_degree: int,
    derivative_floor: float = 1.0e-12,
) -> SoundSpeedCurve:

    return reconstruct_sound_speed_curve_gradient_check(
        densities, energy_density, derivative_floor=derivative_floor
    )


def reconstruct_sound_speed_curve_gradient_check(
    densities: list[float],
    energy_density: list[float],
    derivative_floor: float = 1.0e-12,
    export_path: str | Path | None = None,
    mev_fm3_factor: float = GEV4_TO_MEV_FM3,
) -> SoundSpeedCurve:

    import numpy as np

    density_array = np.asarray(densities, dtype=float)
    energy_array = np.asarray(energy_density, dtype=float)

    if density_array.ndim != 1 or energy_array.ndim != 1:
        raise ValueError("Sound-speed inputs must be one-dimensional.")
    if density_array.size != energy_array.size:
        raise ValueError("Density and energy-density arrays must have the same length.")
    if density_array.size < 3:
        raise ValueError("`numpy.gradient(..., edge_order=2)` requires at least three points.")

    # Worked example: derivatives from a table of total energy density.
    # These illustrative values use epsilon = 939*n_B + 30*n_B**2;
    # they are not simulation results. Include rest-mass energy in epsilon.
    #
    #   n_B [fm^-3] | epsilon [MeV fm^-3] | mu_B [MeV]
    #   ------------|---------------------|-----------
    #       0.10    |          94.20      |   945.0
    #       0.20    |         189.00      |   951.0
    #       0.30    |         284.40      |   957.0
    #       0.40    |         380.40      |   963.0
    #       0.50    |         477.00      |   969.0
    #
    # On a uniform grid with spacing h, the first gradient uses
    #   mu_B[i] = (epsilon[i+1] - epsilon[i-1]) / (2*h).
    # At n_B = 0.30, h = 0.10:
    #   mu_B = (380.40 - 189.00) / 0.20 = 957 MeV.
    # The second gradient applies the same rule to the mu_B column:
    #   dmu_B/dn_B = (963 - 951) / 0.20 = 60 MeV fm^3.
    # Zero-temperature thermodynamics then gives
    #   P = n_B*mu_B - epsilon = 0.30*957 - 284.40 = 2.70 MeV fm^-3,
    #   vs2 = (c_s/c)**2 = dP/depsilon = n_B*(dmu_B/dn_B)/mu_B
    #       = 0.30*60/957 = 0.0188088, so c_s/c = sqrt(vs2) = 0.137145.
    #
    # At the endpoints, edge_order=2 uses one-sided differences; for example,
    #   mu_B[0] = (-3*epsilon[0] + 4*epsilon[1] - epsilon[2]) / (2*h).
    # For nonuniform grids, numpy.gradient uses the actual neighboring
    # spacings supplied in density_array, rather than a constant h.
    # Both gradients are finite-difference approximations without smoothing.
    # Applying gradient twice is not generally the same stencil as the
    # nearest-neighbor second difference of epsilon.
    mu_b_array = np.gradient(energy_array, density_array, edge_order=2)
    mu_bb_array = np.gradient(mu_b_array, density_array, edge_order=2)
    pressure_array = density_array * mu_b_array - energy_array
    dP_dn_array = density_array * mu_bb_array

    vs2_array = np.full(density_array.shape, np.nan, dtype=float)
    valid_mask = (
        np.isfinite(mu_b_array)
        & np.isfinite(mu_bb_array)
        & np.isfinite(dP_dn_array)
        & (np.abs(mu_b_array) >= derivative_floor)
    )
    vs2_array[valid_mask] = density_array[valid_mask] / mu_b_array[valid_mask] * mu_bb_array[valid_mask]

    if export_path is not None:
        export_array = np.column_stack((energy_array * mev_fm3_factor, pressure_array * mev_fm3_factor))
        export_target = Path(export_path)
        export_target.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            export_target,
            export_array,
            header="MeV/fm^3               MeV/fm^3\nenergy_density         pressure",
            fmt="%.10e",
            comments="# ",
        )

    return SoundSpeedCurve(
        energy_density_smoothed=energy_array.tolist(),
        chemical_potential=mu_b_array.tolist(),
        pressure=pressure_array.tolist(),
        dP_dn=dP_dn_array.tolist(),
        d2eps_dn2=mu_bb_array.tolist(),
        vs2=vs2_array.tolist(),
    )


def compute_vs2_only(densities, energy_density):
    """Return the direct `numpy.gradient` estimate of `v_s^2`."""
    import numpy as np

    n = np.asarray(densities, dtype=float)
    E = np.asarray(energy_density, dtype=float)
    mu_b = np.gradient(E, n, edge_order=2)
    mu_bb = np.gradient(mu_b, n, edge_order=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return n * mu_bb / mu_b
