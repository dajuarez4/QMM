"""Sound-speed reconstruction from a zero-temperature energy-density curve."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .numerics import local_polynomial_regression

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
    """Compute `v_s^2` from `epsilon(n_B)` using local polynomial derivatives.

    The reconstruction follows the zero-temperature identities used in the
    quarkyonic workflow:

    - `mu_B = d epsilon / d n_B`
    - `P = n_B * mu_B - epsilon`
    - `dP/dn_B = n_B * d^2 epsilon / d n_B^2`
    - `v_s^2 = (dP/dn_B) / mu_B`
    """
    eps_smooth, mu_b_values, d2eps_values = local_polynomial_regression(
        densities,
        energy_density,
        smoothing_window,
        smoothing_degree,
    )

    pressure_values = [
        density * mu_b - eps
        for density, mu_b, eps in zip(densities, mu_b_values, eps_smooth)
    ]
    dP_dn_values = [
        density * d2eps
        for density, d2eps in zip(densities, d2eps_values)
    ]

    vs2_values: list[float] = []
    for mu_b, dP_dn, d2eps in zip(mu_b_values, dP_dn_values, d2eps_values):
        if (
            not math.isfinite(mu_b)
            or not math.isfinite(dP_dn)
            or not math.isfinite(d2eps)
            or abs(mu_b) < derivative_floor
        ):
            vs2_values.append(float("nan"))
            continue
        vs2_values.append(dP_dn / mu_b)

    return SoundSpeedCurve(
        energy_density_smoothed=eps_smooth,
        chemical_potential=mu_b_values,
        pressure=pressure_values,
        dP_dn=dP_dn_values,
        d2eps_dn2=d2eps_values,
        vs2=vs2_values,
    )


def reconstruct_sound_speed_curve_gradient_check(
    densities: list[float],
    energy_density: list[float],
    derivative_floor: float = 1.0e-12,
    export_path: str | Path | None = None,
    mev_fm3_factor: float = GEV4_TO_MEV_FM3,
) -> SoundSpeedCurve:
    """Diagnostic `v_s^2` reconstruction using the direct `numpy.gradient` recipe.

    This follows the check requested in the notebook workflow:

    - `mu_B = gradient(epsilon, n_B)`
    - `mu_B' = gradient(mu_B, n_B)`
    - `P = n_B * mu_B - epsilon`
    - `v_s^2 = (n_B / mu_B) * mu_B'`

    The routine intentionally works on the provided energy-density array
    directly, without the local-polynomial reconstruction used by the main
    workflow. It is therefore best used as a cross-check, not as the default
    production method.
    """
    import numpy as np

    density_array = np.asarray(densities, dtype=float)
    energy_array = np.asarray(energy_density, dtype=float)

    if density_array.ndim != 1 or energy_array.ndim != 1:
        raise ValueError("Sound-speed inputs must be one-dimensional.")
    if density_array.size != energy_array.size:
        raise ValueError("Density and energy-density arrays must have the same length.")
    if density_array.size < 3:
        raise ValueError("`numpy.gradient(..., edge_order=2)` requires at least three points.")

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
