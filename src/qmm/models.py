"""Interaction models and their repulsive mappings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

from .constants import DEFAULT_CLAUSIUS_CS_RANGE, DEFAULT_CLAUSIUS_RANGE, DEFAULT_CLAUSIUS_TVM_RANGE, DEFAULT_DIETERICI_RANGE


ScalarModelFunction = Callable[[float, float, Optional[float]], Optional[float]]
DensityMap = Callable[[float, float], Optional[float]]
VolumeFractionMap = Callable[[float, float], Optional[float]]
SpeciesVolumeFractionMap = Callable[[float], Optional[float]]


def u_vdw(n: float, b: float, parameter: float | None = None) -> float:
    del b, parameter
    return -n


def du_vdw(n: float, b: float, parameter: float | None = None) -> float:
    del n, b, parameter
    return -1.0


def u_rks(n: float, b: float, parameter: float | None = None) -> float | None:
    del parameter
    if abs(b) < 1.0e-14:
        return -n
    return -(1.0 / b) * math.log(1.0 + b * n)


def du_rks(n: float, b: float, parameter: float | None = None) -> float:
    del parameter
    return -1.0 / (1.0 + b * n)


def u_pr(n: float, b: float, parameter: float | None = None) -> float | None:
    del parameter
    if abs(b) < 1.0e-14:
        return -n
    sqrt_two = math.sqrt(2.0)
    num = 1.0 + (1.0 + sqrt_two) * b * n
    den = 1.0 + (1.0 - sqrt_two) * b * n
    if num <= 0.0 or den <= 0.0:
        return None
    return -(1.0 / (2.0 * sqrt_two * b)) * math.log(num / den)


def du_pr(n: float, b: float, parameter: float | None = None) -> float:
    del parameter
    return -1.0 / (1.0 + 2.0 * b * n - (b * n) ** 2)


def u_clausius(n: float, b: float, parameter: float | None) -> float | None:
    del b
    if parameter is None:
        return None
    den = 1.0 + parameter * n
    if den <= 0.0:
        return None
    return -n / den


def du_clausius(n: float, b: float, parameter: float | None) -> float | None:
    del b
    if parameter is None:
        return None
    den = 1.0 + parameter * n
    if den <= 0.0:
        return None
    return -1.0 / (den * den)


def u_dieterici(n: float, b: float, parameter: float | None) -> float | None:
    del b
    if parameter is None or abs(parameter - 1.0) < 1.0e-14 or n <= 0.0:
        return None
    return -(n ** (parameter - 1.0)) / (parameter - 1.0)


def du_dieterici(n: float, b: float, parameter: float | None) -> float | None:
    del b
    if parameter is None or n <= 0.0:
        return None
    return -(n ** (parameter - 2.0))


def volume_fraction_ev(n: float, b: float) -> float | None:
    den = 1.0 - b * n
    if den <= 0.0:
        return None
    return den


def species_fraction_ev(x_value: float) -> float | None:
    den = 1.0 - x_value
    if den <= 0.0:
        return None
    return den


def nid_from_n_ev(n: float, b: float) -> float | None:
    fraction = volume_fraction_ev(n, b)
    if fraction is None:
        return None
    return n / fraction


def n_from_nid_ev(n_id: float, b: float) -> float | None:
    den = 1.0 + b * n_id
    if den <= 0.0:
        return None
    return n_id / den


def pressure_prefactor_ev(n: float, b: float) -> float | None:
    del n, b
    return 1.0


def _cs_exponent(x_value: float) -> float | None:
    den = 4.0 - x_value
    if den <= 0.0:
        return None
    return -(3.0 * x_value) / den - (4.0 * x_value) / (den * den)


def volume_fraction_cs(n: float, b: float) -> float | None:
    return species_fraction_cs(b * n)


def species_fraction_cs(x_value: float) -> float | None:
    exponent = _cs_exponent(x_value)
    if exponent is None:
        return None
    return math.exp(exponent)


def _volume_fraction_cs_derivative(n: float, b: float) -> float | None:
    x_value = b * n
    den = 4.0 - x_value
    if den <= 0.0:
        return None
    fraction = volume_fraction_cs(n, b)
    if fraction is None:
        return None
    dg_dx = -8.0 * (8.0 - x_value) / (den * den * den)
    return fraction * b * dg_dx


def nid_from_n_cs(n: float, b: float) -> float | None:
    fraction = volume_fraction_cs(n, b)
    if fraction is None or fraction <= 0.0:
        return None
    return n / fraction


def pressure_prefactor_cs(n: float, b: float) -> float | None:
    fraction = volume_fraction_cs(n, b)
    derivative = _volume_fraction_cs_derivative(n, b)
    if fraction is None or derivative is None:
        return None
    prefactor = fraction - n * derivative
    if prefactor <= 0.0:
        return None
    return prefactor


def volume_fraction_tvm(n: float, b: float) -> float:
    return species_fraction_tvm(b * n)


def species_fraction_tvm(x_value: float) -> float:
    return math.exp(-x_value - 0.5 * x_value * x_value)


def _volume_fraction_tvm_derivative(n: float, b: float) -> float:
    x_value = b * n
    fraction = volume_fraction_tvm(n, b)
    return -b * (1.0 + x_value) * fraction


def nid_from_n_tvm(n: float, b: float) -> float:
    return n / volume_fraction_tvm(n, b)


def pressure_prefactor_tvm(n: float, b: float) -> float:
    fraction = volume_fraction_tvm(n, b)
    derivative = _volume_fraction_tvm_derivative(n, b)
    return fraction - n * derivative


def n_from_nid_generic(
    n_id: float,
    b: float,
    volume_fraction: VolumeFractionMap,
    max_density: float | None = None,
) -> float | None:
    """Invert `n_id = n / f(n)` numerically."""
    if n_id < 0.0:
        return None
    if n_id == 0.0:
        return 0.0
    if b <= 0.0:
        return n_id

    def target(n_value: float) -> float | None:
        fraction = volume_fraction(n_value, b)
        if fraction is None or fraction <= 0.0:
            return None
        return n_value / fraction - n_id

    left = 0.0
    right = min(n_id, 0.5 * max_density) if max_density is not None else max(n_id, 1.0 / b)
    if right <= 0.0:
        right = 1.0 / b

    value_right = target(right)
    if value_right is None and max_density is not None:
        right = max_density * (1.0 - 1.0e-12)
        value_right = target(right)
    if value_right is None:
        return None

    attempts = 0
    while value_right < 0.0 and attempts < 200:
        if max_density is not None:
            next_right = 0.5 * (right + max_density)
            if next_right <= right:
                break
            right = next_right
        else:
            right *= 2.0
        value_right = target(right)
        if value_right is None:
            return None
        attempts += 1

    if value_right < 0.0:
        return None

    for _ in range(200):
        mid = 0.5 * (left + right)
        value_mid = target(mid)
        if value_mid is None:
            return None
        if abs(value_mid) <= 1.0e-12 * max(1.0, n_id):
            return mid
        if value_mid > 0.0:
            right = mid
        else:
            left = mid
    return 0.5 * (left + right)


def n_from_nid_cs(n_id: float, b: float) -> float | None:
    max_density = None if b <= 0.0 else 4.0 / b
    return n_from_nid_generic(n_id, b, volume_fraction_cs, max_density=max_density)


def n_from_nid_tvm(n_id: float, b: float) -> float | None:
    return n_from_nid_generic(n_id, b, volume_fraction_tvm)


@dataclass(frozen=True)
class InteractionModel:
    """Bundle the functions that define one interaction model.

    To add a new real-gas model, define the thermodynamic building blocks
    above and register one more `InteractionModel` in `MODELS`.

    The workflow expects:

    - an attractive contribution `U(n,b,parameter)` and `dU/dn`
    - a map from physical to ideal density and its inverse
    - total and species-level volume fractions
    - a pressure prefactor compatible with the excluded-volume rule
    - capability flags declaring which workflow branches are supported
    """

    name: str
    parameter_name: str | None
    parameter_range: tuple[float, float] | None
    supports_quantum: bool
    supports_symmetric_quarkyonic: bool
    supports_asymmetric_hadronic: bool
    supports_asymmetric_quarkyonic: bool
    U: ScalarModelFunction
    dU: ScalarModelFunction
    nid_from_n: DensityMap
    n_from_nid: DensityMap
    volume_fraction: VolumeFractionMap
    species_volume_fraction: SpeciesVolumeFractionMap
    pressure_prefactor: VolumeFractionMap


MODELS: dict[str, InteractionModel] = {
    "vdw": InteractionModel(
        name="vdw",
        parameter_name=None,
        parameter_range=None,
        supports_quantum=True,
        supports_symmetric_quarkyonic=True,
        supports_asymmetric_hadronic=True,
        supports_asymmetric_quarkyonic=True,
        U=u_vdw,
        dU=du_vdw,
        nid_from_n=nid_from_n_ev,
        n_from_nid=n_from_nid_ev,
        volume_fraction=volume_fraction_ev,
        species_volume_fraction=species_fraction_ev,
        pressure_prefactor=pressure_prefactor_ev,
    ),
    "rks": InteractionModel(
        name="rks",
        parameter_name=None,
        parameter_range=None,
        supports_quantum=True,
        supports_symmetric_quarkyonic=True,
        supports_asymmetric_hadronic=False,
        supports_asymmetric_quarkyonic=False,
        U=u_rks,
        dU=du_rks,
        nid_from_n=nid_from_n_ev,
        n_from_nid=n_from_nid_ev,
        volume_fraction=volume_fraction_ev,
        species_volume_fraction=species_fraction_ev,
        pressure_prefactor=pressure_prefactor_ev,
    ),
    "pr": InteractionModel(
        name="pr",
        parameter_name=None,
        parameter_range=None,
        supports_quantum=True,
        supports_symmetric_quarkyonic=True,
        supports_asymmetric_hadronic=False,
        supports_asymmetric_quarkyonic=False,
        U=u_pr,
        dU=du_pr,
        nid_from_n=nid_from_n_ev,
        n_from_nid=n_from_nid_ev,
        volume_fraction=volume_fraction_ev,
        species_volume_fraction=species_fraction_ev,
        pressure_prefactor=pressure_prefactor_ev,
    ),
    "clausius": InteractionModel(
        name="clausius",
        parameter_name="c",
        parameter_range=DEFAULT_CLAUSIUS_RANGE,
        supports_quantum=True,
        supports_symmetric_quarkyonic=True,
        supports_asymmetric_hadronic=True,
        supports_asymmetric_quarkyonic=True,
        U=u_clausius,
        dU=du_clausius,
        nid_from_n=nid_from_n_ev,
        n_from_nid=n_from_nid_ev,
        volume_fraction=volume_fraction_ev,
        species_volume_fraction=species_fraction_ev,
        pressure_prefactor=pressure_prefactor_ev,
    ),
    "clausius_cs": InteractionModel(
        name="clausius_cs",
        parameter_name="c",
        parameter_range=DEFAULT_CLAUSIUS_CS_RANGE,
        supports_quantum=True,
        supports_symmetric_quarkyonic=True,
        supports_asymmetric_hadronic=True,
        supports_asymmetric_quarkyonic=True,
        U=u_clausius,
        dU=du_clausius,
        nid_from_n=nid_from_n_cs,
        n_from_nid=n_from_nid_cs,
        volume_fraction=volume_fraction_cs,
        species_volume_fraction=species_fraction_cs,
        pressure_prefactor=pressure_prefactor_cs,
    ),
    "clausius_tvm": InteractionModel(
        name="clausius_tvm",
        parameter_name="c",
        parameter_range=DEFAULT_CLAUSIUS_TVM_RANGE,
        supports_quantum=True,
        supports_symmetric_quarkyonic=True,
        supports_asymmetric_hadronic=True,
        supports_asymmetric_quarkyonic=True,
        U=u_clausius,
        dU=du_clausius,
        nid_from_n=nid_from_n_tvm,
        n_from_nid=n_from_nid_tvm,
        volume_fraction=volume_fraction_tvm,
        species_volume_fraction=species_fraction_tvm,
        pressure_prefactor=pressure_prefactor_tvm,
    ),
    "dieterici": InteractionModel(
        name="dieterici",
        parameter_name="alpha",
        parameter_range=DEFAULT_DIETERICI_RANGE,
        supports_quantum=True,
        supports_symmetric_quarkyonic=True,
        supports_asymmetric_hadronic=True,
        supports_asymmetric_quarkyonic=True,
        U=u_dieterici,
        dU=du_dieterici,
        nid_from_n=nid_from_n_ev,
        n_from_nid=n_from_nid_ev,
        volume_fraction=volume_fraction_ev,
        species_volume_fraction=species_fraction_ev,
        pressure_prefactor=pressure_prefactor_ev,
    ),
    "cs": InteractionModel(
        name="cs",
        parameter_name=None,
        parameter_range=None,
        supports_quantum=True,
        supports_symmetric_quarkyonic=True,
        supports_asymmetric_hadronic=True,
        supports_asymmetric_quarkyonic=True,
        U=u_vdw,
        dU=du_vdw,
        nid_from_n=nid_from_n_cs,
        n_from_nid=n_from_nid_cs,
        volume_fraction=volume_fraction_cs,
        species_volume_fraction=species_fraction_cs,
        pressure_prefactor=pressure_prefactor_cs,
    ),
    "tvm": InteractionModel(
        name="tvm",
        parameter_name=None,
        parameter_range=None,
        supports_quantum=True,
        supports_symmetric_quarkyonic=True,
        supports_asymmetric_hadronic=True,
        supports_asymmetric_quarkyonic=True,
        U=u_vdw,
        dU=du_vdw,
        nid_from_n=nid_from_n_tvm,
        n_from_nid=n_from_nid_tvm,
        volume_fraction=volume_fraction_tvm,
        species_volume_fraction=species_fraction_tvm,
        pressure_prefactor=pressure_prefactor_tvm,
    ),
}


def get_model(model_name: str) -> InteractionModel:
    """Return a model by name with a clear error if it is missing."""
    if model_name not in MODELS:
        available = ", ".join(sorted(MODELS))
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}.")
    return MODELS[model_name]
