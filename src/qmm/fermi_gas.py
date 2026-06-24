"""Ideal relativistic Fermi-gas relations."""

from __future__ import annotations

import math

from .constants import DEFAULT_GROUND_STATE_SETTINGS, DEFAULT_PHYSICAL_CONSTANTS, GroundStateSettings, PhysicalConstants


def energy_dispersion(k_value: float, physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS) -> float:
    """Single-particle relativistic dispersion relation."""
    return math.sqrt((physical.hbarc * k_value) ** 2 + physical.m_nucleon ** 2)


def kf_from_nid(n_id: float, degeneracy: float, physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS) -> float:
    """Return the Fermi momentum for an ideal density."""
    if n_id <= 0.0:
        return 0.0
    return (6.0 * math.pi ** 2 * n_id / degeneracy) ** (1.0 / 3.0)


def eps_id_from_kf(
    kf: float,
    degeneracy: float,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> float:
    """Return the ideal-gas energy density at `T=0`."""
    del settings
    if kf <= 0.0:
        return 0.0

    p_f = physical.hbarc * kf
    e_f = math.sqrt(p_f * p_f + physical.m_nucleon * physical.m_nucleon)
    log_term = math.log((p_f + e_f) / physical.m_nucleon)
    bracket = p_f * e_f * (2.0 * p_f * p_f + physical.m_nucleon * physical.m_nucleon)
    bracket -= physical.m_nucleon ** 4 * log_term
    return degeneracy * bracket / (16.0 * math.pi ** 2 * physical.hbarc ** 3)


def p_id_from_kf(
    kf: float,
    degeneracy: float,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> float:
    """Return the ideal-gas pressure at `T=0`."""
    del settings
    if kf <= 0.0:
        return 0.0

    p_f = physical.hbarc * kf
    e_f = math.sqrt(p_f * p_f + physical.m_nucleon * physical.m_nucleon)
    log_term = math.log((p_f + e_f) / physical.m_nucleon)
    bracket = p_f * e_f * (2.0 * p_f * p_f - 3.0 * physical.m_nucleon * physical.m_nucleon)
    bracket += 3.0 * physical.m_nucleon ** 4 * log_term
    return degeneracy * bracket / (48.0 * math.pi ** 2 * physical.hbarc ** 3)


def eps_id_from_nid(
    n_id: float,
    degeneracy: float,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> float:
    """Return the ideal-gas energy density from the ideal density."""
    return eps_id_from_kf(kf_from_nid(n_id, degeneracy, physical), degeneracy, physical, settings)


def p_id_from_nid(
    n_id: float,
    degeneracy: float,
    physical: PhysicalConstants = DEFAULT_PHYSICAL_CONSTANTS,
    settings: GroundStateSettings = DEFAULT_GROUND_STATE_SETTINGS,
) -> float:
    """Return the ideal-gas pressure from the ideal density."""
    return p_id_from_kf(kf_from_nid(n_id, degeneracy, physical), degeneracy, physical, settings)
