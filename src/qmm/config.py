"""Configuration loading for unified workflow runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_ASYMMETRIC_SETTINGS,
    DEFAULT_GROUND_STATE_SETTINGS,
    DEFAULT_HADRONIC_EOS_SETTINGS,
    DEFAULT_NEUTRON_STAR_SETTINGS,
    DEFAULT_PHYSICAL_CONSTANTS,
    DEFAULT_QUANTUM_SETTINGS,
    DEFAULT_QUARKYONIC_SETTINGS,
    AsymmetricFitSettings,
    GroundStateSettings,
    HadronicEOSSettings,
    NeutronStarSettings,
    OutputSettings,
    ParameterSearchSettings,
    PhysicalConstants,
    QuantumCriticalSettings,
    QuarkyonicSettings,
)


@dataclass(frozen=True)
class WorkflowSelection:
    """Booleans selecting the workflows to execute."""

    ground_state: bool = True
    quantum_critical: bool = False
    hadronic_eos_table: bool = False
    symmetric_quarkyonic: bool = False
    asymmetric_fit: bool = False
    asymmetric_quarkyonic: bool = False
    neutron_star: bool = False


@dataclass(frozen=True)
class RunConfig:
    """Complete workflow configuration."""

    run_name: str
    model_name: str
    parameter_value: float | None
    parameter_search: ParameterSearchSettings
    workflows: WorkflowSelection
    physical: PhysicalConstants
    ground_state: GroundStateSettings
    quantum: QuantumCriticalSettings
    hadronic_eos: HadronicEOSSettings
    quarkyonic: QuarkyonicSettings
    asymmetric: AsymmetricFitSettings
    neutron_star: NeutronStarSettings
    output: OutputSettings


def _merge_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    if not updates:
        return instance
    allowed = {field_name for field_name in instance.__dataclass_fields__}
    filtered = {key: value for key, value in updates.items() if key in allowed}
    return replace(instance, **filtered)


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run_config(path: str | Path) -> RunConfig:
    """Load a JSON configuration file into a strongly typed config.

    The loader reads the executable workflow blocks and ignores extra
    descriptive keys. This allows one JSON file to carry both runnable
    settings and human-readable notes such as LaTeX equations for the mean
    field or excluded-volume prescription.
    """
    data = _load_json(path)
    model_data = data.get("model", {})
    workflow_data = data.get("workflows", {})

    if "name" not in model_data:
        raise ValueError("The configuration must provide `model.name`.")

    parameter_search = _merge_dataclass(
        ParameterSearchSettings(),
        model_data.get("parameter_search", {}),
    )

    physical = _merge_dataclass(DEFAULT_PHYSICAL_CONSTANTS, data.get("physical", {}))
    ground_state = _merge_dataclass(DEFAULT_GROUND_STATE_SETTINGS, data.get("ground_state", {}))

    quantum_data = dict(data.get("quantum", {}))
    quantum_quick = bool(quantum_data.pop("quick", False))
    quantum_base = QuantumCriticalSettings.quick() if quantum_quick else DEFAULT_QUANTUM_SETTINGS
    quantum = _merge_dataclass(quantum_base, quantum_data)
    hadronic_eos = _merge_dataclass(DEFAULT_HADRONIC_EOS_SETTINGS, data.get("hadronic_eos", {}))

    quarkyonic_data = dict(data.get("quarkyonic", {}))
    quarkyonic_quick = bool(quarkyonic_data.pop("quick", False))
    quarkyonic_base = QuarkyonicSettings.quick() if quarkyonic_quick else DEFAULT_QUARKYONIC_SETTINGS
    quarkyonic = _merge_dataclass(quarkyonic_base, quarkyonic_data)

    asymmetric = _merge_dataclass(DEFAULT_ASYMMETRIC_SETTINGS, data.get("asymmetric", {}))
    neutron_star = _merge_dataclass(DEFAULT_NEUTRON_STAR_SETTINGS, data.get("neutron_star", {}))
    output = _merge_dataclass(OutputSettings(), data.get("output", {}))
    workflows = _merge_dataclass(WorkflowSelection(), workflow_data)

    return RunConfig(
        run_name=data.get("run_name", "qmm_run"),
        model_name=str(model_data["name"]),
        parameter_value=model_data.get("parameter_value"),
        parameter_search=parameter_search,
        workflows=workflows,
        physical=physical,
        ground_state=ground_state,
        quantum=quantum,
        hadronic_eos=hadronic_eos,
        quarkyonic=quarkyonic,
        asymmetric=asymmetric,
        neutron_star=neutron_star,
        output=output,
    )
