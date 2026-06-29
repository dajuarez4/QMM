"""Helpers for building and running QMM JSON configs from a notebook.

This module stays stdlib-only so the notebook can run even if the current
kernel is not the same Python environment that will execute QMM.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


PRESET_DESCRIPTIONS: Dict[str, str] = {
    "ground_state_only": "Fit symmetric nuclear matter at saturation only.",
    "critical_point": "Ground state plus the finite-temperature critical point.",
    "symmetric_quarkyonic": "Ground state plus symmetric quarkyonic matter.",
    "full_symmetric": "Ground state, critical point, hadronic EOS, and symmetric quarkyonic matter.",
    "asymmetric_fixed_y": "Asymmetric fixed-y quarkyonic matter without beta equilibrium.",
    "asymmetric_beta": "Asymmetric quarkyonic matter in beta equilibrium.",
    "asymmetric_beta_neutron_star": "Asymmetric beta-equilibrium matter plus a neutron-star sequence.",
    "baryquark_symmetric": "Symmetric baryquark matter demo.",
    "custom": "Choose each workflow block manually.",
}


def _normalize_branch_mode(branch_mode: str) -> str:
    normalized = str(branch_mode).strip().lower()
    if normalized in {"target_l", "unequal_b", "split_b"}:
        return "target_l"
    if normalized in {"equal_b", "eq", "delta_b_zero"}:
        return "equal_b"
    if normalized == "both":
        raise ValueError("`branch_mode='both'` requires `build_configs()`, not `build_config()`.")
    raise ValueError("`branch_mode` must be one of: target_l or equal_b.")


def _normalize_branch_modes(
    branch_mode: Any,
    branch_modes: Any = None,
) -> List[str]:
    if branch_modes is None:
        raw_value = "target_l" if branch_mode is None else branch_mode
        if str(raw_value).strip().lower() == "both":
            items = ["target_l", "equal_b"]
        else:
            items = [raw_value]
    elif isinstance(branch_modes, (list, tuple, set)):
        items = list(branch_modes)
    else:
        items = [branch_modes]

    normalized: List[str] = []
    for item in items:
        branch = _normalize_branch_mode(str(item))
        if branch not in normalized:
            normalized.append(branch)
    return normalized or ["target_l"]


def _needs_asymmetric_branch(request: Mapping[str, Any]) -> bool:
    workflow = _workflow_from_preset(
        str(request.get("preset", default_request()["preset"])).strip(),
        request.get("custom_workflows"),
    )
    return bool(
        workflow["asymmetric_fit"]
        or workflow["asymmetric_quarkyonic"]
        or workflow["neutron_star"]
    )


def expand_branch_requests(request: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Expand one notebook request into one or more branch-specific requests."""
    normalized = _normalize_request(request)
    if not _needs_asymmetric_branch(normalized):
        normalized["branch_mode"] = _normalize_branch_modes(
            normalized.get("branch_mode", "target_l"),
            normalized.get("branch_modes"),
        )[0]
        normalized.pop("branch_modes", None)
        return [normalized]

    branch_modes = _normalize_branch_modes(
        normalized.get("branch_mode", "target_l"),
        normalized.get("branch_modes"),
    )
    if len(branch_modes) == 1:
        normalized["branch_mode"] = branch_modes[0]
        normalized.pop("branch_modes", None)
        return [normalized]

    variants: List[Dict[str, Any]] = []
    base_run_name = normalized["run_name"]
    base_output_subdir = normalized.get("output_subdir")
    for branch_mode in branch_modes:
        variant = dict(normalized)
        suffix = f"_{branch_mode}"
        variant["branch_mode"] = branch_mode
        variant.pop("branch_modes", None)
        if not str(base_run_name).endswith(suffix):
            variant["run_name"] = f"{base_run_name}{suffix}"
        if base_output_subdir:
            variant["output_subdir"] = (
                str(base_output_subdir)
                if str(base_output_subdir).endswith(suffix)
                else f"{base_output_subdir}{suffix}"
            )
        variants.append(variant)
    return variants


def locate_qmm_root(start: Optional[Path] = None) -> Path:
    """Locate the `QMM` project root from the notebook directory."""
    current = (start or Path.cwd()).resolve()
    candidates = [current] + list(current.parents)
    for candidate in candidates:
        if (candidate / "src" / "qmm").exists() and (candidate / "examples").exists():
            return candidate
    raise RuntimeError("Could not locate the QMM project root.")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_blank_template(root: Path) -> Dict[str, Any]:
    return load_json(root / "examples" / "qmm_blank_template.json")


def _run_python_json(root: Path, python_bin: str, code: str) -> Dict[str, Any]:
    env = _qmm_env(root)
    completed = subprocess.run(
        [python_bin, "-c", code],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _python_candidates(root: Path) -> Iterable[str]:
    local_venv = root.parent / ".venv" / "bin" / "python"
    seen = set()
    candidates = [
        sys.executable,
        str(local_venv) if local_venv.exists() else None,
        "/opt/homebrew/bin/python3.12",
        "python3.12",
        "python3",
    ]
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            yield candidate


def python_details(root: Path, python_bin: str) -> Dict[str, Any]:
    probe = (
        "import json, sys\n"
        "payload = {'ok': False, 'version': list(sys.version_info[:3]), 'has_matplotlib': False}\n"
        "if sys.version_info >= (3, 11):\n"
        "    payload['ok'] = True\n"
        "    try:\n"
        "        import matplotlib\n"
        "        payload['has_matplotlib'] = True\n"
        "    except Exception:\n"
        "        pass\n"
        "else:\n"
        "    payload['error'] = 'Python 3.11+ is required.'\n"
        "print(json.dumps(payload))\n"
    )
    try:
        return _run_python_json(root, python_bin, probe)
    except Exception:
        return {"ok": False, "version": None, "has_matplotlib": False}


def choose_python(root: Path) -> str:
    """Return a Python 3.11+ interpreter that can execute QMM."""
    for candidate in _python_candidates(root):
        if python_details(root, candidate).get("ok"):
            return candidate
    raise RuntimeError(
        "Could not find a Python 3.11+ interpreter for QMM."
    )


def _qmm_env(root: Path) -> Dict[str, str]:
    env = dict(os.environ)
    src_path = str(root / "src")
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = src_path + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = src_path
    return env


def available_models(root: Path, python_bin: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Read registered model metadata using a QMM-compatible interpreter."""
    interpreter = python_bin or choose_python(root)
    code = (
        "import json\n"
        "from qmm.models import MODELS\n"
        "payload = {}\n"
        "for name, model in MODELS.items():\n"
        "    payload[name] = {\n"
        "        'parameter_name': model.parameter_name,\n"
        "        'parameter_range': list(model.parameter_range) if model.parameter_range is not None else None,\n"
        "        'supports_quantum': model.supports_quantum,\n"
        "        'supports_symmetric_quarkyonic': model.supports_symmetric_quarkyonic,\n"
        "        'supports_asymmetric_hadronic': model.supports_asymmetric_hadronic,\n"
        "        'supports_asymmetric_quarkyonic': model.supports_asymmetric_quarkyonic,\n"
        "    }\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )
    payload = _run_python_json(root, interpreter, code)
    for name, metadata in payload.items():
        mean_field, excluded_volume = split_model_name(name)
        metadata["mean_field"] = mean_field
        metadata["excluded_volume"] = excluded_volume
    return payload


def split_model_name(model_name: str) -> tuple[str, str]:
    normalized = str(model_name).strip().lower()
    if normalized == "cs":
        return "vdw", "cs"
    if normalized == "tvm":
        return "vdw", "tvm"
    if normalized.endswith("_cs"):
        return normalized[:-3], "cs"
    if normalized.endswith("_tvm"):
        return normalized[:-4], "tvm"
    return normalized, "vdw"


def available_model_families(root: Path, python_bin: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    families: Dict[str, Dict[str, Any]] = {}
    for model_name, metadata in available_models(root, python_bin).items():
        mean_field = str(metadata["mean_field"])
        excluded_volume = str(metadata["excluded_volume"])
        family = families.setdefault(
            mean_field,
            {
                "excluded_volumes": [],
                "models": {},
                "parameter_name": metadata["parameter_name"],
                "parameter_range": metadata["parameter_range"],
            },
        )
        if excluded_volume not in family["excluded_volumes"]:
            family["excluded_volumes"].append(excluded_volume)
        family["models"][excluded_volume] = model_name
    order = {"vdw": 0, "cs": 1, "tvm": 2}
    for family in families.values():
        family["excluded_volumes"].sort(key=lambda value: order.get(value, 999))
    return dict(sorted(families.items()))


def resolve_model_name(
    mean_field: str,
    excluded_volume: str,
    models: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> str:
    normalized_mean_field = mean_field.strip().lower()
    normalized_excluded = excluded_volume.strip().lower()
    if normalized_excluded not in {"vdw", "cs", "tvm"}:
        raise ValueError("`excluded_volume` must be one of: vdw, cs, tvm.")

    if normalized_mean_field == "vdw":
        candidate = {"vdw": "vdw", "cs": "cs", "tvm": "tvm"}[normalized_excluded]
    elif normalized_excluded == "vdw":
        candidate = normalized_mean_field
    else:
        candidate = f"{normalized_mean_field}_{normalized_excluded}"

    if models is not None and candidate not in models:
        family_choices = sorted(
            metadata["excluded_volume"]
            for metadata in models.values()
            if metadata.get("mean_field") == normalized_mean_field
        )
        if family_choices:
            raise ValueError(
                f"Unsupported combination mean_field='{mean_field}', excluded_volume='{excluded_volume}'. "
                f"Available excluded-volume choices for '{mean_field}' are: {', '.join(family_choices)}."
            )
        raise ValueError(f"Unknown mean field '{mean_field}'.")
    return candidate


def preset_names() -> List[str]:
    return list(PRESET_DESCRIPTIONS)


def default_request() -> Dict[str, Any]:
    return {
        "run_name": "my_qmm_run",
        "mean_field": "vdw",
        "excluded_volume": "cs",
        "model_name": "cs",
        "preset": "full_symmetric",
        "parameter_mode": "search",
        "parameter_value": None,
        "target_k0": 280.0,
        "quick_mode": True,
        "momentum_mode": "quarkyonic",
        "lambda_momentum_mev": 300.0,
        "branch_mode": "target_l",
        "beta_equilibrium": True,
        "proton_fraction_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "write_json": True,
        "write_csv": True,
        "write_plots": False,
        "plot_formats": ["png"],
        "output_subdir": None,
        "custom_workflows": None,
    }


def parse_yes_no(raw: str, default: bool) -> bool:
    text = raw.strip().lower()
    if not text:
        return default
    if text in {"y", "yes", "true", "1"}:
        return True
    if text in {"n", "no", "false", "0"}:
        return False
    raise ValueError("Expected yes or no.")


def parse_float_list(raw: str, default: Iterable[float]) -> List[float]:
    text = raw.strip()
    if not text:
        return [float(value) for value in default]
    values = []
    for item in text.split(","):
        stripped = item.strip()
        if stripped:
            values.append(float(stripped))
    if not values:
        raise ValueError("At least one numeric value is required.")
    return values


def _workflow_from_preset(preset: str, custom_workflows: Optional[Mapping[str, Any]]) -> Dict[str, bool]:
    if preset == "custom":
        if not custom_workflows:
            raise ValueError("`custom_workflows` is required when preset='custom'.")
        workflow = {
            "ground_state": bool(custom_workflows.get("ground_state", False)),
            "quantum_critical": bool(custom_workflows.get("quantum_critical", False)),
            "hadronic_eos_table": bool(custom_workflows.get("hadronic_eos_table", False)),
            "symmetric_quarkyonic": bool(custom_workflows.get("symmetric_quarkyonic", False)),
            "asymmetric_fit": bool(custom_workflows.get("asymmetric_fit", False)),
            "asymmetric_quarkyonic": bool(custom_workflows.get("asymmetric_quarkyonic", False)),
            "neutron_star": bool(custom_workflows.get("neutron_star", False)),
        }
    else:
        workflow = {
            "ground_state": False,
            "quantum_critical": False,
            "hadronic_eos_table": False,
            "symmetric_quarkyonic": False,
            "asymmetric_fit": False,
            "asymmetric_quarkyonic": False,
            "neutron_star": False,
        }
        if preset == "ground_state_only":
            workflow["ground_state"] = True
        elif preset == "critical_point":
            workflow["ground_state"] = True
            workflow["quantum_critical"] = True
        elif preset == "symmetric_quarkyonic":
            workflow["ground_state"] = True
            workflow["symmetric_quarkyonic"] = True
        elif preset == "full_symmetric":
            workflow["ground_state"] = True
            workflow["quantum_critical"] = True
            workflow["hadronic_eos_table"] = True
            workflow["symmetric_quarkyonic"] = True
        elif preset == "asymmetric_fixed_y":
            workflow["ground_state"] = True
            workflow["hadronic_eos_table"] = True
            workflow["asymmetric_fit"] = True
            workflow["asymmetric_quarkyonic"] = True
        elif preset == "asymmetric_beta":
            workflow["ground_state"] = True
            workflow["hadronic_eos_table"] = True
            workflow["asymmetric_fit"] = True
            workflow["asymmetric_quarkyonic"] = True
        elif preset == "asymmetric_beta_neutron_star":
            workflow["ground_state"] = True
            workflow["hadronic_eos_table"] = True
            workflow["asymmetric_fit"] = True
            workflow["asymmetric_quarkyonic"] = True
            workflow["neutron_star"] = True
        elif preset == "baryquark_symmetric":
            workflow["ground_state"] = True
            workflow["symmetric_quarkyonic"] = True
        else:
            raise ValueError(f"Unsupported preset '{preset}'.")

    if workflow["neutron_star"]:
        workflow["asymmetric_fit"] = True
        workflow["asymmetric_quarkyonic"] = True
    if workflow["asymmetric_quarkyonic"]:
        workflow["asymmetric_fit"] = True
    if any(workflow.values()):
        workflow["ground_state"] = True
    return workflow


def _default_momentum_mode(preset: str, requested_mode: Optional[str]) -> str:
    mode = (requested_mode or "").strip().lower()
    if not mode:
        return "baryquark" if preset == "baryquark_symmetric" else "quarkyonic"
    if mode not in {"quarkyonic", "baryquark"}:
        raise ValueError("`momentum_mode` must be 'quarkyonic' or 'baryquark'.")
    return mode


def _default_lambda(workflow: Mapping[str, bool], raw_lambda: Any) -> float:
    if raw_lambda is not None:
        return float(raw_lambda)
    if workflow["asymmetric_quarkyonic"] or workflow["neutron_star"]:
        return 200.0
    return 300.0


def _apply_quick_mode_overrides(config: Dict[str, Any]) -> None:
    """Apply reduced numerical settings for notebook smoke tests."""
    config["ground_state"]["b_scan_steps"] = 1600
    config["ground_state"]["n_integral_points"] = 1500
    config["quantum"]["quick"] = True
    config["quantum"]["n_k_fd"] = 220
    config["quantum"]["coarse_t_count"] = 11
    config["quantum"]["coarse_n_count"] = 11
    config["quantum"]["outer_max_iter"] = 24
    config["quarkyonic"]["quick"] = True
    config["quarkyonic"]["n_points"] = 60
    config["quarkyonic"]["fq_scan_points"] = 61
    config["quarkyonic"]["shell_integral_points"] = 200
    config["quarkyonic"]["quark_integral_points"] = 200
    config["hadronic_eos"]["n_points"] = 80
    config["neutron_star"]["sequence_points"] = 24


def _normalize_request(request: Mapping[str, Any]) -> Dict[str, Any]:
    merged = default_request()
    merged.update(dict(request))
    merged["run_name"] = str(merged["run_name"]).strip()
    merged["model_name"] = str(merged.get("model_name", "")).strip()
    merged["mean_field"] = str(merged.get("mean_field", "")).strip().lower()
    merged["excluded_volume"] = str(merged.get("excluded_volume", "")).strip().lower()
    merged["preset"] = str(merged["preset"]).strip()
    if not merged["run_name"]:
        raise ValueError("`run_name` cannot be empty.")
    if not merged["model_name"] and not (merged["mean_field"] and merged["excluded_volume"]):
        raise ValueError("Provide `model_name` or both `mean_field` and `excluded_volume`.")
    return merged


def build_config(
    root: Path,
    request: Mapping[str, Any],
    python_bin: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a runnable QMM config from a simple notebook request."""
    has_explicit_components = "mean_field" in request or "excluded_volume" in request
    normalized = _normalize_request(request)
    interpreter = python_bin or choose_python(root)
    models = available_models(root, interpreter)
    if has_explicit_components and normalized["mean_field"] and normalized["excluded_volume"]:
        normalized["model_name"] = resolve_model_name(
            normalized["mean_field"],
            normalized["excluded_volume"],
            models=models,
        )
    elif normalized["model_name"]:
        mean_field, excluded_volume = split_model_name(normalized["model_name"])
        normalized["mean_field"] = mean_field
        normalized["excluded_volume"] = excluded_volume
    if normalized["model_name"] not in models:
        raise ValueError(
            f"Unknown model '{normalized['model_name']}'. Choose one of: {', '.join(sorted(models))}."
        )

    model_info = models[normalized["model_name"]]
    preset = normalized["preset"]
    if preset not in PRESET_DESCRIPTIONS:
        raise ValueError(
            f"Unknown preset '{preset}'. Choose one of: {', '.join(PRESET_DESCRIPTIONS)}."
        )

    workflow = _workflow_from_preset(preset, normalized.get("custom_workflows"))
    momentum_mode = _default_momentum_mode(preset, normalized.get("momentum_mode"))

    if workflow["quantum_critical"] and not model_info["supports_quantum"]:
        raise ValueError(f"Model '{normalized['model_name']}' does not support the quantum critical workflow.")
    if workflow["symmetric_quarkyonic"] and not model_info["supports_symmetric_quarkyonic"]:
        raise ValueError(
            f"Model '{normalized['model_name']}' does not support the symmetric quarkyonic workflow."
        )
    if workflow["asymmetric_fit"] and not model_info["supports_asymmetric_hadronic"]:
        raise ValueError(
            f"Model '{normalized['model_name']}' does not support the asymmetric hadronic workflow."
        )
    if workflow["asymmetric_quarkyonic"] and not model_info["supports_asymmetric_quarkyonic"]:
        raise ValueError(
            f"Model '{normalized['model_name']}' does not support the asymmetric quarkyonic workflow."
        )
    if momentum_mode == "baryquark" and (workflow["asymmetric_quarkyonic"] or workflow["neutron_star"]):
        raise ValueError(
            "The current baryquark implementation only supports the symmetric workflow. "
            "Use momentum_mode='quarkyonic' for asymmetric or neutron-star runs."
        )

    config = copy.deepcopy(load_blank_template(root))
    config["run_name"] = normalized["run_name"]
    config["model"]["name"] = normalized["model_name"]
    config["model"]["mean_field"] = normalized["mean_field"]
    config["model"]["excluded_volume"] = normalized["excluded_volume"]

    parameter_name = model_info["parameter_name"]
    parameter_mode = str(normalized.get("parameter_mode", "search")).strip().lower()
    if parameter_name is None:
        config["model"]["parameter_value"] = None
        config["model"]["parameter_search"]["enabled"] = False
        config["model"]["parameter_search"]["target_k0"] = None
        config["model"]["parameter_search"]["parameter_min"] = None
        config["model"]["parameter_search"]["parameter_max"] = None
    elif parameter_mode == "value":
        parameter_value = normalized.get("parameter_value")
        if parameter_value is None:
            raise ValueError(
                f"Model '{normalized['model_name']}' requires {parameter_name}. "
                "Provide `parameter_value` or choose parameter_mode='search'."
            )
        config["model"]["parameter_value"] = float(parameter_value)
        config["model"]["parameter_search"]["enabled"] = False
        config["model"]["parameter_search"]["target_k0"] = None
    else:
        parameter_range = model_info["parameter_range"]
        target_k0 = normalized.get("target_k0")
        if target_k0 is None:
            target_k0 = 280.0
        config["model"]["parameter_value"] = None
        config["model"]["parameter_search"]["enabled"] = True
        config["model"]["parameter_search"]["target_k0"] = float(target_k0)
        config["model"]["parameter_search"]["parameter_min"] = parameter_range[0]
        config["model"]["parameter_search"]["parameter_max"] = parameter_range[1]

    config["workflows"] = workflow

    config["quarkyonic"]["momentum_mode"] = momentum_mode
    config["quarkyonic"]["lambda_momentum_mev"] = _default_lambda(
        workflow,
        normalized.get("lambda_momentum_mev"),
    )

    needs_asymmetric = workflow["asymmetric_fit"] or workflow["asymmetric_quarkyonic"] or workflow["neutron_star"]
    beta_equilibrium = bool(normalized.get("beta_equilibrium", preset != "asymmetric_fixed_y"))
    if workflow["neutron_star"]:
        beta_equilibrium = True
    if needs_asymmetric:
        config["asymmetric"]["enabled"] = True
        config["asymmetric"]["branch_mode"] = _normalize_branch_mode(
            str(normalized.get("branch_mode", "target_l"))
        )
        config["asymmetric"]["target_k0"] = config["model"]["parameter_search"]["target_k0"]
        config["asymmetric"]["beta_equilibrium"] = beta_equilibrium
        config["asymmetric"]["proton_fraction_values"] = [
            float(value) for value in normalized.get("proton_fraction_values", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ]
    else:
        config["asymmetric"]["enabled"] = False
        config["asymmetric"]["beta_equilibrium"] = False

    write_json = bool(normalized.get("write_json", True))
    write_csv = bool(normalized.get("write_csv", True))
    write_plots = bool(normalized.get("write_plots", True))
    plot_formats = [str(value).strip() for value in normalized.get("plot_formats", ["png"]) if str(value).strip()]
    if write_plots and not python_details(root, interpreter).get("has_matplotlib"):
        raise ValueError(
            f"The selected interpreter '{interpreter}' does not have matplotlib installed. "
            "Set write_plots=False or install matplotlib in that Python environment."
        )
    output_subdir = normalized.get("output_subdir") or normalized["run_name"]
    config["output"]["directory"] = f"results/generated/{output_subdir}"
    config["output"]["write_json"] = write_json
    config["output"]["write_csv"] = write_csv
    config["output"]["write_plots"] = write_plots
    config["output"]["plot_formats"] = plot_formats if write_plots else []

    if bool(normalized.get("quick_mode", False)):
        _apply_quick_mode_overrides(config)

    return config


def build_configs(
    root: Path,
    request: Mapping[str, Any],
    python_bin: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build one or more runnable configs from a notebook request."""
    return [
        build_config(root, expanded_request, python_bin)
        for expanded_request in expand_branch_requests(request)
    ]


def save_config(root: Path, config: Mapping[str, Any], filename: Optional[str] = None) -> Path:
    """Write a config into `examples/generated/`."""
    target_dir = root / "examples" / "generated"
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = filename or str(config["run_name"])
    if not stem.endswith(".json"):
        stem = f"{stem}.json"
    target_path = target_dir / stem
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return target_path


def save_configs(root: Path, configs: Iterable[Mapping[str, Any]]) -> List[Path]:
    """Write one or more configs into `examples/generated/`."""
    return [save_config(root, config) for config in configs]


def run_config(root: Path, config_path: Path, python_bin: Optional[str] = None) -> Dict[str, Any]:
    """Execute QMM for one saved config and return the parsed summary."""
    interpreter = python_bin or choose_python(root)
    config_arg = str(config_path.resolve().relative_to(root.resolve()))
    command = [interpreter, "-m", "qmm", config_arg]
    completed = subprocess.run(
        command,
        cwd=root,
        env=_qmm_env(root),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "QMM run failed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
    summary = json.loads(completed.stdout)
    summary_path = summary.get("outputs", {}).get("summary_json")
    return {
        "command": " ".join(command),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "summary": summary,
        "summary_path": Path(summary_path) if summary_path else None,
    }


def run_configs(
    root: Path,
    config_paths: Iterable[Path],
    python_bin: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Execute QMM for one or more saved configs."""
    return [run_config(root, config_path, python_bin) for config_path in config_paths]


def summarize_run(summary: Mapping[str, Any]) -> str:
    outputs = summary.get("outputs", {})
    lines = [
        f"run_name: {summary.get('run_name')}",
        f"model: {summary.get('model', {}).get('name')}",
        f"summary_json: {outputs.get('summary_json', 'not written')}",
    ]
    for key in sorted(outputs):
        if key != "summary_json":
            lines.append(f"{key}: {outputs[key]}")
    return "\n".join(lines)
