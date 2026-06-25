"""Compute the vdW excluded-volume-only symmetric control curve.

This helper uses the vdW repulsive mapping and the vdW ground-state `b`
parameter, but removes the attractive mean-field contribution from the
quarkyonic/baryquark energy functional. It is meant for notebook-level
comparison plots, not as a registered production model.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from qmm.config import load_run_config
from qmm.fermi_gas import eps_id_from_kf
from qmm.ground_state import compute_ground_state_point
from qmm.models import get_model
from qmm.numerics import golden_section_min, linspace
from qmm.quarkyonic import (
    _kbu_from_quark_density,
    _lower_quark_fraction_bound,
    _nucleon_energy_shell,
    _quark_density_from_kbu,
    _quark_energy_density_between_momenta,
    _quark_energy_density_from_kbu,
)
from qmm.sound_speed import (
    reconstruct_sound_speed_curve,
    reconstruct_sound_speed_curve_gradient_check,
)


@dataclass(frozen=True)
class EVOnlyState:
    n_b: float
    n_over_n0: float
    quark_fraction: float
    n_q: float
    n_n: float
    k_bu: float
    k_f: float
    energy_density: float


def ev_only_state_at_fraction(ground_state, n_b, fq, settings, physical):
    model = get_model("vdw")
    if fq < 0.0 or fq > 1.0:
        return None

    n_q = n_b * fq
    n_n = n_b - n_q
    if n_n < 0.0:
        return None

    if settings.momentum_mode == "quarkyonic":
        k_bu = _kbu_from_quark_density(n_q, settings, physical)
        if n_n <= 1.0e-14:
            n_n_id = 0.0
            k_f = k_bu
            shell_energy = 0.0
        else:
            n_n_id = model.nid_from_n(n_n, ground_state.b)
            if n_n_id is None:
                return None
            k_f = (k_bu ** 3 + (6.0 * math.pi ** 2 * n_n_id) / physical.degeneracy_symmetric) ** (1.0 / 3.0)
            shell_energy = _nucleon_energy_shell(k_bu, k_f, settings.shell_integral_points, physical)
            volume_fraction = n_n / n_n_id if n_n_id > 0.0 else 1.0
            shell_energy *= volume_fraction
        quark_energy = _quark_energy_density_from_kbu(k_bu, settings, physical)
        total_energy = shell_energy + quark_energy
    else:
        if n_n <= 1.0e-14:
            n_n_id = 0.0
            k_f = 0.0
            baryon_energy = 0.0
        else:
            n_n_id = model.nid_from_n(n_n, ground_state.b)
            if n_n_id is None:
                return None
            k_f = ((6.0 * math.pi ** 2 * n_n_id) / physical.degeneracy_symmetric) ** (1.0 / 3.0)
            volume_fraction = n_n / n_n_id if n_n_id > 0.0 else 1.0
            baryon_energy = volume_fraction * eps_id_from_kf(k_f, physical.degeneracy_symmetric, physical)
        inner_quark_density = _quark_density_from_kbu(k_f, settings, physical)
        k_bu = _kbu_from_quark_density(n_q + inner_quark_density, settings, physical)
        quark_energy = _quark_energy_density_between_momenta(k_f, k_bu, settings, physical)
        total_energy = baryon_energy + quark_energy

    if not math.isfinite(total_energy):
        return None

    return EVOnlyState(
        n_b=n_b,
        n_over_n0=n_b / physical.n0,
        quark_fraction=fq,
        n_q=n_q,
        n_n=n_n,
        k_bu=k_bu,
        k_f=k_f,
        energy_density=total_energy,
    )


def minimize_ev_only_state(ground_state, n_b, settings, physical):
    lower = _lower_quark_fraction_bound("vdw", n_b, ground_state.b, settings)
    if lower >= 1.0:
        return None

    fq_grid = linspace(lower, 1.0, settings.fq_scan_points)
    trial_states = [ev_only_state_at_fraction(ground_state, n_b, fq, settings, physical) for fq in fq_grid]
    finite_trials = [(idx, state) for idx, state in enumerate(trial_states) if state is not None]
    if not finite_trials:
        return None

    best_state = min((state for _, state in finite_trials), key=lambda item: item.energy_density)
    finite_energies = [state.energy_density if state is not None else float("inf") for state in trial_states]
    candidate_intervals = []

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

    def objective(fq_value):
        state = ev_only_state_at_fraction(ground_state, n_b, fq_value, settings, physical)
        if state is None:
            return float("inf")
        return state.energy_density

    for left, right in candidate_intervals:
        fq_star, _ = golden_section_min(objective, left, right, tol=settings.refine_tol, max_iter=settings.refine_max_iter)
        refined = ev_only_state_at_fraction(ground_state, n_b, fq_star, settings, physical)
        if refined is not None and refined.energy_density < best_state.energy_density:
            best_state = refined
    return best_state


def compute_curve(example_path: Path):
    config = load_run_config(example_path)
    physical = config.physical
    settings = config.quarkyonic
    ground_state = compute_ground_state_point(
        "vdw",
        config.parameter_value,
        config.parameter_search,
        physical,
        config.ground_state,
    )

    states = []
    for ratio in linspace(settings.n_min_ratio, settings.n_max_ratio, settings.n_points):
        n_b = ratio * physical.n0
        state = minimize_ev_only_state(ground_state, n_b, settings, physical)
        if state is not None:
            states.append(state)

    if len(states) < 5:
        raise ValueError("Excluded-volume-only control curve failed.")

    n_values = [state.n_b for state in states]
    n_over_n0 = [state.n_over_n0 for state in states]
    eps_raw = [state.energy_density for state in states]
    fq_values = [state.quark_fraction for state in states]
    n_q_values = [state.n_q for state in states]
    n_n_values = [state.n_n for state in states]
    k_bu_values = [state.k_bu for state in states]
    k_f_values = [state.k_f for state in states]

    smooth = reconstruct_sound_speed_curve(
        n_values,
        eps_raw,
        settings.smoothing_window,
        settings.smoothing_degree,
        settings.derivative_floor,
    )
    raw = reconstruct_sound_speed_curve_gradient_check(
        n_values,
        eps_raw,
        settings.derivative_floor,
    )

    vs2_plot = raw.vs2 if settings.momentum_mode == "baryquark" else smooth.vs2

    rows = []
    for idx in range(len(n_values)):
        rows.append(
            {
                "model": "vdw_excluded_volume_only",
                "momentum_mode": settings.momentum_mode,
                "a": ground_state.a,
                "b": ground_state.b,
                "K0": ground_state.K0,
                "n": n_values[idx],
                "n_over_n0": n_over_n0[idx],
                "eps_raw": eps_raw[idx],
                "eps": smooth.energy_density_smoothed[idx],
                "mu_b": smooth.chemical_potential[idx],
                "pressure": smooth.pressure[idx],
                "dP_dn": smooth.dP_dn[idx],
                "d2eps_dn2": smooth.d2eps_dn2[idx],
                "vs2": smooth.vs2[idx],
                "vs2_raw": raw.vs2[idx],
                "vs2_plot": vs2_plot[idx],
                "quark_fraction": fq_values[idx],
                "n_q": n_q_values[idx],
                "n_n": n_n_values[idx],
                "k_bu": k_bu_values[idx],
                "k_f": k_f_values[idx],
            }
        )

    finite_candidates = [
        (idx, value)
        for idx, value in enumerate(vs2_plot)
        if isinstance(value, float) and math.isfinite(value)
    ]
    peak_index, peak_value = max(finite_candidates, key=lambda item: item[1])
    summary = {
        "source_example": str(example_path),
        "model": "vdw_excluded_volume_only",
        "momentum_mode": settings.momentum_mode,
        "lambda_mev": settings.lambda_momentum_mev,
        "a_MeV_fm3": ground_state.a,
        "b_fm3": ground_state.b,
        "K0_MeV": ground_state.K0,
        "vs2_plot_max": peak_value,
        "n_tr_over_n0_plot": n_over_n0[peak_index],
    }
    return summary, rows


def write_csv(path: Path, rows):
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(
            "Usage: compute_vdw_excluded_volume_only.py <example_json> <curve_csv_out> <summary_json_out>",
            file=sys.stderr,
        )
        return 1

    example_path = Path(argv[1]).resolve()
    curve_csv_out = Path(argv[2]).resolve()
    summary_json_out = Path(argv[3]).resolve()

    summary, rows = compute_curve(example_path)
    write_csv(curve_csv_out, rows)
    summary_json_out.parent.mkdir(parents=True, exist_ok=True)
    summary_json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
