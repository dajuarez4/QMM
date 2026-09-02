"""Compute fixed-y critical points from a completed asymmetric beta suite."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


HERE = Path(__file__).resolve().parent
QMM = HERE.parent.parent
sys.path.insert(0, str(QMM / "src"))

from qmm.asymmetry import AsymmetricFitResult  # noqa: E402
from qmm.constants import DEFAULT_QUANTUM_SETTINGS  # noqa: E402
from qmm.fixed_y_quantum import (  # noqa: E402
    FixedYEVPressureEvaluator,
    FixedYQuantumCriticalPoint,
    compute_fixed_y_critical_point,
    fixed_y_quantum_settings,
)
from qmm.ground_state import GroundStateResult  # noqa: E402
from qmm.quantum import QuantumPressureEvaluator, solve_quantum_critical_point  # noqa: E402


SOURCE = QMM / "results/generated/guided_clausius_tvm_beta_lambda300_highres"
SUITE = "guided_clausius_tvm_beta_lambda300_highres"
OUT = HERE / "results_clausius_tvm_lambda300_highres" / "critical_points"
K0_VALUES = tuple(range(260, 316, 5))
BRANCHES = ("target_l", "equal_b")
Y_VALUES = (0.1, 0.2, 0.3, 0.4)
DIETERICI_HIGH_ACCURACY = False


def solve_dieterici_robust(fit: AsymmetricFitResult, y: float) -> FixedYQuantumCriticalPoint | None:
    """Resolve the low-T Dieterici critical point without coarse-grid artifacts."""
    settings = replace(
        fixed_y_quantum_settings(),
        n_k_fd=800,
        k_max_fd=5.0,
        mu_scf_tol=1.0e-11,
        mu_scf_max_iter=200,
        mu_delta=0.10,
        cache_round_digits=12,
        t_min_cp=0.05,
        t_max_cp=25.0,
        n_min_cp=5.0e-4,
        n_max_cp=0.10,
        outer_max_iter=60,
    )
    evaluator = FixedYEVPressureEvaluator(fit, y, settings=settings)
    seed_by_y = {
        0.1: (1.30, 0.011),
        0.2: (7.1, 0.040),
        0.3: (10.7, 0.052),
        0.4: (12.6, 0.058),
        0.5: (13.3, 0.060),
    }
    primary = seed_by_y[min(seed_by_y, key=lambda value: abs(value - y))]
    seeds = (primary, (max(0.2, primary[0] * 0.85), primary[1] * 0.85),
             (primary[0] * 1.15, min(0.085, primary[1] * 1.15)))

    def residual(values: np.ndarray) -> np.ndarray:
        dPdn, d2Pdn2, _, _ = evaluator.critical_equations(float(values[0]), float(values[1]))
        if dPdn is None or d2Pdn2 is None or not np.isfinite(dPdn + d2Pdn2):
            return np.asarray((1.0e4, 1.0e4))
        return np.asarray((dPdn / 10.0, d2Pdn2 / 1000.0))

    best = None
    for seed in seeds:
        solved = least_squares(
            residual,
            seed,
            bounds=([settings.t_min_cp, settings.n_min_cp + settings.h_n],
                    [settings.t_max_cp, settings.n_max_cp - settings.h_n]),
            x_scale=[5.0, 0.04],
            xtol=3.0e-11,
            ftol=3.0e-11,
            gtol=3.0e-11,
            max_nfev=350,
            diff_step=[1.0e-5, 1.0e-5],
        )
        raw_dPdn, raw_d2Pdn2 = residual(solved.x) * np.asarray((10.0, 1000.0))
        quality = abs(raw_dPdn) + 0.01 * abs(raw_d2Pdn2)
        if best is None or quality < best[0]:
            best = (quality, solved, float(raw_dPdn), float(raw_d2Pdn2))
    if best is None or abs(best[2]) > 1.0e-4 or abs(best[3]) > 1.0e-3:
        return None

    _, solved, dPdn, d2Pdn2 = best
    temperature, density = map(float, solved.x)
    pressure, mu_p, mu_n = evaluator.pressure(temperature, density)
    if pressure is None or mu_p is None or mu_n is None:
        return None
    return FixedYQuantumCriticalPoint(
        model=fit.model,
        parameter_name=fit.parameter_name,
        parameter_value=fit.parameter_value,
        target_k0=fit.target_k0,
        y=float(y),
        Tc=temperature,
        nc=density,
        Pc=float(pressure),
        dPdn=dPdn,
        d2Pdn2=d2Pdn2,
        score=float(best[0]),
        iterations=int(solved.nfev),
        mu_p_star=float(mu_p),
        mu_n_star=float(mu_n),
        a_avg=fit.a_avg,
        b_avg=fit.b_avg,
        a_n=fit.a_n,
        a_pn=fit.a_pn,
        b_n=fit.b_n,
        b_pn=fit.b_pn,
    )


def summary(k0: int, branch: str) -> dict:
    path = SOURCE / f"K0_{k0}_{branch}" / f"{SUITE}_k0_{k0}_{branch}_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def solve_asymmetric(k0: int, branch: str, y: float) -> dict[str, object]:
    fit = AsymmetricFitResult(**summary(k0, branch)["asymmetric_fit"])
    if DIETERICI_HIGH_ACCURACY:
        solved = solve_dieterici_robust(fit, y)
    else:
        solved = compute_fixed_y_critical_point(fit, y, settings=fixed_y_quantum_settings())
    if solved is None:
        return {"K0_target_MeV": k0, "branch_mode": branch, "y": y, "status": "failed"}
    row: dict[str, object] = asdict(solved)
    row.update({
        "K0_target_MeV": k0,
        "branch_mode": branch,
        "y": y,
        "status": "ok",
        "source_method": "fixed-y asymmetric",
    })
    return row


def solve_symmetric(k0: int) -> dict[str, object]:
    payload = summary(k0, "equal_b")
    if DIETERICI_HIGH_ACCURACY:
        fit = AsymmetricFitResult(**payload["asymmetric_fit"])
        solved = solve_dieterici_robust(fit, 0.5)
        if solved is None:
            return {"K0_target_MeV": k0, "y": 0.5, "status": "failed"}
        row = asdict(solved)
        row.update({
            "K0_target_MeV": k0,
            "y": 0.5,
            "status": "ok",
            "source_method": "symmetric limit of high-accuracy fixed-y asymmetric solver",
        })
        return row
    ground_state = GroundStateResult(**payload["ground_state"])
    evaluator = QuantumPressureEvaluator(ground_state, settings=DEFAULT_QUANTUM_SETTINGS)
    solved = solve_quantum_critical_point(evaluator, DEFAULT_QUANTUM_SETTINGS)
    if solved is None:
        return {"K0_target_MeV": k0, "y": 0.5, "status": "failed"}
    row: dict[str, object] = asdict(solved)
    row.update({
        "K0_target_MeV": k0,
        "y": 0.5,
        "status": "ok",
        "source_method": "symmetric quantum critical point",
    })
    return row


def main() -> None:
    global SOURCE, SUITE, OUT, DIETERICI_HIGH_ACCURACY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--suite", default=SUITE)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--tag", default="clausius_tvm_lambda300")
    parser.add_argument("--dieterici-high-accuracy", action="store_true")
    args = parser.parse_args()
    SOURCE = args.source.resolve()
    SUITE = args.suite
    OUT = args.output_dir.resolve()
    DIETERICI_HIGH_ACCURACY = args.dieterici_high_accuracy
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    tasks = [(k0, branch, y) for k0 in K0_VALUES for branch in BRANCHES for y in Y_VALUES]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(solve_asymmetric, *task): task for task in tasks}
        symmetric = {pool.submit(solve_symmetric, k0): k0 for k0 in K0_VALUES}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(row["K0_target_MeV"], row["branch_mode"], row["y"], row["status"], flush=True)
        symmetric_rows = [future.result() for future in as_completed(symmetric)]
    for row in symmetric_rows:
        for branch in BRANCHES:
            duplicate = dict(row)
            duplicate["branch_mode"] = branch
            rows.append(duplicate)
    failures = [row for row in rows if row["status"] != "ok"]
    if failures:
        (OUT / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"{len(failures)} critical-point calculations failed")
    rows.sort(key=lambda row: (str(row["branch_mode"]), float(row["y"]), int(row["K0_target_MeV"])))
    fields = sorted({key for row in rows for key in row})
    csv_path = OUT / f"{args.tag}_fixed_y_critical_points.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / f"{args.tag}_fixed_y_critical_points.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(rows)} critical-point rows under {OUT}")


if __name__ == "__main__":
    main()
