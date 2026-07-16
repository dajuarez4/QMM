from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from qmm.constants import DEFAULT_PHYSICAL_CONSTANTS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = ROOT / "results" / "reports" / "asymmetric_ev_full_suite" / "run_manifest.csv"
DEFAULT_FIGURE_DIR = ROOT / "results" / "reports" / "asymmetric_ev_full_suite" / "figures"
MANIFEST_PATH = DEFAULT_MANIFEST_PATH
FIGURE_DIR = DEFAULT_FIGURE_DIR
X_LIMITS = (0.05, 5.0)
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 12
LEGEND_SIZE = 11
LINE_WIDTH = 2.6
REFERENCE_LINE_WIDTH = 1.1

MODEL_ORDER = ("clausius", "clausius_cs", "clausius_tvm")
MODEL_LABELS = {
    "clausius": "Clausius-VDW",
    "clausius_cs": "Clausius-CS",
    "clausius_tvm": "Clausius-TVM",
}
MODEL_COLORS = {
    "clausius": "#0F4C81",
    "clausius_cs": "#BC4749",
    "clausius_tvm": "#2A9D8F",
}
FALLBACK_COLORS = ("#0F4C81", "#BC4749", "#2A9D8F", "#7A4EAB", "#C76B28")


def load_manifest_rows() -> list[dict[str, str]]:
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_k0(value: str | float) -> float:
    return round(float(value), 9)


def k0_tag(value: float) -> str:
    rounded = round(float(value), 3)
    if abs(rounded - round(rounded)) < 1.0e-9:
        return str(int(round(rounded)))
    return f"{rounded:.3f}".replace(".", "p")


def available_k0_values() -> list[float]:
    return sorted({normalize_k0(row["target_k0"]) for row in load_manifest_rows()})


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def energy_per_baryon_minus_m(energy_density: float, density: float) -> float:
    if density <= 0.0:
        return math.nan
    return energy_density / density - DEFAULT_PHYSICAL_CONSTANTS.m_nucleon


def local_polynomial_regression(
    x_values: np.ndarray,
    y_values: np.ndarray,
    window_size: int,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if window_size % 2 == 0:
        window_size += 1
    half = window_size // 2
    count = len(x_values)
    smoothed = np.empty(count, dtype=float)
    first = np.empty(count, dtype=float)
    second = np.empty(count, dtype=float)

    for index in range(count):
        left = max(0, index - half)
        right = min(count, index + half + 1)
        if right - left < degree + 1:
            if left == 0:
                right = min(count, degree + 1)
            else:
                left = max(0, count - degree - 1)

        x_center = x_values[index]
        x_local = x_values[left:right] - x_center
        y_local = y_values[left:right]
        design = np.vstack([x_local**power for power in range(degree + 1)]).T
        coeffs, *_ = np.linalg.lstsq(design, y_local, rcond=None)

        smoothed[index] = coeffs[0]
        first[index] = coeffs[1] if degree >= 1 else 0.0
        second[index] = 2.0 * coeffs[2] if degree >= 2 else 0.0

    return smoothed, first, second


def reconstruct_pressure_mu_vs2(
    densities: np.ndarray,
    energy_density: np.ndarray,
    window_size: int,
    degree: int,
    derivative_floor: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps_smooth, mu_b, d2eps = local_polynomial_regression(densities, energy_density, window_size, degree)
    pressure = densities * mu_b - eps_smooth
    dP_dn = densities * d2eps
    vs2 = np.full_like(mu_b, np.nan)
    valid = (
        np.isfinite(mu_b)
        & np.isfinite(d2eps)
        & np.isfinite(dP_dn)
        & (np.abs(mu_b) >= derivative_floor)
    )
    vs2[valid] = dP_dn[valid] / mu_b[valid]
    return pressure, mu_b, vs2


def load_model_curves(target_k0: float, window_size: int, degree: int) -> dict[str, dict[str, list[float]]]:
    rows = load_manifest_rows()
    selected = [
        row
        for row in rows
        if normalize_k0(row["target_k0"]) == normalize_k0(target_k0) and row["model"] in MODEL_ORDER
    ]
    by_model = {row["model"]: row for row in selected}
    if not by_model:
        raise FileNotFoundError(f"No manifest rows found for K0={target_k0}.")

    curves: dict[str, dict[str, list[float]]] = {}
    for model in [model for model in MODEL_ORDER if model in by_model]:
        beta_path = ROOT / by_model[model]["asymmetric_beta_csv"]
        beta_rows = load_csv_rows(beta_path)
        n_over_n0 = np.array([parse_float(row, "n_over_n0") for row in beta_rows], dtype=float)
        density = np.array([parse_float(row, "n_b") for row in beta_rows], dtype=float)
        energy_density = np.array([parse_float(row, "energy_density") for row in beta_rows], dtype=float)
        pressure_reconstructed, _, vs2_reconstructed = reconstruct_pressure_mu_vs2(
            density,
            energy_density,
            window_size,
            degree,
        )
        valid = n_over_n0 >= 0.05
        if int(np.count_nonzero(valid)) < 3:
            valid = np.ones_like(n_over_n0, dtype=bool)

        curves[model] = {
            "n_over_n0": n_over_n0[valid].tolist(),
            "y": np.array([parse_float(row, "y") for row in beta_rows], dtype=float)[valid].tolist(),
            "vs2": vs2_reconstructed[valid].tolist(),
            "pressure": pressure_reconstructed[valid].tolist(),
            "energy_density": energy_density[valid].tolist(),
            "energy_per_baryon": [
                energy_per_baryon_minus_m(energy_density[index], density[index])
                for index, keep in enumerate(valid)
                if keep
            ],
            "quark_fraction": np.array([parse_float(row, "quark_fraction") for row in beta_rows], dtype=float)[valid].tolist(),
        }
    return curves


def ordered_models(curves: dict[str, dict[str, list[float]]]) -> list[str]:
    ordered = [model for model in MODEL_ORDER if model in curves]
    extras = [model for model in curves if model not in ordered]
    return ordered + extras


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def model_color(model: str, index: int) -> str:
    return MODEL_COLORS.get(model, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def curve_set_tag(curves: dict[str, dict[str, list[float]]]) -> str:
    models = ordered_models(curves)
    if models == list(MODEL_ORDER):
        return "three_model"
    if len(models) == 1:
        return models[0]
    return f"{len(models)}model"


def vs2_limits(curves: dict[str, dict[str, list[float]]]) -> tuple[float, float]:
    values = [
        value
        for curve in curves.values()
        for value in curve["vs2"]
        if math.isfinite(value)
    ]
    if not values:
        return -0.05, 1.0
    peak = max(values)
    upper = max(1.0, 1.08 * peak)
    return -0.05, 0.1 * math.ceil(upper / 0.1)


def save_figure(fig, base_name: str) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in ("png", "pdf"):
        output_path = FIGURE_DIR / f"{base_name}.{suffix}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        written.append(output_path)
    return written


def configure_matplotlib():
    mpl_cache_dir = Path(tempfile.gettempdir()) / "qmm_mplconfig"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.linewidth": 0.9,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "legend.frameon": False,
            "legend.fontsize": LEGEND_SIZE,
        }
    )
    import matplotlib.pyplot as plt

    return plt


def style_axis(axis, *, x_limits: tuple[float, float] | None = None) -> None:
    axis.grid(True, color="#d6d6d6", alpha=0.45, lw=0.6)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", which="major", width=0.9, length=5)
    axis.tick_params(axis="both", which="minor", width=0.7, length=3)
    if x_limits is not None:
        axis.set_xlim(*x_limits)


def make_four_panel(
    curves: dict[str, dict[str, list[float]]],
    target_k0: float,
    window_size: int,
    degree: int,
) -> list[Path]:
    del window_size, degree
    plt = configure_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), constrained_layout=True)
    lower_vs2, upper_vs2 = vs2_limits(curves)

    for index, model in enumerate(ordered_models(curves)):
        curve = curves[model]
        color = model_color(model, index)
        label = model_label(model)
        axes[0, 0].plot(curve["n_over_n0"], curve["y"], lw=LINE_WIDTH, color=color, label=label, solid_capstyle="round")
        axes[0, 1].plot(curve["n_over_n0"], curve["vs2"], lw=LINE_WIDTH, color=color, label=label, solid_capstyle="round")
        axes[1, 0].plot(curve["n_over_n0"], curve["energy_per_baryon"], lw=LINE_WIDTH, color=color, label=label, solid_capstyle="round")
        axes[1, 1].plot(curve["n_over_n0"], curve["quark_fraction"], lw=LINE_WIDTH, color=color, label=label, solid_capstyle="round")

    axes[0, 0].set_xlabel(r"$n / n_0$")
    axes[0, 0].set_ylabel(r"$y$")

    axes[0, 1].set_xlabel(r"$n / n_0$")
    axes[0, 1].set_ylabel(r"$v_s^2$")
    axes[0, 1].set_ylim(lower_vs2, upper_vs2)
    axes[0, 1].axhline(1.0 / 3.0, color="#7a7a7a", ls=(0, (5, 5)), lw=REFERENCE_LINE_WIDTH)

    axes[1, 0].set_xlabel(r"$n / n_0$")
    axes[1, 0].set_ylabel(r"$\varepsilon / n_B - m_N$ [MeV]")
    axes[1, 0].axhline(0.0, color="#7a7a7a", ls="--", lw=REFERENCE_LINE_WIDTH)

    axes[1, 1].set_xlabel(r"$n / n_0$")
    axes[1, 1].set_ylabel(r"$f_Q$")

    for axis in axes.flat:
        style_axis(axis, x_limits=X_LIMITS)
    axes[0, 0].legend(loc="lower right", handlelength=2.8)

    set_tag = curve_set_tag(curves)
    written = save_figure(fig, f"k0_{k0_tag(target_k0)}_{set_tag}_beta_equilibrium_4panel")
    plt.close(fig)
    return written


def make_full_six_panel(
    curves: dict[str, dict[str, list[float]]],
    target_k0: float,
    window_size: int,
    degree: int,
) -> list[Path]:
    del window_size, degree
    plt = configure_matplotlib()

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.4), constrained_layout=True)
    lower_vs2, upper_vs2 = vs2_limits(curves)

    for index, model in enumerate(ordered_models(curves)):
        curve = curves[model]
        color = model_color(model, index)
        label = model_label(model)
        axes[0, 0].plot(curve["n_over_n0"], curve["y"], lw=LINE_WIDTH, color=color, label=label, solid_capstyle="round")
        axes[0, 1].plot(curve["n_over_n0"], curve["quark_fraction"], lw=LINE_WIDTH, color=color, label=label, solid_capstyle="round")
        axes[0, 2].plot(curve["n_over_n0"], curve["vs2"], lw=LINE_WIDTH, color=color, label=label, solid_capstyle="round")
        axes[1, 0].plot(curve["n_over_n0"], curve["pressure"], lw=LINE_WIDTH, color=color, label=label, solid_capstyle="round")
        axes[1, 1].plot(curve["n_over_n0"], curve["energy_per_baryon"], lw=LINE_WIDTH, color=color, label=label, solid_capstyle="round")
        axes[1, 2].plot(curve["energy_density"], curve["pressure"], lw=LINE_WIDTH, color=color, label=label, solid_capstyle="round")

    axes[0, 0].set_xlabel(r"$n / n_0$")
    axes[0, 0].set_ylabel(r"$y$")

    axes[0, 1].set_xlabel(r"$n / n_0$")
    axes[0, 1].set_ylabel(r"$f_Q$")

    axes[0, 2].set_xlabel(r"$n / n_0$")
    axes[0, 2].set_ylabel(r"$v_s^2$")
    axes[0, 2].set_ylim(lower_vs2, upper_vs2)
    axes[0, 2].axhline(1.0 / 3.0, color="#7a7a7a", ls=(0, (5, 5)), lw=REFERENCE_LINE_WIDTH)

    axes[1, 0].set_xlabel(r"$n / n_0$")
    axes[1, 0].set_ylabel(r"$P$ [MeV fm$^{-3}$]")
    axes[1, 0].axhline(0.0, color="#7a7a7a", ls="--", lw=REFERENCE_LINE_WIDTH)

    axes[1, 1].set_xlabel(r"$n / n_0$")
    axes[1, 1].set_ylabel(r"$\varepsilon / n_B - m_N$ [MeV]")
    axes[1, 1].axhline(0.0, color="#7a7a7a", ls="--", lw=REFERENCE_LINE_WIDTH)

    axes[1, 2].set_xlabel(r"$\varepsilon$ [MeV fm$^{-3}$]")
    axes[1, 2].set_ylabel(r"$P$ [MeV fm$^{-3}$]")

    for index, axis in enumerate(axes.flat):
        style_axis(axis, x_limits=X_LIMITS if index < 5 else None)
    axes[0, 0].legend(loc="lower right", handlelength=2.8)

    set_tag = curve_set_tag(curves)
    written = save_figure(fig, f"k0_{k0_tag(target_k0)}_{set_tag}_beta_equilibrium_full")
    plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay asymmetric beta-equilibrium curves for one K0.")
    parser.add_argument("--k0", type=float, default=250.0, help="Target incompressibility K0 in MeV.")
    parser.add_argument("--window", type=int, default=29, help="Local polynomial window size.")
    parser.add_argument("--degree", type=int, default=3, help="Local polynomial degree.")
    parser.add_argument("--all-k0", action="store_true", help="Generate figures for every K0 in the manifest.")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(DEFAULT_MANIFEST_PATH.relative_to(ROOT)),
        help="Manifest CSV path relative to the QMM root.",
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default=str(DEFAULT_FIGURE_DIR.relative_to(ROOT)),
        help="Figure output directory relative to the QMM root.",
    )
    args = parser.parse_args()

    global MANIFEST_PATH, FIGURE_DIR
    MANIFEST_PATH = ROOT / args.manifest
    FIGURE_DIR = ROOT / args.figure_dir

    written = []
    target_k0_values = available_k0_values() if args.all_k0 else [args.k0]
    for target_k0 in target_k0_values:
        curves = load_model_curves(target_k0, args.window, args.degree)
        written.extend(make_four_panel(curves, target_k0, args.window, args.degree))
        written.extend(make_full_six_panel(curves, target_k0, args.window, args.degree))

    for path in written:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
