#!/usr/bin/env python3
"""Recompute beta-equilibrium sound-speed overlays from stored EOS tables."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


K0_VALUES = [250, 260, 270, 280, 290, 300, 315]
SELECTED_K0_VALUES = [250, 280, 315]
DEFAULT_X_LIMITS = (0.05, 5.0)
X_LIMITS = DEFAULT_X_LIMITS
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 15
LEGEND_SIZE = 16
COLORBAR_LABEL_SIZE = 16
COLORBAR_TICK_SIZE = 14
COLORBAR_TITLE_SIZE = 14


def _local_polynomial_regression(
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


def _reconstruct_vs2(
    densities: np.ndarray,
    energy_density: np.ndarray,
    window_size: int,
    degree: int,
    derivative_floor: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps_smooth, mu_b, d2eps = _local_polynomial_regression(
        densities,
        energy_density,
        window_size,
        degree,
    )
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


def _load_branch(base: Path, prefix: str) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for k0 in K0_VALUES:
        path = base / f"K0_{k0}" / f"{prefix}{k0}_asymmetric_beta.csv"
        if not path.exists():
            continue
        out[k0] = pd.read_csv(path).sort_values("n_over_n0").reset_index(drop=True)
    return out


def _add_recomputed_columns(
    branch_data: dict[int, pd.DataFrame],
    window_size: int,
    degree: int,
) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for k0, df in branch_data.items():
        densities = df["n_b"].to_numpy(dtype=float)
        energy = df["energy_density"].to_numpy(dtype=float)
        pressure, mu_b, vs2 = _reconstruct_vs2(densities, energy, window_size, degree)
        updated = df.copy()
        updated["mu_b_reconstructed"] = mu_b
        updated["pressure_reconstructed"] = pressure
        updated["vs2_reconstructed"] = vs2
        out[k0] = updated
    return out


def _branch_vs2_max(branch_data: dict[int, pd.DataFrame]) -> float:
    peaks = []
    for df in branch_data.values():
        values = df["vs2_reconstructed"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            peaks.append(float(values.max()))
    return max(peaks) if peaks else math.nan


def _write_peak_summary(
    unequal: dict[int, pd.DataFrame],
    equal: dict[int, pd.DataFrame],
    output_dir: Path,
) -> Path:
    rows: list[dict[str, float | int | str]] = []
    for branch_name, branch_data in (
        ("bpn_ne_bn", unequal),
        ("bpn_eq_bn", equal),
    ):
        for k0, df in branch_data.items():
            idx = int(np.nanargmax(df["vs2_reconstructed"].to_numpy(dtype=float)))
            rows.append(
                {
                    "branch": branch_name,
                    "K0_MeV": k0,
                    "n_over_n0_at_peak": float(df.loc[idx, "n_over_n0"]),
                    "quark_fraction_at_peak": float(df.loc[idx, "quark_fraction"]),
                    "vs2_peak": float(df.loc[idx, "vs2_reconstructed"]),
                }
            )
    summary = pd.DataFrame(rows).sort_values(["branch", "K0_MeV"])
    path = output_dir / "beta_sound_speed_peak_summary.csv"
    summary.to_csv(path, index=False)
    return path


def _add_vertical_k0_colorbar(
    figure: plt.Figure,
    box: list[float],
    cmap,
    norm: Normalize,
    title: str,
) -> None:
    cax = figure.add_axes(box)
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    colorbar = figure.colorbar(scalar_map, cax=cax)
    colorbar.set_ticks(K0_VALUES)
    colorbar.set_label(r"$K_0$ [MeV]", fontsize=COLORBAR_LABEL_SIZE)
    colorbar.ax.set_title(title, fontsize=COLORBAR_TITLE_SIZE, pad=8)
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)


def _make_branch_overlay(
    branch_data: dict[int, pd.DataFrame],
    colors: dict[int, tuple[float, float, float, float] | tuple[float, float, float]],
    cmap,
    norm: Normalize,
    linestyle: str,
    title_prefix: str,
    branch_title: str,
    output_path: Path,
    vs_ylim: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_fq, ax_vs, ax_e, ax_p = axes.flat

    for k0 in K0_VALUES:
        df = branch_data[k0]
        x = df["n_over_n0"].to_numpy(dtype=float)
        color = colors[k0]
        ax_fq.plot(x, df["quark_fraction"].to_numpy(dtype=float), color=color, lw=2.5, ls=linestyle)
        ax_vs.plot(x, df["vs2_reconstructed"].to_numpy(dtype=float), color=color, lw=2.5, ls=linestyle)
        ax_e.plot(
            x,
            df["energy_density"].to_numpy(dtype=float) / df["n_b"].to_numpy(dtype=float) / 1000.0 - 0.939,
            color=color,
            lw=2.5,
            ls=linestyle,
        )
        ax_p.plot(x, df["pressure_reconstructed"].to_numpy(dtype=float), color=color, lw=2.5, ls=linestyle)

    ax_fq.set_xlabel(r"$n/n_0$", fontsize=AXIS_LABEL_SIZE)
    ax_fq.set_ylabel(r"$f_Q$", fontsize=AXIS_LABEL_SIZE)
    ax_fq.set_ylim(0.0, 1.05)

    ax_vs.set_xlabel(r"$n/n_0$", fontsize=AXIS_LABEL_SIZE)
    ax_vs.set_ylabel(r"$v_s^2$", fontsize=AXIS_LABEL_SIZE)
    ax_vs.axhline(1.0 / 3.0, color="gray", lw=1.6, ls=(0, (5, 5)))
    ax_vs.set_ylim(-0.02, vs_ylim)

    ax_e.set_xlabel(r"$n/n_0$", fontsize=AXIS_LABEL_SIZE)
    ax_e.set_ylabel(r"$\epsilon/\rho_B - m_N$ [GeV]", fontsize=AXIS_LABEL_SIZE)

    ax_p.set_xlabel(r"$n/n_0$", fontsize=AXIS_LABEL_SIZE)
    ax_p.set_ylabel(r"$P$ [MeV fm$^{-3}$]", fontsize=AXIS_LABEL_SIZE)

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        axis.set_xlim(*X_LIMITS)
        axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    fig.tight_layout(rect=[0.0, 0.0, 0.9, 1.0])
    _add_vertical_k0_colorbar(
        fig,
        [0.92, 0.17, 0.02, 0.66],
        cmap,
        norm,
        branch_title,
    )
    fig.savefig(output_path.with_suffix(".png"), dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _make_comparison_overlay(
    unequal: dict[int, pd.DataFrame],
    equal: dict[int, pd.DataFrame],
    unequal_cmap,
    equal_cmap,
    norm: Normalize,
    base_colors: dict[int, tuple[float, float, float, float]],
    equal_colors: dict[int, tuple[float, float, float]],
    output_path: Path,
    vs_ylim: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_fq, ax_vs, ax_e, ax_p = axes.flat

    for k0 in K0_VALUES:
        unequal_df = unequal[k0]
        equal_df = equal[k0]

        x_unequal = unequal_df["n_over_n0"].to_numpy(dtype=float)
        x_equal = equal_df["n_over_n0"].to_numpy(dtype=float)

        ax_fq.plot(x_unequal, unequal_df["quark_fraction"].to_numpy(dtype=float), color=base_colors[k0], lw=2.5, ls="-")
        ax_fq.plot(x_equal, equal_df["quark_fraction"].to_numpy(dtype=float), color=equal_colors[k0], lw=2.5, ls="--")

        ax_vs.plot(x_unequal, unequal_df["vs2_reconstructed"].to_numpy(dtype=float), color=base_colors[k0], lw=2.5, ls="-")
        ax_vs.plot(x_equal, equal_df["vs2_reconstructed"].to_numpy(dtype=float), color=equal_colors[k0], lw=2.5, ls="--")

        ax_e.plot(
            x_unequal,
            unequal_df["energy_density"].to_numpy(dtype=float) / unequal_df["n_b"].to_numpy(dtype=float) / 1000.0 - 0.939,
            color=base_colors[k0],
            lw=2.5,
            ls="-",
        )
        ax_e.plot(
            x_equal,
            equal_df["energy_density"].to_numpy(dtype=float) / equal_df["n_b"].to_numpy(dtype=float) / 1000.0 - 0.939,
            color=equal_colors[k0],
            lw=2.5,
            ls="--",
        )

        ax_p.plot(x_unequal, unequal_df["pressure_reconstructed"].to_numpy(dtype=float), color=base_colors[k0], lw=2.5, ls="-")
        ax_p.plot(x_equal, equal_df["pressure_reconstructed"].to_numpy(dtype=float), color=equal_colors[k0], lw=2.5, ls="--")

    ax_fq.set_xlabel(r"$n/n_0$", fontsize=AXIS_LABEL_SIZE)
    ax_fq.set_ylabel(r"$f_Q$", fontsize=AXIS_LABEL_SIZE)
    ax_fq.set_ylim(0.0, 1.05)

    branch_handles = [
        Line2D([0], [0], color="black", lw=2.5, ls="-", label=r"$b_{pn} \neq b_n$"),
        Line2D([0], [0], color="black", lw=2.5, ls="--", label=r"$b_{pn} = b_n$"),
    ]
    ax_fq.legend(handles=branch_handles, loc="upper left", frameon=True, fontsize=LEGEND_SIZE)

    ax_vs.set_xlabel(r"$n/n_0$", fontsize=AXIS_LABEL_SIZE)
    ax_vs.set_ylabel(r"$v_s^2$", fontsize=AXIS_LABEL_SIZE)
    ax_vs.axhline(1.0 / 3.0, color="gray", lw=1.6, ls=(0, (5, 5)))
    ax_vs.set_ylim(-0.02, vs_ylim)

    ax_e.set_xlabel(r"$n/n_0$", fontsize=AXIS_LABEL_SIZE)
    ax_e.set_ylabel(r"$\epsilon/\rho_B - m_N$ [GeV]", fontsize=AXIS_LABEL_SIZE)

    ax_p.set_xlabel(r"$n/n_0$", fontsize=AXIS_LABEL_SIZE)
    ax_p.set_ylabel(r"$P$ [MeV fm$^{-3}$]", fontsize=AXIS_LABEL_SIZE)

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        axis.set_xlim(*X_LIMITS)
        axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    fig.tight_layout(rect=[0.0, 0.0, 0.88, 1.0])
    _add_vertical_k0_colorbar(
        fig,
        [0.90, 0.56, 0.02, 0.27],
        unequal_cmap,
        norm,
        r"$b_{pn} \neq b_n$",
    )
    _add_vertical_k0_colorbar(
        fig,
        [0.90, 0.16, 0.02, 0.27],
        equal_cmap,
        norm,
        r"$b_{pn} = b_n$",
    )
    fig.savefig(output_path.with_suffix(".png"), dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _make_paired_k0_overlay(
    unequal: dict[int, pd.DataFrame],
    equal: dict[int, pd.DataFrame],
    paired_cmap,
    norm: Normalize,
    paired_colors: dict[int, tuple[float, float, float, float]],
    output_path: Path,
    vs_ylim: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_fq, ax_vs, ax_e, ax_p = axes.flat

    for k0 in K0_VALUES:
        unequal_df = unequal[k0]
        equal_df = equal[k0]
        color = paired_colors[k0]

        x_unequal = unequal_df["n_over_n0"].to_numpy(dtype=float)
        x_equal = equal_df["n_over_n0"].to_numpy(dtype=float)

        ax_fq.plot(x_unequal, unequal_df["quark_fraction"].to_numpy(dtype=float), color=color, lw=3.0, ls="-")
        ax_fq.plot(x_equal, equal_df["quark_fraction"].to_numpy(dtype=float), color=color, lw=2.7, ls="--", alpha=0.95)

        ax_vs.plot(x_unequal, unequal_df["vs2_reconstructed"].to_numpy(dtype=float), color=color, lw=3.0, ls="-")
        ax_vs.plot(x_equal, equal_df["vs2_reconstructed"].to_numpy(dtype=float), color=color, lw=2.7, ls="--", alpha=0.95)

        ax_e.plot(
            x_unequal,
            unequal_df["energy_density"].to_numpy(dtype=float) / unequal_df["n_b"].to_numpy(dtype=float) / 1000.0 - 0.939,
            color=color,
            lw=3.0,
            ls="-",
        )
        ax_e.plot(
            x_equal,
            equal_df["energy_density"].to_numpy(dtype=float) / equal_df["n_b"].to_numpy(dtype=float) / 1000.0 - 0.939,
            color=color,
            lw=2.7,
            ls="--",
            alpha=0.95,
        )

        ax_p.plot(x_unequal, unequal_df["pressure_reconstructed"].to_numpy(dtype=float), color=color, lw=3.0, ls="-")
        ax_p.plot(x_equal, equal_df["pressure_reconstructed"].to_numpy(dtype=float), color=color, lw=2.7, ls="--", alpha=0.95)

    ax_fq.set_xlabel(r"$n/n_0$", fontsize=AXIS_LABEL_SIZE)
    ax_fq.set_ylabel(r"$f_Q$", fontsize=AXIS_LABEL_SIZE)
    ax_fq.set_ylim(0.0, 1.05)

    branch_handles = [
        Line2D([0], [0], color="black", lw=3.0, ls="-", label=r"$b_{pn} \neq b_n$"),
        Line2D([0], [0], color="black", lw=2.7, ls="--", label=r"$b_{pn} = b_n$"),
    ]
    ax_fq.legend(handles=branch_handles, loc="upper left", frameon=True, fontsize=LEGEND_SIZE)

    ax_vs.set_xlabel(r"$n/n_0$", fontsize=AXIS_LABEL_SIZE)
    ax_vs.set_ylabel(r"$v_s^2$", fontsize=AXIS_LABEL_SIZE)
    ax_vs.axhline(1.0 / 3.0, color="gray", lw=1.6, ls=(0, (5, 5)))
    ax_vs.set_ylim(-0.02, vs_ylim)

    ax_e.set_xlabel(r"$n/n_0$", fontsize=AXIS_LABEL_SIZE)
    ax_e.set_ylabel(r"$\epsilon/\rho_B - m_N$ [GeV]", fontsize=AXIS_LABEL_SIZE)

    ax_p.set_xlabel(r"$n/n_0$", fontsize=AXIS_LABEL_SIZE)
    ax_p.set_ylabel(r"$P$ [MeV fm$^{-3}$]", fontsize=AXIS_LABEL_SIZE)

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        axis.set_xlim(*X_LIMITS)
        axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    fig.tight_layout(rect=[0.0, 0.0, 0.9, 1.0])
    _add_vertical_k0_colorbar(
        fig,
        [0.92, 0.17, 0.02, 0.66],
        paired_cmap,
        norm,
        "same color per $K_0$",
    )
    fig.savefig(output_path.with_suffix(".png"), dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _make_selected_k0_overlay(
    unequal: dict[int, pd.DataFrame],
    equal: dict[int, pd.DataFrame],
    output_path: Path,
    vs_ylim: float,
) -> None:
    selected_colors = {
        250: "#1f77b4",
        280: "#2ca02c",
        315: "#d62728",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_fq, ax_vs, ax_e, ax_p = axes.flat

    for k0 in SELECTED_K0_VALUES:
        unequal_df = unequal[k0]
        equal_df = equal[k0]
        color = selected_colors[k0]

        x_unequal = unequal_df["n_over_n0"].to_numpy(dtype=float)
        x_equal = equal_df["n_over_n0"].to_numpy(dtype=float)

        ax_fq.plot(x_unequal, unequal_df["quark_fraction"].to_numpy(dtype=float), color=color, lw=3.0, ls="-")
        ax_fq.plot(x_equal, equal_df["quark_fraction"].to_numpy(dtype=float), color=color, lw=2.8, ls="--")

        ax_vs.plot(x_unequal, unequal_df["vs2_reconstructed"].to_numpy(dtype=float), color=color, lw=3.0, ls="-")
        ax_vs.plot(x_equal, equal_df["vs2_reconstructed"].to_numpy(dtype=float), color=color, lw=2.8, ls="--")

        ax_e.plot(
            x_unequal,
            unequal_df["energy_density"].to_numpy(dtype=float) / unequal_df["n_b"].to_numpy(dtype=float) / 1000.0 - 0.939,
            color=color,
            lw=3.0,
            ls="-",
        )
        ax_e.plot(
            x_equal,
            equal_df["energy_density"].to_numpy(dtype=float) / equal_df["n_b"].to_numpy(dtype=float) / 1000.0 - 0.939,
            color=color,
            lw=2.8,
            ls="--",
        )

        ax_p.plot(x_unequal, unequal_df["pressure_reconstructed"].to_numpy(dtype=float), color=color, lw=3.0, ls="-")
        ax_p.plot(x_equal, equal_df["pressure_reconstructed"].to_numpy(dtype=float), color=color, lw=2.8, ls="--")

    ax_fq.set_xlabel(r"$n/n_0$", fontsize=AXIS_LABEL_SIZE)
    ax_fq.set_ylabel(r"$f_Q$", fontsize=AXIS_LABEL_SIZE)
    ax_fq.set_ylim(0.0, 1.05)

    branch_handles = [
        Line2D([0], [0], color="black", lw=3.0, ls="-", label=r"$b_{pn} \neq b_n$"),
        Line2D([0], [0], color="black", lw=2.8, ls="--", label=r"$b_{pn} = b_n$"),
    ]
    ax_fq.legend(handles=branch_handles, loc="upper left", frameon=True, fontsize=LEGEND_SIZE)

    k0_handles = [
        Line2D([0], [0], color=selected_colors[k0], lw=3.0, label=rf"$K_0={k0}\,\mathrm{{MeV}}$")
        for k0 in SELECTED_K0_VALUES
    ]
    fig.legend(
        handles=k0_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=True,
        fontsize=LEGEND_SIZE,
    )

    ax_vs.set_xlabel(r"$n/n_0$", fontsize=AXIS_LABEL_SIZE)
    ax_vs.set_ylabel(r"$v_s^2$", fontsize=AXIS_LABEL_SIZE)
    ax_vs.axhline(1.0 / 3.0, color="gray", lw=1.6, ls=(0, (5, 5)))
    ax_vs.set_ylim(-0.02, vs_ylim)

    ax_e.set_xlabel(r"$n/n_0$", fontsize=AXIS_LABEL_SIZE)
    ax_e.set_ylabel(r"$\epsilon/\rho_B - m_N$ [GeV]", fontsize=AXIS_LABEL_SIZE)

    ax_p.set_xlabel(r"$n/n_0$", fontsize=AXIS_LABEL_SIZE)
    ax_p.set_ylabel(r"$P$ [MeV fm$^{-3}$]", fontsize=AXIS_LABEL_SIZE)

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        axis.set_xlim(*X_LIMITS)
        axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(output_path.with_suffix(".png"), dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unequal-root",
        type=Path,
        default=Path("QMM/results/generated/clausius_asymmetric_scan_lambda200_smooth"),
        help="Root folder for the b_pn != b_n beta-equilibrium CSV files.",
    )
    parser.add_argument(
        "--equal-root",
        type=Path,
        default=Path("QMM/results/generated/clausius_equal_b_scan_fullrange_lambda200_smooth"),
        help="Root folder for the b_pn = b_n beta-equilibrium CSV files.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=29,
        help="Local polynomial window size used for derivative reconstruction.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=3,
        help="Polynomial degree used for the local reconstruction.",
    )
    parser.add_argument(
        "--xmin",
        type=float,
        default=DEFAULT_X_LIMITS[0],
        help="Lower x-axis limit for the overlay plots in units of n/n0.",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=DEFAULT_X_LIMITS[1],
        help="Upper x-axis limit for the overlay plots in units of n/n0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("QMM/results/generated/clausius_beta_vs2_postprocess_window29"),
        help="Target directory for the recomputed overlays and summary table.",
    )
    args = parser.parse_args()
    if args.xmax <= args.xmin:
        raise SystemExit(f"Invalid x-axis limits: xmin={args.xmin}, xmax={args.xmax}")

    global X_LIMITS
    X_LIMITS = (args.xmin, args.xmax)

    unequal_raw = _load_branch(args.unequal_root, "clausius_k0_")
    equal_raw = _load_branch(args.equal_root, "clausius_equal_b_k0_")
    missing_unequal = [k0 for k0 in K0_VALUES if k0 not in unequal_raw]
    missing_equal = [k0 for k0 in K0_VALUES if k0 not in equal_raw]
    if missing_unequal or missing_equal:
        raise SystemExit(
            f"Missing beta CSV files. unequal={missing_unequal}, equal={missing_equal}"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    unequal = _add_recomputed_columns(unequal_raw, args.window, args.degree)
    equal = _add_recomputed_columns(equal_raw, args.window, args.degree)

    unequal_cmap = plt.get_cmap("Blues")
    equal_cmap = plt.get_cmap("Oranges")
    paired_cmap = plt.get_cmap("cividis")
    norm = Normalize(vmin=min(K0_VALUES), vmax=max(K0_VALUES))
    base_colors = {
        k0: unequal_cmap(norm(k0))
        for k0 in K0_VALUES
    }
    equal_colors = {
        k0: equal_cmap(norm(k0))
        for k0 in K0_VALUES
    }
    paired_colors = {
        k0: paired_cmap(norm(k0))
        for k0 in K0_VALUES
    }

    vs_max = max(_branch_vs2_max(unequal), _branch_vs2_max(equal))
    vs_ylim = max(1.0, 1.05 * vs_max)

    _make_branch_overlay(
        unequal,
        base_colors,
        unequal_cmap,
        norm,
        "-",
        "Beta-equilibrium",
        r"$b_{pn} \neq b_n$",
        output_dir / "clausius_unequal_b_beta_equilibrium_k0_overlay",
        vs_ylim,
    )
    _make_branch_overlay(
        equal,
        equal_colors,
        equal_cmap,
        norm,
        "--",
        "Beta-equilibrium",
        r"$b_{pn} = b_n$",
        output_dir / "clausius_equal_b_beta_equilibrium_k0_overlay",
        vs_ylim,
    )
    _make_comparison_overlay(
        unequal,
        equal,
        unequal_cmap,
        equal_cmap,
        norm,
        base_colors,
        equal_colors,
        output_dir / "clausius_branch_comparison_k0_overlay",
        vs_ylim,
    )
    _make_paired_k0_overlay(
        unequal,
        equal,
        paired_cmap,
        norm,
        paired_colors,
        output_dir / "clausius_branch_comparison_same_k0_colors",
        vs_ylim,
    )
    _make_selected_k0_overlay(
        unequal,
        equal,
        output_dir / "clausius_branch_comparison_k0_250_280_315",
        vs_ylim,
    )
    summary_path = _write_peak_summary(unequal, equal, output_dir)

    print(f"window={args.window}")
    print(f"degree={args.degree}")
    print(f"x_limits={X_LIMITS}")
    print(f"vs2_max_unequal={_branch_vs2_max(unequal):.6f}")
    print(f"vs2_max_equal={_branch_vs2_max(equal):.6f}")
    print(f"summary={summary_path}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
