#!/usr/bin/env python3
"""Compare beta-equilibrium branch diagnostics from saved QMM CSV outputs.

This script is meant for exactly the question that comes up when one branch
shows a strong dip in ``v_s^2`` and a corresponding pressure shoulder while the
other branch does not.  It works from the saved beta-equilibrium CSV tables, so
you do not need to rerun the full asymmetric workflow.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from qmm.constants import QuarkyonicSettings
from qmm.sound_speed import (
    reconstruct_sound_speed_curve,
    reconstruct_sound_speed_curve_gradient_check,
)


@dataclass(frozen=True)
class BranchDiagnostics:
    """One branch loaded from stored beta-equilibrium workflow outputs."""

    label: str
    branch_mode: str
    color: str
    fit_row: dict[str, float | str]
    frame: pd.DataFrame


def _discover_single(directory: Path, suffix: str) -> Path:
    matches = sorted(directory.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '*{suffix}' file in {directory}, found {len(matches)}."
        )
    return matches[0]


def _branch_label(branch_mode: str) -> str:
    return r"$b_{pn} = b_n$" if branch_mode == "equal_b" else r"$b_{pn} \neq b_n$"


def _fit_label(fit_row: dict[str, float | str]) -> str:
    branch_mode = str(fit_row["branch_mode"])
    l_value = float(fit_row["L"])
    return f"{_branch_label(branch_mode)}, L={l_value:.1f} MeV"


def _load_fit_row(path: Path) -> dict[str, float | str]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Fit CSV has no rows: {path}")
    return frame.iloc[0].to_dict()


def _build_branch(
    beta_csv: Path,
    fit_csv: Path,
    color: str,
    settings: QuarkyonicSettings,
) -> BranchDiagnostics:
    frame = pd.read_csv(beta_csv).sort_values("n_over_n0").reset_index(drop=True)
    fit_row = _load_fit_row(fit_csv)

    densities = frame["n_b"].to_numpy(dtype=float)
    energy_density = frame["energy_density"].to_numpy(dtype=float)

    smooth = reconstruct_sound_speed_curve(
        densities.tolist(),
        energy_density.tolist(),
        settings.smoothing_window,
        settings.smoothing_degree,
        settings.derivative_floor,
    )
    direct = reconstruct_sound_speed_curve_gradient_check(
        densities.tolist(),
        energy_density.tolist(),
        settings.derivative_floor,
    )

    enriched = frame.copy()
    enriched["pressure_smooth"] = np.asarray(smooth.pressure, dtype=float)
    enriched["dP_dn_smooth"] = np.asarray(smooth.dP_dn, dtype=float)
    enriched["vs2_smooth"] = np.asarray(smooth.vs2, dtype=float)
    enriched["pressure_direct"] = np.asarray(direct.pressure, dtype=float)
    enriched["dP_dn_direct"] = np.asarray(direct.dP_dn, dtype=float)
    enriched["vs2_direct"] = np.asarray(direct.vs2, dtype=float)

    return BranchDiagnostics(
        label=_fit_label(fit_row),
        branch_mode=str(fit_row["branch_mode"]),
        color=color,
        fit_row=fit_row,
        frame=enriched,
    )


def _feature_summary_row(
    branch: BranchDiagnostics,
    feature_min: float,
    feature_max: float,
) -> dict[str, float | str]:
    frame = branch.frame
    mask = (frame["n_over_n0"] >= feature_min) & (frame["n_over_n0"] <= feature_max)
    window = frame.loc[mask].reset_index(drop=True)
    if window.empty:
        raise ValueError(
            f"No rows found in feature window [{feature_min}, {feature_max}] for {branch.branch_mode}."
        )

    peak_index = int(window["vs2_smooth"].idxmax())
    post_peak = window.loc[window.index >= peak_index]
    dip_index = int(post_peak["vs2_smooth"].idxmin())
    dPdn_index = int(window["dP_dn_smooth"].idxmin())

    dip_row = window.loc[dip_index]
    peak_row = window.loc[peak_index]
    dpdn_row = window.loc[dPdn_index]

    return {
        "branch_mode": branch.branch_mode,
        "label": branch.label,
        "L_MeV": float(branch.fit_row["L"]),
        "n_over_n0_vs2_peak": float(peak_row["n_over_n0"]),
        "vs2_peak_smooth": float(peak_row["vs2_smooth"]),
        "n_over_n0_vs2_dip": float(dip_row["n_over_n0"]),
        "vs2_dip_smooth": float(dip_row["vs2_smooth"]),
        "vs2_dip_direct": float(dip_row["vs2_direct"]),
        "pressure_at_dip_MeV_fm3": float(dip_row["pressure_smooth"]),
        "dP_dn_at_dip": float(dip_row["dP_dn_smooth"]),
        "y_at_dip": float(dip_row["y"]),
        "quark_fraction_at_dip": float(dip_row["quark_fraction"]),
        "mu_e_at_dip_MeV": float(dip_row["mu_e"]),
        "n_over_n0_min_dP_dn": float(dpdn_row["n_over_n0"]),
        "min_dP_dn": float(dpdn_row["dP_dn_smooth"]),
    }


def _plot_comparison(
    branches: list[BranchDiagnostics],
    output_prefix: Path,
    feature_min: float,
    feature_max: float,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)
    ax_y, ax_fq, ax_pressure, ax_dpdn, ax_vs2, ax_feature = axes.flat
    ax_feature_vs2 = ax_feature.twinx()

    summary_rows = {
        branch.branch_mode: _feature_summary_row(branch, feature_min, feature_max)
        for branch in branches
    }

    for branch in branches:
        frame = branch.frame
        x_values = frame["n_over_n0"].to_numpy(dtype=float)

        ax_y.plot(x_values, frame["y"].to_numpy(dtype=float), color=branch.color, lw=2.4, label=branch.label)
        ax_fq.plot(
            x_values,
            frame["quark_fraction"].to_numpy(dtype=float),
            color=branch.color,
            lw=2.4,
            label=branch.label,
        )
        ax_pressure.plot(
            x_values,
            frame["pressure_smooth"].to_numpy(dtype=float),
            color=branch.color,
            lw=2.4,
            label=branch.label,
        )
        ax_dpdn.plot(
            x_values,
            frame["dP_dn_smooth"].to_numpy(dtype=float),
            color=branch.color,
            lw=2.4,
            label=branch.label,
        )

        ax_vs2.plot(
            x_values,
            frame["vs2_smooth"].to_numpy(dtype=float),
            color=branch.color,
            lw=2.4,
        )
        ax_vs2.plot(
            x_values,
            frame["vs2_direct"].to_numpy(dtype=float),
            color=branch.color,
            lw=1.8,
            ls="--",
        )

        feature_mask = (frame["n_over_n0"] >= feature_min) & (frame["n_over_n0"] <= feature_max)
        feature_frame = frame.loc[feature_mask]
        feature_x = feature_frame["n_over_n0"].to_numpy(dtype=float)
        ax_feature.plot(
            feature_x,
            feature_frame["pressure_smooth"].to_numpy(dtype=float),
            color=branch.color,
            lw=2.4,
        )
        ax_feature_vs2.plot(
            feature_x,
            feature_frame["vs2_smooth"].to_numpy(dtype=float),
            color=branch.color,
            lw=2.0,
            ls="--",
        )

        dip_row = summary_rows[branch.branch_mode]
        ax_feature.scatter(
            [dip_row["n_over_n0_vs2_dip"]],
            [dip_row["pressure_at_dip_MeV_fm3"]],
            color=branch.color,
            s=32,
            zorder=4,
        )
        ax_feature_vs2.scatter(
            [dip_row["n_over_n0_vs2_dip"]],
            [dip_row["vs2_dip_smooth"]],
            color=branch.color,
            s=32,
            zorder=4,
        )

    ax_y.set_title(r"Beta-equilibrium charge fraction $y_\beta$")
    ax_y.set_xlabel(r"$n / n_0$")
    ax_y.set_ylabel(r"$y$")
    ax_y.legend(fontsize=9)

    ax_fq.set_title("Beta-equilibrium quark fraction")
    ax_fq.set_xlabel(r"$n / n_0$")
    ax_fq.set_ylabel(r"$f_Q$")

    ax_pressure.set_title("Pressure reconstructed from $\\epsilon(n_B)$")
    ax_pressure.set_xlabel(r"$n / n_0$")
    ax_pressure.set_ylabel(r"$P$ [MeV fm$^{-3}$]")

    ax_dpdn.set_title(r"Pressure slope $dP/dn_B$")
    ax_dpdn.set_xlabel(r"$n / n_0$")
    ax_dpdn.set_ylabel(r"$dP/dn_B$")

    ax_vs2.set_title(r"Sound speed: smoothed vs direct gradient")
    ax_vs2.set_xlabel(r"$n / n_0$")
    ax_vs2.set_ylabel(r"$v_s^2$")
    ax_vs2.axhline(1.0 / 3.0, color="gray", lw=1.4, ls=(0, (5, 5)))
    vs2_style_handles = [
        Line2D([0], [0], color="black", lw=2.2, ls="-", label="workflow smoothing"),
        Line2D([0], [0], color="black", lw=1.8, ls="--", label="direct gradient"),
    ]
    ax_vs2.legend(handles=vs2_style_handles, fontsize=9, loc="upper right")

    ax_feature.set_title("Feature window: pressure shoulder and $v_s^2$ dip")
    ax_feature.set_xlabel(r"$n / n_0$")
    ax_feature.set_ylabel(r"$P$ [MeV fm$^{-3}$]")
    ax_feature_vs2.set_ylabel(r"$v_s^2$")
    ax_feature.set_xlim(feature_min, feature_max)
    ax_feature_vs2.axhline(1.0 / 3.0, color="gray", lw=1.2, ls=(0, (5, 5)))

    for axis in (ax_y, ax_fq, ax_pressure, ax_dpdn, ax_vs2, ax_feature, ax_feature_vs2):
        axis.grid(alpha=0.25)

    fig.suptitle("Beta-equilibrium branch diagnostics: equal_b versus target_l", fontsize=13)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(summary_rows.values()).sort_values("branch_mode")
    summary.to_csv(output_prefix.with_name(f"{output_prefix.name}_summary.csv"), index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare equal_b and target_l beta-equilibrium diagnostics from stored CSV outputs."
    )
    parser.add_argument(
        "--equal-run-dir",
        type=Path,
        default=Path("results/generated/assym_tvm_vdw_test_equal_b"),
        help="Directory containing one '*_asymmetric_beta.csv' and one '*_asymmetric_fit.csv' for equal_b.",
    )
    parser.add_argument(
        "--target-run-dir",
        type=Path,
        default=Path("results/generated/assym_tvm_vdw_test_target_l"),
        help="Directory containing one '*_asymmetric_beta.csv' and one '*_asymmetric_fit.csv' for target_l.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/generated/assym_tvm_vdw_test_branch_diagnostics/tvm_beta_branch_diagnostics"),
        help="Prefix for the output figure and summary CSV.",
    )
    parser.add_argument(
        "--feature-min",
        type=float,
        default=1.5,
        help="Lower x-limit for the feature-window panel.",
    )
    parser.add_argument(
        "--feature-max",
        type=float,
        default=3.2,
        help="Upper x-limit for the feature-window panel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = QuarkyonicSettings()

    equal_beta_csv = _discover_single(args.equal_run_dir, "_asymmetric_beta.csv")
    equal_fit_csv = _discover_single(args.equal_run_dir, "_asymmetric_fit.csv")
    target_beta_csv = _discover_single(args.target_run_dir, "_asymmetric_beta.csv")
    target_fit_csv = _discover_single(args.target_run_dir, "_asymmetric_fit.csv")

    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    branches = [
        _build_branch(equal_beta_csv, equal_fit_csv, "#E76F51", settings),
        _build_branch(target_beta_csv, target_fit_csv, "#0F4C81", settings),
    ]

    _plot_comparison(
        branches,
        output_prefix,
        args.feature_min,
        args.feature_max,
    )

    print(f"saved: {output_prefix.with_suffix('.png')}")
    print(f"saved: {output_prefix.with_suffix('.pdf')}")
    print(f"saved: {output_prefix.with_name(f'{output_prefix.name}_summary.csv')}")


if __name__ == "__main__":
    main()
