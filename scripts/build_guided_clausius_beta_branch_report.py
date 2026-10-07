#!/usr/bin/env python3
"""Build an Overleaf-ready report for guided_clausius_beta_branch_recompute.

This script packages the saved outputs from
`notebooks/02_some_results/guided_clausius_beta_branch_recompute.ipynb` into:

- a clean folder with copied raw CSV/JSON artifacts
- derived summary tables in CSV and LaTeX
- publication-style PDF/PNG figures
- an Overleaf-ready `main.tex`
- a `.tar.gz` bundle for upload
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import textwrap
import zipfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "results" / "generated" / "guided_clausius_beta_branch_recompute"
REPORT_DIR = ROOT / "results" / "reports" / "guided_clausius_beta_branch_report_overleaf"
ZIP_PATH = ROOT / "results" / "reports" / "guided_clausius_beta_branch_report_overleaf.zip"

WINDOW = 29
DEGREE = 3
DERIVATIVE_FLOOR = 1.0e-12
NUCLEON_MASS_MEV = 938.0
SELECTED_K0 = [250, 280, 315]
SOURCE_NOTEBOOK = ROOT / "notebooks" / "guided_clausius_beta_branch_recompute.ipynb"

BRANCH_LABELS = {
    "target_l": r"$b_{pn} \neq b_n$",
    "equal_b": r"$b_{pn} = b_n$",
}
BRANCH_TEXT = {
    "target_l": "target_l",
    "equal_b": "equal_b",
}
BRANCH_LINESTYLES = {
    "target_l": "-",
    "equal_b": "--",
}
BRANCH_MARKERS = {
    "target_l": "o",
    "equal_b": "s",
}
BRANCH_COLORS = {
    "target_l": "#145A7B",
    "equal_b": "#B05A27",
}


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt_float(value: float, places: int = 6) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    return f"{value:.{places}f}"


def fmt_sci(value: float, places: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    return f"{value:.{places}e}"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def branch_sort_key(branch_mode: str) -> tuple[int, str]:
    order = {"target_l": 0, "equal_b": 1}
    return (order.get(branch_mode, 99), branch_mode)


def parse_branch_dir(path: Path) -> tuple[int, str]:
    # Expected layout: K0_250_target_l
    parts = path.name.split("_", 2)
    if len(parts) != 3 or parts[0] != "K0":
        raise ValueError(f"Unexpected branch directory format: {path.name}")
    return int(parts[1]), parts[2]


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "lines.linewidth": 2.4,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.facecolor": "white",
        }
    )


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
    derivative_floor: float = DERIVATIVE_FLOOR,
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


def load_numeric_curve(path: Path) -> dict[str, np.ndarray]:
    rows = read_csv_rows(path)
    fields = list(rows[0].keys())
    curve: dict[str, np.ndarray] = {}
    for field in fields:
        try:
            curve[field] = np.array(
                [
                    float(row[field]) if row.get(field, "") not in {"", "nan", "None"} else math.nan
                    for row in rows
                ],
                dtype=float,
            )
        except ValueError:
            continue
    return curve


def add_file_index_row(rows: list[dict[str, object]], relative_path: Path) -> None:
    parent = relative_path.parent.name
    if parent.startswith("K0_"):
        try:
            k0, branch_mode = parse_branch_dir(Path(parent))
        except ValueError:
            k0 = None
            branch_mode = None
    else:
        k0 = None
        branch_mode = None

    filename = relative_path.name
    kind = "other"
    y_value = None
    if filename.endswith("_asymmetric_fit.csv"):
        kind = "asymmetric_fit"
    elif filename.endswith("_asymmetric_beta.csv"):
        kind = "asymmetric_beta"
    elif "_asymmetric_y_" in filename and filename.endswith(".csv"):
        kind = "fixed_y_profile"
        y_value = filename.split("_asymmetric_y_")[-1].replace(".csv", "")
    elif filename.endswith("_hadronic_eos.csv"):
        kind = "hadronic_eos"
    elif filename.endswith("_ground_state.csv"):
        kind = "ground_state"
    elif filename.endswith("_neutron_star_sequence.csv"):
        kind = "neutron_star_sequence"
    elif filename.endswith("_summary.json"):
        kind = "summary_json"
    elif filename == "fixed_y_quantum_summary.csv":
        kind = "fixed_y_quantum_summary"
    elif filename == "fixed_y_quantum_branch_status.csv":
        kind = "fixed_y_quantum_status"
    elif filename == "beta_sound_speed_peak_summary.csv":
        kind = "beta_sound_speed_peak_summary"

    rows.append(
        {
            "relative_path": relative_path.as_posix(),
            "kind": kind,
            "K0_MeV": "" if k0 is None else k0,
            "branch_mode": "" if branch_mode is None else branch_mode,
            "y_value": "" if y_value is None else y_value,
        }
    )


def write_longtable(
    path: Path,
    caption: str,
    label: str,
    headers: list[str],
    rows: list[list[str]],
    column_spec: str,
) -> None:
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    header_line = " & ".join(headers) + r" \\"
    tex = textwrap.dedent(
        f"""
        \\begin{{longtable}}{{{column_spec}}}
        \\caption{{{caption}}}\\label{{{label}}} \\\\
        \\toprule
        {header_line}
        \\midrule
        \\endfirsthead
        \\caption[]{{{caption} (continued)}} \\\\
        \\toprule
        {header_line}
        \\midrule
        \\endhead
        \\midrule
        \\multicolumn{{{len(headers)}}}{{r}}{{Continued on next page}} \\\\
        \\midrule
        \\endfoot
        \\bottomrule
        \\endlastfoot
        {body}
        \\end{{longtable}}
        """
    ).strip()
    path.write_text(tex + "\n", encoding="utf-8")


def write_table(
    path: Path,
    caption: str,
    label: str,
    headers: list[str],
    rows: list[list[str]],
    column_spec: str,
) -> None:
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    header_line = " & ".join(headers) + r" \\"
    tex = textwrap.dedent(
        f"""
        \\begin{{table}}[htbp]
        \\centering
        \\caption{{{caption}}}
        \\label{{{label}}}
        \\begin{{tabular}}{{{column_spec}}}
        \\toprule
        {header_line}
        \\midrule
        {body}
        \\bottomrule
        \\end{{tabular}}
        \\end{{table}}
        """
    ).strip()
    path.write_text(tex + "\n", encoding="utf-8")


def build_report() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

    ensure_clean_dir(REPORT_DIR)
    (REPORT_DIR / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "data" / "derived").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "sections").mkdir(parents=True, exist_ok=True)

    setup_matplotlib()

    fit_rows: list[dict[str, object]] = []
    neutron_star_rows: list[dict[str, object]] = []
    artifact_index_rows: list[dict[str, object]] = []

    branch_dirs = sorted(
        [path for path in SOURCE_DIR.iterdir() if path.is_dir()],
        key=lambda path: (*parse_branch_dir(path),),
    )

    for branch_dir in branch_dirs:
        k0_value, branch_mode = parse_branch_dir(branch_dir)
        target_dir = REPORT_DIR / "data" / "raw" / branch_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        for source_path in sorted(branch_dir.glob("*")):
            if source_path.suffix.lower() not in {".csv", ".json"}:
                continue
            destination = target_dir / source_path.name
            shutil.copy2(source_path, destination)
            add_file_index_row(artifact_index_rows, destination.relative_to(REPORT_DIR / "data" / "raw"))

        fit_path = branch_dir / f"guided_clausius_beta_branch_recompute_k0_{k0_value}_{branch_mode}_asymmetric_fit.csv"
        fit_row = read_csv_rows(fit_path)[0]
        fit_row["K0_requested_int"] = k0_value
        fit_rows.append(fit_row)

        summary_path = branch_dir / f"guided_clausius_beta_branch_recompute_k0_{k0_value}_{branch_mode}_summary.json"
        summary = load_json(summary_path)
        neutron_star = dict(summary["neutron_star"])
        neutron_star["branch_mode"] = branch_mode
        neutron_star["target_k0"] = float(summary["asymmetric_fit"]["target_k0"])
        neutron_star_rows.append(neutron_star)

    for source_path in sorted(SOURCE_DIR.glob("*")):
        if source_path.is_dir():
            continue
        if source_path.suffix.lower() not in {".csv", ".pdf", ".png"}:
            continue
        destination = REPORT_DIR / "data" / "derived" / source_path.name
        shutil.copy2(source_path, destination)
        add_file_index_row(artifact_index_rows, destination.relative_to(REPORT_DIR / "data" / "derived"))

    fit_rows.sort(key=lambda row: (int(round(float(row["target_k0"]))), branch_sort_key(str(row["branch_mode"]))))
    neutron_star_rows.sort(key=lambda row: (int(round(float(row["target_k0"]))), branch_sort_key(str(row["branch_mode"]))))

    fixed_y_rows = read_csv_rows(SOURCE_DIR / "fixed_y_quantum_summary.csv")
    fixed_y_rows.sort(
        key=lambda row: (
            branch_sort_key(str(row["branch_mode"])),
            int(round(float(row["target_k0"]))),
            float(row["y"]),
        )
    )
    fixed_y_status_rows = read_csv_rows(SOURCE_DIR / "fixed_y_quantum_branch_status.csv")
    fixed_y_status_rows.sort(
        key=lambda row: (
            branch_sort_key(str(row["branch_mode"])),
            int(round(float(row["target_k0"]))),
            float(row["y"]),
        )
    )
    beta_peak_rows = read_csv_rows(SOURCE_DIR / "beta_sound_speed_peak_summary.csv")
    beta_peak_rows.sort(key=lambda row: (int(row["K0_MeV"]), branch_sort_key(str(row["branch_mode"]))))

    # Derived CSV tables.
    fit_fieldnames = [
        "K0_requested_int",
        "branch_mode",
        "parameter_name",
        "parameter_value",
        "a_avg",
        "b_avg",
        "a_n",
        "a_pn",
        "b_n",
        "b_pn",
        "ratio_a_pn_over_a_n",
        "ratio_b_pn_over_b_n",
        "J",
        "L",
        "K0",
        "binding",
        "n_sat",
        "p_sat",
        "j_residual",
        "l_residual",
        "k0_residual",
    ]
    write_csv(REPORT_DIR / "data" / "derived" / "fit_parameters_all.csv", fit_rows, fit_fieldnames)
    write_csv(
        REPORT_DIR / "data" / "derived" / "fixed_y_critical_points_all.csv",
        fixed_y_rows,
        list(fixed_y_rows[0].keys()),
    )
    write_csv(
        REPORT_DIR / "data" / "derived" / "fixed_y_critical_point_status.csv",
        fixed_y_status_rows,
        list(fixed_y_status_rows[0].keys()),
    )
    write_csv(
        REPORT_DIR / "data" / "derived" / "beta_sound_speed_peak_summary.csv",
        beta_peak_rows,
        list(beta_peak_rows[0].keys()),
    )
    write_csv(
        REPORT_DIR / "data" / "derived" / "neutron_star_maxima.csv",
        neutron_star_rows,
        list(neutron_star_rows[0].keys()),
    )
    artifact_index_rows.sort(key=lambda row: (str(row["relative_path"])))
    write_csv(
        REPORT_DIR / "data" / "derived" / "artifact_index.csv",
        artifact_index_rows,
        list(artifact_index_rows[0].keys()),
    )

    # Fixed-y pivot tables.
    y_values = sorted({float(row["y"]) for row in fixed_y_rows})
    k0_values = sorted({int(round(float(row["target_k0"]))) for row in fixed_y_rows})
    for branch_mode in ["target_l", "equal_b"]:
        for quantity in ["Tc", "nc", "Pc"]:
            pivot_rows: list[dict[str, object]] = []
            lookup = {
                (int(round(float(row["target_k0"]))), float(row["y"])): row
                for row in fixed_y_rows
                if row["branch_mode"] == branch_mode
            }
            for k0_value in k0_values:
                row = {"K0_MeV": k0_value}
                for y_value in y_values:
                    item = lookup.get((k0_value, y_value))
                    row[f"y_{y_value:.3f}"] = "" if item is None else item[quantity]
                pivot_rows.append(row)
            write_csv(
                REPORT_DIR / "data" / "derived" / f"fixed_y_{quantity.lower()}_{branch_mode}_pivot.csv",
                pivot_rows,
                list(pivot_rows[0].keys()),
            )

    # Publication-style figures.
    cividis = plt.get_cmap("cividis")
    k0_colors = {k0: cividis(i / max(len(k0_values) - 1, 1)) for i, k0 in enumerate(k0_values)}
    viridis = plt.get_cmap("viridis")
    y_colors = {y: viridis(i / max(len(y_values) - 1, 1)) for i, y in enumerate(y_values)}

    # Fit parameters vs K0.
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)
    fit_specs = [
        ("a_n", r"$a_n$"),
        ("a_pn", r"$a_{pn}$"),
        ("b_n", r"$b_n$"),
        ("b_pn", r"$b_{pn}$"),
    ]
    for axis, (field, title) in zip(axes.flat, fit_specs):
        for branch_mode in ["target_l", "equal_b"]:
            branch_rows = [row for row in fit_rows if row["branch_mode"] == branch_mode]
            x_values = [int(row["K0_requested_int"]) for row in branch_rows]
            y_field = [float(row[field]) for row in branch_rows]
            axis.plot(
                x_values,
                y_field,
                color=BRANCH_COLORS[branch_mode],
                ls=BRANCH_LINESTYLES[branch_mode],
                marker=BRANCH_MARKERS[branch_mode],
                ms=5.0,
                label=BRANCH_LABELS[branch_mode],
            )
        axis.set_title(title)
        axis.set_xlabel(r"$K_0$ [MeV]")
    axes[0, 0].set_ylabel("fit value")
    axes[1, 0].set_ylabel("fit value")
    axes[0, 0].legend(loc="best", fontsize=10)
    fit_param_pdf = REPORT_DIR / "figures" / "fit_parameters_vs_k0.pdf"
    fit_param_png = REPORT_DIR / "figures" / "fit_parameters_vs_k0.png"
    fig.savefig(fit_param_pdf, bbox_inches="tight")
    fig.savefig(fit_param_png, bbox_inches="tight")
    plt.close(fig)

    # Clausius c vs K0 from the common symmetric fit.
    c_rows = sorted(
        [row for row in fit_rows if row["branch_mode"] == "target_l"],
        key=lambda row: int(row["K0_requested_int"]),
    )
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.6), constrained_layout=True)
    ax.plot(
        [int(row["K0_requested_int"]) for row in c_rows],
        [float(row["parameter_value"]) for row in c_rows],
        color="#5C2E91",
        marker="o",
        ms=5.5,
        lw=2.6,
    )
    ax.set_title(r"Clausius $c$ versus $K_0$")
    ax.set_xlabel(r"$K_0$ [MeV]")
    ax.set_ylabel(r"$c$")
    c_vs_k0_pdf = REPORT_DIR / "figures" / "fit_c_vs_k0.pdf"
    c_vs_k0_png = REPORT_DIR / "figures" / "fit_c_vs_k0.png"
    fig.savefig(c_vs_k0_pdf, bbox_inches="tight")
    fig.savefig(c_vs_k0_png, bbox_inches="tight")
    plt.close(fig)

    # Symmetry slope and max-mass summaries.
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    for branch_mode in ["target_l", "equal_b"]:
        branch_rows = [row for row in fit_rows if row["branch_mode"] == branch_mode]
        x_values = [int(row["K0_requested_int"]) for row in branch_rows]
        l_values = [float(row["L"]) for row in branch_rows]
        axes[0].plot(
            x_values,
            l_values,
            color=BRANCH_COLORS[branch_mode],
            ls=BRANCH_LINESTYLES[branch_mode],
            marker=BRANCH_MARKERS[branch_mode],
            ms=5.0,
            label=BRANCH_LABELS[branch_mode],
        )
        ns_rows = [row for row in neutron_star_rows if row["branch_mode"] == branch_mode]
        mass_values = [float(row["max_mass_msun"]) for row in ns_rows]
        axes[1].plot(
            x_values,
            mass_values,
            color=BRANCH_COLORS[branch_mode],
            ls=BRANCH_LINESTYLES[branch_mode],
            marker=BRANCH_MARKERS[branch_mode],
            ms=5.0,
            label=BRANCH_LABELS[branch_mode],
        )
    axes[0].set_title(r"$L$ from the asymmetric fit")
    axes[0].set_xlabel(r"$K_0$ [MeV]")
    axes[0].set_ylabel(r"$L$ [MeV]")
    axes[1].set_title(r"Neutron-star maximum mass")
    axes[1].set_xlabel(r"$K_0$ [MeV]")
    axes[1].set_ylabel(r"$M_{\max}$ [$M_\odot$]")
    axes[0].legend(loc="best", fontsize=10)
    slope_ns_pdf = REPORT_DIR / "figures" / "fit_slope_and_maxmass_vs_k0.pdf"
    slope_ns_png = REPORT_DIR / "figures" / "fit_slope_and_maxmass_vs_k0.png"
    fig.savefig(slope_ns_pdf, bbox_inches="tight")
    fig.savefig(slope_ns_png, bbox_inches="tight")
    plt.close(fig)

    # Fixed-y critical points vs y.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for k0_value in k0_values:
        color = k0_colors[k0_value]
        for branch_mode in ["target_l", "equal_b"]:
            rows = [
                row for row in fixed_y_rows
                if row["branch_mode"] == branch_mode and int(round(float(row["target_k0"]))) == k0_value
            ]
            if not rows:
                continue
            x_values = [float(row["y"]) for row in rows]
            tc_values = [float(row["Tc"]) for row in rows]
            nc_values = [float(row["nc"]) for row in rows]
            axes[0].plot(x_values, tc_values, color=color, ls=BRANCH_LINESTYLES[branch_mode], alpha=0.95)
            axes[1].plot(x_values, nc_values, color=color, ls=BRANCH_LINESTYLES[branch_mode], alpha=0.95)
    axes[0].set_title(r"$T_c$ versus fixed proton fraction")
    axes[0].set_xlabel(r"$y$")
    axes[0].set_ylabel(r"$T_c$ [MeV]")
    axes[1].set_title(r"$n_c$ versus fixed proton fraction")
    axes[1].set_xlabel(r"$y$")
    axes[1].set_ylabel(r"$n_c$ [fm$^{-3}$]")
    k0_handles = [
        Line2D([0], [0], color=k0_colors[k0], lw=2.6, label=fr"$K_0={k0}\,\mathrm{{MeV}}$")
        for k0 in k0_values[:: max(len(k0_values) // 6, 1)]
    ]
    branch_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            lw=2.3,
            ls=BRANCH_LINESTYLES[branch_mode],
            label=BRANCH_LABELS[branch_mode],
        )
        for branch_mode in ["target_l", "equal_b"]
    ]
    axes[0].legend(handles=k0_handles, loc="best", fontsize=8, title=r"$K_0$")
    axes[1].legend(handles=branch_handles, loc="best", fontsize=10, title="branch")
    fixed_y_by_y_pdf = REPORT_DIR / "figures" / "fixed_y_critical_vs_y.pdf"
    fixed_y_by_y_png = REPORT_DIR / "figures" / "fixed_y_critical_vs_y.png"
    fig.savefig(fixed_y_by_y_pdf, bbox_inches="tight")
    fig.savefig(fixed_y_by_y_png, bbox_inches="tight")
    plt.close(fig)

    # Fixed-y critical points vs K0.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for y_value in y_values:
        color = y_colors[y_value]
        for branch_mode in ["target_l", "equal_b"]:
            rows = [
                row for row in fixed_y_rows
                if row["branch_mode"] == branch_mode and math.isclose(float(row["y"]), y_value)
            ]
            if not rows:
                continue
            x_values = [int(round(float(row["target_k0"]))) for row in rows]
            tc_values = [float(row["Tc"]) for row in rows]
            nc_values = [float(row["nc"]) for row in rows]
            axes[0].plot(
                x_values,
                tc_values,
                color=color,
                ls=BRANCH_LINESTYLES[branch_mode],
                marker=BRANCH_MARKERS[branch_mode],
                ms=4.0,
            )
            axes[1].plot(
                x_values,
                nc_values,
                color=color,
                ls=BRANCH_LINESTYLES[branch_mode],
                marker=BRANCH_MARKERS[branch_mode],
                ms=4.0,
            )
    axes[0].set_title(r"$T_c$ versus $K_0$")
    axes[0].set_xlabel(r"$K_0$ [MeV]")
    axes[0].set_ylabel(r"$T_c$ [MeV]")
    axes[1].set_title(r"$n_c$ versus $K_0$")
    axes[1].set_xlabel(r"$K_0$ [MeV]")
    axes[1].set_ylabel(r"$n_c$ [fm$^{-3}$]")
    y_handles = [
        Line2D([0], [0], color=y_colors[y], lw=2.4, label=fr"$y={y:.3f}$")
        for y in y_values
    ]
    axes[0].legend(handles=y_handles, loc="best", fontsize=7, title=r"$y$", ncol=2)
    axes[1].legend(handles=branch_handles, loc="best", fontsize=10, title="branch")
    fixed_y_by_k0_pdf = REPORT_DIR / "figures" / "fixed_y_critical_vs_k0.pdf"
    fixed_y_by_k0_png = REPORT_DIR / "figures" / "fixed_y_critical_vs_k0.png"
    fig.savefig(fixed_y_by_k0_pdf, bbox_inches="tight")
    fig.savefig(fixed_y_by_k0_png, bbox_inches="tight")
    plt.close(fig)

    # Beta-equilibrium selected-K0 curves.
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.8), constrained_layout=True)
    ax_y, ax_fq, ax_vs, ax_epb, ax_eos, ax_legend = axes.flat
    for k0_value in SELECTED_K0:
        color = k0_colors[k0_value]
        for branch_mode in ["target_l", "equal_b"]:
            beta_path = (
                SOURCE_DIR
                / f"K0_{k0_value}_{branch_mode}"
                / f"guided_clausius_beta_branch_recompute_k0_{k0_value}_{branch_mode}_asymmetric_beta.csv"
            )
            curve = load_numeric_curve(beta_path)
            x_values = curve["n_over_n0"]
            density = curve["n_b"]
            energy = curve["energy_density"]
            energy_per_baryon = np.full_like(energy, np.nan)
            positive_density = density > 0.0
            energy_per_baryon[positive_density] = energy[positive_density] / density[positive_density] - NUCLEON_MASS_MEV
            pressure_reconstructed, _, vs2_reconstructed = reconstruct_pressure_mu_vs2(density, energy, WINDOW, DEGREE)
            valid = x_values >= 0.05
            ax_y.plot(
                x_values[valid],
                curve["y"][valid],
                color=color,
                ls=BRANCH_LINESTYLES[branch_mode],
                lw=2.6,
            )
            ax_fq.plot(
                x_values[valid],
                curve["quark_fraction"][valid],
                color=color,
                ls=BRANCH_LINESTYLES[branch_mode],
                lw=2.6,
            )
            ax_vs.plot(
                x_values[valid],
                vs2_reconstructed[valid],
                color=color,
                ls=BRANCH_LINESTYLES[branch_mode],
                lw=2.6,
            )
            ax_epb.plot(
                x_values[valid],
                energy_per_baryon[valid],
                color=color,
                ls=BRANCH_LINESTYLES[branch_mode],
                lw=2.6,
            )
            ax_eos.plot(
                energy[valid],
                pressure_reconstructed[valid],
                color=color,
                ls=BRANCH_LINESTYLES[branch_mode],
                lw=2.6,
            )
    ax_y.set_title(r"Beta-equilibrium charge fraction")
    ax_y.set_xlabel(r"$n/n_0$")
    ax_y.set_ylabel(r"$y$")
    ax_y.set_xlim(0.05, 5.0)
    ax_fq.set_title(r"Beta-equilibrium quark fraction")
    ax_fq.set_xlabel(r"$n/n_0$")
    ax_fq.set_ylabel(r"$f_Q$")
    ax_fq.set_xlim(0.05, 5.0)
    ax_fq.set_ylim(0.0, 1.02)
    ax_vs.set_title(fr"Postprocessed beta $v_s^2$ (window={WINDOW}, degree={DEGREE})")
    ax_vs.set_xlabel(r"$n/n_0$")
    ax_vs.set_ylabel(r"$v_s^2$")
    ax_vs.set_xlim(0.05, 5.0)
    ax_vs.axhline(1.0 / 3.0, color="gray", lw=1.3, ls=(0, (5, 4)))
    ax_epb.set_title(r"Beta-equilibrium $\varepsilon / \rho_B - m_N$")
    ax_epb.set_xlabel(r"$n/n_0$")
    ax_epb.set_ylabel(r"$\varepsilon / \rho_B - m_N$ [MeV]")
    ax_epb.set_xlim(0.05, 5.0)
    ax_eos.set_title(r"Beta-equilibrium EOS")
    ax_eos.set_xlabel(r"$\epsilon$ [MeV fm$^{-3}$]")
    ax_eos.set_ylabel(r"$P$ [MeV fm$^{-3}$]")
    color_handles = [
        Line2D([0], [0], color=k0_colors[k0], lw=2.6, label=fr"$K_0={k0}\,\mathrm{{MeV}}$")
        for k0 in SELECTED_K0
    ]
    ax_legend.axis("off")
    legend_k0 = ax_legend.legend(handles=color_handles, loc="upper left", fontsize=10, title=r"$K_0$")
    ax_legend.add_artist(legend_k0)
    ax_legend.legend(handles=branch_handles, loc="center left", fontsize=10, title="branch")
    beta_selected_pdf = REPORT_DIR / "figures" / "beta_profiles_selected_k0.pdf"
    beta_selected_png = REPORT_DIR / "figures" / "beta_profiles_selected_k0.png"
    fig.savefig(beta_selected_pdf, bbox_inches="tight")
    fig.savefig(beta_selected_png, bbox_inches="tight")
    plt.close(fig)

    # Beta sound-speed peak summaries.
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    for branch_mode in ["target_l", "equal_b"]:
        rows = [row for row in beta_peak_rows if row["branch_mode"] == branch_mode]
        x_values = [int(row["K0_MeV"]) for row in rows]
        peak_vs2 = [float(row["vs2_peak"]) for row in rows]
        peak_n = [float(row["n_over_n0_at_peak"]) for row in rows]
        axes[0].plot(
            x_values,
            peak_vs2,
            color=BRANCH_COLORS[branch_mode],
            ls=BRANCH_LINESTYLES[branch_mode],
            marker=BRANCH_MARKERS[branch_mode],
            ms=5.0,
            label=BRANCH_LABELS[branch_mode],
        )
        axes[1].plot(
            x_values,
            peak_n,
            color=BRANCH_COLORS[branch_mode],
            ls=BRANCH_LINESTYLES[branch_mode],
            marker=BRANCH_MARKERS[branch_mode],
            ms=5.0,
            label=BRANCH_LABELS[branch_mode],
        )
    axes[0].set_title(r"Peak beta-equilibrium $v_s^2$")
    axes[0].set_xlabel(r"$K_0$ [MeV]")
    axes[0].set_ylabel(r"$\max(v_s^2)$")
    axes[1].set_title(r"Density at the beta-equilibrium $v_s^2$ peak")
    axes[1].set_xlabel(r"$K_0$ [MeV]")
    axes[1].set_ylabel(r"$n/n_0$ at peak")
    axes[0].legend(loc="best", fontsize=10)
    beta_peak_pdf = REPORT_DIR / "figures" / "beta_peak_summary_vs_k0.pdf"
    beta_peak_png = REPORT_DIR / "figures" / "beta_peak_summary_vs_k0.png"
    fig.savefig(beta_peak_pdf, bbox_inches="tight")
    fig.savefig(beta_peak_png, bbox_inches="tight")
    plt.close(fig)

    # Tables for LaTeX.
    selected_fit_rows = [
        row
        for row in fit_rows
        if int(row["K0_requested_int"]) in SELECTED_K0
    ]
    write_table(
        REPORT_DIR / "tables" / "fit_parameters_selected.tex",
        "Selected asymmetric-fit couplings from the guided Clausius beta-branch recompute notebook.",
        "tab:fit-selected",
        [r"$K_0$", "branch", r"$c$", r"$a_n$", r"$a_{pn}$", r"$b_n$", r"$b_{pn}$", r"$L$"],
        [
            [
                str(int(row["K0_requested_int"])),
                BRANCH_LABELS[str(row["branch_mode"])],
                fmt_float(float(row["parameter_value"]), 6),
                fmt_float(float(row["a_n"]), 6),
                fmt_float(float(row["a_pn"]), 6),
                fmt_float(float(row["b_n"]), 6),
                fmt_float(float(row["b_pn"]), 6),
                fmt_float(float(row["L"]), 6),
            ]
            for row in selected_fit_rows
        ],
        "r l r r r r r r",
    )
    write_longtable(
        REPORT_DIR / "tables" / "fit_parameters_all.tex",
        "Full asymmetric-fit parameter table copied from the notebook scan.",
        "tab:fit-all",
        [r"$K_0$", "branch", r"$c$", r"$a_n$", r"$a_{pn}$", r"$b_n$", r"$b_{pn}$", r"$J$", r"$L$"],
        [
            [
                str(int(row["K0_requested_int"])),
                BRANCH_LABELS[str(row["branch_mode"])],
                fmt_float(float(row["parameter_value"]), 9),
                fmt_float(float(row["a_n"]), 6),
                fmt_float(float(row["a_pn"]), 6),
                fmt_float(float(row["b_n"]), 6),
                fmt_float(float(row["b_pn"]), 6),
                fmt_float(float(row["J"]), 6),
                fmt_float(float(row["L"]), 6),
            ]
            for row in fit_rows
        ],
        "r l r r r r r r r",
    )
    write_longtable(
        REPORT_DIR / "tables" / "fixed_y_critical_points_all.tex",
        "Full fixed-$y$ critical-point table derived from the saved notebook outputs.",
        "tab:fixed-y-all",
        ["branch", r"$K_0$", r"$y$", r"$T_c$", r"$n_c$", r"$P_c$"],
        [
            [
                BRANCH_LABELS[str(row["branch_mode"])],
                str(int(round(float(row["target_k0"])))),
                fmt_float(float(row["y"]), 6),
                fmt_float(float(row["Tc"]), 6),
                fmt_float(float(row["nc"]), 6),
                fmt_float(float(row["Pc"]), 6),
            ]
            for row in fixed_y_rows
        ],
        "l r r r r r",
    )
    write_longtable(
        REPORT_DIR / "tables" / "beta_peak_summary_all.tex",
        "Beta-equilibrium sound-speed peaks from the notebook postprocessing step.",
        "tab:beta-peak-all",
        ["branch", r"$K_0$", r"$n/n_0$ at peak", r"$f_Q$ at peak", r"$\max(v_s^2)$"],
        [
            [
                BRANCH_LABELS[str(row["branch_mode"])],
                row["K0_MeV"],
                fmt_float(float(row["n_over_n0_at_peak"]), 6),
                fmt_float(float(row["quark_fraction_at_peak"]), 6),
                fmt_float(float(row["vs2_peak"]), 6),
            ]
            for row in beta_peak_rows
        ],
        "l r r r r",
    )
    write_longtable(
        REPORT_DIR / "tables" / "neutron_star_maxima_all.tex",
        "Neutron-star maximum-mass summary extracted from the saved JSON outputs.",
        "tab:ns-max-all",
        ["branch", r"$K_0$", r"$M_{\max}$", r"$R(M_{\max})$", r"$P_c(M_{\max})$"],
        [
            [
                BRANCH_LABELS[str(row["branch_mode"])],
                str(int(round(float(row["target_k0"])))),
                fmt_float(float(row["max_mass_msun"]), 6),
                fmt_float(float(row["radius_at_max_mass_km"]), 6),
                fmt_float(float(row["central_pressure_at_max_mass_mev_fm3"]), 6),
            ]
            for row in neutron_star_rows
        ],
        "l r r r r",
    )

    readme_text = textwrap.dedent(
        f"""
        # Guided Clausius Beta-Branch Report Package

        This folder is an Overleaf-ready report built from the saved outputs of:

        - `{SOURCE_NOTEBOOK}`

        The source numerical outputs were read from:

        - `{SOURCE_DIR}`

        ## Contents

        - `main.tex`: report entry point for Overleaf
        - `figures/`: regenerated publication-style PDF and PNG figures
        - `tables/`: LaTeX tables used by `main.tex`
        - `data/raw/`: copied per-run CSV and JSON outputs from the notebook scan
        - `data/derived/`: combined machine-readable summary CSV tables

        ## Important Numerical Note

        The beta-equilibrium sound-speed figures in this package were rebuilt from
        the saved `energy_density(n_B)` curves using the same notebook-side
        postprocessing choice:

        - smoothing window = {WINDOW}
        - polynomial degree = {DEGREE}

        This matches the notebook's recomputed `v_s^2` workflow, not the original
        runner-side `v_s^2` columns at lower smoothing.

        ## Suggested Overleaf Entry Point

        Upload this whole directory or the accompanying tarball and compile:

        - `main.tex`
        """
    ).strip()
    (REPORT_DIR / "README.md").write_text(readme_text + "\n", encoding="utf-8")

    main_tex = textwrap.dedent(
        f"""
        \\documentclass[11pt]{{article}}
        \\usepackage[margin=1in]{{geometry}}
        \\usepackage{{graphicx}}
        \\usepackage{{booktabs}}
        \\usepackage{{longtable}}
        \\usepackage{{array}}
        \\usepackage{{pdflscape}}
        \\usepackage{{caption}}
        \\usepackage{{hyperref}}
        \\usepackage{{siunitx}}
        \\usepackage{{float}}
        \\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue, citecolor=blue}}
        \\sisetup{{group-separator={{,}}, group-minimum-digits=4}}

        \\title{{Clausius Beta-Branch Recompute Report}}
        \\author{{Generated from \\texttt{{{latex_escape(str(SOURCE_NOTEBOOK.relative_to(ROOT)))}}}}}
        \\date{{Generated from saved QMM outputs}}

        \\begin{{document}}
        \\maketitle

        \\begin{{abstract}}
        This package summarizes the saved outputs of the guided Clausius
        beta-branch recompute notebook. It collects the asymmetric hadronic
        couplings $(a_n, a_{{pn}}, b_n, b_{{pn}})$ for every scanned
        incompressibility $K_0$, the fixed-$y$ critical-point results
        $(T_c, n_c, P_c)$, the beta-equilibrium sound-speed peaks, and the
        corresponding raw EOS/profile tables in one clean Overleaf-ready folder.
        The copied machine-readable data are stored in \\texttt{{data/raw/}} and
        \\texttt{{data/derived/}}.
        \\end{{abstract}}

        \\section{{Source and Scope}}
        The report was built from the saved outputs of:
        \\begin{{itemize}}
        \\item \\texttt{{{latex_escape(str(SOURCE_NOTEBOOK.relative_to(ROOT)))}}}
        \\item \\texttt{{{latex_escape(str(SOURCE_DIR.relative_to(ROOT)))}}}
        \\end{{itemize}}

        The scan covers:
        \\begin{{itemize}}
        \\item Clausius mean field with the notebook's asymmetric-beta branch setup
        \\item both asymmetric branches: \\texttt{{target\\_l}} and \\texttt{{equal\\_b}}
        \\item $K_0 = 250, 255, \\ldots, 315~\\mathrm{{MeV}}$
        \\item saved beta-equilibrium profiles, fixed-$y$ profiles, hadronic EOS tables,
              and neutron-star sequences
        \\end{{itemize}}

        \\section{{Asymmetric Couplings}}
        Figure~\\ref{{fig:fit-c-vs-k0}} shows the Clausius parameter $c$, and
        Figure~\\ref{{fig:fit-params}} shows the fitted asymmetric couplings across
        the full $K_0$ scan. Table~\\ref{{tab:fit-selected}} lists selected points
        that are often used as representative cases, while the full machine table
        is included in Appendix~\\ref{{sec:appendix-fit}}.

        \\begin{{figure}}[H]
          \\centering
          \\includegraphics[width=0.62\\textwidth]{{figures/fit_c_vs_k0.pdf}}
          \\caption{{Clausius parameter $c$ as a function of the requested incompressibility $K_0$. This parameter comes from the common symmetric-matter fit, so the two branches coincide in this plot.}}
          \\label{{fig:fit-c-vs-k0}}
        \\end{{figure}}

        \\begin{{figure}}[H]
          \\centering
          \\includegraphics[width=0.96\\textwidth]{{figures/fit_parameters_vs_k0.pdf}}
          \\caption{{Asymmetric Clausius couplings as functions of the requested incompressibility $K_0$.}}
          \\label{{fig:fit-params}}
        \\end{{figure}}

        \\input{{tables/fit_parameters_selected.tex}}

        \\begin{{figure}}[H]
          \\centering
          \\includegraphics[width=0.96\\textwidth]{{figures/fit_slope_and_maxmass_vs_k0.pdf}}
          \\caption{{Auxiliary scan diagnostics: fitted symmetry slope $L$ and neutron-star maximum mass.}}
          \\label{{fig:slope-maxmass}}
        \\end{{figure}}

        \\section{{Fixed-$y$ Critical Points}}
        Figures~\\ref{{fig:fixed-y-vs-y}} and~\\ref{{fig:fixed-y-vs-k0}} summarize
        the fixed-$y$ critical-point families computed from the saved notebook
        outputs. The full table of $(K_0, y, T_c, n_c, P_c)$ values is included in
        Appendix~\\ref{{sec:appendix-fixed-y}} and in the derived CSV files.

        \\begin{{figure}}[H]
          \\centering
          \\includegraphics[width=0.95\\textwidth]{{figures/fixed_y_critical_vs_y.pdf}}
          \\caption{{Critical temperature and critical density versus fixed proton fraction $y$.}}
          \\label{{fig:fixed-y-vs-y}}
        \\end{{figure}}

        \\begin{{figure}}[H]
          \\centering
          \\includegraphics[width=0.95\\textwidth]{{figures/fixed_y_critical_vs_k0.pdf}}
          \\caption{{Critical temperature and critical density versus incompressibility $K_0$. The right panel gives the requested $K_0$ versus $n_c$ view explicitly.}}
          \\label{{fig:fixed-y-vs-k0}}
        \\end{{figure}}

        \\section{{Beta-Equilibrium Profiles and EOS}}
        Figure~\\ref{{fig:beta-selected}} shows representative beta-equilibrium
        curves for the selected $K_0 = 250, 280, 315~\\mathrm{{MeV}}$ cases, including
        the charge fraction, quark fraction, postprocessed sound speed, the
        requested $\\varepsilon / \\rho_B - m_N$ profile, and the EOS.
        The sound-speed peak summary from the notebook-side reconstruction is shown
        in Figure~\\ref{{fig:beta-peak}} and tabulated in Appendix~\\ref{{sec:appendix-beta-peak}}.

        \\begin{{figure}}[H]
          \\centering
          \\includegraphics[width=0.97\\textwidth]{{figures/beta_profiles_selected_k0.pdf}}
          \\caption{{Representative beta-equilibrium profiles, $\\varepsilon / \\rho_B - m_N$, and EOS from the saved notebook outputs. The sound-speed reconstruction uses the notebook postprocessing choices window={WINDOW}, degree={DEGREE}.}}
          \\label{{fig:beta-selected}}
        \\end{{figure}}

        \\begin{{figure}}[H]
          \\centering
          \\includegraphics[width=0.95\\textwidth]{{figures/beta_peak_summary_vs_k0.pdf}}
          \\caption{{Summary of the beta-equilibrium sound-speed peaks from the notebook postprocessing step.}}
          \\label{{fig:beta-peak}}
        \\end{{figure}}

        \\section{{Data Layout}}
        The project contains both copied raw outputs and derived summary files:
        \\begin{{itemize}}
        \\item \\texttt{{data/raw/}}: per-run CSV and JSON files copied from the source notebook output tree
        \\item \\texttt{{data/derived/fit\\_parameters\\_all.csv}}: combined asymmetric-fit table
        \\item \\texttt{{data/derived/fixed\\_y\\_critical\\_points\\_all.csv}}: combined fixed-$y$ critical-point table
        \\item \\texttt{{data/derived/beta\\_sound\\_speed\\_peak\\_summary.csv}}: peak $v_s^2$ summary
        \\item \\texttt{{data/derived/neutron\\_star\\_maxima.csv}}: neutron-star maximum-mass summary
        \\end{{itemize}}

        \\appendix

        \\section{{Full Asymmetric-Fit Table}}
        \\label{{sec:appendix-fit}}
        \\input{{tables/fit_parameters_all.tex}}

        \\section{{Full Fixed-$y$ Critical-Point Table}}
        \\label{{sec:appendix-fixed-y}}
        \\input{{tables/fixed_y_critical_points_all.tex}}

        \\section{{Beta-Equilibrium Peak Summary}}
        \\label{{sec:appendix-beta-peak}}
        \\input{{tables/beta_peak_summary_all.tex}}

        \\section{{Neutron-Star Maximum-Mass Summary}}
        \\label{{sec:appendix-ns}}
        \\input{{tables/neutron_star_maxima_all.tex}}

        \\end{{document}}
        """
    ).strip()
    (REPORT_DIR / "main.tex").write_text(main_tex + "\n", encoding="utf-8")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for source_path in sorted(REPORT_DIR.rglob("*")):
            if source_path.is_dir():
                continue
            arcname = Path(REPORT_DIR.name) / source_path.relative_to(REPORT_DIR)
            handle.write(source_path, arcname.as_posix())

    print(f"Report directory: {REPORT_DIR}")
    print(f"Zip archive: {ZIP_PATH}")


if __name__ == "__main__":
    build_report()
