"""Plot saved Lambda300 observables against alpha/c with calibrated K0 bands.

Run with .venv/bin/python scripts/plot_empirical_parameter_bands.py.
Only the saturation fits at the band edges are recomputed, not beta/TOV curves.
"""
from pathlib import Path
from dataclasses import replace
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'qmm_mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq
from qmm.config import load_run_config
from qmm.ground_state import compute_ground_state_point_explicit
from qmm.models import MODELS, get_model

OUTPUT = ROOT / 'Paper/empirical_K0_250_315_parameter_bands'
SOURCE = ROOT / 'Paper/Clausius_Dieterici_3EV_Lambda300_K0_200_750_step50_combined/tables'
LIMITS = (250.0, 315.0)
COLORS = {'vdw': '#333333', 'cs': '#0072B2', 'tvm': '#D55E00'}
BRANCHES = {'target_l': ('-', r'$b_{pn}\ne b_n$'), 'equal_b': ('--', r'$b_{pn}=b_n$')}


def register_models():
    for ev in ('cs', 'tvm'):
        repulsive = get_model('clausius_' + ev)
        MODELS['dieterici_' + ev] = replace(
            get_model('dieterici'), name='dieterici_' + ev,
            **{name: getattr(repulsive, name) for name in (
                'nid_from_n', 'n_from_nid', 'volume_fraction',
                'species_volume_fraction', 'pressure_prefactor')})


def calibrate_bands(results):
    rows = []
    for key, data in results.groupby('model_key', sort=True):
        mapping = data[['K0_MeV', 'parameter_value']].drop_duplicates().sort_values('K0_MeV')
        if mapping.K0_MeV.duplicated().any():
            raise ValueError(f'Inconsistent branch parameter mapping: {key}')
        config = load_run_config(ROOT / 'examples/generated' /
            f'guided_{key}_lambda300_k0_200_750_step50_k0_250_target_l.json')
        def ground(parameter):
            point = compute_ground_state_point_explicit(
                config.model_name, float(parameter), config.physical, config.ground_state)
            if point is None:
                raise ValueError(f'Ground-state fit failed: {key}, {parameter}')
            return point
        edge_values = []
        for target in LIMITS:
            # Bracket with neighboring saved fits; solve the boundary directly.
            lower = mapping[mapping.K0_MeV < target].iloc[-1]
            upper = mapping[mapping.K0_MeV > target].iloc[0]
            a, b = sorted((lower.parameter_value, upper.parameter_value))
            parameter = brentq(lambda x: ground(x).K0-target, a, b, xtol=1e-10)
            point = ground(parameter)
            if abs(point.K0-target) > 0.01:
                raise ValueError(f'Boundary residual too large: {key}, {target}')
            edge_values.append(parameter)
            rows.append(dict(model_key=key, parameter='alpha' if key.startswith('dieterici') else 'c',
                target_K0_MeV=target, fitted_K0_MeV=point.K0, parameter_value=parameter,
                a=point.a, b=point.b, n_sat_fm3=point.n_sat,
                binding_MeV=point.binding))
        print(key, 'parameter band:', *sorted(edge_values), flush=True)
    return pd.DataFrame(rows)


def save(fig, stem):
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(OUTPUT / 'figures' / f'{stem}.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def band_axis(ax, bounds, ev):
    low, high = sorted(bounds.parameter_value)
    ax.axvspan(low, high, color=COLORS[ev], alpha=.14, zorder=0)
    for edge in (low, high):
        ax.axvline(edge, color=COLORS[ev], lw=.8, alpha=.7)
    pad = (high-low)*.08
    ax.set_xlim(low-pad, high+pad)
    ax.grid(alpha=.2)


def main():
    register_models()
    for folder in ('figures', 'tables'):
        (OUTPUT / folder).mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(SOURCE / 'six_models_lambda300_plot_results.csv')
    bounds = calibrate_bands(results)
    bounds.to_csv(OUTPUT / 'tables/empirical_parameter_boundaries.csv', index=False)
    selected = results[results.K0_MeV.between(*LIMITS)].copy()
    selected.to_csv(OUTPUT / 'tables/empirical_saved_observables.csv', index=False)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'mathtext.fontset': 'stix'})
    fields = [('maximum_mass_msun', r'$M_{\max}$ [$M_\odot$]'),
              ('radius_at_maximum_mass_km', r'$R_{M_{\max}}$ [km]'),
              ('maximum_vs2_beta', r'$\max c_s^2$')]
    for family, xlabel in [('dieterici', r'$\alpha$'), ('clausius', r'$c$ [fm$^3$]')]:
        fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
        for row, ev in enumerate(COLORS):
            key = family + '_' + ev
            edges = bounds[bounds.model_key == key]
            for ax, (field, ylabel) in zip(axes[row], fields):
                band_axis(ax, edges, ev)
                for branch, (style, label) in BRANCHES.items():
                    part = selected[(selected.model_key == key) & (selected.branch_mode == branch)].sort_values('parameter_value')
                    ax.plot(part.parameter_value, part[field], style, marker='o', color=COLORS[ev], label=label)
                ax.set(xlabel=xlabel, ylabel=ylabel, title=f'{family.title()}–{ev.upper()}')
            axes[row, 0].legend(fontsize=9)
        fig.suptitle(f'{family.title()}: empirical parameter bands, '+r'$250\leq K_0\leq315$ MeV, $\Lambda=300$ MeV'
                     +'\nShading: calibrated bounds. Markers: saved K₀ = 250, 300 MeV; lines guide the eye.', fontsize=12)
        save(fig, f'{family}_empirical_observables')
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row, family in enumerate(('dieterici', 'clausius')):
        for col, ev in enumerate(COLORS):
            key = family + '_' + ev
            edges = bounds[bounds.model_key == key]
            ax = axes[row, col]
            band_axis(ax, edges, ev)
            data = results[results.model_key == key][['parameter_value', 'K0_MeV']].drop_duplicates()
            data = data[data.K0_MeV.between(*LIMITS)]
            ax.scatter(data.parameter_value, data.K0_MeV, color=COLORS[ev], label='Saved fit')
            ax.scatter(edges.parameter_value, edges.fitted_K0_MeV, marker='x', color='black', label='Solved boundary')
            ax.axhline(250, color='.5', ls=':', lw=.8)
            ax.axhline(315, color='.5', ls=':', lw=.8)
            ax.set(xlabel=r'$\alpha$' if family == 'dieterici' else r'$c$ [fm$^3$]',
                   ylabel=r'$K_0$ [MeV]', ylim=(242, 323), title=f'{family.title()}–{ev.upper()}')
            ax.legend(fontsize=8)
    fig.suptitle('Empirical K₀ interval mapped to each interaction parameter', fontsize=14)
    save(fig, 'empirical_parameter_mapping')
    (OUTPUT / 'README.md').write_text(
        '# Empirical parameter bands\n\n'
        'The shaded intervals map the user-selected 250–315 MeV K0 constraint to alpha '
        '(dimensionless) for Dieterici and c (fm³) for Clausius, separately for VDW, CS and TVM. '
        'Both endpoints are solved with the QMM saturation fit and the original scan settings; '
        'they are not interpolated from the step50 grid.\n\n'
        'Observable panels use only saved K0=250 and 300 MeV results at Lambda=300 MeV. '
        'Connecting lines guide the eye; the shaded region is an allowed parameter interval, '
        'not an uncertainty envelope for the observables. No beta/TOV calculation at 315 MeV '
        'or dense empirical scan has been performed. NaN observables remain gaps.\n\n'
        'Reproduce with `.venv/bin/python scripts/plot_empirical_parameter_bands.py` '
        'or the companion `notebooks/03_six_models_paper/empirical_parameter_bands.ipynb`.\n')
    return bounds


if __name__ == '__main__':
    main()
