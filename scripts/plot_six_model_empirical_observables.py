"""Plot the saved symmetric empirical scan; no physics is recomputed.

Run: .venv/bin/python scripts/plot_six_model_empirical_observables.py
"""
from pathlib import Path
import os
import argparse
import json
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'qmm_mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'Paper/empirical_critical_K0_250_315/critical_points.csv'
OUTPUT = ROOT / 'Paper/six_model_empirical_observables'
MODELS = {
    'clausius_vdw': ('Clausius–VDW', '#e600cc', 's', '-'),
    'clausius_cs': ('Clausius–CS', '#f078aa', 'o', '-'),
    'clausius_tvm': ('Clausius–TVM', '#9e205f', '^', '-'),
    'dieterici_vdw': ('Dieterici–VDW', '#008000', 's', '-'),
    'dieterici_cs': ('Dieterici–CS', '#70b85c', 'o', '-'),
    'dieterici_tvm': ('Dieterici–TVM', '#005b40', '^', '-'),
}
FIELDS = {
    'Tc_MeV': ('critical_temperature_vs_K0', r'$T_c$ [MeV]'),
    'nc_fm3': ('critical_density_vs_K0', r'$n_c$ [fm$^{-3}$]'),
    'Pc_MeV_fm3': ('critical_pressure_vs_K0', r'$P_c$ [MeV fm$^{-3}$]'),
}


def draw(ax, frame, field, annotate_endpoints=True, branch_style=None):
    for key, (label, color, marker, style) in MODELS.items():
        part = frame[frame.model_key.eq(key)].set_index('K0_target_MeV')
        # Missing or failed solves break curves instead of being interpolated.
        part = part.reindex(sorted(frame.K0_target_MeV.unique()))
        values = part[field].where(part.status.eq('ok'))
        ax.plot(part.index, values, color=color, linestyle=branch_style or style,
                linewidth=1.4, label=label)
    high = frame.K0_target_MeV.max()
    pad = (high-250)*.045
    ax.set(xlabel=r'$K_0$ [MeV]', ylabel=FIELDS[field][1], xlim=(250-pad, high+pad))
    ax.set_xticks([250, 260, 270, 280, 290, 300, 310] if high <= 315 else [250, 350, 450, 550, 650, 750])
    if high > 750:
        ax.set_xticks([250, 350, 450, 550, 650, high])
    if high > 315 and annotate_endpoints:
        endpoints = []
        for key in ('clausius_vdw', 'dieterici_vdw'):
            part = frame[frame.model_key.eq(key) & frame.status.eq('ok')].sort_values('K0_target_MeV')
            start, end = part.iloc[0], part.iloc[-1]
            color = MODELS[key][1]
            family = key.split('_')[0].title()
            parameter = (rf'$c={start.parameter_value:.2f}$ fm$^3$' if key.startswith('clausius')
                         else rf'$\alpha={start.parameter_value:.3f}$')
            ax.plot(start.K0_target_MeV, start[field], 's', color=color, ms=3.8, zorder=6)
            above = field == 'Pc_MeV_fm3' and key == 'clausius_vdw'
            ax.annotate(f'{family}\n{parameter}', xy=(start.K0_target_MeV, start[field]),
                        xytext=(0, 9 if above else -9), textcoords='offset points', ha='left', va='bottom' if above else 'top',
                        fontsize=8, color=color)
            ax.plot(end.K0_target_MeV, end[field], 's', color='black', ms=3, zorder=6)
            endpoints.append(end[field])
        ax.annotate('VDW', xy=(high, min(endpoints)), xytext=(-2, -10),
                    textcoords='offset points', ha='right', va='top', fontsize=8, color='black')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.margins(y=.25)


def save(fig, name):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUTPUT / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extend-to-750', action='store_true')
    args = parser.parse_args()
    frame = pd.read_csv(SOURCE)
    frame = frame[frame.K0_target_MeV.between(250, 315)].copy()
    frame['source'] = str(SOURCE.relative_to(ROOT))
    if args.extend_to_750:
        OUTPUT = ROOT / 'Paper/six_model_observables_K0_250_750'
        source = ROOT / 'Paper/Clausius_Dieterici_3EV_Lambda300_K0_200_750_step50_combined/tables/six_models_lambda300_fixed_y_critical_points.csv'
        saved = pd.read_csv(source)
        saved = saved[saved.y.eq(.5) & saved.branch_mode.eq('equal_b') & saved.K0_MeV.gt(315)].copy()
        saved['K0_MeV'] = saved.K0_MeV.astype(float)
        saved['K0_target_MeV'] = saved.K0_MeV
        saved['source'] = str(source.relative_to(ROOT))
        for i, row in saved[saved.status.ne('ok')].iterrows():
            recovered = OUTPUT / 'recovered' / f'{row.model_key}_K0_{row.K0_MeV:g}.json'
            if recovered.exists():
                result = json.loads(recovered.read_text())
                for field, value in result.items():
                    saved.loc[i, field] = value
                saved.loc[i, 'source'] = str(recovered.relative_to(ROOT))
        frame = pd.concat([frame, saved], ignore_index=True).sort_values(['model_key', 'K0_target_MeV'])
    if set(frame.model_key) != set(MODELS):
        raise ValueError('Expected all six empirical models')
    if frame.duplicated(['model_key', 'K0_target_MeV']).any():
        raise ValueError('Duplicate model/K0 samples')
    if not np.allclose(frame.y, .5):
        raise ValueError('Expected symmetric matter (y=0.5)')
    good = frame.status.eq('ok')
    if not np.isfinite(frame.loc[good, list(FIELDS)].to_numpy()).all():
        raise ValueError('Nonfinite observable in a successful solve')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / 'plotted_critical_points.csv', index=False)
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix',
                         'font.size': 11, 'axes.linewidth': .8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    for field, (stem, _) in FIELDS.items():
        fig, ax = plt.subplots(figsize=(5.4, 4.5), layout='constrained')
        draw(ax, frame, field)
        ax.legend(loc='best', frameon=False, fontsize=8.5, ncol=2)
        save(fig, stem)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for ax, field, tag in zip(axes, FIELDS, ('(a)', '(b)', '(c)')):
        draw(ax, frame, field)
        ax.set_title(tag, loc='left', fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, frameon=False,
               fontsize=10, bbox_to_anchor=(.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, .90))
    save(fig, 'six_model_critical_observables')
    (OUTPUT / 'README.md').write_text(
        '# Six-model empirical critical observables\n\n'
        'Symmetric hadronic matter (y = 0.5), K0 = 250–315 MeV. '
        'Each panel overlays Clausius and Dieterici with VDW, CS, and TVM repulsion. '
        'Tc, nc, and Pc are plotted against the target saturation incompressibility. '
        'Straight segments connect adjacent successful '
        '5-MeV samples. Missing or failed samples break the curves. No extrapolation, '
        'smoothing, or new physics calculation is used.\n\n'
        'Source: `../empirical_critical_K0_250_315/critical_points.csv`; '
        'the exact plotted table is copied to `plotted_critical_points.csv`. '
        'The source README and resolution_check.json document numerical validation. '
        'Lambda and the asymmetric branches do not enter these symmetric observables.\n\n'
        'The supplied reference informs the visual style. Its experimental rectangle '
        'is omitted because numerical bounds and a source were not specified; the '
        'selected K0 range is not a joint experimental constraint on Tc, nc, or Pc.\n\n'
        'Outputs: three individual figures and a combined three-panel figure, each '
        'in PNG (300 dpi), PDF, and SVG. Reproduce from the repository root with '
        '`.venv/bin/python scripts/plot_six_model_empirical_observables.py`.\n'
    )
    if args.extend_to_750:
        (OUTPUT / 'README.md').write_text(
            '# Six-model symmetric critical observables, K0 = 250–750 MeV\n\n'
            'Tc, nc and Pc for symmetric hadronic matter (y=0.5), not beta equilibrium. '
            'Colored squares and parameter labels mark the VDW-family starts at K0=250 MeV; black squares mark their K0=750 MeV endpoints. No shaded band is drawn; '
            'the rest of the plotted domain extends beyond that interval.\n\n'
            'The dense empirical scan supplies 250–315 MeV at 5-MeV spacing. '
            'Saved six-model symmetric critical points supply 350–750 MeV at 50-MeV spacing. '
            'The equal_b table is used at y=0.5, where both asymmetric branches coincide '
            '(saved Tc differences below 3e-10 MeV). The two source scans agree at '
            'K0=250 and 300 to within 9e-6 MeV in Tc, 1e-8 fm^-3 in nc, and '
            '6e-7 MeV fm^-3 in Pc. Five failed saved points were retried with '
            'compute_empirical_critical_scan.solve_point; successful results are in recovered/. '
            'plotted_critical_points.csv records per-row provenance.\n\n'
            'Lines connect actual samples, with gaps for failed/missing results; '
            'there is no extrapolation. The original scan permits negative c for '
            'Clausius–CS (K0>=650) and Clausius–TVM (K0>=600). These points continue '
            'that model parameterization, rather than imposing c>=0. No experimental '
            'Tc/nc/Pc constraint is drawn.\n\n'
            'PNG, PDF, and SVG are provided for each observable and the three-panel figure. '
            'Replot: `.venv/bin/python scripts/plot_six_model_empirical_observables.py --extend-to-750`.\n'
        )
    print(f'Plotted {good.sum()} successful points across {frame.model_key.nunique()} models: {OUTPUT}')


if __name__ == '__main__':
    main()
