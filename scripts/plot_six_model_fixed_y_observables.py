"""Replot saved fixed-proton-fraction critical points for both coupling branches."""
from pathlib import Path
import numpy as np
import pandas as pd
import plot_six_model_empirical_observables as style
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'Paper/six_model_fixed_y_observables_K0_250_750'
TABLES = ROOT / 'Paper/Clausius_Dieterici_3EV_Lambda300_K0_200_750_step50_combined/tables'
BRANCHES = {'equal_b': r'$b_{pn}=b_n$', 'target_l': r'$b_{pn}\ne b_n$'}


def load_data():
    source = TABLES / 'six_models_lambda300_fixed_y_critical_points.csv'
    data = pd.read_csv(source)
    data['source_file'] = str(source.relative_to(ROOT))
    parameters = pd.read_csv(TABLES / 'six_models_lambda300_plot_results.csv')
    data = data.merge(parameters[['model_key', 'K0_MeV', 'branch_mode', 'parameter_value']],
                      on=['model_key', 'K0_MeV', 'branch_mode'], validate='many_to_one')
    # Prefer the newer converged empirical calculations, including K0=315.
    empirical_source = ROOT / 'Paper/empirical_asymmetric_lambda300/fixed_y_critical_points.csv'
    empirical = pd.read_csv(empirical_source)
    empirical = empirical[empirical.status.eq('ok')].copy()
    empirical['source_file'] = str(empirical_source.relative_to(ROOT))
    data = pd.concat([data, empirical], ignore_index=True)
    keys = ['model_key', 'K0_MeV', 'branch_mode', 'y']
    data = data.drop_duplicates(keys, keep='last')
    data = data[data.K0_MeV.between(250, 750) & data.y.lt(.5)].copy()
    data['K0_target_MeV'] = data.K0_MeV
    symmetric = pd.read_csv(ROOT / 'Paper/six_model_observables_K0_250_750/plotted_critical_points.csv')
    symmetric['source_file'] = symmetric['source']
    rows = [data]
    for branch in BRANCHES:
        part = symmetric.copy()
        part['branch_mode'] = branch
        rows.append(part)
    result = pd.concat(rows, ignore_index=True)
    recovery_file = OUTPUT / 'recovery/recovered_points.csv'
    if recovery_file.exists():
        recovered = pd.read_csv(recovery_file)
        recovered = recovered[recovered.status.eq('ok')].copy()
        recovered['source_file'] = str(recovery_file.relative_to(ROOT))
        result = pd.concat([result, recovered], ignore_index=True).drop_duplicates(
            ['model_key', 'K0_target_MeV', 'branch_mode', 'y'], keep='last')
    assert not result.duplicated(['model_key', 'K0_target_MeV', 'branch_mode', 'y']).any()
    assert np.isfinite(result.loc[result.status.eq('ok'), list(style.FIELDS)].to_numpy()).all()
    return result


def endpoints(ax, data, field, xfield="K0_target_MeV"):
    ends = []
    high = float(data.K0_target_MeV.max())
    for key in ('clausius_vdw', 'dieterici_vdw'):
        part = data[data.model_key.eq(key) & data.status.eq('ok') & data.branch_mode.eq('equal_b')]
        first = part[part.K0_target_MeV.eq(250)]
        color = style.MODELS[key][1]
        if not first.empty:
            start = first.iloc[0]
            parameter = (rf'$c={start.parameter_value:.2f}$ fm$^3$' if key.startswith('clausius')
                         else rf'$\alpha={start.parameter_value:.3f}$')
            above = xfield == 'K0_target_MeV' and key == 'clausius_vdw' and (field == 'Pc_MeV_fm3' or float(start.y) < .5)
            ax.plot(start[xfield], start[field], 's', color=color, ms=3.8, zorder=6)
            label_y = start[field]
            if key == 'dieterici_vdw' and xfield == 'K0_target_MeV':
                starts = data[data.model_key.eq(key) & data.K0_target_MeV.eq(250) & data.status.eq('ok')]
                label_y = starts[field].min()
            ax.annotate(key.split('_')[0].title()+'\n'+parameter, (start[xfield], label_y),
                        xytext=(0, 9 if above else -9), textcoords='offset points',
                        ha='left', va='bottom' if above else 'top', fontsize=8, color=color)
        last = part[part.K0_target_MeV.eq(high)]
        if not last.empty:
            value = last.iloc[0][field]
            ax.plot(last.iloc[0][xfield], value, 's', color='black', ms=3, zorder=6)
            ends.append((value, last.iloc[0][xfield]))
    # Asymmetric VDW curves need not meet: label each distinct endpoint.
    gap = .07 * np.ptp(ax.get_ylim())
    for i, (value, xpos) in enumerate(sorted(ends)):
        if i and abs(value-min(ends)[0]) < gap:
            continue
        ax.annotate('VDW', (xpos, value), xytext=(-2, -10), textcoords='offset points',
                    ha='right', va='top', fontsize=8)


def main():
    data = load_data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT / 'plotted_critical_points.csv', index=False)
    coverage = data.groupby(['branch_mode', 'y', 'model_key']).agg(
        saved_points=('status', 'size'), successful_points=('status', lambda s: s.eq('ok').sum()))
    coverage.to_csv(OUTPUT / 'coverage.csv')
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix',
                         'font.size': 11, 'axes.linewidth': .8, 'pdf.fonttype': 42})
    directory = OUTPUT / ('combined_branches_to760' if data.K0_target_MeV.max() > 750 else 'combined_branches')
    directory.mkdir(exist_ok=True)
    with PdfPages(directory / 'all_five_proton_fractions.pdf') as pdf:
        for y in (.1, .2, .3, .4, .5):
            part = data[np.isclose(data.y, y)]
            fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
            branches = [('equal_b', '-')] if y == .5 else [('equal_b', '-'), ('target_l', ':')]
            for ax, field in zip(axes, ('Tc_MeV', 'nc_fm3')):
                for branch, line in branches:
                    selected = part[part.branch_mode.eq(branch)]
                    assert set(selected.model_key) == set(style.MODELS)
                    style.draw(ax, selected, field, annotate_endpoints=False, branch_style=line)
                endpoints(ax, part, field)
            axes[0].text(.97, .97, rf'$y={y:.1f}$', transform=axes[0].transAxes,
                         ha='right', va='top', fontsize=12)
            handles = [Line2D([], [], color=spec[1], lw=1.6, label=spec[0])
                       for spec in style.MODELS.values()]
            if y != .5:
                handles += [Line2D([], [], color='black', ls=line, lw=1.5, label=BRANCHES[branch])
                            for branch, line in branches]
            fig.legend(handles=handles, loc='upper center', ncol=3 if y == .5 else 4,
                       frameon=False, fontsize=9, bbox_to_anchor=(.5, 1.0),
                       columnspacing=1.5, handlelength=2.0, labelspacing=.35)
            fig.subplots_adjust(left=.085, right=.985, bottom=.16, top=.82, wspace=.30)
            for ext in ('png', 'pdf', 'svg'):
                fig.savefig(directory / f'critical_observables_y_{y:.1f}.{ext}', dpi=300, bbox_inches='tight')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print('Saved five figures with both branches overlaid', flush=True)
    (OUTPUT / 'README.md').write_text(
        '# Six-model fixed-y critical observables\n\n'
        'Five combined-branch figures: y=0.1, 0.2, 0.3, 0.4, 0.5. Current extended figures are in combined_branches_to760/. Each figure contains '
        'Tc versus K0 and nc versus K0, using K0=250–760 MeV. equal_b uses b_pn=b_n; target_l fits '
        'L=58.9 MeV with b_pn!=b_n. J=32.5 MeV. These are fixed-composition hadronic '
        'liquid–gas critical points, not beta-equilibrium results. Both sets share '
        'the same symmetric y=0.5 limit.\n\n'
        'Saved step50 results are supplemented by successful newer empirical calculations '
        'at 250, 300, and 315 MeV. y=0.5 reuses the validated extended symmetric table, '
        'including its five recovered points and 5-MeV empirical grid. Per-row sources '
        'are recorded in plotted_critical_points.csv. The recovery/ directory contains new continuation-based solves of the previously failed samples and all 60 endpoints at K0=760 MeV. '
        'Lines join adjacent saved samples; failed samples produce gaps and are never '
        'extrapolated. In the earlier input tables, several high-K0 Dieterici y=0.1 points did not '
        'converge; retries recovered these roots. The earlier failures did not establish absence of a physical critical point. '
        'coverage.csv reports all six models even when their curves end early.\n\n'
        'Pink shades represent Clausius and green shades Dieterici. Solid lines are equal_b; dotted lines are target_l for every model. The coincident symmetric branches at y=0.5 are drawn only once, with no redundant branch legend. One shared legend serves both panels. Endpoint annotations refer to the solid equal_b branch. Only VDW-family '
        'starting points at 250 have colored square markers and parameter labels; '
        'successful endpoints at the largest plotted K0 have black squares. Asymmetric VDW endpoints '
        'need not coincide. No shaded range or experimental constraint is drawn. '
        'The inherited high-K0 Clausius-CS/TVM fits include negative c.\n\n'
        'combined_branches_to760/ contains five PNG/PDF/SVG figures and a five-page PDF. Earlier exports are retained in combined_branches/, equal_b/, and target_l/. '
        'Reproduce with `.venv/bin/python scripts/plot_six_model_fixed_y_observables.py`.\n'
    )


if __name__ == '__main__':
    main()
