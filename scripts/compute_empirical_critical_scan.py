"""Resumable symmetric liquid-gas critical scan in the empirical K0 interval."""
from pathlib import Path
from dataclasses import asdict, replace
import hashlib
import json
import sys
import os
import tempfile

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / 'src', ROOT / 'scripts'):
    sys.path.insert(0, str(directory))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'qmm_mpl_cache'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq, least_squares
from qmm.constants import DEFAULT_QUANTUM_SETTINGS
from qmm.config import load_run_config
from qmm.ground_state import compute_ground_state_point_explicit
from qmm.quantum import QuantumPressureEvaluator
from plot_empirical_parameter_bands import register_models

OUTPUT = ROOT / 'Paper/empirical_critical_K0_250_315'
MODELS = [f'{family}_{ev}' for family in ('dieterici', 'clausius') for ev in ('vdw', 'cs', 'tvm')]
SETTINGS = replace(DEFAULT_QUANTUM_SETTINGS, n_k_fd=800, k_max_fd=5.,
    mu_scf_tol=1e-11, mu_scf_max_iter=200, mu_delta=.1,
    cache_round_digits=12, t_min_cp=1., t_max_cp=30., n_min_cp=.005, n_max_cp=.12)
COLORS = {'dieterici': '#228833', 'clausius': '#cc33cc'}
MARKERS = {'vdw': 'o', 'cs': 's', 'tvm': '^'}


def solve_point(key, k0):
    register_models()
    config = load_run_config(ROOT / 'examples/generated' /
        f'guided_{key}_lambda300_k0_200_750_step50_k0_250_target_l.json')
    saved = pd.read_csv(ROOT / 'Paper/Clausius_Dieterici_3EV_Lambda300_K0_200_750_step50_combined/tables/six_models_lambda300_plot_results.csv')
    mapping = saved[saved.model_key == key][['K0_MeV', 'parameter_value']].drop_duplicates().sort_values('K0_MeV')
    lower = mapping[mapping.K0_MeV < k0].iloc[-1].parameter_value
    upper = mapping[mapping.K0_MeV > k0].iloc[0].parameter_value
    def ground(parameter):
        point = compute_ground_state_point_explicit(config.model_name, parameter, config.physical, config.ground_state)
        if point is None:
            raise ValueError(f'No saturation fit: {key}, {parameter}')
        return point
    parameter = brentq(lambda x: ground(x).K0-k0, *sorted((lower, upper)), xtol=1e-10)
    gs = ground(parameter)
    evaluator = QuantumPressureEvaluator(gs, physical=config.physical, settings=SETTINGS)
    def residual(values):
        d1, d2, _ = evaluator.critical_equations(*map(float, values))
        if d1 is None or d2 is None or not np.isfinite(d1+d2):
            return np.array([1e4, 1e4])
        return np.array([d1/10., d2/1000.])
    best = None
    for seed in ((15., .055), (12., .045), (19., .065)):
        solved = least_squares(residual, seed, bounds=([1., .006], [30., .119]),
            x_scale=[5., .04], diff_step=[1e-5, 1e-5],
            xtol=3e-11, ftol=3e-11, gtol=3e-11, max_nfev=250)
        d1, d2 = residual(solved.x)*[10., 1000.]
        if abs(d1)<1e-4 and abs(d2)<1e-3:
            best = solved
            break
    if best is None:
        raise RuntimeError(f'Critical equations did not converge: dPdn={d1}, d2Pdn2={d2}')
    tc, nc = map(float, best.x)
    pc, _ = evaluator.pressure(tc, nc)
    if abs(gs.K0-k0)>.01 or pc is None or pc<=0:
        raise RuntimeError('Invalid ground-state residual or critical pressure')
    return dict(model_key=key, K0_target_MeV=float(k0), K0_MeV=gs.K0,
        parameter_name=gs.parameter_name, parameter_value=parameter, y=.5,
        Tc_MeV=tc, nc_fm3=nc, Pc_MeV_fm3=pc,
        dPdn=float(d1), d2Pdn2=float(d2), a=gs.a, b=gs.b,
        status='ok')


def run_scan(k0_values=range(250, 316, 5), models=MODELS, force=False):
    """Save each converged point atomically; settings/source changes invalidate caches."""
    values = list(map(float, k0_values))
    if not values or any(k<250 or k>315 for k in values):
        raise ValueError('Choose K0 values within 250–315 MeV.')
    cache = OUTPUT / 'checkpoints'
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for source in sorted((ROOT/'src/qmm').glob('*.py')):
        digest.update(source.read_bytes())
    digest.update(json.dumps(asdict(SETTINGS), sort_keys=True).encode())
    rows = []
    for key in models:
        if key not in MODELS:
            raise ValueError(f'Unknown model {key}')
        config_path=ROOT/'examples/generated'/f'guided_{key}_lambda300_k0_200_750_step50_k0_250_target_l.json'
        signature=hashlib.sha256(digest.digest()+config_path.read_bytes()).hexdigest()
        for k0 in values:
            path = cache / f'{key}_K0_{k0:g}.json'
            row = json.loads(path.read_text()) if path.exists() and not force else {}
            if row.get('signature') != signature or row.get('status') != 'ok':
                try:
                    row = solve_point(key, k0)
                except Exception as exc:
                    row = dict(model_key=key, K0_target_MeV=k0, status='failed', error=repr(exc))
                row['signature'] = signature
                temporary=path.with_suffix('.tmp')
                temporary.write_text(json.dumps(row, indent=2)+'\n')
                temporary.replace(path)
            rows.append(row)
            print(f'{key} K0={k0:g}: {row["status"]}', flush=True)
    frame = pd.DataFrame(rows).sort_values(['model_key', 'K0_target_MeV'])
    frame.to_csv(OUTPUT / 'critical_points.csv', index=False)
    return frame


def plot_scan(frame):
    """Plot actual computed points; failed values break curves rather than being bridged."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame=frame.copy()
    for field in ('parameter_value', 'Tc_MeV', 'nc_fm3'):
        if field not in frame: frame[field]=np.nan
        frame.loc[frame.status!='ok', field]=np.nan
    paths=[]
    def save(fig, name):
        for ext in ('png','pdf','svg'):
            path=OUTPUT/f'{name}.{ext}'
            fig.savefig(path,dpi=180,bbox_inches='tight')
            if ext=='png': paths.append(path)
        plt.close(fig)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':10})
    fig, axes=plt.subplots(3,2,figsize=(10,11),constrained_layout=True)
    for row,ev in enumerate(MARKERS):
        for family,color in COLORS.items():
            data=frame[frame.model_key==family+'_'+ev].sort_values('K0_target_MeV')
            if data.empty: continue
            axes[row,0].plot(data.K0_target_MeV,data.nc_fm3,'-o',ms=3,color=color,label=family.title())
            axes[row,1].plot(data.Tc_MeV,data.nc_fm3,'-o',ms=3,color=color,label=family.title())
            valid=data[data.status=='ok']
            if len(valid)==len(data) and {250.,315.}.issubset(set(valid.K0_target_MeV)):
                axes[row,1].axvspan(valid.Tc_MeV.min(),valid.Tc_MeV.max(),color=color,alpha=.09)
        axes[row,0].axvspan(250,315,color='.5',alpha=.18,label=r'$250\leq K_0\leq315$ MeV')
        axes[row,0].set(xlabel=r'$K_0$ [MeV]',xlim=(246,319))
        axes[row,1].set_xlabel(r'$T_c$ [MeV]')
        for ax in axes[row]:
            ax.set(ylabel=r'$n_c$ [fm$^{-3}$]',title=ev.upper())
            ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('Symmetric matter: computed liquid–gas critical points, y = 0.5\n'
                 'Gray: selected K₀ interval. Colored T_c bands: model predictions, not experimental T_c bounds.',fontsize=11)
    save(fig,'critical_correlations_empirical')
    fig, axes=plt.subplots(3,2,figsize=(10,11),constrained_layout=True)
    for col,(family,color) in enumerate(COLORS.items()):
        for row,ev in enumerate(MARKERS):
            data=frame[frame.model_key==family+'_'+ev].sort_values('K0_target_MeV')
            ax=axes[row,col]
            ax.plot(data.parameter_value,data.nc_fm3,'-o',ms=3,color=color)
            valid=data[data.status=='ok']
            if len(valid)==len(data) and {250.,315.}.issubset(set(valid.K0_target_MeV)):
                ax.axvspan(valid.parameter_value.min(),valid.parameter_value.max(),color=color,alpha=.12)
            ax.set(xlabel=r'$\alpha$' if family=='dieterici' else r'$c$ [fm$^3$]',
                ylabel=r'$n_c$ [fm$^{-3}$]',title=f'{family.title()}–{ev.upper()}')
            ax.grid(alpha=.15)
    fig.suptitle('Critical density versus interaction parameter\nShading: parameter range selected by 250 ≤ K₀ ≤ 315 MeV; markers: computed points',fontsize=11)
    save(fig,'critical_density_vs_alpha_c')
    return paths


if __name__=='__main__':
    data=run_scan()
    plot_scan(data)
    if (data.status!='ok').any():
        raise SystemExit('Some points failed; inspect critical_points.csv and rerun to retry.')
