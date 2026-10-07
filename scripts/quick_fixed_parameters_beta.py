"""Low-resolution smoke run of fixed-parameter six-model beta EOS and stars.

Run from QMM: .venv/bin/python scripts/quick_fixed_parameters_beta.py
Separate caches/figures under Paper/TEST_fixed_parameters_beta; no production writes.
Coarse peaks and stellar masses are exploratory, not convergence-tested results.
"""
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'qmm_mpl_cache'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse

from pathlib import Path
from dataclasses import asdict, replace
from types import MethodType
import sys, json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
def display(value):
    print(value.to_string(index=False) if hasattr(value, "to_string") else value)
ROOT=Path(__file__).resolve().parents[1]
if ROOT is None: raise RuntimeError('Open inside QMM.')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'src/TOVsolver')]
import empirical_asymmetric_suite as suite
from qmm.config import load_run_config
from qmm.ground_state import compute_ground_state_point
from qmm.asymmetry import compute_asymmetric_fit
from qmm.quarkyonic import AsymmetricQuarkyonicEOS, compute_symmetric_quarkyonic_curve
from qmm.numerics import golden_section_min_safe
from qmm.sound_speed import reconstruct_sound_speed_curve
suite.register()
MODELS={'clausius_vdw':('Clausius–VDW','#e600cc'),'clausius_cs':('Clausius–CS','#f078aa'),
 'clausius_tvm':('Clausius–TVM','#9e205f'),'dieterici_vdw':('Dieterici–VDW','#008000'),
 'dieterici_cs':('Dieterici–CS','#70b85c'),'dieterici_tvm':('Dieterici–TVM','#005b40')}
OUT=ROOT/'Paper/TEST_fixed_parameters_beta'



def global_beta(self,n_b,fq_min=0.,fq_max=1.,y_scan=None):
    lower=max(fq_min,self.beta_lower_quark_fraction_bound(n_b))
    if lower>=fq_max:return np.nan,np.nan
    (fq,energy),intervals=self._coarse_beta_candidates(n_b,lower,fq_max,y_scan)
    for left,right in intervals:
        f,e=golden_section_min_safe(lambda q:self.total_energy_beta(q,n_b,y_scan=y_scan),
            left,right,tol=self.settings.refine_tol,max_iter=self.settings.refine_max_iter)
        if np.isfinite(e) and e<energy:fq,energy=f,e
    return fq,energy


def finish_thermodynamics(frame):
    thermo=reconstruct_sound_speed_curve(frame.n_b.tolist(),frame.energy_density.tolist(),WINDOW,DEGREE)
    frame=frame.copy()
    frame['pressure']=thermo.pressure
    frame['energy_smoothed']=thermo.energy_density_smoothed
    frame['mu_b']=thermo.chemical_potential
    frame['vs2']=thermo.vs2
    if not np.isfinite(frame[['pressure','energy_smoothed','mu_b','vs2']]).all().all():
        raise RuntimeError('Nonfinite EOS derivatives; inspect checkpoints, do not interpolate over failures.')
    return frame


def ensure_beta(job):
    folder,config,fit=job['folder'],job['config'],job['fit']
    if (folder/'eos.csv').exists():return pd.read_csv(folder/'eos.csv')
    eos=AsymmetricQuarkyonicEOS(fit,config.physical,config.quarkyonic)
    eos.solve_fq_beta=MethodType(global_beta,eos)
    rows=[];grid=suite.density_grid(CONTROLS)
    for i,ratio in enumerate(grid):
        path=folder/f'density_{i:05d}.json'
        if path.exists():row=json.loads(path.read_text())
        else:
            state=eos.build_beta_row(float(ratio*config.physical.n0))
            if state is None:raise RuntimeError(f'No beta solution at {ratio} n0')
            row={k:v for k,v in asdict(state).items() if v is not None}
            if not np.isfinite(list(row.values())).all():raise RuntimeError('Nonfinite state')
            assert 0<=row['y']<=.5 and 0<=row['quark_fraction']<=1
            assert abs(row['n_p']+row['n_n']-row['n_b']*(1-row['quark_fraction']))<1e-7
            residual=eos.neutrality_residual(row['y'],row['n_b'],row['quark_fraction'])
            if not np.isfinite(residual) or abs(residual)>1e-6:
                raise RuntimeError(f'Charge-neutrality residual {residual} at {ratio} n0')
            row['charge_residual_fm3']=residual
            suite.write_json(path,row)
        rows.append(row)
        if (i+1)%10==0 or i+1==len(grid):print(folder.parent.name,folder.name,i+1,'/',len(grid),flush=True)
    frame=finish_thermodynamics(pd.DataFrame(rows))
    suite.write_csv(folder/'eos.csv',frame)
    return frame


def main():
    global ALPHA, C_FM3, LAMBDA_MEV, SELECTED_MODELS, BRANCHES, WINDOW, DEGREE, QUAD_POINTS, FRACTION_POINTS, CONTROLS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=list(MODELS), help='Default: all six models')
    parser.add_argument('--branches', nargs='+', choices=['equal_b','target_l'], help='Default: both branches')
    parser.add_argument('--skip-tov', action='store_true', help='Only EOS/parameters/plots, no stellar integration')
    parser.add_argument('--plot-only', action='store_true', help='Reuse completed quick-run EOS and TOV checkpoints')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    print('TEST resolution: 81 densities per profile (61 + 20), 160 quadrature intervals, 41 fraction samples.')
    print('Outputs:', OUT)

    ALPHA, C_FM3, LAMBDA_MEV = 5/3, 4.74, 306.0
    SELECTED_MODELS=args.models or list(MODELS)
    BRANCHES=args.branches or ['equal_b','target_l']
    COMPUTE_BETA=not args.plot_only
    COMPUTE_TOV=not args.plot_only
    RECOMPUTE_SYMMETRIC=False
    WINDOW, DEGREE = 9, 3
    QUAD_POINTS, FRACTION_POINTS = 160, 41
    CONTROLS={**suite.DEFAULT_CONTROLS,'lambda_mev':LAMBDA_MEV,
     'beta_points':61,'extension_points':21,'maximum_density_ratio':15.,
     'integral_points':QUAD_POINTS,'fq_scan_points':FRACTION_POINTS,
     'tov_points':60,'tov_step_cm':500.,'tov_rmax_km':50.}
    assert set(SELECTED_MODELS)<=set(MODELS) and SELECTED_MODELS
    assert WINDOW%2==1 and DEGREE<WINDOW
    assert CONTROLS['maximum_density_ratio']>5

    jobs={};symmetric={};parameter_rows=[]
    for key in SELECTED_MODELS:
        config=load_run_config(ROOT/'examples/generated'/f'guided_{key}_lambda300_k0_200_750_step50_k0_250_target_l.json')
        parameter=ALPHA if key.startswith('dieterici') else C_FM3
        gs=compute_ground_state_point(config.model_name,parameter_value=parameter,
            physical=config.physical,settings=config.ground_state)
        symmetric[key]=(config,gs,parameter)
        for branch in BRANCHES:
            aset=replace(config.asymmetric,branch_mode=branch,target_k0=None,target_j=32.5,target_l=58.9)
            qset=replace(config.quarkyonic,lambda_momentum_mev=LAMBDA_MEV,
                shell_integral_points=QUAD_POINTS,quark_integral_points=QUAD_POINTS,
                fq_scan_points=FRACTION_POINTS,beta_y_scan_points=41,refine_tol=1e-6,refine_max_iter=160)
            cfg=replace(config,asymmetric=aset,quarkyonic=qset)
            fit=compute_asymmetric_fit(cfg.model_name,parameter_value=parameter,fit_settings=aset,
                physical=cfg.physical,ground_state_settings=cfg.ground_state)
            assert abs(fit.J-32.5)<.01
            if branch=='target_l':assert abs(fit.L-58.9)<.01
            else:assert abs(fit.b_pn-fit.b_n)<1e-8
            manifest=dict(config=asdict(cfg),controls=CONTROLS,parameter=parameter,
                window=WINDOW,degree=DEGREE,method='global-beta-minima-v1')
            digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode())
            for p in sorted((ROOT/'src/qmm').glob('*.py')):digest.update(p.read_bytes())
            for name in ['empirical_asymmetric_suite.py','plot_empirical_parameter_bands.py']:
                digest.update((ROOT/'scripts'/name).read_bytes())
            for p in sorted((ROOT/'src/TOVsolver').glob('*.py')):digest.update(p.read_bytes())
            folder=OUT/'runs'/key/f'{branch}_{digest.hexdigest()[:16]}'
            folder.mkdir(parents=True,exist_ok=True)
            suite.write_json(folder/'inputs.json',manifest)
            suite.write_json(folder/'fit.json',dict(symmetric=asdict(gs),asymmetric=asdict(fit)))
            jobs[key,branch]=dict(folder=folder,config=cfg,fit=fit)
            parameter_rows.append(dict(model=key,branch=branch,alpha=parameter if key.startswith('dieterici') else np.nan,
                c_fm3=parameter if key.startswith('clausius') else np.nan,a=gs.a,b=gs.b,K0_MeV=gs.K0,
                a_n=fit.a_n,a_pn=fit.a_pn,b_n=fit.b_n,b_pn=fit.b_pn,J_MeV=fit.J,L_MeV=fit.L))
    parameters=pd.DataFrame(parameter_rows)
    parameters.to_csv(OUT/'all_parameters.csv',index=False)
    display(parameters.round(6))

    sym_curves={}
    for key,(cfg,gs,parameter) in symmetric.items():
        old=ROOT/'Paper/symmetric_sound_speed_alpha_5over3_c_4p74'
        saved=old/f'{key}.json';use=False
        if not RECOMPUTE_SYMMETRIC and saved.exists() and (old/'settings.json').exists():
            r=json.loads(saved.read_text());settings=json.loads((old/'settings.json').read_text())
            use=np.isclose(r['parameter_value'],parameter) and np.isclose(settings['lambda_momentum_mev'],LAMBDA_MEV) and np.isclose(r['K0'],gs.K0)
        if not use:
            qset=replace(cfg.quarkyonic,n_min_ratio=.05,n_max_ratio=5.,n_points=CONTROLS['beta_points'],
                lambda_momentum_mev=LAMBDA_MEV,fq_scan_points=FRACTION_POINTS,
                shell_integral_points=QUAD_POINTS,quark_integral_points=QUAD_POINTS,
                smoothing_window=WINDOW,smoothing_degree=DEGREE,refine_tol=1e-8)
            signature=hashlib.sha256(json.dumps(dict(settings=asdict(qset),ground_state=asdict(gs)),sort_keys=True).encode()).hexdigest()[:16]
            cache=OUT/f'{key}_symmetric_{signature}.json'
            if cache.exists():r=json.loads(cache.read_text())
            else:
                r=asdict(compute_symmetric_quarkyonic_curve(cfg.model_name,parameter_value=parameter,
                    settings=qset,physical=cfg.physical,gs_settings=cfg.ground_state))
                suite.write_json(cache,r)
        f=pd.DataFrame(dict(n_b=r['n'],n_over_n0=r['n_over_n0'],quark_fraction=r['quark_fraction'],
            energy_density=r['eps_raw'],energy_smoothed=r['eps'],pressure=r['P'],vs2=r['vs2']))
        f['energy_per_baryon_minus_mN']=f.energy_density/f.n_b-cfg.physical.m_nucleon
        sym_curves[key]=f;f.to_csv(OUT/f'{key}_symmetric.csv',index=False)
        print(key, 'saved reference' if use else 'high-resolution cache',len(f),'points')

    beta_curves={}
    for (key,branch),job in jobs.items():
        if COMPUTE_BETA:f=ensure_beta(job)
        else:f=pd.read_csv(job['folder']/'eos.csv')
        f['energy_per_baryon_minus_mN']=f.energy_density/f.n_b-job['config'].physical.m_nucleon
        beta_curves[key,branch]=f
        f.to_csv(OUT/f'{key}_{branch}_beta.csv',index=False)
    print('All selected beta-equilibrium profiles available.')

    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':10,'pdf.fonttype':42})
    def finish(fig,stem):
        for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{stem}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    handles=[Line2D([],[],color=MODELS[k][1],label=MODELS[k][0]) for k in SELECTED_MODELS]
    branch_handles=[Line2D([],[],color='black',ls='-' if b=='equal_b' else ':',
        label=r'$b_{pn}=b_n$' if b=='equal_b' else r'$b_{pn}\ne b_n$') for b in BRANCHES]
    panels=[('n_over_n0','quark_fraction',r'$n_B/n_0$',r'$f_Q$'),
     ('n_over_n0','energy_per_baryon_minus_mN',r'$n_B/n_0$',r'$\epsilon/n_B-m_N$ [MeV]'),
     ('n_over_n0','vs2',r'$n_B/n_0$',r'$c_s^2/c^2$'),
     ('energy_smoothed','pressure',r'$\epsilon$ [MeV fm$^{-3}$]',r'$P$ [MeV fm$^{-3}$]')]
    for case in ['symmetric','beta']:
        fig,axes=plt.subplots(2,2,figsize=(9.5,6.5))
        for ax,(x,y,xlabel,ylabel) in zip(axes.flat,panels):
            for key in SELECTED_MODELS:
                for branch in (BRANCHES if case=='beta' else ['symmetric']):
                    frame=beta_curves[key,branch] if case=='beta' else sym_curves[key]
                    f=frame[frame.n_over_n0<=5+1e-8]
                    ax.plot(f[x],f[y],color=MODELS[key][1],ls=':' if branch=='target_l' else '-',lw=1.4)
            ax.set(xlabel=xlabel,ylabel=ylabel)
            if x=='n_over_n0':ax.set_xlim(0,5)
            if y=='vs2':ax.axhline(1/3,color='.5',ls='--',lw=.8)
            ax.tick_params(direction='in',top=True,right=True)
        fig.legend(handles=handles+(branch_handles if case=='beta' else []),loc='upper center',
                   ncol=4 if case=='beta' else 3,frameon=False,fontsize=9)
        fig.subplots_adjust(left=.09,right=.98,bottom=.09,top=.86,wspace=.3,hspace=.35)
        finish(fig,f'{case}_four_eos_panels')
    fig,ax=plt.subplots(figsize=(7,4.5))
    for (key,branch),f in beta_curves.items():
        ax.plot(f.n_over_n0,f.y,color=MODELS[key][1],ls='-' if branch=='equal_b' else ':')
    ax.set(xlabel=r'$n_B/n_0$',ylabel=r'$y(n_B)$',xlim=(0,5))
    fig.legend(handles=handles+branch_handles,loc='upper center',ncol=4,frameon=False,fontsize=8)
    fig.subplots_adjust(top=.8,bottom=.14)
    finish(fig,'beta_composition')
    if args.skip_tov:
        print("EOS test completed; TOV skipped.")
        return

    star_rows=[];star_curves={}
    for (key,branch),job in jobs.items():
        if COMPUTE_TOV:suite.ensure_tov(job['folder'],CONTROLS)
        path=job['folder']
        meta=json.loads((path/'tov_meta.json').read_text())
        stars=pd.read_csv(path/'mass_radius.csv')
        star_curves[key,branch]=stars
        f=beta_curves[key,branch]
        star_rows.append(dict(model=key,branch=branch,**{k:meta.get(k) for k in [
            'maximum_mass_msun','radius_at_maximum_mass_km','radius_at_1p4_msun_km',
            'maximum_at_eos_boundary','integration_warning_count','core_nonmonotone_rows_removed',
            'core_nonpositive_or_nonfinite_rows_removed']},maximum_cs2=float(f.vs2.max()),
            minimum_cs2=float(f.vs2.min()),maximum_charge_residual=float(f.charge_residual_fm3.abs().max())))
    stars_table=pd.DataFrame(star_rows)
    full_table=parameters.merge(stars_table,on=['model','branch'],validate='one_to_one')
    full_table.to_csv(OUT/'parameters_and_neutron_stars.csv',index=False)
    display(full_table)

    fig,ax=plt.subplots(figsize=(7,4.8))
    for (key,branch),stars in star_curves.items():
        # Keep the full central-density sequence, including the post-maximum branch.
        f=stars
        ax.plot(f.radius_km,f.mass_msun,color=MODELS[key][1],ls='-' if branch=='equal_b' else ':',lw=1.4)
    ax.set(xlabel=r'$R$ [km]',ylabel=r'$M/M_\odot$',xlim=(10,18))
    ax.tick_params(direction='in',top=True,right=True)
    fig.legend(handles=handles+branch_handles,loc='upper center',ncol=4,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.12,right=.98,bottom=.14,top=.8)
    finish(fig,'beta_mass_radius')
    print('Tables, figures and checkpoints:',OUT)
    print("Quick run finished. Results are exploratory; check convergence before physical interpretation.")


if __name__ == "__main__":
    main()
