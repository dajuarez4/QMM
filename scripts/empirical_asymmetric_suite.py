"""Checkpointed six-model, two-branch empirical EoS/TOV/fixed-y workflow."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import warnings

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'src/TOVsolver', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/qmm_mpl_cache')
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from qmm.config import load_run_config
from qmm.asymmetry import AsymmetricFitResult, compute_asymmetric_fit
from qmm.ground_state import compute_ground_state_point_explicit
from qmm.quarkyonic import AsymmetricQuarkyonicEOS
from qmm.fixed_y_quantum import FixedYEVPressureEvaluator
import qmm.fixed_y_quantum as fixed_y_module
from qmm.constants import DEFAULT_QUANTUM_SETTINGS
from plot_empirical_parameter_bands import register_models
from clausius_asymmetric_tov import local_thermodynamics
from TOV_solver_code import stitch_crust, solve_sequence, radius_at_mass

MODELS = [f'{family}_{ev}' for family in ('dieterici','clausius') for ev in ('vdw','cs','tvm')]
BRANCHES = ['target_l','equal_b']
OUTPUT = ROOT / 'Paper/empirical_asymmetric_lambda300'
DEFAULT_CONTROLS = dict(lambda_mev=300., beta_points=120, extension_points=31,
    maximum_density_ratio=15., fq_scan_points=141, integral_points=400,
    tov_points=260, tov_step_cm=250., tov_rmax_km=50., y_values=[.1,.2,.3,.4,.5])
CP_SETTINGS = replace(DEFAULT_QUANTUM_SETTINGS, n_k_fd=800, k_max_fd=5.,
    mu_scf_tol=1e-11, mu_scf_max_iter=200, mu_delta=.1, cache_round_digits=12,
    t_min_cp=.05, t_max_cp=30., n_min_cp=.0005, n_max_cp=.12)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def write_csv(path, rows):
    temporary=path.with_suffix('.csv.tmp')
    pd.DataFrame(rows).to_csv(temporary,index=False)
    temporary.replace(path)


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def template_path(key, k0=250, branch='target_l'):
    return ROOT/'examples/generated'/f'guided_{key}_lambda300_k0_200_750_step50_k0_{k0}_{branch}.json'


def register():
    register_models()
    fixed_y_module.FIXED_Y_MODELS |= {'dieterici_cs','dieterici_tvm'}


def config_for(key, k0, branch, controls):
    config=load_run_config(template_path(key))
    return replace(config, asymmetric=replace(config.asymmetric, branch_mode=branch,
        target_k0=float(k0), target_j=32.5, target_l=58.9),
        quarkyonic=replace(config.quarkyonic, lambda_momentum_mev=controls['lambda_mev'],
            n_min_ratio=.05,n_max_ratio=5.,n_points=controls['beta_points'],
            fq_scan_points=controls['fq_scan_points'],
            shell_integral_points=controls['integral_points'],
            quark_integral_points=controls['integral_points']))


def job_directory(key,k0,branch,controls):
    # Separate runs when physical/numerical inputs or core implementation change.
    fingerprint=hashlib.sha256(json.dumps(controls,sort_keys=True).encode())
    fingerprint.update(template_path(key).read_bytes())
    fingerprint.update(json.dumps(asdict(CP_SETTINGS),sort_keys=True).encode())
    for path in sorted((ROOT/'src/qmm').glob('*.py')):
        fingerprint.update(path.read_bytes())
    for path in ('clausius_asymmetric_tov.py','TOV_solver_code.py','tov.py'):
        fingerprint.update((ROOT/'src/TOVsolver'/path).read_bytes())
    fingerprint.update(b'empirical-asymmetric-pipeline-v1')
    return OUTPUT/'runs'/fingerprint.hexdigest()[:16]/f'{key}_K0_{k0:g}_{branch}'


def fit_valid(fit,k0,branch):
    assert abs(fit.K0-k0)<.01, 'K0 fit residual'
    assert abs(fit.J-32.5)<.01, 'J fit residual'
    if branch=='target_l': assert abs(fit.L-58.9)<.01, 'L fit residual'
    else: assert abs(fit.b_pn-fit.b_n)<1e-8, 'Equal-b branch constraint'


def ensure_fit(key,k0,branch,config,path):
    stored=read_json(path/'fit.json')
    if stored:
        fit=AsymmetricFitResult(**stored['asymmetric_fit'])
        fit_valid(fit,k0,branch)
        return fit
    symmetric_path=ROOT/'Paper/empirical_critical_K0_250_315/checkpoints'/f'{key}_K0_{k0:g}.json'
    parameter=read_json(symmetric_path,{}).get('parameter_value')
    def ground(p):
        point=compute_ground_state_point_explicit(config.model_name,float(p),config.physical,config.ground_state)
        if point is None: raise ValueError('Saturation fit failed')
        return point
    if parameter is None or abs(ground(parameter).K0-k0)>.01:
        # Search bounds from the established model-specific scan.
        from qmm.ground_state import resolve_parameter_value
        parameter=resolve_parameter_value(config.model_name,None,
            replace(config.parameter_search,target_k0=float(k0)),config.physical,config.ground_state)
    gs=ground(parameter)
    fit=compute_asymmetric_fit(config.model_name,parameter_value=parameter,
        fit_settings=config.asymmetric,physical=config.physical,ground_state_settings=config.ground_state)
    fit_valid(fit,k0,branch)
    write_json(path/'fit.json',{'ground_state':asdict(gs),'asymmetric_fit':asdict(fit),'source':'computed'})
    return fit


def density_grid(controls):
    return np.concatenate((np.linspace(.05,5.,controls['beta_points']),
        np.linspace(5.,controls['maximum_density_ratio'],controls['extension_points'])[1:]))


def import_saved(key,k0,branch,config,path,controls):
    """Import compatible completed step50 products with their original provenance."""
    source_config_path=template_path(key,int(k0),branch)
    if k0!=int(k0) or not source_config_path.exists():return
    old=load_run_config(source_config_path)
    if any(asdict(getattr(old,field))!=asdict(getattr(config,field))
           for field in ('physical','ground_state','asymmetric','quarkyonic')):return
    suite=f'guided_{key}_lambda300_k0_200_750_step50'
    source=ROOT/'results/generated'/suite/f'K0_{k0:g}_{branch}'/f'{suite}_k0_{k0:g}_{branch}_summary.json'
    if not source.exists():return
    payload=read_json(source)
    fit=AsymmetricFitResult(**payload['asymmetric_fit'])
    fit_valid(fit,k0,branch)
    if not (path/'fit.json').exists():
        write_json(path/'fit.json',{'ground_state':payload['ground_state'],'asymmetric_fit':asdict(fit),'source':str(source.relative_to(ROOT))})
    previous=ROOT/'results/generated'/f'{suite}_tov'
    eos_source=previous/'extended_eos'/f'K0_{k0:g}_{branch}_extended_beta_eos.csv'
    if eos_source.exists() and not (path/'eos.csv').exists():
        eos=pd.read_csv(eos_source)
        grid=density_grid(controls)
        if (len(eos)==len(grid) and np.allclose(eos.n_over_n0,grid,atol=1e-9)
            and np.isfinite(eos[['energy_density','pressure','n_b']]).all().all()):
            write_csv(path/'eos.csv',eos)
            write_json(path/'eos_meta.json',{'status':'ok','source':str(eos_source.relative_to(ROOT)),
                'points':len(eos),'maximum_density_ratio':float(eos.n_over_n0.max())})
    # Only reuse the original 260-star integration with its original controls.
    mr_source=previous/'curves'/f'K0_{k0:g}_{branch}_mass_radius.csv'
    if ((path/'eos.csv').exists() and mr_source.exists() and controls['tov_points']==260
        and controls['tov_step_cm']==250. and controls['tov_rmax_km']==50.
        and not (path/'tov_meta.json').exists()):
        rows=read_json(previous/'model_tov_summary.json',[])
        matches=[r for r in rows if r['K0_MeV']==k0 and r['branch_mode']==branch]
        if matches:
            mr=pd.read_csv(mr_source)
            if len(mr)>=4 and np.isfinite(mr[['radius_km','mass_msun']]).all().all():
                write_csv(path/'mass_radius.csv',mr)
                write_json(path/'tov_meta.json',{**matches[0],'status':'ok','source':str(mr_source.relative_to(ROOT))})


def ensure_eos(fit,config,path,controls):
    if (path/'eos.csv').exists() and read_json(path/'eos_meta.json',{}).get('status')=='ok':return
    evaluator=AsymmetricQuarkyonicEOS(fit,config.physical,config.quarkyonic)
    checkpoint=path/'density_rows.json'
    saved=read_json(checkpoint,{})
    grid=density_grid(controls)
    for index,ratio in enumerate(grid):
        tag=str(index)
        if tag in saved:continue
        result=evaluator.build_beta_row(float(ratio*config.physical.n0))
        if result is None:raise RuntimeError(f'No beta solution at n/n0={ratio:g}')
        row={k:v for k,v in asdict(result).items() if k not in ('mu_b','pressure','vs2')}
        if any(v is None or not np.isfinite(v) for v in row.values()):
            raise RuntimeError(f'Nonfinite beta solution at n/n0={ratio:g}')
        closure=abs(row['n_p']+row['n_n']-row['n_b']*(1-row['quark_fraction']))
        if closure>1e-7 or not (0<=row['y']<=.5 and 0<=row['quark_fraction']<=1):
            raise RuntimeError(f'Composition check failed at n/n0={ratio:g}')
        saved[tag]=row
        write_json(checkpoint,saved)
        print(f'density {index+1}/{len(grid)}: n/n0={ratio:.4f}',flush=True)
    rows=[saved[str(i)] for i in range(len(grid))]
    local_thermodynamics(rows)
    frame=pd.DataFrame(rows)
    if not np.isfinite(frame[['pressure','energy_density','vs2']]).all().all():
        raise RuntimeError('Nonfinite reconstructed thermodynamics')
    write_csv(path/'eos.csv',frame)
    write_json(path/'eos_meta.json',{'status':'ok','source':'computed','points':len(frame),
        'maximum_density_ratio':float(frame.n_over_n0.max()),
        'density_closure_max':float(np.max(np.abs(frame.n_p+frame.n_n-frame.n_b*(1-frame.quark_fraction)))),
        'maximum_vs2':float(frame.vs2.max()),'minimum_vs2':float(frame.vs2.min())})


def ensure_tov(path,controls,force=False):
    if not force and (path/'mass_radius.csv').exists() and read_json(path/'tov_meta.json',{}).get('status')=='ok':return
    frame=pd.read_csv(path/'eos.csv')
    valid=frame[np.isfinite(frame.energy_density)&np.isfinite(frame.pressure)&(frame.energy_density>0)&(frame.pressure>0)]
    valid=valid.sort_values('energy_density')
    keep=[];last_p=last_e=-np.inf
    for row in valid.itertuples():
        if row.pressure>last_p and row.energy_density>last_e:
            keep.append((row.energy_density,row.pressure));last_e=row.energy_density;last_p=row.pressure
    if len(keep)<8:raise RuntimeError('Insufficient monotone positive core EoS samples')
    energy,pressure=np.asarray(keep).T
    stitched_e,stitched_p,transition_e,transition_p=stitch_crust(energy,pressure)
    write_csv(path/'stitched_eos.csv',{'energy_density':stitched_e,'pressure':stitched_p})
    with warnings.catch_warnings(record=True) as integration_warnings:
        warnings.simplefilter('always')
        rows=solve_sequence(stitched_e,stitched_p,transition_e,controls['tov_points'],
                            controls['tov_rmax_km'],controls['tov_step_cm'])
    peak_i=int(np.argmax([r['mass_msun'] for r in rows]));peak=rows[peak_i]
    write_csv(path/'mass_radius.csv',rows)
    write_json(path/'tov_meta.json',{'status':'ok','source':'computed',
        'maximum_mass_msun':peak['mass_msun'],'radius_at_maximum_mass_km':peak['radius_km'],
        'radius_at_1p4_msun_km':radius_at_mass(rows,1.4),
        'maximum_at_eos_boundary':peak_i==len(rows)-1,
        'maximum_density_ratio_reached':float(frame.n_over_n0.max()),
        'core_nonpositive_or_nonfinite_rows_removed':len(frame)-len(valid),
        'core_nonmonotone_rows_removed':len(valid)-len(keep),
        'crust_transition_energy_mev_fm3':transition_e,'crust_transition_pressure_mev_fm3':transition_p,
        'maximum_vs2':float(frame.vs2.max()),
        'integration_warning_count':len(integration_warnings),
        'integration_warnings':sorted({str(w.message) for w in integration_warnings})})


def ensure_critical(fit,config,path,y):
    output=path/f'critical_y_{y:g}.json'
    if read_json(output,{}).get('status')=='ok':return
    evaluator=FixedYEVPressureEvaluator(fit,y,physical=config.physical,settings=CP_SETTINGS)
    seeds={.1:(1.3,.011),.2:(7.1,.04),.3:(10.7,.052),.4:(12.6,.058),.5:(14.,.06)}
    primary=seeds[min(seeds,key=lambda v:abs(v-y))]
    def residual(values):
        d1,d2,_,_=evaluator.critical_equations(*map(float,values))
        if d1 is None or d2 is None or not np.isfinite(d1+d2):return np.array([1e4,1e4])
        return np.array([d1/10.,d2/1000.])
    best=None
    for seed in (primary,(primary[0]*.8,primary[1]*.8),(primary[0]*1.3,primary[1]*1.15),(17.,.05)):
        solved=least_squares(residual,seed,bounds=([.05,.0008],[30.,.119]),
            x_scale=[5.,.04],diff_step=[1e-5,1e-5],xtol=3e-11,ftol=3e-11,gtol=3e-11,max_nfev=350)
        d1,d2=residual(solved.x)*[10.,1000.]
        if abs(d1)<1e-4 and abs(d2)<1e-3:
            best=solved;break
    if best is None:
        write_json(output,{'status':'failed','y':y,'error':'No converged positive-temperature critical point; this does not prove absence of a critical point.'})
        return
    temperature,density=map(float,best.x)
    pressure,_,_=evaluator.pressure(temperature,density)
    if pressure is None or not np.isfinite(pressure) or pressure<=0:raise RuntimeError('Invalid critical pressure')
    write_json(output,{'status':'ok','y':y,'Tc_MeV':temperature,'nc_fm3':density,'Pc_MeV_fm3':pressure,
        'dPdn':float(d1),'d2Pdn2':float(d2),'parameter_value':fit.parameter_value,'source':'computed fixed-y'})
    print(f'critical y={y:g}: Tc={temperature:.6f}, nc={density:.7f}',flush=True)


def run_job(key,k0,branch,controls,stages,compute_missing=True):
    register()
    path=job_directory(key,k0,branch,controls);path.mkdir(parents=True,exist_ok=True)
    config=config_for(key,k0,branch,controls)
    write_json(path/'inputs.json',{'model_key':key,'K0_MeV':k0,'branch_mode':branch,
        'controls':controls,'physical':asdict(config.physical),'quarkyonic':asdict(config.quarkyonic)})
    import_saved(key,k0,branch,config,path,controls)
    if not compute_missing:return
    errors={}
    try:fit=ensure_fit(key,k0,branch,config,path)
    except Exception as exc:
        write_json(path/'errors.json',{'fit':repr(exc)});return
    for stage in stages:
        try:
            if stage=='eos':ensure_eos(fit,config,path,controls)
            elif stage=='tov':ensure_tov(path,controls)
            elif stage=='critical':
                for y in controls['y_values']:
                    try:ensure_critical(fit,config,path,float(y))
                    except Exception as exc:
                        write_json(path/f'critical_y_{y:g}.json',{'status':'failed','y':y,'error':repr(exc)})
            elif stage!='fit':raise ValueError(f'Unknown stage: {stage}')
        except Exception as exc:
            errors[stage]=repr(exc);print(f'{stage} failed: {exc}',flush=True)
    write_json(path/'errors.json',errors)


def collect(k0_values,models=MODELS,branches=BRANCHES,controls=None):
    controls=DEFAULT_CONTROLS if controls is None else controls
    rows=[];critical=[]
    for key in models:
        for k0 in k0_values:
            for branch in branches:
                path=job_directory(key,k0,branch,controls)
                identity={'model_key':key,'K0_MeV':k0,'branch_mode':branch,'directory':str(path)}
                row=dict(identity)
                stored=read_json(path/'fit.json',{})
                fit=stored.get('asymmetric_fit',{})
                row.update({k:fit.get(k,np.nan) for k in ('parameter_value','J','L','b_n','b_pn','a_n','a_pn')})
                row['fit_ok']=bool(stored)
                eos=read_json(path/'eos_meta.json',{});tov=read_json(path/'tov_meta.json',{})
                row.update(eos_ok=eos.get('status')=='ok' and (path/'eos.csv').exists(),
                    tov_ok=tov.get('status')=='ok' and (path/'mass_radius.csv').exists(),
                    density_points_cached=len(read_json(path/'density_rows.json',{})),
                    maximum_mass_msun=tov.get('maximum_mass_msun',np.nan),
                    radius_at_maximum_mass_km=tov.get('radius_at_maximum_mass_km',np.nan),
                    maximum_at_eos_boundary=tov.get('maximum_at_eos_boundary',False),
                    maximum_vs2=eos.get('maximum_vs2',tov.get('maximum_vs2',np.nan)))
                if row['eos_ok'] and not np.isfinite(row['maximum_vs2']):
                    row['maximum_vs2']=float(pd.read_csv(path/'eos.csv').vs2.max())
                row['integration_warning_count']=tov.get('integration_warning_count',np.nan)
                rows.append(row)
                for y in controls['y_values']:
                    cp=read_json(path/f'critical_y_{y:g}.json',{'status':'missing','y':y})
                    critical.append({**identity,'parameter_value':row['parameter_value'],**cp})
    results=pd.DataFrame(rows);critical_frame=pd.DataFrame(critical)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    write_csv(OUTPUT/'configuration_status.csv',results)
    write_csv(OUTPUT/'fixed_y_critical_points.csv',critical_frame)
    return results,critical_frame


def run_suite(k0_values=range(250,316,5),models=MODELS,branches=BRANCHES,controls=None,
              stages=('fit','eos','tov','critical'),workers=4,compute_missing=True):
    controls=dict(DEFAULT_CONTROLS if controls is None else controls)
    values=list(map(float,k0_values));models=list(models);branches=list(branches)
    if not values or any(v<250 or v>315 for v in values):raise ValueError('K0 must be within 250–315 MeV')
    if set(models)-set(MODELS) or set(branches)-set(BRANCHES):raise ValueError('Unknown model or branch')
    if not models or not branches or workers<1:raise ValueError('Nonempty model/branch lists and positive workers required')
    if set(stages)-{'fit','eos','tov','critical'}:raise ValueError('Unknown stage')
    if controls['beta_points']<9 or controls['extension_points']<2:raise ValueError('Insufficient density samples')
    if controls['maximum_density_ratio']<=5:raise ValueError('Extension must exceed 5 n0')
    if any(y<=0 or y>.5 for y in controls['y_values']):raise ValueError('Use 0 < y <= 0.5')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    settings_path=OUTPUT/('controls_'+hashlib.sha256(json.dumps(controls,sort_keys=True).encode()).hexdigest()[:12]+'.json')
    write_json(settings_path,controls)
    jobs=[(key,k0,branch) for key in models for k0 in values for branch in branches]
    if not compute_missing:
        for job in jobs:run_job(*job,controls,stages,False)
        return collect(values,models,branches,controls)
    processes={};lock=threading.Lock();stopping=threading.Event()
    def launch(job):
        key,k0,branch=job;path=job_directory(key,k0,branch,controls);path.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (path/'worker.log').open('a') as log:
            with lock:
                if stopping.is_set(): return job, -15
                child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',key,str(k0),branch,
                    '--controls',str(settings_path),'--stages',*stages],stdout=log,stderr=subprocess.STDOUT,env=env)
                processes[job]=child
            code=child.wait()
            with lock:processes.pop(job,None)
        return job,code
    executor=ThreadPoolExecutor(max_workers=workers)
    futures={executor.submit(launch,job) for job in jobs};pending=set(futures);done_count=0
    try:
        while pending:
            done,pending=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                job,code=future.result();done_count+=1
                print(f'Finished worker {done_count}/{len(jobs)}: {job}, exit={code}. See status table for stage success.',flush=True)
            if not done:print(f'Running: {done_count}/{len(jobs)} workers finished; per-density progress in worker.log.',flush=True)
    except BaseException:
        stopping.set()
        for future in futures:future.cancel()
        with lock:
            for child in processes.values():child.terminate()
        raise
    finally:executor.shutdown(wait=True)
    return collect(values,models,branches,controls)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',nargs=3,required=True,metavar=('MODEL','K0','BRANCH'))
    parser.add_argument('--controls',type=Path,required=True)
    parser.add_argument('--stages',nargs='+',default=['fit','eos','tov','critical'])
    args=parser.parse_args();key,k0,branch=args.worker
    run_job(key,float(k0),branch,read_json(args.controls),args.stages)
