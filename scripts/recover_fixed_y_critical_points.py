"""Retry missing fixed-y critical points and extend both branches to K0=760."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import empirical_asymmetric_suite as suite

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'Paper/six_model_fixed_y_observables_K0_250_750/recovery'


def solve(fit, config, y, seeds):
    evaluator = suite.FixedYEVPressureEvaluator(fit, y, physical=config.physical, settings=suite.CP_SETTINGS)
    def residual(x):
        d1,d2,*_ = evaluator.critical_equations(*map(float,x))
        if d1 is None or d2 is None or not np.isfinite(d1+d2): return np.array([1e4,1e4])
        return np.array([d1/10,d2/1000])
    attempts=[]
    for seed in seeds+[(8.,.04),(15.,.06),(20.,.08),(3.,.015)]:
        result=least_squares(residual,seed,bounds=([.05,.0008],[30.,.119]),
            x_scale=[5.,.04],diff_step=[1e-5,1e-5],xtol=3e-11,ftol=3e-11,gtol=3e-11,max_nfev=350)
        tc,nc=map(float,result.x);d1,d2=residual(result.x)*[10,1000]
        attempts.append(dict(seed=list(map(float,seed)),Tc_MeV=tc,nc_fm3=nc,dPdn=float(d1),d2Pdn2=float(d2),nfev=result.nfev))
        if abs(d1)<1e-4 and abs(d2)<1e-3:
            pc,*_=evaluator.pressure(tc,nc)
            if pc is not None and np.isfinite(pc) and pc>0:
                return dict(status='ok',Tc_MeV=tc,nc_fm3=nc,Pc_MeV_fm3=pc,dPdn=float(d1),d2Pdn2=float(d2),attempts=attempts)
    return dict(status='failed',attempts=attempts,error='No accepted critical root after continuation and multi-start search; not proof of absence.')


def main():
    suite.register();OUTPUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(OUTPUT.parent/'plotted_critical_points.csv')
    jobs=set(map(tuple,data.loc[data.status.ne('ok'),['model_key','branch_mode','K0_target_MeV','y']].to_numpy()))
    existing = OUTPUT/'recovered_points.csv'
    if existing.exists():
        jobs.update(map(tuple,pd.read_csv(existing)[['model_key','branch_mode','K0_target_MeV','y']].to_numpy()))
    jobs.update((key,branch,760.,y) for key in suite.MODELS for branch in suite.BRANCHES for y in [.1,.2,.3,.4,.5])
    recovered=[]
    for key,branch,k0,y in sorted(jobs):
        directory=OUTPUT/f'{key}_K0_{k0:g}_{branch}'
        path=directory/f'critical_y_{y:g}.json'
        config=suite.config_for(key,k0,branch,suite.DEFAULT_CONTROLS)
        identity=dict(model_key=key,branch_mode=branch,K0_target_MeV=k0,y=y)
        if path.exists() and json.loads(path.read_text()).get('status')=='ok':
            row=json.loads(path.read_text())
        else:
            try:
                fit=suite.ensure_fit(key,k0,branch,config,directory)
                nearby=pd.concat([data,pd.DataFrame(recovered)],ignore_index=True)
                nearby=nearby[nearby.model_key.eq(key)&nearby.branch_mode.eq(branch)&np.isclose(nearby.y,y)&nearby.status.eq('ok')].copy()
                nearby['distance']=abs(nearby.K0_target_MeV-k0)
                seeds=[(r.Tc_MeV,r.nc_fm3) for r in nearby.sort_values('distance').head(2).itertuples()]
                row={**identity,**solve(fit,config,y,seeds), 'parameter_value':fit.parameter_value,'K0_MeV':fit.K0,'settings':asdict(suite.CP_SETTINGS)}
            except Exception as exc:row={**identity,'status':'failed','error':repr(exc)}
            suite.write_json(path,row)
        recovered.append({k:v for k,v in row.items() if k not in ('attempts','settings')})
        print(key,branch,k0,y,row['status'],row.get('Tc_MeV',row.get('error','')),flush=True)
        pd.DataFrame(recovered).to_csv(OUTPUT/'recovered_points.csv',index=False)
    print('Complete:',len(recovered),'attempted;',sum(r['status']=='ok' for r in recovered),'converged',flush=True)

if __name__=='__main__': main()
