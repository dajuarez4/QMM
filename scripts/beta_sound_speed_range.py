"""Checkpoint beta-equilibrium EOS and compare six-model sound speeds to 5 n0."""
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import contextlib
import json
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
import empirical_asymmetric_suite as suite
import dieterici_simpson_acceleration as acceleration
import plot_six_model_empirical_observables as style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'Paper/beta_sound_speed_K0_250_760'
VALUES=[250,300,315,*range(350,751,50),760]
CONTROLS={**suite.DEFAULT_CONTROLS,'beta_points':120,'extension_points':2,'maximum_density_ratio':5.05}


def worker(job):
    key,k0,branch=job
    suite.register();acceleration.install()
    directory=OUTPUT/'runs'/f'{key}_K0_{k0}_{branch}'
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'worker.log').open('a') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            config=suite.config_for(key,k0,branch,CONTROLS)
            suite.write_json(directory/'inputs.json',{'controls':CONTROLS,'config':asdict(config)})
            fit=suite.ensure_fit(key,k0,branch,config,directory)
            suite.ensure_eos(fit,config,directory,CONTROLS)
            return key,k0,branch,'ok'
        except Exception as exc:
            suite.write_json(directory/'error.json',{'error':repr(exc)})
            return key,k0,branch,repr(exc)


def collect():
    curves={};audit=[]
    (OUTPUT/'curves').mkdir(exist_ok=True)
    for key in style.MODELS:
        for k0 in VALUES:
            for branch in ('equal_b','target_l'):
                if k0 in (315,760):
                    source=OUTPUT/'runs'/f'{key}_K0_{k0}_{branch}'/'eos.csv'
                else:
                    source=ROOT/'results/generated'/f'guided_{key}_lambda300_k0_200_750_step50_tov/extended_eos'/f'K0_{k0}_{branch}_extended_beta_eos.csv'
                if not source.exists():raise RuntimeError(f'Missing EOS: {source}')
                frame=pd.read_csv(source)
                frame=frame[frame.n_over_n0<=5+1e-8].copy().sort_values('n_b')
                # Consistent derivative windows restricted to the plotted density domain.
                rows=frame.to_dict('records');suite.local_thermodynamics(rows);frame=pd.DataFrame(rows)
                if len(frame)<100 or not np.isfinite(frame[['energy_density','pressure','vs2']].to_numpy()).all() or (frame.energy_density<=0).any():
                    raise RuntimeError(f'Invalid EOS for plotting: {source}')
                closure=float(abs(frame.n_p+frame.n_n-frame.n_b*(1-frame.quark_fraction)).max())
                if closure>1e-7:raise RuntimeError(f'Composition closure failure: {source}')
                target=OUTPUT/'curves'/f'{key}_K0_{k0}_{branch}.csv'
                frame.to_csv(target,index=False);curves[key,k0,branch]=frame
                audit.append(dict(model_key=key,K0_MeV=k0,branch_mode=branch,points=len(frame),
                    minimum_vs2=float(frame.vs2.min()),maximum_vs2=float(frame.vs2.max()),
                    density_closure_max=closure,superluminal_points=int(frame.vs2.gt(1).sum()),
                    negative_vs2_points=int(frame.vs2.lt(0).sum()),source=str(source.relative_to(ROOT))))
    pd.DataFrame(audit).to_csv(OUTPUT/'coverage.csv',index=False)
    return curves


def plot(curves):
    directory=OUTPUT/'figures';directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':10,'pdf.fonttype':42})
    handles=[Line2D([],[],color=s[1],lw=1.5,label=s[0]) for s in style.MODELS.values()]
    handles += [Line2D([],[],color='black',ls=ls,label=label) for ls,label in [('-',r'$b_{pn}=b_n$'),(':',r'$b_{pn}\ne b_n$')]]
    with PdfPages(directory/'all_K0_sound_speed.pdf') as pdf:
        for k0 in VALUES:
            fig,ax=plt.subplots(figsize=(7.6,4.8))
            for key,spec in style.MODELS.items():
                for branch,line in [('equal_b','-'),('target_l',':')]:
                    frame=curves[key,k0,branch]
                    ax.plot(frame.n_over_n0,frame.vs2,color=spec[1],ls=line,lw=1.4)
            ax.axhline(1,color='.45',ls='--',lw=.8)
            ax.set(xlabel=r'$n_B/n_0$',ylabel=r'$c_s^2/c^2=dP/d\epsilon$',xlim=(.05,5))
            ax.text(.03,.96,rf'$K_0={k0}$ MeV',transform=ax.transAxes,ha='left',va='top')
            ax.tick_params(which='both',direction='in',top=True,right=True)
            ax.xaxis.set_minor_locator(style.AutoMinorLocator(2));ax.yaxis.set_minor_locator(style.AutoMinorLocator(2))
            fig.legend(handles=handles,ncol=4,loc='upper center',frameon=False,fontsize=8.5)
            fig.subplots_adjust(left=.12,right=.98,bottom=.14,top=.81)
            for ext in ('png','pdf','svg'):fig.savefig(directory/f'sound_speed_K0_{k0}.{ext}',dpi=240,bbox_inches='tight')
            pdf.savefig(fig,bbox_inches='tight');plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    jobs=[(key,k0,branch) for key in style.MODELS for k0 in (315,760) for branch in ('equal_b','target_l')]
    failures=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(worker,job) for job in jobs};finished=0
        while pending:
            done,pending=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                result=future.result();finished+=1;print('Finished',finished,'/',len(jobs),result,flush=True)
                if result[-1]!='ok':failures.append(result)
            if not done:
                counts={p.parent.name:len(json.loads(p.read_text())) for p in (OUTPUT/'runs').glob('*/density_rows.json')}
                print('Density progress:',counts,flush=True)
    if failures:raise RuntimeError(failures)
    curves=collect();plot(curves)
    print('Complete:',len(curves),'EOS curves;',len(VALUES),'six-model comparison pages.',flush=True)

if __name__=='__main__':main()
