"""Symmetric T=0 sound speeds at alpha=5/3 and c=4.74 fm^3."""
from pathlib import Path
from dataclasses import replace,asdict
import json
import numpy as np
import pandas as pd
import plot_six_model_empirical_observables as style
from plot_empirical_parameter_bands import register_models
from qmm.constants import DEFAULT_QUARKYONIC_SETTINGS
from qmm.config import load_run_config
from qmm.quarkyonic import compute_symmetric_quarkyonic_curve
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'Paper/symmetric_sound_speed_alpha_5over3_c_4p74'
SETTINGS=replace(DEFAULT_QUARKYONIC_SETTINGS,n_min_ratio=.05,n_max_ratio=5.,n_points=241,
                 fq_scan_points=141,shell_integral_points=400,quark_integral_points=400,
                 lambda_momentum_mev=306.,smoothing_window=9)

def main():
    register_models();OUTPUT.mkdir(parents=True,exist_ok=True)
    curves={};summary=[]
    for key in style.MODELS:
        config=load_run_config(ROOT/'examples/generated'/f'guided_{key}_lambda300_k0_200_750_step50_k0_250_target_l.json')
        parameter=5/3 if key.startswith('dieterici') else 4.74
        cache=OUTPUT/f'{key}.json'
        if cache.exists():r=json.loads(cache.read_text())
        else:
            result=compute_symmetric_quarkyonic_curve(config.model_name,parameter_value=parameter,
                settings=SETTINGS,physical=config.physical,gs_settings=config.ground_state)
            r=asdict(result);cache.write_text(json.dumps(r,indent=2)+'\n')
        assert len(r['n'])==SETTINGS.n_points
        assert np.isfinite(r['vs2']).all()
        frame=pd.DataFrame({k:r[k] for k in ['n','n_over_n0','eps_raw','eps','P','mu_b','vs2','quark_fraction']})
        frame.to_csv(OUTPUT/f'{key}.csv',index=False);curves[key]=frame
        summary.append({'model_key':key,'parameter_value':parameter,'K0_MeV':r['K0'],'a':r['a'],'b':r['b'],
                        'maximum_vs2':r['vs2_max'],'peak_density_over_n0':r['n_tr_over_n0']})
        print(key,'K0=',r['K0'],'peak=',r['vs2_max'],flush=True)
    pd.DataFrame(summary).to_csv(OUTPUT/'model_parameters_and_peaks.csv',index=False)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':10,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(7,4.7))
    for key,spec in style.MODELS.items():
        frame=curves[key];ax.plot(frame.n_over_n0,frame.vs2,color=spec[1],lw=1.5,label=spec[0])
    ax.axhline(1/3,color='.45',ls='--',lw=.8)
    ax.text(.15,1/3+.012,r'$c_s^2/c^2=1/3$',color='.4',fontsize=9)
    ax.set(xlabel=r'$n_B/n_0$',ylabel=r'$c_s^2/c^2$',xlim=(0,5))
    ax.tick_params(which='both',direction='in',top=True,right=True)
    ax.xaxis.set_minor_locator(style.AutoMinorLocator(2));ax.yaxis.set_minor_locator(style.AutoMinorLocator(2))
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,fontsize=9)
    ax.text(.97,.95,r'$y=0.5,\ T=0$'+'\n'+r'$\alpha=5/3,\ c=4.74\ {\rm fm}^3$',transform=ax.transAxes,ha='right',va='top',fontsize=10)
    fig.subplots_adjust(left=.12,right=.98,bottom=.14,top=.83)
    for ext in ['png','pdf','svg']:fig.savefig(OUTPUT/f'six_models_sound_speed.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    (OUTPUT/'settings.json').write_text(json.dumps(asdict(SETTINGS),indent=2)+'\n')
    (OUTPUT/'README.md').write_text('# Symmetric fixed-parameter sound speed\n\nSymmetric quarkyonic matter at T=0 and y=0.5; no leptons or beta-equilibrium constraint. Dieterici alpha is exactly 5/3; Clausius c is 4.74 fm^3. All three repulsive prescriptions (VDW, CS, TVM) are retained. Saturation fitting determines a, b and K0 independently for each model; K0 is not fixed at 250 MeV. Lambda=306 MeV follows the cited Lysenko et al. article. Only the VDW-repulsion members directly correspond to its Dieterici and Clausius choices.\n\n241 density points from 0.05 to 5 n0, 141 fraction-scan points, and 400 quadrature intervals. The QMM symmetric solver reconstructs sound speed with a nine-point cubic smoothing window. settings.json records all numerical controls. The gray dashed line is the conformal value 1/3, not the causal bound. Numerical peaks depend on density resolution and smoothing.\n\nmodel_parameters_and_peaks.csv records fitted incompressibilities and peak properties. JSON and CSV files preserve all computed curves. Reproduce with .venv/bin/python scripts/symmetric_sound_speed_fixed_parameters.py.\n')

if __name__=='__main__':main()
