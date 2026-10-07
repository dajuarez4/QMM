"""Compare saved beta-equilibrium EOS with direct derivatives; preserve originals."""
from pathlib import Path
import os
import sys
import json
import hashlib
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/qmm_mpl_cache')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from qmm.sound_speed import reconstruct_sound_speed_curve
from qmm.numerics import local_polynomial_regression
SOURCE = ROOT/'Paper/TEST_fixed_parameters_beta'
OUT = ROOT/'Paper/beta_fixed_parameters_direct_derivatives'
MODELS = {'clausius_vdw': ('Clausius–VDW', '#e600cc'), 'clausius_cs': ('Clausius–CS', '#f078aa'),
          'clausius_tvm': ('Clausius–TVM', '#9e205f'), 'dieterici_vdw': ('Dieterici–VDW', '#008000'),
          'dieterici_cs': ('Dieterici–CS', '#70b85c'), 'dieterici_tvm': ('Dieterici–TVM', '#005b40')}
BRANCHES = {'equal_b': '-', 'target_l': ':'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    curves, summary, provenance = {}, [], []
    for model in MODELS:
        for branch in BRANCHES:
            path = SOURCE/f'{model}_{branch}_beta.csv'
            f = pd.read_csv(path)
            assert np.isfinite(f[['n_b','energy_density','vs2']]).all().all()
            assert (np.diff(f.n_b)>0).all()
            matches = []
            for candidate in (SOURCE/'runs'/model).glob(f'{branch}_*/eos.csv'):
                raw = pd.read_csv(candidate)
                if len(raw)==len(f) and np.allclose(raw.energy_density, f.energy_density, rtol=0, atol=1e-10):
                    meta = json.loads((candidate.parent/'inputs.json').read_text())
                    assert np.isclose(meta['parameter'], 5/3 if model.startswith('dieterici') else 4.74)
                    assert np.isclose(meta['config']['quarkyonic']['lambda_momentum_mev'],306)
                    matches.append((candidate, meta))
            if not matches: raise RuntimeError(f'No matching beta provenance: {path}')
            meta = matches[0][1]
            _, mu_old, d2_old = local_polynomial_regression(f.n_b.tolist(), f.energy_density.tolist(),meta['window'],meta['degree'])
            np.testing.assert_allclose(f.vs2, f.n_b*np.array(d2_old)/np.array(mu_old),rtol=1e-5,atol=1e-6)
            new = reconstruct_sound_speed_curve(f.n_b.tolist(),f.energy_density.tolist(),9,3)
            f = f.rename(columns={'vs2':'vs2_old','pressure':'pressure_old','mu_b':'mu_b_old','energy_smoothed':'energy_smoothed_old'})
            f['vs2'] = new.vs2
            f['pressure'] = new.pressure
            f['mu_b'] = new.chemical_potential
            f['d2eps_dn2'] = new.d2eps_dn2
            f['delta_vs2'] = f.vs2-f.vs2_old
            assert np.isfinite(f[['vs2','pressure','mu_b']]).all().all()
            f.to_csv(OUT/path.name,index=False)
            g=f[f.n_over_n0<=5+1e-8]
            a,b=g.vs2_old.idxmax(),g.vs2.idxmax()
            summary.append(dict(model=model,branch=branch,points_total=len(f),points_to_5n0=len(g),
                old_peak=g.loc[a,'vs2_old'],new_peak=g.loc[b,'vs2'],old_peak_density=g.loc[a,'n_over_n0'],
                new_peak_density=g.loc[b,'n_over_n0'],peak_change_percent=100*(g.loc[b,'vs2']/g.loc[a,'vs2_old']-1),
                max_abs_change=g.delta_vs2.abs().max(),rms_change=np.sqrt(np.mean(g.delta_vs2**2))))
            provenance.append(dict(source=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                matching_inputs=[str((p.parent/'inputs.json').relative_to(ROOT)) for p,_ in matches]))
            curves[model,branch]=g
    table=pd.DataFrame(summary)
    table.to_csv(OUT/'comparison_summary.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':10})
    fig,axes=plt.subplots(3,2,figsize=(11,11),sharex=True)
    for ax,(model,(label,color)) in zip(axes.flat,MODELS.items()):
        for branch,ls in BRANCHES.items():
            f=curves[model,branch]
            ax.plot(f.n_over_n0,f.vs2_old,color='0.6',ls=ls,lw=2)
            ax.plot(f.n_over_n0,f.vs2,color=color,ls=ls,lw=1.3)
        ax.set(title=label,ylabel=r'$c_s^2/c^2$',xlabel=r'$n_B/n_0$',xlim=(0,5))
        ax.tick_params(direction='in',top=True,right=True)
    fig.legend(handles=[Line2D([],[],color='0.6',label='Previous: 9-point polynomial'),
        Line2D([],[],color='black',label='New: direct derivatives'),
        Line2D([],[],color='black',ls='-',label=r'$b_{pn}=b_n$'),Line2D([],[],color='black',ls=':',label=r'$b_{pn}\ne b_n$')],loc='upper center',ncol=2)
    fig.suptitle('Beta equilibrium — same energy, different derivatives\n61 points up to 5 n₀; α=5/3, c=4.74 fm³, Λ=306 MeV',y=.94)
    fig.tight_layout(rect=(0,0,1,.89))
    for ext in ['png','pdf']:fig.savefig(OUT/f'sound_speed_comparison.{ext}',dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    panels=[('n_over_n0','quark_fraction',r'$n_B/n_0$',r'$f_Q$'),('n_over_n0','energy_per_baryon_minus_mN',r'$n_B/n_0$',r'$\epsilon/n_B-m_N$ [MeV]'),('n_over_n0','vs2',r'$n_B/n_0$',r'$c_s^2/c^2$'),('energy_density','pressure',r'$\epsilon$ [MeV fm$^{-3}$]',r'$P$ [MeV fm$^{-3}$]')]
    for ax,(x,y,xlabel,ylabel) in zip(axes.flat,panels):
        for (model,branch),f in curves.items():ax.plot(f[x],f[y],color=MODELS[model][1],ls=BRANCHES[branch],lw=1.3)
        ax.set(xlabel=xlabel,ylabel=ylabel)
        if x=='n_over_n0':ax.set_xlim(0,5)
        ax.tick_params(direction='in',top=True,right=True)
    handles=[Line2D([],[],color=c,label=n) for n,c in MODELS.values()]
    handles += [Line2D([],[],color='black',ls=l,label=b) for b,l in BRANCHES.items()]
    fig.legend(handles=handles,loc='upper center',ncol=4,fontsize=9)
    fig.suptitle('Beta equilibrium: direct derivatives, no smoothing (test grid)',y=.9)
    fig.tight_layout(rect=(0,0,1,.86))
    for ext in ['png','pdf']:fig.savefig(OUT/f'beta_four_eos_panels_direct.{ext}',dpi=220)
    plt.close(fig)
    (OUT/'README.md').write_text('# Beta equilibrium: derivative comparison\n\nSix models, two branches; α=5/3, c=4.74 fm³, Λ=306 MeV.\nOriginal data: ../TEST_fixed_parameters_beta/, calculated in September.\nTotal energy (including leptons) and composition are reused unchanged.\nChemical potential, pressure and squared sound speed are recomputed using\nnumpy.gradient twice, with edge_order=2 and no smoothing.\nEnergy minimization and TOV are not repeated. Original files are preserved.\n\nEach profile has 81 densities: 61 up to 5 n0 and 20 more up to 15 n0.\nDerivatives use the full grid before plots are restricted to 5 n0.\nThe summary compares peaks and differences up to 5 n0, including endpoints.\nThe previous method was verified against the nine-point cubic reconstruction.\nThis test grid does not establish convergence; new peaks can include numerical noise.\nNegative or superluminal values are not clipped.\n\nRun from QMM: `.venv/bin/python scripts/compare_beta_direct_derivatives.py`.\n')
    print(table.to_string(index=False))
    return table

if __name__=='__main__':main()
