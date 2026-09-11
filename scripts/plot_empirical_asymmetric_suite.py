"""Plot both empirical branches without treating missing calculations as samples."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import empirical_asymmetric_suite as suite

BRANCH_LABELS={'target_l':r'$b_{pn}\ne b_n$, $L=58.9$ MeV','equal_b':r'$b_{pn}=b_n$'}
STYLES={'target_l':'-','equal_b':'--'}
COLORS={'target_l':'#0072B2','equal_b':'#D55E00'}


def stable_curve(frame):
    if 'stable_branch' in frame:
        selected=frame.stable_branch.astype(str).str.lower().isin(['true','1'])
        return frame[selected]
    return frame.iloc[:int(np.argmax(frame.mass_msun))+1]


def plot_results(results,critical,k0_values,controls=None):
    controls=suite.DEFAULT_CONTROLS if controls is None else controls
    values=sorted(map(float,k0_values));expected=len(values)
    active_branches=[b for b in suite.BRANCHES if b in set(results.branch_mode)]
    output=suite.OUTPUT/'figures';output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':9})
    paths=[]
    def finish(fig,name):
        for ext in ('png','pdf','svg'):
            path=output/f'{name}.{ext}';fig.savefig(path,dpi=160,bbox_inches='tight')
            if ext=='png':paths.append(path)
        plt.close(fig)
    critical=critical.copy()
    for field in ('Tc_MeV','nc_fm3'):
        if field not in critical:critical[field]=np.nan
        critical.loc[critical.status!='ok',field]=np.nan
    for key,data in results.groupby('model_key',sort=False):
        family,ev=key.split('_');title=f'{family.title()}–{ev.upper()}'
        xlabel=r'$\alpha$' if family=='dieterici' else r'$c$ [fm$^3$]'
        valid_parameters=data.parameter_value.dropna()
        if valid_parameters.empty:continue
        norm=matplotlib.colors.Normalize(valid_parameters.min(),valid_parameters.max() if valid_parameters.max()>valid_parameters.min() else valid_parameters.min()+.01)
        cmap=matplotlib.colormaps['viridis']
        fig,axes=plt.subplots(len(active_branches),2,figsize=(10,4*len(active_branches)),squeeze=False,constrained_layout=True)
        for row,branch in enumerate(active_branches):
            subset=data[data.branch_mode==branch].sort_values('K0_MeV')
            eos_curves=[];mr_curves=[]
            for item in subset.itertuples():
                directory=Path(item.directory);color=cmap(norm(item.parameter_value))
                if item.eos_ok:
                    eos=pd.read_csv(directory/'eos.csv')
                    eos=eos[np.isfinite(eos.energy_density)&np.isfinite(eos.pressure)&(eos.energy_density>0)&(eos.pressure>0)].sort_values('energy_density').drop_duplicates('energy_density')
                    if len(eos):
                        eos_curves.append(eos)
                        axes[row,0].plot(eos.energy_density/1000,eos.pressure/1000,color=color,ls=STYLES[branch],lw=.9)
                if item.tov_ok:
                    mr=stable_curve(pd.read_csv(directory/'mass_radius.csv'))
                    mr=mr.sort_values('mass_msun').drop_duplicates('mass_msun')
                    if len(mr):
                        mr_curves.append(mr)
                        axes[row,1].plot(mr.radius_km,mr.mass_msun,color=color,ls=STYLES[branch],lw=.9)
            if len(eos_curves)>1:
                lower=max(c.energy_density.min() for c in eos_curves);upper=min(c.energy_density.max() for c in eos_curves)
                if upper>lower:
                    grid=np.linspace(lower,upper,240)
                    predictions=np.array([np.interp(grid,c.energy_density,c.pressure) for c in eos_curves])
                    axes[row,0].fill_between(grid/1000,predictions.min(axis=0)/1000,predictions.max(axis=0)/1000,color='.5',alpha=.2,label='Sampled envelope')
            if len(mr_curves)>1:
                lower=max(c.mass_msun.min() for c in mr_curves);upper=min(c.mass_msun.max() for c in mr_curves)
                if upper>lower:
                    grid=np.linspace(lower,upper,240)
                    predictions=np.array([np.interp(grid,c.mass_msun,c.radius_km) for c in mr_curves])
                    axes[row,1].fill_betweenx(grid,predictions.min(axis=0),predictions.max(axis=0),color='.5',alpha=.2,label='Sampled envelope')
            axes[row,0].set(xlabel=r'$\epsilon$ [GeV fm$^{-3}$]',ylabel=r'$P$ [GeV fm$^{-3}$]',title=BRANCH_LABELS[branch]+f': {len(eos_curves)}/{expected} EoS')
            axes[row,1].set(xlabel=r'$R$ [km]',ylabel=r'$M$ [$M_\odot$]',xlim=(8,18),ylim=(.5,max(2.0,float(data.maximum_mass_msun.max())*1.06)),title=BRANCH_LABELS[branch]+f': {len(mr_curves)}/{expected} TOV')
            for ax in axes[row]:
                ax.grid(alpha=.15)
                if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
        scalar=matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap)
        fig.colorbar(scalar,ax=axes,label=xlabel,shrink=.85)
        fig.suptitle(title+r': $\Lambda=300$ MeV, $250\leq K_0\leq315$ MeV'+'\nStable M–R segments; shaded envelope uses available samples only.',fontsize=11)
        finish(fig,key+'_eos_mass_radius')

        fig,axes=plt.subplots(len(controls['y_values']),3,figsize=(12,2.7*len(controls['y_values'])),squeeze=False,constrained_layout=True)
        endpoints=data[data.K0_MeV.isin([250.,315.])].dropna(subset=['parameter_value'])
        have_bounds={250.,315.}.issubset(set(endpoints.K0_MeV))
        for row,y in enumerate(controls['y_values']):
            for branch in active_branches:
                part=critical[(critical.model_key==key)&(critical.branch_mode==branch)&np.isclose(critical.y,y)]
                part=part.set_index('K0_MeV').reindex(values)
                # Reindex preserves missing samples as breaks, never bridges failures.
                for ax,xfield,yfield in ((axes[row,0],'parameter_value','nc_fm3'),(axes[row,1],'parameter_value','Tc_MeV'),(axes[row,2],'Tc_MeV','nc_fm3')):
                    ax.plot(part[xfield],part[yfield],ls=STYLES[branch],marker='o',ms=2.8,color=COLORS[branch],label=BRANCH_LABELS[branch])
                for ax in axes[row]:ax.grid(alpha=.15)
            axes[row,0].set(xlabel=xlabel,ylabel=r'$n_c$ [fm$^{-3}$]',title=f'y = {y:g}')
            axes[row,1].set(xlabel=xlabel,ylabel=r'$T_c$ [MeV]',title=f'y = {y:g}')
            axes[row,2].set(xlabel=r'$T_c$ [MeV]',ylabel=r'$n_c$ [fm$^{-3}$]',title=f'y = {y:g}')
            if have_bounds:
                for ax in axes[row,:2]:ax.axvspan(endpoints.parameter_value.min(),endpoints.parameter_value.max(),color='.6',alpha=.12)
        axes[0,0].legend(fontsize=7)
        completed=int(((critical.model_key==key)&(critical.status=='ok')).sum())
        fig.suptitle(title+f': fixed-composition critical points ({completed}/{expected*len(active_branches)*len(controls["y_values"])} solved)\n'
            +'Gray: empirical parameter interval. Markers: solved points; gaps: missing/failed.',fontsize=11)
        finish(fig,key+'_critical_branches')
    return paths
