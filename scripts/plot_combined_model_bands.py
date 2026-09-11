"""Combined six-model bands, K0-colored markers and black vdW lines."""
from pathlib import Path
import json
import hashlib
import sys
import warnings
import os

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/'src',ROOT/'src/TOVsolver'):
    sys.path.insert(0,str(p))
os.environ.setdefault('MPLCONFIGDIR','/tmp/qmm_mpl_cache')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.collections import PolyCollection
from scipy.integrate import ODEintWarning
from TOV_solver_code import stitch_crust
from TOVsolver.tov import TOV

OUTPUT=ROOT/'Paper/combined_models_K0_200_350'
K0_VALUES=[200,250,300,350]
BRANCHES=['target_l','equal_b']
SPECS={
    'dieterici_vdw':('Dieterici–vdW','D','#729eaa',''),
    'dieterici_cs':('Dieterici–CS','P','#449977','/'),
    'dieterici_tvm':('Dieterici–TVM','X','#779944','\\'),
    'clausius_vdw':('Clausius–vdW','o','#aa7799',''),
    'clausius_cs':('Clausius–CS','s','#aa7744','/'),
    'clausius_tvm':('Clausius–TVM','^','#7766aa','\\'),
}
BRANCH_LABEL={'target_l':r'$b_{pn}\ne b_n$, $L=58.9$ MeV','equal_b':r'$b_{pn}=b_n$'}


def recover_missing_tov(eos,path):
    """An invalid low-density star must not abort the entire sequence."""
    rows=eos[np.isfinite(eos.energy_density)&np.isfinite(eos.pressure)&(eos.energy_density>0)&(eos.pressure>0)].sort_values('energy_density')
    kept=[];last_e=last_p=-np.inf
    for r in rows.itertuples():
        if r.energy_density>last_e and r.pressure>last_p:
            kept.append((r.energy_density,r.pressure));last_e=r.energy_density;last_p=r.pressure
    energy,pressure=np.asarray(kept).T
    energy,pressure,transition,_=stitch_crust(energy,pressure)
    solver=TOV(energy.copy(),pressure.copy(),add_crust=False,plot_eos=False)
    results=[]
    for central in np.geomspace(transition*1.001,energy[-1]*(1-1e-10),260):
        row={'central_energy_density_mev_fm3':float(central),'central_pressure_mev_fm3':float(np.interp(central,energy,pressure)),
             'radius_km':np.nan,'mass_msun':np.nan,'status':'failed','error':''}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                radius,mass,_=solver.solve(float(central),rmax=50e5,dr=250.)
            if any(issubclass(w.category,ODEintWarning) for w in caught):
                raise RuntimeError('ODE integrator reported nonconvergence')
            if not np.isfinite(radius+mass) or not (0<radius<49.5 and mass>0):
                raise RuntimeError('Invalid star or radius boundary reached')
            row.update(radius_km=float(radius),mass_msun=float(mass),status='ok')
        except Exception as exc:row['error']=str(exc)
        results.append(row)
    frame=pd.DataFrame(results)
    if frame.status.eq('ok').sum()<4:raise RuntimeError('Insufficient converged stars')
    peak=int(frame.mass_msun.idxmax())
    frame['stable_branch']=(frame.index<=peak)&frame.status.eq('ok')
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False)
    path.with_suffix('.json').write_text(json.dumps({'eos_digest':hashlib.sha256(eos.to_csv(index=False).encode()).hexdigest(),
        'valid_stars':int(frame.status.eq('ok').sum()),'rejected_stars':int(frame.status.ne('ok').sum())},indent=2)+'\n')
    return frame


def load_curves(recover=True):
    curves={};coverage=[]
    for key in SPECS:
        source=ROOT/'results/generated'/f'guided_{key}_lambda300_k0_200_750_step50_tov'
        for branch in BRANCHES:
            for k0 in K0_VALUES:
                ep=source/'extended_eos'/f'K0_{k0}_{branch}_extended_beta_eos.csv'
                mp=source/'curves'/f'K0_{k0}_{branch}_mass_radius.csv'
                eos=pd.read_csv(ep) if ep.exists() else None
                if key=='dieterici_tvm' and branch=='target_l' and k0==350:
                    corrected=OUTPUT/'recovered'/f'{key}_K0_{k0}_{branch}_eos.csv'
                    if recover:
                        from repair_combined_band_eos import repair
                        eos=repair(ep,corrected);ep=corrected
                    elif corrected.exists():
                        eos=pd.read_csv(corrected);ep=corrected
                recovered=OUTPUT/'recovered'/f'{key}_K0_{k0}_{branch}_mass_radius.csv'
                if mp.exists():mr=pd.read_csv(mp);mr_source=mp
                elif recovered.exists() and recovered.with_suffix('.json').exists() and json.loads(recovered.with_suffix('.json').read_text()).get('eos_digest')==hashlib.sha256(eos.to_csv(index=False).encode()).hexdigest():mr=pd.read_csv(recovered);mr_source=recovered
                elif recover and eos is not None:mr=recover_missing_tov(eos,recovered);mr_source=recovered
                else:mr=None;mr_source=None
                curves[key,branch,k0]={'eos':eos,'mr':mr}
                coverage.append({'model':key,'branch':branch,'K0_MeV':k0,
                    'eos_points':len(eos) if eos is not None else 0,
                    'max_n_over_n0':float(eos.n_over_n0.max()) if eos is not None else np.nan,
                    'valid_stars':int(mr.mass_msun.notna().sum()) if mr is not None else 0,
                    'eos_source':str(ep.relative_to(ROOT)) if eos is not None else '',
                    'mr_source':str(mr_source.relative_to(ROOT)) if mr_source else ''})
    OUTPUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(coverage).to_csv(OUTPUT/'coverage.csv',index=False)
    return curves,pd.DataFrame(coverage)


def samples(curve,observable):
    if observable=='mr':
        frame=curve['mr']
        if frame is None:return None
        return (frame.central_energy_density_mev_fm3.to_numpy(),frame.radius_km.to_numpy(),frame.mass_msun.to_numpy())
    frame=curve['eos']
    if frame is None:return None
    if observable=='eos':return (frame.energy_density.to_numpy()/1000,frame.energy_density.to_numpy()/1000,frame.pressure.to_numpy()/1000)
    return (frame.n_over_n0.to_numpy(),frame.n_over_n0.to_numpy(),frame['vs2' if observable=='sound' else 'quark_fraction'].to_numpy())


def align(data,count=350):
    lower=max(d[0][np.isfinite(d[1])&np.isfinite(d[2])].min() for d in data)
    upper=min(d[0][np.isfinite(d[1])&np.isfinite(d[2])].max() for d in data)
    if upper<=lower:raise ValueError('No common domain for the model band')
    grid=np.linspace(lower,upper,count)
    coords=[]
    for t,x,y in data:
        valid=np.isfinite(t)&np.isfinite(x)&np.isfinite(y)
        order=np.argsort(t[valid]);tt=t[valid][order]
        xx=np.interp(grid,tt,x[valid][order]);yy=np.interp(grid,tt,y[valid][order])
        # Keep rejected-star gaps; do not silently connect across failed solves.
        for i in np.flatnonzero(~valid):
            left=t[max(0,i-1)];right=t[min(len(t)-1,i+1)]
            mask=(grid>=left)&(grid<=right);xx[mask]=np.nan;yy[mask]=np.nan
        coords.append((xx,yy))
    return grid,coords


def draw(ax,curves,branch,observable,center_k0):
    norm=matplotlib.colors.Normalize(200,350);cmap=matplotlib.colormaps['turbo']
    interpolation_used=False
    for key,(label,marker,fill,hatch) in SPECS.items():
        entries=[(k,samples(curves[key,branch,k],observable)) for k in K0_VALUES]
        entries=[(k,d) for k,d in entries if d is not None]
        if not entries:continue
        grid,aligned=align([d for _,d in entries]);ks=[k for k,_ in entries]
        if observable=='mr':
            # Parameterize by central energy, not radius: descending branches fold in R.
            polygons=[]
            for (x1,y1),(x2,y2) in zip(aligned[:-1],aligned[1:]):
                for i in range(len(grid)-1):
                    vertices=[(x1[i],y1[i]),(x1[i+1],y1[i+1]),(x2[i+1],y2[i+1]),(x2[i],y2[i])]
                    if np.isfinite(vertices).all():polygons.append(vertices)
            ax.add_collection(PolyCollection(polygons,facecolor=fill,alpha=.09,edgecolors='none',zorder=1))
        else:
            values=np.array([a[1] for a in aligned])
            ax.fill_between(grid,np.min(values,axis=0),np.max(values,axis=0),facecolor=fill,
                            edgecolor=fill,hatch=hatch,linewidth=.15,alpha=.13,zorder=1)
        # Colored boundary markers identify the sampled K0 endpoints.
        for k0,data in (entries[0],entries[-1]):
            _,x,y=data
            ax.plot(x,y,color='black' if key.endswith('_vdw') else cmap(norm(k0)),lw=.45,alpha=.65,
                    marker=marker,ms=2.8,markevery=max(1,len(x)//7),mfc=cmap(norm(k0)),mec='black',mew=.2,zorder=2)
        if center_k0 in ks:
            _,x,y=entries[ks.index(center_k0)][1]
        else:
            left=max(k for k in ks if k<center_k0);right=min(k for k in ks if k>center_k0)
            _,coords=align([entries[ks.index(left)][1],entries[ks.index(right)][1]])
            weight=(center_k0-left)/(right-left)
            x=(1-weight)*coords[0][0]+weight*coords[1][0]
            y=(1-weight)*coords[0][1]+weight*coords[1][1]
            interpolation_used=True
        ax.plot(x,y,color='black' if key.endswith('_vdw') else cmap(norm(center_k0)),lw=1.15,
            marker=marker,ms=4,markevery=max(1,len(x)//11),mfc=cmap(norm(center_k0)),mec='black',mew=.45,zorder=4)
    ax.grid(alpha=.16)
    if observable=='sound':ax.set(xlabel=r'$n_B/n_0$',ylabel=r'$c_s^2$',xlim=(0,15))
    elif observable=='fq':ax.set(xlabel=r'$n_B/n_0$',ylabel=r'$f_q$',xlim=(0,15),ylim=(0,1))
    elif observable=='eos':ax.set(xlabel=r'$\epsilon$ [GeV fm$^{-3}$]',ylabel=r'$P$ [GeV fm$^{-3}$]')
    else:ax.set(xlabel=r'$R$ [km]',ylabel=r'$M$ [$M_\odot$]')
    ax.set_title(BRANCH_LABEL[branch],fontsize=10)
    return interpolation_used


def decorate(fig,axes,center_k0,interpolated):
    handles=[Line2D([],[],marker=s[1],color='black' if k.endswith('_vdw') else '.5',mfc=s[2],mec='black',lw=1,label=s[0]) for k,s in SPECS.items()]
    fig.legend(handles=handles,loc='outside lower center',ncol=6,fontsize=8)
    norm=matplotlib.colors.Normalize(200,350)
    scalar=matplotlib.cm.ScalarMappable(norm=norm,cmap='turbo')
    fig.colorbar(scalar,ax=np.asarray(axes).ravel().tolist(),label=r'$K_0$ [MeV]',shrink=.83,pad=.015,ticks=[200,250,center_k0,300,350] if center_k0!=300 else [200,250,300,350])
    note=f'Central line: K₀={center_k0:g} MeV'+(' (interpolated display guide)' if interpolated else ' (computed)')
    fig.suptitle(r'All six models, $\Lambda=300$ MeV: $200\leq K_0\leq350$ MeV'+'\n'+note+'; markers: models; black lines: vdW repulsion.',fontsize=12)


def save(fig,name):
    for ext in ('png','pdf','svg'):
        path=OUTPUT/f'{name}.{ext}';fig.savefig(path,dpi=180,bbox_inches='tight')
        if ext=='svg':path.write_text('\n'.join(l.rstrip() for l in path.read_text().splitlines())+'\n')
    plt.close(fig)
    return OUTPUT/f'{name}.png'


def main(center_k0=275.,recover=True):
    if not 200<center_k0<350:raise ValueError('Choose a central K0 strictly inside 200–350 MeV')
    curves,coverage=load_curves(recover)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':9})
    fields=['sound','mr','eos','fq'];paths=[]
    fig,axes=plt.subplots(2,4,figsize=(20,9),layout='constrained');interpolated=False
    for row,branch in enumerate(BRANCHES):
        for col,field in enumerate(fields):interpolated|=draw(axes[row,col],curves,branch,field,center_k0)
    decorate(fig,axes,center_k0,interpolated);paths.append(save(fig,'all_models_all_observables'))
    for field in fields:
        fig,axes=plt.subplots(1,2,figsize=(13,5.8),layout='constrained');interpolated=False
        for ax,branch in zip(axes,BRANCHES):
            interpolated|=draw(ax,curves,branch,field,center_k0)
            if field=='mr':
                detail=ax.inset_axes([.39,.34,.57,.59])
                draw(detail,curves,branch,field,center_k0)
                detail.set(xlim=(8,16),ylim=(1,3),xlabel='',ylabel='',title='Mass–radius detail')
                detail.tick_params(labelsize=7)
                detail.title.set_fontsize(8)
        decorate(fig,axes,center_k0,interpolated);paths.append(save(fig,'all_models_'+field))
    (OUTPUT/'plot_settings.json').write_text(json.dumps({'K0_samples':K0_VALUES,'center_K0_MeV':center_k0,
        'center_interpolated':center_k0 not in K0_VALUES,'lambda_MeV':300,'full_mass_radius_sequences':True},indent=2)+'\n')
    return paths,coverage


if __name__=='__main__':
    main()
