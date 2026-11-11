"""Recompute the main-folder incompressibility figures from the QMM functional.

Run from QMM: .venv/bin/python scripts/reproduce_incompressibility_paper.py
The notebook calls the same functions; no calculations happen on import.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'src/TOVsolver')]
os.environ.setdefault('MPLCONFIGDIR', '/tmp/qmm_paper_reproduction_mpl')

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import empirical_asymmetric_suite as suite
from qmm.asymmetry import AsymmetricFitResult, compute_asymmetric_fit
from qmm.paper_reproduction import PaperEOS, thermodynamics, species_power_couplings
from qmm.quarkyonic import AsymmetricQuarkyonicEOS

DEFAULT_OUTPUT = ROOT/'results/generated/paper_replication_lambda300'
DEFAULT_PAPER = ROOT.parents[1]/'Effects_of_Nuclear_Incompressibility_on_Quarkyonic_Neutron_Stars'
DEFAULTS = dict(lambda_mev=300., branch='target_l', density_step=.004,
                dieterici_attraction='species_power',
                maximum_density=2.4, extension_step=.02, fq_scan_points=61,
                beta_y_scan_points=25, refine_tol=2e-10, integral_points=400,
                tov_points=90, tov_step_cm=500., tov_rmax_km=50.)
EVS = ('tvm','cs','vdw')
TITLES = dict(tvm='Trivirial model', cs='Carnahan–Starling', vdw='van der Waals')
COLORS = {('dieterici',220.):'#32cd32', ('dieterici',300.):'#006400',
          ('clausius',220.):'#f08080', ('clausius',300.):'#8b0000', ('reference',None):'black'}


def make_jobs(output=DEFAULT_OUTPUT, controls=None):
    controls = {**DEFAULTS, **(controls or {})}
    if controls['branch'] not in ('target_l','equal_b'):
        raise ValueError('Choose target_l or equal_b explicitly.')
    output = Path(output)
    suite.register()
    source_files = sorted((ROOT/'src/qmm').glob('*.py'))+sorted((ROOT/'src/TOVsolver').glob('*.py'))
    source_files += [Path(__file__), ROOT/'scripts/empirical_asymmetric_suite.py',
                     ROOT/'scripts/plot_empirical_parameter_bands.py']
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    jobs = []
    for ev in EVS:
        for family, k0 in [('dieterici',220.),('dieterici',300.),('clausius',220.),('clausius',300.),('reference',None)]:
            key = f'{family}_{ev}'
            cfg = suite.config_for(key if family != 'reference' else f'clausius_{ev}',
                                   k0 or 300., controls['branch'], {**suite.DEFAULT_CONTROLS, **controls})
            cfg = replace(cfg, quarkyonic=replace(cfg.quarkyonic,
                           beta_y_scan_points=controls['beta_y_scan_points'], refine_tol=controls['refine_tol']))
            if family == 'reference':
                # A vdW *attraction* reference for EACH excluded-volume family.
                cfg = replace(cfg, model_name=ev, parameter_value=None,
                              asymmetric=replace(cfg.asymmetric,target_k0=None))
            provenance = dict(pipeline='paper-reproduction-v3', config=asdict(cfg),
                              controls=controls, source_sha256=hashes)
            digest = hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()[:16]
            folder = output/'runs'/f'{key}_K{k0}_{digest}'
            suite.write_json(folder/'inputs.json', provenance)
            if (folder/'fit.json').exists():
                fit = AsymmetricFitResult(**suite.read_json(folder/'fit.json')['asymmetric_fit'])
            elif family == 'reference':
                fit = compute_asymmetric_fit(ev,fit_settings=cfg.asymmetric,
                            physical=cfg.physical,ground_state_settings=cfg.ground_state)
                suite.write_json(folder/'fit.json',dict(asymmetric_fit=asdict(fit)))
            else:
                fit = suite.ensure_fit(key,k0,controls['branch'],cfg,folder)
                if family == 'dieterici' and controls['dieterici_attraction']=='species_power':
                    fit = species_power_couplings(fit)
                    payload = suite.read_json(folder/'fit.json')
                    payload['asymmetric_fit'] = asdict(fit)
                    payload['attraction_convention'] = 'species_power; a_avg rescaled by 2**(alpha-2)'
                    suite.write_json(folder/'fit.json',payload)
            if k0 is not None:
                suite.fit_valid(fit,k0,controls['branch'])
            if abs(fit.J-32.5)>.01 or abs(fit.binding+16)>.001 or abs(fit.n_sat-.16)>1e-5:
                raise RuntimeError(f'Calibration failed: {key}')
            if controls['branch']=='target_l' and abs(fit.L-58.9)>.01:
                raise RuntimeError(f'Symmetry slope failed: {key}')
            jobs.append(dict(family=family,ev=ev,k0=k0,key=key,folder=folder,
                             config=cfg,fit=fit,controls=controls))
    pd.DataFrame([dict(family=j['family'],ev=j['ev'],requested_K0=j['k0'],**asdict(j['fit']))
                  for j in jobs]).to_csv(output/'parameters.csv',index=False)
    return jobs


def density_grid(controls):
    # Preserve familiar plotting endpoints; extend the EoS for stellar centers.
    step = controls['density_step']
    core = np.arange(.008,1.24+step/2,step)
    tail = np.arange(core[-1]+controls['extension_step'],controls['maximum_density']+1e-8,controls['extension_step'])
    return np.r_[core,tail]


def evaluator(job):
    return PaperEOS(job['fit'],job['config'].physical,job['config'].quarkyonic,
                    attraction_mode=job['controls']['dieterici_attraction'])


def independent_beta_residual(eos,n,fq,y,mu_e):
    # Fourth-order stencil resolves the steep species-power derivative near
    # Y=0; a step of Y/4 with a second-order stencil has appreciable bias.
    h=min(2e-5,y/20,(.5-y)/20)
    em2,em1,ep1,ep2=[eos.energy_hq_of_fq_y(fq,n,y+d*h) for d in (-2,-1,1,2)]
    return (em2-8*em1+8*ep1-ep2)/(12*h*n)+mu_e


def compute_eos(job, progress=True):
    folder, eos = job['folder'], evaluator(job)
    saved = suite.read_json(folder/'density_checkpoint.json',{})
    grid = density_grid(job['controls'])
    for index,n in enumerate(grid):
        tag = str(index)
        if tag in saved:
            continue
        state = eos.build_beta_row(float(n))
        if state is None:
            raise RuntimeError(f"No beta equilibrium: {job['key']}, n={n:g}")
        row = {k:v for k,v in asdict(state).items() if v is not None}
        if not np.isfinite(list(row.values())).all():
            raise RuntimeError(f'Nonfinite state at n={n:g}')
        y,fq = row['y'],row['quark_fraction']
        row['n_u'] = eos.nuid(row['k_bu'],y)  # quark NUMBER densities
        row['n_d'] = eos.ndid(row['k_bu'])
        row['n_e'] = eos.lepton_density_from_mu(row['mu_e'],.511)
        row['n_mu'] = eos.lepton_density_from_mu(row['mu_e'],105.66)
        row['charge_residual_fm3'] = row['n_p']+(2*row['n_u']-row['n_d'])/3-row['n_e']-row['n_mu']
        row['baryon_residual_fm3'] = row['n_p']+row['n_n']+(row['n_u']+row['n_d'])/3-n
        # Independent finite-difference check of the analytic beta derivative.
        row['beta_residual_fd_mev'] = independent_beta_residual(eos,n,fq,y,row['mu_e'])
        if abs(row['charge_residual_fm3'])>1e-8 or abs(row['baryon_residual_fm3'])>1e-8:
            raise RuntimeError(f'Conservation failure at n={n:g}')
        saved[tag] = row
        suite.write_json(folder/'density_checkpoint.json',saved)
        if progress and index%100==0:
            print(f"{job['key']} K={job['k0']}: density {index+1}/{len(grid)}",flush=True)
    frame = pd.DataFrame([saved[str(i)] for i in range(len(grid))])
    frame['beta_residual_fd_mev']=[independent_beta_residual(eos,r.n_b,r.quark_fraction,r.y,r.mu_e)
                                  for r in frame.itertuples()]
    for key,value in thermodynamics(frame.n_b,frame.energy_density).items():
        frame[key] = value
    frame['electron_kf_GeV'] = np.sqrt(np.maximum(frame.mu_e**2-.511**2,0))/1000
    frame['proton_fraction_total'] = frame.n_p/frame.n_b
    frame['energy_per_baryon_minus_m'] = frame.energy_density/frame.n_b-eos.physical.m_nucleon
    suite.write_csv(folder/'eos.csv',frame)
    job['eos'] = frame
    report = dict(points=len(frame),max_charge_residual_fm3=float(frame.charge_residual_fm3.abs().max()),
                  max_baryon_residual_fm3=float(frame.baryon_residual_fm3.abs().max()),
                  max_beta_residual_fd_mev=float(frame.beta_residual_fd_mev.abs().max()),
                  minimum_vs2=float(frame.vs2.min()),maximum_vs2=float(frame.vs2.max()))
    suite.write_json(folder/'eos_meta.json',report)
    return frame


def check_peak_resolution(job):
    """Actually re-solve twice as densely around the sound-speed maximum."""
    path = job['folder']/'peak_convergence.json'
    if path.exists():
        return suite.read_json(path)
    frame, eos = job['eos'], evaluator(job)
    visible = frame[frame.n_b.between(.1,1.19)]
    row = visible.loc[visible.vs2.idxmax()]
    step = job['controls']['density_step']
    grid = np.arange(max(.04,row.n_b-8*step),row.n_b+8.01*step,step/2)
    fine = [eos.build_beta_row(float(n)) for n in grid]
    if any(r is None for r in fine):
        raise RuntimeError('Peak refinement lost a beta solution.')
    energy = np.array([r.energy_density for r in fine])
    refined = thermodynamics(grid,energy)
    usable = np.arange(3,len(grid)-3)
    i = usable[np.argmax(refined['vs2'][usable])]
    data = dict(coarse_density_step_fm3=step,fine_density_step_fm3=step/2,
                coarse_peak_cs2=float(row.vs2),coarse_peak_density_fm3=float(row.n_b),
                fine_peak_cs2=float(refined['vs2'][i]),fine_peak_density_fm3=float(grid[i]),
                peak_relative_change=float(abs(refined['vs2'][i]-row.vs2)/abs(refined['vs2'][i])),
                fine_peak_at_scan_edge=bool(i in (usable[0],usable[-1])))
    suite.write_csv(job['folder']/'peak_refinement.csv',dict(n_b=grid,energy_density=energy,
                    quark_fraction=[r.quark_fraction for r in fine],**refined))
    suite.write_json(path,data)
    return data


def compute_stars(job, tides=True):
    """Use the bundled BPS stitch and TOV/Love solver; expose all exclusions."""
    from TOV_solver_code import stitch_crust,solve_sequence,radius_at_mass
    from TOVsolver.tov import TOV
    folder, controls, frame = job['folder'],job['controls'],job['eos']
    if (folder/'mass_radius.csv').exists():
        job['stars'] = pd.read_csv(folder/'mass_radius.csv')
    else:
        # Reject an unstable core instead of silently deleting its samples.
        core = frame[frame.n_b >= .04].copy()
        if (core.pressure<=0).any() or not np.all(np.diff(core.pressure)>0) or not np.all(np.diff(core.energy_density)>0):
            raise RuntimeError('Nonmonotone core: phase construction required before TOV.')
        energy,pressure,et,pt = stitch_crust(core.energy_density.to_numpy(),core.pressure.to_numpy())
        suite.write_csv(folder/'stitched_eos.csv',dict(energy_density=energy,pressure=pressure))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            rows = solve_sequence(energy,pressure,et,controls['tov_points'],controls['tov_rmax_km'],controls['tov_step_cm'])
        stars = pd.DataFrame(rows)
        stars['central_density_over_n0'] = np.interp(stars.central_energy_density_mev_fm3,frame.energy_density,frame.n_over_n0)
        stars['central_quark_fraction'] = np.interp(stars.central_energy_density_mev_fm3,frame.energy_density,frame.quark_fraction)
        suite.write_csv(folder/'mass_radius.csv',stars)
        peak = stars.loc[stars.mass_msun.idxmax()]
        suite.write_json(folder/'tov_meta.json',dict(maximum_mass_msun=float(peak.mass_msun),
                         radius_at_maximum_mass_km=float(peak.radius_km),
                         radius_at_1p4_msun_km=radius_at_mass(rows,1.4),
                         maximum_at_eos_boundary=bool(peak.name==len(stars)-1),
                         crust_transition_energy_mev_fm3=et,crust_transition_pressure_mev_fm3=pt,
                         removed_above_crust_rows=0,warnings=sorted(set(str(w.message) for w in caught))))
        job['stars'] = stars
    if not tides:
        return
    stitched = pd.read_csv(folder/'stitched_eos.csv')
    solver = TOV(stitched.energy_density.to_numpy().copy(),stitched.pressure.to_numpy().copy(),add_crust=False)
    saved = suite.read_json(folder/'tidal_checkpoint.json',{})
    for index,star in job['stars'].iterrows():
        if str(index) in saved:
            continue
        row = dict(central_energy_density_mev_fm3=float(star.central_energy_density_mev_fm3))
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                r,m,c,k2,*_ = solver.solve_tidal(float(star.central_energy_density_mev_fm3),
                                 rmax=controls['tov_rmax_km']*1e5,dr=controls['tov_step_cm'])
            if not np.isfinite([r,m,c,k2]).all() or not (0<r<.99*controls['tov_rmax_km'] and m>0 and 0<c<.5 and k2>0):
                raise ValueError('Invalid tidal solution')
            row.update(radius_km=float(r),mass_msun=float(m),compactness=float(c),k2=float(k2),
                       Lambda=float(2*k2/(3*c**5)),status='ok',warnings=sorted(set(str(w.message) for w in caught)))
        except Exception as exc:
            row.update(status='failed',error=str(exc))
        saved[str(index)] = row
        suite.write_json(folder/'tidal_checkpoint.json',saved)
    table = pd.DataFrame([saved[str(i)] for i in range(len(job['stars']))])
    for key in ('radius_km','mass_msun','compactness','k2','Lambda'):
        if key not in table:
            table[key] = np.nan
    suite.write_csv(folder/'tides.csv',table)
    job['tides'] = table


def label(job):
    return 'van der Waals attraction limit' if job['family']=='reference' else f"{job['family'].title()} ($K_0={job['k0']:g}$ MeV)"


def plot_all(jobs,output=DEFAULT_OUTPUT):
    import matplotlib.pyplot as plt
    output = Path(output)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':10})
    figures = []
    def draw(ax,j,table,x,y):
        data = j[table]
        ax.plot(data[x],data[y],color=COLORS[j['family'],j['k0']],
                ls='--' if j['family']=='reference' else '-',lw=1.6,label=label(j))
    def finish(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):
            path = output/f'{name}.{ext}';fig.savefig(path,dpi=180,bbox_inches='tight');figures.append(path)
        plt.close(fig)
    for ev in EVS:
        selected = [j for j in jobs if j['ev']==ev]
        for stem,specs in [
            ('fract_speed',[('eos','n_b','vs2',r'$n_B$ [fm$^{-3}$]',r'$c_s^2/c^2$'),
                            ('eos','n_b','quark_fraction',r'$n_B$ [fm$^{-3}$]',r'$n_Q/n_B$')]),
            ('charge_fermi',[('eos','n_b','y',r'$n_B$ [fm$^{-3}$]',r'$Y=\rho_Q/n_B$'),
                             ('eos','n_b','electron_kf_GeV',r'$n_B$ [fm$^{-3}$]',r'$k_{F,e}$ [GeV]')]),
            ('MR',[('eos','energy_density','pressure',r'$\varepsilon$ [MeV fm$^{-3}$]',r'$P$ [MeV fm$^{-3}$]'),
                   ('stars','radius_km','mass_msun',r'$R$ [km]',r'$M/M_\odot$')]),
            ('central_fract',[('stars','mass_msun','central_density_over_n0',r'$M/M_\odot$',r'$n_{B,c}/n_0$'),
                              ('stars','mass_msun','central_quark_fraction',r'$M/M_\odot$',r'$n_{Q,c}/n_{B,c}$')])]:
            if any(spec[0] not in j for spec in specs for j in selected):
                continue
            if stem=='fract_speed' and ev!='tvm':
                specs=specs[::-1]
            fig,axes = plt.subplots(1,2,figsize=(13,4.8))
            for ax,(table,x,y,xlab,ylab) in zip(axes,specs):
                for j in selected:
                    draw(ax,j,table,x,y)
                ax.set(xlabel=xlab,ylabel=ylab);ax.grid(alpha=.4,ls='--')
                if x=='n_b':
                    ax.set_xlim(0,1.2)
                if x=='radius_km':
                    ax.set_xlim(10,20)
                if y=='vs2':
                    ax.axhline(1/3,color='.5',ls='--',lw=1)
                if y=='central_density_over_n0':
                    ax.set_ylim(0,8.4)
                if x=='energy_density':
                    ax.set_xscale('log');ax.set_yscale('log')
                    ax.set_xlim(.1,2000);ax.set_ylim(1e-6,1e3)
                ax.legend(fontsize=8)
            fig.suptitle(TITLES[ev]+r' — $\Lambda=300$ MeV, '+selected[0]['controls']['branch'])
            finish(fig,f'{ev}_{stem}')
        if all('tides' in j for j in selected):
            fig,axes = plt.subplots(2,2,figsize=(11,8))
            for ax,(x,y,xlab,ylab) in zip(axes.flat,[('mass_msun','compactness',r'$M/M_\odot$','C'),
                    ('mass_msun','k2',r'$M/M_\odot$',r'$k_2$'),('radius_km','Lambda','R [km]',r'$\Lambda_{\mathrm{tidal}}$'),
                    ('mass_msun','Lambda',r'$M/M_\odot$',r'$\Lambda_{\mathrm{tidal}}$')]):
                for j in selected:
                    draw(ax,j,'tides',x,y)
                ax.set(xlabel=xlab,ylabel=ylab);ax.grid(alpha=.4,ls='--')
                if y=='Lambda':
                    ax.set_ylim(0,1000)
                if x=='radius_km':
                    ax.set_xlim(10,20)
            axes[0,0].legend(fontsize=8);finish(fig,f'{ev}_tidal')
    fig,axes = plt.subplots(1,2,figsize=(13,5))
    colors = {'tvm':('#32cd32','#008000'),'cs':('#ff737c','#a00000'),'vdw':('#4169e1','#00008b')}
    for j in jobs:
        if j['family']!='dieterici':
            continue
        for ax,y in zip(axes,('vs2','quark_fraction')):
            ax.plot(j['eos'].n_b,j['eos'][y],color=colors[j['ev']][int(j['k0']==300)],
                    label=f"{j['ev'].upper()} + Dieterici ($K_0={j['k0']:g}$ MeV)")
            ax.set_xlim(0,1.2);ax.set_xlabel(r'$n_B$ [fm$^{-3}$]');ax.grid(alpha=.4,ls='--');ax.legend(fontsize=8)
    axes[0].set_ylabel(r'$c_s^2/c^2$');axes[0].axhline(1/3,color='.5',ls='--')
    axes[1].set_ylabel(r'$n_Q/n_B$');finish(fig,'Speed_comparison')
    return figures


def summaries(jobs,output=DEFAULT_OUTPUT):
    rows=[]
    for j in jobs:
        row=dict(model=j['key'],K0_MeV=j['k0'],**suite.read_json(j['folder']/'eos_meta.json'))
        row.update(suite.read_json(j['folder']/'peak_convergence.json',{}))
        row.update(suite.read_json(j['folder']/'tov_meta.json',{}))
        if 'tides' in j:
            row['failed_tides']=int((j['tides'].status!='ok').sum())
        rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(Path(output)/'validation_summary.csv',index=False)
    return frame


def run(output=DEFAULT_OUTPUT,controls=None,stars=True,tides=True,convergence=True):
    jobs=make_jobs(output,controls)
    for j in jobs:
        compute_eos(j)
        if convergence:check_peak_resolution(j)
        if stars:compute_stars(j,tides)
        print('Completed',j['key'],j['k0'],flush=True)
    summaries(jobs,output);plot_all(jobs,output)
    return jobs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--density-step',type=float,default=DEFAULTS['density_step'])
    parser.add_argument('--no-stars',action='store_true')
    parser.add_argument('--no-tides',action='store_true')
    parser.add_argument('--branch',choices=('target_l','equal_b'),default='target_l')
    args=parser.parse_args()
    run(args.output,dict(density_step=args.density_step,branch=args.branch),not args.no_stars,not args.no_tides)


if __name__=='__main__':
    main()
