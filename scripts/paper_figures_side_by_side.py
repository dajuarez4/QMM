"""Match the active manuscript figures using the already validated EOS tables.

This is a presentation-only workflow: it never refits or reruns the EOS/TOV.
The source snapshot, active figure blocks and curve/table provenance are saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

os.environ.setdefault('MPLCONFIGDIR', '/tmp/qmm_paper_style_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parents[1]/'Effects_of_Nuclear_Incompressibility_on_Quarkyonic_Neutron_Stars'
SOURCE = ROOT/'docs/paper_figures_source_snapshot.tex'
RESULTS = ROOT/'results/generated/paper_replication_lambda300'
OUTPUT = ROOT/'results/generated/paper_figures_side_by_side'

# Exact line RGB values read from the supplied PNGs.
ORDER = [('dieterici',220.),('clausius',220.),('dieterici',300.),('clausius',300.),('reference',None)]
COLORS = {('dieterici',220.):'#32cd32', ('dieterici',300.):'#006400',
          ('clausius',220.):'#f08080', ('clausius',300.):'#8b0000', ('reference',None):'#000000'}
SPEED_COLORS = {'tvm':('#32cd32','#006400'), 'cs':('#f08080','#8b0000'),
                'vdw':('#4169e1','#00008b')}
EV_NAMES = {'tvm':'Trivirial Model','cs':'Carnahan-Starling','vdw':'van der Waals'}
SUPPORTED = {'Speed_comparison.png','TVM_fract_speed.png','TVM_qfract_fermi.png',
             'TVM_MR.png','TVM_central_fract.png','TVM_tidal.png',
             'CS_MR.png','vdW_MR.png','CS_tidal.png','vdW_tidal.png'}


def active_figures(source=SOURCE):
    """Read figure blocks after removing TeX line comments and disabled blocks."""
    raw = Path(source).read_text()
    text = '\n'.join(re.split(r'(?<!\\)%',line)[0] for line in raw.splitlines())
    text = re.sub(r'\\begin\{comment\}.*?\\end\{comment\}', '', text, flags=re.S)
    text = re.sub(r'\\iffalse\b.*?\\fi\b', '', text, flags=re.S)
    blocks = []
    for block in re.findall(r'\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}',text,re.S):
        files = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',block,re.S)
        if not files:
            continue
        unknown = set(files)-SUPPORTED
        if unknown:
            raise ValueError(f'Active figures need a renderer: {sorted(unknown)}')
        labels = re.findall(r'\\label\{([^}]+)\}',block)
        blocks.append(dict(number=len(blocks)+1,label=labels[0] if labels else '',images=files))
    if not blocks:
        raise ValueError('The manuscript source has no active figure blocks.')
    return blocks


def cached_runs(results=RESULTS):
    manifest = Path(results)/'active_runs.json'
    runs = {}
    for item in json.loads(manifest.read_text()):
        folder = ROOT/item['folder']
        inputs = json.loads((folder/'inputs.json').read_text())
        if inputs['controls']['lambda_mev'] != 300.:
            raise ValueError('The selected paper comparison requires Lambda=300 MeV.')
        family,ev = item['model'].rsplit('_',1)
        runs[ev,family,item['K0_MeV']] = dict(folder=folder,
            eos=pd.read_csv(folder/'eos.csv'),stars=pd.read_csv(folder/'mass_radius.csv'),
            tides=pd.read_csv(folder/'tides.csv'),stitched=pd.read_csv(folder/'stitched_eos.csv'))
    return runs


def configure_style():
    plt.rcParams.update({'font.family':'serif',
        'font.serif':['Times New Roman','Times','STIXGeneral'],
        'mathtext.fontset':'stix','font.size':10,'axes.titlesize':12,
        'axes.labelsize':10,'legend.fontsize':9,'axes.linewidth':.8,
        'lines.linewidth':1.8,'xtick.labelsize':10,'ytick.labelsize':10,
        'grid.color':'#d3d3d3','grid.linestyle':'--','grid.linewidth':.8,
        'legend.frameon':True,'legend.framealpha':.8,'legend.facecolor':'white',
        'legend.edgecolor':'#cccccc','savefig.facecolor':'white'})


def curve_label(family,k0):
    return 'van der Waals limit' if family=='reference' else f'{family.title()} (K = {k0/1000:.3f} GeV)'


def style_axis(ax):
    ax.grid(True,alpha=.95)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)


def two_panel(filename,paper=PAPER,top_title=None):
    w,h = Image.open(Path(paper)/filename).size
    # Avoid an integer canvas losing one pixel to binary floating rounding.
    size = (np.nextafter(w/100,np.inf),np.nextafter(h/100,np.inf))
    fig,axes = plt.subplots(1,2,figsize=size,dpi=100)
    if filename=='Speed_comparison.png':
        left,right,bottom,top = 74/1400,1376/1400,63/600,565/600
    elif filename=='TVM_fract_speed.png':
        left,right,bottom,top = 67/1400,1376/1400,63/600,565/600
    elif filename=='TVM_qfract_fermi.png':
        left,right,bottom,top = 69/1400,1376/1400,63/600,530/600
    else:
        left,right,bottom,top = .0577,.983,.124,.9364
    fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top,wspace=.115)
    if top_title:
        fig.suptitle(top_title,y=.981,fontsize=15)
    for ax in axes:
        style_axis(ax)
    return fig,axes


def plot_curves(ax,runs,ev,table,x,y,central_max=8.,legend=False,**plot_kwargs):
    for family,k0 in ORDER:
        run = runs[ev,family,k0]
        frame = run[table]
        if central_max is not None and table in ('stars','tides'):
            if table=='stars':
                keep = frame.central_density_over_n0 <= central_max
            else:
                cutoff = np.interp(central_max,run['eos'].n_over_n0,run['eos'].energy_density)
                keep = frame.central_energy_density_mev_fm3 <= cutoff
            frame = frame[keep]
        ax.plot(frame[x],frame[y],color=COLORS[family,k0],
                linestyle='--' if family=='reference' else '-',
                linewidth=2. if family=='reference' else 1.8,
                label=curve_label(family,k0),**plot_kwargs)
    if legend:
        ax.legend()


def render(filename,runs,paper=PAPER,central_max=8.):
    density_label = r'Baryon Density $n_B$ (fm$^{-3}$)'
    if filename=='Speed_comparison.png':
        fig,axes = two_panel(filename,paper)
        for k0 in (220.,300.):
            for ev in ('tvm','cs','vdw'):
                frame = runs[ev,'dieterici',k0]['eos']
                tag = 'vdW' if ev=='vdw' else ev.upper()
                label = f'{tag} + Dieterici (K = {k0/1000:.3f} GeV)'
                for ax,column in zip(axes,('vs2','quark_fraction')):
                    ax.plot(frame.n_b,frame[column],color=SPEED_COLORS[ev][int(k0==300.)],label=label)
        axes[0].axhline(1/3,color='#999999',ls='--',lw=1.4,label=r'Conformal limit ($c_s^2=1/3$)')
        axes[0].set(title='Squared Speed of Sound vs. Baryon Density',ylabel=r'$c_s^2$',ylim=(-.09,1.97))
        axes[1].set(title='Quark Fraction vs. Baryon Density',ylabel='Quark Fraction',ylim=(-.036,.792))
        for ax,location in zip(axes,('upper right','upper left')):
            ax.set(xlim=(0,1.2),xlabel=density_label)
            ax.xaxis.set_major_locator(MultipleLocator(.2));ax.legend(loc=location,fontsize=8)
        return fig
    if filename=='TVM_fract_speed.png':
        fig,axes = two_panel(filename,paper)
        for ax,column in zip(axes,('vs2','quark_fraction')):
            plot_curves(ax,runs,'tvm','eos','n_b',column)
            ax.set(xlim=(0,1.2),xlabel=density_label)
            ax.xaxis.set_major_locator(MultipleLocator(.2))
        axes[0].axhline(1/3,color='#999999',ls='--',lw=1.4,label=r'Conformal limit ($c_s^2=1/3$)')
        axes[0].set(title='Squared Speed of Sound vs. Baryon Density',ylabel=r'$c_s^2$',ylim=(-.034,.725))
        axes[1].set(title='Quark Fraction vs. Baryon Density',ylabel='Quark Fraction',ylim=(-.03,.647))
        for ax in axes:
            ax.legend(loc='upper left',fontsize=9)
        return fig
    if filename=='TVM_qfract_fermi.png':
        fig,axes = two_panel(filename,paper,'Trivirial Model Excluded Volume')
        for ax,column in zip(axes,('y','electron_kf_GeV')):
            plot_curves(ax,runs,'tvm','eos','n_b',column)
            ax.set(xlim=(0,1.2),xlabel=density_label)
            ax.xaxis.set_major_locator(MultipleLocator(.2))
        axes[0].set(title='Charge Fraction vs. Baryon Density',ylabel='Charge Fraction',ylim=(-.0045,.099))
        axes[1].set(title='Electron Fermi Momentum vs. Baryon Density',ylabel='Momentum (GeV)',ylim=(-.006,.227))
        axes[0].legend(loc='upper right',fontsize=9);axes[1].legend(loc='upper left',fontsize=9)
        return fig
    ev = {'TVM':'tvm','CS':'cs','vdW':'vdw'}[filename.split('_')[0]]
    if filename.endswith('_MR.png'):
        fig,axes = two_panel(filename,paper)
        # The manuscript's P(epsilon) includes the same BPS crust used by TOV.
        plot_curves(axes[0],runs,ev,'stitched','energy_density','pressure')
        plot_curves(axes[1],runs,ev,'stars','radius_km','mass_msun',central_max)
        prefix = '' if ev=='tvm' else EV_NAMES[ev]+': '
        axes[0].set(title=prefix+'Equation of State'+(' (EoS)' if prefix else ''),
                    xlabel=r'$\epsilon$ (MeV/fm$^3$)',ylabel=r'P (MeV/fm$^3$)',
                    xscale='log',yscale='log',xlim=(.1,2000),ylim=(1e-6,1100))
        axes[1].set(title=prefix+'Mass-Radius (M-R) Relations',xlabel='R (km)',
                    ylabel=r'M/M$_\odot$',xlim=(10,20),ylim=(-.04,2.97 if ev=='vdw' else 2.53))
        axes[0].legend(loc='upper left',fontsize=10);axes[1].legend(loc='upper right',fontsize=10)
        return fig
    if filename=='TVM_central_fract.png':
        fig,axes = two_panel(filename,paper)
        for ax,col in zip(axes,('central_density_over_n0','central_quark_fraction')):
            plot_curves(ax,runs,ev,'stars','mass_msun',col,central_max)
            ax.set(xlabel=r'M/M$_\odot$',xlim=(-.04,2.54));ax.legend(loc='upper left',fontsize=10)
        axes[0].set(title='Central Scaled Baryon Density vs. Stellar Mass',
                    ylabel=r'Scaled Central Baryon Density ($N_B/N_0$)',ylim=(-.4,8.4))
        axes[1].set(title='Central Quark Fraction vs. Stellar Mass',
                    ylabel=r'Central Quark Fraction ($N_q/N_B$)',ylim=(-.03,.647))
        return fig
    if filename.endswith('_tidal.png'):
        w,h = Image.open(Path(paper)/filename).size
        size = (np.nextafter(w/100,np.inf),np.nextafter(h/100,np.inf))
        fig,axes = plt.subplots(2,2,figsize=size,dpi=100)
        fig.subplots_adjust(left=.06,right=.98,bottom=.069,top=.944,wspace=.15,hspace=.185)
        fig.suptitle(('' if ev=='tvm' else EV_NAMES[ev]+': ')+'Tidal Properties',fontsize=15,y=.981)
        for ax,x,y in ((axes[0,0],'mass_msun','compactness'),(axes[0,1],'mass_msun','k2'),
                       (axes[1,0],'radius_km','Lambda'),(axes[1,1],'mass_msun','Lambda')):
            plot_curves(ax,runs,ev,'tides',x,y,central_max)
            style_axis(ax)
            ax.set(xlabel='R (km)' if x=='radius_km' else r'M/M$_\odot$',
                   ylabel={'compactness':'C = M/R','k2':r'k$_2$','Lambda':r'$\Lambda$'}[y])
        maximum = 2.95 if ev=='vdw' else 2.56
        axes[0,0].set(xlim=(-.04,maximum),ylim=(-.013,.307))
        axes[0,1].set(xlim=(-.04,maximum),ylim=(-.006,.144 if ev=='vdw' else .137))
        axes[1,0].set(xlim=(10,20),ylim=(0,1000))
        axes[1,1].set(xlim=(0,2.5),ylim=(0,1000))
        axes[1,1].legend(loc='upper right',fontsize=9)
        return fig
    raise ValueError(f'No renderer for {filename}')


def export_comparison(block,reproductions,paper=PAPER,output=OUTPUT):
    """Make a comparison page, preserving the manuscript's stacked rows."""
    names = block['images']
    ratios = [Image.open(Path(paper)/name).height/Image.open(Path(paper)/name).width for name in names]
    fig,axes = plt.subplots(len(names),2,squeeze=False,
                         figsize=(28,14*sum(ratios)+.65),gridspec_kw={'height_ratios':ratios})
    fig.subplots_adjust(left=.008,right=.992,bottom=.008,top=.966,wspace=.012,hspace=.02)
    fig.text(.25,.99,'Original del paper',ha='center',va='top',fontsize=17)
    fig.text(.75,.99,'Reproducción QMM · Λ = 300 MeV',ha='center',va='top',fontsize=17)
    for row,name in enumerate(names):
        for col,path in enumerate((Path(paper)/name,reproductions[name])):
            with Image.open(path) as im:
                axes[row,col].imshow(np.asarray(im.convert('RGB')))
            axes[row,col].set_axis_off()
    base = Path(output)/f"figure_{block['number']:02d}_side_by_side"
    for ext in ('png','pdf'):
        fig.savefig(base.with_suffix('.'+ext),dpi=100)
    plt.close(fig)
    return base.with_suffix('.png')


def build(source=SOURCE,paper=PAPER,results=RESULTS,output=OUTPUT,central_max=8.):
    output = Path(output);(output/'reproductions').mkdir(parents=True,exist_ok=True)
    blocks,runs = active_figures(source),cached_runs(results)
    configure_style()
    names = list(dict.fromkeys(name for b in blocks for name in b['images']))
    reproductions = {}
    for name in names:
        fig = render(name,runs,paper,central_max)
        for ext in ('png','pdf'):
            fig.savefig(output/'reproductions'/Path(name).with_suffix('.'+ext),dpi=100)
        plt.close(fig)
        reproductions[name] = output/'reproductions'/name
    for block in blocks:
        block['comparison'] = str(export_comparison(block,reproductions,paper,output))
    provenance = dict(source=str(Path(source).resolve()),source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
         active_figures=blocks,central_density_display_max_over_n0=central_max,
         curve_colors={f'{a}_{b}':v for (a,b),v in COLORS.items()},
         reference_linestyle='--',conformal_limit_linestyle='--',
         original_sha256={name:hashlib.sha256((Path(paper)/name).read_bytes()).hexdigest() for name in names},
         active_runs=json.loads((Path(results)/'active_runs.json').read_text()),
         context_note='Original observational bands are shown only in the original; no source contour data available.',
         calculation_note='Only cached numerical tables are plotted. No EOS/TOV recalculation or parameter changes.')
    (output/'selection_and_style.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return blocks,reproductions


def display_comparison(number,blocks,reproductions,paper=PAPER):
    """Embed the two-column page as a PNG, supported by notebook viewers."""
    from IPython.display import Image as NotebookImage,display
    block = next(b for b in blocks if b['number']==number)
    display(NotebookImage(filename=block['comparison']))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args = parser.parse_args()
    figures,_ = build(source=args.source,output=args.output)
    print(f"Exported {len(figures)} active manuscript figures to {args.output}")
