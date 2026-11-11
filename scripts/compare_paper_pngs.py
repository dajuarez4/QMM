"""Digitize supplied PNG curves by color, with explicit pixel calibrations.

These are approximate reference observations, not the original numerical data.
No model calculation is used to select pixels or calibrate the axes.
Requires Pillow (available in matplotlib's dependencies).
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image

import reproduce_incompressibility_paper as rep

RGB = {('dieterici',220.):(50,205,50),('dieterici',300.):(0,100,0),
       ('clausius',220.):(240,128,128),('clausius',300.):(139,0,0),('reference',None):(0,0,0)}
# (image, observable, x_left, x_right, y_pixel_zero, pixels_per_y_unit, legend rectangle)
# Axis scales come from the labeled ticks in each original 1400x600 PNG.
PANELS = {
 'tvm':[
  ('TVM_fract_speed.png','vs2',67,687,515,662.8,(76,41,311,163)),
  ('TVM_fract_speed.png','quark_fraction',757,1376,515,741.,(765,41,1002,148)),
  ('TVM_qfract_fermi.png','y',69,684,517,4512.5,(465,76,682,192)),
  ('TVM_qfract_fermi.png','electron_kf_GeV',761,1376,525,2005.,(765,76,998,192))],
 'cs':[
  ('CS_fract_speed.png','quark_fraction',62,684,516,660.,(68,76,282,188)),
  ('CS_fract_speed.png','vs2',754,1376,517,592.8,(760,76,987,190)),
  ('CS_qfract_fermi.png','y',69,680,517,4650.,(499,76,679,178)),
  ('CS_qfract_fermi.png','electron_kf_GeV',765,1376,526,2228.,(981,435,1163,536))],
 'vdw':[
  ('vdW_fract_sound.png','quark_fraction',62,680,516,519.,(68,76,284,187)),
  ('vdW_fract_sound.png','vs2',758,1376,516,213.6,(1140,76,1375,188)),
  ('vdW_qfract_fermi.png','y',69,684,517,4875.,(499,76,682,178)),
  ('vdW_qfract_fermi.png','electron_kf_GeV',761,1376,524,1960.,(765,76,950,178))]
}


def digitize(paper,output):
    output=Path(output);target=output/'digitized_reference';target.mkdir(parents=True,exist_ok=True)
    records=[]
    for ev,panels in PANELS.items():
        for filename,observable,x0,x1,y0,scale,legend in panels:
            array=np.array(Image.open(Path(paper)/filename).convert('RGB')).astype(float)
            for (family,k0),rgb in RGB.items():
                mask=np.max(np.abs(array-np.array(rgb)),axis=2)<18
                # Keep only the axes interior; remove the legend box.
                top=35 if filename=='TVM_fract_speed.png' else 70
                mask[:top+3]=False;mask[534:]=False;mask[:,:x0+3]=False;mask[:,x1-2:]=False
                left,upper,right,lower=legend;mask[upper:lower,left:right]=False
                xs,ys=[],[]
                for x in range(x0+3,x1-2):
                    pixels=np.flatnonzero(mask[:,x])
                    if not len(pixels) or np.ptp(pixels)>12:
                        continue
                    xs.append(1.2*(x-x0)/(x1-x0));ys.append((y0-float(np.median(pixels)))/scale)
                frame=pd.DataFrame(dict(n_b=xs,**{observable:ys}))
                stem=f'{family}_{ev}_K{k0}_{observable}'
                frame.to_csv(target/f'{stem}.csv',index=False)
                records.append(dict(ev=ev,family=family,K0=k0,observable=observable,source=filename,
                                    file=str(target/f'{stem}.csv'),points=len(frame),
                                    horizontal_pixel_fm3=1.2/(x1-x0),vertical_pixel_units=1/scale))
    manifest=pd.DataFrame(records);manifest.to_csv(target/'manifest.csv',index=False)
    return manifest


def compare(jobs,paper=rep.DEFAULT_PAPER,output=rep.DEFAULT_OUTPUT):
    output=Path(output);manifest=digitize(paper,output);errors=[];legacy=[]
    for j in jobs:
        selection=manifest[(manifest.ev==j['ev'])&(manifest.family==j['family'])]
        selection=selection[selection.K0.isna()] if j['k0'] is None else selection[selection.K0==j['k0']]
        f=j['eos']
        for r in selection.itertuples():
            ref=pd.read_csv(r.file)
            ref=ref[ref.n_b.between(.1,1.18)]
            if len(ref)<3:
                continue
            delta=np.interp(ref.n_b,f.n_b,f[r.observable])-ref[r.observable]
            errors.append(dict(model=j['key'],K0_MeV=j['k0'],observable=r.observable,points=len(ref),
                               RMSE=float(np.sqrt(np.mean(delta**2))),max_abs_error=float(abs(delta).max()),
                               vertical_pixel_units=r.vertical_pixel_units,source=r.source))
            # Compare the previous equal-b notebook with the SAME digitized pixels.
            previous=rep.ROOT/'results/generated/screenshot_model_comparison/runs'
            glob='vdw_reference_None_*/eos.csv' if j['family']=='reference' else f"{j['key']}_{j['k0']}_*/eos.csv"
            paths=list(previous.glob(glob))
            if len(paths)==1:
                old=pd.read_csv(paths[0])
                old_delta=np.interp(ref.n_b,old.n_b,old[r.observable])-ref[r.observable]
                legacy.append(dict(model=j['key'],K0_MeV=j['k0'],observable=r.observable,
                                   old_RMSE=float(np.sqrt(np.mean(old_delta**2))),
                                   corrected_RMSE=float(np.sqrt(np.mean(delta**2)))))
    pd.DataFrame(errors).to_csv(output/'reference_errors.csv',index=False)
    pd.DataFrame(legacy).to_csv(output/'legacy_comparison.csv',index=False)
    # Display image-calibrated curves against computed curves without changing fits.
    import matplotlib.pyplot as plt
    for ev in rep.EVS:
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,obs in zip(axes.flat,('vs2','quark_fraction','y','electron_kf_GeV')):
            for j in [j for j in jobs if j['ev']==ev]:
                color=rep.COLORS[j['family'],j['k0']]
                f=j['eos'];ax.plot(f.n_b,f[obs],color=color,lw=1.5,label=rep.label(j))
                subset=manifest[(manifest.ev==ev)&(manifest.family==j['family'])&(manifest.observable==obs)]
                subset=subset[subset.K0.isna()] if j['k0'] is None else subset[subset.K0==j['k0']]
                ref=pd.read_csv(subset.iloc[0].file)
                # Points stay separate across legend occlusions; no extrapolation.
                ax.scatter(ref.n_b[::8],ref[obs][::8],s=9,facecolors='none',edgecolors=color,alpha=.65)
            ax.set(xlim=(0,1.2),xlabel=r'$n_B$ [fm$^{-3}$]',ylabel=obs);ax.grid(alpha=.25)
        axes[0,0].legend(fontsize=7)
        fig.suptitle(rep.TITLES[ev]+': computed lines / digitized image circles')
        fig.tight_layout();fig.savefig(output/f'{ev}_reference_overlay.png',dpi=180);plt.close(fig)
    return pd.DataFrame(errors)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=rep.DEFAULT_OUTPUT)
    parser.add_argument('--paper',type=Path,default=rep.DEFAULT_PAPER)
    args=parser.parse_args();jobs=rep.make_jobs(args.output)
    for j in jobs:
        j['eos']=pd.read_csv(j['folder']/'eos.csv')
    print(compare(jobs,args.paper,args.output).to_string(index=False))


if __name__=='__main__':
    main()
