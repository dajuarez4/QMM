"""Recheck the four discontinuous high-density points used by the combined bands.

Uses the existing 141-point global quark-fraction scan, then refines every
resolved local minimum. The established solver's first finite golden-section
answer is not sufficient when the objective has disconnected valid regions.
Only a new output copy is written; the original scan is retained.
"""
from pathlib import Path
import sys
import json
import hashlib
import math
from dataclasses import asdict
from types import MethodType

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/'src',ROOT/'src/TOVsolver',ROOT/'scripts'):
    sys.path.insert(0,str(p))
import pandas as pd
from plot_empirical_parameter_bands import register_models
from qmm.config import load_run_config
from qmm.asymmetry import AsymmetricFitResult
from qmm.quarkyonic import AsymmetricQuarkyonicEOS
from qmm.numerics import golden_section_min_safe
from clausius_asymmetric_tov import local_thermodynamics


def repair(source,destination):
    config_path=ROOT/'examples/generated/guided_dieterici_tvm_lambda300_k0_200_750_step50_k0_350_target_l.json'
    summary_path=ROOT/'results/generated/guided_dieterici_tvm_lambda300_k0_200_750_step50/K0_350_target_l/guided_dieterici_tvm_lambda300_k0_200_750_step50_k0_350_target_l_summary.json'
    digest=hashlib.sha256(source.read_bytes()+config_path.read_bytes()+summary_path.read_bytes()).hexdigest()
    audit_path=destination.with_suffix('.json')
    if destination.exists() and audit_path.exists() and json.loads(audit_path.read_text()).get('source_digest')==digest:
        return pd.read_csv(destination)
    register_models()
    config=load_run_config(config_path)
    fit=AsymmetricFitResult(**json.loads(summary_path.read_text())['asymmetric_fit'])
    evaluator=AsymmetricQuarkyonicEOS(fit,config.physical,config.quarkyonic)
    def global_scan(self,n_b,fq_min=0.,fq_max=1.,y_scan=None):
        lower=max(fq_min,self.beta_lower_quark_fraction_bound(n_b))
        (best_fq,best_energy),intervals=self._coarse_beta_candidates(n_b,lower,fq_max,y_scan)
        if not math.isfinite(best_energy):raise ValueError('No finite energy in global scan')
        objective=lambda f:self.total_energy_beta(f,n_b,y_scan=y_scan)
        for left,right in intervals:
            f,e=golden_section_min_safe(objective,left,right,tol=self.settings.refine_tol,max_iter=self.settings.refine_max_iter)
            if math.isfinite(e) and e<best_energy:best_fq,best_energy=f,e
        return best_fq,best_energy
    evaluator.solve_fq_beta=MethodType(global_scan,evaluator)
    frame=pd.read_csv(source);audit=[]
    for i in frame.index[frame.n_over_n0>=14-1e-8]:
        old=frame.loc[i].copy();row=evaluator.build_beta_row(float(old.n_b))
        if row is None:raise RuntimeError('Global energy search failed')
        values=asdict(row)
        if row.energy_density>=old.energy_density:raise RuntimeError('No lower-energy replacement; inspect manually')
        for field in frame.columns:
            if field in values and values[field] is not None:frame.at[i,field]=values[field]
        audit.append({'n_over_n0':row.n_over_n0,'old_energy_density':float(old.energy_density),
            'new_energy_density':row.energy_density,'old_fq':float(old.quark_fraction),'new_fq':row.quark_fraction})
        print('Rechecked high-density minimum:',audit[-1],flush=True)
    rows=frame.to_dict('records');local_thermodynamics(rows)
    frame=pd.DataFrame(rows)
    destination.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(destination,index=False)
    audit_path.write_text(json.dumps({'source_digest':digest,'source':str(source.relative_to(ROOT)),
        'method':'141-point global fq scan plus refinement of local minima; original physical settings',
        'replacements':audit},indent=2)+'\n')
    return pd.read_csv(destination)
