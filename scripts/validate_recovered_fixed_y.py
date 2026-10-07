"""Resolution check of all twelve low-y endpoints at K0=760 MeV."""
from dataclasses import replace,asdict
import json
import pandas as pd
import recover_fixed_y_critical_points as recovery
suite=recovery.suite
suite.register()
base=pd.read_csv(recovery.OUTPUT/'recovered_points.csv')
base=base[base.K0_target_MeV.eq(760)&base.y.eq(.1)&base.status.eq('ok')]
assert len(base)==12, 'Wait for the recovery scan to complete.'
suite.CP_SETTINGS=replace(suite.CP_SETTINGS,n_k_fd=1600,h_n=suite.CP_SETTINGS.h_n/2)
checks=[]
for item in base.itertuples():
    config=suite.config_for(item.model_key,760,item.branch_mode,suite.DEFAULT_CONTROLS)
    directory=recovery.OUTPUT/f'{item.model_key}_K0_760_{item.branch_mode}'
    fit=suite.ensure_fit(item.model_key,760,item.branch_mode,config,directory)
    refined=recovery.solve(fit,config,.1,[(item.Tc_MeV,item.nc_fm3)])
    assert refined['status']=='ok', refined
    checks.append(dict(model_key=item.model_key,branch_mode=item.branch_mode,
        delta_Tc_MeV=refined['Tc_MeV']-item.Tc_MeV,
        delta_nc_fm3=refined['nc_fm3']-item.nc_fm3,
        delta_Pc_MeV_fm3=refined['Pc_MeV_fm3']-item.Pc_MeV_fm3,
        refined=refined))
    print(item.model_key,item.branch_mode,checks[-1]['delta_Tc_MeV'],flush=True)
(recovery.OUTPUT/'resolution_check.json').write_text(json.dumps(dict(settings=asdict(suite.CP_SETTINGS),checks=checks),indent=2)+'\n')
assert max(abs(c['delta_Tc_MeV']) for c in checks)<1e-3
assert max(abs(c['delta_nc_fm3']) for c in checks)<1e-5
print('All twelve endpoints passed resolution checks.')
