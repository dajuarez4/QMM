# Empirical asymmetric QMM: both branches, critical points, EoS and TOV

Open `notebooks/empirical_asymmetric_eos_mass_radius.ipynb` and run all cells to complete the calculation. It uses Dieterici/Clausius × VDW/CS/TVM, both `equal_b` (b_pn = b_n) and `target_l` (L = 58.9 MeV), J = 32.5 MeV, and Lambda = 300 MeV for the quarkyonic beta-equilibrium EoS.

The default grid is K0 = 250, 255, ..., 315 MeV: 168 configurations and 840 fixed-y critical points (y = 0.1, ..., 0.5). Each EoS contains 150 density points and each TOV sequence samples 260 central energy densities. The complete EoS grid is a many-hour calculation; the notebook checkpoints every completed density and can be interrupted and resumed. It uses independent worker processes, with four workers by default. Changing numerical controls isolates the cache. Stop an active run before starting another notebook over the same configurations.

## Results included in this delivery

- 36 asymmetric fits: all six models, both branches, K0 = 250, 300, 315 MeV.
- 180 newly solved fixed-y critical points at those K0 values, all converged.
- 24 compatible existing EoS tables at K0 = 250 and 300 MeV, imported with explicit source provenance.
- 24 newly reintegrated TOV sequences from those EoS tables; maximum masses differ from the prior results by at most 2.6e-8 solar masses.
- Figures in PNG/PDF/SVG and a notebook with read-only previews embedded. The long calculation cells remain unexecuted in the delivered notebook and are enabled for Run All.

**The complete 5-MeV grid has not been run.** There are 144 EoS/TOV configurations and 660 critical points still pending. `configuration_status.csv` and `fixed_y_critical_points.csv` show actual coverage. Figure titles show completed/expected counts; missing critical points leave gaps rather than artificial connecting curves. Gray EoS/M–R envelopes bound only the available sampled curves over their common domain; they are not confidence intervals. The M–R plots zoom to R = 8–18 km and M >= 0.5 solar masses; full computed sequences remain in CSV.

## Numerical diagnostics

Both branches reproduce the same y = 0.5 limit to about 1.2e-11 in Tc and nc. All computed fixed-y points satisfy |dP/dn| < 1e-4 and |d²P/dn²| < 1e-3; actual maxima are in `validation.json`. A two-density interruption/resume test at K0 = 315 MeV used real beta-equilibrium evaluations in a temporary directory and passed.

The inherited TOV integrator emits numerical integration and runtime warnings for some stars. These are captured in each `tov_meta.json`, with counts in the coverage table. Agreement of maximum masses with the old calculation is a reproducibility check, not an independent convergence or physical-validity proof for every star. The workflow retains the prior positive/monotone core-EoS filtering and records excluded samples for new TOV integrations; it does not impose a causal repair. `maximum_vs2` and `maximum_at_eos_boundary` remain available for physical review.

`runs/` contains input manifests, fits, per-density checkpoints, EoS, stitched EoS, TOV tables, critical points, stage errors and worker logs. Source paths for imported products are relative to the repository. The helper scripts are `scripts/empirical_asymmetric_suite.py` and `scripts/plot_empirical_asymmetric_suite.py`.
