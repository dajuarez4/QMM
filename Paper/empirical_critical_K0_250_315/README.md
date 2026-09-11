# Dense empirical critical-point scan

The scan computes symmetric hadronic liquid–gas critical points (y = 0.5) for Dieterici and Clausius with VDW, CS, and TVM repulsion. There are 14 independently solved K0 values per model: 250, 255, ..., 315 MeV (84 points). Saturation parameters are fitted separately at every point. Neither a beta-equilibrium nor a TOV calculation is required; Lambda does not enter this symmetric hadronic calculation.

Run `notebooks/empirical_critical_dense_scan.ipynb` or `.venv/bin/python scripts/compute_empirical_critical_scan.py`. The notebook can use a 1 MeV step. Successful checkpoints are reused only when the source, numerical settings, and model configuration signatures match. Failed points are retried and break plotted curves.

`critical_correlations_empirical` plots nc against K0 and Tc, following the supplied reference's layout. `critical_density_vs_alpha_c` uses the interaction parameters instead. The gray band denotes the selected 250–315 MeV K0 interval. Colored Tc bands are the range predicted by each model over the computed empirical scan, not an independent experimental Tc constraint. All connecting lines join actual computed samples.

All 84 points converged. The maximum absolute residuals are about 5.3e-11 for dP/dn and 5.7e-7 for d²P/dn². The finite-temperature momentum quadrature uses 800 intervals over 0–5 fm⁻¹ and the density derivative step is 2e-4 fm⁻³. A check at K0=315 MeV for all six models doubles the quadrature resolution to 1600 and halves the derivative step to 1e-4 fm⁻³; changes remain below 0.00005 MeV in Tc and 0.00000004 fm⁻³ in nc. Details are saved in `resolution_check.json`; this is a numerical stability check, not a physical uncertainty estimate.
