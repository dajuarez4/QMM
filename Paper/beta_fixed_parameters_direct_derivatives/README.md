# Beta equilibrium: derivative comparison

Six models, two branches; α=5/3, c=4.74 fm³, Λ=306 MeV.
Original data: ../TEST_fixed_parameters_beta/, calculated in September.
Total energy (including leptons) and composition are reused unchanged.
Chemical potential, pressure and squared sound speed are recomputed using
numpy.gradient twice, with edge_order=2 and no smoothing.
Energy minimization and TOV are not repeated. Original files are preserved.

Each profile has 81 densities: 61 up to 5 n0 and 20 more up to 15 n0.
Derivatives use the full grid before plots are restricted to 5 n0.
The summary compares peaks and differences up to 5 n0, including endpoints.
The previous method was verified against the nine-point cubic reconstruction.
This test grid does not establish convergence; new peaks can include numerical noise.
Negative or superluminal values are not clipped.

Run from QMM: `.venv/bin/python scripts/compare_beta_direct_derivatives.py`.
