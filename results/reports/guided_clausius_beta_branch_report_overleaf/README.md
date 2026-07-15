# Guided Clausius Beta-Branch Report Package

This folder is an Overleaf-ready report built from the saved outputs of:

- `/Users/dajuarez4/Documents/QuarkMatt/good_repo/QMM/notebooks/guided_clausius_beta_branch_recompute.ipynb`

The source numerical outputs were read from:

- `/Users/dajuarez4/Documents/QuarkMatt/good_repo/QMM/results/generated/guided_clausius_beta_branch_recompute`

## Contents

- `main.tex`: report entry point for Overleaf
- `figures/`: regenerated publication-style PDF and PNG figures
- `tables/`: LaTeX tables used by `main.tex`
- `data/raw/`: copied per-run CSV and JSON outputs from the notebook scan
- `data/derived/`: combined machine-readable summary CSV tables

## Important Numerical Note

The beta-equilibrium sound-speed figures in this package were rebuilt from
the saved `energy_density(n_B)` curves using the same notebook-side
postprocessing choice:

- smoothing window = 29
- polynomial degree = 3

This matches the notebook's recomputed `v_s^2` workflow, not the original
runner-side `v_s^2` columns at lower smoothing.

## Suggested Overleaf Entry Point

Upload this whole directory or the accompanying tarball and compile:

- `main.tex`
