# QMM

`QMM` is a clean, self-contained workflow for nuclear matter, quarkyonic matter, baryquark matter, asymmetric matter, leptonic beta equilibrium, and neutron stars. It is organized so that the same package can be used in two ways:

1. as a production workflow for the models already implemented in `src/qmm`
2. as a framework that you can extend with a new mean field and a new excluded-volume prescription

## Layout

- `src/qmm/`: documented production source code
- `notebooks/QMM_workflow.ipynb`: the single overview notebook
- `notebooks/clausius_beta_vs2_postprocess.ipynb`: Clausius beta-equilibrium `v_s^2` postprocessing notebook
- `examples/`: JSON inputs, including a blank template
- `docs/qmm_complete_workflow_manual.tex`: formal workflow manual
- `results/`: curated local reference outputs used by the notebook

## What QMM Can Control From JSON

The JSON file decides which physics blocks are executed and which outputs are written.

| Block | Main controls | What it does |
| --- | --- | --- |
| `model` | `name`, `parameter_value`, `parameter_search` | Choose the real-gas model and optional extra parameter |
| `workflows` | booleans | Turn on or off symmetric ground state, liquid critical point, symmetric quarkyonic matter, asymmetric fit, asymmetric quarkyonic matter, and neutron stars |
| `hadronic_eos` | density and smoothing controls | Control the dense pure-hadronic EOS table and its reconstructed `v_s^2` |
| `physical` | constants | Override `n0`, `m_nucleon`, `J`, `L`, degeneracies, binding target |
| `ground_state` | scan/derivative settings | Control the symmetric `a,b` fit |
| `quantum` | SCF and search settings | Control the finite-temperature liquid critical-point solver |
| `quarkyonic` | density range, quadrature, `lambda_momentum_mev`, `momentum_mode` | Control the symmetric/asymmetric quarkyonic EOS construction |
| `asymmetric` | `branch_mode`, `target_j`, `target_l`, `beta_equilibrium` | Choose `b_pn=b_n` or `b_pn!=b_n`, fixed-`y` scans, and leptons on/off |
| `neutron_star` | TOV grid settings | Control the mass-radius sequence |
| `output` | output flags | Decide whether CSV, JSON, and plots are written |

The executable blank template is [examples/qmm_blank_template.json](/Users/dajuarez4/Documents/QuarkMatt/QMM/examples/qmm_blank_template.json).

The richer all-in-one template is [examples/qmm_all_in_one_template.json](/Users/dajuarez4/Documents/QuarkMatt/QMM/examples/qmm_all_in_one_template.json). It includes:

- LaTeX fields for the mean field and excluded-volume equations
- a human-readable calculation summary
- the actual runnable workflow blocks

## What The JSON Can And Cannot Do

The JSON file can:

- choose a model that is already registered in `src/qmm/models.py`
- solve symmetric `a,b`
- solve an extra model parameter from a target `K0`
- compute the liquid critical point
- compute symmetric quarkyonic matter
- switch between `quarkyonic` and `baryquark` momentum-space fillings for the symmetric solver
- compute asymmetric couplings
- compute fixed-`y` asymmetric quarkyonic matter
- include or exclude leptons through `beta_equilibrium`
- compute or skip neutron stars
- save or skip CSV, JSON, and plot outputs

The all-in-one template can also carry:

- `documentation.mean_field_latex`
- `documentation.excluded_volume_latex`
- `documentation.ideal_density_map_latex`
- `documentation.pressure_prefactor_latex`
- `requested_calculations.*`

These fields are safe to keep in the same JSON file because `QMM` ignores unknown descriptive keys at runtime.

The JSON file cannot, by itself:

- define a brand-new analytic mean field `U(n)`
- define a brand-new excluded-volume map
- add a symbolic new model without one code registration step

For a genuinely new model, you must first register the physics in [src/qmm/models.py](/Users/dajuarez4/Documents/QuarkMatt/QMM/src/qmm/models.py). After that, the same JSON interface works for it.

## Quarkyonic Versus Baryquark Momentum Modes

The `quarkyonic` block now accepts:

- `momentum_mode = "quarkyonic"`
- `momentum_mode = "baryquark"`

These correspond to two opposite momentum-space fillings:

- `quarkyonic`: quark Fermi sea plus baryonic shell
- `baryquark`: baryonic Fermi sea plus quark shell

The current `QMM` implementation supports:

- symmetric baryquark matter

and intentionally does not support yet:

- asymmetric baryquark matter
- beta-equilibrium baryquark matter
- neutron stars built from baryquark beta-equilibrium EOS

This restriction is deliberate: the asymmetric baryquark extension needs an explicit prescription for the proton/neutron and `u/d` shell boundaries, and that is not hardcoded here without a dedicated model choice.

## Workflow Decision Map

Use the `workflows` block like this.

- Only symmetric `a,b,K0`:
  set `ground_state=true` and the rest `false`.
- Add the liquid critical point:
  set `quantum_critical=true`.
- Add symmetric quarkyonic EOS and sound speed:
  set `symmetric_quarkyonic=true`.
- Add a pure hadronic EOS table before quarkyonic matter:
  set `hadronic_eos_table=true`.
- Add asymmetric couplings only:
  set `asymmetric_fit=true`.
- Add asymmetric fixed-`y` matter without leptons:
  set `asymmetric_quarkyonic=true` and `asymmetric.beta_equilibrium=false`.
- Add beta equilibrium with electrons and muons:
  set `asymmetric_quarkyonic=true` and `asymmetric.beta_equilibrium=true`.
- Add neutron stars:
  set `neutron_star=true`.

Important practical detail:
if you request a downstream workflow, `QMM` automatically computes the symmetric ground-state point first even if `workflows.ground_state=false`, because the higher-level solvers depend on it.

## Sound Speed And EOS Outputs

Within the current `QMM` design:

- pure hadronic EOS tables are written when `workflows.hadronic_eos_table=true`
- sound speed is reconstructed automatically when a quarkyonic profile is requested
- EOS tables are written when `output.write_csv=true`
- plots are written when `output.write_plots=true`

So the practical way to skip sound-speed or EOS generation is to disable the corresponding workflow branch or disable the output writing.

## Install And Run

Use Python `3.11+`. On this machine, `python3` points to `3.9`, so the safe interpreter is `/opt/homebrew/bin/python3.12`.

From this `QMM` directory:

```bash
/opt/homebrew/bin/python3.12 -m pip install -e .
/opt/homebrew/bin/python3.12 -m qmm examples/cs_full.json
/opt/homebrew/bin/python3.12 -m qmm examples/tvm_full.json
/opt/homebrew/bin/python3.12 -m qmm examples/clausius_reference.json
/opt/homebrew/bin/python3.12 -m qmm examples/clausius_asymmetric_base.json
/opt/homebrew/bin/python3.12 -m qmm examples/clausius_equal_b_base.json
/opt/homebrew/bin/python3.12 -m qmm examples/clausius_cs_hybrid_base.json
/opt/homebrew/bin/python3.12 -m qmm examples/clausius_tvm_hybrid_base.json
/opt/homebrew/bin/python3.12 -m qmm examples/clausius_baryquark_demo.json
```

If you prefer not to install the package:

```bash
PYTHONPATH=src /opt/homebrew/bin/python3.12 -m qmm examples/cs_full.json
```

## Blank Template

The fully general blank template is:

- [examples/qmm_blank_template.json](/Users/dajuarez4/Documents/QuarkMatt/QMM/examples/qmm_blank_template.json)
- [examples/qmm_all_in_one_template.json](/Users/dajuarez4/Documents/QuarkMatt/QMM/examples/qmm_all_in_one_template.json)
- [examples/clausius_asymmetric_base.json](/Users/dajuarez4/Documents/QuarkMatt/QMM/examples/clausius_asymmetric_base.json)
- [examples/clausius_baryquark_demo.json](/Users/dajuarez4/Documents/QuarkMatt/QMM/examples/clausius_baryquark_demo.json)
- [examples/clausius_equal_b_base.json](/Users/dajuarez4/Documents/QuarkMatt/QMM/examples/clausius_equal_b_base.json)
- [examples/clausius_cs_hybrid_base.json](/Users/dajuarez4/Documents/QuarkMatt/QMM/examples/clausius_cs_hybrid_base.json)
- [examples/clausius_tvm_hybrid_base.json](/Users/dajuarez4/Documents/QuarkMatt/QMM/examples/clausius_tvm_hybrid_base.json)

Use `qmm_blank_template.json` if you want the cleanest runnable file.

Use `qmm_all_in_one_template.json` if you want one single JSON that also documents:

- the mean field in LaTeX
- the excluded-volume rule in LaTeX
- whether you conceptually want symmetric or asymmetric matter
- whether you want EOS, sound speed, liquid critical point, leptons, or neutron stars

Then fill:

- the model name
- the extra parameter or parameter-search block
- which workflows you want
- whether leptons are included
- whether outputs are written
- the numerical settings you want to override

## Adding Your Own Real-Gas Model

To add a new model, edit [src/qmm/models.py](/Users/dajuarez4/Documents/QuarkMatt/QMM/src/qmm/models.py) and register one more `InteractionModel`.

The solver needs:

1. an attractive function `U(n,b,parameter)` and `dU/dn`
2. a map `nid_from_n`
3. an inverse map `n_from_nid`
4. a total volume fraction
5. a species-level volume fraction
6. a pressure prefactor
7. capability flags declaring which workflow branches are supported

If the model has an extra free parameter, define:

- `parameter_name`
- `parameter_range`

and then provide either:

- `model.parameter_value`

or

- `model.parameter_search`

in the JSON input.

## Documentation

The formal manual is:

- [docs/qmm_complete_workflow_manual.tex](/Users/dajuarez4/Documents/QuarkMatt/QMM/docs/qmm_complete_workflow_manual.tex)

The notebook overview is:

- [notebooks/QMM_workflow.ipynb](/Users/dajuarez4/Documents/QuarkMatt/QMM/notebooks/QMM_workflow.ipynb)
- [notebooks/baryquark_demo.ipynb](/Users/dajuarez4/Documents/QuarkMatt/QMM/notebooks/baryquark_demo.ipynb)
- [notebooks/clausius_beta_vs2_postprocess.ipynb](/Users/dajuarez4/Documents/QuarkMatt/QMM/notebooks/clausius_beta_vs2_postprocess.ipynb)
- [notebooks/clausius_asymmetric_workflow.ipynb](/Users/dajuarez4/Documents/QuarkMatt/QMM/notebooks/clausius_asymmetric_workflow.ipynb)
- [notebooks/clausius_cs_hybrid_scan.ipynb](/Users/dajuarez4/Documents/QuarkMatt/QMM/notebooks/clausius_cs_hybrid_scan.ipynb)
- [notebooks/clausius_tvm_hybrid_scan.ipynb](/Users/dajuarez4/Documents/QuarkMatt/QMM/notebooks/clausius_tvm_hybrid_scan.ipynb)

## Baryquark Demo

The runnable symmetric demo is:

- [examples/clausius_baryquark_demo.json](/Users/dajuarez4/Documents/QuarkMatt/QMM/examples/clausius_baryquark_demo.json)

It uses:

- `model.name = "clausius"`
- `quarkyonic.momentum_mode = "baryquark"`
- a `K_0 = 280 MeV` parameter search for the Clausius parameter `c`

The companion notebook is:

- [notebooks/baryquark_demo.ipynb](/Users/dajuarez4/Documents/QuarkMatt/QMM/notebooks/baryquark_demo.ipynb)

## Clausius Beta-Equilibrium `v_s^2` Postprocessing

The Clausius beta-equilibrium comparison plots discussed during the smoothing study are now collected in:

- [notebooks/clausius_beta_vs2_postprocess.ipynb](/Users/dajuarez4/Documents/QuarkMatt/QMM/notebooks/clausius_beta_vs2_postprocess.ipynb)

The script used to regenerate them is:

- [scripts/postprocess_beta_sound_speed.py](/Users/dajuarez4/Documents/QuarkMatt/QMM/scripts/postprocess_beta_sound_speed.py)

This postprocessing step:

- starts from the saved beta-equilibrium CSV tables
- recomputes `\mu_B`, `P`, and `v_s^2` from `\epsilon(n_B)`
- applies a larger local-polynomial smoothing window to the derivative reconstruction
- writes cleaned comparison figures without rerunning the full Clausius scan

Current production choice:

- `window = 29`
- `degree = 3`

Regenerate the plots with:

```bash
cd QMM
python3 scripts/postprocess_beta_sound_speed.py \
  --window 29 \
  --degree 3 \
  --output-dir results/generated/clausius_beta_vs2_postprocess_window29
```

The current outputs are stored in:

- [results/generated/clausius_beta_vs2_postprocess_window29](/Users/dajuarez4/Documents/QuarkMatt/QMM/results/generated/clausius_beta_vs2_postprocess_window29)

Key comparison figures:

- [results/generated/clausius_beta_vs2_postprocess_window29/clausius_branch_comparison_k0_overlay.png](/Users/dajuarez4/Documents/QuarkMatt/QMM/results/generated/clausius_beta_vs2_postprocess_window29/clausius_branch_comparison_k0_overlay.png)
- [results/generated/clausius_beta_vs2_postprocess_window29/clausius_branch_comparison_same_k0_colors.png](/Users/dajuarez4/Documents/QuarkMatt/QMM/results/generated/clausius_beta_vs2_postprocess_window29/clausius_branch_comparison_same_k0_colors.png)
- [results/generated/clausius_beta_vs2_postprocess_window29/clausius_branch_comparison_k0_250_280_315.png](/Users/dajuarez4/Documents/QuarkMatt/QMM/results/generated/clausius_beta_vs2_postprocess_window29/clausius_branch_comparison_k0_250_280_315.png)

## Included Reference Results

The curated local `results/` tree contains:

- combined symmetric nuclear-matter and liquid critical-point tables
- symmetric quarkyonic survey curves
- asymmetric CS, TVM, and Clausius outputs
- beta-equilibrium and neutron-star summary tables

Everything used by the notebook lives inside `QMM`.
