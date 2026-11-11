# Audit and reproduction of the incompressibility paper figures

Completed using the files in the **main paper folder**, on 2026-11-11. `old_stuff/` was not used. The calculations use `good_repo/QMM`, not the other QMM copy in the workspace.

The large discrepancies had identifiable causes. The corrected calculation closely follows the supplied composition and sound-speed images. These are recomputed model results, not image tracings presented as calculations. Exact numerical identity cannot be certified without the original numerical tables and full numerical settings.

## Run the notebook

Open `notebooks/03_six_models_paper/paper_replication_lambda300_audited.ipynb` in the QMM Python environment and **Run All**. The historical `screenshot_excluded_volume_comparison.ipynb` now points to this replacement. It retains its old calculations for provenance.

The notebook and `scripts/reproduce_incompressibility_paper.py` use the same implementation. From the QMM directory:

```sh
.venv/bin/python scripts/reproduce_incompressibility_paper.py
```

For a fresh installation: `python -m pip install -e '.[notebook]'`. The default output is `results/generated/paper_replication_lambda300`. Compatible density and stellar checkpoints are reused. Different physical/numerical controls or source hashes select different run directories. `active_runs.json` identifies the final 15 runs; intermediate diagnostic runs remain with their own inputs and source hashes. Any verified checkpoint imports have an explicit `checkpoint_import.json` record.

## 1. Nuclear calibration: the old notebook used the wrong branch

The manuscript states n₀ = 0.16 fm⁻³, binding = −16 MeV, J = 32.5 MeV, L = 58.9 MeV, and K₀ = 220 or 300 MeV. The old notebook assumed `equal_b`: it imposed bpn = bn and **predicted** L. This was not the paper's stated five-constraint calibration. The replacement defaults to `target_l`. Λ is explicitly **300 MeV** in every run; mN = 938 MeV matches the repository and the preceding published model.

`parameters.csv` records the actual fitted couplings and residuals. The vdW-attraction reference has no extra parameter to tune K₀; its K₀ is predicted separately for each excluded-volume family.

## 2. Dieterici: the manuscript body and the original code have different composition dependence

The body of `main_revised.tex` defines

$$v_i^D=-\sum_j a_{ij}\frac{n_j^{\alpha-1}}{\alpha-1}.$$

Consequently, for hadronic species densities np, nn,

$$\epsilon_{\mathrm{att}}^D=-\frac{a_n(n_p^\alpha+n_n^\alpha)+a_{pn}(n_p n_n^{\alpha-1}+n_n n_p^{\alpha-1})}{\alpha-1}.$$

The old QMM implementation instead used

$$-a_{\mathrm{eff}}(Y)\frac{n_H^\alpha}{\alpha-1},\qquad a_{\mathrm{eff}}=a_n[Y^2+(1-Y)^2]+2a_{pn}Y(1-Y).$$

These are different at intermediate Y when α ≠ 2. In particular, they give different beta-equilibrium charge fractions even when the usual calibration checks pass. The appendix repeats the quadratic-coefficient construction and therefore conflicts with the manuscript body. **The supplied PNGs agree with the species-power form in the body.**

To retain exactly the same symmetric and pure-neutron calibration, the paper implementation uses

$$\bar a_{\mathrm{paper}}=2^{\alpha-2}\bar a_{\mathrm{old}},\quad a_{n,\mathrm{paper}}=a_{n,\mathrm{old}},\quad a_{pn,\mathrm{paper}}=2\bar a_{\mathrm{paper}}-a_n.$$

The b couplings and α stay unchanged. The symmetric- and pure-neutron energies then agree at **every** density, preserving binding, K₀, J = EPNM − ESNM, and L. This explains why the old form could pass saturation/J/L checks while still disagreeing with the composition figures. The exported parameter table contains the converted couplings. `dieterici_attraction='qmm'` is available for an explicitly labeled sensitivity comparison.

For example, TVM + Dieterici at K₀ = 300 MeV gives, at nB = 1.2 fm⁻³, approximately **fQ = 0.4682, Y = 0.01967, kFe = 0.1565 GeV**, consistent with the supplied figure.

## 3. The dashed reference is an attraction limit, with the same EV family

The old notebook repeated one **vdW excluded-volume** reference in every row. The paper instead uses **vdW mean-field attraction**, combined with TVM, CS, or vdW excluded volume in the corresponding figure. The new calculation therefore has **15 runs**: four tunable models and one separate reference for each of three excluded-volume families.

## 4. Charge, beta equilibrium, and shell construction

In the exported EOS, nu and nd are **quark number densities**, including color degeneracy. The conserved charges are

$$n_B=n_p+n_n+(n_u+n_d)/3,$$
$$\rho_Q=n_p+(2n_u-n_d)/3=Yn_B=n_e+n_\mu.$$

The manuscript sometimes writes nu and nd as **baryon-normalized** quark contributions. In that convention the equivalent charge is np + 2nuᴮ − ndᴮ. Those conventions must not be mixed. The current original QMM `charge_density_hq` already has the correct factors for its number-density functions; no additional factor of three was introduced here.

The beta condition is

$$\mu_e=\mu_\mu=-\frac{1}{n_B}\left(\frac{\partial\epsilon_{HQ}}{\partial Y}\right)_{n_B,f_Q}.$$

This derivative includes the Y dependence of the common inner shell momentum, both species-dependent volume fractions, and the interaction. Lepton states vanish below their mass thresholds. Total energy, including both leptons, is minimized over fQ. The new evaluator scans for beta roots and competing quark-fraction minima before refinement, instead of assuming one whole-interval golden-section search establishes a global minimum. It scans the physical fraction interval without the original conservative constant lower bound that could discard allowed unequal-b states.

The regulated quark density and energy integrals are evaluated analytically. They agree with adaptive quadrature and the original high-resolution Simpson evaluation. The total-density QMM functional remains available; the manuscript-specific Dieterici correction is explicit in `qmm/paper_reproduction.py`. A separate fix in `quarkyonic.py` makes the pure-quark Dieterici endpoint finite: the vanishing hadronic interaction must not be evaluated as U(0), which was undefined in the original code.

## 5. Pressure and sound speed: the formula is correct, with total equilibrium energy

Use total energy density, including rest mass and leptons, after re-equilibrating composition at each density:

$$\mu_B=\frac{d\epsilon_{\mathrm{eq}}}{dn_B},\quad P=n_B\mu_B-\epsilon_{\mathrm{eq}},\quad c_s^2=\frac{n_B\epsilon_{\mathrm{eq}}''}{\epsilon_{\mathrm{eq}}'}.$$

The original `numpy.gradient` identity is correct. The large discrepancies were primarily the model/calibration choices above. The replacement retains an independent direct-gradient column and uses an **interpolating cubic spline with no smoothing** for its primary derivatives. It solves a grid twice as dense near each peak to measure resolution error. No negative or superluminal values are clipped.

The text's description of μB as the derivative of **strong-sector energy along the equilibrium path** needs correction. Along that path, the charge fraction changes and the lepton derivative contributes. The derivative of total neutral equilibrium energy gives μB. Equivalently, the strong-sector partial derivative for μB holds electric charge fixed. Differentiating only strong-sector energy along the beta path is not the same operation.

## 6. Crust, TOV, and tides

The bundled BPS crust is joined at the highest pressure crossing with the uniform core. The core is required to be strictly increasing and positive above 0.04 fm⁻³; this workflow does not silently delete unstable core samples. The EOS extends to 2.4 fm⁻³ for stellar centers. All 15 sequences have their maximum inside the sampled range. The full sequence, including the descending mass branch, is retained and flagged in `stable_branch`.

Three smaller errors were corrected in the shared `TOVsolver/tov.py`:

- The old solver inferred the surface from a mass plateau after a discontinuous RHS stop, which could produce LSODA failures and uninitialized post-surface rows. The surface is now the interpolated pressure crossing at the tabulated minimum; trial RHS values continue smoothly beyond it and are excluded from the returned star.
- The initial enclosed mass is 4πr³ρ/3; the original expression omitted the division by three.
- The old tidal dρ/dP stencil sampled pressures ±1.5% away from the current pressure and left the EOS table near its boundaries. It now differentiates the same interpolating EOS analytically. Radial fields use monotone interpolation, with boundary values held during LSODA trial stages beyond the requested last radius.

Tidal failures are recorded rather than silently connected in a plot. **All retained points in the final 15 tidal sequences succeeded.** **The final TOV sequences also have zero integration warnings.** Each `tov_meta.json` and `tov_warning_audit.csv` records this. `legacy_tov_warning_audit.csv` preserves the evidence that led to the surface correction. The original observational shaded bands are not recreated because their source contour data are unavailable.

## 7. Measured agreement with the main-folder PNGs

`scripts/compare_paper_pngs.py` digitizes the original colored curves with explicit tick/pixel calibrations, excludes legends, and compares only nB = 0.1–1.18 fm⁻³. The image data never influence model parameters or computed states. Figures named `*_reference_overlay.png` show computed lines and digitized image circles. Curve overlap, antialiasing and the 1400×600 pixel resolution limit the comparison.

The following are **mean per-curve RMSEs**, comparing the same reference pixels before and after the corrections. Electron momenta are in GeV; the other quantities are dimensionless.

| Observable | Old mean curve RMSE | Corrected mean curve RMSE |
| --- | --- | --- |
| vs2 | 0.120606 | 0.0041439 |
| quark_fraction | 0.106605 | 0.0023876 |
| y | 0.0774506 | 0.0003779 |
| electron_kf_GeV | 0.0710079 | 0.000343245 |

The sharp vdW sound-speed peaks are the largest remaining image discrepancy. They are especially sensitive to sampling and numerical differentiation. The model still produces the superluminal peaks visible in the reference; this is a property of that EV choice, not something corrected by clipping.

## 8. Validation and numerical limits

- **26 automated tests pass**, including adaptive quadrature comparisons, analytic versus finite-difference beta derivatives, conservation and muon thresholds, preservation of the Dieterici SNM/PNM calibration, independent two-variable neutral-energy minimization, the CS close-packing regression, analytic sound-speed checks, and tidal derivative/integration checks.
- Across the final tables, maximum charge closure error is **2.31×10⁻¹² fm⁻³** and baryon closure error is **2.00×10⁻¹⁵ fm⁻³**. The independent fourth-order beta residual is below **1.91×10⁻⁴ MeV**; most models are below 3×10⁻⁶ MeV. The larger CS residual occurs near very small Y where energy differencing is ill-conditioned.
- Independent denser minimization scans (fQ scan 61 → 121, Y scan 25 → 81) at four densities for all 15 models changed energy by at most **5.78×10⁻¹¹ MeV fm⁻³**, Y by **5.41×10⁻⁸**, and fQ by **3.01×10⁻⁸**. These are sampled convergence checks, not a mathematical proof that no unsampled minimum exists anywhere.
- Halving the density step from 0.004 to 0.002 fm⁻³ near each sound-speed maximum changes peak height by less than **0.009% for TVM/CS**, and by at most **0.914% for vdW**. Full refinement results are in `peak_refinement.csv` and `peak_convergence.json` in each active run.
- Three representative stellar checks near 1.4 M☉, reducing dr from 500 to 250 cm and tightening ODE tolerance, changed mass by at most **0.0019%**, radius by **0.0198%**, k₂ by **0.0965%**, and tidal Λ by **0.0120%**. See `stellar_convergence.csv`.

The final peak and stellar results are:

| Model | K₀ [MeV] | Fine peak cₛ² | Peak nB [fm⁻³] | Mmax [M☉] |
| --- | --- | --- | --- | --- |
| dieterici_tvm | 220 | 0.625255 | 0.606 | 2.2127 |
| dieterici_tvm | 300 | 0.576614 | 0.524 | 2.2903 |
| clausius_tvm | 220 | 0.692782 | 0.786 | 2.0633 |
| clausius_tvm | 300 | 0.597631 | 0.618 | 2.2081 |
| reference_tvm | reference | 0.498127 | 0.392 | 2.4187 |
| dieterici_cs | 220 | 0.644002 | 0.576 | 2.2553 |
| dieterici_cs | 300 | 0.588667 | 0.502 | 2.3301 |
| clausius_cs | 220 | 0.720355 | 0.750 | 2.0960 |
| clausius_cs | 300 | 0.613995 | 0.602 | 2.2378 |
| reference_cs | reference | 0.506671 | 0.372 | 2.4628 |
| dieterici_vdw | 220 | 1.892544 | 0.374 | 2.6309 |
| dieterici_vdw | 300 | 1.796884 | 0.318 | 2.7192 |
| clausius_vdw | 220 | 1.156440 | 0.608 | 2.3191 |
| clausius_vdw | 300 | 2.015422 | 0.392 | 2.5836 |
| reference_vdw | reference | 1.475000 | 0.234 | 2.8304 |

`parameters.csv`, `validation_summary.csv`, `minimization_convergence.csv`, `stellar_convergence.csv`, `reference_errors.csv` and `legacy_comparison.csv` contain the numerical details. Per-run EOS tables include composition, both derivative estimates, and residuals. Stars and tides use that same stitched EOS.

## 9. Manuscript corrections to make before publication

1. Reconcile the Dieterici interaction in the body with the appendix. The supplied figures favor the species-power expression; document its coupling normalization explicitly.
2. Define whether each nu, nd is a quark number density or a baryon-normalized contribution, and use the corresponding factors consistently.
3. The infrared density-of-states multiplier should be sqrt(q²+Λ²)/q for a radial integrand q sqrt(q²+Λ²). The printed multiplier with q² in its denominator conflicts with that integrand. Integrals already written with a radial q² or k² factor should use dq or dk after angular integration; writing d³q or d³k as well double counts the momentum measure.
4. Clarify that the attractive interaction uses **hadronic** density nH = np + nn. The symbol nB in the real-gas subsection should not be interpreted as including deconfined quarks in that denominator.
5. State that the figure calibration enforces L = 58.9 MeV with unequal b couplings; equal-b results are a different calibration branch.
6. Correct the chemical-potential derivative discussion as described above. `main.tex` also has a negative sign in its stated symmetry-energy target; `main_revised.tex` correctly uses +32.5 MeV.

The previous asymmetric-quarkyonic paper provides an independent primary-source check on the conserved-charge and beta-equilibrium conventions: [Moss, Poberezhniuk and Vovchenko, arXiv:2411.11996, Sec. V](https://arxiv.org/html/2411.11996v1#S5). The specific Dieterici extension here follows your supplied manuscript and is checked against your PNGs.
