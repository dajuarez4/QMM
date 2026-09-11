# Combined model bands, K0 = 200–350 MeV

Run `notebooks/combined_models_bands_K0_200_350.ipynb` or `.venv/bin/python scripts/plot_combined_model_bands.py`.

The master figure has eight panels: sound speed, mass–radius, pressure versus energy density, and quark fraction versus n_B/n_0 for both asymmetric branches. Four separate two-panel figures provide larger versions. Each panel overlays all six Dieterici/Clausius × vdW/CS/TVM models. Marker shape identifies the model, curve/marker color identifies K0, and black lines identify Dieterici–vdW and Clausius–vdW. Those black lines are not a separate pure-vdW model.

Bands use computed samples at K0 = 200, 250, 300 and 350 MeV, with Lambda = 300 MeV. The default central line at 275 MeV is an explicitly labeled linear interpolation between the 250 and 300 MeV curves, not a new physical calculation. Set `CENTER_K0 = 300` in the notebook for a computed reference instead. Band fill colors identify models; the colored marker faces identify incompressibilities.

All EoS tables extend to n_B/n_0 = 15. Full saved mass–radius sequences are shown, including the post-maximum branch, with no fixed radius or mass-axis cropping in the main axes. The separate M–R figure adds a zoomed inset for readability. At a given energy density or baryon density, an EoS/sound-speed/fq band spans the sampled model values on their common domain. For mass–radius, bands are swept between neighboring K0 trajectories at common central energy densities; radius is never sorted into an artificial single-valued M(R) function. Failed-star gaps in the recovered sequence remain gaps. These are sampled model bands, not statistical confidence intervals or a claim of a dense K0 scan.

## Rechecked Dieterici–TVM point

The saved K0 = 350 MeV, unequal-b EoS had a high-density energy jump and no completed TOV file. The original golden-section minimizer could return a finite local answer without checking other valid regions of the quark-fraction objective. For n_B/n_0 = 14, 14⅓, 14⅔ and 15, `repair_combined_band_eos.py` applies the existing 141-point global quark-fraction scan and refines its local minima with unchanged physical parameters and integration settings. Lower-energy solutions remove the jump. Pressure and sound speed are reconstructed from the resulting energy table using the existing local-thermodynamics method.

Corrected EoS and a newly integrated TOV sequence are in `recovered/`, alongside provenance and diagnostics. The original saved scan remains intact. The recovery records a status for each of 260 central-density trials and rejects stars with ODE nonconvergence, exceptions, invalid mass/radius, or the radius boundary. It does not pretend every attempted star converged. The other 47 saved TOV curves retain their original numerical limitations.

`coverage.csv` lists every model/K0/branch input. `validation.json` records coverage, charge neutrality, density closure, local energy checks, and the recovered TOV sequence. `plot_settings.json` records the chosen central line. PNG, PDF and SVG files are saved beside this README.

Validation of the corrected points gives charge-neutrality residuals below 1.4e-10 and baryon-density closure errors at roundoff. The recovered TOV table contains 215 accepted stars out of 260 trials. Halving the radial output spacing and tightening the integration tolerance for the maximum-mass and highest-density stars changes mass by less than 1.4e-5 solar masses and radius by at most 0.00125 km. These checks concern the repaired case, not a new audit of the other 47 legacy sequences.
