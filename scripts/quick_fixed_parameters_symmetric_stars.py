"""Formal symmetric-matter mass-radius comparison, y=0.5, through 15 n0.

Run: .venv/bin/python scripts/quick_fixed_parameters_symmetric_stars.py
This uses the same crust stitching as the beta comparison; it is not a
self-consistent symmetric crust or a beta-equilibrated neutron-star model.
"""
from quick_fixed_parameters_beta import (
    ROOT, MODELS, suite, load_run_config, compute_symmetric_quarkyonic_curve,
    replace, asdict, np, pd, plt, json, hashlib, argparse,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    parser.add_argument('--points', type=int, default=181, help='EOS samples from 0.05 to 15 n0')
    parser.add_argument('--tov-points', type=int, default=120)
    parser.add_argument('--plot-only', action='store_true', help='Require existing EOS and TOV caches')
    args = parser.parse_args()
    if args.points < 9 or args.tov_points < 3:
        parser.error('Use at least 9 EOS points and 3 TOV points.')
    out = ROOT/'Paper/TEST_fixed_parameters_symmetric_stars'
    out.mkdir(parents=True, exist_ok=True)
    curves, tables = {}, []
    controls = {**suite.DEFAULT_CONTROLS, 'tov_points': args.tov_points,
                'tov_step_cm': 500., 'tov_rmax_km': 50.}
    for key in args.models:
        cfg = load_run_config(ROOT/'examples/generated'/f'guided_{key}_lambda300_k0_200_750_step50_k0_250_target_l.json')
        parameter = 5/3 if key.startswith('dieterici') else 4.74
        settings = replace(cfg.quarkyonic, n_min_ratio=.05, n_max_ratio=15., n_points=args.points,
            lambda_momentum_mev=306., fq_scan_points=41, shell_integral_points=160,
            quark_integral_points=160, smoothing_window=9, smoothing_degree=3, refine_tol=1e-8)
        manifest = dict(model=key, parameter=parameter, settings=asdict(settings),
                        physical=asdict(cfg.physical), ground_state=asdict(cfg.ground_state),
                        controls=controls, composition='symmetric y=0.5, no leptons',
                        crust='same stitched crust prescription as beta comparison')
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode())
        for directory in ['src/qmm', 'src/TOVsolver']:
            for path in sorted((ROOT/directory).glob('*.py')):
                digest.update(path.read_bytes())
        digest.update((ROOT/'scripts/empirical_asymmetric_suite.py').read_bytes())
        folder = out/'runs'/key/digest.hexdigest()[:16]
        cache = folder/'symmetric.json'
        if args.plot_only and not all((folder/name).exists() for name in ['symmetric.json', 'mass_radius.csv', 'tov_meta.json']):
            raise FileNotFoundError(f'Run calculation first: {folder}')
        folder.mkdir(parents=True, exist_ok=True)
        suite.write_json(folder/'inputs.json', manifest)
        if cache.exists():
            result = json.loads(cache.read_text())
        else:
            print(f'{key}: computing {args.points} symmetric EOS points to 15 n0', flush=True)
            result = asdict(compute_symmetric_quarkyonic_curve(cfg.model_name,
                parameter_value=parameter, settings=settings, physical=cfg.physical,
                gs_settings=cfg.ground_state))
            suite.write_json(cache, result)
        frame = pd.DataFrame(dict(n_b=result['n'], n_over_n0=result['n_over_n0'],
            energy_density=result['eps_raw'], pressure=result['P'], vs2=result['vs2'],
            quark_fraction=result['quark_fraction']))
        if not np.isfinite(frame.to_numpy()).all() or not np.isclose(frame.n_over_n0.max(), 15.):
            raise ValueError(f'{key}: invalid or incomplete symmetric EOS')
        suite.write_csv(folder/'eos.csv', frame)
        if not args.plot_only:
            suite.ensure_tov(folder, controls)
        stars = pd.read_csv(folder/'mass_radius.csv')
        curves[key] = stars
        suite.write_csv(out/f'{key}_mass_radius.csv', stars)
        meta = json.loads((folder/'tov_meta.json').read_text())
        tables.append(dict(model=key, y=.5, a=result['a'], b=result['b'], K0_MeV=result['K0'],
            alpha=parameter if key.startswith('dieterici') else np.nan,
            c_fm3=parameter if key.startswith('clausius') else np.nan, **meta))
        print(key, 'Mmax =', meta['maximum_mass_msun'], flush=True)
    suite.write_csv(out/'parameters_and_stars.csv', pd.DataFrame(tables))
    plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'stix', 'pdf.fonttype':42})
    fig, ax = plt.subplots(figsize=(7,4.8))
    for key, stars in curves.items():
        # Retain central-density order and include configurations beyond maximum mass.
        ax.plot(stars.radius_km, stars.mass_msun, color=MODELS[key][1], label=MODELS[key][0], lw=1.4)
    ax.set(xlabel=r'$R$ [km]', ylabel=r'$M/M_\odot$')
    ax.tick_params(direction='in', top=True, right=True)
    fig.legend(loc='upper center', ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.12, right=.98, bottom=.14, top=.82)
    for stem, limits in [('symmetric_mass_radius_full', None), ('symmetric_mass_radius', (10,18))]:
        if limits is not None:
            ax.set_xlim(*limits)
        for ext in ['pdf', 'png', 'svg']:
            fig.savefig(out/f'{stem}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    (out/'README.md').write_text(__doc__ + '\nThe complete saved sequence includes the post-maximum branch. '
        'Results use coarse test resolution. CSV metadata retain EOS filtering and integration warnings.\n')
    print('Saved:', out)


if __name__ == '__main__':
    main()
