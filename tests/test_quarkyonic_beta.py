import unittest
import math

from qmm.asymmetry import compute_asymmetric_fit
from qmm.constants import AsymmetricFitSettings, QuarkyonicSettings
from qmm.quarkyonic import AsymmetricQuarkyonicEOS, _max_hadronic_density


class QuarkyonicBetaChargeTests(unittest.TestCase):
    def test_hadronic_density_bound_recognizes_prefixed_ev_families(self) -> None:
        b_value = 2.5
        for model_name in ("cs", "clausius_cs"):
            with self.subTest(model=model_name):
                self.assertAlmostEqual(_max_hadronic_density(model_name, b_value), 4.0 / b_value)
        for model_name in ("tvm", "clausius_tvm"):
            with self.subTest(model=model_name):
                self.assertTrue(math.isinf(_max_hadronic_density(model_name, b_value)))
        for model_name in ("vdw", "clausius"):
            with self.subTest(model=model_name):
                self.assertAlmostEqual(_max_hadronic_density(model_name, b_value), 1.0 / b_value)

    def test_total_charge_density_matches_y_times_baryon_density(self) -> None:
        fit = compute_asymmetric_fit(
            "cs",
            fit_settings=AsymmetricFitSettings(
                enabled=True,
                branch_mode="equal_b",
                beta_equilibrium=True,
            ),
        )
        eos = AsymmetricQuarkyonicEOS(
            fit,
            settings=QuarkyonicSettings(
                lambda_momentum_mev=200.0,
                shell_integral_points=40,
                quark_integral_points=40,
            ),
        )

        n_b = 0.8
        for fq in (0.0, 0.2, 0.6):
            for y_value in (0.03, 0.1, 0.2):
                with self.subTest(fq=fq, y_value=y_value):
                    charge_density = eos.charge_density_hq(n_b, fq, y_value)
                    self.assertAlmostEqual(charge_density, y_value * n_b, places=8)

    def test_vdw_beta_row_survives_to_five_n0(self) -> None:
        fit = compute_asymmetric_fit(
            "vdw",
            fit_settings=AsymmetricFitSettings(
                enabled=True,
                branch_mode="equal_b",
                beta_equilibrium=True,
            ),
        )
        eos = AsymmetricQuarkyonicEOS(
            fit,
            settings=QuarkyonicSettings(
                lambda_momentum_mev=200.0,
                fq_scan_points=41,
                shell_integral_points=40,
                quark_integral_points=40,
            ),
        )

        n_b = 5.0 * eos.physical.n0
        row = eos.build_beta_row(n_b)
        self.assertIsNotNone(row)
        self.assertAlmostEqual(row.n_over_n0, 5.0, places=8)

    def test_vdw_fixed_y_equal_b_row_survives_above_previous_cutoff(self) -> None:
        fit = compute_asymmetric_fit(
            "vdw",
            fit_settings=AsymmetricFitSettings(
                enabled=True,
                branch_mode="equal_b",
                beta_equilibrium=True,
            ),
        )
        eos = AsymmetricQuarkyonicEOS(
            fit,
            settings=QuarkyonicSettings(
                n_min_ratio=0.005,
                n_max_ratio=5.0,
                n_points=200,
                lambda_momentum_mev=200.0,
                fq_scan_points=41,
                shell_integral_points=40,
                quark_integral_points=40,
            ),
        )

        n_b = 4.799195979899497 * eos.physical.n0
        row = eos.build_fixed_y_row(n_b, 0.1)
        self.assertIsNotNone(row)
        self.assertAlmostEqual(row.n_over_n0, 4.799195979899497, places=8)

    def test_vdw_fixed_y_target_l_row_survives_above_previous_cutoff(self) -> None:
        fit = compute_asymmetric_fit(
            "vdw",
            fit_settings=AsymmetricFitSettings(
                enabled=True,
                branch_mode="target_l",
                beta_equilibrium=True,
            ),
        )
        eos = AsymmetricQuarkyonicEOS(
            fit,
            settings=QuarkyonicSettings(
                n_min_ratio=0.005,
                n_max_ratio=5.0,
                n_points=200,
                lambda_momentum_mev=200.0,
                fq_scan_points=41,
                shell_integral_points=40,
                quark_integral_points=40,
            ),
        )

        n_b = 3.920678391959799 * eos.physical.n0
        row = eos.build_fixed_y_row(n_b, 0.1)
        self.assertIsNotNone(row)
        self.assertAlmostEqual(row.n_over_n0, 3.920678391959799, places=8)


if __name__ == "__main__":
    unittest.main()
