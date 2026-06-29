import unittest

from qmm.asymmetry import compute_asymmetric_fit
from qmm.constants import AsymmetricFitSettings, QuarkyonicSettings
from qmm.quarkyonic import AsymmetricQuarkyonicEOS


class QuarkyonicBetaChargeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
