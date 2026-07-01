import unittest

from qmm.plotting import _vs2_axis_limits
from qmm.qmm_json_helper import _apply_numeric_request_overrides, _workflow_from_preset, default_request


class GuidedPresetAndPlottingTests(unittest.TestCase):
    def test_asymmetric_beta_critical_point_preset_enables_quantum_and_asymmetric_workflows(self) -> None:
        workflow = _workflow_from_preset("asymmetric_beta_critical_point", None)
        self.assertTrue(workflow["ground_state"])
        self.assertTrue(workflow["quantum_critical"])
        self.assertTrue(workflow["hadronic_eos_table"])
        self.assertTrue(workflow["asymmetric_fit"])
        self.assertTrue(workflow["asymmetric_quarkyonic"])
        self.assertFalse(workflow["neutron_star"])

    def test_vs2_axis_limits_expand_when_peak_exceeds_default_ceiling(self) -> None:
        lower, upper = _vs2_axis_limits([0.2, 0.9, 1.31])
        self.assertEqual(lower, -0.1)
        self.assertGreater(upper, 1.31)
        self.assertGreaterEqual(upper, 1.4)

    def test_default_request_exposes_quarkyonic_numeric_override_fields(self) -> None:
        request = default_request()
        for key in (
            "quarkyonic_n_min_ratio",
            "quarkyonic_n_max_ratio",
            "quarkyonic_n_points",
            "quarkyonic_fq_scan_points",
            "quarkyonic_shell_integral_points",
            "quarkyonic_quark_integral_points",
        ):
            self.assertIn(key, request)
            self.assertIsNone(request[key])

    def test_numeric_request_overrides_update_quarkyonic_config(self) -> None:
        config = {
            "quarkyonic": {
                "n_min_ratio": 0.05,
                "n_max_ratio": 5.0,
                "n_points": 120,
                "fq_scan_points": 121,
                "shell_integral_points": 400,
                "quark_integral_points": 400,
            }
        }
        request = {
            "quarkyonic_n_min_ratio": 0.10,
            "quarkyonic_n_max_ratio": 6.0,
            "quarkyonic_n_points": 180,
            "quarkyonic_fq_scan_points": 181,
            "quarkyonic_shell_integral_points": 500,
            "quarkyonic_quark_integral_points": 520,
        }

        _apply_numeric_request_overrides(config, request)

        self.assertEqual(config["quarkyonic"]["n_min_ratio"], 0.10)
        self.assertEqual(config["quarkyonic"]["n_max_ratio"], 6.0)
        self.assertEqual(config["quarkyonic"]["n_points"], 180)
        self.assertEqual(config["quarkyonic"]["fq_scan_points"], 181)
        self.assertEqual(config["quarkyonic"]["shell_integral_points"], 500)
        self.assertEqual(config["quarkyonic"]["quark_integral_points"], 520)


if __name__ == "__main__":
    unittest.main()
