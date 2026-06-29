import unittest

from qmm.plotting import _vs2_axis_limits
from qmm.qmm_json_helper import _workflow_from_preset


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


if __name__ == "__main__":
    unittest.main()
