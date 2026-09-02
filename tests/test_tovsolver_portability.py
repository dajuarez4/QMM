from pathlib import Path
import unittest

import numpy as np

from TOVsolver.TOV_solver_code import load_bps_crust
from TOVsolver.tov import TOV


class TOVSolverPortabilityTests(unittest.TestCase):
    def test_bundled_bps_crust_is_available_and_monotone(self) -> None:
        energy, pressure = load_bps_crust()
        self.assertGreater(len(energy), 10)
        self.assertEqual(len(energy), len(pressure))
        self.assertTrue(np.all(np.isfinite(energy)))
        self.assertTrue(np.all(np.isfinite(pressure)))
        self.assertTrue(np.all(np.diff(energy) > 0.0))
        self.assertTrue(np.all(np.diff(pressure) > 0.0))

    def test_tov_package_has_no_external_workspace_dependency(self) -> None:
        import TOVsolver

        package_root = Path(TOVsolver.__file__).resolve().parent
        self.assertTrue((package_root / "data" / "Baym_eos.dat").is_file())
        self.assertEqual(TOV.__module__, "TOVsolver.tov")


if __name__ == "__main__":
    unittest.main()
