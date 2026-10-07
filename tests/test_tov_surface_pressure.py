import unittest
from unittest.mock import patch
import numpy as np
from scipy.interpolate import interp1d
from TOVsolver.tov import TOV


class SurfacePressureTests(unittest.TestCase):
    def solver(self):
        solver = object.__new__(TOV)
        solver.min_p = 9.744e18
        solver.max_p = 2e19
        solver.check_density = lambda value: None
        solver.press = lambda value: 1.5e19
        solver.en_dens = interp1d([solver.min_p, solver.max_p], [1., 2.], bounds_error=True)
        return solver

    def run_profile(self, pressure):
        solver = self.solver()
        profile = np.column_stack([pressure, [1., 2., 3., 4., 5., 5.]])
        with patch('TOVsolver.tov.odeint', return_value=profile):
            return solver.solve(100., rmax=600., dr=100.)

    def test_surface_overshoot_does_not_break_profile(self):
        r, m, (_, energy, pressure, _) = self.run_profile(
            [1.5e19, 1.2e19, 9.74398168610125e18, 9.74398168610125e18, 9.74398168610125e18, 9.74398168610125e18])
        self.assertTrue(np.isfinite(energy).all())
        self.assertTrue((pressure >= 9.744e18).all())
        baseline = self.run_profile([1.5e19, 1.2e19, 9.744e18, 9.744e18, 9.744e18, 9.744e18])
        self.assertEqual((r, m), baseline[:2])

    def test_upper_bound_error_is_not_hidden(self):
        with self.assertRaises(ValueError):
            self.run_profile([1.5e19, 3e19, 1.2e19, 1.1e19, 1e19, 1e19])

if __name__ == '__main__':
    unittest.main()
