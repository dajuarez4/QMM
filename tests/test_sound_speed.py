import unittest

import numpy as np

from qmm.sound_speed import reconstruct_sound_speed_curve


class SoundSpeedTests(unittest.TestCase):
    def test_quadratic_energy_on_nonuniform_grid(self):
        n = np.array([0.1, 0.16, 0.25, 0.4, 0.6, 0.9])
        energy = 939 * n + 30 * n**2
        curve = reconstruct_sound_speed_curve(n.tolist(), energy.tolist(), 9, 3)
        np.testing.assert_allclose(curve.chemical_potential, 939 + 60 * n)
        np.testing.assert_allclose(curve.pressure, 30 * n**2)
        np.testing.assert_allclose(curve.vs2, 60 * n / (939 + 60 * n))
        np.testing.assert_array_equal(curve.energy_density_smoothed, energy)

    def test_smoothing_settings_do_not_change_raw_energy_or_derivatives(self):
        n = np.linspace(0.1, 1, 21)
        energy = 939 * n + n**2 + 0.01 * np.sin(50 * n)
        first = reconstruct_sound_speed_curve(n.tolist(), energy.tolist(), 9, 3)
        second = reconstruct_sound_speed_curve(n.tolist(), energy.tolist(), 19, 2)
        self.assertEqual(first, second)
        np.testing.assert_array_equal(first.energy_density_smoothed, energy)


if __name__ == '__main__':
    unittest.main()
