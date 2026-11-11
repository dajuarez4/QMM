import unittest

import numpy as np

from TOVsolver.tov import TOV
from TOVsolver.constants import c,MeV_fm3_to_pa_cgs


class TidalEOSDerivativeTests(unittest.TestCase):
    def test_analytic_cubic_eos_derivative_including_table_boundaries(self):
        pressure=np.linspace(.01,100.,80)
        energy=7+3*pressure+.002*pressure**2
        solver=TOV(energy.copy(),pressure.copy(),add_crust=False)
        for p in (pressure[0],pressure[0]*1.001,10.,pressure[-1]*.999,pressure[-1]):
            # TOV stores mass density epsilon/c² against pressure in cgs.
            actual=solver.dedp(1.,(None,lambda r:p*MeV_fm3_to_pa_cgs,None))
            expected=(3+.004*p)/c**2
            np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=0.)

    def test_tidal_integration_and_radial_resolution_for_smooth_eos(self):
        energy=np.geomspace(1e-7,4000.,800)
        pressure=1e-4*energy**2
        solver=TOV(energy.copy(),pressure.copy(),add_crust=False)
        coarse=solver.solve_tidal(800.,rmax=50e5,dr=1000.)
        fine=solver.solve_tidal(800.,rmax=50e5,dr=500.)
        self.assertTrue(np.isfinite(coarse).all())
        self.assertTrue(np.isfinite(fine).all())
        self.assertTrue(0<fine[3]<.75)
        np.testing.assert_allclose(coarse[:4],fine[:4],rtol=.02)


if __name__=='__main__':
    unittest.main()
