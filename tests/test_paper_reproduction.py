"""Independent quadrature, variational, and conservation checks for the paper."""
import math
import unittest
from dataclasses import replace

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, differential_evolution

from qmm.asymmetry import compute_asymmetric_fit
from qmm.constants import AsymmetricFitSettings, QuarkyonicSettings
from qmm.paper_reproduction import PaperEOS, regulated_integrals, thermodynamics, species_power_couplings
from qmm.quarkyonic import AsymmetricQuarkyonicEOS


class PaperReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fit = compute_asymmetric_fit('clausius_tvm',parameter_value=3.3711189950675404,
                  fit_settings=AsymmetricFitSettings(enabled=True,branch_mode='target_l'))
        cls.settings = QuarkyonicSettings(lambda_momentum_mev=300.,beta_y_scan_points=31,
                         fq_scan_points=61,refine_tol=1e-10,quark_integral_points=800)
        cls.eos = PaperEOS(cls.fit,settings=cls.settings)
        cls.original = AsymmetricQuarkyonicEOS(cls.fit,settings=cls.settings)

    def test_integrals_against_adaptive_quadrature(self):
        for regulator in (0.,.01,1.52,3.):
            for q in (1e-5,.01,.3,1.,5.):
                with self.subTest(regulator=regulator,q=q):
                    number,energy = regulated_integrals(q,regulator,1.58,197.3)
                    refn = quad(lambda x:x*math.sqrt(x*x+regulator**2),0,q,epsabs=1e-15)[0]
                    refe = quad(lambda x:x*math.sqrt(x*x+regulator**2)*math.sqrt(x*x+1.58**2)*197.3,
                                0,q,epsabs=1e-15)[0]
                    np.testing.assert_allclose([number,energy],[refn,refe],rtol=1e-7,atol=1e-13)

    def test_same_energy_as_original_simpson_functional(self):
        for n,fq,y in ((.2,0.,.07),(.6,.3,.1),(1.2,.75,.03)):
            self.assertAlmostEqual(self.eos.energy_hq_of_fq_y(fq,n,y),
                                   self.original.energy_hq_of_fq_y(fq,n,y),places=8)

    def test_analytic_beta_derivative_against_finite_differences(self):
        # Exercise all three EV derivatives, including species-dependent b.
        for name in ('clausius','clausius_cs','clausius_tvm'):
            fit = replace(self.fit,model=name)
            eos = PaperEOS(fit,settings=self.settings)
            for n,fq,y in ((.2,0.,.07),(.6,.7,.1),(1.2,.85,.03)):
                exact = eos.mu_q_analytic(n,fq,y)
                fd = eos.mu_q_fd(n,fq,y,dy=5e-5)
                self.assertTrue(math.isfinite(exact+fd))
                self.assertAlmostEqual(exact,fd,delta=2e-4)

    def test_neutral_variational_minimum_independent_of_beta_root(self):
        # Minimize the charge-neutral total energy directly in (fq,Y), using
        # leptons obtained from charge only, with no mu_Q or beta root solver.
        n = .7
        eos = self.eos
        def neutral_energy(point):
            fq,y = point
            mu = brentq(lambda m:eos.lepton_density_from_mu(m,.511)+
                        eos.lepton_density_from_mu(m,105.66)-n*y,.511,1500.)
            return eos.energy_hq_of_fq_y(fq,n,y)+eos.lepton_energy_density_from_mu(mu,.511)+eos.lepton_energy_density_from_mu(mu,105.66)
        independent = differential_evolution(neutral_energy,[(0.,1.),(1e-7,.499999)],
                           seed=42,tol=1e-10,popsize=12,polish=True)
        row = eos.build_beta_row(n)
        self.assertIsNotNone(row)
        self.assertAlmostEqual(row.energy_density,independent.fun,delta=2e-6)
        self.assertAlmostEqual(row.quark_fraction,independent.x[0],delta=3e-5)
        self.assertAlmostEqual(row.y,independent.x[1],delta=3e-5)
        self.assertLess(abs(eos.neutrality_residual(row.y,n,row.quark_fraction)),1e-9)

    def test_conservation_and_muon_threshold(self):
        eos = self.eos
        for n in (.08,.2,.6,1.2):
            row=eos.build_beta_row(n)
            self.assertIsNotNone(row)
            nu,nd=eos.nuid(row.k_bu,row.y),eos.ndid(row.k_bu)
            self.assertAlmostEqual(row.n_p+row.n_n+(nu+nd)/3,n,places=9)
            self.assertAlmostEqual(row.n_p+(2*nu-nd)/3,row.y*n,places=9)
            self.assertLess(abs(eos.mu_q_fd(n,row.quark_fraction,row.y)+row.mu_e),2e-4)
        self.assertEqual(eos.lepton_density_from_mu(105.,105.66),0.)
        self.assertEqual(eos.lepton_energy_density_from_mu(105.,105.66),0.)

    def test_dieterici_pure_quark_endpoint_has_finite_energy(self):
        fit=replace(self.fit,model='dieterici',parameter_name='alpha',parameter_value=1.7)
        original=AsymmetricQuarkyonicEOS(fit,settings=self.settings)
        self.assertTrue(math.isfinite(original.energy_hq_of_fq_y(1.,.6,.1)))

    def test_species_power_preserves_snm_pnm_calibration_and_beta_derivative(self):
        oldfit=compute_asymmetric_fit('dieterici',parameter_value=1.7103654736168037,
                      fit_settings=AsymmetricFitSettings(enabled=True,branch_mode='target_l'))
        newfit=species_power_couplings(oldfit)
        old=PaperEOS(oldfit,settings=self.settings)
        new=PaperEOS(newfit,settings=self.settings,attraction_mode='species_power')
        def symmetry(n):
            return (new.energy_hq_of_fq_y(0.,n,0.)-new.energy_hq_of_fq_y(0.,n,.5))/n
        for n in (.12,.16,.2):
            for y in (0.,.5):
                self.assertAlmostEqual(old.energy_hq_of_fq_y(0.,n,y),new.energy_hq_of_fq_y(0.,n,y),places=9)
        self.assertAlmostEqual(symmetry(.16),32.5,delta=1e-5)
        self.assertAlmostEqual(3*.16*(symmetry(.1601)-symmetry(.1599))/.0002,58.9,delta=1e-4)
        for n,fq,y in ((.2,0.,.07),(.6,.7,.1),(1.2,.85,.03)):
            self.assertAlmostEqual(new.mu_q_analytic(n,fq,y),new.mu_q_fd(n,fq,y,dy=2e-5),delta=3e-4)
        row=new.build_beta_row(.6)
        self.assertIsNotNone(row)
        self.assertLess(abs(new.mu_q_fd(.6,row.quark_fraction,row.y)+row.mu_e),1e-4)
        def neutral_energy(point):
            fq,y=point
            energy=new.energy_hq_of_fq_y(fq,.6,y)
            if not math.isfinite(energy):
                return 1e40
            mu=brentq(lambda m:new.lepton_density_from_mu(m,.511)+
                      new.lepton_density_from_mu(m,105.66)-.6*y,.511,1500.)
            return energy+new.lepton_energy_density_from_mu(mu,.511)+new.lepton_energy_density_from_mu(mu,105.66)
        direct=differential_evolution(neutral_energy,[(0.,1.),(1e-7,.499999)],
                      seed=19,tol=1e-10,popsize=12,polish=True)
        self.assertAlmostEqual(row.energy_density,direct.fun,delta=2e-6)
        self.assertAlmostEqual(row.y,direct.x[1],delta=3e-5)
        self.assertAlmostEqual(row.quark_fraction,direct.x[0],delta=3e-5)

    def test_cs_close_packing_overflow_cannot_abort_global_search(self):
        from qmm.models import MODELS,get_model
        # Same mapping as the established six-model registration.
        base,repulsive=get_model('dieterici'),get_model('clausius_cs')
        MODELS['dieterici_cs']=replace(base,name='dieterici_cs',**{name:getattr(repulsive,name)
                for name in ('nid_from_n','n_from_nid','volume_fraction','species_volume_fraction','pressure_prefactor')})
        fit=compute_asymmetric_fit('dieterici_cs',parameter_value=1.660938774,
                    fit_settings=AsymmetricFitSettings(enabled=True,branch_mode='target_l'))
        eos=PaperEOS(species_power_couplings(fit),settings=self.settings,attraction_mode='species_power')
        row=eos.build_beta_row(1.192)
        self.assertIsNotNone(row)
        self.assertTrue(math.isfinite(row.energy_density))

    def test_total_energy_thermodynamics_and_refinement(self):
        def error(step):
            n=np.arange(.1,1.+step/2,step)
            energy=938*n+25*n**(7/3)
            curve=thermodynamics(n,energy)
            mu=938+(7/3)*25*n**(4/3)
            exact=n*((7/3)*(4/3)*25*n**(1/3))/mu
            np.testing.assert_allclose(curve['pressure'][2:-2],((4/3)*25*n**(7/3))[2:-2],rtol=2e-4)
            return np.max(abs(curve['vs2'][2:-2]-exact[2:-2]))
        self.assertLess(error(.01),error(.02))


if __name__=='__main__':
    unittest.main()
