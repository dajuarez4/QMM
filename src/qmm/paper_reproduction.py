"""Auditable evaluator for the Lambda=300 paper comparison.

Uses the existing QMM energy functional and fits. Analytic integrals and an
analytic composition derivative remove nested quadrature/finite-difference
noise. Beta roots and quark-fraction minima are searched before refinement.
No parameter is adjusted to the reference images.
"""
from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .quarkyonic import AsymmetricQuarkyonicEOS


def species_power_couplings(fit):
    """Map the existing SNM/PNM calibration to the manuscript's Dieterici form.

    In SNM, sum_i n_i v_i = -2**(2-alpha)*a_avg*nH**alpha/(alpha-1).
    Thus a_avg(paper)=2**(alpha-2)*a_avg(QMM). PNM fixes the SAME a_n;
    a_pn then follows from the symmetric average. SNM and PNM energies agree
    at every density, so K0, binding, J=E_PNM-E_SNM, L and b's are preserved.
    The intermediate-composition energy, hence beta equilibrium, changes.
    """
    if not fit.model.startswith('dieterici'):
        return fit
    average = fit.a_avg*2**(fit.parameter_value-2)
    cross = 2*average-fit.a_n
    if min(average,cross,fit.a_n) <= 0:
        raise ValueError('Nonpositive species-power Dieterici coupling.')
    return replace(fit,a_avg=average,a_pn=cross,ratio_a_pn_over_a_n=cross/fit.a_n)


def species_power_interaction(np_, nn, a_n, a_pn, alpha):
    return -(a_n*(np_**alpha+nn**alpha)+a_pn*(np_*nn**(alpha-1)+nn*np_**(alpha-1)))/(alpha-1)


def species_power_derivative(nh, y, a_n, a_pn, alpha):
    """Partial interaction-energy derivative with respect to Y at fixed nH."""
    if nh == 0:
        return 0.
    np_,nn = nh*y,nh*(1-y)
    return -nh/(alpha-1)*(a_n*alpha*(np_**(alpha-1)-nn**(alpha-1))+
             a_pn*(nn**(alpha-1)-np_**(alpha-1)+(alpha-1)*
                    (nn*np_**(alpha-2)-np_*nn**(alpha-2))))


def regulated_integrals(q: float, regulator: float, mass: float, hbarc: float):
    """Return integral q sqrt(q²+L²) dq and its energy-weighted analogue.

    Momenta/regulator/mass are in fm^-1; the energy integral is in MeV fm^-3.
    The substitution t=q² makes the energy integral elementary. log1p and
    rationalized root differences avoid subtracting two large primitives.
    """
    if q <= 0:
        return 0.0, 0.0
    t, a, b = q*q, regulator*regulator, mass*mass
    number = ((t+a)**1.5)/3 if a == 0 else a**1.5*math.expm1(1.5*math.log1p(t/a))/3
    s, d = (a+b)/2, (a-b)/2
    r0, r1 = math.sqrt(a*b), math.sqrt((t+a)*(t+b))
    dr = t*(t+2*s)/(r1+r0)
    if s+r0 == 0:
        energy = hbarc*t*t/4
    elif min(a,b) == 0 and t < max(a,b)*1e-3:
        # In this limit the closed form cancels to O(t**1.5).
        nodes, weights = _gauss()
        v = q*(nodes+1)/2
        energy = hbarc*q/2*float(weights @ (v*np.sqrt(v*v+a)*np.sqrt(v*v+b)))
    else:
        energy = hbarc*(t*r1+s*dr-d*d*math.log1p((t+dr)/(s+r0)))/4
    return number, energy


@lru_cache(maxsize=1)
def _gauss():
    return np.polynomial.legendre.leggauss(48)


class PaperEOS(AsymmetricQuarkyonicEOS):
    """Same QMM functional, with checked integrals and global candidate scans."""

    def __init__(self, *args, attraction_mode='qmm', **kwargs):
        super().__init__(*args, **kwargs)
        if attraction_mode not in ('qmm','species_power'):
            raise ValueError('Choose qmm or species_power attraction explicitly.')
        self.attraction_mode = attraction_mode
        if self.settings.nc != 3 or self.settings.momentum_mode != 'quarkyonic':
            raise ValueError('PaperEOS requires Nc=3 and quarkyonic momentum space.')

    def energy_hq_of_fq_y(self, fq, n_b, y):
        energy = super().energy_hq_of_fq_y(fq,n_b,y)
        if self.attraction_mode != 'species_power' or not self.fit_result.model.startswith('dieterici'):
            return energy
        np_,nn = self.hadronic_densities(n_b,fq,y)
        nh = np_+nn
        if nh == 0 or not math.isfinite(energy):
            return energy
        from .asymmetry import attractive_coefficient
        f = self.fit_result
        old = nh*attractive_coefficient(y,f.a_n,f.a_pn)*self.model.U(nh,0.,f.parameter_value)
        return energy-old+species_power_interaction(np_,nn,f.a_n,f.a_pn,f.parameter_value)

    @lru_cache(maxsize=1024)
    def _integrals(self, k_bu):
        return regulated_integrals(k_bu/3, self.kappa,
                                   self.m_q/self.physical.hbarc, self.physical.hbarc)

    def ndid(self, k_bu):
        return 3*self.physical.degeneracy_species/(2*math.pi**2)*self._integrals(k_bu)[0]

    def nuid(self, k_bu, y_value):
        return self.ndid(k_bu)*(1+y_value)/(2-y_value)

    def edid(self, k_bu):
        return 3*self.physical.degeneracy_species/(2*math.pi**2)*self._integrals(k_bu)[1]

    def euid(self, k_bu, y_value):
        return self.edid(k_bu)*(1+y_value)/(2-y_value)

    def lepton_energy_density_from_mu(self, mu_l, mass_l):
        if mu_l <= mass_l:
            return 0.0
        p = math.sqrt(mu_l*mu_l-mass_l*mass_l)
        if p < mass_l*.1:
            nodes, weights = _gauss()
            v = p*(nodes+1)/2
            return p/(2*math.pi**2*self.physical.hbarc**3)*float(
                weights @ (v*v*np.sqrt(v*v+mass_l*mass_l)))
        return (p*mu_l*(2*p*p+mass_l*mass_l)-mass_l**4*math.asinh(p/mass_l))/(8*math.pi**2*self.physical.hbarc**3)

    def mu_q_analytic(self, n_b, fq, y):
        """(1/nB) partial epsilon_HQ/partial Y at fixed nB and fq.

        Includes the Y dependence of the common inner momentum (at fixed
        quark baryon density), both volume fractions, and the attraction.
        """
        k = self.kbu_solver(n_b, fq, y)
        kp, kn = self.kfp_solver(n_b, fq, y, k), self.kfn_solver(n_b, fq, y, k)
        if not all(map(math.isfinite, (kp, kn))):
            return math.nan
        pref = self.physical.degeneracy_species/(2*math.pi**2)
        number, energy = self._integrals(k)
        hc, m = self.physical.hbarc, self.physical.m_nucleon
        eq = math.sqrt((hc*k/3)**2+self.m_q**2)
        dk = -3*number/((k/3)*math.sqrt((k/3)**2+self.kappa**2)*(2-y)) if k > 0 else 0.0
        derivative = 9*pref*(energy-number*eq)/(2-y)**2
        nh = n_b*(1-fq)
        np_, nn = self.hadronic_densities(n_b, fq, y)
        fp, fn = self.volume_fractions(n_b, fq, y)
        ebu = math.sqrt((hc*k)**2+m*m)
        for ni, fi, ki, sign in ((np_, fp, kp, 1), (nn, fn, kn, -1)):
            if fi is None or fi <= 0:
                return math.nan
            xp = nh*(self.fit_result.b_n-self.fit_result.b_pn)*sign
            # d(log f)/dx, for exactly the three EV functions in QMM.
            name = self.fit_result.model
            if name == 'cs' or name.endswith('_cs'):
                x = (self.fit_result.b_n*ni+self.fit_result.b_pn*(nh-ni))
                gprime = -8*(8-x)/(4-x)**3
            elif name == 'tvm' or name.endswith('_tvm'):
                x = (self.fit_result.b_n*ni+self.fit_result.b_pn*(nh-ni))
                gprime = -1-x
            else:
                gprime = -1/fi
            df = fi*gprime*xp
            ei = math.sqrt((hc*ki)**2+m*m)
            shell = self.enid(k, ki)
            derivative += ei*(sign*nh-ni*df/fi)+df*shell+fi*pref*k*k*(ei-ebu)*dk
        u = self.model.U(nh, 0., self.fit_result.parameter_value) if nh > 0 else 0.
        if u is None:
            return math.nan
        derivative += nh*2*(self.fit_result.a_pn-self.fit_result.a_n)*(1-2*y)*u
        if self.attraction_mode == 'species_power' and self.fit_result.model.startswith('dieterici'):
            derivative -= nh*2*(self.fit_result.a_pn-self.fit_result.a_n)*(1-2*y)*u
            derivative += species_power_derivative(nh,y,self.fit_result.a_n,self.fit_result.a_pn,
                                                   self.fit_result.parameter_value)
        return derivative/n_b

    def mu_e_from_y(self, n_b, fq, y_value):
        return -self.mu_q_analytic(n_b, fq, y_value)

    def _y_bounds(self, n_b, fq, y_min, y_max):
        name = self.fit_result.model
        if name == 'tvm' or name.endswith('_tvm'):
            return y_min, y_max
        limit = 4. if name == 'cs' or name.endswith('_cs') else 1.
        nh = n_b*(1-fq)
        for offset, slope in ((self.fit_result.b_pn, self.fit_result.b_n-self.fit_result.b_pn),
                              (self.fit_result.b_n, self.fit_result.b_pn-self.fit_result.b_n)):
            offset, slope = nh*offset, nh*slope
            if slope == 0:
                if offset >= limit:
                    return 1., 0.
            elif slope > 0:
                y_max = min(y_max, (limit*(1-1e-11)-offset)/slope)
            else:
                y_min = max(y_min, (limit*(1-1e-11)-offset)/slope)
        return y_min, y_max

    def solve_y_beta_equilibrium(self, n_b, fq, y_min=1e-8, y_max=.5-1e-8, n_scan=None):
        left, right = self._y_bounds(n_b, fq, y_min, y_max)
        if left >= right:
            return (math.nan,)*4
        grid = np.linspace(left, right, n_scan or self.settings.beta_y_scan_points)
        residuals = [self.neutrality_residual(float(y), n_b, fq) for y in grid]
        candidates = []
        for a, b, fa, fb in zip(grid[:-1], grid[1:], residuals[:-1], residuals[1:]):
            if not (math.isfinite(fa) and math.isfinite(fb)):
                continue
            if fa == 0:
                y = float(a)
            elif (fa < 0) != (fb < 0):
                try:
                    y = brentq(lambda v: self.neutrality_residual(v, n_b, fq), a, b, xtol=2e-13)
                except ValueError as exc:
                    # Near CS close packing, f can be nonzero while epsilon
                    # overflows. Finite bracket endpoints do not guarantee a
                    # representable interior. Such a bracket is not a root.
                    if 'NaN' in str(exc):
                        continue
                    raise
            else:
                continue
            mu = self.mu_e_from_y(n_b, fq, y)
            ee = self.lepton_energy_density_from_mu(mu, .511)
            em = self.lepton_energy_density_from_mu(mu, 105.66)
            candidates.append((self.energy_hq_of_fq_y(fq, n_b, y)+ee+em, (y, mu, ee, em)))
        return min(candidates, key=lambda row: row[0])[1] if candidates else (math.nan,)*4

    def solve_fq_beta(self, n_b, fq_min=0., fq_max=1., y_scan=None):
        # A whole-interval golden search assumes unimodality. Scan first and
        # refine every finite local minimum, keeping both physical endpoints.
        objective = lambda f: self.total_energy_beta(float(f), n_b, y_scan)
        grid = np.linspace(fq_min, fq_max, self.settings.fq_scan_points)
        # Resolve onset without depending on the much coarser full-fraction grid.
        grid = np.unique(np.r_[grid, np.geomspace(1e-10, min(.04, fq_max), 16)])
        grid = grid[(grid >= fq_min) & (grid <= fq_max)]
        values = [objective(f) for f in grid]
        candidates = [(float(f), e) for f, e in zip(grid, values) if math.isfinite(e)]
        if not candidates:
            return math.nan, math.nan
        for i, value in enumerate(values):
            if not math.isfinite(value):
                continue
            prev = values[i-1] if i > 0 and math.isfinite(values[i-1]) else math.inf
            after = values[i+1] if i+1 < len(values) and math.isfinite(values[i+1]) else math.inf
            if value <= prev and value <= after:
                a, b = float(grid[max(i-1, 0)]), float(grid[min(i+1, len(grid)-1)])
                if b > a:
                    def finite_objective(f):
                        e = objective(f)
                        return e if math.isfinite(e) else 1e100
                    result = minimize_scalar(finite_objective,
                                             bounds=(a,b), method='bounded',
                                             options={'xatol': self.settings.refine_tol})
                    if result.success and math.isfinite(result.fun):
                        candidates.append((float(result.x), float(result.fun)))
        return min(candidates, key=lambda row: row[1])


def thermodynamics(n, energy):
    """Unsmoothed local cubic derivatives and the independent gradient check.

    CubicSpline is interpolating (no smoothing, no clipping). Grid refinement,
    rather than a fixed moving window, controls sharp peak resolution.
    """
    from scipy.interpolate import CubicSpline
    from .sound_speed import reconstruct_sound_speed_curve_gradient_check
    n, energy = np.asarray(n), np.asarray(energy)
    if len(n) < 4 or not np.all(np.diff(n) > 0) or not np.isfinite(energy).all():
        raise ValueError('Need >=4 finite energy samples at increasing densities.')
    spline = CubicSpline(n, energy)
    mu, second = spline(n, 1), spline(n, 2)
    gradient = reconstruct_sound_speed_curve_gradient_check(n.tolist(), energy.tolist())
    return dict(mu_b=mu, pressure=n*mu-energy, vs2=n*second/mu,
                pressure_gradient=np.asarray(gradient.pressure), vs2_gradient=np.asarray(gradient.vs2))
