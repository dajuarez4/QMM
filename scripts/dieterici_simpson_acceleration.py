"""Vectorize the existing Simpson sums without changing integration grids."""
import math
from functools import lru_cache
import numpy as np
from qmm.quarkyonic import AsymmetricQuarkyonicEOS

@lru_cache(maxsize=8)
def grid(n):
    n += n % 2
    fractions = np.arange(n + 1, dtype=float) / n
    weights = np.ones(n + 1)
    weights[1:-1:2] = 4
    weights[2:-1:2] = 2
    return fractions, weights / (3*n)

def edid(self, k_bu):
    upper = k_bu / self.settings.nc
    if abs(upper) < 1e-14:
        return 0.
    x, w = grid(self.settings.quark_integral_points)
    q = upper*x
    values = q*np.sqrt(self.kappa**2+q*q)*np.sqrt((self.physical.hbarc*q)**2+self.m_q**2)
    return self.settings.nc*self.physical.degeneracy_species/(2*math.pi**2)*upper*float(w@values)

def euid(self, k_bu, y_value):
    return edid(self, k_bu)*(1+y_value)/(2-y_value)

def ndid(self, k_bu):
    upper = k_bu/self.settings.nc
    if abs(upper) < 1e-14:
        return 0.
    x, w = grid(self.settings.quark_integral_points)
    q = upper*x
    return self.settings.nc*self.physical.degeneracy_species/(2*math.pi**2)*upper*float(w@(q*np.sqrt(self.kappa**2+q*q)))

def nuid(self, k_bu, y_value):
    return ndid(self, k_bu)*(1+y_value)/(2-y_value)

def lepton_energy_density_from_mu(self, mu_l, mass_l):
    if mu_l <= mass_l:
        return 0.
    upper = self.lepton_kf_from_mu(mu_l, mass_l)
    if abs(upper) < 1e-14:
        return 0.
    x, w = grid(self.settings.shell_integral_points)
    k = upper*x
    return upper*float(w@(k*k*np.sqrt((self.physical.hbarc*k)**2+mass_l*mass_l)))/math.pi**2

def install():
    for function in (edid,euid,ndid,nuid,lepton_energy_density_from_mu):
        setattr(AsymmetricQuarkyonicEOS,function.__name__,function)
