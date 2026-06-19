"""
Physical constants and unit conversion factors used throughout the axionbloch package.

Provides:
- Nuclear magneton helper ``mu_N``
- Proton and Xe-129 gyromagnetic ratios / magnetic moments

All constants with units are astropy Quantity objects.
"""

from astropy import units as unit
from astropy.constants import codata2018 as const


def mu_N(m):
    """Return the nuclear magneton e*hbar / (2*m) for a nucleus of mass m."""
    return (const.e * const.hbar) / (2 * m)


# Magnetic dipole moment of proton
g_p = 5.585694713
I_p = 0.5 * const.hbar
mu_p = g_p * mu_N(const.m_p) * I_p / const.hbar

# Gyromagnetic ratio of H-1 nucleus (proton)
gamma_p = 2.6752218708e8 * unit.rad * unit.Hz / unit.T

# Magnetic dipole moment of Xe nucleus
mu_Xe129N = -0.777969 * mu_N(const.m_p)

# Gyromagnetic ratio of Xe129 nucleus
gamma_Xe129N = -7.441e7 * unit.rad * unit.Hz / unit.T
