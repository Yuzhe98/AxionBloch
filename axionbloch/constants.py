# Physical constants
from astropy import units as unit
from astropy.constants import codata2018 as const

# Nuclear magneton
def mu_N(m):
    return (const.e * const.hbar) / (2 * m)


# Magnetic dipole moment of proton
g_p = 5.585694713
I_p = 0.5 * const.hbar
mu_p = g_p * mu_N(const.m_p) * I_p / const.hbar

# Gyromagnetic ratio of proton
gamma_p = 2.6752218708e8 * unit.Hz / unit.T

# Magnetic dipole moment of Xe nucleus
mu_Xe129 = -0.777969 * mu_N(const.m_p)

# Gyromagnetic ratio of Xe129
gamma_Xe129 = -7.441e7 * unit.Hz / unit.T

# Earth radius: 6.371e6 (meter)
earth_radius = 1 * unit.R_earth


# atomic unit
# conversion constants to atomic units
class AtomicUnits:
    hbar = 1.0
    m_e = 1.0
    a_0 = 1.0
    e = 1.0
    hartree = 1.0
    Eh = hartree

    nm = 1.8897261246257702e1
    Å = 1.8897261246257702
    eV = 0.03674932217565499
    ps = 4.134137333518212e4
    picoseconds = 4.134137333518212e4

    fs = 4.134137333518212 * 10.0
    femtoseconds = 4.134137333518212 * 10.0

    V = 0.03674932217565499
    V_m = 1.9446903811488876e-12
    T = 4.254382157326325e-06
    meter = m = 1.8897261246257702e10
    km = 1.8897261246257702e13
    C = 6.241509074460763e18
    s = 4.134137333518173e16
    Hz = 2.4188843265857225e-17
    kg = 1.0977691057577634e30
    joule = J = 2.293712278396328e17
    A = 150.974884744557

    # some physical constants, expressed in atomic units

    k = 0.5  # hbar**2 / (2*m_e)
    m_p = 1836.1526734400013
    𝜇0 = 0.0006691762566207213
    ε0 = 0.0795774715459477
    c = 137.035999083818
    α = 0.0072973525693

    # earth profiles
    earth_radius = 6371.0e3 * m  #
