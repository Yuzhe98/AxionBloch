# Physical constants
from axionbloch.enphylope import PhysicalQuantity as PQ

# Electron charge / "elementary_charge" in pint: 1.602176634e-19 coulomb
e = PQ(1, "e")

# mol to number by Avogadro's number
mol_to_num = PQ(6e23, "mol**(-1)")

# Light speed  speed_of_light: 299792458 (m / s)
c = PQ(1, "c")

# # Atomic mass unit: 931.4941037185688 megaelectron_volt / speed_of_light ** 2
# u = PQ(1, "unified_atomic_mass_unit")

# Boltzmann constant in eV K^-1
# k_B = kB = PQ(8.617333262145e-5, "eV / kelvin")
k_B = kB = PQ(1, "k_B")

# Reduced Planck constant
hbar = PQ(1, "hbar")

# Planck constant: 4.135667696e-15 (eV * s)
h_Planck = PQ(1, "planck_constant")

# Hartree energy in eV
Eh = E_hartree = PQ(1, "Eh")

# Masses of electron, proton, and neutron
m_e = PQ(1, "m_e")
m_p = PQ(1, "m_p")
m_n = PQ(1, "m_n")

# Bohr magneton: 5.788381798194462e-05 (eV / tesla)
mu_B = PQ(1, "mu_B")

# vacuum permeability: 1.25663706212e-6 (henry / m)
mu_0 = PQ(1, "mu_0")


# Nuclear magneton
def mu_N(m):
    return (-1.0 * e * hbar) / (2 * m)  # * c **2?? TODO: check the unit of mu_N


# Magnetic dipole moment of proton
g_p = PQ(5.585694713, "")
I_p = PQ(1 / 2, "") * hbar
mu_p = g_p * mu_N(m_p) * I_p / hbar

# Gyromagnetic ratio of proton
gamma_p = PQ(2.6752218708e8, "hertz / tesla")

# Magnetic dipole moment of Xe nucleus
mu_Xe129 = PQ(-0.777969, "dimensionless") * mu_N(m_p)

# Gyromagnetic ratio of Xe129
gamma_Xe129 = PQ(-7.441e7, "hertz / tesla")

grav_const = gravitational_constant = PQ(1, "gravitational_constant")

# Earth radius: 6.371e6 (meter)
earth_radius = PQ(1, "earth_radius")


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
