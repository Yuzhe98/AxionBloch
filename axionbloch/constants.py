# Physical constants
from .enphylope import PhysicalQuantity

# Electron charge / "elementary_charge" in pint: 1.602176634e-19 coulomb
e = PhysicalQuantity(1, "e")

# mol to number by Avogadro's number
mol_to_num = PhysicalQuantity(6e23, "mol**(-1)")

# Light speed  speed_of_light: 299792458 (m / s)
c = PhysicalQuantity(1, "c")

# Atomic mass unit: 931.4941037185688 megaelectron_volt / speed_of_light ** 2
u = PhysicalQuantity(1, "unified_atomic_mass_unit")

# Boltzmann constant in eV K^-1
kB = PhysicalQuantity(8.617333262145e-5, "eV / kelvin")

# Reduced Planck constant
hbar = PhysicalQuantity(1, "hbar")

# Planck constant: 4.135667696e-15 (eV * s)
h_Planck = PhysicalQuantity(1, "planck_constant")

# Masses of electron, proton, and neutron
m_e = PhysicalQuantity(1, "m_e")
m_p = PhysicalQuantity(1, "m_p")
m_n = PhysicalQuantity(1, "m_n")

# Bohr magneton: 5.788381798194462e-05 (eV / tesla)
mu_B = PhysicalQuantity(1, "mu_B")

# vacuum permeability: 1.25663706212e-6 (henry / m)
mu_0 = PhysicalQuantity(1, "mu_0")


# Nuclear magneton
def mu_N(m):
    return (-1.0 * e * hbar) / (2 * m)  # * c **2


# Magnetic dipole moment of proton
g_p = PhysicalQuantity(5.585694713, "")
I_p = PhysicalQuantity(1 / 2, "") * hbar
mu_p = g_p * mu_N(m_p) * I_p / hbar

# Gyromagnetic ratio of proton
gamma_p = PhysicalQuantity(2.6752218708e8, "hertz / tesla")

# Magnetic dipole moment of Xe nucleus
mu_Xe129 = PhysicalQuantity(-0.777969, "dimensionless") * mu_N(m_p)

# Gyromagnetic ratio of Xe129
gamma_Xe129 = PhysicalQuantity(-7.441e7, "hertz / tesla")


# Earth radius
earth_radius = PhysicalQuantity(6.371e6, "meter")