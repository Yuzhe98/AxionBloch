from .enphylope import PhysicalQuantity

# Electron charge
e = PhysicalQuantity(-1.602176634e-19, "coulomb")

# mol to number by Avogadro's number
mol_to_num = PhysicalQuantity(6e23, "mol**(-1)")

# Constants

# Light speed  speed_of_light
# c = PhysicalQuantity(299792458, "m / s")
c = PhysicalQuantity(1, "c")

# Atomic mass unit
u = PhysicalQuantity(931.49410242, "MeV / c**2")

# Boltzmann constant in eV K^-1
kB = PhysicalQuantity(8.617333262145e-5, "eV / kelvin")
# Boltzmann constant in J K^-1
# kB = PhysicalQuantity(1.380649e-23, "joule / kelvin")

# Planck constant
h_Planck = PhysicalQuantity(4.135667696e-15, "eV * s")

# Reduced Planck constant
hbar = PhysicalQuantity(1, "hbar")

# Masses of electron, proton, and neutron
m_e = PhysicalQuantity(0.51099895000, "MeV / c**2")
m_p = PhysicalQuantity(938.27208816, "MeV / c**2")
m_n = PhysicalQuantity(939.56542052, "MeV / c**2")

# Bohr magneton
mu_B = PhysicalQuantity(5.7883818012e-5, "eV / tesla")

# Magnetic permeability of free space
mu_0 = PhysicalQuantity(1.25663706212e-6, "henry / m")


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
