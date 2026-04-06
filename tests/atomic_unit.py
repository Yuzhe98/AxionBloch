##############################################
# for illustrating atomic and SI unit system
##############################################
from pint import UnitRegistry

# Create a Unit Registry for managing units
ureg = UnitRegistry()
ureg.define("tau = 2.4188843265863416e-17 second")

# Atomic units are defined such that certain fundamental physical constants are set to 1:
# hbar = 1, m_e = 1, e = 1, 4 \pi varepsilon_0 = 1

hbar = ureg("hbar")  # reduced Planck constant
m_e = ureg("m_e")  # electron mass
e = ureg("e")  # elementary charge
FourPiEpsilon0 = ureg("4*pi*epsilon_0")  # 4 times pi times vacuum permittivity
a_0 = ureg("a_0")  # Bohr radius = 4 * pi * epsilon_0 * hbar**2 / (m_e * e**2)
Eh = ureg("Eh")  # Hartree energy = hbar**2 / (m_e * a_0**2)
# 4.134137333518173e16

print("in SI units:")
print(f"hbar: {hbar.to('J*s')}")
print(f"m_e: {m_e.to('kg')}")
print(f"e: {e.to('C')}")
print(f"4 pi epsilon_0: {FourPiEpsilon0.to('F/m')}")
print(f"a_0: {a_0.to('m')}")
print(f"a_0: {ureg('4*pi*epsilon_0 * hbar**2 / (m_e * e**2)').to('m')}")
print(f"Eh: {Eh.to('eV')}")
print(f"Eh: {ureg('hbar**2 / (m_e * a_0**2)').to('eV')}")

print("more quantities in SI units:")
tau = ureg("hbar / Eh")

print(f"tau: {tau.to('s')}")

# conversions
# In atomic units:
# 1 kg mass: 1.0977691042634432e+30 electron_mass
# 1 m length: 18897261259.08201 bohr
# 1 s time: 4.1341373335171144e+16 tau
# 1 A current: 150.97488474459564 elementary_charge / tau
# 1 K temperature: 3.1668115634564225e-06 hartree / boltzmann_constant
mass = 1 * ureg("kg")
length = 1 * ureg("m")
time = 1 * ureg("s")
current = 1 * ureg("A")
temperature = 1 * ureg("K")
print("In atomic units:")
print(f"1 kg mass: {mass.to('m_e')}")
print(f"1 m length: {length.to('a_0')}")
print(f"1 s time: {time.to('tau')}")
print(f"1 A current: {current.to('e/tau')}")
print(f"1 K temperature: {temperature.to('Eh/k_B')}")
