##############################################
# for unit management and physical calculation
##############################################

from pint import UnitRegistry
from typing import Optional
import numpy as np

# Create a Unit Registry for managing units
ureg = UnitRegistry()
# # Define units
# ureg.define("Gauss = 1e-4 * tesla = G")  # Gauss
# ureg.define(
#     "parsec = 30856775814913673 * meter = pc"
# )  # Parsec. The built-in parsec is not precise
# ureg.define("solar_mass = 1.98847e30 * kilogram = M_sun")  # Solar Mass
# ureg.define("earth_mass = 5.9722e24 * kilogram = M_earth")  # Earth Mass

# ureg.define("ppb = 1e-9")  # parts per billion
# ureg.define("ppt = 1e-12")  # parts per trillion
# ureg.define("ppq = 1e-15")  # parts per quadrillion
# ureg.define("ppqu = 1e-18")  # parts per quintillion
# ureg.define("ppmu = 1e-21")  # parts per sextillion
# ureg.define("ppbmu = 1e-24")  # parts per septillion

# ureg.define("m_e = 1 * kg = me")
# ureg.define("a_0 = 1 * meter = a0")
# ureg.define("tau = 1 * second = tau")
# ureg.define("e = 1 * ampere * second = e")

# print(ureg("m_e").to("kg"))
# print(ureg("m_p").to("kg"))
# print(ureg("e").to("C"))

# print(ureg("kg").to("m_e"))

# print(ureg("e").to("e"))
# print(ureg("hbar").to_base_units())


# convert using atomic unit system
# q = 1 * ureg("kg*m* s  *A*  K")
# q_au = q.to(
#     ureg.m_e ** q.dimensionality.get("[mass]", 0)
#     * ureg.a_0 ** q.dimensionality.get("[length]", 0)
#     * ureg.tau ** q.dimensionality.get("[time]", 0)
#     * ureg.e ** q.dimensionality.get("[current]", 0)
# )

# print(ureg.get_dimensionality("kg* m* s  *A*  K  *  cd"))
# print(q.dimensionality.get("[mass]", 0))
# print(q.dimensionality.get("[length]", 0))
# print(q.dimensionality.get("[time]", 0))
# print(q.dimensionality.get("[current]", 0))
# print(q.dimensionality.get("[temperature]", 0))
# print(q.dimensionality.get("[substance]", 0))
# print(q.dimensionality.get("[luminosity]", 0))

# q = 1 * ureg("kg * s")
# print(q.to_reduced_units())
# print(q.to_base_units())
# print(q.to_compact())
# print(q.to("a_0", "m_e", "tau", "e"))
# print((q*q).to("m_e**2 * s**2"))




tau_AU = ureg("tau")

print(tau_AU.to("s"))