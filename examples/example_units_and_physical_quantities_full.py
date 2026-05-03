from axionbloch.dependency import *

from axionbloch.constants import gamma_p

# dimensionless quantities with python scalars
# create dimensionless Quantity objects so adding to scalars works
a = 1.0 + 1.0 * unit.one + 1.0 * unit.dimensionless_unscaled
print("dimensionless quantities with python scalars:")
print(a)
# 3.0

# linewidth in units of ppm / ppb / ppt
# create dimensionless Quantity objects so adding to scalars works
linewidth = 1.0e-6 + 1.0 * ppm + 1.0e-6 * unit.dimensionless_unscaled
print("linewidth in ppm:", linewidth.to(ppm))
# 3.0 ppm

# define a physical quantity with Quantity
speed = Quantity(1.0, unit.km / unit.s)
print("speed in SI units:", speed.si)
# 1000.0 m / s

speed = Quantity(1.0, "km / s")
print("speed in CGS units:", speed.cgs)
# 100000.0 cm / s

# Define a quantity from scalars and units:
# a magnetic field of 1 Gauss
B = 1.0 * unit.gauss
print("magnetic field in SI units:", B.si)
# 0.0001 T


# find the 90 degree pulse duration
t90 = 0.5 * np.pi * unit.radian / (gamma_p * B)
print("90 degree pulse duration converted to microseconds:", t90.to(unit.microsecond))
# 58.71648792722992 us

# ------------- numpy operations ------------- #

# sine of an array of angles
tStamps = np.array([0, 1 / 3, 1]) * t90
phases = gamma_p * B * tStamps
print("phases at different time stamps:", np.sin(phases))
# [0.  0.5 1. ]

# absolute value
speed = -220.0 * unit.km / unit.s
print("speed:", speed)
print("absolute value of speed:", np.abs(speed))
