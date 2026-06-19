# Define a quantity from scalars and units:
# a magnetic field of 1 Gauss
from astropy import units as unit

B = 1.0 * unit.gauss
print(B.si)
# <Quantity 0.0001 T>

import numpy as np

# Import constants and use them with scalars
from axionbloch.constants import gamma_H1

# find the 90 degree pulse duration
t90 = np.pi / 2 / (gamma_H1 * B)
print(t90.to(unit.microsecond))
# <Quantity 5.87164879e-05 s>

# Operation on an array of quantities with numpy
tStamps = np.array([0, 1 / 3, 1]) * t90
phases = gamma_H1 * B * tStamps * unit.radian
print(np.sin(phases))
# <Quantity [0. , 0.5, 1. ]>
