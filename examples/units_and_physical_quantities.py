# Define a quantity from scalars and units:
# a magnetic field of 1 Gauss
from axionbloch.enphylope import PhysicalQuantity as PQ

B = PQ(1.0, "Gauss")
print(B.to("tesla"))
# <0.0001 tesla>

# Import constants and use them in calculations with scalars
from axionbloch.constants import gamma_p
import numpy as np

# find the 90 degree pulse duration
t90 = np.pi / 2 / (gamma_p * B)
print(t90.to("microsecond"))
# <58.71648792722992 microsecond>

# Operation on an array of quantities with numpy
tStamps = np.array([0, 1 / 3, 2 / 3, 1]) * t90
phases = (gamma_p * B * tStamps).to("")
print(np.sin(phases))
# <[0.0 0.49999999999999983 0.8660254037844385 1.0] dimensionless>
