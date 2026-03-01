from axionbloch.axionwind import AxionWind

import numpy as np

from axionbloch.enphylope import PhysicalQuantity

axion = AxionWind(
    name="axion",
    nu_a = PhysicalQuantity(1.0e6, "Hz"),  # compton frequency
    gaNN = PhysicalQuantity(1.0e-9, "GeV**(-1)"),  #
    Qa = None,
    v_0 = PhysicalQuantity(
        220.0, "km/s"
    ),  # Local (@ solar radius) galaxy circular rotation speed
    v_lab = PhysicalQuantity(
        233.0, "km/s"
    ),  # Laboratory speed relative to the galactic rest frame
    windAngle = None,
    rho_E_DM = PhysicalQuantity(0.3, "GeV/cm**3"),
    verbose=False,
    )

axion.getRabiFreq(verbose=True)
frequencies = np.linspace(0.9e6, 1.1e6, 1000)  # in Hz
spec = axion.getAmpSpectra(frequencies=frequencies, verbose=True)
print(spec.shape)