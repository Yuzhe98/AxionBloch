from math import tau

from axionbloch.utils import coh_time_g1

import numpy as np
import time


timeStamp = np.linspace(
    0, 1e1, 10_000
)  # Simulated time stamps corresponding to the tau values
T2 = 1.93e-1
nu = 1e2  # Frequency in Hz
# signal = np.exp(-timeStamp / T2) * np.exp(-1j * 2 * np.pi * nu * timeStamp)
signal = 1j * np.random.uniform(-1, 1, size=timeStamp.shape) + np.random.uniform(
    -1, 1, size=timeStamp.shape
)
# Simulated signal


# ---------------------------------
# Integral definition
# τ_c = ∫ |g1(τ)|^2 dτ
# ---------------------------------

# Trapezoidal integration
tau_coh = coh_time_g1(
    signal, timeStamp[1] - timeStamp[0]
)  # Using the sampling interval from timeStamp

print(f"Integral coherence time: {tau_coh:.3f} s")
