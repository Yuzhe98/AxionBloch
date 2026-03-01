# a more careful estimation of the coherence time, by looking at the interference of axion fields of different eigenstates.
# (n_r)  (l)  Principal(n)   Name   Eigen E (eV)   Kinetic T (eV)  Mean v (m/s)
# -----------------------------------------------------------------
# 0      0    1              1s     8.196e-19       3.836e-19       4.083e+03
# 1      0    2              2s     2.229e-18       3.836e-19       4.083e+03
# 0      1    2              2p     3.259e-18       2.969e-19       3.592e+03
# 2      0    3              3s     3.131e-18       6.412e-19       5.279e+03
# 1      1    3              3p     4.223e-18       7.976e-19       5.888e+03
# 0      2    3              3d     3.964e-18       2.396e-19       3.227e+03
# 3      0    4              4s     3.703e-18       6.412e-19       5.279e+03
# 2      1    4              4p     4.861e-18       8.957e-19       6.240e+03
# 1      2    4              4d     4.692e-18       4.601e-19       4.472e+03
# 0      3    4              4f     4.478e-18       1.854e-19       2.839e+03
# 4      0    5              5s     4.159e-18       5.068e-19       4.693e+03
# 3      1    5              5p     5.158e-18       6.380e-19       5.266e+03
# 2      2    5              5d     5.072e-18       3.448e-19       3.871e+03
# 1      3    5              5f     5.015e-18       2.420e-19       3.243e+03
# 0      4    5              5g     nan             nan             nan

import signal
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from axionbloch.enphylope import PhysicalQuantity
from axionbloch.constants import c as c_SI, h_Planck
from axionbloch.utils import Lorentzian, coh_time_g1

max_n_r = 20  # maximum principal quantum number to plot
quantum_l = 0  # angular momentum quantum number

# axion Compton frequency
# 1 MHz
nu_a = PhysicalQuantity(1, "MHz")
# axion mass
m_a = nu_a * h_Planck / c_SI**2
print("axion Compton frequency =", nu_a)
print("axion mass =", m_a.convert_to("kg"), " =", m_a.convert_to("eV/c**2"))


duration_s = 1e2  # in seconds
sampRate_Hz = 1e1  # in Hz
dataLen = int(duration_s * sampRate_Hz)
timeStamp_s = np.linspace(0, duration_s, dataLen)  # in seconds
dt = timeStamp_s[1] - timeStamp_s[0]  # sampling interval in seconds

freq = np.fft.fftfreq(dataLen, d=dt)
freq_shifted = np.fft.fftshift(freq)
FWHM_Hz = .1
spectrum = Lorentzian(freq_shifted, center=0, FWHM=FWHM_Hz) * np.exp(
    -1j * np.random.uniform(0, 2 * np.pi, dataLen)
)
spectrum_shifted = np.fft.fftshift(spectrum)

axion_field = np.fft.ifft(spectrum_shifted)

fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures

ax00 = fig.add_subplot(gs[0, 0])
ax00.plot(timeStamp_s, axion_field.real, label="real part")
ax00.plot(timeStamp_s, axion_field.imag, label="imaginary part")
ax00.set_xlabel("time (s)")
ax00.set_ylabel("field amplitude (arb. units)")
ax00.legend()
fig.suptitle("", wrap=True)

plt.tight_layout()
plt.show()


freq = np.fft.fftfreq(len(axion_field), d=timeStamp_s[1] - timeStamp_s[0])
axion_fft = np.fft.fft(axion_field)
freq_shifted = np.fft.fftshift(freq)
axion_fft_shifted = np.fft.fftshift(axion_fft)

fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures

ax00 = fig.add_subplot(gs[0, 0])
ax00.plot(freq_shifted, np.abs(axion_fft_shifted), label="fft")
# ax00.plot(timeStamp_s, axion_field.imag, label="imaginary part")
ax00.set_xlabel("frequency (Hz)")
ax00.set_ylabel("field amplitude (arb. units)")
# ax00.set_xscale('log')
# ax00.set_yscale('log')
ax00.legend()
# #############################################################################
fig.suptitle("", wrap=True)

plt.tight_layout()
# plt.savefig('example figure - one-column.png', transparent=False)
plt.show()


# A_t = np.exp(
#     -timeStamp_s / 1.2
# )  # an exponential decay envelope with a characteristic time of 0.1 seconds
# axion_field = A_t * np.exp(-1j * (2 * np.pi * 1e1 * timeStamp_s + 0.2))  # in seconds

print(
    "axion field coherence time:",
    coh_time_g1(axion_field, timeStamp_s[1] - timeStamp_s[0]),
    "s",
)
