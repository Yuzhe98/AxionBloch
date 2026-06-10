from axionbloch.dependency import *
from axionbloch.constants import gamma_p
from axionbloch.utils import Lorentzian

# dimensionless quantities with python scalars
# create dimensionless Quantity objects so adding to scalars works
a = 1.0 + 1.0 * unit.one + 1.0 * unit.one
print("dimensionless quantities with python scalars:")
print("1.0 + 1.0 * unit.one + 1.0 * unit.one =", a)
# 3.0

# linewidth in units of ppm / ppb / ppt
# create dimensionless Quantity objects so adding to scalars works
linewidth = 1.0e-6 + 1.0 * ppm + 1.0e-6 * unit.one
print("linewidth in ppm:", linewidth.to(ppm))
# 3.0 ppm

# Lorentzian
print("\nLorentzian")
frequencies = np.linspace(-10, 10, 21) * unit.kHz
PSD = Lorentzian(x=frequencies, center=0, FWHM=1.0e3 * unit.Hz, area=1, offset=0)
print("Lorentzian PSD:", PSD.to(unit.kHz ** (-1)))

# define a physical quantity with Quantity
speed = Quantity(1.0, unit.km / unit.s)
print("speed in SI units:", speed.si)
# 1000.0 m / s

speed = Quantity(1.0, "km / s")
print("speed (defined by unit string) in CGS units:", speed.cgs)
# 100000.0 cm / s
print("speed in unit of light speed:", speed.to(const.c))
# 3.3356409519815205e-06 2.99792e+08 m / s

# Define a quantity from scalars and units:
# a magnetic field of 1 Gauss
B = 1.0 * unit.gauss
print("magnetic field in SI units:", B.si)
# 0.0001 T

mag_flux = 1 * magnetic_flux_quantum
print("electron charge", (1 * const.e).to(unit.coulomb))
print("magnetic_flux_quantum unit:", magnetic_flux_quantum)
print("magnetic flux:", mag_flux)
print("magnetic flux.to(unit.Wb):", mag_flux.to(unit.Wb))

# find the 90 degree pulse duration
t90 = 0.5 * np.pi * unit.radian / (gamma_p * B)
print("90 degree pulse duration converted to microseconds:", t90.to(unit.microsecond))
# 58.71648792722992 us

# max min
print("\nmax or min:")
rate0 = 2 * unit.Hz
rate1 = 1 * unit.MHz
rate2 = -3 * unit.kHz
# array = np.asanyarray([rate0, rate1, rate2])
max_rate = max([rate0, rate1, rate2])  # → 1.0 MHz
min_rate = min([rate0, rate1, rate2])  # → -3.0 kHz
print("max_rate =", max_rate)
print("min_rate =", min_rate)

# ------------- numpy operations ------------- #
print("\n# ------------- numpy operations ------------- #")
# sine of an array of angles
tStamps = np.array([0, 1 / 3, 1]) * t90
phases = gamma_p * B * tStamps
print("phases at different time stamps:", np.sin(phases))
# [0.  0.5 1. ]

# absolute value
speed: Quantity[unit.km / unit.s] = -220.0 * unit.km / unit.s
print("speed:", speed)
speed = np.abs(speed)
print("absolute value of speed:", np.abs(speed))

# convert to numpy array (real signal)
signal_list = [
    1,
    1,
    54,
    6,
    7,
    87,
] * unit.milliVolt
print("signal as a list", signal_list)
print("np.array(signal) (unit info lost)", np.array(signal_list))
print("np.asarray(signal) (unit info lost)", np.asarray(signal_list))
print("np.asanyarray(signal) (unit info kept!)", np.asanyarray(signal_list))
signal_arr = np.asanyarray(signal_list)
print("signal array mean", signal_arr.mean())
print("signal array std", signal_arr.std())
print("signal array var", signal_arr.var())

# convert to numpy array (complex signal)
complex_signal_list = [
    1 + 1j,
    1 + 1j * 11.0,
    54,
    6 - 1j * 121.0,
    7,
    87,
] * unit.milliVolt
complex_signal_arr = np.asanyarray(complex_signal_list)
print("complex signal array mean", complex_signal_arr.mean())
print("complex signal array std", complex_signal_arr.std())
print("complex signal array var", complex_signal_arr.var())
print("complex signal array real", complex_signal_arr.real)
print("complex signal array imag", complex_signal_arr.imag)
print("complex signal array reshaped", complex_signal_arr.reshape(2, 3))
print("complex signal array numpy.ones_like", np.ones_like(complex_signal_arr))
print("complex signal array numpy.zeros_like", np.zeros_like(complex_signal_arr))

# numpy linspace
print("\nnp.linspace with Quantity:")
print("complex signal array value", complex_signal_arr.value)
#
duration = 5 * unit.s
timeStamp = np.linspace(0, duration, num=5)
print("timeStamp", timeStamp)
#
try:
    timeStamp = np.linspace(1.23, duration, num=5)
except Exception as exc:
    print(
        f"np.linspace(1.23, duration, num=5) failed because the start/stop must be a quantity (unless zero/infinity/nan)"
    )
    print(f"error message: {exc}")

print("\nnp.arange with Quantity:")
try:
    timeStamp = np.arange(0, duration + 1e-9 * unit.s, step=1 * unit.s)
except Exception as exc:
    print(f"np.arange does not work with Quantity")
    print(f"error message: {exc}")

# np.where
print("\nnp.where:")
nu_a = 1.0e6 * unit.Hz
nu = np.linspace(0.9, 1.1, 10) * unit.MHz
positive_indices = np.where(nu > nu_a)[0]
print("\npositive_indices:", positive_indices)

# np.isclose works with Quantity
# if a Quantity is close to a scalar, it will be treated as close if the value is close and the unit is dimensionless (or has the same dimension as the scalar)
print("\nnp.isclose:")
a = 1.0 * ppm
b = 1.0e-6 * unit.one
c = 1.0e-6
print("a:", a)
print("b:", b)
print("c:", c)
print("np.isclose(a, b):", np.isclose(a, b))
print("np.isclose(a, c):", np.isclose(a, c))
print("np.isclose(b, c):", np.isclose(b, c))

dur_0 = 1.0 * unit.microsecond
dur_1 = 1.0e-6 * unit.s
print("\nduration 0:", dur_0)
print("duration 1:", dur_1)
print("np.isclose(dur_0, dur_1):", np.isclose(dur_0, dur_1))

# np.tanh with Quantity
print("\nnumpy sinh cosh tanh with Quantity:")
beta = 0.5 * np.pi * unit.one * unit.radian
beta_float = 0.5 * np.pi
# print("np.tanh(beta)", np.tanh(beta))
# print("np.tanh(beta_float)", np.tanh(beta_float))
assert np.isclose(
    np.sinh(beta).value, np.sinh(beta_float)
), "np.sinh with Quantity should give the same result as np.sinh with float"
assert np.isclose(
    np.cosh(beta).value, np.cosh(beta_float)
), "np.cosh with Quantity should give the same result as np.cosh with float"

assert np.isclose(
    np.tanh(beta).value, np.tanh(beta_float)
), "np.tanh with Quantity should give the same result as np.tanh with float"
print("np.sinh / cosh / tanh accepts Quantity[unit.radian]")

# histogram
print("\nnp.histogram:")
B_space = np.random.random(100) * unit.T
hist, bin_edges = np.histogram(B_space)
print("histogram counts:", hist)
print("histogram bin edges:", bin_edges)

# ceil
print("\nnp.ceil:")
rate = 1.01 * unit.Hz
duration = 1.01 * unit.s
print("np.ceil(rate * duration).to(unit.one) =", np.ceil(rate * duration).to(unit.one))
