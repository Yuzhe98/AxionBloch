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
print("\n# ------------- numpy operations ------------- #")
# sine of an array of angles
tStamps = np.array([0, 1 / 3, 1]) * t90
phases = gamma_p * B * tStamps
print("phases at different time stamps:", np.sin(phases))
# [0.  0.5 1. ]

# absolute value
speed = -220.0 * unit.km / unit.s
print("speed:", speed)
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
    print(
        f"error message: {exc}"
    )
