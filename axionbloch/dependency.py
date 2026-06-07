# ################################################################### #
# This file is meant to centralize imports of external dependencies
# (e.g. numpy, matplotlib, astropy) so that they can be easily managed
# and updated in one place.
# This also helps to avoid circular imports and makes it clear
# which libraries are being used throughout the project.
# ################################################################### #

# numerical computing
import numpy as np

# ----------- physical units and constants ----------- #
import astropy
from astropy import units as unit
from astropy.units import Quantity, CompositeUnit
from astropy.constants import codata2018 as const
from astropy.time import Time

# pi with unit.radian
PI = np.pi * unit.rad

# dimensionless scale units
# parts per million
ppm = unit.def_unit("ppm", 1e-6 * unit.one)
# parts per billion
ppb = unit.def_unit("ppb", 1e-9 * unit.one)
# parts per trillion
ppt = unit.def_unit("ppt", 1e-12 * unit.one)
# parts per quadrillion
ppq = unit.def_unit("ppq", 1e-15 * unit.one)
# parts per quintillion
ppqu = unit.def_unit("ppqu", 1e-18 * unit.one)
# parts per sextillion
ppmu = unit.def_unit("ppmu", 1e-21 * unit.one)
# parts per septillion
ppbmu = unit.def_unit("ppbmu", 1e-24 * unit.one)

# Magnetic flux quantum
Phi0 = magnetic_flux_quantum = unit.def_unit(
    "magnetic_flux_quantum", 1.0 * const.h / (2 * const.e)
)
# ----------- ---------------------------- ----------- #

# --------------- plotting --------------- #
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # for creating subplots
import matplotlib.ticker as mticker
from matplotlib.axes import Axes
import textwrap

# plot style
plt.rc("font", size=6)  # font size for all figures
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# Make math text match Times New Roman
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "Times New Roman"

plt.rcParams["lines.linewidth"] = 1.0  # thickness of lines
plt.rcParams["lines.markersize"] = 0.5  # size of markers

plt.rcParams["figure.dpi"] = 300  # resolution on screen
plt.rcParams["savefig.dpi"] = 300  # resolution when saving
# --------------- -------- --------------- #
