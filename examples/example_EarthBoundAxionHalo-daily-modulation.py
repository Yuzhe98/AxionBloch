# plot the daily modulation of the axion gradient at various stations on Earth, for an Earth-bound axion halo with a given Compton frequency and axion-nucleon coupling. The modulation is due to the rotation of the Earth, which changes the direction of the station relative to the axion halo. The code solves the time-independent Schrödinger equation for the axion halo and then computes the gradient at each station as a function of time over one day. Finally, it plots the results.
# TODO: to be completed
import os

from axionbloch.dependency import *
from axionbloch.EarthBoundAxionHalo import (
    EarthBoundAxionHalo,
)
from axionbloch.utils import check
from axionbloch.Station import (
    Mainz,
    Baltimore,
    Sanya,
    Tokyo,
    Geneva,
    Sydney,
    BuenosAires,
    CapeTown,
    Mumbai,
)

stations = [
    Mainz,
    Geneva,
    Baltimore,
    Tokyo,
    Mumbai,
    Sanya,
    Sydney,
    CapeTown,
    BuenosAires,
]
# stations = [
#     CapeTown,
#     Mumbai,
# ]
# station = Baltimore
rhoE_DM = 0.3 * unit.GeV / unit.cm**3

states_to_check = ["1s", "2s", "2p", "3s", "3p", "3d", "4s", "4p", "4d"]
# states_to_check = ["2p", "3p"]

halo = EarthBoundAxionHalo(
    nu_a=1.348 * unit.MHz,  # axion Compton frequency in Hz
    N=int(2**12),  # number of grid points
    extent=128.0 * unit.R_earth,  # spatial extent of the grid in units of earth radius
    g_aNN=1e-9 * unit.GeV**-1,  # axion-nucleon coupling
    verbose=True,
)

halo.showValueAndUnits()


halo.solve_TISE_3D(
    l_vals=[0, 1, 2],  # angular momentum quantum number
    max_n_r=64,  # maximum principal quantum number to plot
    verbose=False,
)

