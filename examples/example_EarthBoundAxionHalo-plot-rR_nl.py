import os
import time
from axionbloch.dependency import *
from axionbloch.EarthBoundAxionHalo import (
    EarthBoundAxionHalo,
)
from axionbloch.Station import Mainz, Baltimore, Sanya

station = Baltimore
rhoE_DM = 0.3 * unit.GeV / unit.cm**3

halo = EarthBoundAxionHalo(
    nu_a=1.348 * unit.MHz,  # axion Compton frequency in Hz
    N=int(2**12),  # number of grid points
    extent=128.0 * unit.R_earth,  # spatial extent of the grid in units of earth radius
    verbose=True,
)

halo.showValueAndUnits()

# tic = time.time()
halo.solve_TISE_3D(
    l_vals=[0, 1, 2, 3, 4],  # angular momentum quantum number
    max_n_r=64,  # maximum principal quantum number to plot
    verbose=False,
)
# toc = time.time()
# print(f"Time taken to solve TISE: {toc - tic:.2e} seconds")

# print(halo.getStateNames())
# print(halo.getStateEnergies())
halo.plotEigenStates(savefig=True)
