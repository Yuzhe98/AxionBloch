import time
from astropy import units as unit
from axionbloch.EarthBoundAxionHalo import (
    EarthBoundAxionHalo,
)
from axionbloch.Station import Mainz, Baltimore

halo = EarthBoundAxionHalo(
    nu_a=1.348 * unit.MHz,  # axion Compton frequency in Hz
    N=int(2**12),  # number of grid points
    extent=128.0 * unit.R_earth,  # spatial extent of the grid in units of earth radius
    verbose=True,
)

halo.showValueAndUnits()

# tic = time.time()
halo.solve_TISE_3D(l_vals=[1],  # angular momentum quantum number
        max_n_r = 64,  # maximum principal quantum number to plot
        verbose=False,
        )
# toc = time.time()
# print(f"Time taken to solve TISE: {toc - tic:.2e} seconds")

# print(halo.getStateNames())
# print(halo.getStateEnergies())

halo.findGradients(
    stateNames=["2p"], station=Baltimore, truncRadius=2 * unit.earthRad, verbose=True
)
