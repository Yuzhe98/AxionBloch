import time
from axionbloch.enphylope import PhysicalQuantity as PQ
from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo, loadPEMdata
from axionbloch.Station import Mainz, Baltimore, Sanya
# data = loadPEMdata()
# print(data["radius_m"])

halo = EarthBoundAxionHalo(
    nu_a=PQ(1.348, "MHz"),  # axion Compton frequency in Hz
    N=int(2**12),  # number of grid points
    extent=PQ(128.0, "earth_radius"),  # spatial extent of the grid in units of earth radius
    verbose=True,
)

tic = time.time()
halo.solve_TISE_3D(l_vals=[1],  # angular momentum quantum number
        max_n_r = 64,  # maximum principal quantum number to plot
        )
toc = time.time()
print(f"Time taken to solve TISE: {toc - tic:.2e} seconds")

print(halo.getStateNames())

# halo.findGradients(stateNames=['2p'], station=Mainz)

halo.findGradients(stateNames=["2p"], station=Baltimore)

halo.findGradients(stateNames=["2p"], station=Sanya)
