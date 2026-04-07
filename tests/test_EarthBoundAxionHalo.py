import time
from axionbloch.enphylope import PhysicalQuantity as PQ
from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo, loadPEMdata

# data = loadPEMdata()
# print(data["radius_m"])

halo = EarthBoundAxionHalo(
    nu_a=PQ(1, "MHz"),  # axion Compton frequency in Hz
    N=int(2**12),  # number of grid points
    extent=PQ(128.0, "earth_radius"),  # spatial extent of the grid in units of earth radius
    verbose=True,
)

tic = time.time()
halo.solve_TISE_3D(l_vals=[0],  # angular momentum quantum number
        max_n_r = 64,  # maximum principal quantum number to plot
        )
toc = time.time()
print(f"Time taken to solve TISE: {toc - tic:.2e} seconds")
      
print(halo.getStateNames())

tic = time.time()
halo.findGradients(stateNames=['1s'])
toc = time.time()
print(f"Time taken to find gradients: {toc - tic:.2e} seconds")