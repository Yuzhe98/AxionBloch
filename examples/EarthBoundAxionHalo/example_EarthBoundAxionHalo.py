import os

from axionbloch.dependency import *
from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo
from axionbloch.Station import Baltimore

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
    l_vals=[1],  # angular momentum quantum number
    max_n_r=64,  # maximum principal quantum number to plot
    verbose=True,
)
# toc = time.time()
# print(f"Time taken to solve TISE: {toc - tic:.2e} seconds")

# print(halo.getStateNames())
# print(halo.getStateEnergies())

r, R_r, r_line, grad_r_line, grad_theta_line, grad_phi_line = (
    halo.findGradientsAtDirection(
        stateNames=["2p"],
        station=station,
        truncRadius=2 * unit.earthRad,
        verbose=True,
        showPlot=False,
    )
)
halo.plotGradients(
    stateNames=["2p"],
    station=station,
    r=r,
    R_r=R_r,
    r_line=r_line,
    grad_r_line=grad_r_line,
    grad_theta_line=grad_theta_line,
    grad_phi_line=grad_phi_line,
)

logPrefix = os.path.abspath(__file__)

earthRad_idx = np.argmin(np.abs(r_line - 1 * unit.earthRad))
print(logPrefix, "r_line index @ station =", earthRad_idx)
print(
    logPrefix,
    "a_0_reduced * grad_r @ station =",
    (halo.a_0_reduced * grad_r_line[earthRad_idx]).to(unit.eV / unit.m),
)
print(
    logPrefix,
    "a_0_reduced * grad_theta @ station =",
    (halo.a_0_reduced * grad_theta_line[earthRad_idx]).to(unit.eV / unit.m),
)
print(
    logPrefix,
    "a_0_reduced * grad_phi @ station =",
    (halo.a_0_reduced * grad_phi_line[earthRad_idx]).to(unit.eV / unit.m),
)
