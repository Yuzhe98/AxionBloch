"""Example: Compare gradient components across different eigenstate combinations.

This script demonstrates how to use findGradientsOverStates() to compute and
visualize gradient components for multiple state combinations at a given station.
"""

from axionbloch.dependency import *
from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo
from axionbloch.Station import Mainz

# Configuration
station = Mainz
meas_time = Time("2022-12-14T12:00:00")  # measurement time

# Initialize the Earth-bound axion halo
halo = EarthBoundAxionHalo(
    nu_a=1.348 * unit.MHz,  # axion Compton frequency in Hz
    N=int(2**12),  # number of grid points
    extent=128.0 * unit.R_earth,  # spatial extent of the grid
    verbose=True,
)

# Solve the Schrödinger equation for multiple angular-momentum channels
halo.solve_TISE_3D(
    l_vals=[1],  # l = 0 (s), 1 (p), 2 (d)
    max_n_r=64,  # number of radial eigenstates per l
    verbose=False,
)

# Define state combinations to compare
# Key: label for the plot, Value: list of eigenstate names
state_combinations = {
    "2p": ["2p"],
    "2p + 3p": ["2p", "3p"],
    # "2p + 3p + 3d": ["2p", "3p", "3d"],
    "2p + 3p + 4p": ["2p", "3p", "4p"],
}

# Compute and plot gradients for all state combinations
results = halo.findGradientsOverStates(
    station=station,
    meas_time=meas_time,
    stateNamesDict=state_combinations,
    truncRadius=2 * unit.earthRad,
    showPlot=True,
    verbose=True,
)

# Extract and print gradient values at Earth's surface
print("\n" + "=" * 70)
print("Gradient values at Earth's surface (r = 1 R_earth):")
print("=" * 70)

r_line = results["r_line"]
earthRad_idx = np.argmin(np.abs(r_line - 1 * unit.earthRad))

for label in state_combinations.keys():
    grad_r = results["grad_r"][label][earthRad_idx]
    grad_theta = results["grad_theta"][label][earthRad_idx]
    grad_phi = results["grad_phi"][label][earthRad_idx]

    print(f"\n{label}:")
    print(f"  ∂_r φ     = {grad_r}")
    print(f"  ∂_θ φ / r = {grad_theta}")
    print(f"  ∂_φ φ / r sinθ = {grad_phi}")
    # print(f"  |∇φ| = {np.sqrt(np.abs(grad_r)**2 + np.abs(grad_theta)**2 + np.abs(grad_phi)**2)}")

print("\n" + "=" * 70)
print("Plot window shows overlaid gradient components for all state combinations.")
print("=" * 70)
