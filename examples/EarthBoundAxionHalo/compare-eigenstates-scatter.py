"""Example: Compare gradient components across eigenstate combinations via scatter plot.

This script demonstrates how to use compareGradientsOverStates() to extract and
visualize gradient values at the station's location for different state combinations.
"""

from axionbloch.dependency import *
from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo
from axionbloch.Station import Mainz

# Configuration
station = Mainz
meas_time = Time("2022-12-14T12:00:00")

# Initialize the Earth-bound axion halo
halo = EarthBoundAxionHalo(
    nu_a=1.348 * unit.MHz,
    N=int(2**12),
    extent=128.0 * unit.R_earth,
    verbose=True,
)

# Solve the Schrödinger equation for multiple angular-momentum channels
halo.solve_TISE_3D(
    l_vals=[1],
    max_n_r=64,
    verbose=False,
)

# Define state combinations to compare
state_combinations = {
    "2p": {"2p": 1.0},
    "2p + 3p": {"2p": 1.0, "3p": 1.0},
    "2p + 3p + 4p": {"2p": 1.0, "3p": 1.0, "4p": 1.0},
}

# Compare gradients across state combinations at station location
comparison = halo.compareGradientsOverStates(
    station=station,
    meas_time=meas_time,
    stateNamesDict=state_combinations,
    truncRadius=2 * unit.earthRad,
    showPlot=True,
    verbose=True,
)

# Print gradient values at specified radius
print("\n" + "=" * 70)
print(f"Gradient values at r = {comparison['r_eval']}:")
print("=" * 70)

for i, label in enumerate(comparison["state_labels"]):
    print(f"\n{label}:")
    print(f"  ∂_r φ     = {comparison['grad_r'][i]:.4e}")
    print(f"  ∂_θ φ / r = {comparison['grad_theta'][i]:.4e}")
    print(f"  ∂_φ φ / r sinθ = {comparison['grad_phi'][i]:.4e}")

print("\n" + "=" * 70)
print("Scatter plot shows gradient magnitudes for all state combinations.")
print("=" * 70)
