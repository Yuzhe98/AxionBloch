"""Plot eigenstate amplitudes against their eigen-energy shifts.

All states from 1s through 4f are included with equal input coefficients.
``plotStateAmplitudeVsEigenEnergy`` normalizes the coefficients before plotting,
so every displayed amplitude is 1 / sqrt(number of states).
"""

from pathlib import Path

from axionbloch.dependency import *
from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo


halo = EarthBoundAxionHalo(
    nu_a=1.348585 * unit.MHz,
    N=2**12,
    extent=128 * unit.R_earth,
    verbose=True,
)

# l = 0, 1, 2, 3 correspond to s, p, d, f. Four radial states per l
# ensure that every state through principal quantum number n = 4 is solved.
halo.solve_TISE_3D(
    l_vals=[0, 1, 2, 3],
    # l_vals=[0],
    max_n_r=4,
    verbose=False,
)

# Equal coefficients provide an equal coherent superposition. The plotting
# method normalizes this dictionary, giving |c_nlm| = 1 / sqrt(10).
stateCoefficients = {
    "1s": 1,
    "2s": 1,
    "2p": 1,
    "3s": 1,
    "3p": 1,
    "3d": 1,
    "4s": 1,
    "4p": 1,
    "4d": 1,
    "4f": 1,
}

fig, _, _, _ = halo.plotStateAmplitudeVsEigenEnergy(
    stateCoefficients=stateCoefficients,
    energy_unit=unit.attoelectronvolt,
    frequency_unit=unit.mHz,
    showPlot=False,
)

output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "EarthHalo-state-amplitude-vs-eigenenergy-1s-to-4f.pdf"
fig.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Saved figure to {output_path}")
