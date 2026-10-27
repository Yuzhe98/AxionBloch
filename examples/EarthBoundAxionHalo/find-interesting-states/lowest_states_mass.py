"""List the 30 lowest Earth-bound axion states and inferred halo masses.

Run from the repository root with:
    python examples/EarthBoundAxionHalo/find-interesting-states/lowest_states_mass.py

Each inferred mass treats the corresponding eigenstate as the entire halo.
"""

from astropy import constants as const
from astropy import units as unit

from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo


# Match the 1 MHz grid preset from DM_overdensity/density_by_shell_mass_sweep.py.
param_1MHz = {
    "nu_a": 1 * unit.MHz,
    "N": int(7.5e4),
    "extent": 1e2 * unit.R_earth,
}

# Frequency requested for this run. Keep the 1 MHz preset's grid unchanged.
param_1MHz["nu_a"] = 1.348575 * unit.MHz

# Match the shell constraint used by density_by_shell_mass_sweep.py.
shell_mass_limit = 4e-9 * unit.M_earth
shell_radii = [12300 * unit.km, 384000 * unit.km]

halo = EarthBoundAxionHalo(**param_1MHz)  # maxTotalDMmass defaults to 1 M_earth.
halo.solve_TISE_3D(l_vals=[0, 1, 2, 3, 4], max_n_r=30)

# solve_TISE_3D sorts all solved states by eigen-energy. Select the first 30,
# then infer each as an independent, single-state halo using inferHaloMass's
# default maxEarthEnclosedMass constraint.
state_names = halo.getStateNames()[:30]
mass_results = halo.inferHaloMass(shell_mass_limit, shell_radii, state_names)

results = [
    (
        name,
        (halo.states[name]["eigenE"] / const.h).to(unit.mHz),
        mass_results[name]["total_mass"].to(unit.M_earth),
    )
    for name in state_names
]

print("State   Eigenenergy shift E/h (mHz)   Total halo mass (M_earth)")
for name, eigen_energy, total_mass in results:
    print(
        f"{name:<5}   {eigen_energy.to_value(unit.mHz):.4g}"
        f"   {total_mass.to_value(unit.M_earth):.4e}"
    )
