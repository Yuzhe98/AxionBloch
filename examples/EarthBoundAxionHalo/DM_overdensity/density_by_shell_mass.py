"""Single-state halo masses and mean densities inferred from a measured shell.

Run from the repository root:
    python -m examples.EarthBoundAxionHalo.DM_overdensity.density_by_shell_mass
"""

import numpy as np
from astropy import constants as const, units as unit
from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo

# Edit these physical inputs.
# axion_mass = 1e-9 * unit.eV / const.c**2
shell_mass = 0.3e-9 * unit.M_earth
mass_uncertainty = 4e-9 * unit.M_earth
shell_radii = [12300 * unit.km, 384000 * unit.km]
states = ["1s", "2s", "2p", "3s", "3p", "3d", "4s", "4p", "4d", "5s", "5p", "5d"]

# at 1 MHz
# the lowest 10 states are: 1s, 2s, 3s, 2p, 4s, 3d, 5s, 3p, 6s, 4f
# 1s: E = -4.549e-18 eV
# 2s: E = -3.445e-18 eV
# 3s: E = -2.540e-18 eV
# 2p: E = -2.478e-18 eV
# 4s: E = -1.861e-18 eV
# 3d: E = -1.680e-18 eV
# 5s: E = -1.398e-18 eV
# 3p: E = -1.347e-18 eV
# 6s: E = -1.090e-18 eV
# 4f: E = -1.088e-18 eV

# mass_factor = 1e-2
# nu_a = mass_factor * 1.0 * unit.MHz
# N=max(1024, int(3e4))
# extent = max(100.0 * unit.R_earth, 3e2 / mass_factor * unit.R_earth)
param_1Hz = {
    "nu_a": 1 * unit.Hz,
    "N": int(4e7),
    "extent": 4e6 * unit.R_earth,
}

param_10Hz = {
    "nu_a": 10 * unit.Hz,
    "N": int(1e7),
    "extent": 1e6 * unit.R_earth,
}

param_100Hz = {
    "nu_a": 100 * unit.Hz,
    "N": int(1e6),
    "extent": 1e6 * unit.R_earth,
}

param_1kHz = {
    "nu_a": 1 * unit.kHz,
    "N": int(1e6),
    "extent": 1e5 * unit.R_earth,
}

param_10kHz = {
    "nu_a": 10 * unit.kHz,
    "N": int(5e6),
    "extent": 1e4 * unit.R_earth,
}

param_100kHz = {
    "nu_a": 100 * unit.kHz,
    "N": int(1e5),
    "extent": 1e3 * unit.R_earth,
}

param_1MHz = {
    "nu_a": 1 * unit.MHz,
    "N": int(1e5),
    "extent": 1e2 * unit.R_earth,
}

param_10MHz = {
    "nu_a": 10 * unit.MHz,
    "N": int(1e5),
    "extent": 1e2 * unit.R_earth,
}

param_30MHz = {
    "nu_a": 30 * unit.MHz,
    "N": int(1e6),
    "extent": 70 * unit.R_earth,
}

# param_100MHz = {
#     "nu_a": 100 * unit.MHz,
#     "N": int(5e6),
#     "extent": 70 * unit.R_earth,
# }
# param_1GHz = {
#     "nu_a": 1 * unit.GHz,
#     "N": int(1e6),
#     "extent": 70 * unit.R_earth,
# }

# params = [param_1Hz, param_10Hz, param_100Hz, param_1kHz, param_10kHz, param_100kHz, param_1MHz, param_10MHz, param_100MHz]

halo = EarthBoundAxionHalo(
    # m_a=axion_mass,
    **param_10kHz
)
halo.solve_TISE_3D(l_vals=[0, 1, 2], max_n_r=10)
fig_u_r, ax_u_r = halo.plot_u_r(state_names=states)

masses = halo.inferHaloMass(shell_mass, mass_uncertainty, shell_radii, states)

for state, result in masses.items():
    print(f"{state}: total = {result['total_mass']:.2g} +/- {result['total_mass_uncertainty']:.2g}")
    print(f"    inside Moon = {result['enclosed_mass']:.2g} +/- {result['enclosed_mass_uncertainty']:.2g}")
    print(f"    shell_fraction = {result['shell_fraction']:.2g}")

# Earth-surface density for the selected single-state hypotheses.
density_at_earth_radius = halo.getDensityAtEarthSurface(state_names=states)
for state, result in density_at_earth_radius.items():
    print(f"{state}: density at Earth radius = ({result['density_ratio']:.2g} +/- {result['uncertainty_ratio']:.2g}) * rho_M_DM_SHM")

# Plot through the last solver grid point; the measured shell only sets the mass scale.
radii = np.geomspace(.01 * unit.R_earth, halo.r[-1], 1000)
# fig, axes, profiles = halo.plotDMdensity(radii, state_names=states, scales=("linear", "linear"))
fig, axes, profiles = halo.plotDMdensity(radii, state_names=states, scales=("log", "log"))