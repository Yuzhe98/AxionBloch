"""Infer independent single-state halo densities for each frequency preset.

Run from the repository root:
    python -m examples.EarthBoundAxionHalo.DM_overdensity.density_by_shell_mass_sweep

Figures and text results are saved beside this script under outputs/frequency_sweep.
The combined density_at_earth_vs_frequency.txt table contains one row per state
and frequency, with densities and absolute uncertainties in units of rho_M_DM_SHM.
The largest presets require substantial memory; edit params to run a subset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as unit

from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo

# Physical inputs and active presets from density_by_shell_mass.py.
shell_mass = 0.3e-9 * unit.M_earth

mass_uncertainty = 4e-9 * unit.M_earth

shell_radii = [12300 * unit.km, 384000 * unit.km]

states = ["1s", "2s", "2p", "3s", "3p", "3d", "4s", "4p", "4d", "5s", "5p", "5d"]

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
    "N": int(1e6),
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

params = [
    param_1Hz, param_10Hz, param_100Hz, param_1kHz, param_10kHz,
    param_100kHz, param_1MHz, param_10MHz, param_30MHz,
]
output_dir = Path(__file__).resolve().parent / "outputs" / "frequency_sweep"
output_dir.mkdir(parents=True, exist_ok=True)

# Start a fresh combined table; append each frequency as soon as it is computed.
density_table_path = output_dir / "density_at_earth_vs_frequency.txt"
density_table_path.write_text(
    "# Density at r = 1 R_earth; each state is an independent hypothesis.\n"
    "# Density and absolute uncertainty are in units of rho_M_DM_SHM.\n"
    "# axion_frequency_Hz\tstate\tdensity_over_rho_M_DM_SHM\tuncertainty_over_rho_M_DM_SHM\n",
    encoding="utf-8",
)

for param in params:
    # Solve once per frequency; each state is a separate mass hypothesis.
    frequency_label = f"{param['nu_a'].to_value(unit.Hz):g}_Hz"
    print(f"\nFrequency: {param['nu_a']}", flush=True)
    halo = EarthBoundAxionHalo(**param)
    halo.solve_TISE_3D(l_vals=[0, 1, 2], max_n_r=10)

    # Save wavefunctions before performing the shell-mass inference.
    fig_u_r, ax_u_r = halo.plot_u_r(state_names=states, showPlot=False)
    fig_u_r.set_size_inches(13 / 2.54, 8 / 2.54)
    fig_u_r.tight_layout()
    fig_u_r.savefig(output_dir / f"u_r_{frequency_label}.png", dpi=300)
    plt.close(fig_u_r)

    masses = halo.inferHaloMass(shell_mass, mass_uncertainty, shell_radii, states)
    density_at_earth_radius = halo.getDensityAtEarthSurface(state_names=states)

    # Convert quantities to numbers only for the tab-separated text output.
    with density_table_path.open("a", encoding="utf-8") as density_table:
        for state, earth in density_at_earth_radius.items():
            density_table.write(
                f"{param['nu_a'].to_value(unit.Hz):.16e}\t{state}\t"
                f"{earth['density_ratio'].to_value(unit.one):.16e}\t"
                f"{earth['uncertainty_ratio'].to_value(unit.one):.16e}\n"
            )

    # Keep readable results with units and quoted absolute uncertainties.
    lines = [f"Frequency: {param['nu_a']}",
             f"N: {param['N']}; extent: {param['extent']}",
             f"Shell radii: {shell_radii}",
             f"Shell mass: {shell_mass} +/- {mass_uncertainty}"]
    for state, result in masses.items():
        earth = density_at_earth_radius[state]
        lines.extend([
            f"{state}: total = {result['total_mass']:.6g} +/- {result['total_mass_uncertainty']:.6g}",
            f"    inside Moon = {result['enclosed_mass']:.6g} +/- {result['enclosed_mass_uncertainty']:.6g}",
            f"    shell_fraction = {result['shell_fraction']:.6g}",
            f"    density at Earth radius = ({earth['density_ratio']:.6g} +/- {earth['uncertainty_ratio']:.6g}) * rho_M_DM_SHM",
        ])
    report = "\n".join(lines)
    print(report, flush=True)
    (output_dir / f"results_{frequency_label}.txt").write_text(report + "\n", encoding="utf-8")

    # Plot the full radial extent, independently of the measured shell boundaries.
    radii = np.geomspace(.01 * unit.R_earth, halo.r[-1], 1000)
    fig, axes, profiles = halo.plotDMdensity(
        radii, state_names=states, showPlot=False, scales=("log", "log")
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"density_{frequency_label}.png", dpi=300)
    plt.close(fig)

    # Release the large eigenstate arrays before constructing the next halo.
    del halo
