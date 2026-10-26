"""Sweep 1 MHz radial resolution and domain size for comparison.

Run from the repository root:
    python -m examples.EarthBoundAxionHalo.DM_overdensity.optimize_1MHz_params

Results are written under outputs/convergence_1MHz. No plots are made during
the numerical runs; use plot_1MHz_optimization.py afterward.
"""

import csv
import warnings
from pathlib import Path
from time import perf_counter

from astropy import units as unit

from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo


frequency = 1 * unit.MHz
shell_mass_limit = 4e-9 * unit.M_earth
shell_radii = [12300 * unit.km, 384000 * unit.km]
states = ["1s", "2s", "2p", "3s", "3p", "3d", "4s", "4p", "4d", "5s", "5p", "5d"]

# Resolution sweep: vary N with the preset 100 R_earth domain held fixed.
# This extended sequence is intended to show where successive changes plateau.
# Measured solve + inference runtimes by N: 50k=1.67, 75k=2.28, 100k=3.20,
# 150k=4.57, 200k=6.56, 300k=9.19, 400k=11.89, 600k=17.37, 800k=23.56,
# 1.2M=36.10, 1.6M=44.26 s (machine-dependent; plots excluded).
resolution_cases = [("resolution", n, 100 * unit.R_earth)
                    for n in [50_000, 75_000, 100_000, 150_000, 200_000,
                              300_000, 400_000, 600_000, 800_000,
                              1_200_000, 1_600_000]]
# Domain sweep: vary the outer boundary while keeping N/extent close to the
# preset sampling density. Every extent includes the 384,000 km shell edge.
# Measured solve + inference runtimes: 70R=2.25, 85R=2.98, 100R=3.19,
# 125R=3.83, 150R=4.36 s (machine-dependent; plots excluded).
domain_cases = [("domain", round(extent * 1_000), extent * unit.R_earth)
                for extent in [70, 85, 100, 125, 150]]
cases = resolution_cases + domain_cases

output_dir = Path(__file__).resolve().parent / "outputs" / "convergence_1MHz"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "results_1MHz.csv"
columns = ["frequency_Hz", "check", "N", "extent_Rearth", "dr_Rearth",
           "runtime_s", "state", "energy_eV", "shell_fraction",
           "total_mass_Mearth", "moon_enclosed_mass_Mearth", "earth_density_ratio"]

with output_path.open("w", newline="", encoding="utf-8") as output:
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for check, count, extent in cases:
        start = perf_counter()
        halo = EarthBoundAxionHalo(nu_a=frequency, N=count, extent=extent)
        halo.solve_TISE_3D(l_vals=[0, 1, 2], max_n_r=10)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*shell_fraction=.*at or below float64 machine epsilon.*"
            )
            masses = halo.inferHaloMass(shell_mass_limit, shell_radii, states)
        densities = halo.getDensityAtEarthSurface(state_names=states)
        runtime = perf_counter() - start
        for state in states:
            writer.writerow({
                "frequency_Hz": frequency.to_value(unit.Hz),
                "check": check,
                "N": count,
                "extent_Rearth": extent.to_value(unit.R_earth),
                "dr_Rearth": halo.dr.to_value(unit.R_earth),
                "runtime_s": runtime,
                "state": state,
                "energy_eV": halo.states[state]["eigenE"].to_value(unit.eV),
                "shell_fraction": masses[state]["shell_fraction"].to_value(unit.one),
                "total_mass_Mearth": masses[state]["total_mass"].to_value(unit.M_earth),
                "moon_enclosed_mass_Mearth": masses[state]["enclosed_mass"].to_value(unit.M_earth),
                "earth_density_ratio": densities[state]["density_ratio"].to_value(unit.one),
            })
        output.flush()
        print(
            f"{check}: N={count:,}, extent={extent.to_value(unit.R_earth):g} R_earth, "
            f"dr={halo.dr.to_value(unit.R_earth):.4g} R_earth, {runtime:.2f} s",
            flush=True,
        )
        del halo

print(f"Wrote {output_path}")
