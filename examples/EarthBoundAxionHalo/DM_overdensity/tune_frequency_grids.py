"""Targeted resolution sweeps for 10 and 30 MHz grid presets.

Run from the repository root. The broader 100 kHz and 1 MHz sweeps are
already saved; this script checks denser grids at 10 and 30 MHz.
"""

import csv
import warnings
from pathlib import Path
from time import perf_counter

from astropy import units as unit

from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo


shell_mass_limit = 4e-9 * unit.M_earth
shell_radii = [12300 * unit.km, 384000 * unit.km]
states = ["1s", "2s", "2p", "3s", "3p", "3d", "4s", "4p", "4d", "5s", "5p", "5d"]
cases = [
    (10 * unit.MHz, 400_000, 100 * unit.R_earth),
    (10 * unit.MHz, 800_000, 100 * unit.R_earth),
    (10 * unit.MHz, 1_600_000, 100 * unit.R_earth),
    (10 * unit.MHz, 3_200_000, 100 * unit.R_earth),
    (30 * unit.MHz, 4_000_000, 70 * unit.R_earth),
    (30 * unit.MHz, 8_000_000, 70 * unit.R_earth),
    (30 * unit.MHz, 16_000_000, 70 * unit.R_earth),
]
output_dir = Path(__file__).resolve().parent / "outputs" / "convergence"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "targeted_frequency_grid_checks.csv"
fields = ["frequency_Hz", "N", "extent_Rearth", "dr_Rearth", "runtime_s", "state",
          "energy_eV", "shell_fraction", "total_mass_Mearth",
          "moon_enclosed_mass_Mearth", "earth_density_ratio"]

with output_path.open("w", newline="", encoding="utf-8") as output:
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for frequency, count, extent in cases:
        start = perf_counter()
        halo = EarthBoundAxionHalo(nu_a=frequency, N=count, extent=extent)
        halo.solve_TISE_3D(l_vals=[0, 1, 2], max_n_r=10)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*shell_fraction=.*at or below float64 machine epsilon.*"
            )
            masses = halo.inferHaloMass(shell_mass_limit, shell_radii, states)
        earth_density = halo.getDensityAtEarthSurface(state_names=states)
        runtime = perf_counter() - start
        for state in states:
            writer.writerow({
                "frequency_Hz": frequency.to_value(unit.Hz),
                "N": count,
                "extent_Rearth": extent.to_value(unit.R_earth),
                "dr_Rearth": halo.dr.to_value(unit.R_earth),
                "runtime_s": runtime,
                "state": state,
                "energy_eV": halo.states[state]["eigenE"].to_value(unit.eV),
                "shell_fraction": masses[state]["shell_fraction"].to_value(unit.one),
                "total_mass_Mearth": masses[state]["total_mass"].to_value(unit.M_earth),
                "moon_enclosed_mass_Mearth": masses[state]["enclosed_mass"].to_value(unit.M_earth),
                "earth_density_ratio": earth_density[state]["density_ratio"].to_value(unit.one),
            })
        output.flush()
        print(f"{frequency:g}: N={count:,}, extent={extent.to_value(unit.R_earth):g} R_earth, "
              f"runtime={runtime:.2f} s", flush=True)
        del halo

print(f"Wrote {output_path}")
