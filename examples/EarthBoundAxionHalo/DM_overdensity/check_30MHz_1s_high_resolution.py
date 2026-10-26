"""Check 30 MHz 1s convergence at fixed 70 Earth radii.

Run from the repository root:
    python -m examples.EarthBoundAxionHalo.DM_overdensity.check_30MHz_1s_high_resolution

Computes only the lowest s state at each grid size to target the unresolved
case without spending time on unrelated eigenstates or plots.
"""

import csv
import warnings
from pathlib import Path
from time import perf_counter

from astropy import units as unit

from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo


frequency = 30 * unit.MHz
extent = 70 * unit.R_earth
shell_mass_limit = 4e-9 * unit.M_earth
shell_radii = [12300 * unit.km, 384000 * unit.km]
# Measured runtimes for l=0, max_n_r=1 plus inference: 1.65, 3.52, 7.28,
# 13.89, and 27.99 s for the counts below (machine-dependent; no plots).
counts = [1_000_000, 2_000_000, 4_000_000, 8_000_000, 16_000_000]
output_path = Path(__file__).resolve().parent / "convergence_30MHz_1s_high_resolution.csv"
fields = ["frequency_Hz", "state", "N", "extent_Rearth", "runtime_s",
          "energy_eV", "shell_fraction", "moon_enclosed_mass_Mearth",
          "earth_density_ratio"]

existing_counts = set()
if output_path.exists() and output_path.stat().st_size:
    with output_path.open(encoding="utf-8", newline="") as prior:
        existing_counts = {int(row["N"]) for row in csv.DictReader(prior)}

with output_path.open("a", newline="", encoding="utf-8") as output:
    writer = csv.DictWriter(output, fieldnames=fields)
    if not existing_counts:
        writer.writeheader()
    for count in counts:
        if count in existing_counts:
            continue
        start = perf_counter()
        halo = EarthBoundAxionHalo(nu_a=frequency, N=count, extent=extent)
        halo.solve_TISE_3D(l_vals=[0], max_n_r=1)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*shell_fraction=.*at or below float64 machine epsilon.*"
            )
            mass = halo.inferHaloMass(shell_mass_limit, shell_radii, ["1s"])["1s"]
        density = halo.getDensityAtEarthSurface(["1s"])["1s"]["density_ratio"]
        runtime = perf_counter() - start
        writer.writerow({
            "frequency_Hz": frequency.to_value(unit.Hz),
            "state": "1s",
            "N": count,
            "extent_Rearth": extent.to_value(unit.R_earth),
            "runtime_s": runtime,
            "energy_eV": halo.states["1s"]["eigenE"].to_value(unit.eV),
            "shell_fraction": mass["shell_fraction"].to_value(unit.one),
            "moon_enclosed_mass_Mearth": mass["enclosed_mass"].to_value(unit.M_earth),
            "earth_density_ratio": density.to_value(unit.one),
        })
        output.flush()
        print(
            f"N={count:,}, extent=70 R_earth, runtime={runtime:.2f} s, "
            f"density ratio={density.to_value(unit.one):.8e}",
            flush=True,
        )
        del halo

print(f"Wrote {output_path}")
