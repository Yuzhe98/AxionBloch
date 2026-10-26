"""Check resolution and domain convergence from 100 Hz through 10 kHz.

Run from the repository root::

    python -m examples.EarthBoundAxionHalo.DM_overdensity.check_low_frequency_convergence

Results are appended after each completed run under ``outputs`` so a long
check can be resumed or inspected without losing completed cases.
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
presets = [
    (100 * unit.Hz, 1_000_000, 1_000_000 * unit.R_earth),
    (1 * unit.kHz, 1_000_000, 100_000 * unit.R_earth),
    (10 * unit.kHz, 1_000_000, 10_000 * unit.R_earth),
]
output_dir = Path(__file__).resolve().parent / "outputs" / "low_frequency_convergence"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "convergence_results.csv"
fields = [
    "frequency_Hz", "check", "scale", "N", "extent_Rearth", "dr_Rearth",
    "runtime_s", "state", "energy_eV", "shell_fraction",
    "total_mass_Mearth", "moon_enclosed_mass_Mearth", "earth_density_ratio",
]

write_header = not output_path.exists() or output_path.stat().st_size == 0
with output_path.open("a", newline="", encoding="utf-8") as output:
    writer = csv.DictWriter(output, fieldnames=fields)
    if write_header:
        writer.writeheader()
        output.flush()
    for frequency, base_n, base_extent in presets:
        runs = [
            ("resolution", scale, max(2, round(base_n * scale)), base_extent)
            for scale in (0.5, 1.0, 2.0)
        ]
        runs += [
            ("extent", scale, max(2, round(base_n * scale)), base_extent * scale)
            for scale in (0.5, 2.0)
        ]
        if frequency == 100 * unit.Hz:
            # Shell-fraction convergence was slower at 100 Hz, so extend only
            # that frequency before committing to a larger preset.
            runs += [
                ("resolution", 3.0, 3 * base_n, base_extent),
                ("resolution", 4.0, 4 * base_n, base_extent),
                ("resolution", 8.0, 8 * base_n, base_extent),
                ("extent", 4.0, 4 * base_n, 4 * base_extent),
            ]
        else:
            # Check whether the 1M preset can be reduced without crossing the
            # 0.01% reference-deviation target.
            candidate_scale = 0.375 if frequency == 1 * unit.kHz else 0.125
            runs.append(("resolution", candidate_scale,
                         max(2, round(base_n * candidate_scale)), base_extent))
        for check, scale, count, extent in runs:
            # Make reruns safe: skip a case already fully present in the CSV.
            already_done = False
            if output_path.exists() and output_path.stat().st_size:
                with output_path.open(encoding="utf-8", newline="") as saved:
                    saved_rows = list(csv.DictReader(saved))
                already_done = any(
                    float(row["frequency_Hz"]) == frequency.to_value(unit.Hz)
                    and row["check"] == check
                    and float(row["scale"]) == scale
                    and int(row["N"]) == count
                    and float(row["extent_Rearth"]) == extent.to_value(unit.R_earth)
                    for row in saved_rows
                )
            if already_done:
                print(f"Skipping saved {frequency:g} {check} scale={scale:g}", flush=True)
                continue

            start = perf_counter()
            halo = EarthBoundAxionHalo(nu_a=frequency, N=count, extent=extent)
            halo.solve_TISE_3D(l_vals=[0, 1, 2], max_n_r=10)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*shell_fraction=.*at or below float64 machine epsilon.*",
                )
                masses = halo.inferHaloMass(shell_mass_limit, shell_radii, states)
            earth_density = halo.getDensityAtEarthSurface(state_names=states)
            runtime = perf_counter() - start
            for state in states:
                writer.writerow({
                    "frequency_Hz": frequency.to_value(unit.Hz),
                    "check": check,
                    "scale": scale,
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
            print(
                f"{frequency:g} {check} scale={scale:g}: N={count:,}, "
                f"extent={extent.to_value(unit.R_earth):g} R_earth, "
                f"dr={halo.dr.to_value(unit.R_earth):g} R_earth, "
                f"runtime={runtime:.1f} s",
                flush=True,
            )
            del halo

print(f"Wrote {output_path}")
