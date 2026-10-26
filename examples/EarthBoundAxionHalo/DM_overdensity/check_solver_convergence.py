"""Compare shell-inferred halo quantities across radial-grid settings.

Run from the repository root:
    python -m examples.EarthBoundAxionHalo.DM_overdensity.check_solver_convergence

This script intentionally creates no plots. Edit ``cases``,
``resolution_scales``, and ``extent_scales`` below to choose the frequencies
and grid refinements to evaluate. Use ``plot_convergence.py`` to plot saved CSV
results without rerunning the solver.
"""

import csv
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
from astropy import units as unit

from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo


shell_mass_limit = 4e-9 * unit.M_earth
shell_radii = [12300 * unit.km, 384000 * unit.km]
states = ["1s", "2s", "2p", "3s", "3p", "3d", "4s", "4p", "4d", "5s", "5p", "5d"]

# Representative presets spanning the higher-frequency portion of the sweep.
# Lower-frequency grids have up to 40 million points and are deliberately not
# run by default; add them here when their memory and runtime are available.
cases = [
    # Baseline solve + inference runtimes, no plots (machine-dependent).
    # 100 kHz: 2.6 s; 1 MHz: 3.2 s (latest sweep); 10 MHz: 3.3 s; 30 MHz: 17.8 s.
    {"nu_a": 100 * unit.kHz, "N": 100_000, "extent": 1_000 * unit.R_earth},
    {"nu_a": 1 * unit.MHz, "N": 100_000, "extent": 100 * unit.R_earth},
    {"nu_a": 10 * unit.MHz, "N": 100_000, "extent": 100 * unit.R_earth},
    {"nu_a": 30 * unit.MHz, "N": 1_000_000, "extent": 70 * unit.R_earth},
]

# Resolution check: hold the outer boundary fixed and scale N.
resolution_scales = [0.5, 1.0, 2.0]
# Domain check: hold grid spacing approximately fixed as extent changes.
# Every domain must contain the outer shell boundary (~60.2 Earth radii).

output_path = Path(__file__).resolve().parent / "convergence_results.csv"
columns = [
    "frequency_Hz", "check", "scale", "N", "extent_Rearth", "runtime_s",
    "state", "energy_eV", "shell_fraction", "moon_enclosed_mass_Mearth",
    "earth_density_ratio",
]


def solve_case(frequency, check, scale, count, extent):
    start = perf_counter()
    halo = EarthBoundAxionHalo(nu_a=frequency, N=count, extent=extent)
    halo.solve_TISE_3D(l_vals=[0, 1, 2], max_n_r=10)
    # These warnings identify tiny shell probabilities (important when
    # interpreting relative convergence); the summary records this limitation.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*shell_fraction=.*at or below float64 machine epsilon.*"
        )
        masses = halo.inferHaloMass(shell_mass_limit, shell_radii, states)
    density = halo.getDensityAtEarthSurface(state_names=states)
    runtime = perf_counter() - start
    rows = []
    for state in states:
        rows.append({
            "frequency_Hz": frequency.to_value(unit.Hz),
            "check": check,
            "scale": scale,
            "N": count,
            "extent_Rearth": extent.to_value(unit.R_earth),
            "runtime_s": runtime,
            "state": state,
            "energy_eV": halo.states[state]["eigenE"].to_value(unit.eV),
            "shell_fraction": masses[state]["shell_fraction"].to_value(unit.one),
            "moon_enclosed_mass_Mearth": masses[state]["enclosed_mass"].to_value(unit.M_earth),
            "earth_density_ratio": density[state]["density_ratio"].to_value(unit.one),
        })
    print(
        f"{frequency:g}, {check} scale={scale:g}: N={count}, "
        f"extent={extent.to_value(unit.R_earth):g} R_earth, {runtime:.1f} s",
        flush=True,
    )
    del halo
    return rows


with output_path.open("w", newline="", encoding="utf-8") as output:
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for case in cases:
        frequency, base_n, base_extent = case["nu_a"], case["N"], case["extent"]
        runs = [("resolution", scale, max(2, round(base_n * scale)), base_extent)
                for scale in resolution_scales]
        domain_scales = ([0.5, 1.0, 2.0] if base_extent >= 121 * unit.R_earth
                         else [1.0, 2.0, 4.0])
        runs += [("extent", scale, max(2, round(base_n * scale)), base_extent * scale)
                 for scale in domain_scales]
        # Avoid computing the identical baseline twice.
        computed = {}
        for check, scale, count, extent in runs:
            key = (count, extent.to_value(unit.m))
            if key in computed:
                if check == "extent" and scale == 1.0:
                    # Re-label the already-computed baseline for direct
                    # baseline-to-domain comparisons without solving twice.
                    writer.writerows([
                        {**row, "check": "extent", "scale": 1.0}
                        for row in computed[key]
                    ])
                    output.flush()
                continue
            result_rows = solve_case(frequency, check, scale, count, extent)
            computed[key] = result_rows
            writer.writerows(result_rows)
            output.flush()

print(f"Wrote {output_path}")
