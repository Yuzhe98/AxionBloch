"""Infer 1 MHz halo density profiles and check numerical convergence.

Run from the repository root::

    python -m examples.EarthBoundAxionHalo.DM_overdensity.density_by_shell_mass

Density profiles, run data, and convergence figures are written to
``examples/EarthBoundAxionHalo/DM_overdensity/outputs/1MHz_shell_mass``.
"""

import csv
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as unit

from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo


frequency = 1 * unit.MHz
shell_mass_limit = 4e-9 * unit.M_earth
shell_radii = [12300 * unit.km, 384000 * unit.km]
states = ["1s", "2s", "2p", "3s", "3p", "3d", "4s", "4p", "4d", "5s", "5p", "5d"]

# Keep the established 1 MHz preset as the profile used for the reported
# distribution. Convergence runs vary N or the outer boundary independently.
baseline_N = 100_000
baseline_extent = 100 * unit.R_earth
# A 0.001% maximum deviation across all states and tracked quantities is a
# strict practical saturation criterion for these inferred profiles.
convergence_threshold_percent = 0.001
resolution_counts = [50_000, 75_000, 100_000, 150_000, 200_000,
                     300_000, 400_000, 600_000, 800_000,
                     1_200_000, 1_600_000, 2_400_000, 3_200_000]
# For the extent sweep, keep dr near 0.001 R_earth (N/extent ~= 1000/R_earth)
# to avoid confusing a larger box with a coarser radial grid.
domain_extents = [70, 85, 100, 125, 150, 200, 300, 500]

output_dir = Path(__file__).resolve().parent / "outputs" / "1MHz_shell_mass"
output_dir.mkdir(parents=True, exist_ok=True)


def solve_and_infer(count, extent):
    start = perf_counter()
    halo = EarthBoundAxionHalo(nu_a=frequency, N=count, extent=extent)
    halo.solve_TISE_3D(l_vals=[0, 1, 2], max_n_r=10)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*shell_fraction=.*at or below float64 machine epsilon.*"
        )
        masses = halo.inferHaloMass(shell_mass_limit, shell_radii, states)
    density = halo.getDensityAtEarthSurface(state_names=states)
    return halo, masses, density, perf_counter() - start


columns = ["check", "N", "extent_Rearth", "dr_Rearth", "runtime_s", "state",
           "energy_eV", "shell_fraction", "total_mass_Mearth",
           "moon_enclosed_mass_Mearth", "earth_density_ratio"]
rows = []
for check, count, extent in (
    [("resolution", n, baseline_extent) for n in resolution_counts]
    + [("extent", int(x * 1_000), x * unit.R_earth) for x in domain_extents]
):
    halo, masses, earth_density, runtime = solve_and_infer(count, extent)
    for state in states:
        rows.append({
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
            "earth_density_ratio": earth_density[state]["density_ratio"].to_value(unit.one),
        })
    print(f"{check}: N={count:,}, extent={extent.to_value(unit.R_earth):g} R_earth, "
          f"dr={halo.dr.to_value(unit.R_earth):.4g} R_earth, {runtime:.2f} s", flush=True)
    del halo

with (output_dir / "convergence_1MHz.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

# Use the baseline resolution/domain to derive the requested radial
# distribution and retain numerical values as a portable CSV.
halo, masses, earth_density, _ = solve_and_infer(baseline_N, baseline_extent)
radii = np.geomspace(0.01, baseline_extent.to_value(unit.R_earth), 1200) * unit.R_earth
fig, axes, profiles = halo.plotDMdensity(
    radii, state_names=states, scales=("log", "log"), showPlot=False
)
fig.savefig(output_dir / "density_profiles_1MHz.png", dpi=180, bbox_inches="tight")
plt.close(fig)

profile_columns = ["radius_Rearth"] + [f"{state}_rho_over_SHM" for state in states]
with (output_dir / "density_profiles_1MHz.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=profile_columns)
    writer.writeheader()
    reference_density = halo.rho_M_DM_SHM.to(
        unit.g / unit.cm**3, equivalencies=unit.mass_energy()
    )
    for index, radius in enumerate(radii):
        writer.writerow({
            "radius_Rearth": radius.to_value(unit.R_earth),
            **{f"{state}_rho_over_SHM": (profiles[state]["density"][index]
                 / reference_density).to_value(unit.one) for state in states},
        })

# Plot convergence of four quantities over all solved states. Errors are
# measured against the finest N or largest domain in the corresponding sweep.
metrics = [
    ("energy_eV", "Eigenenergy"),
    ("shell_fraction", "Shell probability fraction"),
    ("total_mass_Mearth", "Total inferred mass in domain"),
    ("moon_enclosed_mass_Mearth", "Moon-enclosed mass"),
    ("earth_density_ratio", "Earth-surface density ratio"),
]
for check, x_label, filename in [
    ("resolution", "Interior grid points N", "convergence_1MHz_resolution.png"),
    ("extent", "Domain extent (Earth radii)", "convergence_1MHz_extent.png"),
]:
    selected = [row for row in rows if row["check"] == check]
    keys = sorted({row["N"] if check == "resolution" else row["extent_Rearth"] for row in selected})
    reference_key = keys[-1]
    reference = {r["state"]: r for r in selected
                 if (r["N"] if check == "resolution" else r["extent_Rearth"]) == reference_key}
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, metrics):
        medians, maxima = [], []
        for key in keys:
            current = {r["state"]: r for r in selected
                       if (r["N"] if check == "resolution" else r["extent_Rearth"]) == key}
            errors = []
            for state, ref in reference.items():
                value, target = current[state][metric], ref[metric]
                errors.append(100 * abs(value - target) / max(abs(value), abs(target), 1e-300))
            medians.append(max(float(np.median(errors)), 1e-10))
            maxima.append(max(float(np.max(errors)), 1e-10))
        ax.plot(keys, medians, "o-", label="Median state")
        ax.plot(keys, maxima, "x--", label="Worst state")
        ax.axhline(convergence_threshold_percent, color="black", linestyle=":",
                   linewidth=1, label="0.1% threshold")
        ax.set_title(title)
        ax.set_ylabel("Relative change from finest/largest (%)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel(x_label)
    axes.flat[-1].set_visible(False)
    fig.suptitle(f"1 MHz {check} convergence; threshold = {convergence_threshold_percent:g}%")
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)

# Record a reproducible threshold assessment using each sweep's finest or
# largest run as its reference. Require every state/metric to pass.
summary_lines = [
    "1 MHz convergence assessment",
    f"Criterion: maximum state-by-state relative deviation <= {convergence_threshold_percent:g}%",
    "Metrics: energy, shell fraction, total inferred mass, Moon-enclosed mass, Earth-surface density.",
    "Reference runs: N=3,200,000 at 100 R_earth; extent=500 R_earth at dr=0.001 R_earth.",
]
for check, x_field in [("resolution", "N"), ("extent", "extent_Rearth")]:
    selected = [row for row in rows if row["check"] == check]
    keys = sorted({row[x_field] for row in selected})
    ref_key = keys[-1]
    reference = {row["state"]: row for row in selected if row[x_field] == ref_key}
    passing = []
    for key in keys:
        current = {row["state"]: row for row in selected if row[x_field] == key}
        worst = 0.0
        worst_label = ""
        for state, ref in reference.items():
            for metric, _ in metrics:
                value, target = float(current[state][metric]), float(ref[metric])
                error = 100 * abs(value - target) / max(abs(value), abs(target), 1e-300)
                if error > worst:
                    worst, worst_label = error, f"{state} {metric}"
        if worst <= convergence_threshold_percent:
            passing.append((key, worst, worst_label))
    if passing:
        recommendation = passing[0]
        unit_label = "N" if check == "resolution" else "R_earth"
        summary_lines.append(
            f"{check}: smallest passing setting is {recommendation[0]:g} {unit_label}; "
            f"worst deviation={recommendation[1]:.6g}% ({recommendation[2]})."
        )
    else:
        summary_lines.append(f"{check}: no tested setting passes the criterion.")
    summary_lines.append(f"{check} reference-relative maximum deviations:")
    for key in keys:
        current = {row["state"]: row for row in selected if row[x_field] == key}
        worst = max(
            100 * abs(float(current[state][metric]) - float(ref[metric]))
            / max(abs(float(current[state][metric])), abs(float(ref[metric])), 1e-300)
            for state, ref in reference.items() for metric, _ in metrics
        )
        summary_lines.append(f"  {key:g}: {worst:.6g}%")
with (output_dir / "convergence_summary_1MHz.txt").open("w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines) + "\n")

# Concise human-readable inference for the baseline profile.
with (output_dir / "inferred_masses_1MHz.txt").open("w", encoding="utf-8") as f:
    f.write("1 MHz single-state halo inference; shell mass limit = 4e-9 M_earth\n")
    f.write("Baseline grid: N=100000, extent=100 R_earth\n\n")
    for state in states:
        mass = masses[state]
        f.write(f"{state}: total={mass['total_mass']:.6g}, "
                f"inside Moon={mass['enclosed_mass']:.6g}, "
                f"shell fraction={mass['shell_fraction']:.6g}, "
                f"Earth density={earth_density[state]['density_ratio']:.6g} rho_M_DM_SHM\n")

print(f"Wrote density profile, inference, and convergence results to {output_dir}")
