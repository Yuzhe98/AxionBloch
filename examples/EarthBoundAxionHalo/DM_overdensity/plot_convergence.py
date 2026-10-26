"""Plot convergence and runtime from convergence_results.csv.

Run from the repository root:
    python -m examples.EarthBoundAxionHalo.DM_overdensity.plot_convergence
"""

import csv
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


folder = Path(__file__).resolve().parent
output_folder = folder / "outputs" / "convergence"
output_folder.mkdir(parents=True, exist_ok=True)
source_path = folder / "convergence_results.csv"
with source_path.open(encoding="utf-8", newline="") as source:
    rows = list(csv.DictReader(source))
if not rows:
    raise RuntimeError("convergence_results.csv has no result rows; run the convergence check first.")
shutil.copy2(source_path, output_folder / source_path.name)

freqs = sorted({float(row["frequency_Hz"]) for row in rows})
colors = dict(zip(freqs, plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(freqs)))))
metrics = [
    ("energy_eV", "Eigenenergy"),
    ("shell_fraction", "Shell probability fraction"),
    ("moon_enclosed_mass_Mearth", "Moon-enclosed mass"),
    ("earth_density_ratio", "Earth-surface density ratio"),
]


def frequency_label(frequency):
    if frequency >= 1e6:
        return f"{frequency / 1e6:g} MHz"
    if frequency >= 1e3:
        return f"{frequency / 1e3:g} kHz"
    return f"{frequency:g} Hz"

# Aggregate each numerical run by state, then compare to the finest grid or
# largest domain tested for that frequency.
data = defaultdict(lambda: defaultdict(dict))
runtime_rows = {}
for row in rows:
    frequency = float(row["frequency_Hz"])
    check = row["check"]
    scale = float(row["scale"])
    state = row["state"]
    values = {metric: float(row[metric]) for metric, _ in metrics}
    values["_N"] = float(row["N"])
    values["_extent"] = float(row["extent_Rearth"])
    data[(frequency, check)][scale][state] = values
    identity = (frequency, check, int(row["N"]), float(row["extent_Rearth"]))
    runtime_rows[identity] = float(row["runtime_s"])

for check, x_label, output_name in [
    ("resolution", "Interior grid points N", "convergence_resolution.png"),
    ("extent", "Domain extent (Earth radii)", "convergence_extent.png"),
]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, metrics):
        for frequency in freqs:
            scale_data = data.get((frequency, check), {})
            if not scale_data:
                continue
            reference_scale = max(scale_data)
            reference = scale_data[reference_scale]
            x_values, median_errors, maximum_errors = [], [], []
            for scale in sorted(scale_data):
                run = scale_data[scale]
                errors = []
                for state, reference_values in reference.items():
                    if state not in run:
                        continue
                    value = run[state][metric]
                    target = reference_values[metric]
                    errors.append(100 * abs(value - target) / max(abs(value), abs(target), 1e-300))
                if not errors:
                    continue
                example = next(iter(run.values()))
                x_values.append(
                    example["_N"] if check == "resolution" else example["_extent"]
                )
                median_errors.append(max(float(np.median(errors)), 1e-8))
                maximum_errors.append(max(float(np.max(errors)), 1e-8))
            label = frequency_label(frequency)
            color = colors[frequency]
            ax.plot(x_values, median_errors, color=color, marker="o", label=label)
            ax.plot(x_values, maximum_errors, color=color, marker="x", linestyle="--", alpha=0.75)
        ax.set_title(title)
        ax.set_ylabel("Relative change from finest/largest (%)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel(x_label)
        ax.set_xscale("log")
    fig.suptitle(f"{check.title()} convergence: median (solid) and worst state (dashed)")
    fig.savefig(output_folder / output_name, dpi=180)
    plt.close(fig)

# Runtime for each unique solve; the star marks the preset run for each frequency.
fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
markers = {"resolution": "o", "extent": "s"}
for (frequency, check, count, extent), seconds in sorted(runtime_rows.items()):
    is_baseline = any(
        float(row["frequency_Hz"]) == frequency
        and row["check"] == check
        and float(row["scale"]) == 1.0
        and int(row["N"]) == count
        and float(row["extent_Rearth"]) == extent
        for row in rows
    )
    ax.scatter(
        count, seconds, marker="*" if is_baseline else markers[check],
        s=105 if is_baseline else 48, color=colors[frequency],
        edgecolor="black" if is_baseline else "none", alpha=0.9,
    )
for frequency, color in colors.items():
    ax.scatter([], [], color=color, label=frequency_label(frequency))
ax.scatter([], [], marker="o", color="gray", label="Resolution runs")
ax.scatter([], [], marker="s", color="gray", label="Domain runs")
ax.scatter([], [], marker="*", color="gray", edgecolor="black", label="Preset baseline")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Interior grid points N")
ax.set_ylabel("Solve + inference runtime (s)")
ax.set_title("Measured convergence-run runtime (plots excluded)")
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=8, ncol=2)
fig.savefig(output_folder / "convergence_runtime.png", dpi=180)
plt.close(fig)

# Keep the prior focused check alongside the main convergence data, if present.
for name in ["convergence_30MHz_1s_high_resolution.csv",
             "convergence_30MHz_1s_high_resolution.png",
             "convergence_findings.md"]:
    prior = folder / name
    if prior.exists():
        shutil.copy2(prior, output_folder / name)

print(f"Wrote convergence figures and copied data to {output_folder}")
