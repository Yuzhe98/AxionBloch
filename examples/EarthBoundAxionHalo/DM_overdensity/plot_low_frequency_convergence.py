"""Plot the saved 100 Hz--10 kHz domain-convergence checks.

This script only reads the saved CSV; it does not run the solver.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


folder = Path(__file__).resolve().parent
output_dir = folder / "outputs" / "low_frequency_convergence"
input_path = output_dir / "convergence_results.csv"
with input_path.open(encoding="utf-8", newline="") as source:
    rows = list(csv.DictReader(source))

metrics = [
    ("energy_eV", "Eigenenergy"),
    ("shell_fraction", "Shell probability fraction"),
    ("moon_enclosed_mass_Mearth", "Moon-enclosed mass"),
    ("earth_density_ratio", "Earth-surface density ratio"),
]
frequencies = sorted({float(row["frequency_Hz"]) for row in rows})
data = defaultdict(lambda: defaultdict(dict))
for row in rows:
    frequency = float(row["frequency_Hz"])
    if row["check"] == "extent":
        scale = float(row["scale"])
    elif row["check"] == "resolution" and float(row["scale"]) == 1.0:
        # The baseline grid is also the extent=1 comparison point.
        scale = 1.0
    else:
        continue
    data[frequency][scale][row["state"]] = row


def frequency_label(frequency):
    if frequency >= 1000:
        return f"{frequency / 1000:g} kHz"
    return f"{frequency:g} Hz"


def plot_frequency(frequency, destination):
    runs = data[frequency]
    reference = runs[max(runs)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, metrics):
        scales = sorted(runs)
        median, worst = [], []
        for scale in scales:
            errors = []
            for state, ref in reference.items():
                current = runs[scale].get(state)
                if current is None:
                    continue
                value, target = float(current[metric]), float(ref[metric])
                errors.append(
                    100 * abs(value - target) / max(abs(value), abs(target), 1e-300)
                )
            median.append(max(float(np.median(errors)), 1e-10))
            worst.append(max(float(np.max(errors)), 1e-10))
        x = [float(next(iter(runs[scale].values()))["extent_Rearth"]) for scale in scales]
        ax.plot(x, median, marker="o", label="Median state")
        ax.plot(x, worst, marker="x", linestyle="--", label="Worst state")
        ax.axhline(0.01, color="black", linestyle=":", linewidth=1,
                   label="0.01% criterion")
        ax.set_title(title)
        ax.set_xlabel("Domain extent (Earth radii)")
        ax.set_ylabel("Relative change from largest domain (%)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Domain convergence at {frequency_label(frequency)}")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


domain_summary = []
for frequency in frequencies:
    runs = data[frequency]
    reference = runs[max(runs)]
    for scale in sorted(runs):
        row = {
            "frequency_Hz": frequency,
            "N": int(next(iter(runs[scale].values()))["N"]),
            "extent_Rearth": float(next(iter(runs[scale].values()))["extent_Rearth"]),
        }
        for metric, _ in metrics:
            errors = []
            for state, ref in reference.items():
                current = runs[scale].get(state)
                if current is None:
                    continue
                value, target = float(current[metric]), float(ref[metric])
                errors.append(
                    100 * abs(value - target) / max(abs(value), abs(target), 1e-300)
                )
            row[f"{metric}_median_error_percent"] = float(np.median(errors))
            row[f"{metric}_worst_error_percent"] = float(np.max(errors))
        domain_summary.append(row)

with (output_dir / "domain_convergence_summary.csv").open(
    "w", encoding="utf-8", newline=""
) as target:
    writer = csv.DictWriter(target, fieldnames=list(domain_summary[0]))
    writer.writeheader()
    writer.writerows(domain_summary)

for frequency in frequencies:
    filename = frequency_label(frequency).replace(" ", "") + "_domain.png"
    plot_frequency(frequency, output_dir / filename)

# Combined four-metric view with one median/worst pair per frequency.
colors = dict(zip(frequencies, plt.get_cmap("viridis")(
    np.linspace(0.08, 0.92, len(frequencies))
)))
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for ax, (metric, title) in zip(axes.flat, metrics):
    for frequency in frequencies:
        runs = data[frequency]
        reference = runs[max(runs)]
        scales = sorted(runs)
        x, medians, worsts = [], [], []
        for scale in scales:
            errors = []
            for state, ref in reference.items():
                current = runs[scale].get(state)
                if current is None:
                    continue
                value, target = float(current[metric]), float(ref[metric])
                errors.append(
                    100 * abs(value - target) / max(abs(value), abs(target), 1e-300)
                )
            x.append(float(next(iter(runs[scale].values()))["extent_Rearth"]))
            medians.append(max(float(np.median(errors)), 1e-10))
            worsts.append(max(float(np.max(errors)), 1e-10))
        label = frequency_label(frequency)
        ax.plot(x, medians, color=colors[frequency], marker="o", label=label)
        ax.plot(x, worsts, color=colors[frequency], marker="x", linestyle="--", alpha=0.8)
    ax.axhline(0.01, color="black", linestyle=":", linewidth=1,
               label="0.01% criterion")
    ax.set_title(title)
    ax.set_xlabel("Domain extent (Earth radii)")
    ax.set_ylabel("Relative change from largest domain (%)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
fig.suptitle("Low-frequency domain convergence: median (solid), worst state (dashed)")
fig.savefig(output_dir / "low_frequency_domain_all.png", dpi=180)
plt.close(fig)
print(f"Wrote per-frequency domain plots and {output_dir / 'low_frequency_domain_all.png'}")
