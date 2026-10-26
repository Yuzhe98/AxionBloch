"""Plot the 1 MHz parameter sweep without clipping data points."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


folder = Path(__file__).resolve().parent / "outputs" / "convergence_1MHz"
with (folder / "results_1MHz.csv").open(encoding="utf-8", newline="") as source:
    rows = list(csv.DictReader(source))
if not rows:
    raise RuntimeError("results_1MHz.csv is empty; run optimize_1MHz_params.py first.")

metrics = [
    ("energy_eV", "Eigenenergy"),
    ("shell_fraction", "Shell probability fraction"),
    ("moon_enclosed_mass_Mearth", "Moon-enclosed mass"),
    ("earth_density_ratio", "Earth-surface density ratio"),
]
data = defaultdict(dict)
for row in rows:
    check = row["check"]
    key = (int(row["N"]), float(row["extent_Rearth"]))
    data[check][key] = {
        "runtime": float(row["runtime_s"]),
        "dr": float(row["dr_Rearth"]),
        "states": {
            existing["state"]: {metric: float(existing[metric]) for metric, _ in metrics}
            for existing in rows
            if existing["check"] == check
            and int(existing["N"]) == key[0]
            and float(existing["extent_Rearth"]) == key[1]
        },
    }

for check, x_label, output_name in [
    ("resolution", "Interior grid points N", "convergence_1MHz_resolution.png"),
    ("domain", "Domain extent (Earth radii)", "convergence_1MHz_domain.png"),
]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    runs = data[check]
    for ax, (metric, title) in zip(axes.flat, metrics):
        reference_key = max(runs, key=(lambda key: key[0]) if check == "resolution"
                            else (lambda key: key[1]))
        reference = runs[reference_key]["states"]
        xs, medians, maxima = [], [], []
        for key in sorted(runs, key=(lambda item: item[0]) if check == "resolution"
                          else (lambda item: item[1])):
            current = runs[key]["states"]
            errors = []
            for state, values in reference.items():
                if state not in current:
                    continue
                ref_value = values[metric]
                value = current[state][metric]
                denom = max(abs(value), abs(ref_value), 1e-300)
                errors.append(100 * abs(value - ref_value) / denom)
            xs.append(key[0] if check == "resolution" else key[1])
            medians.append(float(np.median(errors)))
            maxima.append(float(np.max(errors)))
        ax.plot(xs, medians, linestyle="-", marker="o", label="Median state")
        ax.plot(xs, maxima, linestyle="--", marker="x", alpha=0.8,
                label="Worst state")
        ax.set_title(title)
        ax.set_ylabel("Relative change from finest/largest (%)")
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel(x_label)
    fig.suptitle(f"1 MHz {check} optimization: all tested points, autoscaled axes")
    fig.savefig(folder / output_name, dpi=180)
    plt.close(fig)

# Runtime and sampling distance for every run, saved as a compact second figure.
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
for check, marker in [("resolution", "o"), ("domain", "s")]:
    keys = sorted(data[check])
    ns = [key[0] for key in keys]
    secs = [data[check][key]["runtime"] for key in keys]
    ax.plot(ns, secs, marker=marker, label=check)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Interior grid points N")
ax.set_ylabel("Solve + inference runtime (s)")
ax.set_title("1 MHz runtime (plots excluded from timing)")
ax.grid(True, which="both", alpha=0.25)
ax.legend()
fig.savefig(folder / "runtime_1MHz.png", dpi=180)
plt.close(fig)
print(f"Wrote convergence_1MHz_resolution.png, convergence_1MHz_domain.png, and runtime_1MHz.png to {folder}")
