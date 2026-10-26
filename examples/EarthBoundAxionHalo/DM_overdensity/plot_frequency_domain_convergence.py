"""Combine saved domain-convergence data across tested frequencies.

Run from the repository root::

    python -m examples.EarthBoundAxionHalo.DM_overdensity.plot_frequency_domain_convergence

No solver runs are performed. The legacy 1 MHz domain rows are replaced by
the denser 1 MHz extent sweep in outputs/1MHz_shell_mass.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


folder = Path(__file__).resolve().parent
output_dir = folder / "outputs" / "convergence"
output_dir.mkdir(parents=True, exist_ok=True)
base_csv = output_dir / "convergence_results.csv"
fine_1mhz_csv = folder / "outputs" / "1MHz_shell_mass" / "convergence_1MHz.csv"
output_csv = output_dir / "convergence_domain_all_frequencies.csv"
output_png = output_dir / "convergence_domain_all_frequencies.png"

metrics = [
    ("energy_eV", "Eigenenergy"),
    ("shell_fraction", "Shell probability fraction"),
    ("moon_enclosed_mass_Mearth", "Moon-enclosed mass"),
    ("earth_density_ratio", "Earth-surface density ratio"),
]

with base_csv.open(encoding="utf-8", newline="") as source:
    rows = list(csv.DictReader(source))
rows = [row for row in rows if not (
    float(row["frequency_Hz"]) == 1e6 and row["check"] == "extent"
)]
with fine_1mhz_csv.open(encoding="utf-8", newline="") as source:
    for row in csv.DictReader(source):
        if row["check"] != "extent":
            continue
        rows.append({
            "frequency_Hz": "1000000",
            "check": "extent",
            "scale": row["extent_Rearth"],
            "N": row["N"],
            "extent_Rearth": row["extent_Rearth"],
            "runtime_s": row["runtime_s"],
            "state": row["state"],
            "energy_eV": row["energy_eV"],
            "shell_fraction": row["shell_fraction"],
            "moon_enclosed_mass_Mearth": row["moon_enclosed_mass_Mearth"],
            "earth_density_ratio": row["earth_density_ratio"],
        })

frequencies = sorted({float(row["frequency_Hz"]) for row in rows
                      if row["check"] == "extent"})
data = defaultdict(dict)
for row in rows:
    if row["check"] != "extent":
        continue
    frequency = float(row["frequency_Hz"])
    key = float(row["extent_Rearth"])
    data[frequency].setdefault(key, {})[row["state"]] = row

curve_rows = []
for frequency in frequencies:
    runs = data[frequency]
    reference = runs[max(runs)]
    for extent in sorted(runs):
        errors_by_metric = {metric: [] for metric, _ in metrics}
        for state, ref in reference.items():
            current = runs[extent].get(state)
            if current is None:
                continue
            for metric, _ in metrics:
                value = float(current[metric])
                target = float(ref[metric])
                error = 100 * abs(value - target) / max(abs(value), abs(target), 1e-300)
                errors_by_metric[metric].append(error)
        example = next(iter(runs[extent].values()))
        output = {
            "frequency_Hz": frequency,
            "N": int(example["N"]),
            "extent_Rearth": extent,
        }
        for metric, _ in metrics:
            values = errors_by_metric[metric]
            output[f"{metric}_median_error_percent"] = float(np.median(values))
            output[f"{metric}_worst_error_percent"] = float(np.max(values))
        curve_rows.append(output)

columns = list(curve_rows[0])
with output_csv.open("w", newline="", encoding="utf-8") as target:
    writer = csv.DictWriter(target, fieldnames=columns)
    writer.writeheader()
    writer.writerows(curve_rows)

colors = dict(zip(frequencies, plt.get_cmap("viridis")(
    np.linspace(0.08, 0.92, len(frequencies))
)))
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for ax, (metric, title) in zip(axes.flat, metrics):
    for frequency in frequencies:
        curve = sorted(
            (row for row in curve_rows if row["frequency_Hz"] == frequency),
            key=lambda row: row["extent_Rearth"],
        )
        label = f"{frequency / 1e6:g} MHz" if frequency >= 1e6 else f"{frequency / 1e3:g} kHz"
        x = [row["extent_Rearth"] for row in curve]
        median = [max(row[f"{metric}_median_error_percent"], 1e-10) for row in curve]
        worst = [max(row[f"{metric}_worst_error_percent"], 1e-10) for row in curve]
        ax.plot(x, median, color=colors[frequency], marker="o", label=label)
        ax.plot(x, worst, color=colors[frequency], marker="x", linestyle="--", alpha=0.8)
    ax.axhline(0.01, color="black", linestyle=":", linewidth=1,
               label="0.01% criterion")
    ax.set_title(title)
    ax.set_xlabel("Domain extent (Earth radii)")
    ax.set_ylabel("Relative change from largest domain (%)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
fig.suptitle("Domain convergence by frequency: median (solid), worst state (dashed)")
fig.savefig(output_png, dpi=180)
plt.close(fig)
print(f"Wrote {output_png} and {output_csv}")
