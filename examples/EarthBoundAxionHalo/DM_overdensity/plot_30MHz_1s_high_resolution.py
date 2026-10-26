"""Plot the targeted high-resolution 30 MHz 1s check."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


folder = Path(__file__).resolve().parent
with (folder / "convergence_30MHz_1s_high_resolution.csv").open(
    encoding="utf-8", newline=""
) as source:
    rows = sorted(csv.DictReader(source), key=lambda row: int(row["N"]))
if not rows:
    raise RuntimeError("The high-resolution CSV has no data.")

n = np.asarray([int(row["N"]) for row in rows])
shell = np.asarray([float(row["shell_fraction"]) for row in rows])
density = np.asarray([float(row["earth_density_ratio"]) for row in rows])
energy = np.asarray([float(row["energy_eV"]) for row in rows])
runtime = np.asarray([float(row["runtime_s"]) for row in rows])
energy_error = np.abs(energy - energy[-1]) / abs(energy[-1]) * 100

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
series = [
    (shell, "Shell probability fraction", "#6a3d9a"),
    (density, "Earth-surface density ratio", "#1f78b4"),
    (np.maximum(energy_error, 1e-14), "Energy difference from N = 16M (%)", "#33a02c"),
]
for ax, (values, title, color) in zip(axes, series):
    ax.plot(n, values, marker="o", color=color, linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Interior grid points N")
    ax.grid(True, which="both", alpha=0.25)
    for count, value, seconds in zip(n, values, runtime):
        ax.annotate(f"{seconds:.0f}s", (count, value), xytext=(4, 5),
                    textcoords="offset points", fontsize=8)
axes[0].set_ylabel("Value (log scale)")
fig.suptitle("30 MHz 1s at fixed 70 Earth radii — shell and density remain unstable")
output = folder / "convergence_30MHz_1s_high_resolution.png"
fig.savefig(output, dpi=180)
plt.close(fig)
print(f"Wrote {output}")
