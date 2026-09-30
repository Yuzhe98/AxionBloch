"""Plot Earth-surface mean overdensity with shaded uncertainty bands from the saved table."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const, units as unit

# Edit the input file and the states to include in the plot.
input_path = Path(__file__).resolve().parent / "outputs" / "frequency_sweep-2nd" / "density_at_earth_vs_frequency.txt"
state_names = ["1s", 
            #    "2s", "2p", "3s", "3p", "3d",
            # "4s", "4p", "4d", "5s", "5p", "5d"
               ]
output_path = input_path.with_name("density_at_earth_vs_frequency_uncertainty_band.png")
show_plot = True
# Match the reference density used to generate the input table.
rho_M_DM_SHM = 0.3 * unit.GeV / (const.c**2 * unit.cm**3)

# The file stores frequency in Hz and both density columns in rho_M_DM_SHM units.
data = np.genfromtxt(
    input_path,
    comments="#",
    dtype=[("frequency", float), ("state", "U16"),
           ("density", float), ("uncertainty", float)],
    ndmin=1,
)
available_states = np.unique(data["state"])
if not state_names or any(state not in available_states for state in state_names):
    raise ValueError(f"Choose state_names from {available_states.tolist()}.")

# Plot the mean with the full quoted absolute uncertainty.
fig, ax = plt.subplots(figsize=(13 / 2.54, 9 / 2.54), dpi=300)
ax.set_xscale("log")
# Bands extending below zero continue off the plot on the logarithmic axis.
# Keep their original uncertainties rather than replacing negative lower bounds.
ax.set_yscale("log", nonpositive="clip")
for state in state_names:
    rows = data[data["state"] == state]
    rows = np.sort(rows, order="frequency")
    overdensity = rows["density"]
    if (np.any(~np.isfinite(rows["frequency"])) or np.any(rows["frequency"] <= 0)
            or np.any(~np.isfinite(overdensity)) or np.any(overdensity <= 0)):
        raise ValueError(f"{state}: log-log plotting requires finite, positive frequencies and mean densities.")
    if np.any(~np.isfinite(rows["uncertainty"])) or np.any(rows["uncertainty"] < 0):
        raise ValueError(f"{state}: uncertainties must be finite and nonnegative.")
    line, = ax.plot(rows["frequency"], overdensity, "o-", markersize=3, label=state)
    ax.fill_between(
        rows["frequency"],
        overdensity - rows["uncertainty"],
        overdensity + rows["uncertainty"],
        color=line.get_color(), alpha=.15, linewidth=0,
    )

# Save at the requested size and resolution, then optionally display the figure.
ax.set_xlabel(r"Axion frequency $\nu_a$ (Hz)")
ax.set_ylabel(r"$\rho_\oplus / \rho_{\mathrm{SHM}}$")
ax.grid(True, which="both", alpha=.25)
ax.legend(title="Eigenstate")
ax.set_ylim(bottom=.5e01)

# Convert the dimensionless overdensity to mass density on the right axis.
rho_reference = rho_M_DM_SHM.to_value(unit.g / unit.cm**3)
density_ax = ax.secondary_yaxis(
    "right",
    functions=(lambda ratio: ratio * rho_reference,
               lambda density: density / rho_reference),
)
density_ax.set_ylabel(r"$\rho_\oplus$ (g/cm$^3$)")

# Since m_a c^2 = h nu_a, h in eV s converts Hz to mass in eV/c^2.
mass_per_hz = const.h.to_value(unit.eV * unit.s)
mass_ax = ax.twiny()
mass_ax.set_xscale("log")
mass_ax.set_xlim(np.asarray(ax.get_xlim()) * mass_per_hz)
mass_ax.set_xlabel(r"Axion mass $m_a$ (eV/$c^2$)")
# Keep the top axis aligned when frequency limits change during zooming.
ax.callbacks.connect(
    "xlim_changed",
    lambda frequency_ax: mass_ax.set_xlim(np.asarray(frequency_ax.get_xlim()) * mass_per_hz),
)
fig.tight_layout()
fig.savefig(output_path, dpi=300)
print(f"Saved {output_path}")
if show_plot:
    plt.show()
