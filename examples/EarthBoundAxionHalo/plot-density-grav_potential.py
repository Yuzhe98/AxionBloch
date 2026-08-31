"""Plot the PREM density, enclosed mass, and gravitational potential.

The figure is formatted for a single journal column (8.5 cm wide).
"""

from pathlib import Path

from axionbloch.dependency import *
from axionbloch.EarthBoundAxionHalo import (
    PREM_density_profile,
    earth_grav_potential_earth_center,
    getCumulativeMass,
)


def make_figure():
    """Return the three-panel Earth-profile figure."""
    radius_m, density_kg_m3 = PREM_density_profile()
    radius = radius_m * unit.meter
    density = density_kg_m3 * unit.kg / unit.meter**3

    mass_radius, enclosed_mass = getCumulativeMass()
    potential, radius_unit, potential_unit = earth_grav_potential_earth_center()

    radius_extended = np.linspace(0, 3 * radius[-1].to_value(radius_unit), 1000)
    potential_extended = potential(radius_extended) * potential_unit

    cm = 1 / 2.54
    fig = plt.figure(figsize=(8.5 * cm, 10.5 * cm), dpi=300)
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.08)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]

    for index, ax in enumerate(axes):
        ax.text(
            0.02,
            0.92,
            f"({chr(ord('a') + index)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
        ax.tick_params(
            direction="in", which="both", top=True, right=True, length=3, pad=2
        )
        ax.grid(alpha=0.25, linewidth=0.4)

    axes[0].plot(
        radius.to_value(unit.R_earth), density.to_value(density.unit), color="tab:blue"
    )
    axes[0].set_ylabel("Density $(\\mathrm{g}\\,\\mathrm{cm}^{-3})$")

    axes[1].plot(
        mass_radius.to_value(unit.R_earth),
        enclosed_mass.to_value(unit.kg) / 1e24,
        color="tab:green",
    )
    axes[1].set_ylabel("Enclosed mass $(10^{24}\\,\\mathrm{kg})$")

    axes[2].plot(
        radius_extended, potential_extended.to_value(potential_unit), color="tab:orange"
    )
    axes[2].axvline(1, color="black", linestyle="dotted", linewidth=0.8)
    axes[2].set_ylabel("$\\Phi_\\oplus\\, (\\mathrm{MJ}\\,\\mathrm{kg}^{-1})$")
    axes[2].set_xlabel("Radius ($R_\\oplus$)")

    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[0].set_xlim(0, 3)
    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.10, top=0.99)
    return fig


if __name__ == "__main__":
    output = Path(__file__).with_name("earth-profiles.pdf")
    figure = make_figure()
    # Do not use bbox_inches="tight": the untrimmed canvas preserves the
    # requested 8.5 cm column width in the exported PDF.
    # figure.savefig(output)
    plt.show()
