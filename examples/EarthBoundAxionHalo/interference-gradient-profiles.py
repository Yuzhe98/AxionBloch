"""Plot spatial-gradient interference between the 2p and 3p eigenstates."""

from pathlib import Path

from axionbloch.dependency import *
from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo
from axionbloch.Station import Mainz


halo = EarthBoundAxionHalo(
    nu_a=1.348585 * unit.MHz,
    N=2**11,
    extent=32 * unit.R_earth,
    verbose=True,
)
halo.solve_TISE_3D(l_vals=[1], max_n_r=8, verbose=False)

meas_time = Time("2022-12-14T12:00:00")
normalization = 1 / np.sqrt(2)
superpositions = {
    "$2p$": {"2p": 1.0},
    "$3p$": {"3p": 1.0},
    "$(2p+3p)/\\sqrt{2}$": {
        "2p": normalization,
        "3p": normalization,
    },
    "$(2p-3p)/\\sqrt{2}$": {
        "2p": normalization,
        "3p": -normalization,
    },
    "$(2p+i3p)/\\sqrt{2}$": {
        "2p": normalization,
        "3p": 1j * normalization,
    },
}

results = {}
for label, coefficients in superpositions.items():
    _, _, r_line, grad_r, grad_theta, grad_phi = halo.findGradients(
        stateCoefficients=coefficients,
        station=Mainz,
        meas_time=meas_time,
        truncRadius=3 * unit.R_earth,
        include_lorentz_boost=False,
        showPlot=False,
    )
    results[label] = {
        "r_line": r_line,
        "grad_r": grad_r,
        "grad_theta": grad_theta,
        "grad_phi": grad_phi,
    }

components = [
    ("grad_r", "\\partial_r\\Psi"),
    ("grad_theta", "\\frac{1}{r}\\partial_\\theta\\Psi"),
    ("grad_phi", "\\frac{1}{r\\!\\sin\\theta}\\partial_\\varphi\\Psi"),
]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
linestyles = [":", ":", "-", "--", "-."]

fig, axes = plt.subplots(
    3,
    2,
    figsize=(17.0 / 2.54, 10.5 / 2.54),
    dpi=300,
    sharex="col",
    sharey="row",
)

for (real_ax, imag_ax), (gradient_key, gradient_label) in zip(axes, components):
    reference_unit = results["$2p$"][gradient_key].unit
    for zorder, ((label, result), color, linestyle) in enumerate(
        zip(results.items(), colors, linestyles),
        start=2,
    ):
        gradient = result[gradient_key].to(
            reference_unit,
            equivalencies=unit.dimensionless_angles(),
        )
        radius = result["r_line"].to_value(unit.R_earth)
        real_ax.plot(
            radius,
            gradient.real.value,
            color=color,
            linestyle=linestyle,
            label=label,
            zorder=zorder,
        )
        imag_ax.plot(
            radius,
            gradient.imag.value,
            color=color,
            linestyle=linestyle,
            label=label,
            zorder=zorder,
        )

    unit_string = reference_unit.to_string("latex_inline")[1:-1]
    real_ax.set_ylabel(
        f"${gradient_label}$\n$\\left({unit_string}\\right)$",
        color="black",
    )
    real_ax.axvline(1.0, color="black", linestyle=":", linewidth=0.8)
    imag_ax.axvline(1.0, color="black", linestyle=":", linewidth=0.8)

axes[0, 0].set_title("$\\mathrm{Re}[\\nabla\\Psi]$")
axes[0, 1].set_title("$\\mathrm{Im}[\\nabla\\Psi]$")
axes[0, 0].legend(
    loc="lower center",
    bbox_to_anchor=(1.05, 1.20),
    ncol=3,
    frameon=False,
    fontsize=7,
)
axes[-1, 0].set_xlabel("$r/R_\\oplus$")
axes[-1, 1].set_xlabel("$r/R_\\oplus$")
fig.suptitle("Eigenstate interference at Mainz")
fig.tight_layout()
output_directory = Path(__file__).with_name("outputs")
output_directory.mkdir(exist_ok=True)
output_path = output_directory / "EarthHalo-interference-gradient-profiles.png"
fig.savefig(output_path, bbox_inches="tight")
print("Saved", output_path)
plt.show()
