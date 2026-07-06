"""Plot daily Omega_a modulation for coherent 2p-3p superpositions.

The coefficients below have fixed relative phases. Bound-state phase evolution
from the small 2p-3p energy splitting is not included.
"""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from axionbloch.dependency import *
from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo
from axionbloch.Station import Mainz

halo = EarthBoundAxionHalo(
    nu_a=1.348585 * unit.MHz,
    N=2**11,
    extent=32 * unit.R_earth,
    g_aNN=1e-9 * unit.GeV**-1,
    verbose=True,
)
halo.solve_TISE_3D(l_vals=[1], max_n_r=8, verbose=False)

t0 = Time(datetime(2022, 12, 13, 7, 0, tzinfo=ZoneInfo("Europe/Berlin")))
t_hours = np.linspace(0, 72, 144 + 1) * unit.hour
meas_times = t0 + t_hours

# Compute each basis state once. Any fixed coherent superposition can then be
# formed without repeating the expensive spatial interpolation.
gradient_by_state = {}
for state_name in ("2p", "3p"):
    gradient_by_state[state_name] = halo.findGradientsOverTime(
        stateCoefficients={state_name: 1.0},
        station=Mainz,
        meas_times=meas_times,
        truncRadius=3 * unit.R_earth,
        include_lorentz_boost=True,
        verbose=True,
    )

normalization = 1 / np.sqrt(2)
superpositions = {
    "$2p$": {"2p": 1.0, "3p": 0.0},
    "$(2p+3p)/\\sqrt{2}$": {
        "2p": normalization,
        "3p": normalization,
    },
    "$(2p+i3p)/\\sqrt{2}$": {
        "2p": normalization,
        "3p": 1j * normalization,
    },
    "$(2p-3p)/\\sqrt{2}$": {
        "2p": normalization,
        "3p": -normalization,
    },
}

Omega_factor = (
    const.c * halo.g_aNN * np.sqrt(halo.N_a * const.hbar**3 * const.c / (2 * halo.m_a))
)
components = [
    ("grad_r", "\\Omega_a^r"),
    ("grad_theta", "\\Omega_a^\\theta"),
    ("grad_phi", "\\Omega_a^\\varphi"),
]
colors = ["tab:blue", "tab:green", "tab:purple", "tab:red"]
linestyles = [":", "-", "-.", "--"]

fig, axes = plt.subplots(
    3,
    1,
    figsize=(7 / 2.54, 8 / 2.54),
    dpi=300,
    sharex=True,
    sharey=True,
)

for ax, (gradient_key, Omega_label) in zip(axes, components):
    gradient_2p = gradient_by_state["2p"][gradient_key]
    gradient_3p = gradient_by_state["3p"][gradient_key].to(
        gradient_2p.unit,
        equivalencies=unit.dimensionless_angles(),
    )

    for zorder, ((label, coefficients), color, linestyle) in enumerate(
        zip(superpositions.items(), colors, linestyles),
        start=2,
    ):
        combined_gradient = (
            coefficients["2p"] * gradient_2p + coefficients["3p"] * gradient_3p
        )
        Omega = (Omega_factor * np.abs(combined_gradient)).to(
            unit.mHz,
            equivalencies=unit.dimensionless_angles(),
        )
        ax.plot(
            t_hours,
            Omega,
            color=color,
            linestyle=linestyle,
            label=label,
            zorder=zorder,
        )

    ax.set_ylabel(f"${Omega_label}$\n$\\left(\\mathrm{{mHz}}\\right)$")

axes[0].legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    frameon=False,
    fontsize=7,
)
axes[-1].set_xlabel("Time (hour) from 2022-12-13 07:00 CET")
# fig.suptitle("Daily eigenstate-interference modulation at Mainz")
fig.tight_layout()
output_directory = Path(__file__).with_name("outputs")
output_directory.mkdir(exist_ok=True)
output_path = output_directory / "EarthHalo-interference-daily-Omega_a.pdf"
fig.savefig(output_path, bbox_inches="tight")
print("Saved", output_path)
plt.show()
