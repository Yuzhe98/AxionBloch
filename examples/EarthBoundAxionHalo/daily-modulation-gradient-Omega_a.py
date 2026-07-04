"""Plot daily wavefunction-gradient and Omega_a modulation at Mainz.

The left column shows the spatial gradient of the time-independent
wavefunction. The right column compares the corresponding Rabi frequency
Omega_a with and without the first-order Lorentz boost.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from axionbloch.dependency import *
from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo
from axionbloch.Station import Mainz


states_to_check = ["2p"]
halo = EarthBoundAxionHalo(
    nu_a=1.348585 * unit.MHz,
    N=int(2**12),
    extent=128.0 * unit.R_earth,
    g_aNN=1e-9 * unit.GeV**-1,
    verbose=True,
)

halo.showValueAndUnits()
halo.solve_TISE_3D(
    l_vals=[1],
    max_n_r=64,
    verbose=False,
)

t0 = Time(datetime(2022, 12, 13, 7, 0, tzinfo=ZoneInfo("Europe/Berlin")))
t_hours = np.linspace(0, 72, 72) * unit.hour
meas_times = t0 + t_hours

gradient_result_no_boost = halo.findGradientsOverTime(
    stateCoefficients={name: 1.0 for name in states_to_check},
    station=Mainz,
    meas_times=meas_times,
    truncRadius=3 * unit.R_earth,
    include_lorentz_boost=False,
    verbose=True,
)

gradient_result_with_boost = halo.findGradientsOverTime(
    stateCoefficients={name: 1.0 for name in states_to_check},
    station=Mainz,
    meas_times=meas_times,
    truncRadius=3 * unit.R_earth,
    include_lorentz_boost=True,
    verbose=True,
)

components = [
    (
        "grad_r",
        "\\partial_r \\Psi",
        "\\Omega_a^r",
    ),
    (
        "grad_theta",
        "\\frac{1}{r}\\partial_\\theta \\Psi",
        "\\Omega_a^\\theta",
    ),
    (
        "grad_phi",
        "\\frac{1}{r\\!\\sin\\theta}\\partial_\\varphi\\Psi",
        "\\Omega_a^\\phi",
    ),
]

Omega_factor = (
    const.c
    * halo.g_aNN
    * np.sqrt(halo.N_a * const.hbar**3 * const.c / (2 * halo.m_a))
)

fig, axes = plt.subplots(
    3,
    2,
    figsize=(17.0 / 2.54, 9.0 / 2.54),
    dpi=300,
    sharex="col",
    sharey="col",
)
gradient_axes = axes[:, 0]
Omega_axes = axes[:, 1]

for gradient_ax, Omega_ax, component in zip(
    gradient_axes, Omega_axes, components
):
    gradient_key, gradient_label, Omega_label = component
    gradient_no_boost = gradient_result_no_boost[gradient_key]
    gradient_with_boost = gradient_result_with_boost[gradient_key]
    Omega_no_boost = (
        Omega_factor * np.abs(gradient_no_boost)
    ).to(unit.mHz, equivalencies=unit.dimensionless_angles())
    Omega_with_boost = (
        Omega_factor * np.abs(gradient_with_boost)
    ).to(unit.mHz, equivalencies=unit.dimensionless_angles())

    gradient_ax.plot(
        t_hours,
        gradient_no_boost.real,
        color="tab:blue",
        linestyle="-",
    )

    Omega_ax.plot(
        t_hours,
        Omega_no_boost,
        color="tab:blue",
        linestyle="--",
        label="Without Lorentz boost",
        zorder=3,
    )
    Omega_ax.plot(
        t_hours,
        Omega_with_boost,
        color="tab:green",
        linestyle="-",
        label="With Lorentz boost",
        zorder=2,
    )

    gradient_unit = gradient_no_boost.unit.to_string("latex_inline")[1:-1]

    gradient_ax.set_ylabel(
        f"${gradient_label}$\n$\\left({gradient_unit}\\right)$",
        color="black",
        # rotation=0,
        loc="center",
        labelpad=10,
    )
    Omega_ax.set_ylabel(
        f"${Omega_label}\\, \\left(\\mathrm{{mHz}}\\right)$",
        color="black",
    )
    gradient_ax.tick_params(axis="y", colors="black")
    Omega_ax.tick_params(axis="y", colors="black")

# gradient_axes[0].set_title("Spatial wavefunction gradient")
# Omega_axes[0].set_title("$\\Omega_a$")
Omega_axes[0].legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.16),
    ncol=2,
    frameon=False,
    fontsize=7,
)

gradient_axes[-1].set_xlabel("Time (hour) from 2022-12-13 07:00 CET")
Omega_axes[-1].set_xlabel("Time (hour) from 2022-12-13 07:00 CET")
fig.suptitle(
    f"Mainz  ·  {states_to_check[0]} state  ·  "
    f"$\\nu_a={halo.nu_a.to_value(unit.MHz):.6f}$ MHz",
    y=0.99,
)

gradient_ylim_abs = 0
for ax in gradient_axes:
    ylim_bottom, ylim_top = ax.get_ylim()
    gradient_ylim_abs = np.amax(
        np.abs([gradient_ylim_abs, ylim_bottom, ylim_top])
    )
for ax in gradient_axes:
    ax.set_ylim(-gradient_ylim_abs, gradient_ylim_abs)

# Omega_ylim_abs = 0
# for ax in Omega_axes:
#     ylim_bottom, ylim_top = ax.get_ylim()
#     Omega_ylim_abs = np.amax(np.abs([Omega_ylim_abs, ylim_bottom, ylim_top]))
# for ax in Omega_axes:
#     ax.set_ylim(top=Omega_ylim_abs)

fig.tight_layout()
plt.show()
