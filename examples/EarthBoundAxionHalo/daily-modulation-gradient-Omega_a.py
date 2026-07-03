"""Plot daily wavefunction-gradient and Omega_a modulation at Mainz.

Each panel shows one spherical gradient component on the left y-axis and the
corresponding axion-nucleon coupling frequency on a twinned right y-axis.
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
    stateNames=states_to_check,
    station=Mainz,
    meas_times=meas_times,
    truncRadius=3 * unit.R_earth,
    include_lorentz_boost=False,
    verbose=True,
)

gradient_result_with_boost = halo.findGradientsOverTime(
    stateNames=states_to_check,
    station=Mainz,
    meas_times=meas_times,
    truncRadius=3 * unit.R_earth,
    include_lorentz_boost=True,
    verbose=True,
)

components = [
    (
        "grad_r",
        "\\partial_r \\phi",
        "\\Omega_a^r",
    ),
    (
        "grad_theta",
        "\\frac{1}{r}\\partial_\\theta \\phi",
        "\\Omega_a^\\theta",
    ),
    (
        "grad_phi",
        "\\frac{1}{r\\!\\sin\\theta}\\partial_\\varphi\\phi",
        "\\Omega_a^\\phi",
    ),
]

Omega_factor = (
    const.c
    * halo.g_aNN
    * np.sqrt(halo.N_a * const.hbar**3 * const.c / (2 * halo.m_a))
)

fig = plt.figure(figsize=(8.5 / 2.54, 8.5 / 2.54), dpi=300)
gs = gridspec.GridSpec(3, 1)
fig.subplots_adjust(left=0.22, bottom=0.12, right=0.67, top=0.93, hspace=0.12)

ax_r = fig.add_subplot(gs[0])
ax_theta = fig.add_subplot(gs[1], sharex=ax_r, sharey=ax_r)
ax_phi = fig.add_subplot(gs[2], sharex=ax_r, sharey=ax_r)
gradient_axes = [ax_r, ax_theta, ax_phi]
Omega_axes = [ax.twinx() for ax in gradient_axes]
Omega_scales = []

for gradient_ax, Omega_ax, component in zip(
    gradient_axes, Omega_axes, components
):
    gradient_key, gradient_label, Omega_label = component
    gradient_no_boost = gradient_result_no_boost[gradient_key]
    gradient_with_boost = gradient_result_with_boost[gradient_key]

    gradient_ax.plot(
        t_hours,
        gradient_no_boost,
        color="tab:blue",
        linestyle="--",
        # linewidth=1.4,
        label="Without Lorentz boost",
        zorder=7,
    )
    gradient_ax.plot(
        t_hours,
        gradient_with_boost,
        color="tab:green",
        linestyle="-",
        # linewidth=1.4,
        label="With Lorentz boost",
    )

    gradient_unit = gradient_no_boost.unit.to_string("latex_inline")[1:-1]
    Omega_scale = (
        Omega_factor * (1 * gradient_no_boost.unit)
    ).to_value(unit.mHz, equivalencies=unit.dimensionless_angles())
    Omega_scales.append(Omega_scale)

    gradient_ax.set_ylabel(
        f"${gradient_label}$\n$\\left({gradient_unit}\\right)$",
        color="black",
        # rotation=0,
        loc="center",
        labelpad=22,
    )
    Omega_ax.set_ylabel(
        f"${Omega_label}\\, \\left(\\mathrm{{mHz}}\\right)$",
        color="black",
    )
    gradient_ax.tick_params(axis="y", colors="black")
    Omega_ax.tick_params(axis="y", colors="black")

gradient_axes[0].legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    frameon=False,
    fontsize=7,
)

for ax in [ax_r, ax_theta]:
    plt.setp(ax.get_xticklabels(), visible=False)

ax_phi.set_xlabel("Time (hour) from 2022-12-13 07:00 CET")
fig.suptitle(
    f"Mainz  ·  {states_to_check[0]} state  ·  "
    f"$\\nu_a={halo.nu_a.to_value(unit.MHz):.6f}$ MHz"
)

gradient_ylim_abs = 0
for ax in gradient_axes:
    ylim_bottom, ylim_top = ax.get_ylim()
    gradient_ylim_abs = np.amax(
        np.abs([gradient_ylim_abs, ylim_bottom, ylim_top])
    )
for ax in gradient_axes:
    ax.set_ylim(-gradient_ylim_abs, gradient_ylim_abs)

for gradient_ax, Omega_ax, Omega_scale in zip(
    gradient_axes, Omega_axes, Omega_scales
):
    gradient_ylim = np.array(gradient_ax.get_ylim())
    Omega_ax.set_ylim(*(gradient_ylim * Omega_scale))

fig.tight_layout()
plt.show()
