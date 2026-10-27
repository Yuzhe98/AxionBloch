"""Plot Rabi-frequency modulation for every m substate of one nl state.

Run from the repository root with:
    python examples/EarthBoundAxionHalo/find-interesting-states/rabi_m_states.py

The quantization axis is the solar-Z axis used by Station.in_solarZ_frame:
z points from Earth to Sun at each epoch. Mass is inferred independently for
the selected (n,l) radial state, then used for each allowed m substate.
"""

from pathlib import Path
from datetime import datetime
from datetime import timedelta, timezone

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as unit
from astropy.time import Time

from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo
from axionbloch.Station import Mainz
from axionbloch.utils import high_contrast_extended


# Select a solved-state label such as "2p", "3d", or "4f".
state_to_plot = "5g"

orbital_labels = {label: l for l, label in enumerate(EarthBoundAxionHalo.orbitalLabels)}
if len(state_to_plot) < 2 or not state_to_plot[:-1].isdigit() or state_to_plot[-1] not in orbital_labels:
    raise ValueError("state_to_plot must look like a state label such as '2p' or '3d'.")
n = int(state_to_plot[:-1])
l = orbital_labels[state_to_plot[-1]]
if n <= l:
    raise ValueError(f"{state_to_plot} is not a valid state label: require n > l.")

# Match param_1MHz in DM_overdensity/density_by_shell_mass_sweep.py, with the
# requested axion frequency. The preset grid is retained at this frequency.
halo = EarthBoundAxionHalo(
    nu_a=1.348575 * unit.MHz,
    N=int(7.5e6),
    extent=1e4 * unit.R_earth,
)
halo.solve_TISE_3D(l_vals=[l], max_n_r=n - l)

# inferHaloMass returns one (n,l) mass hypothesis, independent of m. Use the
# same shell upper limit and radial boundaries as the density sweep. Its
# default maxEarthEnclosedMass and the constructor's default maxTotalDMmass
# remain active.
shell_mass_limit = 4e-9 * unit.M_earth
shell_radii = [12300 * unit.km, 384000 * unit.km]
mass_inference = halo.inferHaloMass(shell_mass_limit, shell_radii, [state_to_plot])[state_to_plot]
halo_mass = mass_inference["total_mass"]
halo.N_a = halo_mass / halo.m_a

# Match the 72-hour time grid and Berlin-local start used by
# EH_ALP-2p-daily_modulation-gradient-Omega_a.py.
t0 = Time(datetime(2022, 12, 13, 7, 0, tzinfo=timezone(timedelta(hours=1))))
t_hours = np.linspace(0, 72, 144) * unit.hour
meas_times = t0 + t_hours

# Use the built-in gradient and Rabi routines with each allowed m value.
# findGradientsOverTime returns the complex gradient phasors; the Rabi method
# converts their magnitudes to real amplitudes and applies the Lorentz boost.
omega_factor = (
    const.c
    * halo.g_aNN
    * np.sqrt(halo.N_a * const.hbar**3 * const.c / (2 * halo.m_a))
)
m_values = list(range(-l, l + 1))
components = {
    "r": ("grad_r", "Omega_a_r"),
    "theta": ("grad_theta", "Omega_a_theta"),
    "phi": ("grad_phi", "Omega_a_phi"),
}
gradients_no_boost = {m: {} for m in m_values}
omega_no_boost = {m: {} for m in m_values}
omega_with_boost = {m: {} for m in m_values}
for m in m_values:
    mode_coefficients = {(state_to_plot, m): 1.0}
    gradients = halo.findGradientsOverTime(
        stateCoefficients=mode_coefficients,
        station=Mainz,
        meas_times=meas_times,
        truncRadius=3 * unit.R_earth,
        include_lorentz_boost=False,
    )
    boosted_omega = halo.findOmega_aOverTime(
        stateCoefficients=mode_coefficients,
        station=Mainz,
        meas_times=meas_times,
        truncRadius=3 * unit.R_earth,
        g_aNN=halo.g_aNN,
        include_lorentz_boost=True,
    )
    for component, (gradient_key, omega_key) in components.items():
        gradient = gradients[gradient_key]
        gradients_no_boost[m][component] = gradient.to(unit.R_earth ** (-2.5))
        omega_no_boost[m][component] = (
            omega_factor * np.abs(gradient)
        ).to(unit.Hz, equivalencies=unit.dimensionless_angles())
        omega_with_boost[m][component] = boosted_omega[omega_key]

print(f"{state_to_plot} inferred total halo mass: {halo_mass.to(unit.M_earth):.3g}", flush=True)
print("RMS Rabi-frequency amplitudes at Mainz (Lorentz boost included):", flush=True)
for m in m_values:
    component_rms = {component: np.sqrt(np.mean(omega_with_boost[m][component] ** 2)) for component in components}
    omega_theta_phi_no_boost = np.sqrt(
        omega_no_boost[m]["theta"] ** 2 + omega_no_boost[m]["phi"] ** 2
    )
    omega_theta_phi_with_boost = np.sqrt(
        omega_with_boost[m]["theta"] ** 2 + omega_with_boost[m]["phi"] ** 2
    )
    rms_omega_theta_phi_no_boost = np.sqrt(
        np.mean(omega_theta_phi_no_boost**2)
    )
    rms_omega_theta_phi_with_boost = np.sqrt(
        np.mean(omega_theta_phi_with_boost**2)
    )
    total_rms = np.sqrt(sum(value**2 for value in component_rms.values()))
    print(
        f"m={m:+d}: Omega_r={component_rms['r'].to(unit.mHz):3g}, "
        f"Omega_theta={component_rms['theta'].to(unit.mHz):3g}, "
        f"Omega_phi={component_rms['phi'].to(unit.mHz):3g}, "
        f"Omega_theta_phi quadrature (no boost)={rms_omega_theta_phi_no_boost.to(unit.mHz):3g}, "
        f"Omega_theta_phi quadrature (with boost)={rms_omega_theta_phi_with_boost.to(unit.mHz):3g}, "
        f"quadrature total={total_rms.to(unit.mHz):3g}",
        flush=True,
    )

# Match the daily-modulation example: real spatial gradient without boost at
# left, real nonnegative Rabi amplitudes with/without boost at right.
fig, axes = plt.subplots(
    3,
    2,
    figsize=(13.0 / 2.54, 7.8 / 2.54),
    dpi=300,
    sharex="col",
)
fig.subplots_adjust(
    left=0.146,
    top=0.824,
    right=0.96,
    bottom=0.154,
    wspace=0.428,
    hspace=0.31,
)
gradient_colors = {
    m: high_contrast_extended[index % len(high_contrast_extended)]
    for index, m in enumerate(m_values)
}
rabi_colors = {
    (m, boost): high_contrast_extended[(2 * index + boost_index) % len(high_contrast_extended)]
    for index, m in enumerate(m_values)
    for boost_index, boost in enumerate(("no boost", "with boost"))
}
gradient_labels = {
    "r": r"\partial_r\psi",
    "theta": r"r^{-1}\partial_\theta\psi",
    "phi": r"(r\sin\theta)^{-1}\partial_\varphi\psi",
}
omega_labels = {"r": r"\Omega_a^r", "theta": r"\Omega_a^\theta", "phi": r"\Omega_a^\phi"}

for row, component in enumerate(components):
    gradient_ax, omega_ax = axes[row]
    for m in m_values:
        gradient_ax.plot(
            t_hours,
            gradients_no_boost[m][component].real.value,
            color=gradient_colors[m],
            label=f"m={m:+d}",
        )
        omega_ax.plot(
            t_hours,
            (omega_with_boost[m][component] / (2 * np.pi)).to_value(unit.mHz),
            color=rabi_colors[(m, "with boost")],
            linestyle="-",
            label=f"m={m:+d}, with boost",
            zorder=2,
        )
        omega_ax.plot(
            t_hours,
            (omega_no_boost[m][component] / (2 * np.pi)).to_value(unit.mHz),
            color=rabi_colors[(m, "no boost")],
            linestyle="--",
            label=f"m={m:+d}, no boost",
            linewidth=1.1,
            zorder=3,
        )
    gradient_unit = gradients_no_boost[m_values[0]][component].unit.to_string("latex_inline")[1:-1]
    gradient_ax.set_ylabel(
        f"${gradient_labels[component]}$\n$\\left({gradient_unit}\\right)$",
        color="black",
        loc="center",
        labelpad=10,
    )
    omega_ax.set_ylabel(
        f"${omega_labels[component]} / (2\\pi)$\n$(\\mathrm{{mHz}})$",
        color="black",
    )
    gradient_ax.tick_params(axis="y", colors="black")
    omega_ax.tick_params(axis="y", colors="black")

# Keep the reference example's top legend placement, with entries for all
# m-state and boost combinations represented in the Rabi panels.
handles, legend_labels = axes[0, 1].get_legend_handles_labels()
fig.legend(
    handles,
    legend_labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.84),
    ncol=3,
    frameon=False,
    fontsize=6,
)
axes[-1, 0].set_xlabel("Time (hour)")
axes[-1, 1].set_xlabel("Time (hour)")

# Follow the reference script's shared, fixed limits across all three gradient
# panels and all three Rabi panels. Derive them once from the full plotted data
# so no component gets clipped and comparisons retain a common scale.
gradient_ylim_abs = max(
    np.max(np.abs(gradients_no_boost[m][component].real.value))
    for m in m_values
    for component in components
)
gradient_ylim_abs *= 1.06
for ax in axes[:, 0]:
    ax.set_ylim(-gradient_ylim_abs, gradient_ylim_abs)

omega_ymax = max(
    np.max((omega[m][component] / (2 * np.pi)).to_value(unit.mHz))
    for omega in (omega_no_boost, omega_with_boost)
    for m in m_values
    for component in components
)
for ax in axes[:, 1]:
    ax.set_ylim(-0.06 * np.abs(omega_ymax), 1.06 * omega_ymax)

output_directory = Path(__file__).with_name("outputs")
output_directory.mkdir(exist_ok=True)
output_path = output_directory / f"EarthHalo-{state_to_plot}-rabi-m-states.png"
fig.savefig(output_path, bbox_inches="tight")
print("Saved", output_path)
plt.close(fig)
