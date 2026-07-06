"""Reproduce the four periodic-modulation panels in Gramolin spectral signature paper."""

from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as unit

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from axionbloch.MilkyWayAxionHalo import MilkyWayAxionHalo
from axionbloch.Station import Boston

FIGURE_DIR = Path(__file__).resolve().parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

station = Boston
axion = MilkyWayAxionHalo(nu_a=1.0 * unit.MHz)

# Equation (34) of Phys. Rev. D 105, 035029 (2022). The time origin is
# January 1; the maximum near June 1 occurs at tau = t_y + 72.4 days.
annual_days = np.linspace(0, 365, 365 + 1)
v_sun = 233 * unit.km / unit.s
v_earth = 29.8 * unit.km / unit.s
eta = 0.982
omega_year = 2 * np.pi / 365
vernal_equinox_day = 31 + 28 + 19
tau = vernal_equinox_day + 72.4
v_lab_annual = np.sqrt(
    v_sun**2
    + v_earth**2
    + eta * v_sun * v_earth * np.cos(omega_year * (annual_days - tau))
)

# Equations (35)-(37) for January 1 at Boston University. The Station object
# supplies the signed latitude and longitude specified in the figure caption.
daily_hours = np.linspace(0, 24, 145)
daily_days = daily_hours / 24
latitude = station.location.lat.to_value(unit.rad)
longitude = station.location.lon.to_value(unit.rad)
b_0 = 0.7589
b_1 = 0.6512
psi = -3.5336
phase = longitude + psi
omega_day = 2 * np.pi / 0.9973
daily_argument = omega_day * daily_days + phase

cos_alpha = {
    "North": (b_0 * np.cos(latitude) - b_1 * np.sin(latitude) * np.cos(daily_argument)),
    "West": b_1 * np.sin(daily_argument),
    "Zenith": (
        b_0 * np.sin(latitude) + b_1 * np.cos(latitude) * np.cos(daily_argument)
    ),
}
styles = {
    "North": {"color": "red", "linestyle": "-"},
    "West": {"color": "blue", "linestyle": "--"},
    "Zenith": {"color": "green", "linestyle": "-."},
}

P_parallel = {}
P_perp = {}
v_lab_january_1 = np.full_like(daily_hours, v_lab_annual[0].value) * v_lab_annual.unit
for label in cos_alpha:
    alpha = np.arccos(np.clip(cos_alpha[label], -1, 1)) * unit.rad
    P_parallel[label] = axion.gradientPowerCoefficient(
        v_0=axion.v_0,
        v_lab=v_lab_january_1,
        alpha=alpha,
        case="grad_par",
    )
    P_perp[label] = axion.gradientPowerCoefficient(
        v_0=axion.v_0,
        v_lab=v_lab_january_1,
        alpha=alpha,
        case="grad_perp",
    )

# Use one common scaling factor for all three orientations in each power panel.
P_parallel_max = max(np.max(values) for values in P_parallel.values())
P_perp_max = max(np.max(values) for values in P_perp.values())
P_parallel_normalized = {
    label: (values / P_parallel_max).to_value(unit.one)
    for label, values in P_parallel.items()
}
P_perp_normalized = {
    label: (values / P_perp_max).to_value(unit.one) for label, values in P_perp.items()
}

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14 / 2.54, 13 / 2.54),
    dpi=300,
)
fig.patch.set_facecolor("white")
ax_a, ax_b = axes[0]
ax_c, ax_d = axes[1]

ax_a.plot(
    annual_days,
    v_lab_annual.to_value(unit.km / unit.s),
    color="black",
)
ax_a.set_ylabel("$v_\\mathrm{lab}$ [km/s]")
month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
ax_a.set_xticks(month_starts)
ax_a.set_xticklabels(
    [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ],
    rotation=60,
    ha="center",
)
ax_a.set_xlim(0, 365)

for label in cos_alpha:
    ax_b.plot(
        daily_hours,
        cos_alpha[label],
        label=label,
        **styles[label],
    )
    ax_c.plot(
        daily_hours,
        P_parallel_normalized[label],
        label=label,
        **styles[label],
    )
    ax_d.plot(
        daily_hours,
        P_perp_normalized[label],
        label=label,
        **styles[label],
    )

ax_b.axhline(0, color="0.65", linewidth=0.6, zorder=0)
ax_b.set_ylabel("$\\cos\\alpha$")
ax_b.set_ylim(-1.05, 1.05)

ax_c.set_ylabel("$P_\\parallel$ [arb. units]")
ax_d.set_ylabel("$P_\\perp$ [arb. units]")
for ax in (ax_c, ax_d):
    ax.set_ylim(0, 1.03)

for ax in (ax_b, ax_c, ax_d):
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel("Time [hour]")

ax_d.legend(loc="lower right", frameon=False, fontsize=7)

for panel_label, ax in zip(("a", "b", "c", "d"), axes.flat):
    ax.text(
        -0.18,
        1.02,
        f"({panel_label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
    )

caption = "FIG. 3. Periodic modulations of $v_\\mathrm{lab}$ and of the signal power in the case of gradient coupling. (a) Annual modulation of $v_\\mathrm{lab}$ due to the Earth's orbital motion around the Sun. (b) Daily modulations of $\\cos\\alpha$ for the three orthogonal orientations of an external magnetic field: towards the north (solid red line), towards the west (dashed blue line), and towards the zenith (dash-dotted green line). (c) Daily modulations of the total signal power $P_\\parallel$ for the three orientations of an external magnetic field. The signal power is normalized such that the maximum value is 1 (the scaling factor is the same for the three cases). (d) Similar to panel (c) but for the power $P_\\perp$. For panels (b)-(d), we assume that the location is the Metcalf Science Center of Boston University ($\\lambda_\\mathrm{lab}=42.3484^{\\circ}$, $\\phi_\\mathrm{lab}=-71.1002^{\\circ}$) and the date is January 1 (from 00:00 to 24:00 in local time)."
fig.text(
    0.02,
    0.035,
    textwrap.fill(caption, width=140),
    ha="left",
    va="bottom",
    fontsize=8,
)

# Watermark follows examples/example_matplotlib.py.
script_path = Path(__file__).resolve()
figure_width_points = fig.get_size_inches()[0] * 72
approximate_character_width_points = 0.6 * 4
maximum_characters = int(figure_width_points / approximate_character_width_points)
watermark = textwrap.fill(
    f"Generated by: {script_path}",
    width=maximum_characters,
)
fig.text(
    0.02,
    0.008,
    watermark,
    fontsize=6,
    color="gray",
    wrap=True,
)

fig.subplots_adjust(
    left=0.12,
    right=0.98,
    top=0.97,
    bottom=0.34,
    wspace=0.32,
    hspace=0.38,
)

output_path = FIGURE_DIR / "MW-axion-periodic-modulations.pdf"
fig.savefig(output_path, facecolor="white", transparent=False)
plt.close(fig)

print(f"Saved {output_path}")
print(
    "Annual speed range:",
    f"{np.min(v_lab_annual).to_value(unit.km / unit.s):.2f}",
    "to",
    f"{np.max(v_lab_annual).to_value(unit.km / unit.s):.2f}",
    "km/s",
)
