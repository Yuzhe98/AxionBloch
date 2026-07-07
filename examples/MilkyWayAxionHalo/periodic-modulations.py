"""Plot MW axion-wind periodic modulations at Boston.

This example intentionally obtains the laboratory speed and wind-orientation
modulations from :mod:`axionbloch.MilkyWayAxionHalo`, rather than retyping the
closed-form approximations in the plotting script.
"""

from pathlib import Path
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as unit
from astropy.time import Time
from astropy.utils import iers

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from axionbloch.MilkyWayAxionHalo import MilkyWayAxionHalo
from axionbloch.Station import Boston

iers.conf.auto_download = False

FIGURE_DIR = Path(__file__).resolve().parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

station = Boston
axion = MilkyWayAxionHalo(nu_a=1.0 * unit.MHz)

# Use the default astropy Galactocentric frame.  This is more appropriate for
# calculations than the simplified Gramolin reproduction convention
# v_sun = 233 km/s, because it includes the Sun's peculiar velocity and a
# modern Galactic rotation speed.

# Annual modulation of the lab speed.  The vectorized projectHaloVelocity call
# uses the same astropy-based halo/lab transformation as findKinematicsOverTime.
annual_days = np.arange(365)
annual_times = Time("2022-01-01T00:00:00", scale="utc") + annual_days * unit.day
v_lab_annual = MilkyWayAxionHalo.projectHaloVelocity(
    time=annual_times,
    station=station,
    axis="magnitude",
)

# Daily modulation on January 1, following the time origin used in
# reporduce-periodic-modulations-in-Gramolin-paper.py.  This is the convention
# that makes the library result directly comparable to Eqs. (35)-(37) there.
daily_hours = np.linspace(0, 24, 145)
daily_times = Time("2022-01-01T00:00:00", scale="utc") + daily_hours * unit.hour

axis_styles = {
    "north": {"label": "North", "color": "red", "linestyle": "-"},
    "west": {"label": "West", "color": "blue", "linestyle": "--"},
    "zenith": {"label": "Zenith", "color": "green", "linestyle": "-."},
}

daily_kinematics = {
    axis: axion.findKinematicsOverTime(
        station=station,
        meas_times=daily_times,
        sensitive_axis=axis,
        include_rotation=True,
    )
    for axis in axis_styles
}

P_parallel = {}
P_perp = {}
for axis in axis_styles:
    alpha = daily_kinematics[axis]["wind_angle"]
    v_lab_daily = daily_kinematics[axis]["v_lab_magnitude"]
    P_parallel[axis] = axion.gradientPowerCoefficient(
        v_0=axion.v_0,
        v_lab=v_lab_daily,
        alpha=alpha,
        case="grad_par",
    )
    P_perp[axis] = axion.gradientPowerCoefficient(
        v_0=axion.v_0,
        v_lab=v_lab_daily,
        alpha=alpha,
        case="grad_perp",
    )

# Use one common scaling factor for all three orientations in each power panel.
P_parallel_max = max(np.max(values) for values in P_parallel.values())
P_perp_max = max(np.max(values) for values in P_perp.values())
P_parallel_normalized = {
    axis: (values / P_parallel_max).to_value(unit.one)
    for axis, values in P_parallel.items()
}
P_perp_normalized = {
    axis: (values / P_perp_max).to_value(unit.one)
    for axis, values in P_perp.items()
}

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14 / 2.54, 13 / 2.54),
    dpi=300,
)
fig.patch.set_facecolor("white")
ax_speed, ax_cos_alpha = axes[0]
ax_P_parallel, ax_P_perp = axes[1]

ax_speed.plot(
    annual_days,
    v_lab_annual.to_value(unit.km / unit.s),
    color="black",
)
ax_speed.set_ylabel("$v_\\mathrm{lab}$ (km/s)")
ax_speed.set_xlim(0, 364)
ax_speed.set_xlabel("Date in 2022")
month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
ax_speed.set_xticks(month_starts)
ax_speed.set_xticklabels(
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

for axis, style in axis_styles.items():
    # findKinematicsOverTime reports cos(alpha) for the axion wind direction,
    # i.e. -v_lab.  The Gramolin Fig. 3 convention plots the angle relative to
    # v_lab itself, so the sign is flipped here.
    line_kwargs = {
        "label": style["label"],
        "color": style["color"],
        "linestyle": style["linestyle"],
    }
    ax_cos_alpha.plot(
        daily_hours,
        -daily_kinematics[axis]["cos_alpha"],
        **line_kwargs,
    )
    ax_P_parallel.plot(
        daily_hours,
        P_parallel_normalized[axis],
        **line_kwargs,
    )
    ax_P_perp.plot(
        daily_hours,
        P_perp_normalized[axis],
        **line_kwargs,
    )

ax_cos_alpha.axhline(0, color="0.65", linewidth=0.6, zorder=0)
ax_cos_alpha.set_ylabel("$\\cos\\alpha$")
ax_cos_alpha.set_xlim(0, 24)
ax_cos_alpha.set_ylim(-1.05, 1.05)
ax_cos_alpha.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    fontsize=7,
    handlelength=2.5,
    columnspacing=1.0,
)

ax_P_parallel.set_ylabel("$P_\\parallel$ (arb. units)")
ax_P_perp.set_ylabel("$P_\\perp$ (arb. units)")
for ax in (ax_P_parallel, ax_P_perp):
    ax.set_ylim(0, 1.03)

for ax in (ax_cos_alpha, ax_P_parallel, ax_P_perp):
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel("Time on Jan. 1 (hour)")

for panel_label, ax in zip(("a", "b", "c", "d"), axes.flat):
    ax.text(
        -0.18,
        1.02,
        f"({panel_label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
    )

for ax in axes.flat:
    ax.grid(True, alpha=0.25)

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

fig.tight_layout(rect=(0, 0.08, 1, 0.94))

output_path = FIGURE_DIR / "MW-axion-annual-daily-modulations.pdf"
fig.savefig(output_path, facecolor="white", transparent=False)
plt.show()

print(f"Saved {output_path}")
print(
    "Annual speed range:",
    f"{np.min(v_lab_annual).to_value(unit.km / unit.s):.2f}",
    "to",
    f"{np.max(v_lab_annual).to_value(unit.km / unit.s):.2f}",
    "km/s",
)
