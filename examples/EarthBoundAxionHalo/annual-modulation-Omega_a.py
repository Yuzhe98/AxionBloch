"""Plot annual modulation of 2p-state Omega_a with daily sampling.

The script evaluates the Rabi frequency once per day for 365 days. The
Lorentz-boost correction is included.
"""

from pathlib import Path
import textwrap

from astropy.utils import iers

from axionbloch.dependency import *
from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo
from axionbloch.Station import Mainz


# Use Astropy's bundled Earth-orientation table. Annual plotting should not
# pause repeatedly when an online IERS table is unavailable.
iers.conf.auto_download = False

halo = EarthBoundAxionHalo(
    nu_a=1.348585 * unit.MHz,
    N=2**11,
    extent=32 * unit.R_earth,
    g_aNN=1e-9 * unit.GeV**-1,
    verbose=True,
)

# The first l=1 solution is 2p, so only one radial state is required.
halo.solve_TISE_3D(l_vals=[1], max_n_r=1, verbose=False)

# Sample once per day for 365 days.
# 06:00 UTC is 07:00 CET on 2022-12-13, matching the reference script while
# avoiding a runtime dependency on the external timezone database.
t0 = Time("2022-01-01T00:00:00", scale="utc")
meas_times = t0 + np.arange(365) * unit.day

# For the single 2p state, m = 0 and
#
#   Y_1^0(theta) = sqrt(3 / (4 pi)) cos(theta).
#
# Evaluating this expression and its derivative directly avoids rebuilding the
# same 3-D angular interpolation grid 365 times. The radial wavefunction still
# comes from the numerical TISE solution, and the boost term is identical to
# the one in GravBoundAxionHalo.findGradientsAtDirection.
state = halo.states["2p"]
start_index = halo.N // 2 + 5
r_positive = halo.r[start_index:]
R_positive = state["R_r"][start_index:]
dR_dr_positive = np.gradient(R_positive, r_positive)
r_surface = 1 * unit.R_earth
R_surface = (
    np.interp(
        r_surface.to_value(r_positive.unit),
        r_positive.to_value(r_positive.unit),
        R_positive.value,
    )
    * R_positive.unit
)
dR_dr_surface = (
    np.interp(
        r_surface.to_value(r_positive.unit),
        r_positive.to_value(r_positive.unit),
        dR_dr_positive.value,
    )
    * dR_dr_positive.unit
)

angular_frequency = (
    (halo.m_a * const.c**2 + state["eigenE_expect"]) / const.hbar
).to(
    1 / unit.s,
    equivalencies=unit.dimensionless_angles(),
)
Y_normalization = np.sqrt(3 / (4 * np.pi))

gradient_components = {
    "grad_r": [],
    "grad_theta": [],
    "grad_phi": [],
}

for meas_time in meas_times:
    position_gcrs, velocity_gcrs = Mainz.location.get_gcrs_posvel(meas_time)
    solarZ_basis = Mainz._solarZ_basis(meas_time)
    position = (
        solarZ_basis.T @ position_gcrs.xyz.to_value(unit.m)
    ) * unit.m
    velocity = (
        solarZ_basis.T
        @ velocity_gcrs.xyz.to_value(unit.m / unit.s)
    ) * unit.m / unit.s

    position_norm = np.linalg.norm(position)
    theta_value = np.arccos(
        np.clip(
            (position[2] / position_norm).to_value(unit.one),
            -1,
            1,
        )
    )
    phi_value = np.arctan2(position[1].value, position[0].value)
    sin_theta, cos_theta = np.sin(theta_value), np.cos(theta_value)
    sin_phi, cos_phi = np.sin(phi_value), np.cos(phi_value)

    Y_10 = Y_normalization * cos_theta
    dY_10_dtheta = -Y_normalization * sin_theta
    wavefunction = R_surface * Y_10

    grad_r = dR_dr_surface * Y_10
    grad_theta = R_surface * dY_10_dtheta / r_surface
    grad_phi = 0 * grad_theta

    velocity_r = (
        velocity[0] * sin_theta * cos_phi
        + velocity[1] * sin_theta * sin_phi
        + velocity[2] * cos_theta
    )
    velocity_theta = (
        velocity[0] * cos_theta * cos_phi
        + velocity[1] * cos_theta * sin_phi
        - velocity[2] * sin_theta
    )
    velocity_phi = -velocity[0] * sin_phi + velocity[1] * cos_phi

    boost_scale = -1j * angular_frequency * wavefunction / const.c**2
    with unit.add_enabled_equivalencies(unit.dimensionless_angles()):
        grad_r = grad_r + boost_scale * velocity_r
        grad_theta = grad_theta + boost_scale * velocity_theta
        grad_phi = grad_phi + boost_scale * velocity_phi

    gradient_components["grad_r"].append(grad_r)
    gradient_components["grad_theta"].append(grad_theta)
    gradient_components["grad_phi"].append(grad_phi)

gradient_components = {
    key: unit.Quantity(values)
    for key, values in gradient_components.items()
}

Omega_factor = (
    const.c
    * halo.g_aNN
    * np.sqrt(halo.N_a * const.hbar**3 * const.c / (2 * halo.m_a))
)
Omega_results = {
    "Omega_a_r": Omega_factor * np.abs(gradient_components["grad_r"]),
    "Omega_a_theta": Omega_factor * np.abs(
        gradient_components["grad_theta"]
    ),
    "Omega_a_phi": Omega_factor * np.abs(gradient_components["grad_phi"]),
}


components = [
    ("Omega_a_r", "\\Omega_a^r"),
    ("Omega_a_theta", "\\Omega_a^\\theta"),
    ("Omega_a_phi", "\\Omega_a^\\varphi"),
]

fig, axes = plt.subplots(
    3,
    1,
    figsize=(14 / 2.54, 8 / 2.54),
    dpi=300,
    sharex=True,
    sharey=True,
)

for ax, (Omega_key, Omega_label) in zip(axes, components):
    Omega = Omega_results[Omega_key].to(
        unit.mHz,
        equivalencies=unit.dimensionless_angles(),
    )
    ax.plot(
        meas_times.datetime,
        Omega,
        color="tab:blue",
        linestyle=":",
        label="$2p$",
        zorder=2,
    )
    ax.set_ylabel(f"${Omega_label}$\n$\\left(\\mathrm{{mHz}}\\right)$")

axes[0].legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    frameon=False,
    fontsize=7,
)
axes[-1].set_xlabel("Date")

# Watermark follows examples/example_matplotlib.py.
script_path = Path(__file__).resolve()
figure_width_points = fig.get_size_inches()[0] * 72
approximate_character_width_points = 0.6 * 4
maximum_characters = int(
    figure_width_points / approximate_character_width_points
)
watermark = textwrap.fill(
    f"Generated by: {script_path}",
    width=maximum_characters,
)
fig.text(
    0.02,
    0.01,
    watermark,
    fontsize=6,
    color="gray",
    wrap=True,
)

fig.tight_layout(rect=(0, 0.04, 1, 1))
output_directory = Path(__file__).with_name("outputs")
output_directory.mkdir(exist_ok=True)
output_path = (
    output_directory
    / "EarthHalo-annual-modulation-Omega_a-2p-365-days.pdf"
)
fig.savefig(output_path)
print("Saved", output_path)
plt.show()
