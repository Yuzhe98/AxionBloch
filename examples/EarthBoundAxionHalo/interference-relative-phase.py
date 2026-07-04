"""Scan the relative phase of a coherent 2p-3p superposition."""

from pathlib import Path

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

meas_time = Time("2022-12-14T12:00:00")
gradient_by_state = {}
for state_name in ("2p", "3p"):
    _, _, r_line, grad_r, grad_theta, grad_phi = halo.findGradients(
        stateCoefficients={state_name: 1.0},
        station=Mainz,
        meas_time=meas_time,
        truncRadius=3 * unit.R_earth,
        include_lorentz_boost=True,
        showPlot=False,
    )
    station_index = np.argmin(np.abs(r_line - Mainz.R))
    gradient_by_state[state_name] = {
        "grad_r": grad_r[station_index],
        "grad_theta": grad_theta[station_index],
        "grad_phi": grad_phi[station_index],
    }

Omega_factor = (
    const.c
    * halo.g_aNN
    * np.sqrt(halo.N_a * const.hbar**3 * const.c / (2 * halo.m_a))
)
relative_phases = np.linspace(0, 2 * np.pi, 181)
normalization = 1 / np.sqrt(2)

components = [
    ("grad_r", "\\Omega_a^r"),
    ("grad_theta", "\\Omega_a^\\theta"),
    ("grad_phi", "\\Omega_a^\\varphi"),
]
fig, axes = plt.subplots(
    3,
    1,
    figsize=(8.5 / 2.54, 9.5 / 2.54),
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
    combined_gradient = normalization * (
        gradient_2p
        + np.exp(1j * relative_phases) * gradient_3p
    )
    Omega = (Omega_factor * np.abs(combined_gradient)).to(
        unit.mHz,
        equivalencies=unit.dimensionless_angles(),
    )

    ax.plot(relative_phases / np.pi, Omega, color="tab:purple")
    ax.set_ylabel(f"${Omega_label}$\n$\\left(\\mathrm{{mHz}}\\right)$")
    ax.axvline(1.0, color="black", linestyle=":", linewidth=0.8)

axes[-1].set_xlabel("Relative phase $\\delta/\\pi$")
fig.suptitle(
    "Interference of "
    "$\\left(2p+e^{i\\delta}3p\\right)/\\sqrt{2}$ at Mainz"
)
fig.tight_layout()
output_directory = Path(__file__).with_name("outputs")
output_directory.mkdir(exist_ok=True)
output_path = output_directory / "interference-relative-phase.png"
fig.savefig(output_path, bbox_inches="tight")
print("Saved", output_path)
plt.show()
