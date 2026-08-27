"""Plot the angularly averaged density profiles for several Earth-bound states.

This example solves the 1s, 2s, 2p, 3s, 3p, and 3d states at a fixed axion
mass and overlays the angularly averaged density profiles.

Using the reduced radial wavefunction ``u_r`` stored by the solver, the
angularly averaged density is approximated as

    rho(r) = M_DM * |u_r(r)|^2 / (4 pi r^2)

on a single plot.
"""

from pathlib import Path

from axionbloch.dependency import *
from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo

halo = EarthBoundAxionHalo(
    nu_a=1.348 * unit.MHz,  # axion Compton frequency in Hz
    # m_a=10 ** (-10.0) * unit.eV / const.c**2,
    N=2**12,
    extent=32 * unit.R_earth,
    verbose=True,
)

# Solve enough radial states for l = 0, 1, 2 to include:
# 1s, 2s, 2p, 3s, 3p, 3d.
halo.solve_TISE_3D(
    l_vals=[0, 1, 2],
    max_n_r=3,
    verbose=False,
)
rhoE_DM_MW = 0.4 * unit.GeV / const.c**2 / unit.cm**3
state_names = ["1s", "2s", "2p", "3s", "3p", "3d"]
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
totalMassEnclosed = halo.totalMassEnclosed

print("\nNormalization check for u_r:")
print(f"{'state':<6} {'int |u_r|^2 dr':>18} {'|norm-1|':>14} {'rescaled':>12}")
print("-" * 42)
for name in state_names:
    u_r = halo.states[name]["u_r"]
    start_idx_norm = halo.N // 2 + 4
    norm = np.trapezoid(np.abs(u_r[start_idx_norm:]) ** 2, halo.r[start_idx_norm:])
    deviation = np.abs(norm - 1.0)
    rescaled = (
        "yes"
        if (not np.isfinite(norm.value) or norm.value <= 0 or deviation.value > 1e-2)
        else ""
    )
    print(f"{name:<6} {norm.value:18.10e} {deviation.value:14.3e} {rescaled:>12}")

fig, ax = plt.subplots(figsize=(8.5 / 2.54, 5.5 / 2.54), dpi=300)

start_idx = halo.N // 2 + 4  # avoid the r=0 singularity
rho_to_rhoMW = {}
rho_gcm3 = {}

for idx, name in enumerate(state_names):
    state = halo.states[name]
    u_r = state["u_r"]
    start_idx_norm = halo.N // 2 + 4
    norm = np.trapezoid(np.abs(u_r[start_idx_norm:]) ** 2, halo.r[start_idx_norm:])
    if np.isfinite(norm.value) and norm.value > 0:
        u_r = u_r / np.sqrt(norm)
    rho_r = totalMassEnclosed * np.abs(u_r) ** 2 / (4 * np.pi * halo.r**2)
    rho_r = np.where(np.isfinite(rho_r), rho_r, 0 * rho_r.unit)
    rho_to_rhoMW[name] = (rho_r / rhoE_DM_MW).to_value(unit.one)
    # rho_gcm3[name] = rho_r.to_value(unit.g / unit.cm**3)
    ax.plot(
        halo.r[start_idx:].to_value(unit.R_earth),
        rho_to_rhoMW[name][start_idx:],
        label=name,
        color=colors[idx % len(colors)],
        linewidth=1.4,
    )

ax.axvline(
    x=1.0,
    color="k",
    linestyle="dotted",
    linewidth=1,
    alpha=0.8,
    label="Earth radius",
)

ax.set_xlabel("r (Earth radii)")
# ax.set_ylabel(r"$\rho(r)\,(\mathrm{GeV}/\mathrm{cm}^3)$")
ax.set_ylabel("$\\rho(r) / \\rho_{\\mathrm{MW}}$")
ax.set_yscale("log")
# ax.legend(loc="upper right", fontsize=8, ncol=2)

# ax2 = ax.twinx()
# ax2.set_yscale("log")
# ax2.set_ylim(
#     min(np.min(rho_gcm3[name][start_idx:]) for name in state_names),
#     max(np.max(rho_gcm3[name][start_idx:]) for name in state_names),
# )
# ax2.set_ylabel(r"$\rho(r)\,(\mathrm{g}/\mathrm{cm}^3)$")

fig.tight_layout()
plt.show()

# output_dir = Path(__file__).resolve().parent / "outputs"
# output_dir.mkdir(parents=True, exist_ok=True)
# output_path = output_dir / "EarthHalo-density-1s-3d.pdf"
# fig.savefig(output_path, dpi=300, bbox_inches="tight")

# print(f"Saved figure to {output_path}")
