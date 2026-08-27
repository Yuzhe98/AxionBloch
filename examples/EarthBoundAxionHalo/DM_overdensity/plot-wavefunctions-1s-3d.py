"""Plot the real part of several Earth-bound radial eigenfunctions.

This example solves the 1s, 2s, 2p, 3s, 3p, and 3d states and overlays the
real part of their radial wavefunctions ``R_r`` on a single plot.
"""

from pathlib import Path

from axionbloch.dependency import *
from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo


halo = EarthBoundAxionHalo(
    nu_a=1e-1 * 1. * unit.MHz,  # axion Compton frequency in Hz
    # m_a=10**(-11.5) * unit.eV / const.c**2,
    N=2**10,
    extent=2**5 * unit.R_earth,
    verbose=True,
)

# Solve enough radial states for l = 0, 1, 2 to include:
# 1s, 2s, 2p, 3s, 3p, 3d.
halo.solve_TISE_3D(
    l_vals=[0, 1, 2],
    max_n_r=3,
    verbose=False,
)

state_names = ["1s", "2s", "2p", "3s", "3p", "3d"]
# state_names = ["2s"]
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, ax = plt.subplots(figsize=(8.5 / 2.54, 5.5 / 2.54), dpi=300)

start_idx = halo.N // 2 + 1  # avoid the r=0 singularity

for idx, name in enumerate(state_names):
    state = halo.states[name]
    ax.plot(
        halo.r[start_idx:].to_value(unit.R_earth),
        state["R_r"][start_idx:].real,
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
ax.set_ylabel("$R_r$")
# ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.set_xlim(-0.02, 2.2)
fig.tight_layout()
plt.show()

# output_dir = Path(__file__).resolve().parent / "outputs"
# output_dir.mkdir(parents=True, exist_ok=True)
# output_path = output_dir / "EarthHalo-wavefunctions-1s-3d-real-Rr.pdf"
# fig.savefig(output_path, dpi=300, bbox_inches="tight")

# print(f"Saved figure to {output_path}")
