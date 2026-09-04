"""Plot the real part of several Earth-bound radial eigenfunctions.

This example solves the 1s, 2s, 2p, 3s, 3p, and 3d states and overlays the
real part of their radial wavefunctions ``R_r`` on a single plot.
"""

from axionbloch.dependency import unit, plt
from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo

halo = EarthBoundAxionHalo(
    # specify axion Compton frequency
    nu_a=1e-0 * 1.0 * unit.MHz,
    # you can use a fixed axion mass instead of nu_a, e.g.
    # m_a=10**(-11.5) * unit.eV / const.c**2,
    # number of grid points for the radial solver
    N=2**12,
    # radial extent of the solver grid
    extent=2**8 * unit.R_earth,
    verbose=True,
)

# Solve enough radial states for l = 0, 1, 2 to include:
# 1s, 2s, 2p, 3s, 3p, 3d.
halo.solve_TISE_3D(
    # l_vals=[0, 1, 2],
    l_vals=[1],
    max_n_r=20,
    verbose=False,
)

# state_names = ["1s", "2s", "2p", "3s", "3p", "3d"]
# state_names = ["2s"]
state_names = ["2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "10p"]
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, ax = plt.subplots(figsize=(8.5 / 2.54, 5.5 / 2.54), dpi=300)

start_idx = halo.N // 2 - 1  # avoid the r=0 singularity

for idx, name in enumerate(state_names):
    state = halo.states[name]
    ax.plot(
        halo.r[start_idx:].to_value(unit.R_earth),
        state["u_r"][start_idx:].real,
        label=name,
        color=colors[idx % len(colors)],
        linewidth=1.4,
    )

ax.set_xlabel("r (Earth radius)")
ax.set_ylabel("$R_r$")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.set_xlim(-0.02, 2.2)
fig.tight_layout()
plt.show()

# output_dir = Path(__file__).resolve().parent / "outputs"
# output_dir.mkdir(parents=True, exist_ok=True)
# output_path = output_dir / "EarthHalo-wavefunctions-1s-3d-real-Rr.pdf"
# fig.savefig(output_path, dpi=300, bbox_inches="tight")

# print(f"Saved figure to {output_path}")
