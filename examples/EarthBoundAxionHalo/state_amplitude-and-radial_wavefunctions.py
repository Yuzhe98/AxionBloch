"""Combine the eigenstate spectrum and radial wavefunctions in a 1 x 2 figure.

Preserve each source example's frequency, grid, and state selection. Set
USE_MARKERS to True to add sparse markers from axionbloch.utils to the curves.
Run from the repository root with the axionbloch environment activated::

    python -m examples.EarthBoundAxionHalo.state-amplitude-and-radial-wavefunctions
"""
"""Plot eigenstate amplitudes against their eigen-energy shifts.

All states from 1s through 4f are included with equal input coefficients.
``plotStateAmplitudeVsEigenEnergy`` normalizes the coefficients before plotting,
so every displayed amplitude is 1 / sqrt(number of states).
"""

from pathlib import Path

from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo


from axionbloch.dependency import unit, plt
from axionbloch.utils import linestyles

halo = EarthBoundAxionHalo(
    # specify axion Compton frequency
    nu_a=1.0 * unit.MHz,
    # you can use a fixed axion mass instead of nu_a, e.g.
    # m_a=10**(-11.5) * unit.eV / const.c**2,
    # number of grid points for the radial solver
    N=2**12,
    # radial extent of the solver grid
    extent=2**8 * unit.R_earth,
    verbose=True,
)

# Solve enough radial states
halo.solve_TISE_3D(
    l_vals=[0, 1, 2, 3],
    max_n_r=20,
    verbose=False,
)
# Plot the radial wavefunctions
state_names = ["1s", "2s", "3s", "2p", "4s", "3d", "5s", "3p", "6s", "4f"]

# Equal coefficients provide an equal coherent superposition. The plotting
# method normalizes this dictionary, giving |c_nlm| = 1 / sqrt(10).
stateCoefficients = {
    "1s": 1,
    "2s": 1,
    "3s": 1,
    "2p": 1,
    "4s": 1,
    "3d": 1,
    "5s": 1,
    "3p": 1,
    "6s": 1,
    "4f": 1,
}


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, ax = plt.subplots(figsize=(13 / 2.54, 5.5 / 2.54), dpi=300)

start_idx = halo.N // 2 + 0

for idx, name in enumerate(state_names):
    state = halo.states[name]
    ax.plot(
        halo.r[start_idx:].to_value(unit.R_earth),
        state["u_r"][start_idx:].real,
        label=name,
        color=colors[idx % len(colors)],
        linestyle=linestyles[idx % len(linestyles)],
        linewidth=1.4,
    )

ax.set_xlabel("$r\\,(R_\\oplus$)")
ax.set_ylabel("$r\\,R(r)\\,(R_\\oplus^{-1/2})$")
# ax.legend(ncol=2, bbox_to_anchor=(1.0, 1.0))
ax.set_xlim(-0.02, 5.2)



fig, _, _, _ = halo.plotStateAmplitudeVsEigenEnergy(
    stateCoefficients=stateCoefficients,
    energy_unit=unit.attoelectronvolt,
    frequency_unit=unit.mHz,
    showPlot=False,
)

# output_dir = Path(__file__).resolve().parent / "outputs"
# output_dir.mkdir(parents=True, exist_ok=True)
# output_path = output_dir / "EarthHalo-state-amplitude-vs-eigenenergy-1s-to-4f.pdf"
# fig.savefig(output_path, dpi=300, bbox_inches="tight")
# print(f"Saved figure to {output_path}")

fig.tight_layout()
plt.show()

"""
halo = EarthBoundAxionHalo(
    # specify axion Compton frequency
    nu_a=1e-0 * 1.0 * unit.MHz,
    # number of grid points for the radial solver
    N=2**12,
    # radial extent of the solver grid
    extent=2**8 * unit.R_earth,
    verbose=True,
)
l_vals=[0, 1, 2, 3, 4, 5, 6]
        s  p  d  f  g  h  i
eigen-energies low to high: 

1s: E = -4.549e-18 eV
2s: E = -3.445e-18 eV
3s: E = -2.540e-18 eV
2p: E = -2.478e-18 eV
4s: E = -1.861e-18 eV
3d: E = -1.680e-18 eV
5s: E = -1.398e-18 eV
3p: E = -1.347e-18 eV
6s: E = -1.090e-18 eV
4f: E = -1.088e-18 eV

4d: E = -9.739e-19 eV
7s: E = -8.663e-19 eV
4p: E = -8.363e-19 eV
5g: E = -7.140e-19 eV
8s: E = -7.069e-19 eV
5f: E = -6.952e-19 eV
5d: E = -6.387e-19 eV
9s: E = -5.855e-19 eV
5p: E = -5.674e-19 eV
6h: E = -4.966e-19 eV

6g: E = -4.955e-19 eV
10s: E = -4.937e-19 eV
6f: E = -4.837e-19 eV
6d: E = -4.514e-19 eV
11s: E = -4.211e-19 eV
6p: E = -4.095e-19 eV
7i: E = -3.649e-19 eV
7h: E = -3.649e-19 eV
7g: E = -3.640e-19 eV
12s: E = -3.638e-19 eV
7f: E = -3.562e-19 eV
7d: E = -3.360e-19 eV
13s: E = -3.170e-19 eV
7p: E = -3.092e-19 eV
8i: E = -2.794e-19 eV
8h: E = -2.794e-19 eV
14s: E = -2.789e-19 eV
8g: E = -2.787e-19 eV
8f: E = -2.734e-19 eV
8d: E = -2.598e-19 eV
15s: E = -2.471e-19 eV
8p: E = -2.417e-19 eV
9i: E = -2.207e-19 eV
9h: E = -2.207e-19 eV
16s: E = -2.206e-19 eV
9g: E = -2.202e-19 eV
9f: E = -2.164e-19 eV
9d: E = -2.069e-19 eV
17s: E = -1.980e-19 eV
9p: E = -1.940e-19 eV
10i: E = -1.788e-19 eV
10h: E = -1.788e-19 eV
18s: E = -1.788e-19 eV
10g: E = -1.784e-19 eV
10f: E = -1.756e-19 eV
10d: E = -1.687e-19 eV
19s: E = -1.621e-19 eV
10p: E = -1.592e-19 eV
20s: E = -1.478e-19 eV
11i: E = -1.478e-19 eV
11h: E = -1.478e-19 eV
11g: E = -1.475e-19 eV
11f: E = -1.453e-19 eV
11d: E = -1.401e-19 eV
11p: E = -1.329e-19 eV
12i: E = -1.242e-19 eV
12h: E = -1.242e-19 eV
12g: E = -1.239e-19 eV
12f: E = -1.223e-19 eV
12d: E = -1.183e-19 eV
12p: E = -1.127e-19 eV
13i: E = -1.058e-19 eV
13h: E = -1.058e-19 eV
13g: E = -1.056e-19 eV
13f: E = -1.043e-19 eV
13d: E = -1.011e-19 eV
13p: E = -9.672e-20 eV
14i: E = -9.122e-20 eV
14h: E = -9.122e-20 eV
14g: E = -9.107e-20 eV
14f: E = -9.002e-20 eV
14d: E = -8.748e-20 eV
14p: E = -8.393e-20 eV
15h: E = -7.946e-20 eV
15i: E = -7.946e-20 eV
15g: E = -7.934e-20 eV
15f: E = -7.848e-20 eV
15d: E = -7.642e-20 eV
15p: E = -7.351e-20 eV
16h: E = -6.984e-20 eV
16i: E = -6.984e-20 eV
16g: E = -6.973e-20 eV
16f: E = -6.903e-20 eV
16d: E = -6.733e-20 eV
16p: E = -6.492e-20 eV
17h: E = -6.186e-20 eV
17i: E = -6.186e-20 eV
17g: E = -6.178e-20 eV
17f: E = -6.119e-20 eV
17d: E = -5.977e-20 eV
17p: E = -5.775e-20 eV
18h: E = -5.518e-20 eV
18i: E = -5.518e-20 eV
18g: E = -5.511e-20 eV
18f: E = -5.461e-20 eV
18d: E = -5.341e-20 eV
18p: E = -5.171e-20 eV
19h: E = -4.953e-20 eV
19i: E = -4.953e-20 eV
19g: E = -4.946e-20 eV
19f: E = -4.904e-20 eV
19d: E = -4.802e-20 eV
19p: E = -4.657e-20 eV
20h: E = -4.470e-20 eV
20i: E = -4.470e-20 eV
20g: E = -4.464e-20 eV
20f: E = -4.428e-20 eV
20d: E = -4.340e-20 eV
20p: E = -4.215e-20 eV
21h: E = -4.054e-20 eV
21i: E = -4.054e-20 eV
21g: E = -4.049e-20 eV
21f: E = -4.018e-20 eV
21d: E = -3.942e-20 eV
21p: E = -3.834e-20 eV
22h: E = -3.694e-20 eV
22i: E = -3.694e-20 eV
22g: E = -3.690e-20 eV
22f: E = -3.662e-20 eV
22d: E = -3.596e-20 eV
23h: E = -3.380e-20 eV
23i: E = -3.380e-20 eV
23g: E = -3.376e-20 eV
23f: E = -3.352e-20 eV
24h: E = -3.104e-20 eV
24i: E = -3.104e-20 eV
24g: E = -3.101e-20 eV
25h: E = -2.860e-20 eV
25i: E = -2.860e-20 eV
26i: E = -2.644e-20 eV
"""
