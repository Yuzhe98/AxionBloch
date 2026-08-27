"""Plot the local axion dark-matter density at Earth radius versus axion mass.

Assumption:
- All dark matter occupies a single eigenstate.
- The local mass density is estimated from the spherical average of that
  normalized eigenstate:

    rho(r) = M_DM * |R_r(r)|^2 / (4 pi)

This script evaluates the density at r = R_earth for a sweep of axion masses
``m_a`` and shows:
- bottom x-axis: ``m_a``
- top x-axis: ``nu_a``
- left y-axis: density in GeV/cm^3
- right y-axis: density in g/cm^3
"""

from datetime import datetime
from pathlib import Path

from axionbloch.dependency import *
from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo


rhoE_DM_MW = 0.4 * unit.GeV / unit.cm**3
totalMassEnclosed = 4e-9 * unit.M_earth

# Sweep m_a over the requested range.
m_a_quantities = np.logspace(-13.5, -8.5, 18) * unit.eV / const.c**2

# Check all states from 1s through 3d.
state_names = ["1s", "2s", "2p", "3s", "3p", "3d"]
l_vals = [0, 1, 2]
max_n_r = 3

rho_gevcm3 = {name: [] for name in state_names}
rho_gcm3 = {name: [] for name in state_names}
m_integral = {name: [] for name in state_names}
m_relerr = {name: [] for name in state_names}
nu_a_values = []

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
data_path = output_dir / f"EarthHalo-density-at-earth-radius-vs-nua-{timestamp}.txt"

with data_path.open("w", encoding="utf-8") as fh:
    fh.write(
        "# nu_a_Hz m_a_eV_c2 "
        + " ".join(
            [
                f"{name}_rho_GeV_cm3 {name}_rho_g_cm3 {name}_M_int_kg {name}_M_relerr"
                for name in state_names
            ]
        )
        + "\n"
    )

    for m_a in m_a_quantities:
        halo = EarthBoundAxionHalo(
            nu_a=None,
            m_a=m_a,
            N=2**12,
            extent=128 * unit.R_earth,
            totalMassEnclosed=totalMassEnclosed,
            verbose=False,
        )
        halo.solve_TISE_3D(
            l_vals=l_vals,
            max_n_r=max_n_r,
            verbose=False,
        )

        r_idx = int(np.argmin(np.abs(halo.r - 1 * unit.R_earth)))
        start_idx = halo.N // 2 + 4
        nu_a_values.append(halo.nu_a.to_value(unit.Hz))

        row = [f"{m_a.to_value(unit.eV / const.c**2):.12e}", f"{nu_a_values[-1]:.12e}"]

        for state_name in state_names:
            state = halo.states[state_name]
            u_r = state["u_r"]
            u_r_norm = np.trapezoid(np.abs(u_r[start_idx:]) ** 2, halo.r[start_idx:])
            print(
                f"{m_a.to_value(unit.eV / const.c**2):.3e} eV/c^2 {state_name}: "
                f"int |u_r|^2 dr = {u_r_norm.value:.10e}, "
                f"|norm-1| = {np.abs(u_r_norm - 1.0).value:.3e}"
            )

            R_at_earth = u_r[r_idx] / halo.r[r_idx]
            r_grid = halo.r[start_idx:]
            u_grid = u_r[start_idx:]

            # Spherically averaged local density from a single occupied eigenstate.
            rho_r = totalMassEnclosed * np.abs(R_at_earth) ** 2 / (4 * np.pi)
            rho_grid = totalMassEnclosed * np.abs(u_grid) ** 2 / (4 * np.pi * r_grid**2)
            mass_int = 4 * np.pi * np.trapezoid(
                rho_grid * r_grid**2,
                r_grid,
            )
            relerr = np.abs((mass_int - totalMassEnclosed) / totalMassEnclosed)

            rho_gevc = rho_r.to_value(
                unit.GeV / unit.cm**3,
                equivalencies=unit.mass_energy(),
            )
            rho_gcc = rho_r.to_value(unit.g / unit.cm**3)
            mass_int_kg = mass_int.to_value(unit.kg)
            relerr_val = relerr.to_value(unit.one)
            rho_gevcm3[state_name].append(rho_gevc)
            rho_gcm3[state_name].append(rho_gcc)
            m_integral[state_name].append(mass_int_kg)
            m_relerr[state_name].append(relerr_val)
            row.extend(
                [
                    f"{rho_gevc:.12e}",
                    f"{rho_gcc:.12e}",
                    f"{mass_int_kg:.12e}",
                    f"{relerr_val:.12e}",
                ]
            )

        fh.write(" ".join(row) + "\n")

for name in state_names:
    rho_gevcm3[name] = np.asarray(rho_gevcm3[name])
    rho_gcm3[name] = np.asarray(rho_gcm3[name])
    m_integral[name] = np.asarray(m_integral[name])
    m_relerr[name] = np.asarray(m_relerr[name])

nu_a_values = np.asarray(nu_a_values)
m_a_values_eV = m_a_quantities.to_value(unit.eV)

fig, ax = plt.subplots(figsize=(8.5 / 2.54, 5.5 / 2.54), dpi=300)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for idx, state_name in enumerate(state_names):
    color = colors[idx % len(colors)]
    ax.plot(
        m_a_values_eV,
        rho_gevcm3[state_name],
        color=color,
        linewidth=1.6,
        label=state_name,
    )
    ax.scatter(
        m_a_values_eV,
        rho_gevcm3[state_name],
        color=color,
        s=14,
        zorder=3,
    )

ax.axhline(
    rhoE_DM_MW.to_value(unit.GeV / unit.cm**3),
    color="k",
    linestyle="dotted",
    linewidth=1.2,
    alpha=0.85,
    label=rf"$\rho_{{E,\mathrm{{DM}}}} = 0.4$ GeV/cm$^3$",
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$m_a\,(\mathrm{eV}/c^2)$")
ax.set_ylabel(r"$\rho(R_\oplus)\,(\mathrm{GeV}/\mathrm{cm}^3)$")
ax.grid(True, which="both", alpha=0.25)

ax2 = ax.twinx()
ax2.set_yscale("log")
ax2.set_ylim(
    min(np.min(rho_gcm3[name]) for name in state_names),
    max(np.max(rho_gcm3[name]) for name in state_names),
)
ax2.set_ylabel(r"$\rho(R_\oplus)\,(\mathrm{g}/\mathrm{cm}^3)$")

ax_top = ax.twiny()
ax_top.set_xscale("log")
ax_top.set_xlim(ax.get_xlim())
tick_idx = np.linspace(0, len(m_a_values_eV) - 1, 5, dtype=int)
ax_top.set_xticks(m_a_values_eV[tick_idx])
ax_top.set_xticklabels([f"{nu_a_values[i]:.1e}" for i in tick_idx])
ax_top.set_xlabel(r"$\nu_a\,(\mathrm{Hz})$")

ax.legend(loc="best", fontsize=8, title="Eigenstate")
fig.tight_layout()

output_path = output_dir / "EarthHalo-density-at-earth-radius-vs-nua.pdf"
fig.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Saved figure to {output_path}")
print(f"Saved data to {data_path}")
print(f"States checked: {', '.join(state_names)}")
print(f"m_a range: {m_a_quantities[0]} to {m_a_quantities[-1]}")
for name in state_names:
    print(
        f"{name}: max relative mass-integral error = "
        f"{np.max(m_relerr[name]):.3e}"
    )
