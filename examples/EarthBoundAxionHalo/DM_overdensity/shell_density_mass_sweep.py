"""Infer single-state halo masses and angularly averaged densities from a shell.

Edit the settings below, then run from the repository root with:
    python -m examples.EarthBoundAxionHalo.DM_overdensity.shell_density_mass_sweep
Each eigenstate is an independent hypothesis, never a population mixture.
The existing Earth-only potential is used (no halo self-gravity).
"""

import json
from pathlib import Path

import numpy as np
from astropy import constants as const, units as unit
import matplotlib.pyplot as plt

from examples.EarthBoundAxionHalo.DM_overdensity.shell_density_helpers import solve, json_numbers


# User settings. Orbital radii are geocentric, not altitudes.
satellite_radius = 12300 * unit.km
moon_radius = 384000 * unit.km

# Quoted shell constraint: (0.3 +/- 4) * 1e-9 M_earth.
# Propagate the quoted uncertainty linearly; no confidence level or positive
# upper-limit construction is assumed. This is consistent with zero mass.
shell_mass = 0.3e-9 * unit.M_earth
shell_mass_uncertainty = 4e-9 * unit.M_earth

masses = np.array([1e-9, 3e-9, 1e-8]) * unit.eV / const.c**2
n = 32768  # Resolves the small shell tails of the most localized states.
extent = 256 * unit.R_earth
output = Path(__file__).resolve().parent / "outputs" / "shell_normalized"
STATES = ["1s", "2s", "2p", "3s", "3p", "3d"]


sat, moon = satellite_radius, moon_radius
if not (0 < sat < moon < extent and moon > 1 * unit.R_earth
        and np.isfinite(shell_mass) and shell_mass > 0
        and np.isfinite(shell_mass_uncertainty) and shell_mass_uncertainty >= 0 and n >= 32
        and all(np.isfinite(m) and m > 0 for m in masses)):
    raise ValueError("Require positive finite masses, N>=32, and 0<satellite<Moon<extent with Moon>1.")
plt.switch_backend("Agg")
output.mkdir(parents=True, exist_ok=True)
radii = np.unique(np.concatenate((
    np.geomspace(.01 * unit.R_earth, moon, 1200),
    unit.Quantity([1 * unit.R_earth, sat, moon]),
)))
uncertainty_scale = (shell_mass_uncertainty / shell_mass).to(unit.one)
summary = []
fig, axes = plt.subplots(2, len(masses), figsize=(5*len(masses), 8), squeeze=False)
print(f"Shell mass = ({shell_mass:g} +/- {shell_mass_uncertainty:g}); "
      f"radii = {satellite_radius:g} to {moon_radius:g}.", flush=True)
print("Central estimates and quoted uncertainties; not detections or upper limits.", flush=True)
for mass_index, (ax, mass) in enumerate(zip(axes[0], masses)):
    uncertainty_ax = axes[1, mass_index]
    print(f"Solving m_a={mass:g} with resolution and extent checks...", flush=True)
    base = solve(mass, n, extent, sat, moon, shell_mass, radii, STATES)
    fine = solve(mass, 2*n+1, extent, sat, moon, shell_mass, radii, STATES)
    # Same fine spacing, twice the outer radius.
    wide = solve(mass, 4*n+3, 2*extent, sat, moon, shell_mass, radii, STATES)
    columns = [radii]
    for name in STATES:
        row, density = wide[name]
        row["mass_ev_c2"] = (mass * const.c**2).to(unit.eV)
        for key in ("moon_mass_earth", "rho_earth_g_cm3"):
            row[key + "_resolution_relchange"] = abs(fine[name][0][key] / base[name][0][key] - 1)
            row[key + "_extent_relchange"] = abs(row[key] / fine[name][0][key] - 1)
        row["converged_1percent"] = all(row[k] < .01 for k in row if k.endswith("relchange"))
        for key in ("moon_mass_earth", "domain_mass_earth", "rho_earth_g_cm3"):
            row[key + "_uncertainty"] = row[key] * uncertainty_scale
        row["central_domain_mass_exceeds_earth"] = row["domain_mass_earth"] >= 1 * unit.M_earth
        summary.append(row)
        rho = density.to(unit.g / unit.cm**3)
        columns.extend([rho, rho * uncertainty_scale])
        uncertainty_ax.loglog(radii.to_value(unit.R_earth), (rho * uncertainty_scale).to_value(unit.g / unit.cm**3), label=name)
        ax.loglog(radii.to_value(unit.R_earth), rho.to_value(unit.g / unit.cm**3), label=name)
        print(f"  {name}: M(<Moon)/M_shell={row['moon_mass_over_shell_mass']:.6g}; "
              f"M(<Moon)=({row['moon_mass_earth']:.6g} +/- {row['moon_mass_earth_uncertainty']:.6g}); "
              f"rho(Earth)=({row['rho_earth_g_cm3']:.6g} +/- {row['rho_earth_g_cm3_uncertainty']:.6g}); "
              f"converged={row['converged_1percent']}", flush=True)
    # Mixed-unit text columns require explicit conversion at the file boundary.
    table = np.column_stack([columns[0].to_value(unit.R_earth)] +
                            [column.to_value(unit.g / unit.cm**3) for column in columns[1:]])
    mass_label = (mass * const.c**2).to_value(unit.eV)  # filename/plot label only
    np.savetxt(output / f"density_{mass_label:.6g}_eV.txt", table,
               header="r_R_earth " + " ".join(f"{s}_rho_g_cm3 {s}_rho_uncertainty_g_cm3" for s in STATES))
    ax.axvline(sat.to_value(unit.R_earth), color="gray", ls=":", label="Satellite")
    ax.axvline(moon.to_value(unit.R_earth), color="gray", ls="--", label="Moon")
    ax.set(title=rf"$m_a={mass_label:g}$ eV/$c^2$", xlabel=r"$r/R_\oplus$", ylabel=r"$\bar\rho$ (g/cm$^3$)")
    uncertainty_ax.set(xlabel=r"$r/R_\oplus$", ylabel=r"Quoted uncertainty in $\bar\rho$ (g/cm$^3$)")
    uncertainty_ax.axvline(sat.to_value(unit.R_earth), color="gray", ls=":")
    uncertainty_ax.axvline(moon.to_value(unit.R_earth), color="gray", ls="--")
    uncertainty_ax.grid(alpha=.2)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, ncol=2)
fig.suptitle("Independent single-state hypotheses: central density (top), quoted uncertainty (bottom)")
fig.tight_layout()
fig.savefig(output / "density_profiles.png", dpi=180)
plt.close(fig)
metadata = {
    "shell_mass_earth": shell_mass.to(unit.M_earth),
    "shell_mass_uncertainty_earth": shell_mass_uncertainty.to(unit.M_earth),
    "satellite_radius_earth": satellite_radius.to(unit.R_earth),
    "moon_radius_earth": moon_radius.to(unit.R_earth),
    "satellite_radius_km": satellite_radius.to(unit.km),
    "moon_radius_km": moon_radius.to(unit.km),
    "masses_ev": (masses * const.c**2).to(unit.eV),
    "n": n,
    "extent_earth": extent.to(unit.R_earth),
    "states": STATES,
}
metadata["output"] = str(output)
metadata["interpretation"] = "Central values and linearly propagated quoted uncertainties; confidence level unspecified; no positivity truncation or upper bound inferred."
(output / "results.json").write_text(json.dumps(json_numbers({"settings": metadata, "results": summary}), indent=2), encoding="utf-8")
print(f"Saved results to {output.resolve()}")
