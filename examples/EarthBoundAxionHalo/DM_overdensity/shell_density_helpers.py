"""Reusable calculations for the shell-normalized density example.

Importing this module does not run simulations or generate output files.
"""

import numpy as np
from astropy import units as unit

from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo


# Compatibility alias for the original sweep and its integration tests.
probability_between = EarthBoundAxionHalo._integrateRadialProbability


def infer_state(halo, name, satellite, moon, shell_mass, radii):
    """Adapt the class API to the detailed sweep's existing output columns.

    Parameters
    ----------
    halo : EarthBoundAxionHalo
        Instance with the requested eigenstate already solved. Its stored
        mass inference is replaced by the inference for this state.
    name : str
        Eigenstate label, for example "1s".
    satellite, moon : astropy.units.Quantity
        Scalar inner and outer geocentric shell radii, in length units.
        Must satisfy 0 <= satellite < moon <= halo.extent.
    shell_mass : astropy.units.Quantity
        Positive scalar shell mass. This wrapper passes zero uncertainty;
        the sweep propagates its quoted uncertainty separately.
    radii : astropy.units.Quantity
        Scalar or 1-D positive radii at which to evaluate density.

    Returns
    -------
    row : dict
        State label, dimensionless probability fractions and mass ratio,
        enclosed/domain masses in Earth masses, and Earth-surface density
        in g/cm**3. Numerical results remain Quantity objects.
    density : astropy.units.Quantity
        Angularly averaged density in g/cm**3, with the shape of radii.

    Notes
    -----
    The grid must extend to at least one Earth radius for the reported
    Earth-surface density to be defined.
    """
    # Infer the mass for one hypothesis; the sweep propagates uncertainty later.
    result = halo.inferHaloMass(
        shell_mass, 0 * shell_mass, [satellite, moon], [name],
    )[name]
    # Query the radial density array and the Earth-surface value from the class.
    profile = halo.getDensity(radii)[name]
    earth_density = halo.getDensity(1 * unit.R_earth)[name]["density"]
    # Measure probability in the outermost 10% as a domain-truncation diagnostic.
    r = np.concatenate((0 * halo.r[:1], halo.r, halo.extent.reshape(1)))
    u = np.pad(halo.states[name]["u_r"], (1, 1))
    outer_fraction = probability_between(r, u, .9 * halo.extent, halo.extent) / probability_between(r, u, r[0], r[-1])
    # Keep the sweep's output schema while retaining quantities until serialization.
    total = result["total_mass"]
    row = {
        "state": name,
        "shell_fraction": result["shell_fraction"],
        "moon_mass_earth": result["enclosed_mass"].to(unit.M_earth),
        "domain_mass_earth": total.to(unit.M_earth),
        "moon_mass_over_shell_mass": (result["enclosed_mass"] / shell_mass).to(unit.one),
        "rho_earth_g_cm3": earth_density,
        "outer_ten_percent_fraction": outer_fraction.to(unit.one),
    }
    return row, profile["density"]



def solve(mass, n, extent, satellite, moon, shell_mass, radii, states):
    """Solve an Earth halo and infer the sweep's single-state profiles.

    Parameters
    ----------
    mass : astropy.units.Quantity
        Scalar axion mass with mass units, e.g. unit.eV / const.c**2.
    n : int
        Number of positive interior radial grid points.
    extent : astropy.units.Quantity
        Outer radial boundary, with length units.
    satellite, moon : astropy.units.Quantity
        Inner and outer geocentric radii of the measured shell.
    shell_mass : astropy.units.Quantity
        Positive scalar mass within that shell.
    radii : astropy.units.Quantity
        Scalar or 1-D positive density-evaluation radii inside the domain.
    states : sequence of str
        Labels to return. Must belong to the three lowest radial states
        of channels l=0, 1, 2, which this wrapper solves.

    Returns
    -------
    results : dict
        Mapping from label to the (row, density) pair from infer_state.
    """
    # Build and solve one halo for this mass and grid configuration.
    halo = EarthBoundAxionHalo(m_a=mass, N=n, extent=extent)
    halo.solve_TISE_3D(l_vals=[0, 1, 2], max_n_r=3)
    # Infer each requested eigenstate separately, never summing their masses.
    return {name: infer_state(halo, name, satellite, moon, shell_mass, radii)
            for name in states}


def json_numbers(value):
    """Strip units only at the JSON boundary; field names specify output units.

    Parameters
    ----------
    value : object
        Quantity, NumPy scalar, nested dict/list/tuple, or JSON-compatible
        Python value. Quantities must already use the desired output units.

    Returns
    -------
    converted : object
        Recursively converted Python scalars, lists and dictionaries.
        Quantity units are dropped, not converted; tuples become lists.
        Other values pass through unchanged.
    """
    # Drop units only at this output boundary; callers already chose the output units.
    if isinstance(value, unit.Quantity):
        return value.value.tolist()
    # Recursively convert nested result containers into JSON-compatible structures.
    if isinstance(value, dict):
        return {key: json_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_numbers(item) for item in value]
    # Convert remaining NumPy scalars; ordinary Python values need no changes.
    if isinstance(value, np.generic):
        return value.item()
    return value
