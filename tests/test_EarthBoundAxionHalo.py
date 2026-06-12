"""Tests for axionbloch.EarthBoundAxionHalo utility functions.

Covers the Preliminary Reference Earth Model (PREM) data loader and the
Earth gravitational-potential helper used by the bound-axion Schrödinger
solver.

Run with::

    pytest tests/test_EarthBoundAxionHalo.py -q
"""

import numpy as np
from astropy import units as unit

from axionbloch.EarthBoundAxionHalo import (earth_grav_potential_earth_center,
                                            loadPEMdata,
                                            plot_earth_grav_potential)


def test_loadPEMdata():
    """loadPEMdata returns a dict with matching-length radius and density arrays."""
    data = loadPEMdata()
    assert "radius_m" in data
    assert "density_kg_m3" in data
    assert len(data["radius_m"]) == len(data["density_kg_m3"])


def test_earth_grav_potential_earth_center_returns_valid_objects():
    """earth_grav_potential_earth_center returns a callable and compatible unit objects."""
    phi_func, r_unit, phi_unit = earth_grav_potential_earth_center()

    assert callable(phi_func)
    assert r_unit.is_equivalent(unit.meter)
    assert phi_unit.is_equivalent(unit.joule / unit.kilogram)


def test_earth_grav_potential_earth_center_is_symmetric_and_finite():
    """The gravitational potential is finite and symmetric about r = 0 (even function)."""
    phi_func, r_unit, _ = earth_grav_potential_earth_center()

    # phi_func takes values in r_unit (R_earth); sample within the interpolation domain
    test_radius = 1e3 * unit.R_earth
    radii = np.linspace(
        -test_radius.to_value(r_unit), test_radius.to_value(r_unit), num=1000
    )
    phi_pos = phi_func(radii)
    phi_neg = phi_func(-radii)

    assert np.all(np.isfinite(phi_pos))
    assert np.all(np.isfinite(phi_neg))
    assert np.allclose(phi_pos, phi_neg, rtol=1e-10, atol=1e-10)


def test_plot_earth_grav_potential_runs_without_error():
    """plot_earth_grav_potential completes without raising when showplot=False."""
    try:
        plot_earth_grav_potential(showplot=False)
    except Exception as e:
        assert False, f"plot_earth_grav_potential raised an exception: {e}"
