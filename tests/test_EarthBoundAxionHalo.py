"""Tests for axionbloch.EarthBoundAxionHalo utility functions.

Covers the Preliminary Reference Earth Model (PREM) data loader and the
Earth gravitational-potential helper used by the bound-axion Schrödinger
solver.

Run with::

    pytest tests/test_EarthBoundAxionHalo.py -q
"""

import numpy as np
from astropy import units as unit
from astropy.time import Time

from axionbloch.EarthBoundAxionHalo import (EarthBoundAxionHalo,
                                            earth_grav_potential_earth_center,
                                            loadPEMdata,
                                            plot_earth_grav_potential)
from axionbloch.Station import Mainz


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


def test_findGradientsOverStates_returns_correct_structure():
    """findGradientsOverStates returns a dict with expected keys and valid arrays."""
    halo = EarthBoundAxionHalo(
        nu_a=1.348 * unit.MHz,
        N=int(2**10),
        extent=128.0 * unit.R_earth,
        verbose=False,
    )

    halo.solve_TISE_3D(
        l_vals=[1],
        max_n_r=10,
        verbose=False,
    )

    state_combinations = {
        "2p": ["2p"],
        "2p + 3p": ["2p", "3p"],
    }

    results = halo.findGradientsOverStates(
        station=Mainz,
        meas_time=Time("2022-12-14T12:00:00"),
        stateNamesDict=state_combinations,
        truncRadius=2 * unit.earthRad,
        showPlot=False,
        verbose=False,
    )

    # Check that results dict has expected keys
    assert isinstance(results, dict)
    assert "state_labels" in results
    assert "grad_r" in results
    assert "grad_theta" in results
    assert "grad_phi" in results
    assert "r_line" in results

    # Check state_labels matches input
    assert results["state_labels"] == list(state_combinations.keys())

    # Check gradient dicts have entries for each state combination
    for label in state_combinations.keys():
        assert label in results["grad_r"]
        assert label in results["grad_theta"]
        assert label in results["grad_phi"]

    # Check gradient arrays have valid shapes and units
    for label in state_combinations.keys():
        assert isinstance(results["grad_r"][label], unit.Quantity)
        assert isinstance(results["grad_theta"][label], unit.Quantity)
        assert isinstance(results["grad_phi"][label], unit.Quantity)
        assert len(results["grad_r"][label]) > 0
        assert len(results["grad_theta"][label]) > 0
        assert len(results["grad_phi"][label]) > 0

    # Check r_line is valid
    assert isinstance(results["r_line"], unit.Quantity)
    assert len(results["r_line"]) > 0
