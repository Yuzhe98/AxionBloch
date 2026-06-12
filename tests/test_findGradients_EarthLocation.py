"""Tests for findGradients / findGradientsAtDirection with EarthLocation interface.

Run with::

    pytest tests/test_findGradients_EarthLocation.py -q
"""

import numpy as np
import pytest

from astropy import units as unit
from astropy.coordinates import EarthLocation

from axionbloch.Station import Mainz, Baltimore, Station

# ---------------------------------------------------------------------------
# Station.location
# ---------------------------------------------------------------------------


def test_station_has_location_attribute():
    assert hasattr(Mainz, "location")
    assert isinstance(Mainz.location, EarthLocation)


def test_station_location_lat_lon_sign():
    """Northern / eastern hemisphere → positive lat / lon."""
    assert Mainz.location.lat.to_value(unit.deg) > 0  # ~50° N
    assert Mainz.location.lon.to_value(unit.deg) > 0  # ~8° E


def test_station_location_southern_western():
    """Southern / western hemisphere → negative lat / lon."""
    from axionbloch.Station import BuenosAires

    assert BuenosAires.location.lat.to_value(unit.deg) < 0  # ~35° S
    assert BuenosAires.location.lon.to_value(unit.deg) < 0  # ~58° W


def test_station_location_elevation_matches():
    """station.location.height should equal station.elevation."""
    assert np.isclose(
        Mainz.location.height.to_value(unit.m),
        Mainz.elevation.to_value(unit.m),
        rtol=1e-9,
    )


def test_station_location_colatitude_conversion():
    """lat=90° (north pole) → colatitude theta=0; lat=0° (equator) → theta=90°."""
    north_pole = Station(
        "NorthPole",
        NSsemisphere="N",
        EWsemisphere="E",
        latitude=90 * unit.deg,
        longitude=0 * unit.deg,
        elevation=0 * unit.m,
    )
    equator = Station(
        "Equator",
        NSsemisphere="N",
        EWsemisphere="E",
        latitude=0 * unit.deg,
        longitude=0 * unit.deg,
        elevation=0 * unit.m,
    )
    theta_np = (90 * unit.deg - north_pole.location.lat).to_value(unit.deg)
    theta_eq = (90 * unit.deg - equator.location.lat).to_value(unit.deg)
    assert np.isclose(theta_np, 0.0, atol=1e-9)
    assert np.isclose(theta_eq, 90.0, atol=1e-9)


# ---------------------------------------------------------------------------
# findGradients interface (requires a solved halo)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def solved_halo():
    """A minimal EarthBoundAxionHalo solved for l=1 with few states."""
    from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo

    halo = EarthBoundAxionHalo(
        nu_a=1.348 * unit.MHz,
        N=int(2**10),  # small grid for speed
        extent=8.0 * unit.R_earth,
        verbose=False,
    )
    halo.solve_TISE_3D(l_vals=[1], max_n_r=4, verbose=False)
    return halo


def test_findGradients_raises_without_location(solved_halo):
    """findGradients raises ValueError when no station is given."""
    with pytest.raises(ValueError):
        solved_halo.findGradients(stateNames=["2p"], showPlot=False)
