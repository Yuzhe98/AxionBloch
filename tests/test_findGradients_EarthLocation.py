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
    assert Mainz.location.lat.to_value(unit.deg) > 0   # ~50° N
    assert Mainz.location.lon.to_value(unit.deg) > 0   # ~8° E


def test_station_location_southern_western():
    """Southern / western hemisphere → negative lat / lon."""
    from axionbloch.Station import BuenosAires
    assert BuenosAires.location.lat.to_value(unit.deg) < 0   # ~35° S
    assert BuenosAires.location.lon.to_value(unit.deg) < 0   # ~58° W


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
        N=int(2**10),           # small grid for speed
        extent=8.0 * unit.R_earth,
        verbose=False,
    )
    halo.solve_TISE_3D(l_vals=[1], max_n_r=4, verbose=False)
    return halo


def test_findGradients_with_station(solved_halo):
    """findGradients accepts a Station object via the station= parameter."""
    result = solved_halo.findGradients(
        stateNames=["2p"],
        station=Baltimore,
        truncRadius=2 * unit.R_earth,
        showPlot=False,
    )
    assert len(result) == 6


def test_findGradients_with_EarthLocation(solved_halo):
    """findGradients accepts an EarthLocation via the location= parameter."""
    loc = EarthLocation(lat=39.3 * unit.deg, lon=-76.6 * unit.deg, height=35 * unit.m)
    result = solved_halo.findGradients(
        stateNames=["2p"],
        location=loc,
        truncRadius=2 * unit.R_earth,
        showPlot=False,
    )
    assert len(result) == 6


def test_findGradients_station_and_location_give_same_result(solved_halo):
    """Station and equivalent EarthLocation should produce identical gradients."""
    # Baltimore: lat≈39.33° N, lon≈-76.62° W
    loc = EarthLocation(
        lat=Baltimore.location.lat,
        lon=Baltimore.location.lon,
        height=Baltimore.location.height,
    )
    r1, _, rl1, gr1, gt1, gp1 = solved_halo.findGradients(
        stateNames=["2p"],
        station=Baltimore,
        truncRadius=2 * unit.R_earth,
        showPlot=False,
    )
    r2, _, rl2, gr2, gt2, gp2 = solved_halo.findGradients(
        stateNames=["2p"],
        location=loc,
        truncRadius=2 * unit.R_earth,
        showPlot=False,
    )
    assert np.allclose(gr1.value, gr2.value, rtol=1e-10)
    assert np.allclose(gt1.value, gt2.value, rtol=1e-10)
    assert np.allclose(gp1.value, gp2.value, rtol=1e-10)


def test_findGradients_raises_without_location(solved_halo):
    """findGradients raises ValueError when neither station nor location is given."""
    with pytest.raises(ValueError):
        solved_halo.findGradients(stateNames=["2p"], showPlot=False)


def test_findGradientsAtDirection_with_EarthLocation(solved_halo):
    """findGradientsAtDirection accepts an EarthLocation."""
    loc = EarthLocation(lat=0 * unit.deg, lon=0 * unit.deg, height=0 * unit.m)
    result = solved_halo.findGradientsAtDirection(
        location=loc,
        stateNames=["2p"],
        truncRadius=2 * unit.R_earth,
        showPlot=False,
    )
    assert len(result) == 6
