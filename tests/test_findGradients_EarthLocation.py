"""Tests for findGradients / findGradientsAtDirection with EarthLocation interface.

Run with::

    pytest tests/test_findGradients_EarthLocation.py -q
"""

import numpy as np
import pytest
from astropy import units as unit
from astropy.coordinates import EarthLocation
from astropy.time import Time

from axionbloch.Station import Mainz, Station

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


def test_station_rotation_velocity_uses_solarZ_frame():
    """Mainz rotation speed is physical and tangent to its geocentric radius."""
    meas_time = Time("2022-12-14T12:00:00")
    position = unit.Quantity(
        Mainz.in_solarZ_frame(meas_time, output="cartesian")
    )
    velocity = Mainz.rotation_velocity_in_solarZ_frame(meas_time)

    speed = np.linalg.norm(velocity)
    assert np.isclose(speed.to_value(unit.m / unit.s), 299.6, rtol=2e-3)
    assert np.isclose(
        np.dot(position, velocity).to_value(unit.m**2 / unit.s),
        0.0,
        atol=1e3,
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


def test_findGradients_resolves_default_measurement_time(monkeypatch):
    """The Earth-bound wrapper resolves meas_time=None before delegation."""
    from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo

    halo = object.__new__(EarthBoundAxionHalo)
    captured = {}

    def fake_find_gradients_at_direction(**kwargs):
        captured.update(kwargs)
        return (None,) * 6

    monkeypatch.setattr(
        halo, "findGradientsAtDirection", fake_find_gradients_at_direction
    )
    before = Time.now()
    halo.findGradients(stateNames=["2p"], station=Mainz, meas_time=None)
    after = Time.now()

    assert isinstance(captured["meas_time"], Time)
    assert before <= captured["meas_time"] <= after


def test_lorentz_boost_zero_velocity_matches_intrinsic_gradient(solved_halo):
    """An explicit zero relative velocity leaves the gradient unchanged."""
    kwargs = {
        "stateNames": ["2p"],
        "station": Mainz,
        "meas_time": Time("2022-12-14T12:00:00"),
        "truncRadius": 2 * unit.R_earth,
        "showPlot": False,
    }
    intrinsic = solved_halo.findGradients(
        **kwargs,
        include_lorentz_boost=False,
    )
    zero_boost = solved_halo.findGradients(
        **kwargs,
        include_lorentz_boost=True,
        relative_velocity=np.zeros(3) * unit.m / unit.s,
    )

    for intrinsic_component, boosted_component in zip(
        intrinsic[3:], zero_boost[3:]
    ):
        boosted_compatible = boosted_component.to(
            intrinsic_component.unit,
            equivalencies=unit.dimensionless_angles(),
        )
        assert np.allclose(
            boosted_compatible.value,
            intrinsic_component.value,
            rtol=1e-12,
            atol=1e-12,
        )


def test_lorentz_boost_scales_linearly_with_relative_velocity(solved_halo):
    """At terrestrial speeds, the boost contribution is linear in v_rel."""
    kwargs = {
        "stateNames": ["2p"],
        "station": Mainz,
        "meas_time": Time("2022-12-14T12:00:00"),
        "truncRadius": 2 * unit.R_earth,
        "showPlot": False,
    }
    intrinsic = solved_halo.findGradients(
        **kwargs,
        include_lorentz_boost=False,
    )
    velocity = np.array([100.0, 0.0, 0.0]) * unit.m / unit.s
    boosted_once = solved_halo.findGradients(
        **kwargs,
        relative_velocity=velocity,
    )
    boosted_twice = solved_halo.findGradients(
        **kwargs,
        relative_velocity=2.0 * velocity,
    )

    nonzero_boost_found = False
    for intrinsic_component, once_component, twice_component in zip(
        intrinsic[3:], boosted_once[3:], boosted_twice[3:]
    ):
        once = once_component.to(
            intrinsic_component.unit,
            equivalencies=unit.dimensionless_angles(),
        )
        twice = twice_component.to(
            intrinsic_component.unit,
            equivalencies=unit.dimensionless_angles(),
        )
        delta_once = once - intrinsic_component
        delta_twice = twice - intrinsic_component
        nonzero_boost_found |= np.any(np.abs(delta_once.value) > 0.0)
        assert np.allclose(
            delta_twice.value,
            2.0 * delta_once.value,
            rtol=1e-9,
            atol=1e-18,
        )

    assert nonzero_boost_found
