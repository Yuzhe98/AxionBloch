"""Tests for findGradients / findGradientsAtDirection with EarthLocation interface.

Run with::

    pytest tests/test_findGradients_EarthLocation.py -q
"""

import numpy as np
import pytest
from astropy import constants as const
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
        solved_halo.findGradients(
            stateCoefficients={"2p": 1.0},
            showPlot=False,
        )


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
    halo.findGradients(
        stateCoefficients={"2p": 1.0},
        station=Mainz,
        meas_time=None,
    )
    after = Time.now()

    assert isinstance(captured["meas_time"], Time)
    assert before <= captured["meas_time"] <= after


def test_lorentz_boost_zero_velocity_matches_intrinsic_gradient(solved_halo):
    """An explicit zero relative velocity leaves the gradient unchanged."""
    kwargs = {
        "stateCoefficients": {"2p": 1.0},
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
        "stateCoefficients": {"2p": 1.0},
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


def test_gradient_superposition_uses_complex_state_coefficients(solved_halo):
    """The total wavefunction and gradient are linear in the supplied c_nlm."""
    kwargs = {
        "station": Mainz,
        "meas_time": Time("2022-12-14T12:00:00"),
        "truncRadius": 2 * unit.R_earth,
        "include_lorentz_boost": True,
        "showPlot": False,
    }
    coefficient_2p = 0.4 + 0.2j
    coefficient_3p = -0.7j
    coefficient_norm = np.sqrt(
        np.abs(coefficient_2p) ** 2 + np.abs(coefficient_3p) ** 2
    )

    result_2p = solved_halo.findGradients(
        **kwargs,
        stateCoefficients={"2p": 1.0},
    )
    result_3p = solved_halo.findGradients(
        **kwargs,
        stateCoefficients={"3p": 1.0},
    )
    result_combined = solved_halo.findGradients(
        **kwargs,
        stateCoefficients={
            "2p": coefficient_2p,
            "3p": coefficient_3p,
        },
    )

    for result_index in (1, 3, 4, 5):
        expected = (
            coefficient_2p * result_2p[result_index]
            + coefficient_3p * result_3p[result_index]
        ) / coefficient_norm
        actual = result_combined[result_index].to(
            expected.unit,
            equivalencies=unit.dimensionless_angles(),
        )
        assert np.allclose(actual.value, expected.value, rtol=2e-10, atol=1e-18)


def test_Omega_a_uses_magnitude_of_combined_gradient(solved_halo):
    """Omega_a is derived after coherently summing the selected states."""
    meas_times = Time(["2022-12-14T12:00:00"])
    coefficients = {"2p": 0.4 + 0.2j, "3p": -0.7j}
    gradient_result = solved_halo.findGradientsOverTime(
        stateCoefficients=coefficients,
        station=Mainz,
        meas_times=meas_times,
        truncRadius=2 * unit.R_earth,
        include_lorentz_boost=True,
    )
    Omega_result = solved_halo.findOmega_aOverTime(
        stateCoefficients=coefficients,
        station=Mainz,
        meas_times=meas_times,
        truncRadius=2 * unit.R_earth,
        include_lorentz_boost=True,
    )
    factor = (
        const.c
        * (1e-9 * unit.GeV**-1)
        * np.sqrt(
            solved_halo.N_a
            * const.hbar**3
            * const.c
            / (2 * solved_halo.m_a)
        )
    )

    for gradient_key, Omega_key in (
        ("grad_r", "Omega_a_r"),
        ("grad_theta", "Omega_a_theta"),
        ("grad_phi", "Omega_a_phi"),
    ):
        expected = (factor * np.abs(gradient_result[gradient_key])).to(
            unit.Hz,
            equivalencies=unit.dimensionless_angles(),
        )
        assert np.allclose(
            Omega_result[Omega_key].value,
            expected.value,
            rtol=2e-10,
            atol=1e-18,
        )


def test_state_coefficients_are_normalized(solved_halo):
    coefficients = solved_halo._resolveStateCoefficients(
        stateCoefficients={"2p": 3.0, "3p": 4.0j},
    )

    assert np.isclose(sum(np.abs(c) ** 2 for c in coefficients.values()), 1.0)
    assert coefficients["2p"] == pytest.approx(0.6)
    assert coefficients["3p"] == pytest.approx(0.8j)


@pytest.mark.parametrize(
    ("state_coefficients", "error_type"),
    [
        (None, ValueError),
        ({}, ValueError),
        (["2p", 1.0], TypeError),
        ({"not-a-state": 1.0}, KeyError),
        ({"2p": np.inf}, ValueError),
        ({"2p": 0.0, "3p": 0.0}, ValueError),
    ],
)
def test_state_coefficient_validation(
    solved_halo,
    state_coefficients,
    error_type,
):
    with pytest.raises(error_type):
        solved_halo._resolveStateCoefficients(
            stateCoefficients=state_coefficients,
        )
