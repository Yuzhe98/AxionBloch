"""Tests for axionbloch.MilkyWay.

Covers:
- Earth velocity in the galactic rest frame (magnitude, annual modulation)
- Sun's galactocentric position and velocity
- Station projection-axis unit vector (unit norm, daily variation)
- Wind angle (range, seasonal change)
- Earth rotation velocity (order of magnitude)
- Heliocentric velocity (order of magnitude)
- plot() smoke test (no errors, correct figure structure)

Run::

    pytest tests/test_MilkyWay.py -q -s
"""

import math

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — no window needed
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as unit
from astropy.time import Time

from axionbloch.MilkyWay import MilkyWay
from axionbloch.Station import Baltimore, Mainz, Sanya

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mw_june():
    """MilkyWay at northern summer solstice, station = Mainz."""
    return MilkyWay(time=Time("2024-06-21"), station=Mainz)


@pytest.fixture
def mw_dec():
    """MilkyWay at northern winter solstice, station = Mainz."""
    return MilkyWay(time=Time("2024-12-21"), station=Mainz)


@pytest.fixture
def mw_no_station():
    """MilkyWay without a station (for testing station-less paths)."""
    return MilkyWay(time=Time("2024-06-21"))


# ---------------------------------------------------------------------------
# Earth velocity / v_lab
# ---------------------------------------------------------------------------


class TestEarthVelocity:
    def test_v_lab_shape(self, mw_june):
        v = mw_june.get_v_lab()
        assert v.shape == (3,)
        assert v.unit.is_equivalent(unit.km / unit.s)

    def test_v_lab_magnitude_reasonable(self, mw_june):
        """v_lab should be in 210–310 km/s (Sun's rotation + Earth's orbit)."""
        v_mag = mw_june.get_v_lab_magnitude().to(unit.km / unit.s).value
        assert 210 < v_mag < 310, f"|v_lab| = {v_mag:.1f} km/s — outside expected range"

    def test_v_lab_annual_modulation(self, mw_june, mw_dec):
        """v_lab magnitude should differ by ~20–40 km/s between June and December."""
        v_june = mw_june.get_v_lab_magnitude().to(unit.km / unit.s).value
        v_dec = mw_dec.get_v_lab_magnitude().to(unit.km / unit.s).value
        delta = abs(v_june - v_dec)
        assert (
            10 < delta < 70
        ), f"Annual |v_lab| modulation = {delta:.1f} km/s — expected ~30 km/s"

    def test_v_lab_dominated_by_galactic_rotation(self, mw_june):
        """The y-component (galactic rotation direction) should be the largest."""
        v = mw_june.get_v_lab().to(unit.km / unit.s).value
        assert abs(v[1]) == max(abs(v))

    def test_v_lab_no_station(self, mw_no_station):
        """v_lab should work without a station set."""
        v = mw_no_station.get_v_lab_magnitude()
        assert v.value > 0


# ---------------------------------------------------------------------------
# Sun and Earth galactocentric positions
# ---------------------------------------------------------------------------


class TestGalactocentricPosition:
    def test_earth_position_shape(self, mw_june):
        pos = mw_june.get_earth_position()
        assert pos.shape == (3,)
        assert pos.unit.is_equivalent(unit.kpc)

    def test_galactocentric_radius_close_to_sun(self, mw_june):
        """Earth–GC distance should be ~8 kpc (close to galcen_distance = 8.122 kpc)."""
        r = mw_june.get_galactocentric_radius().to(unit.kpc).value
        assert 7.5 < r < 8.8, f"r_GC = {r:.3f} kpc"

    def test_sun_position_shape(self, mw_june):
        pos, vel = mw_june.get_sun_galactocentric()
        assert pos.shape == (3,)
        assert vel.shape == (3,)

    def test_sun_galactocentric_radius(self, mw_june):
        """Sun should be at ~8.122 kpc from GC."""
        pos, _ = mw_june.get_sun_galactocentric()
        r = float(np.sqrt(np.sum(pos.to(unit.kpc).value ** 2)))
        assert abs(r - 8.122) < 0.05, f"Sun–GC = {r:.3f} kpc"

    def test_sun_velocity_magnitude(self, mw_june):
        """Sun's galactic velocity should be ~245–250 km/s."""
        _, vel = mw_june.get_sun_galactocentric()
        v_mag = float(np.sqrt(np.sum(vel.to(unit.km / unit.s).value ** 2)))
        assert 230 < v_mag < 260, f"|v_sun| = {v_mag:.1f} km/s"


# ---------------------------------------------------------------------------
# Earth heliocentric velocity
# ---------------------------------------------------------------------------


class TestHeliocentricVelocity:
    def test_magnitude(self, mw_june):
        """Earth's heliocentric speed should be ~29–31 km/s."""
        v = mw_june.get_earth_heliocentric_velocity()
        v_mag = float(np.sqrt(np.sum(v.to(unit.km / unit.s).value ** 2)))
        assert 25 < v_mag < 35, f"|v_helio| = {v_mag:.2f} km/s"

    def test_opposite_seasons(self, mw_june, mw_dec):
        """Heliocentric velocities in June and December should be roughly anti-parallel."""
        v_jun = mw_june.get_earth_heliocentric_velocity().to(unit.km / unit.s).value
        v_dec = mw_dec.get_earth_heliocentric_velocity().to(unit.km / unit.s).value
        cos_angle = np.dot(v_jun, v_dec) / (
            np.linalg.norm(v_jun) * np.linalg.norm(v_dec)
        )
        assert (
            cos_angle < -0.8
        ), f"Helio velocities not anti-parallel: cos = {cos_angle:.3f}"


# ---------------------------------------------------------------------------
# Station projection axis (nvec)
# ---------------------------------------------------------------------------


class TestNvecGalactic:
    """get_nvec_galactic() returns the Earth/Sun direction from the galactic
    centre — essentially the same for all stations since Earth (6 371 km) is
    negligibly small at 8 kpc.  It is used for GravBoundAxionHalo gradient
    calculations, not for wind-angle computation."""

    def test_unit_norm(self, mw_june):
        n = mw_june.get_nvec_galactic()
        assert abs(np.linalg.norm(n) - 1.0) < 1e-6

    def test_shape(self, mw_june):
        assert mw_june.get_nvec_galactic().shape == (3,)

    def test_raises_without_station(self, mw_no_station):
        with pytest.raises(ValueError, match="station"):
            mw_no_station.get_nvec_galactic()

    def test_points_away_from_gc(self, mw_june):
        """nvec_galactic should be close to the -x galactocentric direction
        (GC is roughly in the -x direction from the Sun)."""
        n = mw_june.get_nvec_galactic()
        # Sun is at positive x in galactocentric frame; Earth is near Sun → n ≈ (-1, 0, 0)
        assert n[0] < -0.99, f"Expected nvec_galactic[0] ≈ -1, got {n[0]:.4f}"


class TestNvecGcrs:
    """get_nvec_gcrs() returns the station's local vertical in GCRS — the
    correct vector for wind-angle calculations (changes with station and time)."""

    def test_unit_norm(self, mw_june):
        assert abs(np.linalg.norm(mw_june.get_nvec_gcrs()) - 1.0) < 1e-6

    def test_shape(self, mw_june):
        assert mw_june.get_nvec_gcrs().shape == (3,)

    def test_daily_rotation(self):
        """nvec_gcrs rotates as Earth spins — the angle after 12 h should be significant.

        At latitude φ the dot product after 12 h (half rotation) is
        sin²φ − cos²φ = −cos(2φ).  For Mainz (φ≈50°) this is ~+0.17,
        so we only assert the vectors are not close (dot < 0.5, i.e. angle > 60°).
        For a near-equatorial station like Sanya the vectors approach anti-parallel.
        """
        t1 = Time("2024-06-21T00:00:00")
        t2 = Time("2024-06-21T12:00:00")
        # Mainz ~50°N: dot ≈ +0.17 after 12 h
        n1 = MilkyWay(t1, station=Mainz).get_nvec_gcrs()
        n2 = MilkyWay(t2, station=Mainz).get_nvec_gcrs()
        assert (
            np.dot(n1, n2) < 0.5
        ), f"nvec_gcrs barely moved at Mainz: cos = {np.dot(n1,n2):.4f}"
        # Sanya ~18°N: dot ≈ −0.81 after 12 h
        n3 = MilkyWay(t1, station=Sanya).get_nvec_gcrs()
        n4 = MilkyWay(t2, station=Sanya).get_nvec_gcrs()
        assert (
            np.dot(n3, n4) < 0.0
        ), f"nvec_gcrs barely moved at Sanya: cos = {np.dot(n3,n4):.4f}"

    def test_different_stations_differ(self, mw_june):
        n_mainz = MilkyWay(mw_june.time, station=Mainz).get_nvec_gcrs()
        n_sanya = MilkyWay(mw_june.time, station=Sanya).get_nvec_gcrs()
        assert not np.allclose(n_mainz, n_sanya, atol=0.01)

    def test_raises_without_station(self, mw_no_station):
        with pytest.raises(ValueError, match="station"):
            mw_no_station.get_nvec_gcrs()


# ---------------------------------------------------------------------------
# Wind angle
# ---------------------------------------------------------------------------


class TestWindAngle:
    def test_range(self, mw_june):
        """Wind angle must be in [0, π]."""
        alpha = mw_june.get_wind_angle().to(unit.rad).value
        assert 0.0 <= alpha <= math.pi

    def test_changes_with_time(self):
        """Wind angle should vary during the day because nvec_gcrs rotates with Earth."""
        times = Time(
            [
                "2024-06-21T00:00:00",
                "2024-06-21T06:00:00",
                "2024-06-21T12:00:00",
                "2024-06-21T18:00:00",
            ]
        )
        angles = [
            MilkyWay(t, station=Mainz).get_wind_angle().to(unit.deg).value
            for t in times
        ]
        # Over 18 h the projection axis sweeps ~270° — expect > 30° variation
        assert (
            max(angles) - min(angles) > 30.0
        ), f"Wind angle variation too small: {[f'{a:.1f}' for a in angles]}"

    def test_raises_without_station(self, mw_no_station):
        with pytest.raises(ValueError, match="station"):
            mw_no_station.get_wind_angle()

    def test_changes_between_stations(self, mw_june):
        """Stations at different latitudes/longitudes give different wind angles."""
        a_mainz = MilkyWay(mw_june.time, Mainz).get_wind_angle().to(unit.deg).value
        a_sanya = MilkyWay(mw_june.time, Sanya).get_wind_angle().to(unit.deg).value
        a_balt = MilkyWay(mw_june.time, Baltimore).get_wind_angle().to(unit.deg).value
        assert (
            abs(a_mainz - a_sanya) > 1.0
        ), f"Mainz {a_mainz:.2f}° vs Sanya {a_sanya:.2f}°"
        assert abs(a_mainz - a_balt) > 0.1


# ---------------------------------------------------------------------------
# Earth rotation velocity
# ---------------------------------------------------------------------------


class TestEarthRotation:
    def test_shape_and_units(self, mw_june):
        v = mw_june.get_earth_rotation_velocity()
        assert v.shape == (3,)
        assert v.unit.is_equivalent(unit.km / unit.s)

    def test_magnitude_at_mainz(self, mw_june):
        """At Mainz (lat ~50°), rotation speed should be ~0.25–0.35 km/s."""
        v_rot = float(
            np.linalg.norm(
                mw_june.get_earth_rotation_velocity().to(unit.km / unit.s).value
            )
        )
        assert 0.20 < v_rot < 0.40, f"|v_rot| Mainz = {v_rot:.4f} km/s"

    def test_magnitude_smaller_than_v_lab(self, mw_june):
        """Rotation speed must be much less than v_lab."""
        v_rot = float(
            np.linalg.norm(
                mw_june.get_earth_rotation_velocity().to(unit.km / unit.s).value
            )
        )
        v_lab = mw_june.get_v_lab_magnitude().to(unit.km / unit.s).value
        assert v_rot < v_lab / 100

    def test_raises_without_station(self, mw_no_station):
        with pytest.raises(ValueError, match="station"):
            mw_no_station.get_earth_rotation_velocity()


# ---------------------------------------------------------------------------
# Galactic (l, b) of station
# ---------------------------------------------------------------------------


class TestStationGalacticLB:
    def test_shape_and_units(self, mw_june):
        lb = mw_june.get_station_galactic_lb()
        assert lb.shape == (2,)
        assert lb.unit.is_equivalent(unit.deg)

    def test_longitude_in_range(self, mw_june):
        l, b = mw_june.get_station_galactic_lb().to(unit.deg).value
        assert -360 <= l <= 360
        assert -90 <= b <= 90


# ---------------------------------------------------------------------------
# plot() smoke test
# ---------------------------------------------------------------------------


class TestPlot:
    def test_returns_figure(self, mw_june):
        from matplotlib.figure import Figure

        fig = mw_june.plot(show=False)
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_two_3d_axes(self, mw_june):
        fig = mw_june.plot(show=False)
        assert len(fig.axes) == 2
        plt.close("all")

    def test_plot_without_station(self, mw_no_station):
        """plot() should work even without a station projection-axis arrow."""
        fig = mw_no_station.plot(show=False)
        assert len(fig.axes) == 2
        plt.close("all")

    def test_plot_different_stations(self):
        """plot() should succeed for various stations."""
        for station in [Mainz, Baltimore, Sanya]:
            fig = MilkyWay(Time("2024-03-20"), station=station).plot(show=False)
            assert len(fig.axes) == 2
            plt.close("all")
