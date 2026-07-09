"""Focused tests for MW axion PSD and FWHM station helpers."""

import numpy as np
from astropy.time import Time
from astropy.utils import iers

from axionbloch.dependency import ppm, unit
from axionbloch.MilkyWayAxionHalo import MilkyWayAxionHalo
from axionbloch.Station import Boston

iers.conf.auto_download = False


def test_station_PSD_and_FWHM_update():
    """Station/time-aware PSD methods return widths and update tau_a."""

    axion = MilkyWayAxionHalo(nu_a=1 * unit.MHz)
    meas_time = Time("2022-01-01T00:00:00", scale="utc")

    lineshape = axion.findLineshape(
        station=Boston,
        meas_time=meas_time,
        case="grad_perp",
        projection_axis="zenith",
        num_frequency_points=2001,
    )
    assert lineshape["lineshape"].shape == lineshape["frequencies"].shape
    assert lineshape["power_spectrum"].shape == lineshape["frequencies"].shape
    assert lineshape["alpha"].unit.is_equivalent(unit.rad)
    assert lineshape["v_lab_magnitude"][0].unit.is_equivalent(unit.km / unit.s)
    assert "FWHM_frequency" not in lineshape
    assert "FWHM_fraction" not in lineshape
    assert "frequency" not in lineshape
    assert "PSD" not in lineshape
    assert lineshape["FWHM_freq"] > 0 * unit.Hz
    assert lineshape["FWHM_freq"].unit == unit.Hz
    assert lineshape["FWHM"] == lineshape["FWHM_a"]
    assert lineshape["FWHM_a"].unit.is_equivalent(ppm)
    assert axion.FWHM_freq == lineshape["FWHM_freq"]

    fwhm = axion.findLineshapeFWHM(
        station=Boston,
        meas_time=meas_time,
        case="grad_perp",
        projection_axis="zenith",
        num_frequency_points=4001,
    )
    assert "FWHM_frequency" not in fwhm
    assert "FWHM_fraction" not in fwhm
    assert fwhm["FWHM_freq"] > 0 * unit.Hz
    assert fwhm["FWHM_freq"].unit == unit.Hz
    assert fwhm["FWHM_a"].unit.is_equivalent(ppm)
    assert fwhm["tau_a"] > 0 * unit.s
    assert axion.FWHM_frequency == fwhm["FWHM_freq"]
    assert axion.FWHM_freq == fwhm["FWHM_freq"]
    assert axion.FWHM == fwhm["FWHM"]
    assert axion.FWHM_a == fwhm["FWHM_a"]
    assert axion.tau_a == fwhm["tau_a"]

    tau_from_width = (1.0 / (np.pi * fwhm["FWHM"] * axion.nu_a.to(unit.Hz))).to(
        unit.s
    )
    assert np.isclose(
        fwhm["tau_a"].to_value(unit.s),
        tau_from_width.to_value(unit.s),
        rtol=1e-12,
    )

    fwhm_power = axion.findLineshapeFWHM(
        station=Boston,
        meas_time=meas_time,
        frequencies=fwhm["frequencies"],
        case="grad_perp",
        projection_axis="zenith",
        spectrum="power_spectrum",
        update=False,
    )
    assert np.isclose(
        fwhm["FWHM_freq"].to_value(unit.Hz),
        fwhm_power["FWHM_freq"].to_value(unit.Hz),
        rtol=1e-12,
    )


def test_FWHM_reuses_cached_station_lineshape(monkeypatch):
    """FWHM uses the stored station/time lineshape when inputs match."""

    axion = MilkyWayAxionHalo(nu_a=1 * unit.MHz)
    meas_time = Time("2022-01-01T00:00:00", scale="utc")

    lineshape = axion.findLineshape(
        station=Boston,
        meas_time=meas_time,
        case="grad_perp",
        projection_axis="zenith",
        num_frequency_points=2001,
        update=False,
    )
    assert axion.lineshapeAtStationAndTimeResult is lineshape

    def raise_if_recomputed(*args, **kwargs):
        raise AssertionError("cached lineshape was not reused")

    monkeypatch.setattr(
        axion,
        "findLineshape",
        raise_if_recomputed,
    )

    fwhm = axion.findLineshapeFWHM(
        station=Boston,
        meas_time=meas_time,
        case="grad_perp",
        projection_axis="zenith",
        num_frequency_points=2001,
    )

    assert fwhm["lineshape"] is lineshape["lineshape"]
    assert fwhm["FWHM_freq"] > 0 * unit.Hz
    assert axion.v_lab == lineshape["v_lab_magnitude"][0]
    assert axion.windAngle == lineshape["alpha"]


def test_kinematics_cache_reuses_matching_inputs(monkeypatch):
    """Repeated kinematics calls reuse the stored result when inputs match."""

    axion = MilkyWayAxionHalo(nu_a=1 * unit.MHz)
    meas_times = Time(
        ["2022-01-01T00:00:00", "2022-01-01T01:00:00"],
        scale="utc",
    )

    kinematics = axion.findKinematicsOverTime(
        station=Boston,
        meas_times=meas_times,
        projection_axis="zenith",
    )
    assert axion.kinematicsOverTimeResult is kinematics

    def raise_if_recomputed(*args, **kwargs):
        raise AssertionError("cached kinematics were not reused")

    monkeypatch.setattr(axion, "projectHaloVelocity", raise_if_recomputed)

    cached = axion.findKinematicsOverTime(
        station=Boston,
        meas_times=meas_times,
        projection_axis="zenith",
    )
    assert cached is kinematics


def test_lineshape_reuses_cached_kinematics_for_new_frequency_grid(monkeypatch):
    """Changing only the frequency grid should not recompute kinematics."""

    axion = MilkyWayAxionHalo(nu_a=1 * unit.MHz)
    meas_time = Time("2022-01-01T00:00:00", scale="utc")

    axion.findLineshape(
        station=Boston,
        meas_time=meas_time,
        case="grad_perp",
        projection_axis="zenith",
        num_frequency_points=1001,
        update=False,
    )

    def raise_if_recomputed(*args, **kwargs):
        raise AssertionError("cached kinematics were not reused")

    monkeypatch.setattr(axion, "projectHaloVelocity", raise_if_recomputed)

    result = axion.findLineshape(
        station=Boston,
        meas_time=meas_time,
        case="grad_perp",
        projection_axis="zenith",
        num_frequency_points=1201,
        update=False,
    )
    assert result["frequencies"].size == 1201
