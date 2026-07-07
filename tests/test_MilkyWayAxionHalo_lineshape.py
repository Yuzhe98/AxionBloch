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

    lineshape = axion.findLineshapeAtStationAndTime(
        station=Boston,
        meas_time=meas_time,
        case="grad_perp",
        sensitive_axis="zenith",
        num_frequency_points=2001,
    )
    assert lineshape["lineshape"].shape == lineshape["frequencies"].shape
    assert lineshape["power_spectrum"].shape == lineshape["frequencies"].shape
    assert lineshape["alpha"].unit.is_equivalent(unit.rad)
    assert lineshape["v_lab_magnitude"][0].unit.is_equivalent(unit.km / unit.s)

    fwhm = axion.findLineshapeFWHMAtStation(
        station=Boston,
        meas_time=meas_time,
        case="grad_perp",
        sensitive_axis="zenith",
        num_frequency_points=4001,
    )
    assert fwhm["FWHM_frequency"] > 0 * unit.Hz
    assert fwhm["FWHM_a"].unit.is_equivalent(ppm)
    assert fwhm["tau_a"] > 0 * unit.s
    assert axion.FWHM_frequency == fwhm["FWHM_frequency"]
    assert axion.FWHM_a == fwhm["FWHM_a"]
    assert axion.tau_a == fwhm["tau_a"]

    tau_from_width = (
        1.0 / (np.pi * fwhm["FWHM_fraction"] * axion.nu_a.to(unit.Hz))
    ).to(unit.s)
    assert np.isclose(
        fwhm["tau_a"].to_value(unit.s),
        tau_from_width.to_value(unit.s),
        rtol=1e-12,
    )

    fwhm_power = axion.findLineshapeFWHMAtStation(
        station=Boston,
        meas_time=meas_time,
        frequencies=fwhm["frequencies"],
        case="grad_perp",
        sensitive_axis="zenith",
        spectrum="power_spectrum",
        update=False,
    )
    assert np.isclose(
        fwhm["FWHM_frequency"].to_value(unit.Hz),
        fwhm_power["FWHM_frequency"].to_value(unit.Hz),
        rtol=1e-12,
    )
