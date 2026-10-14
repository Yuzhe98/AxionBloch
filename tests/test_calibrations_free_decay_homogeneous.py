"""Homogeneous free-precession and T2 decay convergence calibration."""

import numpy as np
import pytest

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.dependency import PI, ppm, unit
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation


_T1 = 1.0e6 * unit.s
_T2 = 1.0 * unit.s
_DURATION = 3.0 * _T2
_DETUNING = 1.0 * unit.Hz
_RATE_SWEEP_HZ = np.geomspace(100.0, 1.0e5, 8)
_LARMOR_FREQUENCIES = (1.0 * unit.kHz, 1.0 * unit.MHz, 1.0 * unit.GHz)


def _simulate_homogeneous_free_decay(
    larmor_frequency: unit.Quantity, rate_hz: float
) -> float:
    """Return transverse-vector relative L2 error for one rate."""
    sample = Sample(
        name="calibration_homogeneous_free_decay",
        gamma=gamma_p,
        massDensity=0.789 * unit.g / unit.cm**3,
        molarMass=46.069 * unit.g / unit.mol,
        numOfSpinsPerMolecule=6 * unit.one,
        T1=_T1,
        T2=_T2,
        vol=1.0 * unit.cm**3,
        mu=mu_p,
        temp=300.0 * unit.K,
        verbose=False,
    )
    rcf_frequency = larmor_frequency + _DETUNING
    magnet = Magnet(
        name="calibration_homogeneous_free_decay_magnet",
        B0=larmor_frequency / (sample.gamma / (2 * PI)),
        FWHM=0.1 * ppm,
        numPt=1,
        nFWHM=0.0,
    )
    simulation = Simulation(
        name=f"free_decay_{larmor_frequency.to_value(unit.Hz):g}Hz_{rate_hz:g}Hz",
        sample=sample,
        magnet=magnet,
        excField=MagField(name="zero_field"),
        RCF_freq=rcf_frequency,
        rate=rate_hz * unit.Hz,
        duration=_DURATION,
        init_M=1.0 * unit.one,
        init_M_theta=90.0 * unit.deg,
        init_M_phi=90.0 * unit.deg,
        verbose=False,
    )
    # One packet removes field spread; only its known carrier detuning remains.
    simulation.magnet.setHomogeneity(numPt=1)
    simulation.excField.B_vec = np.zeros((simulation.timeLen - 1, 3)) * unit.T
    simulation.generateTrajectories(integrator="RK4")

    time_s = simulation.getTimeStamp().to_value(unit.s)
    actual = simulation.trjry[0, :, 0] + 1j * simulation.trjry[0, :, 1]
    detuning_hz = (larmor_frequency - rcf_frequency).to_value(unit.Hz)
    expected = 1j * np.exp(
        -2j * np.pi * detuning_hz * time_s - time_s / _T2.to_value(unit.s)
    )
    return float(np.sqrt(np.sum(np.abs(actual - expected) ** 2) / np.sum(np.abs(expected) ** 2)))


@pytest.mark.parametrize(
    "larmor_frequency",
    _LARMOR_FREQUENCIES,
    ids=lambda frequency: f"{frequency.to_value(unit.Hz):g}Hz",
)
def test_homogeneous_free_decay_converges_with_rate(larmor_frequency):
    """Off-resonance phase and T2 envelope match the exact free-decay law."""
    errors = np.array(
        [
            _simulate_homogeneous_free_decay(larmor_frequency, rate)
            for rate in _RATE_SWEEP_HZ
        ]
    )

    # At 1 GHz, representing B0 and subtracting a 1 Hz carrier offset creates
    # a small floating-point floor in the requested detuning.
    assert errors[-1] < 2e-7, f"High-rate free-decay error was {errors[-1]:.3e}"
    assert errors[-1] < max(errors[0] * 2e-4, 1e-7), (
        f"Free-decay error did not converge at {larmor_frequency}: "
        f"{errors[0]:.3e} -> {errors[-1]:.3e}"
    )
