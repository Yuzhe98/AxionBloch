"""Weak continuous-wave drive calibration for a homogeneous spin packet."""

import numpy as np
import pytest
from scipy.linalg import eig

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.dependency import PI, ppm, unit
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation


_T1 = 1.0e6 * unit.s
_T2 = 0.1 * unit.s
_DURATION = 5.0 * _T2
_TIP_SCALE = 1.0e-3 * unit.rad
_RATE_SWEEP_HZ = np.geomspace(100.0, 1.0e5, 8)
_LARMOR_FREQUENCIES = (1.0 * unit.kHz, 1.0 * unit.MHz, 1.0 * unit.GHz)


def _simulate_cw_buildup(larmor_frequency: unit.Quantity, rate_hz: float) -> float:
    """Return transverse-vector relative L2 error for weak resonant CW drive."""
    sample = Sample(
        name="calibration_homogeneous_cw",
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
    magnet = Magnet(
        name="calibration_homogeneous_cw_magnet",
        B0=larmor_frequency / (sample.gamma / (2 * PI)),
        FWHM=0.1 * ppm,
        numPt=1,
        nFWHM=0.0,
    )
    # The field uses peak-amplitude convention; the effective rotating field
    # is B1/2. The weak drive keeps longitudinal depletion negligible.
    b1_peak = (2.0 * _TIP_SCALE / (gamma_p * _T2)).to(unit.T)
    simulation = Simulation(
        name=f"cw_{larmor_frequency.to_value(unit.Hz):g}Hz_{rate_hz:g}Hz",
        sample=sample,
        magnet=magnet,
        excField=MagField(name="weak_cw_drive"),
        RCF_freq=larmor_frequency,
        rate=rate_hz * unit.Hz,
        duration=_DURATION,
        verbose=False,
    )
    simulation.magnet.setHomogeneity(numPt=1)
    simulation.excField.setXYPulse(
        timeStep=simulation.timeStep,
        timeLen=simulation.timeLen,
        B1=b1_peak,
        nu_rot=0.0 * unit.Hz,
    )
    simulation.generateTrajectories(integrator="RK4")

    time_s = simulation.getTimeStamp().to_value(unit.s)
    actual = simulation.trjry[0, :, :2]
    # Exact constant-field Bloch propagation, including longitudinal T1
    # recovery and depletion, evaluated at every simulation timestamp.
    gamma_value = gamma_p.to_value(unit.rad * unit.Hz / unit.T)
    b1_eff_t = 0.5 * b1_peak.to_value(unit.T)
    r1 = 1.0 / _T1.to_value(unit.s)
    r2 = 1.0 / _T2.to_value(unit.s)
    omega_x = gamma_value * b1_eff_t
    generator = np.array(
        [
            [-r2, 0.0, 0.0, 0.0],
            [0.0, -r2, omega_x, 0.0],
            [0.0, -omega_x, -r1, r1],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    eigenvalues, eigenvectors = eig(generator)
    coefficients = np.linalg.solve(eigenvectors, np.array([0.0, 0.0, 1.0, 1.0]))
    expected_states = eigenvectors @ (
        coefficients[:, None] * np.exp(eigenvalues[:, None] * time_s[None, :])
    )
    expected = expected_states[:2].real.T
    denominator = np.sum(expected**2)
    return float(np.sqrt(np.sum((actual - expected) ** 2) / denominator))


@pytest.mark.parametrize(
    "larmor_frequency",
    _LARMOR_FREQUENCIES,
    ids=lambda frequency: f"{frequency.to_value(unit.Hz):g}Hz",
)
def test_homogeneous_cw_buildup_converges_with_rate(larmor_frequency):
    """Weak on-resonance CW response matches the analytic buildup curve."""
    errors = np.array(
        [_simulate_cw_buildup(larmor_frequency, rate) for rate in _RATE_SWEEP_HZ]
    )

    assert errors[-1] < 1e-8, f"High-rate CW error was {errors[-1]:.3e}"
    assert errors[-1] < errors[0] * 2e-4, (
        f"CW error did not converge at {larmor_frequency}: "
        f"{errors[0]:.3e} -> {errors[-1]:.3e}"
    )
