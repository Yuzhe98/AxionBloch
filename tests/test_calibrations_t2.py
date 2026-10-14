"""Single-process Bloch calibration checks with analytic references.

These tests isolate one dynamic at a time. The first case checks transverse
T2 decay in a homogeneous, on-resonance field, then repeats the rate
convergence sweep at several Larmor frequencies.
"""

import numpy as np
import pytest

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.dependency import PI, ppm, unit
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation


_T1 = 1.0e6 * unit.s  # Suppress longitudinal relaxation during this test.
_T2 = 1.0 * unit.s
_DURATION = 5.0 * _T2
_RATE_SWEEP_HZ = np.geomspace(100.0, 1.0e5, 8)
_LARMOR_FREQUENCIES = (1.0 * unit.kHz, 1.0 * unit.MHz, 1.0 * unit.GHz)


def _simulate_t2_decay(larmor_frequency: unit.Quantity, rate_hz: float) -> float:
    """Return relative L2 error for pure transverse T2 decay at one rate."""
    sample = Sample(
        name="calibration_t2_decay",
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
        name="calibration_t2_homogeneous_magnet",
        B0=larmor_frequency / (sample.gamma / (2 * PI)),
        FWHM=0.1 * ppm,
        numPt=1,
        nFWHM=0.0,
    )
    simulation = Simulation(
        name=f"t2_{larmor_frequency.to_value(unit.Hz):g}Hz_{rate_hz:g}Hz",
        sample=sample,
        magnet=magnet,
        excField=MagField(name="zero_field"),
        RCF_freq=larmor_frequency,
        rate=rate_hz * unit.Hz,
        duration=_DURATION,
        init_M=1.0 * unit.one,
        init_M_theta=90.0 * unit.deg,
        init_M_phi=90.0 * unit.deg,
        verbose=False,
    )
    # nFWHM=0 and one packet isolate intrinsic T2 relaxation from field spread.
    simulation.magnet.setHomogeneity(numPt=1)
    simulation.excField.B_vec = np.zeros((simulation.timeLen - 1, 3)) * unit.T
    simulation.generateTrajectories(integrator="RK4")

    time_s = simulation.getTimeStamp().to_value(unit.s)
    my_simulated = simulation.trjry[0, :, 1]
    my_expected = np.exp(-time_s / _T2.to_value(unit.s))
    return float(np.sqrt(np.sum((my_simulated - my_expected) ** 2) / np.sum(my_expected**2)))


@pytest.mark.parametrize(
    "larmor_frequency",
    _LARMOR_FREQUENCIES,
    ids=lambda frequency: f"{frequency.to_value(unit.Hz):g}Hz",
)
def test_pure_t2_decay_converges_with_rate_at_larmor_frequency(larmor_frequency):
    """On-resonance T2 decay follows exp(-t/T2) as RK4 rate increases."""
    errors = np.array(
        [_simulate_t2_decay(larmor_frequency, rate) for rate in _RATE_SWEEP_HZ]
    )

    assert errors[-1] < 1e-8, f"High-rate T2 error was {errors[-1]:.3e}"
    assert errors[-1] < errors[0] * 2e-4, (
        f"T2 error did not converge at {larmor_frequency}: "
        f"{errors[0]:.3e} -> {errors[-1]:.3e}"
    )
