"""Pure longitudinal T1 relaxation calibration across frequency and rate."""

import numpy as np
import pytest

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.dependency import PI, ppm, unit
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation


_T1 = 1.0 * unit.s
_T2 = 0.5 * _T1
_DURATION = 5.0 * _T1
_RATE_SWEEP_HZ = np.geomspace(100.0, 1.0e5, 8)
_LARMOR_FREQUENCIES = (1.0 * unit.kHz, 1.0 * unit.MHz, 1.0 * unit.GHz)


def _simulate_t1_decay(
    larmor_frequency: unit.Quantity, rate_hz: float, initial_mz: float
) -> float:
    """Return relative L2 error for longitudinal relaxation from initial_mz."""
    sample = Sample(
        name="calibration_t1_decay",
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
        name="calibration_t1_homogeneous_magnet",
        B0=larmor_frequency / (sample.gamma / (2 * PI)),
        FWHM=0.01 * ppm,
        numPt=1,
        nFWHM=0.0,
    )
    simulation = Simulation(
        name=(
            f"t1_{larmor_frequency.to_value(unit.Hz):g}Hz_"
            f"{rate_hz:g}Hz_Mz{initial_mz:g}"
        ),
        sample=sample,
        magnet=magnet,
        excField=MagField(name="zero_field"),
        RCF_freq=larmor_frequency,
        rate=rate_hz * unit.Hz,
        duration=_DURATION,
        init_M=initial_mz * unit.one,
        init_M_theta=0.0 * unit.deg,
        init_M_phi=0.0 * unit.deg,
        verbose=False,
    )
    simulation.magnet.setHomogeneity(numPt=1)
    simulation.excField.B_vec = np.zeros((simulation.timeLen - 1, 3)) * unit.T
    simulation.generateTrajectories(integrator="RK4")

    time_s = simulation.getTimeStamp().to_value(unit.s)
    mz_simulated = simulation.trjry[0, :, 2]
    mz_expected = 1.0 + (initial_mz - 1.0) * np.exp(
        -time_s / _T1.to_value(unit.s)
    )
    return float(
        np.sqrt(np.sum((mz_simulated - mz_expected) ** 2) / np.sum(mz_expected**2))
    )


@pytest.mark.parametrize(
    "larmor_frequency",
    _LARMOR_FREQUENCIES,
    ids=lambda frequency: f"{frequency.to_value(unit.Hz):g}Hz",
)
@pytest.mark.parametrize("initial_mz", (0.0, 1.0, 1.0e2, 1.0e4), ids=("0", "1", "1e2", "1e4"))
def test_pure_t1_decay_converges_with_rate_at_larmor_frequency(
    larmor_frequency, initial_mz
):
    """T1 evolution from below, at, and above equilibrium follows its exact law."""
    errors = np.array(
        [
            _simulate_t1_decay(larmor_frequency, rate, initial_mz)
            for rate in _RATE_SWEEP_HZ
        ]
    )

    assert errors[-1] < 1e-8, f"High-rate T1 error was {errors[-1]:.3e}"
    if initial_mz != 1.0:
        assert errors[-1] < errors[0] * 2e-4, (
            f"T1 error did not converge at {larmor_frequency}, Mz(0)={initial_mz:g}: "
            f"{errors[0]:.3e} -> {errors[-1]:.3e}"
        )
