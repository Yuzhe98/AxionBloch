"""Hard-pulse tip calibration and rate convergence over a 360-degree sweep."""

import numpy as np
import pytest
from scipy.linalg import expm

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.dependency import PI, ppm, unit
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation


_RCF_FREQ = 1.0 * unit.MHz
_T1 = 1.0e6 * unit.s
_T2 = 1.0e5 * unit.s
_PULSE_DURATION_S = 0.1
_RATE_SWEEP_HZ = np.geomspace(100.0, 1.0e5, 8)
_TIP_ANGLES_DEG = np.linspace(0.0, 360.0, 50)
_LARMOR_FREQUENCIES = (1.0 * unit.kHz, 1.0 * unit.MHz, 1.0 * unit.GHz)


def _simulate_pulse_sweep(
    larmor_frequency: unit.Quantity, rate_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return simulated and exact continuous-time states at pulse end."""
    pulse_steps = int(round(_PULSE_DURATION_S * rate_hz))
    dt_s = 1.0 / rate_hz
    sample = Sample(
        name="calibration_hard_pulse",
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
        name="calibration_pulse_homogeneous_magnet",
        B0=larmor_frequency / (sample.gamma / (2 * PI)),
        FWHM=0.1 * ppm,
        numPt=1,
        nFWHM=0.0,
    )
    # Keep exactly pulse_steps integration intervals, including at the
    # minimum rate where the Simulation constructor enforces a five-step floor.
    total_steps = max(pulse_steps, 5)
    simulation = Simulation(
        name=f"pulse_{larmor_frequency.to_value(unit.Hz):g}Hz_{rate_hz:g}Hz",
        sample=sample,
        magnet=magnet,
        excField=MagField(name="hard_x_pulse"),
        RCF_freq=larmor_frequency,
        rate=rate_hz * unit.Hz,
        duration=(total_steps + 1) * dt_s * unit.s,
        verbose=False,
    )
    simulation.magnet.setHomogeneity(numPt=1)

    states = []
    theory = []
    for angle_deg in _TIP_ANGLES_DEG:
        simulation.excField.setXYPulse_new(
            timeStep=dt_s * unit.s,
            timeLen=simulation.timeLen,
            gamma=gamma_p,
            pulseSteps=pulse_steps,
            nu_rot=0.0 * unit.Hz,
            tip_angle=angle_deg * unit.deg,
        )
        simulation.generateTrajectories(integrator="RK4")
        states.append(simulation.trjry[0, pulse_steps, :3].copy())

        # Analytic Bloch propagation uses the requested physical flip angle,
        # not the discrete-RK4 amplitude correction used by the pulse helper.
        angle_rad = np.deg2rad(angle_deg)
        omega_x = angle_rad / _PULSE_DURATION_S
        r1 = 1.0 / _T1.to_value(unit.s)
        r2 = 1.0 / _T2.to_value(unit.s)
        generator = np.array(
            [
                [-r2, 0.0, 0.0, 0.0],
                [0.0, -r2, omega_x, 0.0],
                [0.0, -omega_x, -r1, r1],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        theory.append((expm(generator * _PULSE_DURATION_S) @ [0.0, 0.0, 1.0, 1.0])[:3])
    return np.asarray(states), np.asarray(theory)


@pytest.mark.parametrize(
    "larmor_frequency",
    _LARMOR_FREQUENCIES,
    ids=lambda frequency: f"{frequency.to_value(unit.Hz):g}Hz",
)
def test_hard_pulse_calibration_converges_over_tip_sweep(larmor_frequency):
    """A 50-angle 360-degree sweep converges at each Larmor frequency."""
    errors = []
    for rate_hz in _RATE_SWEEP_HZ:
        simulated, expected = _simulate_pulse_sweep(larmor_frequency, rate_hz)
        # Normalize vector L2 residual by the nonzero transverse reference
        # norm, so near-zero Mx values do not create a singular metric.
        denominator = np.sum(expected[:, 1] ** 2)
        errors.append(
            np.sqrt(np.sum((simulated[:, :2] - expected[:, :2]) ** 2) / denominator)
        )
    errors = np.asarray(errors)

    assert np.all(np.isfinite(errors))
    assert errors[-1] < 1e-8, f"High-rate pulse error was {errors[-1]:.3e}"
    assert errors[-1] < errors[0] * 2e-4, (
        f"Pulse error did not converge at {larmor_frequency}: "
        f"{errors[0]:.3e} -> {errors[-1]:.3e}"
    )
