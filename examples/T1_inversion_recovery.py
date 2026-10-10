"""Inversion-recovery T1 measurement with a discrete hard 180-degree pulse.

The sequence is

    pi -- wait(tau) -- measure Mz,

repeated for a range of wait times. The theory includes T1/T2 relaxation
during the finite inversion pulse. The lower plot shows the signed error
``Mz_simulation - Mz_theory`` directly.

Run from the repository root with::

    python examples/T1_inversion_recovery.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.dependency import PI, ppm, unit
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation


RCF_FREQ = 1.0 * unit.MHz
T1 = 1.0 * unit.s
T2 = 0.1 * unit.s
FWHM = 1.0 * ppm
N_FWHM = 0.0
RATE = 1.0 * unit.kHz
PULSE_STEPS = 10
WAIT_TIMES = np.linspace(0.0, 5.0, 21) * unit.s


def _apply_x_pulse(
    state: np.ndarray, bx_t: float, pulse_duration_s: float
) -> np.ndarray:
    """Apply the exact affine Bloch evolution for one constant x pulse."""
    gamma = gamma_p.to_value(unit.rad * unit.Hz / unit.T)
    r1 = 1.0 / T1.to_value(unit.s)
    r2 = 1.0 / T2.to_value(unit.s)
    omega_x = gamma * bx_t
    generator = np.array(
        [
            [-r2, 0.0, 0.0, 0.0],
            [0.0, -r2, omega_x, 0.0],
            [0.0, -omega_x, -r1, r1],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    affine_state = np.append(state, 1.0)
    return (expm(generator * pulse_duration_s) @ affine_state)[:3]


def _predict_mz_after_inversion(
    wait_duration_s: float, b180_component_t: float
) -> float:
    """Predict Mz after inversion and wait, including pulse relaxation."""
    pulse_duration_s = PULSE_STEPS / RATE.to_value(unit.Hz)
    state = _apply_x_pulse(
        np.array([0.0, 0.0, 1.0]), b180_component_t, pulse_duration_s
    )
    mz_after_wait = 1.0 + (state[2] - 1.0) * np.exp(
        -wait_duration_s / T1.to_value(unit.s)
    )
    return float(mz_after_wait)


def simulate_mz(wait_time) -> tuple[float, float]:
    """Run one inversion-recovery case and return simulated/theory Mz."""
    sample = Sample(
        name="T1_inversion_recovery",
        gamma=gamma_p,
        massDensity=0.789 * unit.g / unit.cm**3,
        molarMass=46.069 * unit.g / unit.mol,
        numOfSpinsPerMolecule=6 * unit.one,
        T1=T1,
        T2=T2,
        vol=1.0 * unit.cm**3,
        mu=mu_p,
        temp=300.0 * unit.K,
        verbose=False,
    )
    B0 = RCF_FREQ / (sample.gamma / (2 * PI))
    magnet = Magnet(
        name="T1_inversion_recovery_magnet",
        B0=B0,
        FWHM=FWHM,
        nFWHM=N_FWHM,
    )
    pulse_duration = PULSE_STEPS / RATE
    # Keep a short trailing window so the pre-readout Mz sample is stored.
    duration = wait_time + pulse_duration + 2.0 / RATE
    simu = Simulation(
        name=f"T1_wait_{wait_time.to_value(unit.s):.3g}s",
        sample=sample,
        magnet=magnet,
        excField=MagField(name="pi_pi_over_2_sequence"),
        RCF_freq=RCF_FREQ,
        rate=RATE,
        duration=duration,
        verbose=False,
    )
    pulse_len = PULSE_STEPS
    wait_len = int(
        np.round(wait_time.to_value(unit.s) * RATE.to_value(unit.Hz))
    )
    readout_index = pulse_len + wait_len

    # Build a discrete-grid calibrated x-axis inversion pulse. Measure Mz at
    # the end of the wait, just before an optional transverse readout pulse.
    b180 = simu.excField._calibratedHardPulseAmplitude(
        timeStep=simu.timeStep,
        gamma=simu.sample.gamma,
        pulseSteps=pulse_len,
        tip_angle=180 * unit.deg,
    )
    num_steps = simu.timeLen - 1
    B_vec = np.zeros((1, num_steps, 3)) * unit.T
    B_vec[0, :pulse_len, 0] = 0.5 * b180
    simu.excField.B_vec = B_vec
    simu.generateTrajectories(integrator="RK4")

    mz_sim = float(simu.trjry[0, readout_index, 2])
    actual_wait_s = wait_len * simu.timeStep.to_value(unit.s)
    mz_theo = _predict_mz_after_inversion(
        actual_wait_s,
        (0.5 * b180).to_value(unit.T),
    )
    return mz_sim, mz_theo


if __name__ == "__main__":
    results = np.array([simulate_mz(wait) for wait in WAIT_TIMES])
    mz_sim = results[:, 0]
    mz_theo = results[:, 1]
    wait_s = WAIT_TIMES.to_value(unit.s)
    ideal_signed = 1.0 - 2.0 * np.exp(-wait_s / T1.to_value(unit.s))
    mz_error = mz_sim - mz_theo
    print(
        "Maximum absolute Mz error (simulation - pulse-aware theory): "
        f"{np.max(np.abs(mz_error)):.3e}"
    )
    for wait, sim, theory, error in zip(wait_s, mz_sim, mz_theo, mz_error):
        print(f"wait={wait:6.3f} s  Mz_sim={sim:+.7f}  Mz_theory={theory:+.7f}  error={error:+.3e}")

    fig, (ax_mz, ax_error) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
    ax_mz.plot(wait_s, mz_sim, "o", label="Simulation")
    ax_mz.plot(wait_s, mz_theo, "-", label="Pulse-aware theory")
    ax_mz.plot(wait_s, ideal_signed, "--", label="Ideal-pulse formula")
    ax_mz.set(
        xlabel=r"Wait time $\tau$ (s)",
        ylabel=r"$M_z$",
        title="T1 inversion recovery",
    )
    ax_mz.axhline(0.0, color="k", linewidth=0.7, alpha=0.5)
    ax_mz.grid(True, alpha=0.3)
    ax_mz.legend(frameon=False)
    ax_error.axhline(0.0, color="k", linewidth=0.7, alpha=0.5)
    ax_error.plot(wait_s, mz_error, "o-")
    ax_error.set(
        xlabel=r"Wait time $\tau$ (s)",
        ylabel=r"$M_{z,sim} - M_{z,theory}$",
        title="Signed prediction error",
    )
    ax_error.grid(True, alpha=0.3)
    fig.suptitle(r"$T_1$ inversion recovery: 10 ms inversion at 1 kHz")
    fig.tight_layout()
    fig.savefig("examples/T1_inversion_recovery.png", dpi=180)
    plt.show()
