"""Inversion-recovery T1 measurement using two hard pulses.

The sequence is

    pi -- wait(tau) -- pi/2 -- readout,

repeated for a range of wait times.  The transverse amplitude after the
readout pulse is plotted against ``tau`` and compared with the expected
inversion-recovery curve.

Run from the repository root with::

    python examples/T1_inversion_recovery.py
"""

import matplotlib.pyplot as plt
import numpy as np

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
RATE = 5.0 * unit.kHz
PULSE_STEPS = 10
WAIT_TIMES = np.linspace(0.0, 5.0, 21) * unit.s


def simulate_readout(wait_time) -> tuple[float, float]:
    """Run one ``pi - wait - pi/2`` sequence and return signed/readout amplitudes."""
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
    # Leave a short trailing acquisition window after the pi/2 pulse so the
    # readout sample is always inside the stored trajectory.
    duration = wait_time + 2.0 * pulse_duration + 10.0 / RATE
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
    readout_start = pulse_len + wait_len
    readout_index = readout_start + pulse_len

    # Build the two x-axis hard pulses explicitly.  This makes the wait
    # interval unambiguous and places the pi/2 readout at a known sample.
    b90 = PI / (simu.sample.gamma * (pulse_len * simu.timeStep))
    b180 = 2.0 * b90
    num_steps = simu.timeLen - 1
    B_vec = np.zeros((1, num_steps, 3)) * unit.T
    B_vec[0, :pulse_len, 0] = 0.5 * b180
    B_vec[0, readout_start:readout_index, 0] = 0.5 * b90
    simu.excField.B_vec = B_vec
    simu.generateTrajectories(integrator="RK4")

    m_readout = simu.trjry[0, readout_index, :2]
    signed_signal = float(m_readout[1])
    amplitude = float(np.linalg.norm(m_readout))
    return signed_signal, amplitude


if __name__ == "__main__":
    results = np.array([simulate_readout(wait) for wait in WAIT_TIMES])
    signed_signal = results[:, 0]
    amplitude = results[:, 1]
    wait_s = WAIT_TIMES.to_value(unit.s)
    expected_signed = 1.0 - 2.0 * np.exp(-wait_s / T1.to_value(unit.s))
    expected_amplitude = np.abs(expected_signed)

    for wait, signal, amp in zip(wait_s, signed_signal, amplitude):
        print(f"wait={wait:6.3f} s  signed_My={signal:+.6e}  amplitude={amp:.6e}")

    plt.figure(figsize=(8.5 / 2.54, 0.65 * 8.5 / 2.54))
    plt.scatter(wait_s, amplitude, label=r"Measured $|M_\perp|$", s=16)
    plt.scatter(
        wait_s,
        expected_amplitude,
        marker="x",
        label=r"Expected $|1-2e^{-\tau/T_1}|$",
    )
    plt.xlabel(r"Wait time $\tau$ (s)")
    plt.ylabel(r"Readout amplitude $|M_\perp|$")
    plt.title(r"$T_1$ inversion-recovery measurement")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
