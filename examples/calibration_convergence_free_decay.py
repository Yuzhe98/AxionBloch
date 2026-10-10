"""Rate-convergence demonstration for the full transverse-vector calibration.

Run from the repository root with::

    python examples/calibration_convergence_free_decay.py

The theoretical curve is initialized with the simulated transverse vector at
the end of the finite-duration pulse.  This isolates convergence of the
post-pulse free evolution, whose theory is not intended to describe the pulse
itself.
"""

import matplotlib.pyplot as plt
import numpy as np

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.dependency import PI, ppm, unit
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation


RCF_FREQ = 1.0 * unit.MHz
DELTA_NU_L = 1.0 * unit.Hz
FWHM = 1.0 * ppm
T1 = 1.0e6 * unit.s
T2 = 10.0 * unit.s
DURATION = 3.0 * unit.s
T90_STEPS = 5
RATES_HZ = [250.0, 500.0, 1000.0, 2000.0, 4000.0]


def run_case(rate_hz: float) -> tuple[float, float]:
    sample = Sample(
        name="free_decay_convergence",
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
    B0 = (RCF_FREQ - DELTA_NU_L) / (sample.gamma / (2 * PI))
    magnet = Magnet(
        name="free_decay_convergence_magnet",
        B0=B0,
        FWHM=FWHM,
        nFWHM=10.0,
    )
    simu = Simulation(
        name=f"free_decay_rate_{rate_hz:g}Hz",
        sample=sample,
        magnet=magnet,
        excField=MagField(name="90deg_pulse"),
        RCF_freq=RCF_FREQ,
        rate=rate_hz * unit.Hz,
        duration=DURATION,
        verbose=False,
    )
    simu.excField.set90DegPulse(
        timeStep=simu.timeStep,
        timeLen=simu.timeLen,
        gamma=simu.sample.gamma,
        t90=T90_STEPS * simu.timeStep,
        nu_rot=DELTA_NU_L,
    )
    simu.generateTrajectories(integrator="RK4")

    t = simu.getTimeStamp().to_value(unit.s)
    t_sig = t[T90_STEPS:] - t[T90_STEPS]
    m_simu = simu.trjry[0, T90_STEPS:, :2]
    z0 = complex(m_simu[0, 0], m_simu[0, 1])
    gamma_over_2pi = gamma_p.to_value(unit.rad * unit.Hz / unit.T) / (2 * np.pi)
    detuning = (
        gamma_over_2pi * simu.magnet.B_spread.to_value(unit.T)
        - simu.RCF_freq.to_value(unit.Hz)
    )
    z_theo = np.zeros_like(t_sig, dtype=complex)
    for detuning_i, weight in zip(detuning, simu.magnet.ratios):
        z_theo += weight * np.exp(-2j * np.pi * detuning_i * t_sig)
    z_theo *= z0 * np.exp(-t_sig / T2.to_value(unit.s))
    m_theo = np.column_stack((z_theo.real, z_theo.imag))
    epsilon_l2_squared = np.sum((m_simu - m_theo) ** 2) / np.sum(m_theo**2)
    return float(np.sqrt(epsilon_l2_squared)), float(epsilon_l2_squared)


if __name__ == "__main__":
    errors = [run_case(rate) for rate in RATES_HZ]
    for rate, (epsilon_l2, epsilon_l2_squared) in zip(RATES_HZ, errors):
        print(
            f"rate={rate:7.1f} Hz  "
            f"epsilon_l2={epsilon_l2:.6e}  "
            f"epsilon_l2_squared={epsilon_l2_squared:.6e}"
        )

    plt.figure(figsize=(8.5 / 2.54, 0.65 * 8.5 / 2.54))
    plt.loglog(RATES_HZ, [item[0] for item in errors], "o", linestyle="none")
    plt.xlabel("Simulation rate (Hz)")
    plt.ylabel(r"$\epsilon_{L^2}$")
    plt.title("Free-decay calibration convergence")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()
