"""Rate-convergence demonstration for the T1 calibration.

Run from the repository root with::

    python examples/calibration_convergence_t1.py

The theoretical recovery is anchored at the simulated post-pulse Mz value,
so the metric tests the T1 evolution after the finite-duration pulse.
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
T2 = 0.1 * T1
FWHM = 1.0 * ppm
N_FWHM = 0.0
T90_STEPS = 5
DURATION = 5.0 * T1
RATES_HZ = [200.0, 400.0, 800.0, 1600.0, 3200.0]


def run_case(rate_hz: float) -> tuple[float, float]:
    sample = Sample(
        name="t1_convergence",
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
        name="t1_convergence_magnet",
        B0=B0,
        FWHM=FWHM,
        nFWHM=N_FWHM,
    )
    simu = Simulation(
        name=f"t1_rate_{rate_hz:g}Hz",
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
        nu_rot=0.0 * unit.Hz,
    )
    simu.generateTrajectories(integrator="RK4")

    t = simu.getTimeStamp().to_value(unit.s)
    t_sig = t[T90_STEPS:] - t[T90_STEPS]
    mz_simu = simu.trjry[0, T90_STEPS:, 2]
    mz0 = float(mz_simu[0])
    mz_theo = 1.0 + (mz0 - 1.0) * np.exp(-t_sig / T1.to_value(unit.s))
    departure = mz_theo - 1.0
    epsilon_l2_squared = np.sum((mz_simu - mz_theo) ** 2) / np.sum(departure**2)
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
    plt.ylabel(r"Relative $T_1$ $\epsilon_{L^2}$")
    plt.title(r"$T_1$ calibration convergence")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()
