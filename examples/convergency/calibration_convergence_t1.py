"""Rate-convergence demonstration for the T1 calibration.

Run from the repository root with::

    python examples/convergency/calibration_convergence_t1.py

The theoretical recovery is anchored at the simulated post-pulse Mz value,
so the metric tests the T1 evolution after the finite-duration pulse. The
transverse linewidth, packet count, pulse length, and rate sweep match the
fixed-numPt free-decay convergence study.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.dependency import PI, ppm, unit
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation


RCF_FREQ = 1.0 * unit.MHz
T1 = 1.0 * unit.s
T2 = 0.1 * T1
FWHM = 0.1 * ppm
N_FWHM = 10.0
NUM_PT = 400
T90_STEPS = 5
DURATION = 5.0 * T1  # Retain a 5*T1 observation window for this T1-specific study.
RATES_HZ = [250.0, 500.0, 1000.0, 2000.0, 4000.0]


def run_case(rate_hz: float) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
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
    simu.magnet.setHomogeneity(numPt=NUM_PT)
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
    return (
        float(np.sqrt(epsilon_l2_squared)),
        float(epsilon_l2_squared),
        t_sig,
        mz_simu,
        mz_theo,
    )


if __name__ == "__main__":
    errors = [run_case(rate) for rate in RATES_HZ]
    for rate, result in zip(RATES_HZ, errors):
        epsilon_l2, epsilon_l2_squared = result[:2]
        print(
            f"rate={rate:7.1f} Hz  "
            f"epsilon_l2={epsilon_l2:.6e}  "
            f"epsilon_l2_squared={epsilon_l2_squared:.6e}"
        )

    fig, (ax_mz, ax_err) = plt.subplots(1, 2, figsize=(12, 4.5))
    for rate, result in zip(RATES_HZ, errors):
        if rate in (RATES_HZ[0], RATES_HZ[-1]):
            _, _, t_sig, mz_simu, mz_theo = result
            ax_mz.plot(t_sig, mz_simu, label=f"{rate:g} Hz simulation")
            if rate == RATES_HZ[-1]:
                ax_mz.plot(t_sig, mz_theo, "k--", label="T1 theory")
    ax_mz.set(xlabel="Time after pulse (s)", ylabel="$M_z$", title="$T_1$ recovery")
    ax_mz.grid(True, alpha=0.3)
    ax_mz.legend()
    ax_err.loglog(RATES_HZ, [item[0] for item in errors], "o-")
    ax_err.set(xlabel="Simulation rate (Hz)", ylabel=r"Relative $T_1$ $\epsilon_{L^2}$",
               title="Rate convergence")
    ax_err.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"T1 calibration: FWHM=0.1 ppm, numPt={NUM_PT}")
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("calibration_convergence_t1.png"), dpi=180)
    plt.show()
