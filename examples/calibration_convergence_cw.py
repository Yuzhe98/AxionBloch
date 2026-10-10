"""Rate-convergence demonstration for the CW full-vector calibration.

Run from the repository root with::

    python examples/calibration_convergence_cw.py
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
TIP_ANGLE = 1.0e-3 * unit.rad
RATES_HZ = [250.0, 500.0, 1000.0, 2000.0, 4000.0]


def run_case(rate_hz: float) -> tuple[float, float]:
    sample = Sample(
        name="cw_convergence",
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
        name="cw_convergence_magnet",
        B0=B0,
        FWHM=FWHM,
        nFWHM=10.0,
    )
    b1_input = (2.0 * TIP_ANGLE / (gamma_p * (1.0 / (np.pi * FWHM * RCF_FREQ)))).to(unit.T)
    simu = Simulation(
        name=f"cw_rate_{rate_hz:g}Hz",
        sample=sample,
        magnet=magnet,
        excField=MagField(name="cw_drive"),
        RCF_freq=RCF_FREQ,
        rate=rate_hz * unit.Hz,
        duration=DURATION,
        verbose=False,
    )
    simu.excField.setXYPulse(
        timeStep=simu.timeStep,
        timeLen=simu.timeLen,
        B1=b1_input,
        nu_rot=DELTA_NU_L,
    )
    simu.generateTrajectories(integrator="RK4")

    t = simu.getTimeStamp().to_value(unit.s)
    dt = t[1] - t[0]
    delta_nu_i = (
        gamma_p.to_value(unit.rad * unit.Hz / unit.T) / (2 * np.pi)
        * simu.magnet.B_spread.to_value(unit.T)
        - (simu.RCF_freq.to_value(unit.Hz) - DELTA_NU_L.to_value(unit.Hz))
    )
    fd = np.zeros_like(t, dtype=complex)
    for detuning_i, weight in zip(delta_nu_i, simu.magnet.ratios):
        fd += weight * np.exp(2j * np.pi * detuning_i * t - t / T2.to_value(unit.s))
    b1_eff = 0.5 * b1_input.to_value(unit.T)
    z_theo = (
        1j
        * np.exp(2j * np.pi * DELTA_NU_L.to_value(unit.Hz) * t)
        * gamma_p.to_value(unit.rad * unit.Hz / unit.T)
        * b1_eff
        * np.cumsum(fd)
        * dt
    )
    m_simu = simu.trjry[0, :, :2]
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
    plt.title("CW calibration convergence")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()
