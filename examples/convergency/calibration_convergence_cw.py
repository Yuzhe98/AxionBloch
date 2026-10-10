"""Rate convergence for a weak CW excitation against a discrete-time theory.

Run from the repository root with::

    python examples/convergency/calibration_convergence_cw.py
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
DELTA_NU_L = 1.0 * unit.Hz
FWHM = 0.1 * ppm
NUM_PT = 400
T1 = 1.0e6 * unit.s
T2 = 10.0 * unit.s
DURATION = 3.0 * unit.s
TIP_ANGLE = 1.0e-3 * unit.rad
RATES_HZ = [250.0, 500.0, 1000.0, 2000.0, 4000.0]


def linear_cw_prediction(magnet: Magnet, pulse_b_t: np.ndarray, dt_s: float) -> np.ndarray:
    """Solve the weak-drive transverse Bloch equation for the sampled CW field.

    For the 1 mrad drive, longitudinal depletion is second order and is
    negligible. Each packet obeys ``dMxy/dt = lambda*Mxy + i*gamma*Bxy``.
    The recurrence below integrates that linear equation exactly over each
    piecewise-constant field sample, including packet detuning and T2.
    """
    gamma = gamma_p.to_value(unit.rad * unit.Hz / unit.T)
    gamma_over_2pi = gamma / (2 * np.pi)
    detuning_hz = (
        gamma_over_2pi * magnet.B_spread.to_value(unit.T)
        - RCF_FREQ.to_value(unit.Hz)
    )
    lambdas = -1.0 / T2.to_value(unit.s) - 2j * np.pi * detuning_hz
    step_factor = np.exp(lambdas * dt_s)
    drive_factor = 1j * gamma * np.expm1(lambdas * dt_s) / lambdas
    packet_z = np.zeros(len(magnet.ratios), dtype=complex)
    mean_z = np.zeros(len(pulse_b_t) + 1, dtype=complex)
    b_complex = pulse_b_t[:, 0] + 1j * pulse_b_t[:, 1]
    for step, drive in enumerate(b_complex):
        packet_z = step_factor * packet_z + drive_factor * drive
        mean_z[step + 1] = np.dot(magnet.ratios, packet_z)
    return mean_z


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
    simu.magnet.setHomogeneity(numPt=NUM_PT)
    simu.excField.setXYPulse(
        timeStep=simu.timeStep,
        timeLen=simu.timeLen,
        B1=b1_input,
        nu_rot=DELTA_NU_L,
    )
    pulse_b_t = simu.excField.B_vec.to_value(unit.T).copy()
    m_theo_complex = linear_cw_prediction(
        simu.magnet,
        pulse_b_t,
        simu.timeStep.to_value(unit.s),
    )
    simu.generateTrajectories(integrator="RK4")
    t = simu.getTimeStamp().to_value(unit.s)
    m_simu = simu.trjry[0, :, :2]
    m_theo = np.column_stack((m_theo_complex.real, m_theo_complex.imag))
    difference = m_simu - m_theo
    denominator = np.sum(m_theo**2)
    epsilon_mx = np.sqrt(np.sum(difference[:, 0] ** 2) / denominator)
    epsilon_my = np.sqrt(np.sum(difference[:, 1] ** 2) / denominator)
    epsilon_total = np.hypot(epsilon_mx, epsilon_my)
    return (
        float(epsilon_mx), float(epsilon_my), float(epsilon_total),
        t, m_simu, m_theo,
    )


if __name__ == "__main__":
    errors = [run_case(rate) for rate in RATES_HZ]
    for rate, result in zip(RATES_HZ, errors):
        epsilon_mx, epsilon_my, epsilon_total = result[:3]
        print(
            f"rate={rate:7.1f} Hz  "
            f"epsilon_Mx={epsilon_mx:.6e}  epsilon_My={epsilon_my:.6e}  "
            f"epsilon_total={epsilon_total:.6e}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for rate, result in zip(RATES_HZ, errors):
        if rate in (RATES_HZ[0], RATES_HZ[-1]):
            _, _, _, t, m_simu, m_theo = result
            suffix = f"{rate:g} Hz"
            axes[0].plot(t, m_simu[:, 0], label=suffix)
            axes[1].plot(t, m_simu[:, 1], label=suffix)
            if rate == RATES_HZ[-1]:
                axes[0].plot(t, m_theo[:, 0], "k--", label="theory")
                axes[1].plot(t, m_theo[:, 1], "k--", label="theory")
    axes[0].set(xlabel="Time (s)", ylabel="$M_x$", title="$M_x$ trajectory")
    axes[1].set(xlabel="Time (s)", ylabel="$M_y$", title="$M_y$ trajectory")
    for ax in axes[:2]:
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[2].loglog(RATES_HZ, [item[0] for item in errors], "o-", label=r"$\epsilon_{M_x}$")
    axes[2].loglog(RATES_HZ, [item[1] for item in errors], "s-", label=r"$\epsilon_{M_y}$")
    axes[2].loglog(RATES_HZ, [item[2] for item in errors], "^-", label="combined")
    axes[2].set(xlabel="Simulation rate (Hz)", ylabel="Normalized component L2 error",
                title="Rate convergence")
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend()
    fig.suptitle(
        f"CW calibration: FWHM=0.1 ppm, numPt={NUM_PT}; sampled-drive theory"
    )
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("calibration_convergence_cw.png"), dpi=180)
    plt.show()
