"""Rate-convergence demonstration for the full transverse-vector calibration.

Run from the repository root with::

    python examples/convergency/calibration_convergence_free_decay.py

The pulse duration is held fixed across rates. The reference propagates each
inhomogeneity packet through the sampled pulse, including detuning and
relaxation, and then evaluates its free decay analytically. This avoids both
rate-dependent pulse bandwidth and the assumption that every packet has the
same transverse state after the pulse.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.dependency import PI, ppm, unit
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation
from examples.convergency.inhomogeneity_numpt_convergence import pulse_aware_prediction


RCF_FREQ = 1.0 * unit.MHz
DELTA_NU_L = 1.0 * unit.Hz
FWHM = 0.1 * ppm
NUM_PT = 400
T1 = 1.0e6 * unit.s
T2 = 10.0 * unit.s
DURATION = 3.0 * unit.s
T90_DURATION = 20.0 * unit.ms  # 5 steps at the lowest rate in the sweep.
RATES_HZ = [250.0, 500.0, 1000.0, 2000.0, 4000.0]


def run_case(rate_hz: float):
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
    simu.magnet.setHomogeneity(numPt=NUM_PT)
    pulse_steps = int(np.round((T90_DURATION / simu.timeStep).to_value(unit.one)))
    simu.excField.set90DegPulse(
        timeStep=simu.timeStep,
        timeLen=simu.timeLen,
        gamma=simu.sample.gamma,
        t90=T90_DURATION,
        nu_rot=DELTA_NU_L,
    )
    pulse_b_t = simu.excField.B_vec.to_value(unit.T)[:pulse_steps].copy()
    t = simu.getTimeStamp().to_value(unit.s)
    t_sig = t[pulse_steps:] - t[pulse_steps]
    z_theo = pulse_aware_prediction(
        simu.magnet,
        pulse_b_t,
        t_sig,
        simu.timeStep.to_value(unit.s),
    )
    simu.generateTrajectories(integrator="RK4")
    m_simu = simu.trjry[0, pulse_steps:, :2]
    m_theo = np.column_stack((z_theo.real, z_theo.imag))
    difference = m_simu - m_theo
    denominator = np.sum(m_theo**2)
    epsilon_mx = np.sqrt(np.sum(difference[:, 0] ** 2) / denominator)
    epsilon_my = np.sqrt(np.sum(difference[:, 1] ** 2) / denominator)
    epsilon_total = np.hypot(epsilon_mx, epsilon_my)
    return (
        float(epsilon_mx), float(epsilon_my), float(epsilon_total),
        t_sig, m_simu, m_theo,
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
            _, _, _, t_sig, m_simu, m_theo = result
            suffix = f"{rate:g} Hz"
            axes[0].plot(t_sig, m_simu[:, 0], label=suffix)
            axes[1].plot(t_sig, m_simu[:, 1], label=suffix)
            if rate == RATES_HZ[-1]:
                axes[0].plot(t_sig, m_theo[:, 0], "k--", label="theory")
                axes[1].plot(t_sig, m_theo[:, 1], "k--", label="theory")
    axes[0].set(xlabel="Time after pulse (s)", ylabel="$M_x$", title="$M_x$ trajectory")
    axes[1].set(xlabel="Time after pulse (s)", ylabel="$M_y$", title="$M_y$ trajectory")
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
        f"Free-decay calibration: FWHM=0.1 ppm, numPt={NUM_PT}, "
        f"T90={T90_DURATION.to_value(unit.ms):g} ms"
    )
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("calibration_convergence_free_decay.png"), dpi=180)
    plt.show()
