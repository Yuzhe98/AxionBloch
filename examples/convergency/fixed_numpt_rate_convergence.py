"""Free-decay time-step convergence at a fixed inhomogeneity packet count.

Run from the repository root with::

    python examples/convergency/fixed_numpt_rate_convergence.py

Unlike the automatic-rate convergence example, this fixes ``numPt`` at 400
for every simulation. The expected post-pulse free decay is evaluated from
that same 400-packet distribution, isolating time integration error from
packet-grid quadrature error. The pulse endpoint is used as the initial
condition, so this measures free decay after the pulse rather than its tip.
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
T90_STEPS = 5
RATES_HZ = [250.0, 500.0, 1000.0, 2000.0, 4000.0]


def run_case(rate_hz: float) -> tuple[float, float, float]:
    sample = Sample(
        name="fixed_numPt_free_decay",
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
    magnet = Magnet(
        name=f"fixed_numPt_{NUM_PT}",
        B0=(RCF_FREQ - DELTA_NU_L) / (sample.gamma / (2 * PI)),
        FWHM=FWHM,
        numPt=NUM_PT,
        nFWHM=10.0,
    )
    simu = Simulation(
        name=f"free_decay_{rate_hz:g}Hz_numPt_{NUM_PT}",
        sample=sample,
        magnet=magnet,
        excField=MagField(name="90deg_pulse"),
        RCF_freq=RCF_FREQ,
        rate=rate_hz * unit.Hz,
        duration=DURATION,
        verbose=False,
    )
    # Simulation initialization estimates numPt from duration and linewidth;
    # restore the selected value after initialization for a controlled sweep.
    simu.magnet.setHomogeneity(numPt=NUM_PT)
    simu.excField.set90DegPulse(
        timeStep=simu.timeStep,
        timeLen=simu.timeLen,
        gamma=sample.gamma,
        t90=T90_STEPS * simu.timeStep,
        nu_rot=DELTA_NU_L,
    )
    simu.generateTrajectories(integrator="RK4")

    t_s = simu.getTimeStamp().to_value(unit.s)
    t_free_s = t_s[T90_STEPS:] - t_s[T90_STEPS]
    simulated = simu.trjry[0, T90_STEPS:, :2]
    z0 = complex(simulated[0, 0], simulated[0, 1])
    gamma_over_2pi = gamma_p.to_value(unit.rad * unit.Hz / unit.T) / (2 * np.pi)
    detuning_hz = (
        gamma_over_2pi * simu.magnet.B_spread.to_value(unit.T)
        - simu.RCF_freq.to_value(unit.Hz)
    )
    z_expected = np.zeros_like(t_free_s, dtype=complex)
    for detuning, weight in zip(detuning_hz, simu.magnet.ratios):
        z_expected += weight * np.exp(-2j * np.pi * detuning * t_free_s)
    z_expected *= z0 * np.exp(-t_free_s / T2.to_value(unit.s))
    expected = np.column_stack((z_expected.real, z_expected.imag))

    difference = simulated - expected
    denominator = np.sum(expected**2)
    epsilon_mx = np.sqrt(np.sum(difference[:, 0] ** 2) / denominator)
    epsilon_my = np.sqrt(np.sum(difference[:, 1] ** 2) / denominator)
    epsilon_total = np.hypot(epsilon_mx, epsilon_my)
    return float(epsilon_mx), float(epsilon_my), float(epsilon_total)


if __name__ == "__main__":
    errors = [run_case(rate) for rate in RATES_HZ]
    print(f"Fixed inhomogeneity packet count: numPt={NUM_PT}; FWHM={FWHM}")
    print("rate (Hz) | epsilon_Mx | epsilon_My | epsilon_total")
    for rate, (epsilon_mx, epsilon_my, epsilon_total) in zip(RATES_HZ, errors):
        print(
            f"{rate:9.1f} | {epsilon_mx:10.4e} | {epsilon_my:10.4e} "
            f"| {epsilon_total:12.4e}"
        )

    plt.loglog(RATES_HZ, [item[0] for item in errors], "o-", label=r"$M_x$")
    plt.loglog(RATES_HZ, [item[1] for item in errors], "s-", label=r"$M_y$")
    plt.loglog(RATES_HZ, [item[2] for item in errors], "^-", label="combined")
    plt.xlabel("Simulation rate (Hz)")
    plt.ylabel("Component error (normalized L2)")
    plt.title(f"Free-decay rate convergence at numPt={NUM_PT}, FWHM=0.1 ppm")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(__file__).with_name("fixed_numpt_rate_convergence.png"), dpi=180)
    plt.show()
