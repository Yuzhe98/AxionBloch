"""Compare nominal and RK4-calibrated hard 90 degree pulses.

Run from the repository root with::

    python examples/excitation-study/hard_pulse_discretization.py

This isolates the pulse by using an on-resonance, long-lived sample. The
reported flip angle is measured from the simulated magnetization immediately
after the pulse, before free decay.
"""

import numpy as np

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.dependency import PI, unit
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation


RATE_HZ = 1000.0
PULSE_STEPS = [1, 2, 3, 5, 10, 20]
NOMINAL_DURATION_EXTRA_STEPS = 0.4
RCF_FREQ = 1.0 * unit.MHz


def run_case(n_steps: int, calibrated: bool) -> float:
    sample = Sample(
        name="pulse_discretization",
        gamma=gamma_p,
        massDensity=0.789 * unit.g / unit.cm**3,
        molarMass=46.069 * unit.g / unit.mol,
        numOfSpinsPerMolecule=6 * unit.one,
        T1=1e12 * unit.s,
        T2=1e12 * unit.s,
        vol=1.0 * unit.cm**3,
        mu=mu_p,
        temp=300 * unit.K,
        verbose=False,
    )
    magnet = Magnet(
        name="on_resonance",
        B0=RCF_FREQ / (sample.gamma / (2 * PI)),
        FWHM=1e-12 * unit.one,
        nFWHM=1.0,
    )
    simu = Simulation(
        name=f"{'calibrated' if calibrated else 'nominal'}_{n_steps}",
        sample=sample,
        magnet=magnet,
        excField=MagField(name="hard pulse"),
        RCF_freq=RCF_FREQ,
        rate=RATE_HZ * unit.Hz,
        duration=0.1 * unit.s,
        verbose=False,
    )
    # Deliberately request a duration between grid points. The legacy pulse
    # rounds its sample count but keeps this original duration for B1.
    pulse_duration = (n_steps + NOMINAL_DURATION_EXTRA_STEPS) * simu.timeStep
    if calibrated:
        simu.excField.setXYPulse_new(
            timeStep=simu.timeStep,
            timeLen=simu.timeLen,
            gamma=sample.gamma,
            pulseSteps=n_steps,
            nu_rot=0.0 * unit.Hz,
            verbose=True,
        )
    else:
        simu.excField.set90DegPulse(
            timeStep=simu.timeStep,
            timeLen=simu.timeLen,
            gamma=sample.gamma,
            t90=pulse_duration,
            nu_rot=0.0 * unit.Hz,
        )
    simu.generateTrajectories(integrator="RK4")
    m = simu.trjry[0, n_steps]
    angle = np.degrees(np.arctan2(np.hypot(m[0], m[1]), m[2]))
    return float(angle)


if __name__ == "__main__":
    print("steps | nominal flip (deg) | calibrated flip (deg) | error (deg)")
    for steps in PULSE_STEPS:
        nominal = run_case(steps, calibrated=False)
        try:
            calibrated = run_case(steps, calibrated=True)
            calibrated_text = f"{calibrated:21.12f} | {calibrated - 90:+.3e}"
        except ValueError as exc:
            calibrated_text = f"unavailable ({exc})"
        print(f"{steps:5d} | {nominal:18.9f} | {calibrated_text}")
