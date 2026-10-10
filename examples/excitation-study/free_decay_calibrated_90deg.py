"""Free decay following an RK4-calibrated 90 degree hard pulse.

Run from the repository root with::

    python examples/excitation-study/free_decay_calibrated_90deg.py
"""

import matplotlib.pyplot as plt
import numpy as np

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.dependency import PI, unit
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation


RCF_FREQ = 1.0 * unit.MHz
T1 = 1.0e12 * unit.s
T2 = 1.0e12 * unit.s  # Keep relaxation negligible while checking the pulse.
TDELTA = 1.0e-1 * unit.s  # Narrow the field distribution to isolate the flip angle.
RATE = 1000.0 * unit.Hz
DURATION = 8.0 * unit.s
PULSE_STEPS = 10


sample = Sample(
    name="Ethanol (C2H5OH)",
    gamma=gamma_p,
    massDensity=0.78945 * unit.g / unit.cm**3,
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
    name="on-resonance detection magnet",
    B0=RCF_FREQ / (sample.gamma / (2 * PI)),
    FWHM=(1.0 / (np.pi * TDELTA * RCF_FREQ)) * unit.one,
    nFWHM=10.0,
)
magnet.setHomogeneity(numPt=301)

simu = Simulation(
    name="free decay after calibrated 90 degree pulse",
    sample=sample,
    magnet=magnet,
    excField=MagField(name="calibrated hard pulse"),
    RCF_freq=RCF_FREQ,
    rate=RATE,
    duration=DURATION,
    verbose=False,
)

simu.excField.setXYPulse_new(
    timeStep=simu.timeStep,
    timeLen=simu.timeLen,
    gamma=sample.gamma,
    pulseSteps=PULSE_STEPS,
    nu_rot=0.0 * unit.Hz,
    verbose=True,
)
simu.generateTrajectories(integrator="RK4")
simu.keepMeanStd()
mean_m = simu.M_mean
pulse_end = PULSE_STEPS
transverse = np.hypot(mean_m[pulse_end, 0], mean_m[pulse_end, 1])
tip_angle = np.degrees(np.arctan2(transverse, mean_m[pulse_end, 2]))
magnetization_norm = np.linalg.norm(mean_m[pulse_end])
print(f"Mean magnetization tip after pulse: {tip_angle:.12f} deg")
print(f"Mean magnetization magnitude after pulse: {magnetization_norm:.12f}")
simu.displayTrjries()
# t_s = np.arange(simu.trjry.shape[1]) * simu.timeStep.to_value(unit.s)
# m = simu.trjry.mean(axis=0)
# mxy = np.hypot(m[:, 0], m[:, 1])
# tip_angle = np.degrees(np.arctan2(mxy[PULSE_STEPS], m[PULSE_STEPS, 2]))
# print(f"Magnetization tip after {PULSE_STEPS} steps: {tip_angle:.12f} deg")

# fig, (ax_m, ax_rf) = plt.subplots(
#     2, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [2.2, 1]}
# )
# ax_m.plot(t_s, m[:, 0], label=r"$M_x$", lw=1.2)
# ax_m.plot(t_s, m[:, 1], label=r"$M_y$", lw=1.2)
# ax_m.plot(t_s, m[:, 2], label=r"$M_z$", lw=1.2)
# ax_m.plot(t_s, mxy, "--", label=r"$|M_{xy}|$", lw=1.2)
# ax_m.axvline(PULSE_STEPS * simu.timeStep.to_value(unit.s), color="k", ls=":", lw=1)
# ax_m.set(
#     xlabel="Time (s)",
#     ylabel="Normalized magnetization",
#     title=f"Free decay after calibrated pulse (tip = {tip_angle:.4f}°)",
# )
# ax_m.grid(alpha=0.3)
# ax_m.legend(ncol=4)

# rf = simu.excField.B_vec[0].to_value(unit.T)
# rf_t_s = np.arange(rf.shape[0]) * simu.timeStep.to_value(unit.s)
# zoom_end_s = min(0.05, t_s[-1])
# mask = rf_t_s <= zoom_end_s
# ax_rf.plot(rf_t_s[mask], rf[mask, 0], label=r"$B_x$", lw=1.2)
# ax_rf.plot(rf_t_s[mask], rf[mask, 1], label=r"$B_y$", lw=1.2)
# ax_rf.set(xlabel="Time (s)", ylabel="RF field (T)", title="Pulse detail")
# ax_rf.grid(alpha=0.3)
# ax_rf.legend()

# fig.tight_layout()
# plt.show()
