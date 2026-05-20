# Example: liquid Xe-129 at 14 T excited by N pulses with arbitrary delays.
#
# Pulse sequence (3 pulses, deliberately unequal interpulse delays):
#
#   90°(x)  ──── τ₁ ────  180°(y)  ──── τ₁ ────  [echo₁]  ──── τ₁+τ₂ ────  180°(y)  ──── τ₂ ────  [echo₂]
#
# with τ₁ = 2·Tdelta, τ₂ = 4·Tdelta (≠ τ₁, demonstrating "arbitrary" delays).
# delays[i] is the free-precession interval BEFORE pulse i.
import time

from axionbloch.dependency import *
from axionbloch.SimuTools import MagField, Simulation
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_Xe129, mu_Xe129
from axionbloch.utils import check

# ──────────────────────────────────────────────────────────────────────────── #
# Physical parameters (all as Quantity)
# ──────────────────────────────────────────────────────────────────────────── #

B0 = 14.0 * unit.T  # magnet field magnitude

# Larmor frequency of Xe-129 at 14 T  (take |gamma| since gamma_Xe129 < 0)
nuL = (np.abs(gamma_Xe129) / (2 * np.pi * unit.rad) * B0).to(
    unit.Hz, equivalencies=unit.dimensionless_angles()
)  # ≈ 165.7 MHz

signalFreqRot = 2.0 * unit.Hz  # signal offset in the rotating frame
RCF_freq = nuL - signalFreqRot  # rotating-frame carrier frequency

T1 = 15 * unit.minute  # very long longitudinal relaxation (not the focus here)
T2 = 10.0 * unit.minute  # transverse relaxation time
Tdelta = 1.0 * unit.ms  # dephasing time set by field inhomogeneity

# ──────────────────────────────────────────────────────────────────────────── #
# Pulse sequence definition (all as Quantity)
# ──────────────────────────────────────────────────────────────────────────── #
Npulses = 3

#   pulse index:  0        1        2
tip_angles = [np.pi / 2, np.pi, np.pi] * unit.rad

# delays[i] = free-precession time from end of pulse i-1 to start of pulse i
# delays = [0.0 * unit.s,  tau1,  tau1 + tau2]
# #            ↑              ↑          ↑
# #       before 90°     before 180°#1  before 180°#2 (after echo₁ + extra τ₂)
# # for i in range(Npulses):
# #     pass
delays = np.arange(5, 5 * (Npulses + 1), 5) * unit.ms

phases = [0.0 * unit.rad, (np.pi / 2) * unit.rad, (np.pi / 2) * unit.rad]
# 90°(x) → 180°(y) → 180°(y)  — CPMG-style phase cycle

simuRate = 10 * unit.kHz
duration = np.sum(delays) + 10.0 * unit.ms  # ≈ 17 s; covers both echoes
check(duration)
# ──────────────────────────────────────────────────────────────────────────── #
# Sample
# ──────────────────────────────────────────────────────────────────────────── #

sample = Sample(
    name="Liquid Xe-129",
    gamma=gamma_Xe129,  # negative for Xe-129
    massDensity=3.1 * unit.g / unit.cm**3,
    molarMass=131.29 * unit.g / unit.mol,
    numOfSpinsPerMolecule=1 * unit.one,
    T2=T2,
    T1=T1,
    vol=1 * unit.cm**3,
    mu=mu_Xe129,
    temp=300 * unit.K,
    verbose=False,
)

# ──────────────────────────────────────────────────────────────────────────── #
# Magnet
# For Xe-129 (gamma < 0) the simulation requires B0 < 0 so that
# gamma * B0 > 0 and the Larmor frequency in the rotating frame is positive.
# Passing B0 directly as a Tesla Quantity avoids a T/rad unit artefact that
# arises when computing B0 = freq / (gamma / 2π).
# ──────────────────────────────────────────────────────────────────────────── #

FWHM = (1.0 / (np.pi * Tdelta) / nuL).to(ppm)

magnet = Magnet(
    name="14 T detection magnet",
    B0=B0, 
    FWHM=FWHM,
    nFWHM=20.0,
)
magnet.setHomogeneity(numPt=500)
print(f"B0           = {magnet.B0:.4g}")
print(f"numPt        = {magnet.numPt}")

# ──────────────────────────────────────────────────────────────────────────── #
# Excitation field
# ──────────────────────────────────────────────────────────────────────────── #

excField = MagField(name="N-pulse arbitrary-delay sequence")

simu = Simulation(
    name="Xe-129 14T N-pulse simulation",
    sample=sample,
    magnet=magnet,
    excField=excField,
    RCF_freq=RCF_freq,
    rate=simuRate,
    duration=duration,
    verbose=False,
)

print(f"Larmor freq  = {nuL:g}")
print(f"T2*          = {simu.T2star_s:g}")
print(f"Tdelta       = {simu.Tdelta_s:g}")
print(f"numSteps     = {simu.numSteps}")
print(f"timeStep     = {simu.timeStep:g}")

simu.excField.setNPulsesArbDelay(
    timeStep=simu.timeStep,
    timeLen=simu.timeLen,
    gamma=simu.sample.gamma,
    t90=(10 * simu.timeStep),
    tip_angles=tip_angles,
    delays=[d for d in delays],
    nu_rot=signalFreqRot,
    phases=[p for p in phases],
    verbose=True,
)

check(simu.excField.B_vec.shape)

# ──────────────────────────────────────────────────────────────────────────── #
# Run simulation
# ──────────────────────────────────────────────────────────────────────────── #

tic = time.perf_counter()
simu.generateTrajectories(integrator="RK4")
toc = time.perf_counter()
print(f"generateTrajectories time = {toc - tic:.3f} s")

simu.keepMeanStd()
simu.displayTrjries(verbose=True)

# ──────────────────────────────────────────────────────────────────────────── #
# Optionally save results
# ──────────────────────────────────────────────────────────────────────────── #

save_data = False
