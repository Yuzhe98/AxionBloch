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
from axionbloch.constants import gamma_Xe129, mu_Xe129, gamma_p, mu_p
from axionbloch.utils import check

# ──────────────────────────────────────────────────────────────────────────── #
# Physical parameters (all as Quantity)
# ──────────────────────────────────────────────────────────────────────────── #


T1 = 15 * unit.minute  # very long longitudinal relaxation (not the focus here)
T2 = 10.0 * unit.min  # transverse relaxation time
Tdelta = 10 * unit.ms  # dephasing time set by field inhomogeneity

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
    temp=163 * unit.K,
    verbose=False,
)
# CH3CH2OH
# sample = Sample(
#     name="Ethanol",
#     gamma=gamma_p,
#     massDensity=0.78945 * unit.g / unit.cm**3,
#     molarMass=46.069 * unit.g / unit.mol,
#     numOfSpinsPerMolecule=6 * unit.one,
#     T2=T2,
#     T1=T1,
#     vol=1 * unit.cm**3,
#     mu=mu_p,
#     temp=300 * unit.K,
#     verbose=False,
# )

B0 = 14.0 * unit.T  # magnet field magnitude

# Larmor frequency of Xe-129 at 14 T  (take |gamma| since gamma_Xe129 < 0)
nu_L = (np.abs(sample.gamma) / (2 * PI) * B0).to(unit.MHz)  # ≈ 165.8 MHz
check(nu_L)

signalFreqRot = 0.0 * unit.Hz  # signal offset in the rotating frame
RCF_freq = nu_L - signalFreqRot  # rotating-frame carrier frequency

# ──────────────────────────────────────────────────────────────────────────── #
# Pulse sequence definition (all as Quantity)
# ──────────────────────────────────────────────────────────────────────────── #
Npulses = 60

tip_angles = (18 / 123) * PI / 2 * np.ones(Npulses)

delays = np.arange(55, 55 * (Npulses + 1), 5) * unit.ms

phases = PI * np.ones(Npulses)


tip_angles = tip_angles[0:Npulses]
delays = delays[0:Npulses]
phases = phases[0:Npulses]

simuRate = 7 * unit.kHz
duration = np.sum(delays) + 10.0 * unit.ms  # ≈ 17 s; covers both echoes
check(duration.si)
# ──────────────────────────────────────────────────────────────────────────── #
# Magnet
# For Xe-129 (gamma < 0) the simulation requires B0 < 0 so that
# gamma * B0 > 0 and the Larmor frequency in the rotating frame is positive.
# Passing B0 directly as a Tesla Quantity avoids a T/rad unit artefact that
# arises when computing B0 = freq / (gamma / 2π).
# ──────────────────────────────────────────────────────────────────────────── #

FWHM = (1.0 / (np.pi * Tdelta) / nu_L).to(ppm)

magnet = Magnet(
    name="14 T detection magnet",
    B0=B0,
    FWHM=FWHM,
    nFWHM=10.0,
)
magnet.setHomogeneity(numPt=500, showPlot=False)
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

print(f"Larmor freq  = {nu_L:g}")
print(f"T2*          = {simu.T2star:g}")
print(f"Tdelta       = {simu.Tdelta:g}")
print(f"numSteps     = {simu.numSteps}")
print(f"timeStep     = {simu.timeStep:g}")

simu.excField.setNPulsesArbDelay(
    timeStep=simu.timeStep,
    timeLen=simu.timeLen,
    gamma=simu.sample.gamma,
    pulseDur=(5 * simu.timeStep),
    tip_angles=tip_angles,
    delays=delays,
    nu_rot=signalFreqRot,
    phases=phases,
    verbose=False,
)

simu.estimateRuntime( verbose=True)
# check(simu.excField.B_vec.shape)

# ──────────────────────────────────────────────────────────────────────────── #
# Run simulation
# ──────────────────────────────────────────────────────────────────────────── #

tic = time.perf_counter()
simu.generateTrajectories()
toc = time.perf_counter()
print(f"generateTrajectories time = {toc - tic:.3f} s")

simu.keepMeanStd()
simu.displayTrjry(verbose=True)

# ──────────────────────────────────────────────────────────────────────────── #
# Optionally save results
# ──────────────────────────────────────────────────────────────────────────── #

save_data = False
