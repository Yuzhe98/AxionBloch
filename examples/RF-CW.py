# Example script: CW (continuous-wave) NMR free-induction-decay simulation
#
# A CW excitation field drives the spin ensemble continuously at signalFreqRot
# in the rotating frame.  The simulation records the magnetization trajectory
# over `duration` seconds.
import time

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.dependency import *
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation

RCF_Freq = 1 * unit.MHz
signalFreqRot = 0 * unit.Hz
T1 = 5 * unit.s

# short Tdelta, long T2
Tdelta = 10 * unit.ms
T2 = 1.0 * unit.s

# # short T2, long Tdelta
# Tdelta = 10.0 * unit.s
# T2 = 1.0 * unit.s

# # short Tdelta and T2
# Tdelta = 1.0 * unit.s
# T2 = 1.0 * unit.s

# # long Tdelta and T2
# Tdelta = 10.0 * unit.s
# T2 = 10.0 * unit.s

simuRate = 2000 * unit.Hz
duration = 0.1 * unit.s

# CH3CH2OH sample
sample = Sample(
    name="Ethanol",
    gamma=gamma_p,
    massDensity=0.78945 * unit.g / unit.cm**3,
    molarMass=46.069 * unit.g / unit.mol,
    numOfSpinsPerMolecule=6 * unit.one,
    T2=T2,
    T1=T1,
    vol=1 * unit.cm**3,
    mu=mu_p,
    temp=300 * unit.K,
    verbose=False,
)

FWHM = (1 / (np.pi * Tdelta) / RCF_Freq) * unit.one

# Detection (bias) magnet — B0 tuned so that the Larmor frequency equals
# the carrier plus signalFreqRot
magnet = Magnet(
    name="detection magnet",
    B0=(RCF_Freq - signalFreqRot) / (sample.gamma / (2 * PI)),
    FWHM=FWHM,
    nFWHM=10.0,
)
magnet.setHomogeneity(numPt=500, showPlot=True, verbose=True)
print(f"numPt for magnet homogeneity = {magnet.numPt}")

# Excitation field (CW)
excField = MagField(name="CW excitation")

simu = Simulation(
    name="CW NMR simulation",
    sample=sample,
    magnet=magnet,
    excField=excField,
    RCF_freq=RCF_Freq,
    rate=simuRate,
    duration=duration,
    verbose=False,
)

# CW drive: constant-envelope XY pulse at the signal frequency
simu.excField.setXYPulse(
    timeStep=simu.timeStep,
    timeLen=simu.timeLen,
    B1=0.005
    * unit.Hz
    / (
        sample.gamma / (2 * PI)
    ),  # B1 field amplitude in Tesla, converted from Rabi frequency in Hz
    nu_rot=signalFreqRot,
)

tic = time.perf_counter()
simu.generateTrajectories(integrator="RK4")
toc = time.perf_counter()
print(f"generateTrajectories time consumption = {toc - tic:.6f} s")

simu.keepMeanStd()
simu.displayTrjries(verbose=True)
# simu.monitorTrajectories(verbose=True)
save_data = False
