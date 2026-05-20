# Example script: CW NMR simulation with a hyperpolarized sample
import time

from axionbloch.dependency import *
from axionbloch.SimuTools import MagField, Simulation
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.utils import check


RCF_Freq = 1 * unit.MHz
signalFreqRot = 1 * unit.Hz
T1 = 1.0 * unit.s

# short Tdelta, long T2
Tdelta = 1.0e-1 * unit.s
T2 = 1.0e-1 * unit.s

# # short T2, long Tdelta
# Tdelta = 10.0 * unit.s
# T2 = 1.0 * unit.s

# # short Tdelta and T2
# Tdelta = 1.0 * unit.s
# T2 = 1.0 * unit.s

# # long Tdelta and T2
# Tdelta = 10.0 * unit.s
# T2 = 10.0 * unit.s

simuRate = 500 * unit.Hz
duration = 20 * unit.s

# CH3CH2OH
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
    pol=1e-2 * unit.one,
    verbose=False,
)

FWHM = (1 / (np.pi * Tdelta) / RCF_Freq) * unit.one

# set detection magnet
magnet = Magnet(
    name="detection magnet",
    B0=(RCF_Freq - signalFreqRot) / (sample.gamma / (2 * PI)),
    FWHM=FWHM,
    nFWHM=20.0,
)
magnet.setHomogeneity(numPt=500)
print(f"numPt for magnet homogeneity = {magnet.numPt}")

# set excitation field
excField = MagField(name="CW excitation")

simu = Simulation(
    name="CW NMR hyperpolarized simulation",
    sample=sample,
    magnet=magnet,
    excField=excField,
    RCF_freq=RCF_Freq,
    rate=simuRate,
    duration=duration,
    verbose=False,
)

# CW drive with zero amplitude — pure free-induction decay
simu.excField.setXYPulse(
    timeStep=simu.timeStep,
    timeLen=simu.timeLen,
    B1=0 * unit.T,
    nu_rot=signalFreqRot,
)

tic = time.perf_counter()
simu.generateTrajectories(integrator="RK4")
toc = time.perf_counter()
print(f"GenerateTrajectory time consumption = {toc - tic:.6f} s")

simu.keepMeanStd()
simu.displayTrjries(verbose=True)

save_data = False
