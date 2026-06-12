# Example script: spin-echo NMR simulation using a CPMG pulse train
import time

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.dependency import *
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation

RCF_Freq = 1 * unit.MHz
signalFreqRot = 1 * unit.Hz
T1 = 1e6 * unit.s

# short Tdelta
Tdelta = 1.0 * unit.s
T2 = 10.0 * unit.s

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

excField = MagField(name="CPMG pulse train")

simu = Simulation(
    name="Spin-echo CPMG simulation",
    sample=sample,
    magnet=magnet,
    excField=excField,
    rate=500 * unit.Hz,
    RCF_freq=RCF_Freq,
    duration=20 * unit.s,
    verbose=False,
)

simu.excField.setCPMGPulseTrain(
    timeStep=simu.timeStep,
    timeLen=simu.timeLen,
    gamma=simu.sample.gamma,
    t90=10 * simu.timeStep,
    tau=4 * Tdelta,
    numEcho=2,
    nu_rot=signalFreqRot,
    init_phase=0 * unit.rad,
    verbose=True,
)

tic = time.perf_counter()
simu.generateTrajectories(integrator="RK4")
toc = time.perf_counter()
print(f"{simu.generateTrajectories.__name__} time consumption = {toc - tic:.3g} s")

simu.keepMeanStd()
simu.displayTrjries(verbose=True)

save_data = False
