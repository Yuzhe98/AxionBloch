# $env:PYTHONPATH = "your:\path\here;$env:PYTHONPATH”

import numpy as np
import time

from axionbloch.SimuTools import MagField, Simulation
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Magnet
from axionbloch.utils import check, giveDateAndTime
from axionbloch.enphylope import PhysicalQuantity
from axionbloch.constants import gamma_p, mu_p


RCF_Freq_Hz = 1e6

T1_s = 1e8

# short Tdelta
Tdelta_s = 0.08
T2_s = 5

# CH3CH2OH
sample = Sample(
    name="Ethanol",  # name of the sample
    gamma=gamma_p,  # [Hz/T]. Remember input it with 2 * np.pi
    massDensity=PhysicalQuantity(0.78945, "g / cm**3 "),
    molarMass=PhysicalQuantity(46.069, "g / mol"),  # molar mass
    numOfSpinsPerMolecule=PhysicalQuantity(6, ""),  # number of spins per molecule
    T2=PhysicalQuantity(T2_s, "s"),  #
    T1=PhysicalQuantity(T1_s, "s"),  #
    vol=PhysicalQuantity(1, "cm**3"),
    mu=mu_p,  # magnetic dipole moment
    verbose=False,
)

magnet = Magnet(
    name="detection magnet",
    B0=PhysicalQuantity(RCF_Freq_Hz, "Hz") / (sample.gamma / (2 * np.pi)),
    FWHM=PhysicalQuantity(1 / (np.pi * Tdelta_s) / RCF_Freq_Hz, ""),
    nFWHM=10.0,
    numPt=10000,
)
magnet.setHomogeneity(
    numPt=10000,
)
check(magnet.numPt)


excField = MagField(name="RF pulse")

rand_seed = 0

simu = Simulation(
    name="NMR simulation",
    sample=sample,
    magnet=magnet,
    excField=excField,
    rate=PhysicalQuantity(10000, "Hz"),  #
    duration=PhysicalQuantity(20, "s"),
    verbose=True,
)
simu.excField.setCPMGPulseTrain(
    timeStep_s=simu.timeStep_s,
    timeLen=simu.timeLen,
    gamma_HzToT=simu.gamma_HzToT,
    t90_s=5 * simu.timeStep_s,
    tau_s=10 * Tdelta_s,
    numEcho=10,
    nu_rot_Hz=0,
    init_phase=0,
    verbose=True,
)

tic = time.perf_counter()
simu.generateTrajectories(cleanup=False, verbose=False)
# generateTrajectories time consumption = 15 s with bs ver 0.1.0
toc = time.perf_counter()
print(f"{simu.generateTrajectories.__name__} time consumption = {toc-tic:.3g} s")

simu.monitorTrajectories(verbose=True)
