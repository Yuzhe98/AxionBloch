# $env:PYTHONPATH = "your:\path\here;$env:PYTHONPATH”

import numpy as np
import time

from astropy import units as unit

from axionbloch.SimuTools import MagField, Simulation
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Magnet
from axionbloch.utils import check
from axionbloch.constants import gamma_p, mu_p


RCF_Freq_Hz = 1e6

T1_s = 1e8

# short Tdelta
Tdelta_s = 0.1
T2_s = 5.0

# CH3CH2OH
sample = Sample(
    name="Ethanol",  # name of the sample
    gamma=gamma_p,  # [Hz/T]. Remember input it with 2 * np.pi
    massDensity=0.78945 * unit.g / unit.cm**3,
    molarMass=46.069 * unit.g / unit.mol,
    numOfSpinsPerMolecule=6 * unit.one,
    T2=T2_s * unit.s,
    T1=T1_s * unit.s,
    vol=1 * unit.cm**3,
    mu=mu_p,  # magnetic dipole moment
    verbose=False,
)

magnet = Magnet(
    name="detection magnet",
    B0=RCF_Freq_Hz * unit.Hz / (sample.gamma / (2 * np.pi)),
    FWHM=(1 / (np.pi * Tdelta_s) / RCF_Freq_Hz) * unit.one,
    nFWHM=10.0,
)
magnet.setHomogeneity(
    numPt=1000,
)
check(magnet.numPt)


excField = MagField(name="RF pulse")

rand_seed = 0

simu = Simulation(
    name="NMR simulation",
    sample=sample,
    magnet=magnet,
    excField=excField,
    rate=1000 * unit.Hz,
    duration=20 * unit.s,
    verbose=True,
)
simu.excField.setCPMGPulseTrain(
    timeStep_s=simu.timeStep.to_value(unit.s),
    timeLen=simu.timeLen,
    gamma_HzToT=simu.gamma_HzToT,
    t90_s=3 * simu.timeStep.to_value(unit.s),
    tau_s=10 * Tdelta_s,
    numEcho=10,
    nu_rot_Hz=0,
    init_phase=0,
    verbose=True,
)

tic = time.perf_counter()
# simu.generateTrajectories(integrator="taylor")  # taylor is bad for long-term stability
simu.generateTrajectories(integrator="RK4")  # RK4 is better
# You can try both and compare the results.
toc = time.perf_counter()
print(f"{simu.generateTrajectories.__name__} time consumption = {toc-tic:.3g} s")

simu.monitorTrajectories(verbose=True)
