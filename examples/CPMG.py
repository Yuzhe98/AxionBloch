# $env:PYTHONPATH = "your:\path\here;$env:PYTHONPATH”

import numpy as np
from functools import partial
import time
from tqdm import tqdm

from axionbloch.SimuTools import MagField, Simulation, gate
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Pickup, Magnet
from axionbloch.utils import check, giveDateAndTime
from axionbloch.Envelope import PhysicalQuantity, gamma_p, mu_p


# excitation_type = "CW"
# excitation_type = "1Pulse"
excitation_type = "CPMG"

RCF_Freq_Hz = 1e6

T1_s = 1e6

# # short Tdelta
# Tdelta_s = 1.0
# T2_s = 1e2

# # short T2
# Tdelta_s = 1e2
# T2_s = 1.0

# # short Tdelta and T2
# Tdelta_s = 1.0
# T2_s = 1.0
# T1_s = 100.0

# # long Tdelta and T2
# Tdelta_s = 1.0e1
# T2_s = 1.0e1

# short Tdelta
Tdelta_s = 0.04
T2_s = 3

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
    numPt=20000,
)

timestr = giveDateAndTime()

excField = MagField(name="RF pulse")


rand_seed = 0

if excitation_type == "CW":
    # CW excitation
    simu = Simulation(
        name="NMR simulation",
        sample=sample,
        magnet=magnet,
        excField=excField,
        rate=None,  #
        duration=None,
        verbose=True,
    )
    simu.excField.setXYPulse(
        timeStamp=simu.timeStamp_s,
        B1=1.0e-6
        * 2
        * np.pi
        / 2.0
        / simu.gamma_HzToT
        / (5 * 1e-3),  # amplitude of the excitation pulse in [T]
        nu_rot=0,
        init_phase=0,
        # direction: np.ndarray,  #  not needed now
        duty_func=partial(gate, start=0, stop=simu.duration_s),
    )
elif excitation_type == "1Pulse":
    # hard pulse excitation

    simu = Simulation(
        name="NMR simulation",
        sample=sample,
        magnet=magnet,
        excField=excField,
        rate=None,  #
        duration=None,
        verbose=True,
    )
    simu.excField.setXYPulse(
        timeStamp=simu.timeStamp_s,
        B1=2
        * np.pi
        / 2.0
        / simu.gamma_HzToT
        / (5 * simu.timeStep_s),  # amplitude of the excitation pulse in [T]
        nu_rot=0,
        init_phase=0,
        # direction: np.ndarray,  #  not needed now
        duty_func=partial(gate, start=0, stop=5 * simu.timeStep_s),
    )
elif excitation_type == "CPMG":

    simu = Simulation(
        name="NMR simulation",
        sample=sample,
        magnet=magnet,
        excField=excField,
        rate=PhysicalQuantity(10000, "Hz"),  #
        duration=PhysicalQuantity(5, "s"),
        verbose=True,
    )
    check(magnet.numPt)
    simu.excField.setCPMGPulseTrain(
        timeStep_s=simu.timeStep_s,
        timeLen=simu.timeLen,
        gamma_HzToT=simu.gamma_HzToT,
        t90_s=5e-3,
        tau_s=10 * Tdelta_s,
        numEcho=10,
        nu_rot_Hz=0,
        init_phase=0,
    )
else:
    raise ValueError("excitation_type not found. ")

simu.excType = excitation_type

tic = time.perf_counter()
simu.generateTrajectories(cleanup=False, verbose=False)
toc = time.perf_counter()
print(f"{simu.generateTrajectories.__name__} time consumption = {toc-tic:.3g} s")

simu.monitorTrajectories(verbose=True)
