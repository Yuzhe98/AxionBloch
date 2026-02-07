# example script to simulate RF-pulse NMR experiment
import numpy as np
from functools import partial
import time
from tqdm import tqdm

from axionbloch.SimuTools import MagField, Simulation, gate
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Magnet
from axionbloch.utils import giveDateAndTime
from axionbloch.Envelope import PhysicalQuantity, gamma_p, mu_p


RCF_Freq_Hz = 1e6
T1_s = 100.0

# short Tdelta, long T2
Tdelta_s = 1.0
T2_s = 10.0

# # short T2, long Tdelta
# Tdelta_s = 10.0
# T2_s = 1.0

# # short Tdelta and T2
# Tdelta_s = 1.0
# T2_s = 1.0

# # long Tdelta and T2
# Tdelta_s = 10.0
# T2_s = 10.0

simuRate = PhysicalQuantity(500, "Hz")  #
duration = PhysicalQuantity(10, "s")
timeLen = int((simuRate * duration).convert_to("").value)


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

# set detection magnet
magnet_det = Magnet(
    name="detection magnet",
    B0=PhysicalQuantity(RCF_Freq_Hz - 0, "Hz") / (sample.gamma / (2 * np.pi)),
    FWHM=PhysicalQuantity(1 / (np.pi * Tdelta_s) / RCF_Freq_Hz, ""),
    nFWHM=20.0,
)
magnet_det.setHomogeneity(
    # numPt=1,
    # numPt=400,
    numPt=int(
        11
        + duration.value_in("s")
        * 2
        * magnet_det.nFWHM
        * magnet_det.FWHM_T
        * sample.gamma.value_in("Hz/T")
        * 1
    ),
)

# set excitation field
excField = MagField(name="RF pulse")

simu = Simulation(
    name="simulation template",
    sample=sample, 
    magnet=magnet_det,
    excField=excField,
    RCF_freq=PhysicalQuantity(RCF_Freq_Hz, "Hz"),
    rate=simuRate,  #
    duration=duration,
    verbose=False,
)

# set excitation pulse: 90 degree hard pulse
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

simu.excType = "pulse NMR"

tic = time.perf_counter()
simu.generateTrajectory_1LoopByNb(verbose=False)
toc = time.perf_counter()
print(f"GenerateTrajectory time consumption = {toc-tic:.6f} s")

simu.monitorTrajectory(verbose=True)
simu.visualizeTrajectory3D(
    verbose=False,
)
