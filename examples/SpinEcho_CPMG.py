# $env:PYTHONPATH = "your:\path\here;$env:PYTHONPATH”
# $env:PYTHONPATH = "C:\Users\zhenf\D\Yu0702\axionbloch;$env:PYTHONPATH”

import numpy as np
import time

from axionbloch.SimuTools import MagField, Simulation
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Magnet
from axionbloch.utils import check
from axionbloch.enphylope import PhysicalQuantity as PQ
from axionbloch.constants import gamma_p, mu_p


RCF_Freq_Hz = 1e6
signalFreqRot_Hz = 1
T1_s = 1e6

# short Tdelta
Tdelta_s = 1.0
T2_s = 10.0

# CH3CH2OH
sample = Sample(
    name="Ethanol",  # name of the sample
    gamma=gamma_p,  # [Hz/T]. Remember input it with 2 * np.pi
    massDensity=PQ(0.78945, "g / cm**3 "),
    molarMass=PQ(46.069, "g / mol"),  # molar mass
    numOfSpinsPerMolecule=PQ(6, ""),  # number of spins per molecule
    T2=PQ(T2_s, "s"),  #
    T1=PQ(T1_s, "s"),  #
    vol=PQ(1, "cm**3"),
    mu=mu_p,  # magnetic dipole moment
    temp=PQ(300, "K"),
    verbose=False,
)

# set detection magnet
magnet = Magnet(
    name="detection magnet",
    B0=PQ(RCF_Freq_Hz - signalFreqRot_Hz, "Hz")
    / (sample.gamma / (2 * np.pi)),
    FWHM=PQ(1 / (np.pi * Tdelta_s) / RCF_Freq_Hz, ""),
    nFWHM=20.0,
)


excField = MagField(name="RF pulse")

rand_seed = 0

simu = Simulation(
    name="NMR simulation",
    sample=sample,
    magnet=magnet,
    excField=excField,
    rate=PQ(500, "Hz"),  #
    RCF_freq=PQ(RCF_Freq_Hz, "Hz"),
    duration=PQ(20, "s"),
    verbose=True,
)


# magnet.setHomogeneity(
#     numPt=int(
#         11
#         + simu.duration.value_in("s")
#         * 2
#         * magnet.nFWHM
#         * magnet.FWHM_T
#         * sample.gamma.value_in("Hz/T")
#         * 1
#     ),
# )
# check(magnet.numPt)
magnet.setHomogeneity(
    numPt=500,
)

simu.excField.setCPMGPulseTrain(
    timeStep_s=simu.timeStep,
    timeLen=simu.timeLen,
    gamma_HzToT=simu.gamma_HzToT,
    t90_s=10 * simu.timeStep,
    tau_s=4 * Tdelta_s,
    numEcho=2,
    nu_rot_Hz=signalFreqRot_Hz,
    init_phase=0,
    verbose=True,
)

tic = time.perf_counter()
# simu.generateTrajectories(integrator="taylor")
simu.generateTrajectories(integrator="RK4")
toc = time.perf_counter()
print(f"{simu.generateTrajectories.__name__} time consumption = {toc-tic:.3g} s")

# magnet.numPt : ? unknown
# 3.18 s for no optimization RK4
# 3.09 s after simplifying dt/2 and dt/6 calculations
# 2.79 s after combining gamma and B field into Omega

# taylor 1.23 s (unstable result) default magnet.numPt : int(8011)
# 0.551 s for RK4 with magnet.numPt = 1000 (stable result)
# 0.284 s for RK4 with magnet.numPt = 500 (stable result)
# RK4 with magnet.numPt = 100 gives unstable result

simu.monitorTrajectories(verbose=True)

save_data = True
if save_data:
    timeStamp_s = simu.getTimeStamp()
    check(simu.excField.B_vec.shape)
    check(simu.trjry.shape)
    np.savez(
        "SpinEcho_CPMG_simu.npz",
        timeStamp_s=timeStamp_s,
        B_vec=simu.excField.B_vec,
        trjry=simu.trjry,
        T2_s=T2_s,
        Tdelta_s=Tdelta_s,
    )
