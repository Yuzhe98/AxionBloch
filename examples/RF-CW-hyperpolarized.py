# example script to simulate CW NMR experiment
import numpy as np
import time

# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from matplotlib import font_manager

from axionbloch.SimuTools import MagField, Simulation
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Magnet
from axionbloch.enphylope import PhysicalQuantity
from axionbloch.constants import gamma_p, mu_p
from axionbloch.utils import check


RCF_Freq_Hz = 1e6
signalFreqRot_Hz = 1
T1_s = 1.0

# short Tdelta, long T2
Tdelta_s = 1.0e-1
T2_s = 1.0e-1

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
duration = PhysicalQuantity(20, "s")
timeLen = int((simuRate * duration).to("").value)


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
    temp=PhysicalQuantity(300, "K"),  # room temperature
    pol=PhysicalQuantity(1e-2, ""),  # polarization
    verbose=False,
)

# set detection magnet
magnet = Magnet(
    name="detection magnet",
    B0=PhysicalQuantity(RCF_Freq_Hz - signalFreqRot_Hz, "Hz")
    / (sample.gamma / (2 * np.pi)),
    FWHM=PhysicalQuantity(1 / (np.pi * Tdelta_s) / RCF_Freq_Hz, ""),
    nFWHM=20.0,
)
# magnet.setHomogeneity(
#     numPt=int(
#         11
#         + duration.value_in("s")
#         * 2
#         * magnet.nFWHM
#         * magnet.FWHM_T
#         * sample.gamma.value_in("Hz/T")
#         * 1
#     ),
# )
magnet.setHomogeneity(
    numPt=500,
)
print(f"numPt for magnet homogeneity = {magnet.numPt}")


# set excitation field
excField = MagField(name="RF pulse")

simu = Simulation(
    name="simulation template",
    sample=sample,
    magnet=magnet,
    excField=excField,
    RCF_freq=PhysicalQuantity(RCF_Freq_Hz, "Hz"),
    rate=simuRate,  #
    duration=duration,
    verbose=False,
)

# set excitation pulse: 90 degree hard pulse
# t90_s = 10 * simu.timeStep_s
simu.excField.setXYPulse(
    timeStep_s=simu.timeStep_s,
    timeLen=simu.timeLen,
    # B1_T=1.0e-11,
    B1_T=0,
    nu_rot_Hz=signalFreqRot_Hz,
)

tic = time.perf_counter()
# simu.generateTrajectories(integrator="taylor")
simu.generateTrajectories(integrator="RK4")
toc = time.perf_counter()
print(f"GenerateTrajectory time consumption = {toc-tic:.6f} s")

simu.monitorTrajectories(verbose=True)
# simu.visualizeTrajectory3D(
#     verbose=False,
# )
save_data = True
if save_data:
    timeStamp_s = simu.getTimeStamp()
    check(simu.excField.B_vec.shape)
    check(simu.trjry.shape)
    np.savez(
        "C:\\Users\\zhenf\\D\\Yu0702\\CASPEr-Collaboration\\AxionBloch-paper/figures/RF_CW_hyperpolarized.npz",
        timeStamp_s=timeStamp_s,
        B_vec=simu.excField.B_vec,
        trjry=simu.trjry,
        T2_s=T2_s,
        Tdelta_s=Tdelta_s,
        T_1_s=T1_s,
        pol=sample.pol,
        init_M=simu.init_M.value_in(""),
    )
