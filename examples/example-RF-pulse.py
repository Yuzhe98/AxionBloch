# example script to simulate RF-pulse NMR experiment
import time

from axionbloch.dependency import *
from axionbloch.SimuTools import MagField, Simulation
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.utils import check

RCF_Freq = 1 * unit.MHz
signalFreqRot = 2 * unit.Hz
T1 = 1e6 * unit.s

# short Tdelta, long T2
Tdelta = 1.0 * unit.s
T2 = 10.0 * unit.s

# # short T2, long Tdelta
# Tdelta_s = 10.0
# T2 = 1.0

# # short Tdelta and T2
# Tdelta_s = 1.0
# T2 = 1.0

# # long Tdelta and T2
# Tdelta_s = 10.0
# T2 = 10.0

simuRate = 1000 * unit.Hz
duration = 30.1 * unit.s

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

# set detection magnet
magnet = Magnet(
    name="detection magnet",
    B0=(RCF_Freq - signalFreqRot) / (sample.gamma / (2 * PI)),
    FWHM=(1 / (np.pi * Tdelta) / RCF_Freq) * unit.one,
    nFWHM=20.0,
)
magnet.setHomogeneity(numPt=500)
print(f"numPt for magnet homogeneity = {magnet.numPt}")

# set excitation field
excField = MagField(name="RF pulse")

simu = Simulation(
    name="simulation template",
    sample=sample,
    magnet=magnet,
    excField=excField,
    RCF_freq=RCF_Freq,
    rate=simuRate,
    duration=duration,
    verbose=False,
)

# set excitation pulse: 90 degree hard pulse
simu.excField.set90DegPulse(
    timeStep=simu.timeStep,
    timeLen=simu.timeLen,
    gamma=simu.sample.gamma,
    t90=10 * simu.timeStep,
    nu_rot=signalFreqRot,
)

tic = time.perf_counter()
simu.generateTrajectories(integrator="RK4")
toc = time.perf_counter()
print(f"GenerateTrajectory time consumption = {toc-tic:.6f} s")

simu.monitorTrajectories(verbose=True)

# save_data = False
# if save_data:
#     timeStamp_s = simu.getTimeStamp()
#     check(simu.excField.B_vec.shape)
#     check(simu.trjry.shape)
#     np.savez(
#         "RF_pulse_simu.npz",
#         timeStamp_s=timeStamp_s,
#         B_vec=simu.excField.B_vec,
#         trjry=simu.trjry,
#         T2=T2,
#         Tdelta_s=Tdelta,
#     )
