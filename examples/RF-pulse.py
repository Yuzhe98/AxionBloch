# example script to simulate RF-pulse NMR experiment
import time

from axionbloch.dependency import *
from axionbloch.SimuTools import MagField, Simulation
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.utils import check


RCF_Freq_Hz = 1e6
signalFreqRot_Hz = 2
T1_s = 1e6

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

simuRate = 500 * unit.Hz
duration = 10.1 * unit.s

# CH3CH2OH
sample = Sample(
    name="Ethanol",
    gamma=gamma_p,
    massDensity=0.78945 * unit.g / unit.cm**3,
    molarMass=46.069 * unit.g / unit.mol,
    numOfSpinsPerMolecule=6 * unit.one,
    T2=T2_s * unit.s,
    T1=T1_s * unit.s,
    vol=1 * unit.cm**3,
    mu=mu_p,
    verbose=False,
)

# set detection magnet
magnet = Magnet(
    name="detection magnet",
    B0=(RCF_Freq_Hz - signalFreqRot_Hz) * unit.Hz / (sample.gamma / (2 * np.pi)),
    FWHM=(1 / (np.pi * Tdelta_s) / RCF_Freq_Hz) * unit.one,
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
    RCF_freq=RCF_Freq_Hz * unit.Hz,
    rate=simuRate,
    duration=duration,
    verbose=False,
)

# set excitation pulse: 90 degree hard pulse
simu.excField.set90DegPulse(
    timeStep_s=simu.timeStep.to_value(unit.s),
    timeLen=simu.timeLen,
    gamma_HzToT=simu.gamma_HzToT,
    t90_s=10 * simu.timeStep.to_value(unit.s),
    nu_rot_Hz=signalFreqRot_Hz,
)

tic = time.perf_counter()
simu.generateTrajectories(integrator="RK4")
toc = time.perf_counter()
print(f"GenerateTrajectory time consumption = {toc-tic:.6f} s")

simu.monitorTrajectories(verbose=True)

save_data = False
if save_data:
    timeStamp_s = simu.getTimeStamp()
    check(simu.excField.B_vec.shape)
    check(simu.trjry.shape)
    np.savez(
        "RF_pulse_simu.npz",
        timeStamp_s=timeStamp_s,
        B_vec=simu.excField.B_vec,
        trjry=simu.trjry,
        T2_s=T2_s,
        Tdelta_s=Tdelta_s,
    )
