# Example script: CW (continuous-wave) NMR free-induction-decay simulation
#
# A CW excitation field drives the spin ensemble continuously at signalFreqRot_Hz
# in the rotating frame.  The simulation records the magnetisation trajectory
# over `duration` seconds.
import time

from axionbloch.dependency import *
from axionbloch.SimuTools import MagField, Simulation
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.utils import check


RCF_Freq_Hz = 1e6       # rotating-frame carrier frequency (Hz)
signalFreqRot_Hz = 1    # signal offset from carrier in the rotating frame (Hz)
T1_s = 1e10             # longitudinal relaxation time (s)

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
duration = 20 * unit.s

# CH3CH2OH sample
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

# Detection (bias) magnet — B0 tuned so that the Larmor frequency equals
# the carrier plus signalFreqRot_Hz
magnet = Magnet(
    name="detection magnet",
    B0=(RCF_Freq_Hz - signalFreqRot_Hz) * unit.Hz / (sample.gamma / (2 * np.pi)),
    FWHM=(1 / (np.pi * Tdelta_s) / RCF_Freq_Hz) * unit.one,
    nFWHM=20.0,
)
magnet.setHomogeneity(numPt=500)
print(f"numPt for magnet homogeneity = {magnet.numPt}")

# Excitation field (CW)
excField = MagField(name="CW excitation")

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

# CW drive: constant-envelope XY pulse at the signal frequency
simu.excField.setXYPulse(
    timeStep_s=simu.timeStep.to_value(unit.s),
    timeLen=simu.timeLen,
    B1_T=1.0e-11,
    nu_rot_Hz=signalFreqRot_Hz,
)

tic = time.perf_counter()
simu.generateTrajectories(integrator="RK4")
toc = time.perf_counter()
print(f"GenerateTrajectory time consumption = {toc - tic:.6f} s")

simu.monitorTrajectories(verbose=True)

save_data = False
if save_data:
    timeStamp_s = simu.getTimeStamp()
    check(simu.excField.B_vec.shape)
    check(simu.trjry.shape)
    np.savez(
        "RF_CW_simu.npz",
        timeStamp_s=timeStamp_s,
        B_vec=simu.excField.B_vec,
        trjry=simu.trjry,
        T2_s=T2_s,
        Tdelta_s=Tdelta_s,
    )
