# example script to simulate CW NMR experiment with hyperpolarized sample
import time

from axionbloch.dependency import *
from axionbloch.SimuTools import MagField, Simulation
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Magnet
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

simuRate = 500 * unit.Hz
duration = 20 * unit.s

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
    temp=300 * unit.K,
    pol=1e-2 * unit.one,
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

# set excitation pulse: CW drive with zero amplitude (free-induction decay)
simu.excField.setXYPulse(
    timeStep_s=simu.timeStep.to_value(unit.s),
    timeLen=simu.timeLen,
    B1_T=0,
    nu_rot_Hz=signalFreqRot_Hz,
)

tic = time.perf_counter()
simu.generateTrajectories(integrator="RK4")
toc = time.perf_counter()
print(f"GenerateTrajectory time consumption = {toc-tic:.6f} s")

simu.monitorTrajectories(verbose=True)

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
        pol=sample.pol.to_value(unit.one),
        init_M=simu.init_M.to_value(unit.dimensionless_unscaled),
    )
