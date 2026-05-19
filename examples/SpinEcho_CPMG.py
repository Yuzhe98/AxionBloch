# $env:PYTHONPATH = "your:\path\here;$env:PYTHONPATH"
# $env:PYTHONPATH = "C:\Users\zhenf\D\Yu0702\axionbloch;$env:PYTHONPATH"
import time

from axionbloch.dependency import *
from axionbloch.SimuTools import MagField, Simulation
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.utils import check


RCF_Freq_Hz = 1e6
signalFreqRot_Hz = 1
T1_s = 1e6

# short Tdelta
Tdelta_s = 1.0
T2_s = 10.0

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
    verbose=False,
)

# set detection magnet
magnet = Magnet(
    name="detection magnet",
    B0=(RCF_Freq_Hz - signalFreqRot_Hz) * unit.Hz / (sample.gamma / (2 * np.pi)),
    FWHM=(1 / (np.pi * Tdelta_s) / RCF_Freq_Hz) * unit.one,
    nFWHM=20.0,
)

excField = MagField(name="RF pulse")

rand_seed = 0

simu = Simulation(
    name="NMR simulation",
    sample=sample,
    magnet=magnet,
    excField=excField,
    rate=500 * unit.Hz,
    RCF_freq=RCF_Freq_Hz * unit.Hz,
    duration=20 * unit.s,
    verbose=True,
)

magnet.setHomogeneity(numPt=500)

simu.excField.setCPMGPulseTrain(
    timeStep_s=simu.timeStep.to_value(unit.s),
    timeLen=simu.timeLen,
    gamma_HzToT=simu.gamma_HzToT,
    t90_s=10 * simu.timeStep.to_value(unit.s),
    tau_s=4 * Tdelta_s,
    numEcho=2,
    nu_rot_Hz=signalFreqRot_Hz,
    init_phase=0,
    verbose=True,
)

tic = time.perf_counter()
simu.generateTrajectories(integrator="RK4")
toc = time.perf_counter()
print(f"{simu.generateTrajectories.__name__} time consumption = {toc-tic:.3g} s")

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
