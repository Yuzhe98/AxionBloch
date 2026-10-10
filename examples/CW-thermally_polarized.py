# Example script: CW (continuous-wave) NMR free-induction-decay simulation
#
# A CW excitation field drives the spin ensemble continuously at signalFreqRot
# in the rotating frame.  The simulation records the magnetization trajectory
# over `duration` seconds.
import time

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.dependency import np, unit, PI
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation

RCF_Freq = 1 * unit.MHz
signalFreqRot = 0.5 * unit.Hz
T1 = 5 * unit.s

Tdelta = 100 * unit.s
T2 = 1.0 * unit.s

simuRate = 500 * unit.Hz
duration = 20 * unit.s

# CH3CH2OH sample
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

FWHM = (1 / (np.pi * Tdelta) / RCF_Freq) * unit.one

# Detection (bias) magnet — B0 tuned so that the Larmor frequency equals
# the carrier plus signalFreqRot
magnet = Magnet(
    name="detection magnet",
    B0=(RCF_Freq - signalFreqRot) / (sample.gamma / (2 * PI)),
    FWHM=FWHM,
    nFWHM=10.0,
)
magnet.setHomogeneity(numPt=2000, showPlot=False, verbose=True)
print(f"numPt for magnet homogeneity = {magnet.numPt}")

# Excitation field (CW)
excField = MagField(name="CW excitation")

simu = Simulation(
    name="CW NMR simulation",
    sample=sample,
    magnet=magnet,
    excField=excField,
    RCF_freq=RCF_Freq,
    rate=simuRate,
    duration=duration,
    verbose=False,
)

# CW drive: constant-envelope XY pulse at the signal frequency
B1 = 0.0005 * unit.Hz / (sample.gamma / (2 * PI))
init_phase = 0 * unit.rad
simu.excField.setXYPulse(
    timeStep=simu.timeStep,
    timeLen=simu.timeLen,
    B1=B1,  # field amplitude corresponding to a 0.005 Hz Rabi frequency
    nu_rot=signalFreqRot,
    init_phase=init_phase,
)

tic = time.perf_counter()
simu.generateTrajectories(integrator="RK4")
toc = time.perf_counter()
print(f"generateTrajectories time consumption = {toc - tic:.6f} s")

# # Save raw plotting arrays before keepMeanStd() releases them.
# timeStamp_s = simu.timeStep.to_value(unit.s) * np.arange(simu.trjry.shape[1])
# output_path = "CW_thermally_polarized_simu.npz"
# np.savez_compressed(
#     output_path,
#     timeStamp_s=timeStamp_s,
#     B_vec=simu.excField.B_vec.to_value(unit.T),
#     trjry=simu.trjry,
#     T2_s=simu.sample.T2.to_value(unit.s),
#     Tdelta_s=simu.Tdelta.to_value(unit.s),
#     signalFreqRot_Hz=signalFreqRot.to_value(unit.Hz),
#     init_phase_rad=init_phase.to_value(unit.rad),
#     B1_rot_T=(0.5 * B1).to_value(unit.T),
#     gamma_rad_s_T=sample.gamma.to_value(unit.rad / unit.s / unit.T),
# )
# print(f"Saved plotting data to {output_path}")

simu.keepMeanStd()
simu.displayTrjries(verbose=True)
# simu.monitorTrajectories(verbose=True)
