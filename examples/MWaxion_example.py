# Example script to run stochastic axion wind NMR simulations
import os

import numpy as np

from axionbloch.enphylope import PhysicalQuantity as PQ
from axionbloch.constants import gamma_Xe129, mu_Xe129

from axionbloch.SimuTools import MagField, Simulations
from axionbloch.SimuTypes import SimuParams

from axionbloch.Sample import Sample
from axionbloch.Apparatus import Magnet

from axionbloch.MilkyWayAxionHalo import MilkyWayAxionHalo

# Define the Xe-129 sample and relaxation parameters used in the simulation.
sample = Sample(
    name="Liquid Xe-129",  # name of the sample
    gamma=gamma_Xe129,  # gyromagnetic ratio
    massDensity=PQ(3.1, "g / cm**3 "),  # mass density at STP
    molarMass=PQ(131.29, "g / mol"),  # molar mass
    numOfSpinsPerMolecule=PQ(1, ""),  # number of spins per molecule
    T2=PQ(10, "minute"),  # transverse relaxation time
    T1=PQ(15, "minute"),  # longitudinal relaxation time
    vol=PQ(1, "cm**3"),
    mu=mu_Xe129,  # magnetic dipole moment
    temp=PQ(300, "K"),  # room temperature
    pol=PQ(0.5, ""),  # polarization
    verbose=False,
)

# Axion Compton frequency
nu_a = PQ(1, "kHz")
# Set the axion-nucleon coupling strength gaNN
gaNN = PQ(1.0e-9, "GeV**(-1)")

axion = MilkyWayAxionHalo(
    name="axion",
    nu_a=nu_a,
    g_aNN=gaNN,
)


# Bias-field inhomogeneity model
mag_FWHM = PQ(2, "ppm")

# set the strength of the pseudomagnetic field (rms of the field)
# by setting the axion-nucleon coupling strength gaNN

# Number of independent random field realizations.
numFields = 1

init_M = PQ(1.0, "")  # initial magnetization vector amplitude
# init_M = None # if None, it will be set to the magnetization determined by the sample polarization
init_M_theta = PQ(0, "rad")
init_M_phi = PQ(0, "rad")

simulations = Simulations()
# list of simulation parameter dictionaries
all_params = []

# for nu_a in nu_a_array:
#     for mag_FWHM in mag_FWHMs:
# nu_a_Hz = nu_a.value_in("Hz")
# print("Axion Compton frequency =", nu_a, flush=True)

# Effective resonance condition includes kinetic broadening of the axion field.
# RCF_Freq_Hz ~= nu_a * (1 + v_a^2/c^2)
RCF_freq: PQ = axion.nu_a_eff
RCF_freq_Hz = RCF_freq.value_in("Hz")

# Tune the static bias field B0 to match the effective resonance frequency.
magnet = Magnet(
    name="detection magnet",
    B0=RCF_freq / (sample.gamma / (2 * np.pi)),
    direction=[0, 0, 1],
    FWHM=mag_FWHM,
    nFWHM=10.0,
)
# initialize excitation field
excField = MagField(name="ALP field gradient")

# Convert gaNN to an equivalent RMS pseudo-magnetic driving field when needed.

B_a_rms = axion.getRabiFreq(verbose=True) / (sample.gamma)
B_a_rms = B_a_rms.to("T")
print(f"Calculated B_a_rms from gaNN = {B_a_rms}", flush=True)

key_info = {"mag_FWHM": mag_FWHM, "nu_a": axion.nu_a}
# duration = 10 * axion.tau_a_est
duration = PQ(4000, "s")
rate = PQ(1, "Hz")
# Bundle all inputs into one simulation configuration dictionary.
params: SimuParams = {
    "key_info": key_info,
    "axion": axion,
    "sample": sample,
    "magnet": magnet,
    "excField": excField,
    "B_a_rms": B_a_rms,
    "numFields": numFields,
    "rand_seed": 10,
    "init_M": None,
    "init_M_theta": init_M_theta,
    "init_M_phi": init_M_phi,
    # "rate": None,
    # "duration": None,
    "rate": rate,
    "duration": duration,
}
all_params.append(params)

# Create and execute the simulation job collection.
simu_all = Simulations(name="Axion-Xe_NMR-simulations", all_params=all_params)
# print("simu_all.run started", flush=True)
simu_all.run(autoStart=False, verbose=True)


# Persist outputs near this example script for later analysis.
simu_all.saveToPkl(dir=os.path.dirname(os.path.abspath(__file__)))

# simu_all.pool[0].simu.monitorTrajectories()
# Post-process each trajectory with summary stats and quick plotting.
for i in range(len(simu_all.pool)):
    simu_all.pool[i].simu.keepMeanStd()
    simu_all.pool[i].simu.displayTrjries()
    # check(simu_all.pool[i].simu.T2star_s)
    # check(simu_all.pool[i].simu.Tdelta_s)
    # check(simu_all.pool[i].simu.T2_s)
