# Example script to run axion NMR simulations

from axionbloch.dependency import *

# For handling quantities with units.
from axionbloch.enphylope import PhysicalQuantity as PQ

# Gyromagnetic ratio and magnetic dipole moment of Xe-129
from axionbloch.constants import gamma_Xe129, mu_Xe129

# classes for simulations
from axionbloch.SimuTools import MagField, Simulations
from axionbloch.SimuTypes import SimuParams
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Magnet
from axionbloch.MilkyWayAxionHalo import MilkyWayAxionHalo

# Define the Xe-129 sample with gyromagnetic ratio,
# mass density, molar mass, number of spins per molecule,
# relaxation times, volume, magnetic dipole moment,
# temperature, and polarization.

sample = Sample(
    name="Liquid Xe-129",  # name of the sample
    gamma=gamma_Xe129,  # [Hz/T]. Remember input it with 2 * np.pi
    massDensity=3.1 * unit.g * unit.cm ** (-3),  # mass density at STP
    molarMass=131.29 * unit.g / unit.mol,  # molar mass [g/mol]
    numOfSpinsPerMolecule=1 * unit.one,  # number of spins per molecule
    T2=10 * unit.minute,  #
    T1=15 * unit.minute,  #
    vol=1 * unit.cm**3,
    mu=mu_Xe129,  # magnetic dipole moment
    verbose=False,
)

# # CH3OH
# methanol = Sample(
#     name="C-12 Methanol",  # name of the sample
#     gamma=gamma_p,  # [Hz/T]. Remember input it with 2 * np.pi
#     massDensity=0.792 * unit.g * unit.cm ** (-3),
#     molarMass=32.04 * unit.g / unit.mol,  # molar mass
#     numOfSpinsPerMolecule=4 * unit.one,  # number of spins per molecule
#     T2=1 * unit.s,  #
#     T1=5 * unit.s,  #
#     vol=1 * unit.cm**3,
#     mu=mu_p,  # magnetic dipole moment
#     # boilpt=337.8,  # [K]
#     # meltpt=175.6,  # [K]
#     verbose=False,
# )

# # CH3CH2OH
# ethanol = Sample(
#     name="Ethanol",  # name of the sample
#     gamma=gamma_p,  # [Hz/T]. Remember input it with 2 * np.pi
#     massDensity=0.78945 * unit.g * unit.cm ** (-3),
#     molarMass=46.069 * unit.g / unit.mol,  # molar mass
#     numOfSpinsPerMolecule=6 * unit.one,  # number of spins per molecule
#     T2=1 * unit.s,  #
#     T1=5 * unit.s,  #
#     vol=1 * unit.cm**3,
#     mu=mu_p,  # magnetic dipole moment
#     # boilpt=351.38,  # [K]
#     # meltpt=159.01,  # [K]
#     verbose=False,
# )

# Define the axion field with axion Compton frequency
# and axion-nucleon coupling strength
axion = MilkyWayAxionHalo(
    name="Milky Way Axion Halo",
    nu_a=1 * unit.kHz,
    g_aNN=1.0e-9 * unit.GeV ** (-1),
    verbose=False,
)

# Set the bias field strength, direction, and homogeneity
magnet = Magnet(
    B0=axion.nu_a_eff / (sample.gamma / (2 * np.pi)),
    direction=[0, 0, 1],
    FWHM=2 * ppm,
)

# Bundle all inputs into one dictionary
params: SimuParams = {
    "key_info": {"nu_a": axion.nu_a},
    "axion": axion,
    "sample": sample,
    "magnet": magnet,
    "excField": MagField(),  # excitation field
    # rms of the pseudomagnetic field
    "B_a_rms": axion.getRabiFreq() / (sample.gamma),
    # Number of random field realizations.
    "numFields": 1000,
    "rand_seed": 10,  # random seed
    # amplitude, polar and azimuthal angle
    # of the initial magnetization
    "init_M": PQ(1.0, ""),
    "init_M_theta": PQ(0, "rad"),
    "init_M_phi": PQ(0, "rad"),
    # sampling rate and duration of the time series
    "rate": PQ(1, "Hz"),
    "duration": PQ(4000, "s"),
}

# Create and execute the simulation job collection
simulations = Simulations(all_params=[params])

# run the simulation
simulations.run(verbose=True)

# Post-process results with summary stats and plotting
for i, item in enumerate(simulations.pool):
    item.simu.keepMeanStd()
    item.simu.displayTrjries()

# Save to .pkl file for later analysis
simulations.saveToPkl(dir="path_to_save")
