# Example script to run axion NMR simulations

import numpy as np

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
    gamma=gamma_Xe129,
    massDensity=PQ(3.1, "g / cm**3 "),
    molarMass=PQ(131.29, "g / mol"),
    numOfSpinsPerMolecule=PQ(1, ""),
    T2=PQ(10, "minute"),
    T1=PQ(15, "minute"),
    vol=PQ(1, "cm**3"),
    mu=mu_Xe129,
    temp=PQ(300, "K"),
    pol=PQ(0.5, ""),
)

# Define the axion field with axion Compton frequency
# and axion-nucleon coupling strength
axion = MilkyWayAxionHalo(
    nu_a=PQ(1, "kHz"),
    g_aNN=PQ(1.0e-9, "GeV**(-1)"),
)

# Set the bias field strength, direction, and homogeneity
magnet = Magnet(
    B0=axion.nu_a_eff / (sample.gamma / (2 * np.pi)),
    direction=[0, 0, 1],
    FWHM=PQ(2, "ppm"),
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
