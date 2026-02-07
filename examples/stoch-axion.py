# Example script to run stochastic axion wind NMR simulations
import os

import numpy as np
import time

from axionbloch.AxionWind import AxionWind
from axionbloch.SimuTools import MagField, Simulations
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Magnet
from axionbloch.Envelope import PhysicalQuantity, gamma_p, mu_p
from axionbloch.SimuTypes import SimuParams

# set directory for saving data
script_path = os.path.abspath(__file__)
savedir = script_path

# set the sample (including T2 and T1)
# CH3OH
sample = Sample(
    name="C-12 Methanol",  # name of the sample
    gamma=gamma_p,  # [Hz/T]. Remember input it with 2 * np.pi
    massDensity=PhysicalQuantity(0.792, "g / cm**3 "),
    molarMass=PhysicalQuantity(32.04, "g / mol"),  # molar mass
    numOfSpinsPerMolecule=PhysicalQuantity(4, ""),  # number of spins per molecule
    T2=PhysicalQuantity(1, "s"), 
    T1=PhysicalQuantity(3, "s"), 
    vol=PhysicalQuantity(1, "cm**3"),
    mu=mu_p,  # magnetic dipole moment
    verbose=False,
)

# axion Compton frequency
nu_a_array = np.array(
    [
        PhysicalQuantity(nu, "Hz")
        for nu in [1e6]
    ]
)

# set magnet homogeneity
mag_FWHMs = np.array([PhysicalQuantity(nu, "ppm") for nu in [1e1]])

# set the strength of the pseudomagnetic field (rms of the field) Brms
B_a_rms = PhysicalQuantity(1e-15, "T")

# set number of simulation runs
numFields = 1

init_M = PhysicalQuantity(1.0, "")  # initial magnetization vector amplitude
init_M_theta = PhysicalQuantity(0, "rad")
init_M_phi = PhysicalQuantity(0, "rad")

simulations = Simulations()
# list of simulation parameter dictionaries
all_params = []

for nu_a in nu_a_array:
    for mag_FWHM in mag_FWHMs:
        nu_a_Hz = nu_a.value_in("Hz")
        print("Axion Compton frequency =", nu_a, flush=True)
        time.sleep(0.1)
        axion = AxionWind(
            name="axion",
            nu_a=nu_a,
        )

        # set RCF frequency to it RCF_Freq_Hz = nu_a*(1+v_a^2/c^2)
        RCF_freq: PhysicalQuantity = axion.nu_a_eff
        RCF_freq_Hz = RCF_freq.value_in("Hz")

        # set the detection magnet (bias field) accordingly
        # also set detection magnet (bias field) homogeneity
        magnet = Magnet(
            name="detection magnet",
            B0=RCF_freq / (sample.gamma / (2 * np.pi)),
            FWHM=mag_FWHM,
            nFWHM=10.0,
        )
        # initialize excitation field
        excField = MagField(name="ALP field gradient")

        key_info = {"mag_FWHM": mag_FWHM, "nu_a": axion.nu_a}
        params: SimuParams = {
            "key_info": key_info,
            "axion": axion,
            "sample": sample,
            "magnet": magnet,
            "excField": excField,
            "B_a_rms": B_a_rms,
            "numFields": numFields,
            "rand_seed": 10,
            "init_M": init_M,
            "init_M_theta": init_M_theta,
            "init_M_phi": init_M_phi,
            "rate": None,
            "duration": None,
        }
        all_params.append(params)

simu_all = Simulations(name="ALP-proton_NMR-simulations", all_params=all_params)
# print("simu_all.run started", flush=True)
simu_all.run(autoStart=False, verbose=True)

simu_all.saveToPkl(
    dir=os.path.dirname(os.path.abspath(__file__)), fname="new_simulation"
)
