# Example script to run stochastic axion wind NMR simulations
import os

import numpy as np
import time

from axionbloch.MilkyWayAxionHalo import MilkyWayAxionHalo
from axionbloch.utils import check, dualLorentzian
from axionbloch.SimuTools import MagField, Simulation, Simulations
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Magnet
from axionbloch.enphylope import PhysicalQuantity as PQ
from axionbloch.constants import gamma_p, mu_p, gamma_Xe129, mu_Xe129
from axionbloch.SimuTypes import SimuParams

# set directory for saving data
script_path = os.path.abspath(__file__)
savedir = script_path

# set the sample (including T2 and T1)
# CH3OH
sample = Sample(
    name="Liquid Xe-129",  # name of the sample
    gamma=gamma_Xe129,  # [Hz/T]. Remember input it with 2 * np.pi
    massDensity=PQ(3.1, "g / cm**3 "),  # mass density at STP
    molarMass=PQ(131.29, "g / mol"),  # molar mass [g/mol]
    numOfSpinsPerMolecule=PQ(1, ""),  # number of spins per molecule
    T2=PQ(10, "minute"),  #
    T1=PQ(15, "minute"),  #
    vol=PQ(1, "cm**3"),
    mu=mu_Xe129,  # magnetic dipole moment
    temp=PQ(300, "K"),  # room temperature
    pol=PQ(0.5, ""),  # polarization
    verbose=False,
)

# axion Compton frequency
nu_a_array = np.array([PQ(nu, "Hz") for nu in [1e3]])
# nu_a_array = np.array([PQ(nu, "Hz") for nu in [1e3, 1e6, 1e9]])

# set magnet homogeneity
mag_FWHMs = np.array([PQ(nu, "ppm") for nu in [2]])

# set the strength of the pseudomagnetic field (rms of the field) Brms
B_a_rms = None
# or by setting the axion-nucleon coupling strength gaNN
gaNN = PQ(1.0e-9, "GeV**(-1)")

# set number of simulation runs
numFields = 1000

init_M = PQ(1.0, "")  # initial magnetization vector amplitude
# init_M = None # if None, it will be set to the magnetization determined by the sample polarization
init_M_theta = PQ(0, "rad")
init_M_phi = PQ(0, "rad")

simulations = Simulations()
# list of simulation parameter dictionaries
all_params = []

for nu_a in nu_a_array:
    for mag_FWHM in mag_FWHMs:
        nu_a_Hz = nu_a.value_in("Hz")
        print("Axion Compton frequency =", nu_a, flush=True)
        time.sleep(0.1)
        axion = MilkyWayAxionHalo(
            name="axion",
            nu_a=nu_a,
            gaNN=gaNN,
        )

        # set RCF frequency to it RCF_Freq_Hz = nu_a*(1+v_a^2/c^2)
        RCF_freq: PQ = axion.nu_a_eff
        RCF_freq_Hz = RCF_freq.value_in("Hz")

        # set the detection magnet (bias field) accordingly
        # also set detection magnet (bias field) homogeneity
        magnet = Magnet(
            name="detection magnet",
            B0=RCF_freq / (sample.gamma / (2 * np.pi)),
            direction=[0, 0, 1],
            FWHM=mag_FWHM,
            nFWHM=10.0,
        )
        # initialize excitation field
        excField = MagField(name="ALP field gradient")

        if B_a_rms is None:
            if axion.gaNN is None:
                raise ValueError("B_a_rms and gaNN cannot be both None")
            else:
                B_a_rms = axion.getRabiFreq(verbose=True) / (sample.gamma)
                B_a_rms = B_a_rms.to("T")
                print(f"Calculated B_a_rms from gaNN = {B_a_rms}", flush=True)
        key_info = {"mag_FWHM": mag_FWHM, "nu_a": axion.nu_a}
        # duration = 10 * axion.tau_a_est
        duration = PQ(4000, "s")
        rate = PQ(1, "Hz")
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

simu_all = Simulations(name="Axion-Xe_NMR-simulations", all_params=all_params)
# print("simu_all.run started", flush=True)
simu_all.run(autoStart=False, verbose=True)

# simu_all.pool[0].simu.monitorTrajectories()
for i in range(len(simu_all.pool)):
    simu_all.pool[i].simu.keepMeanStd()
    simu_all.pool[i].simu.displayTrjries()
    check(simu_all.pool[i].simu.T2star_s)
    check(simu_all.pool[i].simu.Tdelta_s)
    check(simu_all.pool[i].simu.T2_s)
simu_all.saveToPkl(
    dir=os.path.dirname(os.path.abspath(__file__))#, fname="new_simulation"
)
save_data = True
if save_data:
    i=0
    simu: Simulation = simu_all.pool[i].simu
    timeStamp_s = simu.getTimeStamp()
    # check(simu.excField.B_vec.shape)
    # check(simu.trjry.shape)
    np.savez(
        "C:\\Users\\zhenf\\D\\Yu0702\\CASPEr-Collaboration\\AxionBloch-paper/figures/Axion-Xe_NMR-simulations.npz",
        timeStamp_s=timeStamp_s,
        B_vec_mean=simu.excField.B_vec_mean,
        B_vec_std=simu.excField.B_vec_std,
        M_mean=simu.M_mean,
        M_std=simu.M_std,
        Mxy_mrs=simu.Mxy_mrs,
        Mxy_srs=simu.Mxy_srs,
        # trjry=simu.trjry,
        T2_s=sample.T2.value_in("s"),
        Tdelta_s=simu_all.pool[i].simu.Tdelta_s,
        T_1_s=sample.T1.value_in("s"),
        pol=sample.pol,
        # init_M=simu.init_M.value_in(""),
    )
