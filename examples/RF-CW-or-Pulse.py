# $env:PYTHONPATH = "your:\path\here;$env:PYTHONPATH”

import numpy as np
from functools import partial
import time
from tqdm import tqdm

from axionbloch.SimuTools import MagField, Simulation, gate
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Pickup, Magnet
from axionbloch.utils import giveDateAndTime
from axionbloch.Envelope import PhysicalQuantity, gamma_p, mu_p


RCF_Freq_Hz = 1e6
T1_s = 500

# # short Tdelta
# Tdelta_s = 1.0
# T2_s = 1e2

# short T2
Tdelta_s = 1e2
T2_s = 1.0

# # short Tdelta and T2
# Tdelta_s = 1.0
# T2_s = 1.0
# T1_s = 100.0

# # long Tdelta and T2
# Tdelta_s = 1.0e1
# T2_s = 1.0e1

num_runs = 1
simuRate = PhysicalQuantity(65, "Hz")  #
duration = PhysicalQuantity(50, "s")
timeLen = int((simuRate * duration).convert_to("").value)


nu_a_offsets = np.array([PhysicalQuantity(Delta_nu, "Hz") for Delta_nu in [0.0]])


gradiometer = Pickup(
    name="(old) gradiometer on PEEK",
    Lcoil=PhysicalQuantity(400, "nH"),
    gV=PhysicalQuantity(37.0, "1/m"),  # sample-to-pickup coupling strength
    # assume cylindrical sample (R=4 mm, H=22.53 mm) coupling to the gradiometer
    vol=PhysicalQuantity(np.pi * 14**2 * 22.53, "mm**3"),
)


# CH3CH2OH
sample = Sample(
    name="Ethanol",  # name of the sample
    gamma=gamma_p,  # [Hz/T]. Remember input it with 2 * np.pi
    massDensity=PhysicalQuantity(0.78945, "g / cm**3 "),
    molarMass=PhysicalQuantity(46.069, "g / mol"),  # molar mass
    numOfSpinsPerMolecule=PhysicalQuantity(6, ""),  # number of spins per molecule
    T2=PhysicalQuantity(T2_s, "s"),  #
    T1=PhysicalQuantity(T1_s, "s"),  #
    vol=PhysicalQuantity(1, "cm**3"),
    mu=mu_p,  # magnetic dipole moment
    # boilpt=351.38,  # [K]
    # meltpt=159.01,  # [K]
    verbose=False,
)

magnet_det = Magnet(
    name="detection magnet",
    B0=PhysicalQuantity(RCF_Freq_Hz - 0, "Hz") / (sample.gamma / (2 * np.pi)),
    FWHM=PhysicalQuantity(1 / (np.pi * Tdelta_s) / RCF_Freq_Hz, ""),
    nFWHM=10.0,
)
magnet_det.setHomogeneity(
    # numPt=1,
    # numPt=400,
    numPt=int(
        11
        # + 2000
        + duration.value_in("s")
        * 2
        * magnet_det.nFWHM
        * magnet_det.FWHM_T
        * sample.gamma.value_in("Hz/T")
        * 1
    ),  # numPt >= 2 pi * Delta_nu * duration
    showplt=False,
    # verbose: bool = False,
)
# magnet_det.setHomogeneity1(
#     # numPt=1,
#     # numPt=800,  # numPt >= 2 pi * Delta_nu * duration
#     # showplt=True,
#     # verbose: bool = False,
# )
print("magnet_det.numPt =", magnet_det.numPt)

Brms = 1e-10
nu_a = -0.7
use_stoch = True

savedir = r"src\tests\20251031-template-for-simulations/"
timestr = giveDateAndTime()


excField = MagField(name="RF pulse")

simu = Simulation(
    name="simulation template",
    sample=sample,  # class Sample
    # pickup=gradiometer,
    # SQUID=C649_O12,
    magnet_pol=None,
    magnet=magnet_det,
    # LIA=LIA,
    init_time=0.0,  # [s]
    station=None,
    init_M=1.0,
    init_M_theta=0.0,  # [rad]
    init_M_phi=0.0,  # [rad]
    RCF_freq=PhysicalQuantity(RCF_Freq_Hz, "Hz"),
    rate=simuRate,  #
    duration=duration,
    excField=excField,
    verbose=True,
)


# print(magnet_det.FWHM.convert_to("ppm"))

# FWHM_Hz = (simu.sample.gamma * magnet_det.B0 * magnet_det.FWHM / (2 * np.pi)).value_in(
#     "Hz"
# )
# numPt = 400
# print(
#     f"simu.duration_s: {simu.duration_s:g} and numPt / lw_Hz: {numPt / FWHM_Hz /(4 * np.pi):g}"
# )

# # the spacing between two Larmor frequencies must be smaller than the RBW
# if (4 * np.pi) * FWHM_Hz / numPt > simu.duration_s:
# # alternatively if simu.duration_s > numPt / FWHM_Hz / (4 * np.pi):
#     print("WARNING: numPt may be too few. ")
# magnet_det.setHomogeneity(
#     numPt=numPt,
#     # lw: Optional[PhysicalQuantity] = None,
#     # verbose: bool = False,
# )

excitation_type = "CW"
# excitation_type = "Pulse"

for j, nu_a_offset in enumerate((nu_a_offsets)):
    for i in tqdm(range(num_runs)):
        rand_seed = i

        if excitation_type == "CW":
            # CW excitation
            simu.excField.setXYPulse(
                timeStamp=simu.timeStamp_s,
                B1=1.0e-6
                * 2
                * np.pi
                / 2.0
                / simu.gamma_HzToT
                / (5 * 1e-3),  # amplitude of the excitation pulse in [T]
                nu_rot=0,
                init_phase=0,
                # direction: np.ndarray,  #  not needed now
                duty_func=partial(gate, start=0, stop=simu.duration_s),
            )
        elif excitation_type == "Pulse":
            # hard pulse excitation
            simu.excField.setXYPulse(
                timeStamp=simu.timeStamp_s,
                B1=2
                * np.pi
                / 2.0
                / simu.gamma_HzToT
                / (5 * simu.timeStep_s),  # amplitude of the excitation pulse in [T]
                nu_rot=0,
                init_phase=0,
                # direction: np.ndarray,  #  not needed now
                duty_func=partial(gate, start=0, stop=5 * simu.timeStep_s),
            )
        else:
            raise ValueError("excitation_type not found. ")

        simu.excType = excitation_type
        # toc = time.perf_counter()
        # print(f"setALP_Field() time consumption = {toc-tic:.3f} s")

        # tic = time.perf_counter()
        # simu.generateTrajectory_1LoopByNb(verbose=False)
        # toc = time.perf_counter()
        # print(
        #     f"{simu.generateTrajectory_1LoopByNb.__name__} time consumption = {toc-tic:.6f} s"
        # )
        # simu.monitorTrajectory(verbose=True)

        # tic = time.perf_counter()
        # simu.generateTrajectory_2LoopsByNb(verbose=False)
        # toc = time.perf_counter()
        # print(
        #     f"{simu.generateTrajectory_2LoopsByNb.__name__} time consumption = {toc-tic:.6f} s"
        # )
        # simu.monitorTrajectory(verbose=True)

        tic = time.perf_counter()
        simu.generateTrajectory_vectorized(verbose=False)
        toc = time.perf_counter()
        print(
            f"{simu.generateTrajectory_vectorized.__name__} time consumption = {toc-tic:.6f} s"
        )
        simu.monitorTrajectory(verbose=True)
        # simu.VisualizeTrajectory3D(
        #     plotrate=1e3,  # [Hz]
        #     # rotframe=True,
        #     verbose=False,
        # )
        # Delta_nu_a = 1.2  # Hz
        # tau_a = 1 / (np.pi * Delta_nu_a)
        # T2 = simu.sample.T2
        # tau = np.sqrt(tau_a * T2)
        # # check(tau)
        # check(1 / (np.pi * T2))
        # check(1 / (np.pi * tau))
        # check(1 / (tau))
        # simu.compareBandSig()

        # normalized magnetization
        # sin_theta = np.sqrt(simu.trjry[0:-1, 0] ** 2 + simu.trjry[0:-1, 1] ** 2)

        # theta = run_simulation(rand_seed)
        # results[i] = sin_theta
        # np.save(f"theta_run_{i}.npy", theta)
        # simu.saveToFile_h5(pathAndName=savedir + "test_save")

        #
        # data_file_name = savedir + "theta_all_runs_" + timestr + f"_{j}.npz"
        # np.savez(
        #     data_file_name,
        #     timeStamp=simu.timeStamp,
        #     sin_theta=results,
        #     simuRate=simuRate,
        #     duration=duration,
        #     demodfreq=demodFreq,
        #     T2=T2,
        #     T1=T1,
        #     Brms=Brms,
        #     nu_a=nu_a,
        #     use_stoch=True,
        #     gyroratio=ExampleSample10MHzT.gamma,
        # )

        # np.save(savedir + f"theta_all_runs_" + timestr + ".npy", results)

        # # Sample data
        # theta = np.random.rand(1000)
        # simuRate = 1000
        # duration = 1.0

        # # Save DataFrame as pickle (preserves attrs)

        # data_file_name = savedir + f"theta_all_runs_" + timestr + ".pkl"
        # df.to_pickle(data_file_name)

        # with open(data_file_name, "wb") as f:
        #     pickle.dump({"df": df, "theta": results}, f)
