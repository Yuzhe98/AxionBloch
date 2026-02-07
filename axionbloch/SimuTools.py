import os
from typing import Optional

import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter, FuncFormatter

# import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D  # for type hinting

from tqdm import tqdm

import numba as nb
from numba import njit

from scipy.stats import uniform, expon
from scipy.fft import ifft

# for saving data
import pickle
import h5py

from axionbloch.utils import (
    PhysicalObject,
    axion_lineshape,
    check,
    save_phys_quantity,
    giveDateAndTime,
    sci_fmt,
)
from axionbloch.DataAnalysis import Signal
from axionbloch.Sample import Sample

from axionbloch.Apparatus import Magnet
from axionbloch.Envelope import PhysicalQuantity
from axionbloch.AxionWind import AxionWind, Station
from axionbloch.SimuTypes import SimuParams, SimuEntry

import blochSimulation_c_ext.blochSimulation_c as bh

RECORD_RUNTIME = True

# estimated run time for a single step in setField
T_SETFIELD_S = 1.2e-7
# estimated run time for a single simulation step
T_SIMUSTEP_S = 1.2e-09


def gate(x: float | np.ndarray, start: float, stop: float) -> float:
    """
    Returns 1 if start <= x <= stop, else returns 0.

    Parameters:
    x : float or array-like
        The input value(s) where the function is evaluated.

    Returns:
    float or array-like
        1 if start <= x <= stop, else 0.
    """
    return np.where((start <= x) & (x < stop), 1.0, 0.0)


class MagField(PhysicalObject):
    """
    DC / AC (pseudo)magnetic fields
    """

    def __init__(
        self,
        name="B field",
    ):
        super().__init__()
        self.name = name
        self.nu_Hz = None

    def setXYPulse(
        self,
        timeStamp: np.ndarray,
        B1: float,  # amplitude of the excitation pulse in (T)
        nu_rot: float,
        init_phase: float,
        # direction: np.ndarray,  #  not needed now
        duty_func,
        verbose: bool = False,
    ):
        """
        generate a pulse in the rotating frame
        """
        # direction_norm = direction / np.dot(direction, direction)

        # excitation along x-axis
        Bx_envelope = (
            1.0
            / 2
            * B1
            * duty_func(timeStamp)
            # * np.dot(np.array([1, 0, 0]), direction_norm)
        )
        # check(Bx_envelope[0:10])
        Bx_envelope = np.multiply(
            Bx_envelope, np.cos(2 * np.pi * nu_rot * timeStamp + init_phase)
        )
        Bx = np.outer(Bx_envelope, np.array([1, 0, 0]))

        # excitation along y-axis
        By_envelope = (
            1.0
            / 2
            * B1
            * duty_func(timeStamp)
            # * np.dot(np.array([0, 1, 0]), direction_norm)
        )
        # check(By_envelope)
        By_envelope = np.multiply(
            By_envelope, np.sin(2 * np.pi * nu_rot * timeStamp + init_phase)
        )
        By = np.outer(By_envelope, np.array([0, 1, 0]))

        # 1st order time-derivate of the excitation along x-axis
        dBxdt_envelope = (
            1.0
            / 2
            * B1
            * duty_func(timeStamp)
            # * np.dot(np.array([1, 0, 0]), direction_norm)
        )
        dBxdt_envelope = np.multiply(
            dBxdt_envelope,
            -2 * np.pi * nu_rot * np.sin(2 * np.pi * nu_rot * timeStamp + init_phase),
        )
        dBxdt = np.outer(dBxdt_envelope, np.array([1, 0, 0]))

        # 1st order time-derivate of the excitation along y-axis
        dBydt_envelope = (
            1.0
            / 2
            * B1
            * duty_func(timeStamp)
            # * np.dot(np.array([0, 1, 0]), direction_norm)
        )
        dBydt_envelope = np.multiply(
            dBydt_envelope,
            2 * np.pi * nu_rot * np.cos(2 * np.pi * nu_rot * timeStamp + init_phase),
        )
        dBydt = np.outer(dBydt_envelope, np.array([0, 1, 0]))

        self.B_vec = Bx + By
        self.dBdt_vec = dBxdt + dBydt
        # sanCheck(Bx, tag="Bx")
        # sanCheck(self.B_vec, tag="self.B_vec")
        # self.dBdt_vec = np.outer(dBxdt + dBydt, direction)
        # self.nu = nu_rot
        # def envelope(timeStamp):
        #     return duty_func(timeStamp) * B1 * np.sin(2 * np.pi * nu_e * timeStamp + init_phase)
        # return

    def setCPMGPulseTrain(
        self,
        timeStep_s: float,
        timeLen: int,
        gamma_HzToT: float,
        t90_s: float,
        tau_s: float,
        numEcho: int,
        nu_rot_Hz: float,
        init_phase: float,
        verbose: bool = False,
    ):
        """
        generate a CPMG pulse train in the rotating frame
        A schematic of the envelop of the pulse train can be found below.
          90deg pulse       180deg pulse                     180deg pulse                  180deg pulse
            ┌───┐            ┌───────┐                       ┌───────┐                       ┌───────┐
            |   |            |       |                       |       |                       |       |
            ┘   └────────────┘       └───────────────────────┘       └───────────────────────┘       └──────────────────────
            ↑   ↑            ↑       ↑                       ↑       ↑                       ↑       ↑
            0  t90          tau   tau+t180                  3tau   3tau+t180                5tau   5tau+t180

        """
        timeStamp_s = timeStep_s * np.arange(timeLen - 1)
        t90Len = int(np.round(t90_s / timeStep_s))
        if t90Len < 3:
            print(f"WARNING: t90Len = {t90Len} < 3")

        t90_s: float = t90Len * timeStep_s
        t180Len: int = 2 * t90Len

        B90_T = 1.01 * np.pi / (gamma_HzToT * t90_s)
        B180_T = (1 - 0.00005) * np.pi / (gamma_HzToT * t90_s)

        tauLen = int(np.round(tau_s / timeStep_s))
        if tauLen < 10 * t180Len:
            print(
                f"WARNING: tauLen = {tauLen} < 10 * t180Len = {10 * t180Len}. Too short! "
            )

        # envelope = np.zeros_like(timeStamp_s)
        Bx = np.zeros_like(timeStamp_s)
        By = np.zeros_like(timeStamp_s)
        dBxdt = np.zeros_like(timeStamp_s)
        dBydt = np.zeros_like(timeStamp_s)

        # set pulses
        # envelope[0:t90Len] += 1

        t90Stamp = 2 * np.pi * nu_rot_Hz * timeStamp_s[0:t90Len] + init_phase
        t180Stamp = 2 * np.pi * nu_rot_Hz * timeStamp_s[0:t180Len] + init_phase

        Bx_90pulse = 0.5 * B90_T * np.cos(t90Stamp)
        By_90pulse = 0.5 * B90_T * np.sin(t90Stamp)
        dBxdt_90pulse = 0.5 * B90_T * (-2) * np.pi * nu_rot_Hz * np.sin(t90Stamp)
        dBydt_90pulse = 0.5 * B90_T * 2 * np.pi * nu_rot_Hz * np.cos(t90Stamp)
        piHalfPulses = [Bx_90pulse, By_90pulse, dBxdt_90pulse, dBydt_90pulse]

        Bx_180pulse = 0.5 * B180_T * np.cos(t180Stamp)
        By_180pulse = 0.5 * B180_T * np.sin(t180Stamp)
        dBxdt_180pulse = 0.5 * B180_T * (-2) * np.pi * nu_rot_Hz * np.sin(t180Stamp)
        dBydt_180pulse = 0.5 * B180_T * 2 * np.pi * nu_rot_Hz * np.cos(t180Stamp)
        piPulses = [Bx_180pulse, By_180pulse, dBxdt_180pulse, dBydt_180pulse]

        # set pi/2 pulses
        for i, B in enumerate([Bx, By, dBxdt, dBydt]):
            B[0:t90Len] = piHalfPulses[i]

        # set pi pulses
        for i in range(numEcho):
            if (1 + i * 2) * tauLen + t180Len >= len(timeStamp_s):
                break
            for j, B in enumerate([Bx, By, dBxdt, dBydt]):
                B[(1 + i * 2) * tauLen : (1 + i * 2) * tauLen + t180Len] = piPulses[j]

        # set pulse amplitude

        # # excitation along x-axis
        # Bx = np.multiply(
        #     envelope, np.cos(2 * np.pi * nu_rot_Hz * timeStamp_s + init_phase)
        # )

        # # excitation along y-axis
        # By = np.multiply(
        #     envelope, np.sin(2 * np.pi * nu_rot_Hz * timeStamp_s + init_phase)
        # )

        # # 1st order time-derivate of the excitation along x-axis
        # dBxdt = np.multiply(
        #     envelope,
        #     -2
        #     * np.pi
        #     * nu_rot_Hz
        #     * np.sin(2 * np.pi * nu_rot_Hz * timeStamp_s + init_phase),
        # )

        # # 1st order time-derivate of the excitation along y-axis
        # dBydt = np.multiply(
        #     envelope,
        #     2
        #     * np.pi
        #     * nu_rot_Hz
        #     * np.cos(2 * np.pi * nu_rot_Hz * timeStamp_s + init_phase),
        # )

        self.B_vec = np.zeros((1, len(Bx), 3))
        self.dBdt_vec = np.zeros((1, len(Bx), 3))

        self.B_vec[:, :, 0] = Bx
        self.B_vec[:, :, 1] = By
        self.dBdt_vec[:, :, 0] = dBxdt
        self.dBdt_vec[:, :, 1] = dBydt
        # sanCheck(Bx, tag="Bx")
        # sanCheck(self.B_vec, tag="self.B_vec")
        # self.dBdt_vec = np.outer(dBxdt + dBydt, direction)
        # self.nu = nu_rot
        # def envelope(timeStamp):
        #     return duty_func(timeStamp) * B1 * np.sin(2 * np.pi * nu_e * timeStamp + init_phase)
        # return

    # def showTSandPSD(
    #     self, dataX: np.ndarray, dataY: np.ndarray, demodfreq, samprate, showplt_opt
    # ):
    #     """
    #     dataX=,
    #     dataY=,
    #     demodfreq=,
    #     samprate=,
    #     showplt_opt=True
    #     """
    #     stream = Signal(
    #         name="ALP field gradient",
    #         # device="Simulation",
    #         # device_id="Simulation",
    #         filelist=[],
    #         verbose=True,
    #     )
    #     stream.attenuation = 0
    #     stream.filterstatus = "off"
    #     stream.DTRC_TC = 0.0
    #     stream.DTRC_order = 0
    #     stream.demodfreq = demodfreq
    #     stream.samprate = samprate

    #     stream.dataX = dataX
    #     stream.dataY = dataY

    #     stream.getNoPulsePSD(
    #         # windowfunction="Hanning",
    #         windowfunction="rectangle",
    #         # decayfactor=-10,
    #         chunksize=None,  # sec
    #         analysisrange=[0, -1],
    #         getstd=False,
    #         stddev_range=None,
    #         selectshots=[],
    #         verbose=False,
    #     )
    #     # stream.FitPSD(
    #     #     fitfunction="Lorentzian",  # 'Lorentzian' 'dualLorentzian' 'tribLorentzian' 'Gaussian 'dualGaussian' 'auto' 'Polyeven'
    #     #     inputfitparas=["auto", "auto", "auto", "auto"],
    #     #     smooth=False,
    #     #     smoothlevel=1,
    #     #     fitrange=["auto", "auto"],
    #     #     alpha=0.05,
    #     #     getresidual=False,
    #     #     getchisq=False,
    #     #     verbose=False,
    #     # )
    #     specxaxis, spectrum, specxunit, specyunit = stream.getSpectrum(
    #         showtimedomain=True,
    #         # showfit=True,
    #         showresidual=False,
    #         showlegend=True,  # !!!!!show or not to show legend
    #         spectype="PSD",  # in 'PSD', 'ASD', 'FLuxPSD', 'FluxASD'
    #         ampunit="V",
    #         specxunit="Hz",  # 'Hz' 'kHz' 'MHz' 'GHz' 'ppm' 'ppb'
    #         # specxlim=[nu_a + demodfreq - 5, nu_a + demodfreq + 20],
    #         # specylim=[0, 4e-23],
    #         specyscale="linear",  # 'log', 'linear'
    #         # showstd=False,
    #         showplt_opt=showplt_opt,
    #         return_opt=True,
    #     )
    #     return specxaxis, spectrum, specxunit, specyunit

    def setALP_Field(
        self,
        method: str,  # 'inverse-FFT' 'time-interfer'
        timeStamp_s: np.ndarray,
        simuRate_Hz: float,
        duration_s: float,
        B_a_rms_T: float,  # amplitude of the pseudo-magnetic field in (T)
        nu_a_rot_Hz: float,  # axion effective frequency in RCF
        use_stoch: bool,
        # direction: np.ndarray,  #  = np.array([1, 0, 0])
        RCF_freq_Hz: float,
        rand_seed: int = None,
        makePlot: bool = False,
        verbose: bool = False,
    ):
        """
        generate a pseudo-magnetic field (ALP field gradient)
        """
        timeStep = np.abs(timeStamp_s[1] - timeStamp_s[0])
        timeLen = len(timeStamp_s)
        self.nu_Hz = nu_a_rot_Hz

        # if verbose:
        #     # check input
        #     print("timeStamp_s.shape =", timeStamp_s.shape)
        #     print("np.mean(timeStamp_s) =", np.mean(timeStamp_s))
        #     print("timeLen =", timeLen)
        #     print("self.nu_Hz  =", self.nu_Hz)
        #     print("timeStep =", timeStep)
        #     print("nu_a_rot_Hz =", nu_a_rot_Hz)
        #     # print(" =", )
        #     # print(" =", )
        #     # print(" =", )

        def setALP_Field_timeIntf():
            """
            generate Bx, By, dBxdt, dBydt
            """
            frequencies = np.linspace(
                -0.5 / timeStep, 0.5 / timeStep, num=timeLen, endpoint=True
            )
            lineshape = axion_lineshape(
                v_0=220e3,
                v_lab=233e3,
                nu_a=nu_a_rot_Hz + RCF_freq_Hz,
                nu=frequencies + RCF_freq_Hz,
                case="grad_perp",
                alpha=0.0,
            )

            rvs_amp = expon.rvs(loc=0.0, scale=1.0, size=timeLen)
            # rvs_amp = 1.0
            rvs_phase = np.exp(1j * uniform.rvs(loc=0, scale=2 * np.pi, size=timeLen))
            # rvs_phase = 1.0

            if use_stoch:
                ax_sq_lineshape = lineshape * rvs_amp
            else:
                ax_sq_lineshape = lineshape
            ax_lineshape = np.sqrt(ax_sq_lineshape)
            # check lineshape sanity
            # for arr in [lineshape, ax_lineshape]:
            #     has_nan = np.isnan(arr).any()  # Check for NaN
            #     has_inf = np.isinf(arr).any()  # Check for Inf

            #     print(f"Contains NaN: {has_nan}")  # Output: True
            #     print(f"Contains Inf: {has_inf}")  # Output: True

            # inverse FFT method
            # ax_FFT = np.sqrt(stoch_a_sq_lineshape) * rvs_phase
            # Ba_t = np.fft.ifft(ax_FFT)
            # Bx = np.outer(np.real(Ba_t), np.array([1, 0, 0]))
            # By = np.outer(np.imag(Ba_t), np.array([1, 0, 0]))
            # self.B_vec = Bx + By

            # Find the index of the first non-zero element
            # nonzero_indices = np.nonzero(ax_lineshape)[0]
            # first_nonzero_index = (
            #     nonzero_indices[0] if nonzero_indices.size > 0 else None
            # )
            positive_indices = np.where(ax_lineshape > 0)[0]
            if positive_indices.size > 0:
                first_positive_index = positive_indices[0]
            else:
                first_positive_index = 0

            Bx_amp = np.zeros(timeLen)
            By_amp = np.zeros(timeLen)
            dBxdt_amp = np.zeros(timeLen)
            dBydt_amp = np.zeros(timeLen)

            if use_stoch:
                init_phase = uniform.rvs(loc=0, scale=2 * np.pi, size=timeLen)
            else:
                init_phase = 0.0 * np.ones(timeLen)

            for i in np.arange(first_positive_index, timeLen, dtype=int):
                nu_rot = frequencies[i]
                ax_amp = ax_lineshape[i]
                #  phase
                # init_phase_0 = init_phase[i]
                # By_init_phase = uniform.rvs(loc=0, scale=2 * np.pi, size=1)
                # # fixed phase
                # Bx_init_phase = 0
                # By_init_phase = 0

                # Bx
                Bx_amp += (
                    0.5
                    * B_a_rms_T
                    * ax_amp
                    * np.cos(2 * np.pi * nu_rot * timeStamp_s + init_phase[i])
                )
                # By
                By_amp += (
                    0.5
                    * B_a_rms_T
                    * ax_amp
                    * np.sin(2 * np.pi * nu_rot * timeStamp_s + init_phase[i])
                )
                # dBx / dt
                dBxdt_amp += (
                    0.5
                    * B_a_rms_T
                    * ax_amp
                    * (2 * np.pi * nu_rot)
                    * np.cos(2 * np.pi * nu_rot * timeStamp_s + init_phase[i])
                )
                # dBy / dt
                dBydt_amp += (
                    0.5
                    * B_a_rms_T
                    * ax_amp
                    * (-2 * np.pi * nu_rot)
                    * np.sin(2 * np.pi * nu_rot * timeStamp_s + init_phase[i])
                )

            Bx = np.outer(Bx_amp, np.array([1, 0, 0]))
            By = np.outer(By_amp, np.array([0, 1, 0]))
            dBxdt = np.outer(dBxdt_amp, np.array([1, 0, 0]))
            dBydt = np.outer(dBydt_amp, np.array([0, 1, 0]))

            self.B_vec = Bx + By
            self.dBdt_vec = dBxdt + dBydt

            if makePlot:
                pass
                # self.B_Stream = Signal(
                #     name="ALP field gradient",
                #     device="Simulation",
                #     device_id="Simulation",
                #     filelist=[],
                #     verbose=True,
                # )
                # self.B_Stream.attenuation = 0
                # self.B_Stream.filterstatus = "off"
                # self.B_Stream.DTRC_TC = 0.0
                # self.B_Stream.DTRC_order = 0
                # self.B_Stream.demodfreq = RCF_freq_Hz
                # saveintv = 1
                # self.B_Stream.samprate = 1.0 / timeStep / saveintv

                # self.B_Stream.dataX = 1 * self.B_vec[0:-1:saveintv, 0]  # * \
                # # np.cos(2 * np.pi * self.nu_rot * self.timestamp[0:-1:saveintv])
                # self.B_Stream.dataY = 1 * self.B_vec[0:-1:saveintv, 1]

                # self.B_Stream.getNoPulsePSD(
                #     windowfunction="Hanning",
                #     # decayfactor=-10,
                #     chunksize=None,  # sec
                #     analysisrange=[0, -1],
                #     getstd=False,
                #     stddev_range=None,
                #     selectshots=[],
                #     verbose=False,
                # )
                # specxaxis, spectrum, specxunit, specyunit = self.B_Stream.getSpectrum(
                #     showtimedomain=True,
                #     showfit=True,
                #     showresidual=False,
                #     showlegend=True,  # !!!!!show or not to show legend
                #     spectype="PSD",  # in 'PSD', 'ASD', 'FLuxPSD', 'FluxASD'
                #     ampunit="V",
                #     specxunit="Hz",  # 'Hz' 'kHz' 'MHz' 'GHz' 'ppm' 'ppb'
                #     specxlim=[
                #         nu_a_rot_Hz + RCF_freq_Hz - 5,
                #         nu_a_rot_Hz + RCF_freq_Hz + 20,
                #     ],
                #     # specylim=[0, 4e-23],
                #     specyscale="linear",  # 'log', 'linear'
                #     showstd=False,
                #     showplt_opt=True,
                #     return_opt=True,
                # )

        def setALP_Field_invFFT():
            """
            generate Bx, By, dBxdt, dBydt
            """
            frequencies = np.linspace(
                -0.5 / timeStep, 0.5 / timeStep, num=timeLen, endpoint=True
            )
            if verbose:
                print("np.mean(frequencies) =", np.mean(frequencies))
                print("np.std(frequencies) =", np.std(frequencies))
                print("RBW =", abs(frequencies[1] - frequencies[0]))

            if verbose:
                tic = time.perf_counter()
            lineshape = axion_lineshape(
                v_0=220e3,
                v_lab=233e3,
                nu_a=nu_a_rot_Hz + RCF_freq_Hz,
                nu=frequencies + RCF_freq_Hz,
                case="grad_perp",
                alpha=0.0,
            )
            if verbose:
                toc = time.perf_counter()
                timeConsumption = toc - tic
                print(f"axion_lineshape time consumption = {timeConsumption:.3e} s")

            # if verbose:
            #     print("np.mean(lineshape) =", np.mean(lineshape))
            #     print("np.std(lineshape) =", np.std(lineshape))

            if verbose:
                tic = time.perf_counter()
            # rng = (
            #     np.random.default_rng(seed=rand_seed) if rand_seed is not None else None
            # )
            np.random.seed(rand_seed)  # set the seed globally for this iteration

            if use_stoch:
                amp_freq = np.random.exponential(scale=1.0, size=timeLen)
                phase_freq = np.exp(1j * 2 * np.pi * np.random.rand(timeLen))
            else:
                amp_freq = 1.0
                phase_freq = np.exp(1j * 2 * np.pi * np.random.rand(timeLen))
            if verbose:
                toc = time.perf_counter()
                timeConsumption = toc - tic
                print(f"rng time consumption = {timeConsumption:.3e} s")

            ax_lineshape = np.sqrt(lineshape * amp_freq)

            # inverse FFT method
            N = len(ax_lineshape)
            freq = np.fft.fftfreq(N, timeStep)

            # ifft_runtimes = []
            if verbose:
                tic = time.perf_counter()

            # ttic = time.perf_counter()
            ax_FFT = (
                ax_lineshape
                * phase_freq
                * B_a_rms_T
                * simuRate_Hz
                * np.sqrt(duration_s)
            )
            # ttoc = time.perf_counter()
            # ifft_runtimes.append(ttoc - ttic)

            # ttic = time.perf_counter()
            ax_FFT_0_pos_neg = np.fft.fftshift(ax_FFT)
            # ttoc = time.perf_counter()
            # ifft_runtimes.append(ttoc - ttic)

            # ttic = time.perf_counter()
            # Ba_t = np.fft.ifft(ax_FFT_0_pos_neg)
            Ba_t = ifft(ax_FFT_0_pos_neg)  # , axis=1 batch IFFT along time axis
            # ttoc = time.perf_counter()
            # ifft_runtimes.append(ttoc - ttic)

            # ttic = time.perf_counter()
            dBadt_FFT = 1j * 2 * np.pi * freq * ax_FFT_0_pos_neg
            # ttoc = time.perf_counter()
            # ifft_runtimes.append(ttoc - ttic)

            # ttic = time.perf_counter()
            # dBadt = np.fft.ifft(dBadt_FFT)
            dBadt = ifft(dBadt_FFT)
            # ttoc = time.perf_counter()
            # ifft_runtimes.append(ttoc - ttic)

            if verbose:
                toc = time.perf_counter()
                timeConsumption = toc - tic
                print(f"ifft total time consumption = {timeConsumption:.3e} s")
                # for i, runtime in enumerate(ifft_runtimes):
                # print("individual runtimes =", ifft_runtimes)

            if verbose:
                tic = time.perf_counter()
            Bx_amp, By_amp = np.real(Ba_t), np.imag(Ba_t)
            dBxdt_amp, dBydt_amp = np.real(dBadt), np.imag(dBadt)

            Bx = np.outer(Bx_amp, np.array([1, 0, 0]))
            By = np.outer(By_amp, np.array([0, 1, 0]))
            dBxdt = np.outer(dBxdt_amp, np.array([1, 0, 0]))
            dBydt = np.outer(dBydt_amp, np.array([0, 1, 0]))

            self.B_vec = Bx + By
            self.dBdt_vec = dBxdt + dBydt
            if verbose:
                toc = time.perf_counter()
                timeConsumption = toc - tic
                print(f"array-asignment time consumption = {timeConsumption:.3e} s")

            if makePlot:
                # check(self.B_vec[::1, 0])
                specxaxis, spectrum0, specxunit, specyunit = self.showTSandPSD(
                    dataX=self.B_vec[::1, 0],
                    dataY=self.B_vec[::1, 1],
                    demodfreq=RCF_freq_Hz,
                    samprate=1.0 / timeStep,
                    showplt_opt=False,
                )
                specxaxis, spectrum1, specxunit, specyunit = self.showTSandPSD(
                    dataX=self.dBdt_vec[::1, 0],
                    dataY=self.dBdt_vec[::1, 1],
                    demodfreq=RCF_freq_Hz,
                    samprate=1.0 / timeStep,
                    showplt_opt=False,
                )
                fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
                gs = gridspec.GridSpec(
                    nrows=2, ncols=2
                )  # create grid for multiple figures
                ax_x = fig.add_subplot(gs[0, 0])
                ax_y = fig.add_subplot(gs[1, 0])
                axPSD = fig.add_subplot(gs[:, 1])

                ax_x.plot(self.B_vec[:, 0] / np.amax(self.B_vec), label="Bx")
                ax_x.plot(self.dBdt_vec[:, 0] / np.amax(self.dBdt_vec), label="dBxdt")

                ax_y.plot(self.B_vec[:, 1] / np.amax(self.B_vec), label="By")
                ax_y.plot(self.dBdt_vec[:, 1] / np.amax(self.dBdt_vec), label="dBydt")

                axPSD.plot(
                    specxaxis,
                    spectrum0 / np.amax(spectrum0),
                    label="ALP field gradient PSD",
                )
                axPSD.plot(
                    specxaxis,
                    spectrum1 / np.amax(spectrum1),
                    label="dBa_dt PSD",
                    linestyle="--",
                )
                axPSD.set_xlabel("")
                axPSD.set_ylabel("")
                # ax00.set_xscale('log')
                # ax00.set_yscale("log")
                # #############################################################################
                ax_x.legend()
                ax_y.legend()
                axPSD.legend()
                # #############################################################################
                fig.suptitle("", wrap=True)
                plt.tight_layout()
                plt.show()

        if method == "inverse-FFT":
            setALP_Field_invFFT()
        elif method == "time-interfer":
            setALP_Field_timeIntf()
        else:
            raise ValueError("method not found")

        # print(f"{self.setALP_Field.__name__}: setALP_Field")

    def setALPfields(
        self,
        method: str,  # 'inverse-FFT'
        # timeStamp_s: np.ndarray,
        timeStep_s: float,
        timeLen: int,
        simuRate_Hz: float,
        duration_s: float,
        B_a_rms_T: float,  # amplitude of the pseudo-magnetic field in (T)
        nu_a_rot_Hz: float,  # axion effective frequency in RCF
        use_stoch: bool,
        # direction: np.ndarray,  #  = np.array([1, 0, 0])
        RCF_freq_Hz: float,
        numFields: int,
        rand_seed: int = None,
        makePlot: bool = False,
        verbose: bool = False,
    ):
        """
        generate a pseudo-magnetic field (ALP field gradient)
        """
        # timeStep = np.abs(timeStamp_s[1] - timeStamp_s[0])
        # timeLen = len(timeStamp_s)
        self.nu_Hz = nu_a_rot_Hz
        self.numFields = numFields
        numSteps = timeLen - 1

        def setALP_Field_invFFT():
            """
            generate Bx, By, dBxdt, dBydt
            """
            frequencies = np.linspace(
                -0.5 / timeStep_s, 0.5 / timeStep_s, num=numSteps, endpoint=True
            )
            if verbose:
                check(timeStep_s)
                check(numSteps)

            tic = time.perf_counter()
            avg_lineshape = axion_lineshape(
                v_0=220e3,
                v_lab=233e3,
                nu_a=nu_a_rot_Hz + RCF_freq_Hz,
                nu=frequencies + RCF_freq_Hz,
                case="grad_perp",
                alpha=0.0,
            )
            toc = time.perf_counter()
            timeConsumption = toc - tic
            if verbose:
                print(f"axion_lineshape time consumption = {timeConsumption:.3e} s")

            tic = time.perf_counter()
            rng = (
                np.random.default_rng(seed=rand_seed)
                if rand_seed is not None
                else np.random.default_rng()
            )

            if use_stoch:
                # amp_freq: shape = (numFields, numSteps), reproducible
                amp_freq = rng.exponential(scale=1.0, size=(numFields, numSteps))
                # phase_freq:
                phase_freq = np.exp(1j * 2 * np.pi * rng.random((numFields, numSteps)))
            else:
                # amp_freq scalar or 1D array, depending on usage
                amp_freq = np.ones((numFields, numSteps), dtype=np.float64)
                # phase_freq: reproducible random phases
                phase_freq = np.exp(1j * 2 * np.pi * rng.random((numFields, numSteps)))
            toc = time.perf_counter()
            timeConsumption = toc - tic
            if verbose:
                print(f"rng time consumption = {timeConsumption:.3e} s")

            # check(amp_freq.shape)  # shape = (numFields, numSteps)
            # check(phase_freq.shape)  # shape = (numFields, numSteps)
            # TODO optimize when only a small fraction of lineshapes is non-zero by using less lengths for amp and phase
            rand_sqrt_lineshapes = (
                np.sqrt(avg_lineshape * amp_freq) * phase_freq
            )  # shape = (numFields, numSteps)
            # check(ax_lineshape.shape)
            freq = np.fft.fftfreq(numSteps, timeStep_s)  # shape = (numSteps)
            if makePlot:
                fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
                gs = gridspec.GridSpec(
                    nrows=1, ncols=1
                )  # create grid for multiple figures
                axPSD = fig.add_subplot(gs[0, 0])

                axPSD.plot(
                    frequencies,
                    avg_lineshape,
                    label="Average ALP-field gradient PSD",
                    color="tab:orange",
                    linestyle="--",
                    zorder=3,
                )
                axPSD.scatter(
                    frequencies,
                    avg_lineshape,
                    # label="Average ALP-field gradient PSD",
                    # linestyle="--",zorder=3,
                )
                if use_stoch:
                    rand_lineshapes = np.abs(rand_sqrt_lineshapes) ** 2
                    axPSD.errorbar(
                        x=frequencies,
                        y=rand_lineshapes.mean(axis=0),
                        yerr=rand_lineshapes.std(axis=0),
                        label="Stochastic ALP-field gradient PSD",
                        linestyle="-",
                    )
                axPSD.set_xlabel(f"Frequency - {RCF_freq_Hz:.0g} (Hz)")
                axPSD.set_ylabel("")
                axPSD.legend()
                # #############################################################################
                fig.suptitle("", wrap=True)
                plt.tight_layout()
                plt.show()

            # ifft_runtimes = []
            if verbose:
                tic = time.perf_counter()

            # ttic = time.perf_counter()
            ax_FFT: np.ndarray = (
                B_a_rms_T * simuRate_Hz * np.sqrt(duration_s) * rand_sqrt_lineshapes
            )
            # ttoc = time.perf_counter()
            # ifft_runtimes.append(ttoc - ttic)

            # ttic = time.perf_counter()
            ax_FFT_0_pos_neg: np.ndarray = np.fft.fftshift(ax_FFT, axes=1)
            # ttoc = time.perf_counter()
            # ifft_runtimes.append(ttoc - ttic)

            # ttic = time.perf_counter()
            # Ba_t = np.fft.ifft(ax_FFT_0_pos_neg, axis=1)
            Ba_t: np.ndarray = ifft(
                ax_FFT_0_pos_neg, axis=1
            )  # batch IFFT along time axis
            # ttoc = time.perf_counter()
            # ifft_runtimes.append(ttoc - ttic)

            # ttic = time.perf_counter()
            dBadt_FFT: np.ndarray = 1j * 2 * np.pi * freq * ax_FFT_0_pos_neg
            # ttoc = time.perf_counter()
            # ifft_runtimes.append(ttoc - ttic)

            # ttic = time.perf_counter()
            # dBadt = np.fft.ifft(dBadt_FFT, axis=1)
            dBadt = ifft(dBadt_FFT, axis=1)
            # ttoc = time.perf_counter()
            # ifft_runtimes.append(ttoc - ttic)

            if verbose:
                toc = time.perf_counter()
                timeConsumption = toc - tic
                print(f"ifft total time consumption = {timeConsumption:.3e} s")
                # for i, runtime in enumerate(ifft_runtimes):
                # print("individual runtimes =", ifft_runtimes)

            if verbose:
                tic = time.perf_counter()

            self.B_vec = np.zeros((numFields, numSteps, 3))
            self.dBdt_vec = np.zeros((numFields, numSteps, 3))

            self.B_vec[:, :, 0] = Ba_t.real
            self.B_vec[:, :, 1] = Ba_t.imag
            self.dBdt_vec[:, :, 0] = dBadt.real
            self.dBdt_vec[:, :, 1] = dBadt.imag

            # # Stack real and imaginary parts along the last axis
            # self.B_vec = np.stack((Ba_t.real, Ba_t.imag, np.zeros_like(Ba_t.real)), axis=2)
            # self.dBdt_vec = np.stack((dBadt.real, dBadt.imag, np.zeros_like(dBadt.real)), axis=2)

            if verbose:
                toc = time.perf_counter()
                timeConsumption = toc - tic
                print(f"array-asignment time consumption = {timeConsumption:.3e} s")

            # check(Bx_amp.shape)
            # check(By_amp.shape)
            # check(dBxdt_amp.shape)
            # check(dBydt_amp.shape)

            # check(Bx.shape)
            # check(By.shape)
            # check(dBxdt.shape)
            # check(dBydt.shape)
            # check(self.B_vec.shape)
            # check(self.dBdt_vec.shape)

        if method == "inverse-FFT":
            setALP_Field_invFFT()
        # elif method == "time-interfer":
        #     setALP_Field_timeIntf()
        else:
            raise ValueError("method not found")

    def plotField(self, demodfreq, samprate, showplt_opt):
        specxaxis, spectrum, specxunit, specyunit = self.showTSandPSD(
            dataX=self.B_vec[:, 0],
            dataY=self.B_vec[:, 1],
            demodfreq=demodfreq,
            samprate=samprate,
            showplt_opt=showplt_opt,
        )
        return specxaxis, spectrum, specxunit, specyunit


class Simulations:
    # Axion NMR simulations
    def __init__(
        self,
        name: str = "NMR simulationS",
        # pool: Optional[list[Simulation]] = None,
        # samples: Optional[list[Sample]] = None,  # class Sample
        # axions: Optional[list[AxionWind]] = None,  # refer to class Station
        # stations: [list[Station]] = None,  # refer to class Station
        all_params: list = None,
        # init_M: Optional[PhysicalQuantity] = PhysicalQuantity(
        #     1.0, ""
        # ),  # initial magnetization vector amplitude
        # init_M_theta: Optional[PhysicalQuantity] = PhysicalQuantity(0, "rad"),
        # init_M_phi: Optional[PhysicalQuantity] = PhysicalQuantity(0, "rad"),
        verbose=True,
    ):
        """ """
        # self.axions = axions
        # self. =
        # self.samples = samples
        # self.magnets = magnets
        # self.excFields = excFields
        # self.init_M = init_M
        # self.init_M_theta = init_M_theta
        # self. =
        # self. =
        # self. =
        # self. =
        # self. =

        # magnets: Optional[list[Magnet]] = None,
        # excFields: Optional[list[MagField]] = None,  # class MagField

        # RCF_freqs: Optional[list[PhysicalQuantity]] = None,
        # rates: Optional[list[PhysicalQuantity]] = None,
        # durations: Optional[list[PhysicalQuantity]] = None,
        self.pool: list[SimuEntry] = []
        self.all_params: list[SimuParams] = all_params

    def setup(self, verbose: bool = False):
        est_runtime = 0.0
        est_setFields_s = 0.0
        est_trjry_s = 0.0
        # initialization
        for params in self.all_params:
            #
            sample = params["sample"]  # set the sample
            axion = params["axion"]
            magnet = params["magnet"]
            excField = params["excField"]  # initailize excitation field
            numFields = params["numFields"]
            init_M = params["init_M"]
            init_M_theta = params["init_M_theta"]
            init_M_phi = params["init_M_phi"]
            rate = params["rate"]
            duration = params["duration"]

            RCF_freq: PhysicalQuantity = axion.nu_a_eff

            use_stoch = True

            # initialize simulation
            simu = Simulation(
                name="simulation",
                axion=axion,
                sample=sample,
                magnet=magnet,
                excField=excField,
                init_M=init_M,
                init_M_theta=init_M_theta,
                init_M_phi=init_M_phi,
                RCF_freq=RCF_freq,
                rate=rate,
                duration=duration,
                verbose=verbose,
            )
            self.pool.append({"simu": simu, "params": params})

            if verbose:
                print("", flush=True)
                for key, pq in params["key_info"].items():
                    print(key, " =", pq, flush=True)
                # print("Axion Compton frequency =", axion.nu_a)
                # print(
                #     f"simulation duration = {duration.value_in('s'):e} (s).", flush=True
                # )
                print("simu.magnet.numPt =", simu.magnet.numPt, flush=True)
                # print("simuRate =", simuRate, flush=True)
                print(f"Number of fields = {numFields}", flush=True)

            # estimate setting fields time
            t_setFields_s = T_SETFIELD_S * numFields * (simu.numSteps + 1)
            # estimate simulation time
            t_trjry_s = T_SIMUSTEP_S * simu.numSteps * simu.magnet.numPt * numFields
            est_setFields_s += t_setFields_s
            est_trjry_s += t_trjry_s
            est_runtime += t_setFields_s + t_trjry_s
            if verbose:
                # print(
                #     "[setALPfields] time consumption estimated =",
                #     t_setFields_s / 60,
                #     "min",
                # )
                # print(
                #     "[generateTrajectories] time consumption estimated =",
                #     t_trjry_s / 60,
                #     "min",
                # )
                print(
                    "Estimated step runtime = "
                    + f"{(t_setFields_s + t_trjry_s) / 60:.2g} min",
                    flush=True,
                )
        return est_runtime, est_setFields_s, est_trjry_s

    def run(self, autoStart: bool = True, verbose: bool = False):
        # est_runtime = 0.0
        # est_setFields_s = 0.0
        # est_trjry_s = 0.0

        actu_runtime = 0.0
        actu_setFields_s = 0.0
        actu_trjry_s = 0.0
        # initialization
        est_runtime, est_setFields_s, est_trjry_s = self.setup(verbose=verbose)
        # for params in self.all_params:
        #     #
        #     sample = params["sample"]  # set the sample
        #     axion = params["axion"]
        #     magnet = params["magnet"]
        #     excField = params["excField"]  # initailize excitation field
        #     numFields = params["numFields"]
        #     init_M = params["init_M"]
        #     init_M_theta = params["init_M_theta"]
        #     init_M_phi = params["init_M_phi"]
        #     rate= params["rate"]
        #     duration= params["duration"]

        #     RCF_freq: PhysicalQuantity = axion.nu_a_eff

        #     use_stoch = True

        #     # initialize simulation
        #     simu = Simulation(
        #         name="simulation",
        #         axion=axion,
        #         sample=sample,
        #         magnet=magnet,
        #         excField=excField,
        #         init_M=init_M,
        #         init_M_theta=init_M_theta,
        #         init_M_phi=init_M_phi,
        #         RCF_freq=RCF_freq,
        #         rate=rate,
        #         duration=duration,
        #         verbose=verbose,
        #     )
        #     self.pool.append({"simu": simu, "params": params})

        #     if verbose:
        #         print("", flush=True)
        #         for key, pq in params["key_info"].items():
        #             print(key, " =", pq, flush=True)
        #         # print("Axion Compton frequency =", axion.nu_a)
        #         # print(
        #         #     f"simulation duration = {duration.value_in('s'):e} (s).", flush=True
        #         # )
        #         print("simu.magnet.numPt =", simu.magnet.numPt, flush=True)
        #         # print("simuRate =", simuRate, flush=True)
        #         print(f"Number of fields = {numFields}", flush=True)

        #     # estimate setting fields time
        #     t_setFields_s = T_SETFIELD_S * numFields * (simu.numSteps + 1)
        #     # estimate simulation time
        #     t_trjry_s = T_SIMUSTEP_S * simu.numSteps * simu.magnet.numPt * numFields
        #     est_setFields_s += t_setFields_s
        #     est_trjry_s += t_trjry_s
        #     est_runtime += t_setFields_s + t_trjry_s
        #     if verbose:
        #         # print(
        #         #     "[setALPfields] time consumption estimated =",
        #         #     t_setFields_s / 60,
        #         #     "min",
        #         # )
        #         # print(
        #         #     "[generateTrajectories] time consumption estimated =",
        #         #     t_trjry_s / 60,
        #         #     "min",
        #         # )
        #         print(
        #             "Estimated step runtime = "
        #             + f"{(t_setFields_s + t_trjry_s) / 60:.2g} min",
        #             flush=True,
        #         )

        if not autoStart:
            print(
                "# ---------------------------------------------------- #", flush=True
            )
            print(
                f"Estimated setFields time = {est_setFields_s / 60.0:.3g} min",
                flush=True,
            )
            print(f"Estimated trjry time = {est_trjry_s / 60.0:.3g} min", flush=True)
            print(f"Estimated runtime = {est_runtime / 60.0:.3g} min", flush=True)
            answer = input("Continue? (y/n): ").strip().lower()
            if answer == "y" or answer == "Y":
                print("Proceeding...", flush=True)
                print(
                    "# ---------------------------------------------------- #",
                    flush=True,
                )
            else:
                print("Stopped.", flush=True)
                print(
                    "# ---------------------------------------------------- #",
                    flush=True,
                )
                exit()

        # run simulations
        for i, params in enumerate(self.all_params):
            simu: Simulation = self.pool[i]["simu"]
            # set fields
            tic = time.perf_counter()
            simu.excField.setALPfields(
                method="inverse-FFT",
                # timeStamp_s=simu.timeStamp_s,
                timeStep_s=simu.timeStep_s,
                timeLen=simu.timeLen,
                simuRate_Hz=simu.rate_Hz,
                duration_s=simu.duration_s,
                B_a_rms_T=params["B_a_rms"].value_in(
                    "T"
                ),  # rms amplitude of the pseudo-magnetic field in [T]
                nu_a_rot_Hz=params["axion"].nu_a.value_in("Hz")
                - simu.RCF_freq_Hz,  # frequency in the rotating frame
                use_stoch=True,
                RCF_freq_Hz=simu.RCF_freq_Hz,
                numFields=params["numFields"],
                rand_seed=params["rand_seed"],
                makePlot=False,
                verbose=False,
            )
            toc = time.perf_counter()
            timeConsumption = toc - tic
            actu_setFields_s += timeConsumption
            if verbose:
                print("", flush=True)
                for key, pq in params["key_info"].items():
                    print(key, "=", pq, flush=True)
                print(
                    f"[setALPfields] time consumption = {timeConsumption:.6f} s = {timeConsumption/60:.1g} min",
                    flush=True,
                )
                print(
                    f"[setALPfields] individual step time consumption = {timeConsumption/(simu.numSteps+1)/simu.excField.numFields:.3e} s",
                    flush=True,
                )
            # ------------------------------------
            # print("simu.numSteps =", simu.numSteps, flush=True)
            # print("len(simu.excField.B_vec) =", (simu.excField.B_vec.shape), flush=True)

            simu.excType = "ALP"

            # ------------------------------------
            tic = time.perf_counter()
            simu.generateTrajectories(verbose=False)
            toc = time.perf_counter()
            timeConsumption = toc - tic
            actu_trjry_s += timeConsumption
            if verbose:
                print(
                    f"[generateTrajectories] time consumption = {timeConsumption:.2g} s = {timeConsumption/60:.1g} min",
                    flush=True,
                )
                print(
                    f"[generateTrajectories] individual step time consumption = {timeConsumption/simu.numSteps/simu.magnet.numPt/simu.excField.numFields:.2e} s",
                    flush=True,
                )
            # ------------------------------------
            simu.keepMeanStd()

        if verbose:
            actu_runtime = actu_setFields_s + actu_trjry_s
            print(
                "# ---------------------------------------------------- #", flush=True
            )
            print(f"Total runtime = {actu_runtime / 60.0:.3g} min", flush=True)
            print(
                "# ---------------------------------------------------- #", flush=True
            )

    def saveToH5(self, dir: str = None, verbose: bool = False):
        """ """
        if dir[-1] != "\\":
            dir += "\\"
        ymd_hms = giveDateAndTime()

        for time_str, pool_item in self.pool.items():
            data_file_name = dir + ymd_hms
            data_file_name += "-simulation"
            for key, pq in pool_item["params"]["key_info"].items():
                data_file_name += "-"
                data_file_name += key
                data_file_name += f"={pq.value:g}" + pq.unit
            pool_item["simu"].saveToH5(path=data_file_name)

    def loadFromH5(self, paths: list[str] = None, verbose: bool = False):
        """
        In progress
        Docstring for loadFromH5

        :param self: Description
        :param paths: Description
        :type paths: list[str]
        :param verbose: Description
        :type verbose: bool
        """

        for i, path in enumerate(paths):
            newSimu = Simulation()
            newSimu.loadFromH5(path=path)
            # self.pool.append(newSimu)
            name = f"simu_{i:d}"
            self.pool[name] = newSimu
        # self.sortByAttrs("RCF_freq_Hz")

    def saveToPkl(
        self,
        dir: str,
        fname: str | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ):
        """
        Save this instance to a pickle file.
        """
        if dir is None:
            raise ValueError("dir must not be None")

        if fname is None:
            fname = giveDateAndTime()

        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, f"{fname}.pkl")

        while os.path.exists(path) and not overwrite:
            print(f"File already exists: {path}")
            new = input(
                "Enter a new filename (without .pkl) or press Enter to overwrite: "
            ).strip()
            if new == "":
                break
            fname = new
            path = os.path.join(dir, f"{fname}.pkl")

        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        if verbose:
            print(f"Saved object to {path}")

        # for time_str, pool_item in self.pool.items():
        #     data_file_name = dir + ymd_hms
        #     data_file_name += "-simulation"
        #     for key, pq in pool_item["params"]["key_info"].items():
        #         data_file_name += "-"
        #         data_file_name += key
        #         data_file_name += f"={pq.value:g}" + pq.unit
        #     pool_item["simu"].saveToH5(path=data_file_name)

    @classmethod
    def loadFromPkl(cls, path: str, verbose: bool = False):
        """
        Load an instance of this class from a pickle file.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Pickle file not found: {path}")

        with open(path, "rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, cls):
            raise TypeError(f"Pickle contains {type(obj)}, expected {cls}")

        if verbose:
            print(f"Loaded object from {path}")

        return obj

    def sortByAttrs(self, *attrs, reverse: bool = False, verbose=False):
        if not attrs:
            raise ValueError("At least one attribute must be specified")

        # check if attrs are valid
        for attr in attrs:
            for simu in self.pool.values():
                if not hasattr(simu, attr):
                    raise AttributeError(f"Simulation has no attribute '{attr}'")
        # sort the map
        self.pool = dict(
            sorted(
                self.pool.items(),
                key=lambda item: tuple(getattr(item[1], attr) for attr in attrs),
                reverse=reverse,
            )
        )

        if verbose:
            for name, simu in self.pool.items():
                print("")
                print(simu.axion.nu_a)
                print(simu.magnet.FWHM)


class Simulation(PhysicalObject):
    # NMR simulation in the rotating frame
    def __init__(
        self,
        name: str = "NMR simulation",
        axion: Optional[AxionWind] = None,
        sample: Optional[Sample] = None,
        magnet: Optional[Magnet] = None,
        excField: Optional[MagField] = None,
        station: Optional[Station] = None,
        init_M: Optional[PhysicalQuantity] = PhysicalQuantity(
            1.0, ""
        ),  # initial magnetization vector amplitude
        init_M_theta: Optional[PhysicalQuantity] = PhysicalQuantity(0, "rad"),
        init_M_phi: Optional[PhysicalQuantity] = PhysicalQuantity(0, "rad"),
        RCF_freq: Optional[PhysicalQuantity] = None,
        rate: Optional[PhysicalQuantity] = None,  # simulation rate
        duration: Optional[PhysicalQuantity] = None,  # simulation duration
        verbose: bool = False,
    ):
        """
        initialize NMR simulation
        The simulation uses rotating coordinate frame (RCF).

        # List of rates / frequencies in the simulation

        ## Fixed parameters
            1 / (T1, T2, T2star or Tdelta) : Relaxation rates
            1 / tau_a : decoherence rate
            nuL : Larmor frequency
            FWHM of Larmor frequencies

        ## Tunable parameters
            rate : simulation rate
            RCF_Freq : rotating frequency of RCF
            1 / duration : inverse of the simulation duration / resolution bandwidth (RBW)
            numPt in Magnet : the number of spin packets

        ## how to choose tunable parameter values
            1. use T1 >> duration, otherwise choose simu.rate >> 1 / T1
            1. simu.rate >> 1 / (T2, T2star, Tdelta or tau_a)
            1. simu.rate >> nuL * (1 +/- FWHM)
            1. simu.rate >> 1 / duration
            1. RBW <= 2 pi * Larmor frequency step


        """
        super().__init__()
        self.physicalQuantities = {
            "RCF_freq": "Hz",
            "rate": "Hz",
            "duration": "s",
            "init_M": "",
            "init_M_theta": "rad",
            "init_M_phi": "rad",
        }

        self.name = name

        if None in [sample, magnet, excField]:
            return

        # Instances
        self.sample = sample
        self.magnet = magnet
        self.axion = axion
        self.station = station
        self.excField = excField

        #
        self.gamma_HzToT = self.sample.gamma.value_in("Hz/T")

        self.init_M = init_M
        self.init_M_theta = init_M_theta
        self.init_M_phi = init_M_phi

        self.init_M_amp = init_M.value_in("")
        self.init_M_theta_rad = init_M_theta.value_in("rad")
        self.init_M_phi_rad = init_M_phi.value_in("rad")

        self.B0z_T = magnet.B0.value_in("T")

        # set rotating frame frequency
        if RCF_freq is None:
            self.RCF_freq = self.sample.gamma / (2 * np.pi) * self.magnet.B0
        else:
            self.RCF_freq = RCF_freq
        self.RCF_freq_Hz = self.RCF_freq.value_in("Hz")

        self.nuL_Hz = (
            abs(self.gamma_HzToT * self.B0z_T / (2 * np.pi)) - self.RCF_freq_Hz
        )  # nuL_Hz is the Larmor frequency of the magnetization in the rotating frame

        self.T2_s = sample.T2.value_in("s")
        self.T1_s = sample.T1.value_in("s")

        self.trjry = None

        # ---------------------------- set duration ----------------------------#
        FWHM_Hz = self.magnet.FWHM_T * sample.gamma.value_in("Hz/T") / (2 * np.pi)
        # find Tdelta
        self.Tdelta_s = 1 / (np.pi * FWHM_Hz)
        self.T2star_s = 1 / (1 / self.Tdelta_s + 1 / sample.T2.value_in("s"))

        if duration is None:
            if self.axion is not None:
                self.duration_s = np.amin(
                    [3.2e3 * self.axion.tau_a_est.value_in("s"), 1.5e2 * self.T2star_s]
                )
            else:
                self.duration_s = 2e2 * self.T2star_s
            self.duration = PhysicalQuantity(self.duration_s, "s")
        else:
            self.duration = duration
            self.duration_s = self.duration.value_in("s")
        # ---------------------------------------------------------------------------- #

        # -----------------------------set magnet--------------------------------------- #
        # estimate the necessary data points for sampling the inhomogeneity
        numPt: float = (self.duration * 2 * magnet.B0_nW * sample.gamma).value_in("")
        if numPt <= 1:
            numPt = 1
        else:
            numPt += 11
        #
        # set the number of data points for the inhomogeneity (could be 1)
        magnet.setHomogeneity(
            numPt=numPt,
            # showplt=False,
            verbose=verbose,
        )
        # ---------------------------------------------------------------------------- #

        # ---------------------------- simulation rate ----------------------------#
        T2star_relaxation_rate_Hz = 1 / self.T2star_s
        if self.axion is not None:
            axion_decoherence_rate_Hz = 1 / self.axion.tau_a_est.value_in("s")
        else:
            axion_decoherence_rate_Hz = 0.0
        nuL_vals_Hz = (
            sample.gamma.value_in("Hz/T") / (2 * np.pi) * magnet.B_vals_T
            - self.RCF_freq_Hz
        )
        nuL_Hz_abs_max = np.amax(np.abs(nuL_vals_Hz))
        nu_Hz_range = np.amax(nuL_vals_Hz) - np.amin(nuL_vals_Hz)
        if rate is None:
            rate_Hz: float = np.amax(
                [
                    50.0 * nu_Hz_range,
                    50.0 * nuL_Hz_abs_max,
                    # 30.0 * nuL_Hz_abs_max * self.duration_s / (1e2 * self.T2star_s),
                    100.0 * T2star_relaxation_rate_Hz,
                    100.0 * axion_decoherence_rate_Hz,
                ]
            )
            rate = PhysicalQuantity(rate_Hz, "Hz")
        self.setRate(rate=rate)
        # -------------------------------------------------------------------------#

        # ----- check if parameter values are within reasonable range -----#
        assert self.numSteps >= 5, "number of steps < 5"
        # numPt >= 2 pi * Delta_nu * duration
        if (
            self.magnet.numPt
            < 2
            * self.magnet.nFWHM
            * self.magnet.FWHM_T
            * self.gamma_HzToT
            * self.duration_s
        ):
            print(
                f"{self.__init__.__name__} WARNING: magnet_det.numPt may be too few. "
            )

        # simulation rate must be 20 times larger than maximum Larmor frequency
        # from experience, simulation rate should be 20 times greater than the max. of signal frequency in the rotating frame
        if self.rate_Hz < 20 * nuL_Hz_abs_max:
            print(
                f"WARNING: the simulation rate ({self.rate_Hz:g} Hz) might be too small compared to signal frequency ({nuL_Hz_abs_max:g} Hz) . "
            )

        if self.T2_s > self.T1_s:
            print("WARNING: T2 is larger than T1")

        if self.rate_Hz <= 10 * (1.0 / self.T2_s):
            print(
                f"WARNING: the simulation rate ({self.rate_Hz:g} Hz) might be too small compared to T2 relaxation rate ({1.0 / self.T2_s:g} Hz) . "
            )
        if self.rate_Hz <= 1 / self.duration_s:
            print(
                f"WARNING: the simulation rate ({self.rate_Hz:g} Hz) might be too small so there are only {self.numSteps:d} step(s) in the duration of {self.duration_s:g} s. "
            )
        # ----- ----------------------------------------------------- -----#

    def setRate(self, rate: PhysicalQuantity):
        assert rate is not None, f"[{self.setRate.__name__}] rate is None"
        self.rate = rate
        self.rate_Hz = rate.value_in("Hz")
        self.timeStep_s = (
            1.0 / self.rate_Hz
        )  # the key parameter in setting simulation timing
        self.timeLen: int = int(np.ceil(self.duration_s * self.rate_Hz))
        self.numSteps: int = self.timeLen - 1
        if self.numSteps > 1e8:
            print(f"WARNING: Simulation.numSteps = {self.numSteps:.1e} > 1e8")
            # return
        # self.timeStamp_s = np.arange(start=0, stop=(self.timeLen) * self.timeStep_s, step=self.timeStep_s)

    def suggestRate(self, verbose: bool = False):
        # ----- check if parameter values are within reasonable range -----#

        # compute the maximum of (absolute) Larmor frequencies
        nuL_Hz_max = (
            self.gamma_HzToT / (2 * np.pi) * self.magnet.B_vals_T.max()
            - self.RCF_freq_Hz
        )
        nuL_Hz_min = (
            self.gamma_HzToT / (2 * np.pi) * self.magnet.B_vals_T.min()
            - self.RCF_freq_Hz
        )
        nuL_Hz_abs_max = max(abs(nuL_Hz_max), abs(nuL_Hz_min))

        # compute T2 relaxation rate
        T2Rate = 1.0 / self.T2_s

        # resolution bandwidth (RBW)
        RBW = 1 / self.duration_s

        if verbose:
            # check(np.amax(self.magnet_det.B_vals_T))
            # check(np.amin(self.magnet_det.B_vals_T))
            print(
                f"{self.suggestRate.__name__}: Larmor frequency range = ({nuL_Hz_min}, {nuL_Hz_max}) Hz"
            )
            print(
                f"{self.suggestRate.__name__}: the maximum of (absolute) Larmor frequencies  = {nuL_Hz_abs_max} Hz"
            )
            print(
                f"{self.suggestRate.__name__}: T2 relaxation rate = {nuL_Hz_abs_max} Hz"
            )
            print(f"{self.suggestRate.__name__}: resolution bandwidth (RBW) = {RBW} Hz")
        # from experience, simulation rate should be 20 times greater than the max. of signal frequency in the rotating frame
        rate_Hz = np.amax([21 * nuL_Hz_abs_max, 10 * T2Rate, 10 * RBW])
        return PhysicalQuantity(rate_Hz, "Hz")

    # def RandomJump(
    #     self,
    #     numofcohT=None,  # float. number of coherent period for simulation
    #     verbose=False,
    # ):
    #     """
    #     Generate simulation parameters by 'RandomJump' method.

    #     Parameters
    #     ----------
    #     numofcohT : float

    #     Number of coherent period for simulation. Can be integer or float values. Default in None.

    #     verbose : bool

    #     """
    #     numofsampling = int(np.ceil(numofcohT))
    #     check(numofsampling)
    #     BALPamp_array = statRayleigh(
    #         sigma=self.excField.BALP,
    #         num=numofsampling,
    #         showplt=verbose,
    #         verbose=verbose,
    #     )
    #     theta_array = (
    #         statUni2Pi(num=numofsampling, showplt=verbose, verbose=verbose) / 2.0
    #     )
    #     phi_array = statUni2Pi(num=numofsampling, showplt=verbose, verbose=verbose)
    #     phase0_array = statUni2Pi(num=numofsampling, showplt=verbose, verbose=verbose)
    #     BALP_list = []

    #     BALPamp_array[0] = self.excField.BALP
    #     check(BALPamp_array)
    #     theta_array[0] = 90.0 * np.pi / 180.0
    #     phi_array[0] = 0.0 * np.pi / 180.0
    #     phase0_array[0] = 0

    #     numofsimustep_perperiod = int(self.simuRate_Hz * self.excField.cohT)
    #     timestamp = np.linspace(
    #         start=0,
    #         stop=numofsampling * self.excField.cohT,
    #         num=int(np.ceil(self.simuRate_Hz * numofsampling * self.excField.cohT)),
    #     )
    #     for i in range(numofsampling - 1):
    #         BALP = 0.5 * BALPamp_array[i]
    #         theta = theta_array[i]
    #         phi = phi_array[i]
    #         Bx = BALP * np.sin(theta) * np.cos(phi)
    #         By = BALP * np.sin(theta) * np.sin(phi)
    #         Bz = BALP * np.cos(theta)
    #         phase0 = phase0_array[i]
    #         for j in range(numofsimustep_perperiod):
    #             BALP_list.append([Bx, By, Bz, phase0].copy())  # room for modification?
    #     for i in [-1]:
    #         BALP = 0.5 * BALPamp_array[i]
    #         theta = theta_array[i]
    #         phi = phi_array[i]
    #         Bx = (
    #             BALP * np.sin(theta) * np.cos(phi) * np.cos(phi)
    #         )  # *np.cos(2*np.pi*self.nu_rot*timestamp)
    #         By = (
    #             BALP * np.sin(theta) * np.sin(phi) * np.sin(phi)
    #         )  # *np.sin(2*np.pi*self.nu_rot*timestamp)
    #         Bz = BALP * np.cos(theta)
    #         phase0 = phase0_array[i]
    #         numofstep_lastperiod = int(
    #             self.simuRate_Hz * self.excField.cohT * (numofcohT - numofsampling + 1)
    #         )
    #         check(numofstep_lastperiod)
    #         for j in range(numofstep_lastperiod):
    #             BALP_list.append(
    #                 [Bx, By, Bz, phase0].copy()
    #             )  # self.BALP_array [Bx, By, Bz, phase0]
    #     del BALPamp_array, theta_array, phi_array, phase0_array
    #     self.Bexc_t_vec = np.array(BALP_list)  # self.BALP_array [Bx, By, Bz, phase0]
    #     self.dBexc_dt_vec = np.zeros(self.Bexc_t_vec.shape)
    #     # check(self.BALP_array)
    #     self.timeStamp = np.arange(len(self.Bexc_t_vec) + 1) * self.timeStep_s

    # @nb.jit(
    #     [
    #         "int16(int64, int64, float64[:], \
    #     float64, float64[:], float64[:], \
    #     float64[:], float64[:,:], float64[:,:],     \
    #     float64[:], float64, float64, \
    #     float64, float64)"
    #     ],
    #     nopython=True,
    # )
    # def ThermalLightLoop(
    #     numofsimustep,  # int
    #     numofALPparticle,  # int
    #     random_phase,  # float[:]
    #     ALP_B,  # float
    #     ALP_nu_rot,  # float[:]
    #     ALP_phase,  # float[:]
    #     ALP_vtova_arr,  # float[:]
    #     BALP_arr,  # float[:,:]
    #     dBALPdt_arr,  # float[:,:]
    #     ALPwind_direction_earth,  # float[:]
    #     init_time,  # float
    #     simustep,  # float
    #     station_theta,  # float
    #     station_phi,  # float
    # ):

    #     # BALP_list = np.array([])
    #     for i in range(numofsimustep):
    #         # if verbose and i%10000 == 0:
    #         #     check(i)
    #         # decide BALP amplitude and phase0 from 2D random walk and decoherence due to 2 particle collision in each step
    #         # update ALP particles' phase array
    #         (
    #             ALP_phase[(2 * i) % (2 * numofALPparticle)],
    #             ALP_phase[(2 * i + 1) % (2 * numofALPparticle)],
    #         ) = (random_phase[2 * i], random_phase[2 * i + 1])
    #         # BALP = abs(rw)  # ALPwind_BALP *
    #         # phase0 = np.angle(rw)
    #         BALP = np.sum(
    #             ALP_B
    #             * ALP_vtova_arr
    #             * np.sin(
    #                 2 * np.pi * ALP_nu_rot * (init_time + i * simustep) + ALP_phase
    #             )
    #         ) / sqrt(2.0 * numofALPparticle)
    #         dBALPdt = np.sum(
    #             ALP_B
    #             * ALP_vtova_arr
    #             * 2
    #             * np.pi
    #             * ALP_nu_rot
    #             * np.cos(
    #                 2 * np.pi * ALP_nu_rot * (init_time + i * simustep) + ALP_phase
    #             )
    #         ) / sqrt(2.0 * numofALPparticle)
    #         # decide the direciton of B_ALP from the experiment time and motion of celestial bodies
    #         theta_e = ALPwind_direction_earth[1]  #
    #         phi_e = ALPwind_direction_earth[2]  #
    #         theta_s = station_theta  # theta_station
    #         phi_s = (
    #             init_time + i * simustep
    #         ) * 2 * 3.141592 / 86164.0 + station_phi  # phi_station

    #         x = sin(theta_e) * cos(theta_s) * cos(phi_e - phi_s) - cos(theta_e) * sin(
    #             theta_s
    #         )
    #         y = sin(theta_e) * sin(phi_e - phi_s)
    #         z = sin(theta_e) * sin(theta_s) * cos(phi_e - phi_s) + cos(theta_e) * cos(
    #             theta_s
    #         )
    #         Bx, By, Bz = 0.5 * BALP * np.array([x, y, z])
    #         dBxdt, dBydt, dBzdt = (
    #             0.5 * dBALPdt * np.array([x, y, z])
    #         )  # to be improved!!!!!!
    #         BALP_arr[i] = [Bx, By, Bz]  # , phase0 to be improved?
    #         dBALPdt_arr[i] = [dBxdt, dBydt, dBzdt]
    #         # BALP_modu_list.append(direction_lab)
    #     # BALP_array = np.array(BALP_list)
    #     # return BALP_array
    #     return 0

    # def ThermalLight(
    #     self,
    #     numofcohT=None,  # float. number of coherent period for simulation
    #     usenumba=True,
    #     verbose=False,
    # ):
    #     """
    #     Generate parameters by 'thermal light' method.

    #     Parameters
    #     ----------
    #     numofcohT : float

    #     Number of coherent period for simulation. Can be integer or float values. Default in None.

    #     verbose : bool

    #     """
    #     numofsimustep = int(self.simuRate_Hz * self.excField.cohT * numofcohT)
    #     # initialize an array of particles
    #     # rw2D = []
    #     # numofALPparticle = int(self.simurate*self.ALPwind.cohT)
    #     self.numofALPparticle = int(self.simuRate_Hz * self.excField.cohT)
    #     if self.numofALPparticle < 1000:
    #         print("WARNING: self.numofALPparticle < 1000")
    #     ALP_phase = statUni2Pi(
    #         num=2 * self.numofALPparticle, showplt=False, verbose=False
    #     )
    #     sin_arr = np.sin(ALP_phase)
    #     cos_arr = np.cos(ALP_phase)
    #     random_phase = uniform.rvs(loc=0, scale=2 * np.pi, size=2 * numofsimustep)

    #     self.ALP_B = (
    #         self.excField.BALP
    #     )  # why? * (1 + self.ALPwind.Gamma * np.random.standard_cauchy(size=2 * self.numofALPparticle))
    #     ALP_nu = self.excField.nu * (1 + (self.ALP_B / self.excField.BALP * 1e-3) ** 2)
    #     ALP_nu_rot = abs(ALP_nu) - abs(self.sample.gamma.value_in("Hz/T") * self.B0z_T / (2 * np.pi))
    #     self.va = 220 * 1e3  # 220 km/s
    #     self.speedtova = maxwell.rvs(size=2 * self.numofALPparticle)

    #     plt.rc("font", size=16)
    #     if verbose:
    #         fig = plt.figure(figsize=(8 * 0.6, 6 * 0.6), dpi=150)  #
    #         gs = gridspec.GridSpec(nrows=1, ncols=1)  #
    #         # fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
    #         #                     bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
    #         ax = fig.add_subplot(gs[0, 0])
    #         ax.hist(self.speedtova, bins=100, density=True, color="tab:green")
    #         ax.set_xlabel("$|v|/v_0$")
    #         ax.set_ylabel("Probability density")
    #         # plt.title('Speed distribution')
    #         plt.tight_layout()
    #         plt.show()
    #         # check(np.amin(self.speedtova))

    #     self.nu_a_arr = self.excField.nu * (1.0 + (self.speedtova * self.va / 3e8) ** 2)
    #     # check(np.amin(self.nu_a_arr))
    #     if verbose:
    #         fig = plt.figure(figsize=(8 * 0.6, 6 * 0.6), dpi=150)  #
    #         gs = gridspec.GridSpec(nrows=1, ncols=1)  #
    #         # fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
    #         #                     bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
    #         ax = fig.add_subplot(gs[0, 0])
    #         ax.hist(
    #             1e6 * (self.nu_a_arr / self.excField.nu - 1),
    #             bins=100,
    #             density=True,
    #             color="tab:purple",
    #         )
    #         ax.set_xlabel("$(\\nu/\\nu_a-1) \\times 10^6$")

    #         ax.set_ylabel("Probability density")
    #         ax.set_xlim(-0.5, 10.5)  # [0, 1, 2, 3, 4, 5, 6, 7,8 , 9, 10]
    #         ax.set_xticks([0, 2, 4, 6, 8, 10])  # [0, 1, 2, 3, 4, 5, 6, 7,8 , 9, 10]
    #         # plt.title('Speed distribution')
    #         plt.tight_layout()
    #         plt.show()

    #         # plt.hist(1e6 * (self.nu_a_arr/self.ALPwind.nu-1), bins=100, density=True)
    #         # plt.xlabel('$(\\nu/\\nu_a-1) \\times 10^6$')
    #         # plt.ylabel('Distribution density')
    #         # plt.title('Frequency distribution')
    #         # plt.show()

    #     ALP_nu_rot_arr = self.nu_a_arr - abs(
    #         self.sample.gamma.value_in("Hz/T") * self.B0z_T / (2 * np.pi)
    #     )
    #     # plt.hist(ALP_B/self.ALPwind.BALP, bins=1000)
    #     # plt.show()

    #     # for i in range(int(self.simurate*self.ALPwind.cohT)):
    #     #     phase = uniform.rvs(loc=0, scale=2*np.pi,size=numofstep)
    #     #     sin_arr = np.sin(phase)
    #     #     cos_arr = np.cos(phase)
    #     #     rw2D.append(np.sum(cos_arr) + 1j*np.sum(sin_arr))
    #     # rw2D = self.ALPwind.BALP*np.array(rw2D, dtype=np.complex64)/np.sqrt(self.numofALPparticle/2.)
    #     rw = (np.sum(cos_arr) + 1j * np.sum(sin_arr)) / np.sqrt(
    #         self.numofALPparticle / 2.0
    #     )

    #     # check(numofsimustep)
    #     self.Bexc_t_vec = np.zeros((numofsimustep, 3))
    #     self.dBexc_dt_vec = np.zeros((numofsimustep, 3))
    #     # loop
    #     if usenumba:
    #         # check(nb.typeof(numofsimustep))
    #         # check(nb.typeof(self.numofALPparticle))
    #         # check(nb.typeof(self.ALP_B))
    #         # check(nb.typeof(ALP_phase))
    #         # check(nb.typeof(rw))
    #         # check(nb.typeof(self.BALP_array))
    #         # check(nb.typeof(self.ALPwind.direction_earth))
    #         # check(nb.typeof(self.ALPwind.BALP))
    #         # check(nb.typeof(self.init_time))
    #         # check(nb.typeof(self.simustep))
    #         # check(nb.typeof(self.station.theta))
    #         # check(nb.typeof(self.station.phi))
    #         # check(nb.typeof(self.nu_rot))

    #         # numofsimustep,  # int
    #         # numofALPparticle,  # int
    #         # random_phase,  # float[:]

    #         # ALP_B,  # float
    #         # ALP_nu_rot,  # float[:]
    #         # ALP_phase,  # float[:]

    #         # ALP_vtova_arr,  # float[:]
    #         # BALP_arr,  # float[:,:]
    #         # dBALPdt_arr,  # float[:,:]

    #         # ALPwind_direction_earth,  # float[:]
    #         # init_time,  # float
    #         # simustep,  # float

    #         # station_theta,  # float
    #         # station_phi,  # float
    #         Simulation.ThermalLightLoop(
    #             numofsimustep=numofsimustep,  # int
    #             numofALPparticle=self.numofALPparticle,  # int
    #             random_phase=random_phase,  # float[:]
    #             ALP_B=self.ALP_B,  # float
    #             ALP_nu_rot=ALP_nu_rot_arr,  # float[:]
    #             ALP_phase=ALP_phase,  # float[:]
    #             ALP_vtova_arr=self.speedtova,  # float[:]
    #             BALP_arr=self.Bexc_t_vec,  # float[:,:]
    #             dBALPdt_arr=self.dBexc_dt_vec,  # float[:,:]
    #             ALPwind_direction_earth=self.excField.direction_earth,  # float[:]
    #             init_time=self.init_time,  # float
    #             simustep=self.timeStep_s,  # float
    #             station_theta=self.station.theta,  # float
    #             station_phi=self.station.phi,  # float
    #         )

    #     else:
    #         for i in range(numofsimustep):
    #             # if verbose and i%10000 == 0:
    #             #     check(i)
    #             # decide BALP amplitude and phase0 from 2D random walk and decoherence due to 2 particle collision in each step
    #             # tic = time.perf_counter()
    #             phase_2i, phase_2ip1 = (
    #                 ALP_phase[(2 * i) % (2 * self.numofALPparticle)],
    #                 ALP_phase[(2 * i + 1) % (2 * self.numofALPparticle)],
    #             )
    #             rw -= (
    #                 np.cos(phase_2i)
    #                 + 1j * np.sin(phase_2i)
    #                 + np.cos(phase_2ip1)
    #                 + 1j * np.sin(phase_2ip1)
    #             ) / np.sqrt(self.numofALPparticle / 2.0)

    #             phase_2i_new, phase_2ip1_new = (
    #                 random_phase[2 * i],
    #                 random_phase[2 * i + 1],
    #             )
    #             rw += (
    #                 np.cos(phase_2i_new)
    #                 + 1j * np.sin(phase_2i_new)
    #                 + np.cos(phase_2ip1_new)
    #                 + 1j * np.sin(phase_2ip1_new)
    #             ) / np.sqrt(2.0 * self.numofALPparticle)
    #             (
    #                 ALP_phase[(2 * i) % (2 * self.numofALPparticle)],
    #                 ALP_phase[(2 * i + 1) % (2 * self.numofALPparticle)],
    #             ) = (phase_2i_new, phase_2ip1_new)

    #             BALP = self.excField.BALP * np.abs(rw)
    #             phase0 = np.angle(rw)
    #             # decide the direciton of B_ALP from the experiment time and motion of celestial bodies
    #             # toc = time.perf_counter()
    #             # newphase_generation += toc-tic
    #             # tic = time.perf_counter()
    #             direction_lab = Npole2station(
    #                 theta_e=self.excField.direction_earth[1],  #
    #                 phi_e=self.excField.direction_earth[2],  #
    #                 theta_s=self.station.theta,  # theta_station
    #                 phi_s=(self.init_time + i * self.timeStep_s) * 2 * np.pi / 86164.0
    #                 + self.station.phi,  # phi_station
    #                 verbose=False,
    #             )
    #             Bx, By, Bz = BALP * direction_lab
    #             self.Bexc_t_vec[i] = [Bx, By, Bz, phase0]
    #             # BALP_modu_list.append(direction_lab)
    #             # toc = time.perf_counter()
    #             # newBALP_generation += toc-tic
    #     # looptime = newphase_generation + newBALP_generation
    #     # if verbose:
    #     #     check(newphase_generation/looptime)
    #     self.timeStamp = np.arange(len(self.Bexc_t_vec)) * self.timeStep_s
    #     check(self.excField.BALP * abs(self.sample.gamma.value_in("Hz/T")))
    #     # check(self.BALP_array[:, 0])
    #     self.BALPsq_arr = (
    #         self.Bexc_t_vec[:, 0] ** 2
    #         + self.Bexc_t_vec[:, 1] ** 2
    #         + self.Bexc_t_vec[:, 2] ** 2
    #     )
    #     check(np.mean(np.sqrt(self.BALPsq_arr)) * abs(self.sample.gamma.value_in("Hz/T")))
    #     check(np.sqrt(np.mean(self.BALPsq_arr)) * abs(self.sample.gamma.value_in("Hz/T")))
    #     if verbose:
    #         fig = plt.figure(figsize=(8 * 0.6, 6 * 0.6), dpi=150)  #
    #         gs = gridspec.GridSpec(nrows=1, ncols=1)  #
    #         # fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
    #         #                     bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
    #         ax = fig.add_subplot(gs[0, 0])
    #         ax.plot(
    #             self.timeStamp,
    #             2 * self.Bexc_t_vec[:, 0] / self.excField.BALP,
    #             color="tab:brown",
    #         )
    #         ax.set_xlabel("time (s)")
    #         ax.set_ylabel("$B_{a,t} / B_{a} $")
    #         plt.tight_layout()
    #         plt.show()
    #     # plt.hist(np.sqrt(self.BALPsq_arr)/self.ALPwind.BALP, bins=30)
    #     # plt.title('BALP_array/ALPwind.BALP')
    #     # plt.show()
    #     if verbose:
    #         fig = plt.figure(figsize=(8 * 0.6, 6 * 0.6), dpi=150)  #
    #         gs = gridspec.GridSpec(nrows=1, ncols=1)  #
    #         # fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
    #         #                     bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
    #         ax = fig.add_subplot(gs[0, 0])
    #         ax.hist(
    #             2 * np.sqrt(self.BALPsq_arr) / self.excField.BALP,
    #             bins=50,
    #             density=True,
    #             color="blue",
    #         )
    #         ax.set_xticks([0, 1, 2, 3, 4])
    #         ax.set_xlabel("$ |B_{a,t} / B_{a}| $")
    #         ax.set_ylabel("Probability density")
    #         plt.tight_layout()
    #         plt.show()

    # def InfCoherence(
    #     self,
    #     numofcohT=None,  # float. number of coherent period for simulation
    #     verbose=False,
    # ):
    #     """
    #     Generate parameters by 'thermal light' method.

    #     Parameters
    #     ----------
    #     numofcohT : float

    #     Number of coherent period for simulation. Can be integer or float values. Default in None.

    #     verbose : bool

    #     """
    #     numofsimustep = int(self.simuRate_Hz * self.excField.cohT * numofcohT)
    #     # initialize an array of particles
    #     # rw2D = []
    #     # numofALPparticle = int(self.simurate*self.ALPwind.cohT)
    #     self.numofALPparticle = int(1234567)
    #     ALP_phase = statUni2Pi(
    #         num=2 * self.numofALPparticle, showplt=False, verbose=False
    #     )
    #     sin_arr = np.sin(ALP_phase)
    #     cos_arr = np.cos(ALP_phase)
    #     rw = (np.sum(cos_arr) + 1j * np.sum(sin_arr)) / np.sqrt(
    #         self.numofALPparticle / 2.0
    #     )
    #     BALP = self.excField.BALP * np.abs(rw)
    #     phase0 = np.angle(rw)
    #     if verbose:
    #         check(numofsimustep)
    #     BALP_list = []
    #     BALP_modu_list = []
    #     # newphase_generation = 0
    #     # newBALP_generation = 0
    #     # loop
    #     for i in range(numofsimustep):
    #         # if verbose and i%10000 == 0:
    #         #     check(i)
    #         # decide the direciton of B_ALP from the experiment time and motion of celestial bodies
    #         # toc = time.perf_counter()
    #         # newphase_generation += toc-tic
    #         # tic = time.perf_counter()
    #         direction_lab = Npole2station(
    #             theta_e=self.excField.direction_earth[1],  #
    #             phi_e=self.excField.direction_earth[2],  #
    #             theta_s=self.station.theta,  # theta_station
    #             phi_s=(self.init_time + i * self.timeStep_s) * 2 * np.pi / 86164.0
    #             + self.station.phi,  # phi_station
    #             verbose=False,
    #         )
    #         Bx, By, Bz = BALP * direction_lab
    #         BALP_list.append([Bx, By, Bz, phase0].copy())
    #         BALP_modu_list.append(direction_lab)
    #         # toc = time.perf_counter()
    #         # newBALP_generation += toc-tic
    #     # looptime = newphase_generation + newBALP_generation
    #     # if verbose:
    #     #     check(newphase_generation/looptime)
    #     self.Bexc_t_vec = np.array(BALP_list)  # [Bx, By, Bz, phase0]
    #     self.BALP_modu_array = np.array(BALP_modu_list) * self.excField.BALP
    #     self.timeStamp = np.arange(len(self.Bexc_t_vec)) * self.timeStep_s

    # def GenerateParam(
    #     self,
    #     numofcohT=None,  # float. number of coherent period for simulation
    #     excType=None,  # 'inf coherence' 'ThermalLight'
    #     showplt=False,  # whether to plot B_ALP
    #     plotrate=None,
    #     verbose=False,
    # ):
    #     """
    #     Generate parameters for simulation.

    #     Parameters
    #     ----------
    #     numofcohT : float

    #     Number of coherent period for simulation. Can be integer or float values. Default in None.

    #     excType : string

    #     The type of excitation for generating simulation parameters.

    #     'RandomJump' - The amplitude, phase, direction of B_1 field will jump to new random values.

    #             The amplitude is sampled from Rayleigh distribution --> see utils.statRayleigh()

    #             phase, direction (theta and phi in spherical coordinates) obey U[0, 2 pi), U[0, pi) and U[0, 2 pi).

    #             theta->azimuthal angle, phi->polar angle.

    #             refer to [1] -> 5. Sensitivity scaling with averaging time.

    #     'thermal light' - Gradually change B_1 by the period of 1 coherent time.

    #             The amplitude, phase, direction of B_1 field obey same distributions as in 'RandomJump' method.

    #             However, there's no sudden jump. The decoherent progress happens gradually over time.

    #             Refer to [2] and [3] for more details in this 'thermal light source' like ensemble.

    #     verbose : bool

    #     Reference
    #     ---------
    #     [1] Budker, D., Graham, P. W., Ledbetter, M., Rajendran, S. & Sushkov, A. O.
    #         Proposal for a cosmic axion spin precession experiment (CASPEr). Phys. Rev.
    #         X 4, 021030 (2014).

    #     [2] Loudon, R. The Quantum Theory of Light 2nd edn (Oxford University Press, 1983).

    #     [3] Dmitry Budker and Alexander O. Sushkov,  Physics on your feet: Berkeley graduate exam questions, DOI: 10.1080/00107514.2016.1156750

    #     """
    #     if self.excField.cohT is None:
    #         raise ValueError("self.ALPwind.cohT is None")
    #     if numofcohT is None:
    #         raise TypeError("numofcohT is None")
    #     # if numofcohT < 1:
    #     #     raise ValueError('numofcohT < 1')
    #     self.numofcohT = numofcohT * 1.0
    #     self.method_dict = {
    #         "RandomJump": Simulation.RandomJump,
    #         "ThermalLight": Simulation.ThermalLight,
    #         "InfCoherence": Simulation.InfCoherence,
    #     }
    #     if excType not in self.method_dict.keys():
    #         raise KeyError("method not in self.method_dict.keys()")
    #     else:
    #         self.excType = excType

    #     self.method_dict[excType](
    #         self,
    #         numofcohT=numofcohT,  # float. number of coherent period for simulation
    #         verbose=verbose,
    #     )
    #     if showplt and excType == "InfCoherence":
    #         if plotrate > self.simuRate_Hz:
    #             print(
    #                 "WARNING: plotrate > self.simurate. plotrate will be decreased to simurate"
    #             )
    #             plotrate = self.simuRate_Hz
    #             plotintv = 1
    #         else:
    #             plotintv = int(1.0 * self.simuRate_Hz / plotrate)
    #         # self.BALP_array = np.array(BALP_list)  # [Bx, By, Bz, phase0]
    #         # self.timestamp = np.arange(len(self.BALP_array)+1)*self.simustep

    #         fig = plt.figure(figsize=(4 * 1.0, 3 * 1.0), dpi=150)  #
    #         gs = gridspec.GridSpec(nrows=1, ncols=1)  #
    #         # fig.subplots_adjust( left=left_spc, top=top_spc, right=right_spc,bottom=bottom_spc,
    #         # wspace=0.1, hspace=0.01)
    #         # BALPamp_ax = fig.add_subplot(gs[0,0])
    #         # BALPamp_ax.plot(self.timestamp[0:-1:plotintv], self.BALP_array[0:-1:plotintv, 3], label='$B_{ALP}$', color='tab:red', alpha=0.9)
    #         # BALPamp_ax.legend(loc='upper right')
    #         # BALPamp_ax.set_ylabel('ALP B field / T')
    #         # BALPamp_ax.set_xlabel('time (s)')

    #         BALPxyz_ax = fig.add_subplot(gs[0, 0])
    #         BALPxyz_ax.plot(
    #             self.timeStamp[0:-1:plotintv],
    #             self.BALP_modu_array[0:-1:plotintv, 0],
    #             label="ALP $B_{x}$",
    #             color="tab:blue",
    #             alpha=0.7,
    #         )  # self.BALP_array[0:-1:plotintv, 0]
    #         BALPxyz_ax.plot(
    #             self.timeStamp[0:-1:plotintv],
    #             self.BALP_modu_array[0:-1:plotintv, 1],
    #             label="ALP $B_{y}$",
    #             color="tab:orange",
    #             alpha=0.7,
    #         )
    #         BALPxyz_ax.plot(
    #             self.timeStamp[0:-1:plotintv],
    #             self.BALP_modu_array[0:-1:plotintv, 2],
    #             label="ALP $B_{z}$",
    #             color="tab:green",
    #             alpha=0.7,
    #         )
    #         BALPxyz_ax.set_ylabel("ALP B field / T")
    #         BALPxyz_ax.set_xlabel("Time / hour")
    #         BALPxyz_ax.legend(loc="upper right")
    #         formatter = mticker.FuncFormatter(lambda y, _: f"{y/3600:.0f}")
    #         BALPxyz_ax.xaxis.set_major_formatter(formatter)
    #         plt.tight_layout()
    #         plt.show()
    #     elif showplt and excType != "InfCoherence":
    #         if plotrate > self.simuRate_Hz:
    #             print(
    #                 "WARNING: plotrate > self.simurate. plotrate will be decreased to simurate"
    #             )
    #             plotrate = self.simuRate_Hz
    #             plotintv = 1
    #         else:
    #             plotintv = int(1.0 * self.simuRate_Hz / plotrate)
    #         # self.BALP_array = np.array(BALP_list)  # [Bx, By, Bz, phase0]
    #         # self.timestamp = np.arange(len(self.BALP_array)+1)*self.simustep

    #         fig = plt.figure(figsize=(7 * 1.0, 3 * 1.0), dpi=150)  #
    #         gs = gridspec.GridSpec(nrows=1, ncols=2)  #
    #         # fig.subplots_adjust( left=left_spc, top=top_spc, right=right_spc,bottom=bottom_spc,
    #         # wspace=0.1, hspace=0.01)
    #         # BALPamp_ax = fig.add_subplot(gs[0,0])
    #         # BALPamp_ax.plot(self.timestamp[0:-1:plotintv], self.BALP_array[0:-1:plotintv, 3], label='$B_{ALP}$', color='tab:red', alpha=0.9)
    #         # BALPamp_ax.legend(loc='upper right')
    #         # BALPamp_ax.set_ylabel('ALP B field / T')
    #         # BALPamp_ax.set_xlabel('time (s)')

    #         BALPxyz_ax = fig.add_subplot(gs[0, 0])
    #         BALPxyz_ax.plot(
    #             self.timeStamp[0:-1:plotintv],
    #             self.Bexc_t_vec[0:-1:plotintv, 0],
    #             label="ALP $B_{x}$",
    #             color="tab:blue",
    #             alpha=0.7,
    #         )  # self.BALP_array[0:-1:plotintv, 0]
    #         BALPxyz_ax.plot(
    #             self.timeStamp[0:-1:plotintv],
    #             self.Bexc_t_vec[0:-1:plotintv, 1],
    #             label="ALP $B_{y}$",
    #             color="tab:orange",
    #             alpha=0.7,
    #         )
    #         BALPxyz_ax.plot(
    #             self.timeStamp[0:-1:plotintv],
    #             self.Bexc_t_vec[0:-1:plotintv, 2],
    #             label="ALP $B_{z}$",
    #             color="tab:green",
    #             alpha=0.7,
    #         )
    #         BALPxyz_ax.set_ylabel("ALP B field / T")
    #         BALPxyz_ax.set_xlabel("Time / hour")
    #         BALPxyz_ax.legend(loc="upper right")
    #         formatter_s2h = mticker.FuncFormatter(lambda x, _: f"{x/3600:.1f}")
    #         BALPxyz_ax.xaxis.set_major_formatter(formatter_s2h)

    #         BALPphase0_ax = fig.add_subplot(gs[0, 1])
    #         BALPphase0_ax.plot(
    #             self.timeStamp[0:-1:plotintv],
    #             self.Bexc_t_vec[0:-1:plotintv, 3],
    #             label="ALP $\Phi_{0}$",
    #             color="tab:cyan",
    #             alpha=1.0,
    #         )  # self.BALP_array[0:-1:plotintv, 0]
    #         BALPphase0_ax.set_ylabel("Phase0/$\pi$ ")
    #         BALPphase0_ax.set_xlabel("Time / hour")
    #         BALPphase0_ax.xaxis.set_major_formatter(formatter_s2h)
    #         # formatter_rad2pi = mticker.FuncFormatter(lambda y, _: f'{y/np.pi:.1f}')
    #         # BALPphase0_ax.yaxis.set_major_formatter(formatter_rad2pi)
    #         BALPphase0_ax.set_ylim(-1 * np.pi, 1 * np.pi)
    #         BALPphase0_ax.set_yticks([-1 * np.pi, 0.0, 1 * np.pi])
    #         BALPphase0_ax.set_yticklabels(["-$\pi$", "0", "$\pi$"])
    #         plt.tight_layout()
    #         plt.show()

    # def generatePulseExcitation(
    #     self,
    #     pulseDur: float = 100e-6,
    #     tipAngle: float = np.pi / 2,
    #     direction: np.ndarray = np.array([1, 0, 0]),
    #     nu_rot: float = None,
    #     showplt: bool = False,  # whether to plot B_ALP
    #     plotrate: float = None,
    #     verbose: bool = False,
    # ):
    #     self.excType = "pulse"
    #     B1 = 2 * tipAngle / (self.sample.gamma.value_in("Hz/T") * pulseDur)
    #     duty_func = partial(gate, start=0, stop=pulseDur)
    #     if nu_rot is None:
    #         nu_rot = self.excField.nu - self.demodFreq_Hz
    #     self.excField.setXYPulse(
    #         timeStamp=self.timeStamp,
    #         B1=B1,  # amplitude of the excitation pulse in (T)
    #         nu_rot=nu_rot,  # Hz
    #         init_phase=0,
    #         direction=direction,
    #         duty_func=duty_func,
    #         verbose=False,
    #     )

    #     # check(duty_func(pulseDur / 2))

    # @nb.jit
    @nb.jit(
        [
            "void(float64[:,:], float64[:,:], \
        float64, float64, float64, \
            float64, float64, float64, float64, float64, float64, \
        float64[:,:], float64[:,:], float64[:,:], float64[:,:])"
        ],
        nopython=True,
    )
    def generateTrajectory_1LoopByNb_loop(
        B_t,
        dBdt,  # \
        Mx,
        My,
        Mz,  # \
        gamma,
        timeStep,
        tsq_half,
        M0eq,
        T2,
        T1,  #
        trjry,
        dMdt,
        McrossB,
        d2Mdt2,
    ):
        N = B_t.shape[0]

        for i in range(N):
            # for i, Bexc in enumerate(B_t):  # self.BALP_array [Bx, By, Bz, phase0]
            # [Bx, By, Bz] = Bexc[
            #     0:3
            # ]  # *np.cos(2*np.pi * nu_rot * (i) * timestep + BALP[-1])
            Bx = B_t[i, 0]
            By = B_t[i, 1]
            Bz = B_t[i, 2]
            dBxdt = dBdt[i, 0]
            dBydt = dBdt[i, 1]
            dBzdt = dBdt[i, 2]

            # --- First derivatives ---
            # [dBxdt, dBydt, dBzdt] = dBdt[i][0:3]  #
            dMxdt = gamma * (My * Bz - Mz * By) - Mx / T2
            dMydt = gamma * (Mz * Bx - Mx * Bz) - My / T2
            dMzdt = gamma * (Mx * By - My * Bx) - (Mz - M0eq) / T1

            # --- Second derivatives ---
            d2Mxdt2 = (
                gamma * (dMydt * Bz + My * dBzdt - dMzdt * By - Mz * dBydt) - dMxdt / T2
            )
            d2Mydt2 = (
                gamma * (dMzdt * Bx + Mz * dBxdt - dMxdt * Bz - Mx * dBzdt) - dMydt / T2
            )
            d2Mzdt2 = (
                gamma * (dMxdt * By + Mx * dBydt - dMydt * Bx - My * dBxdt) - dMzdt / T1
            )

            # --- Integrate via truncated Taylor expansion ---
            Mx += dMxdt * timeStep + d2Mxdt2 * tsq_half
            My += dMydt * timeStep + d2Mydt2 * tsq_half
            Mz += dMzdt * timeStep + d2Mzdt2 * tsq_half
            # Mx1 = Mx + dMxdt * timeStep + d2Mxdt2 / 2.0 * timeStep**2
            # My1 = My + dMydt * timeStep + d2Mydt2 / 2.0 * timeStep**2
            # Mz1 = Mz + dMzdt * timeStep + d2Mzdt2 / 2.0 * timeStep**2

            # record magnetization. only for recording. trjry does not involve in the kinetic simulation
            trjry[i + 1, 0] += Mx
            trjry[i + 1, 1] += My
            trjry[i + 1, 2] += Mz

            # ------ if we want to monitor the process ------
            # dMdt[i, 0] = dMxdt
            # dMdt[i, 1] = dMydt
            # dMdt[i, 2] = dMzdt

            # McrossB[i, 0] = My * Bz - Mz * By
            # McrossB[i, 1] = Mz * Bx - Mx * Bz
            # McrossB[i, 2] = Mx * By - My * Bx

            # d2Mdt2[i, 0] = d2Mxdt2
            # d2Mdt2[i, 1] = d2Mydt2
            # d2Mdt2[i, 2] = d2Mzdt2
            # ------ --------------------------------- ------

            # [Mx, My, Mz] = [Mx1, My1, Mz1].copy()

    def generateTrajectory_1LoopByNb(self, verbose: bool = False):
        """
        Generate trajectory of magnetization vector in Cartesian coordinate system
        based on kinetic simulation for Bloch equations.
        """
        self.trjry = np.zeros((len(self.excField.B_vec) + 1, 3))
        M = self.init_M_amp
        theta = self.init_M_theta_rad
        phi = self.init_M_phi_rad
        [Mx0, My0, Mz0] = np.array(
            [
                M * np.sin(theta) * np.cos(phi),
                M * np.sin(theta) * np.sin(phi),
                M * np.cos(theta),
            ]
        )
        M_init = np.array([Mx0, My0, Mz0])
        # Magnetization at equilibrium
        M0equ = 1.0
        #
        self.trjry[0] = M_init
        tsq_half = 0.5 * self.timeStep_s**2
        for i, B0_i in enumerate(self.magnet.B_vals_T):
            ratio = self.magnet.ratios[i]

            self.dMdt = np.zeros((len(self.excField.B_vec), 3))
            self.McrossB = np.zeros((len(self.excField.B_vec), 3))
            self.d2Mdt2 = np.zeros((len(self.excField.B_vec), 3))

            B0z_rot_amp = B0_i - self.RCF_freq_Hz / (
                self.sample.gamma.value_in("Hz/T") / (2 * np.pi)
            )  # scalar
            B0z_rot = B0z_rot_amp * np.ones(
                len(self.excField.B_vec)
            )  # B0z_rot.shape is (N)
            B0_rot = np.outer(B0z_rot, np.array([0, 0, 1]))  # B0_rot.shape is (N, 3)

            Simulation.generateTrajectory_1LoopByNb_loop(
                B_t=self.excField.B_vec + B0_rot,
                dBdt=self.excField.dBdt_vec,
                Mx=ratio * Mx0,
                My=ratio * My0,
                Mz=ratio * Mz0,
                gamma=self.sample.gamma.value_in("Hz/T"),
                timeStep=self.timeStep_s,
                tsq_half=tsq_half,
                M0eq=ratio * M0equ,
                T2=self.T2_s,
                T1=self.T1_s,
                trjry=self.trjry,
                dMdt=self.dMdt,
                McrossB=self.McrossB,
                d2Mdt2=self.d2Mdt2,
            )

    @nb.jit(
        [
            "void("
            "float64[:,:], float64[:,:], float64[:], float64[:], "
            "float64, float64, float64, float64, float64, "
            "float64, float64, float64, float64, float64,"
            "float64[:,:])"
        ],
        nopython=True,
    )
    # @nb.njit
    def generateTrajectory_2LoopsByNb_loop(
        B_vec,
        dBdt_vec,
        B_vals_T,
        ratios,
        gamma,
        timeStep,
        tSqHalf,
        T1,
        T2,
        RCF_Freq_Hz,
        Mx0,
        My0,
        Mz0,
        M0eqb,
        trjry,
    ):
        N = B_vec.shape[0]
        K = B_vals_T.shape[0]

        # trjry = np.zeros((K, N+1, 3))

        # trjry = np.zeros((N+1, 3))
        trjry[0, 0] = Mx0
        trjry[0, 1] = My0
        trjry[0, 2] = Mz0

        for k in range(K):  # OUTER LOOP (now compiled)
            Mx = ratios[k] * Mx0
            My = ratios[k] * My0
            Mz = ratios[k] * Mz0

            M0eqb_k = ratios[k] * M0eqb

            B0z_rot_amp = B_vals_T[k] - RCF_Freq_Hz / (gamma / (2 * np.pi))

            for i in range(N):  # INNER LOOP (compiled)
                Bx = B_vec[i, 0]
                By = B_vec[i, 1]
                Bz = B_vec[i, 2] + B0z_rot_amp

                dBxdt = dBdt_vec[i, 0]
                dBydt = dBdt_vec[i, 1]
                dBzdt = dBdt_vec[i, 2]

                # derivatives
                dMxdt = gamma * (My * Bz - Mz * By) - Mx / T2
                dMydt = gamma * (Mz * Bx - Mx * Bz) - My / T2
                dMzdt = gamma * (Mx * By - My * Bx) - (Mz - M0eqb_k) / T1

                d2Mxdt2 = (
                    gamma * (dMydt * Bz + My * dBzdt - dMzdt * By - Mz * dBydt)
                    - dMxdt / T2
                )
                d2Mydt2 = (
                    gamma * (dMzdt * Bx + Mz * dBxdt - dMxdt * Bz - Mx * dBzdt)
                    - dMydt / T2
                )
                d2Mzdt2 = (
                    gamma * (dMxdt * By + Mx * dBydt - dMydt * Bx - My * dBxdt)
                    - dMzdt / T1
                )

                Mx += dMxdt * timeStep + tSqHalf * d2Mxdt2
                My += dMydt * timeStep + tSqHalf * d2Mydt2
                Mz += dMzdt * timeStep + tSqHalf * d2Mzdt2

                # trjry[k, i+1, 0] = Mx
                # trjry[k, i+1, 1] = My
                # trjry[k, i+1, 2] = Mz
                trjry[i + 1, 0] += Mx
                trjry[i + 1, 1] += My
                trjry[i + 1, 2] += Mz

        # return trjry

    def generateTrajectory_2LoopsByNb(self, verbose: bool = False):
        """
        Generate trajectory of magnetization vector in Cartesian coordinate system
        based on kinetic simulation for Bloch equations.
        """
        self.trjry = np.zeros((len(self.excField.B_vec) + 1, 3))
        self.dMdt = np.zeros((len(self.excField.B_vec), 3))
        self.McrossB = np.zeros((len(self.excField.B_vec), 3))
        self.d2Mdt2 = np.zeros((len(self.excField.B_vec), 3))

        M = self.init_M_amp
        theta = self.init_M_theta_rad
        phi = self.init_M_phi_rad
        [Mx0, My0, Mz0] = np.array(
            [
                M * np.sin(theta) * np.cos(phi),
                M * np.sin(theta) * np.sin(phi),
                M * np.cos(theta),
            ]
        )
        M_init = np.array([Mx0, My0, Mz0])
        # Magnetization at equilibrium
        M0eqb = 1.0
        #
        # self.trjry[0] = M_init
        tSqHalf = 0.5 * self.timeStep_s**2
        Simulation.generateTrajectory_2LoopsByNb_loop(
            B_vec=self.excField.B_vec,
            dBdt_vec=self.excField.dBdt_vec,
            B_vals_T=self.magnet.B_vals_T,
            ratios=self.magnet.ratios,
            gamma=self.sample.gamma.value_in("Hz/T"),
            timeStep=self.timeStep_s,
            tSqHalf=tSqHalf,
            T2=self.T2_s,
            T1=self.T1_s,
            RCF_Freq_Hz=self.RCF_freq_Hz,
            Mx0=Mx0,
            My0=My0,
            Mz0=Mz0,
            M0eqb=M0eqb,
            trjry=self.trjry,
        )

        # for i, B0_i in enumerate(self.magnet_det.B_vals_T):
        #     ratio = self.magnet_det.ratios[i]

        #     self.dMdt = np.zeros((len(self.excField.B_vec), 3))
        #     self.McrossB = np.zeros((len(self.excField.B_vec), 3))
        #     self.d2Mdt2 = np.zeros((len(self.excField.B_vec), 3))

        #     B0z_rot_amp = B0_i - self.RCF_Freq_Hz / (
        #         self.sample.gamma.value_in("Hz/T") / (2 * np.pi)
        #     )  # scalar
        #     B0z_rot = B0z_rot_amp * np.ones(
        #         len(self.excField.B_vec)
        #     )  # B0z_rot.shape is (N)
        #     B0_rot = np.outer(B0z_rot, np.array([0, 0, 1]))  # B0_rot.shape is (N, 3)

        #     Simulation.generateTrajectoryLoop_doubleLoop(
        #         B_t=self.excField.B_vec + B0_rot,
        #         dBdt=self.excField.dBdt_vec,
        #         Mx=ratio * Mx,
        #         My=ratio * My,
        #         Mz=ratio * Mz,
        #         gamma=self.sample.gamma.value_in("Hz/T"),
        #         timeStep=self.timeStep_s,
        #         tSqHalf=tSqHalf,
        #         M0equ=ratio * M0equ,
        #         T2=self.T2_s,
        #         T1=self.T1_s,
        #         trjry=self.trjry,
        #         dMdt=self.dMdt,
        #         McrossB=self.McrossB,
        #         d2Mdt2=self.d2Mdt2,
        #     )

    def generateTrajectory_noNb(self, use_numba: bool = False, verbose: bool = False):
        """
        Generate trajectory of magnetization vector in Cartesian coordinate system
        based on kinetic simulation for Bloch equations.
        """
        self.trjry = np.zeros((len(self.excField.B_vec) + 1, 3))
        self.dMdt = np.zeros((len(self.excField.B_vec), 3))
        self.McrossB = np.zeros((len(self.excField.B_vec), 3))
        self.d2Mdt2 = np.zeros((len(self.excField.B_vec), 3))
        M = self.init_M_amp
        theta = self.init_M_theta_rad
        phi = self.init_M_phi_rad
        [Mx0, My0, Mz0] = np.array(
            [
                M * np.sin(theta) * np.cos(phi),
                M * np.sin(theta) * np.sin(phi),
                M * np.cos(theta),
            ]
        )
        M_init = np.array([Mx0, My0, Mz0])
        # vecM0 = np.array([Mx, My, Mz])
        # Magnetization at equilibrium
        M0equ = 1.0
        #
        self.trjry[0] = M_init
        #
        timeStep = self.timeStep_s
        gamma = self.sample.gamma.value_in("Hz/T")
        B0z_rot_amp = self.B0z_T - self.RCF_freq_Hz / (
            self.sample.gamma.value_in("Hz/T") / (2 * np.pi)
        )
        B0z_rot = B0z_rot_amp * np.ones(len(self.excField.B_vec))
        B0_rot = np.outer(B0z_rot, np.array([0, 0, 1]))

        # does not support inhomogeneous bias field
        for i, BALP in enumerate(self.excField.B_vec):
            [Bx, By, Bz] = BALP[0:3] * np.cos(
                2 * np.pi * self.nuL_Hz * (i) * timeStep + BALP[-1]
            )
            [dBxdt, dBydt, dBzdt] = (
                (-1.0)
                * BALP[0:3]
                * 2
                * np.pi
                * self.nuL_Hz
                * np.sin(2 * np.pi * self.nuL_Hz * (i) * timeStep + BALP[-1])
            )
            dMxdt = gamma * (My * Bz - Mz * By) - Mx / self.T2_s
            dMydt = gamma * (Mz * Bx - Mx * Bz) - My / self.T2_s
            dMzdt = gamma * (Mx * By - My * Bx) - (Mz - M0equ) / self.T1_s

            d2Mxdt2 = (
                gamma * (dMydt * Bz + My * dBzdt - dMzdt * By - Mz * dBydt)
                - dMxdt / self.T2_s
            )
            d2Mydt2 = (
                gamma * (dMzdt * Bx + Mz * dBxdt - dMxdt * Bz - Mx * dBzdt)
                - dMydt / self.T2_s
            )
            d2Mzdt2 = (
                gamma * (dMxdt * By + Mx * dBydt - dMydt * Bx - My * dBxdt)
                - dMzdt / self.T1_s
            )

            Mx1 = Mx + dMxdt * timeStep + d2Mxdt2 / 2.0 * timeStep**2
            My1 = My + dMydt * timeStep + d2Mydt2 / 2.0 * timeStep**2
            Mz1 = Mz + dMzdt * timeStep + d2Mzdt2 / 2.0 * timeStep**2

            self.trjry[i + 1] = [Mx1, My1, Mz1]
            self.dMdt[i] = [dMxdt, dMydt, dMzdt]
            self.McrossB[i] = [
                My * Bz - Mz * By,
                Mz * Bx - Mx * Bz,
                Mx * By - My * Bx,
            ]
            self.d2Mdt2[i] = [d2Mxdt2, d2Mydt2, d2Mzdt2]
            # [Mx0, My0, Mz0] = [Mx, My, Mz].copy()
            [Mx, My, Mz] = [Mx1, My1, Mz1].copy()

        if verbose:
            check(self.trjry.shape)

    @njit(
        [
            "void(float64[:,:], float64[:,:], float64[:], float64[:], "
            "float64, float64, float64, float64, float64, "
            "float64, float64, float64, float64, float64, "
            "float64[:,:])"
        ],
        nopython=True,
    )
    def generateTrajectory_vectorized_loop(
        B_vec,
        dBdt_vec,
        B_vals_T,
        ratios,
        gamma,
        timeStep,
        tSqHalf,
        T1,
        T2,
        RCF_freq_Hz,
        Mx0,
        My0,
        Mz0,
        M0eqb,
        trjry,
    ):
        # numFields = B_vec.shape[0]  # number of time steps
        numTimeSteps = B_vec.shape[0]  # number of time steps
        # K = B_vals_T.shape[0]  # number of ratios/B_vals

        # Initialize magnetization for all spin packets
        # M shape = (numPt)
        Mx = ratios * Mx0
        My = ratios * My0
        Mz = ratios * Mz0
        M0eqb_arr = ratios * M0eqb

        # Precompute B0z_rot_amp for all K
        B0z_rot_amp = B_vals_T - RCF_freq_Hz / (gamma / (2 * np.pi))

        # Initialize trajectory array for accumulation
        trjry[0, 0] = Mx0
        trjry[0, 1] = My0
        trjry[0, 2] = Mz0

        # Loop over time steps
        for i in range(numTimeSteps):
            # Extract B and dBdt at this time step
            Bx = B_vec[i, 0]
            By = B_vec[i, 1]
            Bz_raw = B_vec[i, 2]
            dBxdt = dBdt_vec[i, 0]
            dBydt = dBdt_vec[i, 1]
            dBzdt = dBdt_vec[i, 2]

            #
            Bz = Bz_raw + B0z_rot_amp

            # First derivatives (vectorized over K)
            dMxdt = gamma * (My * Bz - Mz * By) - Mx / T2
            dMydt = gamma * (Mz * Bx - Mx * Bz) - My / T2
            dMzdt = gamma * (Mx * By - My * Bx) - (Mz - M0eqb_arr) / T1

            # Second derivatives (vectorized over K)
            d2Mxdt2 = (
                gamma * (dMydt * Bz + My * dBzdt - dMzdt * By - Mz * dBydt) - dMxdt / T2
            )
            d2Mydt2 = (
                gamma * (dMzdt * Bx + Mz * dBxdt - dMxdt * Bz - Mx * dBzdt) - dMydt / T2
            )
            d2Mzdt2 = (
                gamma * (dMxdt * By + Mx * dBydt - dMydt * Bx - My * dBxdt) - dMzdt / T1
            )

            # Update M for all K
            Mx += dMxdt * timeStep + tSqHalf * d2Mxdt2
            My += dMydt * timeStep + tSqHalf * d2Mydt2
            Mz += dMzdt * timeStep + tSqHalf * d2Mzdt2

            # Accumulate trajectory
            trjry[i + 1, 0] += Mx.sum()
            trjry[i + 1, 1] += My.sum()
            trjry[i + 1, 2] += Mz.sum()

    def generateTrajectory_vectorized(self, verbose: bool = False):
        """
        Generate trajectory of magnetization vector in Cartesian coordinate system
        based on kinetic simulation for Bloch equations.
        """
        # numFields = self.excField.B_vec.shape[0]
        self.trjry = np.zeros((self.numSteps + 1, 3))
        self.dMdt = np.zeros((self.numSteps, 3))
        self.McrossB = np.zeros((self.numSteps, 3))
        self.d2Mdt2 = np.zeros((self.numSteps, 3))
        # print(f"{self.generateTrajectory_vectorized.__name__}: self.trjry.shape = ", self.trjry.shape)
        # print(f"{self.generateTrajectory_vectorized.__name__}: self.trjry[0] = ", self.trjry[0])
        [Mx0, My0, Mz0] = np.array(
            [
                self.init_M_amp
                * np.sin(self.init_M_theta_rad)
                * np.cos(self.init_M_phi_rad),
                self.init_M_amp
                * np.sin(self.init_M_theta_rad)
                * np.sin(self.init_M_phi_rad),
                self.init_M_amp * np.cos(self.init_M_theta_rad),
            ]
        )
        M_init = np.array([Mx0, My0, Mz0])
        # Magnetization at equilibrium
        M0eqb = 1.0
        #
        # self.trjry[0] = M_init
        tSqHalf = 0.5 * self.timeStep_s**2

        # @record_runtime_YorN(RECORD_RUNTIME)
        Simulation.generateTrajectory_vectorized_loop(
            B_vec=self.excField.B_vec,
            dBdt_vec=self.excField.dBdt_vec,
            B_vals_T=self.magnet.B_vals_T,
            ratios=self.magnet.ratios,
            gamma=self.sample.gamma.value_in("Hz/T"),
            timeStep=self.timeStep_s,
            tSqHalf=tSqHalf,
            T2=self.T2_s,
            T1=self.T1_s,
            RCF_freq_Hz=self.RCF_freq_Hz,
            Mx0=Mx0,
            My0=My0,
            Mz0=Mz0,
            M0eqb=M0eqb,
            trjry=self.trjry,
        )

    def generateTrajectories(self, cleanup: bool = True, verbose: bool = False):
        """
        Generate trajectory of magnetization vector in Cartesian coordinate system
        based on kinetic simulation for Bloch equations.
        """
        numFields = self.excField.B_vec.shape[0]
        # self.trjry = np.zeros((numFields, self.numSteps + 1, 3))
        # self.dMdt = np.zeros((numFields,self.numSteps, 3))
        # self.McrossB = np.zeros((numFields, self.numSteps, 3))
        # self.d2Mdt2 = np.zeros((numFields, self.numSteps, 3))
        # print(f"{self.generateTrajectory_vectorized.__name__}: self.trjry.shape = ", self.trjry.shape)
        # print(f"{self.generateTrajectory_vectorized.__name__}: self.trjry[0] = ", self.trjry[0])
        M = self.init_M_amp
        theta = self.init_M_theta_rad
        phi = self.init_M_phi_rad
        [Mx0, My0, Mz0] = np.array(
            [
                M * np.sin(theta) * np.cos(phi),
                M * np.sin(theta) * np.sin(phi),
                M * np.cos(theta),
            ]
        )
        # M_init = np.array([Mx0, My0, Mz0])
        # Magnetization at equilibrium
        M0eqb = 1.0
        #
        # self.trjry[0] = M_init
        tSqHalf = 0.5 * self.timeStep_s**2

        self.trjry, self.dMdt, self.McrossB, self.d2Mdt2 = bh.generateTrajectories(
            self.excField.B_vec,
            self.excField.dBdt_vec,
            self.magnet.B_vals_T,
            self.magnet.ratios,
            self.sample.gamma.value_in("Hz/T"),
            self.timeStep_s,
            tSqHalf,
            self.T1_s,
            self.T2_s,
            self.RCF_freq_Hz,
            Mx0,
            My0,
            Mz0,
            M0eqb,
        )
        if cleanup:
            del self.excField.B_vec, self.excField.dBdt_vec
            del self.dMdt, self.McrossB, self.d2Mdt2

    @njit(
        [
            "void(float64[:,:], float64[:,:], float64[:], float64[:], "
            "float64, float64, float64, float64, float64, "
            "float64, float64, float64, float64, float64, "
            "float64[:,:])"
        ],
        nopython=True,
    )
    def generateTrajectory_vec2_loop(
        B_vec,
        dBdt_vec,
        B_vals_T,
        ratios,
        gamma,
        timeStep,
        tSqHalf,
        T1,
        T2,
        RCF_freq_Hz,
        Mx0,
        My0,
        Mz0,
        M0eqb,
        trjry,
    ):
        # numFields = B_vec.shape[0]  # number of time steps
        numTimeSteps = B_vec.shape[0]  # number of time steps
        # K = B_vals_T.shape[0]  # number of ratios/B_vals

        # Initialize magnetization for all spin packets
        # M shape = (numPt)
        Mx = ratios * Mx0
        My = ratios * My0
        Mz = ratios * Mz0
        M0eqb_arr = ratios * M0eqb

        # Precompute B0z_rot_amp for all K
        B0z_rot_amp = B_vals_T - RCF_freq_Hz / (gamma / (2 * np.pi))

        # Initialize trajectory array for accumulation
        trjry[:, 0] = Mx0
        trjry[:, 1] = My0
        trjry[:, 2] = Mz0

        # Loop over time steps
        for i in range(numTimeSteps):
            # Extract B and dBdt at this time step
            Bx = B_vec[i, 0]
            By = B_vec[i, 1]
            Bz_raw = B_vec[i, 2]
            dBxdt = dBdt_vec[i, 0]
            dBydt = dBdt_vec[i, 1]
            dBzdt = dBdt_vec[i, 2]

            #
            Bz = Bz_raw + B0z_rot_amp

            # First derivatives (vectorized over K)
            dMxdt = gamma * (My * Bz - Mz * By) - Mx / T2
            dMydt = gamma * (Mz * Bx - Mx * Bz) - My / T2
            dMzdt = gamma * (Mx * By - My * Bx) - (Mz - M0eqb_arr) / T1

            # Second derivatives (vectorized over K)
            d2Mxdt2 = (
                gamma * (dMydt * Bz + My * dBzdt - dMzdt * By - Mz * dBydt) - dMxdt / T2
            )
            d2Mydt2 = (
                gamma * (dMzdt * Bx + Mz * dBxdt - dMxdt * Bz - Mx * dBzdt) - dMydt / T2
            )
            d2Mzdt2 = (
                gamma * (dMxdt * By + Mx * dBydt - dMydt * Bx - My * dBxdt) - dMzdt / T1
            )

            # Update M for all K
            Mx += dMxdt * timeStep + tSqHalf * d2Mxdt2
            My += dMydt * timeStep + tSqHalf * d2Mydt2
            Mz += dMzdt * timeStep + tSqHalf * d2Mzdt2

            # Accumulate trajectory across K
            # for f in range(numFields):
            trjry[i + 1, 0] += Mx.sum()
            trjry[i + 1, 1] += My.sum()
            trjry[i + 1, 2] += Mz.sum()

    def generateTrajectory_vec2(self, verbose: bool = False):
        """
        Generate trajectory of magnetization vector in Cartesian coordinate system
        based on kinetic simulation for Bloch equations.
        """
        # numFields = self.excField.B_vec.shape[0]
        self.trjry = np.zeros((self.numSteps + 1, 3))
        self.dMdt = np.zeros((self.numSteps, 3))
        self.McrossB = np.zeros((self.numSteps, 3))
        self.d2Mdt2 = np.zeros((self.numSteps, 3))
        # print(f"{self.generateTrajectory_vectorized.__name__}: self.trjry.shape = ", self.trjry.shape)
        # print(f"{self.generateTrajectory_vectorized.__name__}: self.trjry[0] = ", self.trjry[0])
        M = self.init_M_amp
        theta = self.init_M_theta_rad
        phi = self.init_M_phi_rad
        [Mx0, My0, Mz0] = np.array(
            [
                M * np.sin(theta) * np.cos(phi),
                M * np.sin(theta) * np.sin(phi),
                M * np.cos(theta),
            ]
        )
        M_init = np.array([Mx0, My0, Mz0])
        # Magnetization at equilibrium
        M0eqb = 1.0
        #
        # self.trjry[0] = M_init
        tSqHalf = 0.5 * self.timeStep_s**2

        # @record_runtime_YorN(RECORD_RUNTIME)
        Simulation.generateTrajectory_vec2_loop(
            B_vec=self.excField.B_vec,
            dBdt_vec=self.excField.dBdt_vec,
            B_vals_T=self.magnet.B_vals_T,
            ratios=self.magnet.ratios,
            gamma=self.sample.gamma.value_in("Hz/T"),
            timeStep=self.timeStep_s,
            tSqHalf=tSqHalf,
            T2=self.T2_s,
            T1=self.T1_s,
            RCF_freq_Hz=self.RCF_freq_Hz,
            Mx0=Mx0,
            My0=My0,
            Mz0=Mz0,
            M0eqb=M0eqb,
            trjry=self.trjry,
        )

    def monitorTrajectory(
        self,
        plotRate: float = None,  #
        verbose: bool = False,
    ):
        if plotRate is None:
            plotRate = self.rate_Hz

        if plotRate > self.rate_Hz:
            print(
                "WARNING: samprate > self.simurate. samprate will be decreased to simurate"
            )
            plotRate = self.rate_Hz
            plotIntv = 1
        else:
            plotIntv = int(1.0 * self.rate_Hz / plotRate)

        self.trjry_mean = self.trjry.copy()

        BALP_array_step = np.concatenate(
            (self.excField.B_vec, [self.excField.B_vec[-1]]), axis=0
        )
        timestamp_step = np.arange(
            start=0, stop=(self.timeLen) * self.timeStep_s, step=self.timeStep_s
        )
        fig = plt.figure(figsize=(15 * 0.8, 7 * 0.8), dpi=150)  #
        gs = gridspec.GridSpec(nrows=2, ncols=4)  #
        # fix the margins
        left = 0.056
        bottom = 0.1
        right = 0.985
        top = 0.924
        wspace = 0.313
        hspace = 0.127
        fig.subplots_adjust(
            left=left, top=top, right=right, bottom=bottom, wspace=wspace, hspace=hspace
        )

        BALPamp_ax = fig.add_subplot(gs[0, 0])
        Mxy_ax = fig.add_subplot(gs[0, 1], sharex=BALPamp_ax)
        Mz_ax = fig.add_subplot(gs[1, 1], sharex=BALPamp_ax)
        dMxydt_ax = fig.add_subplot(gs[0, 2], sharex=BALPamp_ax)
        dMzdt_ax = fig.add_subplot(gs[1, 2], sharex=BALPamp_ax)
        d2Mxydt_ax = fig.add_subplot(gs[0, 3], sharex=BALPamp_ax)
        d2Mzdt_ax = fig.add_subplot(gs[1, 3], sharex=BALPamp_ax)

        # if self.excType == "RandomJump":
        #     lastnum = -2
        # else:
        #     lastnum = -1
        lastIndx = -1
        # print("np.std(BALP_array_step[:, 0]) =", np.std(BALP_array_step[:, 0]))
        # print("np.std(BALP_array_step[:, 1]) =", np.std(BALP_array_step[:, 1]))

        # print("np.std(self.trjry[:, 0]) =", np.std(self.trjry[:, 0]))
        # print("np.std(self.trjry[:, 1]) =", np.std(self.trjry[:, 1]))
        BALPamp_ax.plot(
            timestamp_step[0:lastIndx:plotIntv],
            BALP_array_step[0:-1:plotIntv, 0],
            label="$B_{x}$",
            color="tab:blue",
            alpha=0.7,
        )  # self.BALP_array[0:-1:plotintv, 0]
        BALPamp_ax.plot(
            timestamp_step[0:lastIndx:plotIntv],
            BALP_array_step[0:-1:plotIntv, 1],
            label="$B_{y}$",
            color="tab:orange",
            alpha=0.7,
        )
        BALPamp_ax.plot(
            timestamp_step[0:lastIndx:plotIntv],
            BALP_array_step[0:-1:plotIntv, 2],
            label="$B_{z}$",
            color="tab:green",
            alpha=0.7,
        )

        BALPamp_ax.set_ylabel("Magnetic field (T)")  # $B_\\mathrm{exc}$
        BALPamp_ax.set_xlabel("time (s)")
        BALPamp_ax.legend(loc="upper right")

        Mtabs = np.sqrt(
            self.trjry[0:-1:plotIntv, 0] ** 2 + self.trjry[0:-1:plotIntv, 1] ** 2
        )

        Mxy_ax.plot(
            timestamp_step[0:lastIndx:plotIntv],
            self.trjry[0:-1:plotIntv, 0],
            label="$M_x$",
            color="tab:orange",
            alpha=1,
        )
        Mxy_ax.plot(
            timestamp_step[0:lastIndx:plotIntv],
            self.trjry[0:-1:plotIntv, 1],
            label="$M_y$",
            color="tab:blue",
            alpha=1,
        )
        Mxy_ax.plot(
            timestamp_step[0:lastIndx:plotIntv],
            Mtabs,
            label="$|M_\\mathrm{transverse}|$",
            color="tab:brown",
            alpha=0.7,
            linestyle="--",
        )

        Mxy_ax.legend(loc="upper right")
        # Mxy_ax.set_xlabel("time (s)")
        Mxy_ax.set_ylabel("")
        Mxy_ax.grid()

        Mz_ax.plot(
            timestamp_step[0:lastIndx:plotIntv],
            self.trjry[0:-1:plotIntv, 2],
            label="$M_z$",
            color="tab:pink",
        )
        Mz_ax.legend(loc="upper right")
        Mz_ax.grid()
        Mz_ax.set_xlabel("time (s)")
        Mz_ax.set_ylabel("")
        Mz_ax.set_ylim(0, 1.1)

        dMxydt_ax.plot(
            timestamp_step[0:-1:plotIntv],
            self.dMdt[0:-1:plotIntv, 0],
            label="$d M_x / dt$",
            color="tab:gray",
            alpha=0.7,
        )
        dMxydt_ax.plot(
            timestamp_step[0:-1:plotIntv],
            self.dMdt[0:-1:plotIntv, 1],
            label="$d M_y / dt$",
            color="tab:olive",
            alpha=0.7,
        )
        dMxydt_ax.legend(loc="upper right")
        dMxydt_ax.grid()
        # dMxydt_ax.set_xlabel('time (s)')
        dMxydt_ax.set_ylabel("")

        dMzdt_ax.plot(
            timestamp_step[0:-1:plotIntv],
            self.dMdt[0:-1:plotIntv, 2],
            label="$d M_z / dt$",
            color="tab:cyan",
            alpha=1,
        )
        dMzdt_ax.legend(loc="upper right")
        dMzdt_ax.grid()
        dMzdt_ax.set_xlabel("time (s)")
        dMzdt_ax.set_ylabel("")

        d2Mxydt_ax.plot(
            timestamp_step[0:-1:plotIntv],
            self.d2Mdt2[0:-1:plotIntv, 0],
            label="$d^2 M_x /d t^2$",
            color="tab:blue",
            alpha=0.7,
        )
        d2Mxydt_ax.plot(
            timestamp_step[0:-1:plotIntv],
            self.d2Mdt2[0:-1:plotIntv, 1],
            label="$d^2 M_y /d t^2$",
            color="tab:cyan",
            alpha=0.7,
        )
        d2Mxydt_ax.legend(loc="upper right")
        d2Mxydt_ax.grid()
        # McrossBxy_ax.set_xlabel('time (s)')
        d2Mxydt_ax.set_ylabel("")

        d2Mzdt_ax.plot(
            timestamp_step[0:-1:plotIntv],
            self.d2Mdt2[0:-1:plotIntv, 2],
            label="$d^2 M_z /d t^2$",
            color="tab:purple",
            alpha=1,
        )
        d2Mzdt_ax.legend(loc="upper right")
        d2Mzdt_ax.grid()
        d2Mzdt_ax.set_xlabel("time (s)")
        d2Mzdt_ax.set_ylabel("")

        fig.suptitle(f"T2={self.T2_s:.1g}s T1={self.T1_s:.1e}s")
        # gaNN={self.excField.gaNN:.0e} axion_nu={self.excField.nu:.1e}\nXe
        # print(f'TrajectoryMonitoring_gaNN={self.ALPwind.gaNN:.0e}_axion_nu={self.ALPwind.nu:.1e}_Xe_T2={self.T2:.1g}s_T1={self.T1:.1e}s')
        # plt.tight_layout()
        plt.show()

    def monitorTrajectories(
        self,
        plotRate: float = None,  #
        verbose: bool = False,
    ):
        if plotRate is None:
            plotRate = self.rate_Hz

        if plotRate > self.rate_Hz:
            print(
                "WARNING: samprate > self.simurate. samprate will be decreased to simurate"
            )
            plotRate = self.rate_Hz
            plotIntv = 1
        else:
            plotIntv = int(1.0 * self.rate_Hz / plotRate)
        numFields, timeLen, _ = self.trjry.shape
        # self.trjry.shape = (numFields, timeLen, 3)
        # self.trjry_visual.shape = (timeLen, 3)
        # self.trjry_mean = np.mean(
        #     self.trjry, axis=0
        # )  # or
        if numFields > 1:
            self.trjry_mean = self.trjry.mean(axis=0)
            self.M_mean = np.sqrt(
                self.trjry[:, :, 0] ** 2 + self.trjry[:, :, 1] ** 2
            ).mean(axis=0)
            self.M_std = np.sqrt(
                self.trjry[:, :, 0] ** 2 + self.trjry[:, :, 1] ** 2
            ).std(axis=0)
            BALP_array_step = np.concatenate(
                (
                    self.excField.B_vec.mean(axis=0),
                    [self.excField.B_vec.mean(axis=0)[-1]],
                ),
                axis=0,
            )
        else:
            self.trjry_mean = self.trjry[0]
            self.M_mean = np.sqrt(self.trjry[0, :, 0] ** 2 + self.trjry[0, :, 1] ** 2)
            self.M_std = None
            BALP_array_step = np.concatenate(
                (self.excField.B_vec[0], [self.excField.B_vec[0][-1]]),
                axis=0,
            )

        timeStamp_s = self.timeStep_s * np.arange(self.timeLen)

        fig = plt.figure(figsize=(15 * 0.8, 7 * 0.8), dpi=150)  #
        gs = gridspec.GridSpec(nrows=2, ncols=4)  #
        # fix the margins
        left = 0.056
        bottom = 0.1
        right = 0.985
        top = 0.924
        wspace = 0.313
        hspace = 0.127
        fig.subplots_adjust(
            left=left, top=top, right=right, bottom=bottom, wspace=wspace, hspace=hspace
        )

        Ba_ax = fig.add_subplot(gs[0, 0])
        M_ax = fig.add_subplot(gs[1, 0])
        Mxy_ax = fig.add_subplot(gs[0, 1], sharex=Ba_ax)
        Mz_ax = fig.add_subplot(gs[1, 1], sharex=Ba_ax)
        dMxydt_ax = fig.add_subplot(gs[0, 2], sharex=Ba_ax)
        dMzdt_ax = fig.add_subplot(gs[1, 2], sharex=Ba_ax)
        d2Mxydt_ax = fig.add_subplot(gs[0, 3], sharex=Ba_ax)
        d2Mzdt_ax = fig.add_subplot(gs[1, 3], sharex=Ba_ax)

        # if self.excType == "RandomJump":
        #     lastnum = -2
        # else:
        #     lastnum = -1
        lastIndx = -1
        # print("np.std(BALP_array_step[:, 0]) =", np.std(BALP_array_step[:, 0]))
        # print("np.std(BALP_array_step[:, 1]) =", np.std(BALP_array_step[:, 1]))

        # print("np.std(self.trjry[:, 0]) =", np.std(self.trjry[:, 0]))
        # print("np.std(self.trjry[:, 1]) =", np.std(self.trjry[:, 1]))
        Ba_ax.plot(
            timeStamp_s[0:lastIndx:plotIntv],
            BALP_array_step[0:lastIndx:plotIntv, 0],
            label="$B_{x}$",
            color="tab:blue",
            alpha=0.7,
        )
        Ba_ax.plot(
            timeStamp_s[0:lastIndx:plotIntv],
            BALP_array_step[0:lastIndx:plotIntv, 1],
            label="$B_{y}$",
            color="tab:orange",
            alpha=0.7,
        )
        Ba_ax.plot(
            timeStamp_s[0:lastIndx:plotIntv],
            BALP_array_step[0:lastIndx:plotIntv, 2],
            label="$B_{z}$",
            color="tab:green",
            alpha=0.7,
        )

        Ba_ax.set_ylabel("Magnetic field (T)")  # $B_\\mathrm{exc}$
        Ba_ax.set_xlabel("time (s)")
        Ba_ax.legend(loc="upper right")

        M_ax.plot(
            timeStamp_s[0:lastIndx:plotIntv],
            self.M_mean[0:lastIndx:plotIntv],
            label="mean",
            color="tab:orange",
            alpha=1,
            zorder=7,
        )
        if self.M_std is not None:
            M_ax.errorbar(
                x=timeStamp_s[0:lastIndx:plotIntv],
                y=self.M_mean[0:lastIndx:plotIntv],
                yerr=self.M_std[0:lastIndx:plotIntv],
                label="standard deviation",
                color="tab:blue",
                alpha=1,
            )

        M_ax.set_ylabel("$M_{xy}$")
        M_ax.set_xlabel("time (s)")
        M_ax.legend(loc="upper right")

        Mxy_ax.plot(
            timeStamp_s[0:lastIndx:plotIntv],
            self.trjry_mean[0:lastIndx:plotIntv, 0],
            label="$M_x$",
            color="tab:orange",
            alpha=1,
        )
        Mxy_ax.plot(
            timeStamp_s[0:lastIndx:plotIntv],
            self.trjry_mean[0:lastIndx:plotIntv, 1],
            label="$M_y$",
            color="tab:blue",
            alpha=1,
        )
        Mxy_ax.plot(
            timeStamp_s[0:lastIndx:plotIntv],
            self.M_mean[0:lastIndx:plotIntv],
            label="$|M_\\mathrm{transverse}|$",
            color="tab:brown",
            alpha=1.0,
            linestyle="--",
        )

        Mxy_ax.legend(loc="upper right")
        # Mxy_ax.set_xlabel("time (s)")
        Mxy_ax.set_ylabel("")
        Mxy_ax.grid()

        Mz_ax.plot(
            timeStamp_s[0:lastIndx:plotIntv],
            self.trjry_mean[0:lastIndx:plotIntv, 2],
            label="$M_z$",
            color="tab:pink",
        )
        Mz_ax.legend(loc="upper right")
        Mz_ax.grid()
        Mz_ax.set_xlabel("time (s)")
        Mz_ax.set_ylabel("")
        Mz_ax.set_ylim(0, 1.1)

        # dMxydt_ax.plot(
        #     self.timeStamp_s[0:lastIndx:plotIntv],
        #     self.dMdt[0 : -1 : plotIntv, 0],
        #     label="$d M_x / dt$",
        #     color="tab:gray",
        #     alpha=0.7,
        # )
        # dMxydt_ax.plot(
        #     self.timeStamp_s[0:lastIndx:plotIntv],
        #     self.dMdt[0 : -1 : plotIntv, 1],
        #     label="$d M_y / dt$",
        #     color="tab:olive",
        #     alpha=0.7,
        # )
        # dMxydt_ax.legend(loc="upper right")
        # dMxydt_ax.grid()
        # # dMxydt_ax.set_xlabel('time (s)')
        # dMxydt_ax.set_ylabel("")

        # dMzdt_ax.plot(
        #     self.timeStamp_s[0:lastIndx:plotIntv],
        #     self.dMdt[0 : -1 : plotIntv, 2],
        #     label="$d M_z / dt$",
        #     color="tab:cyan",
        #     alpha=1,
        # )
        # dMzdt_ax.legend(loc="upper right")
        # dMzdt_ax.grid()
        # dMzdt_ax.set_xlabel("time (s)")
        # dMzdt_ax.set_ylabel("")

        # d2Mxydt_ax.plot(
        #     self.timeStamp_s[0:lastIndx:plotIntv],
        #     self.d2Mdt2[0 : -1 : plotIntv, 0],
        #     label="$d^2 M_x /d t^2$",
        #     color="tab:blue",
        #     alpha=0.7,
        # )
        # d2Mxydt_ax.plot(
        #     self.timeStamp_s[0:lastIndx:plotIntv],
        #     self.d2Mdt2[0 : -1 : plotIntv, 1],
        #     label="$d^2 M_y /d t^2$",
        #     color="tab:cyan",
        #     alpha=0.7,
        # )
        # d2Mxydt_ax.legend(loc="upper right")
        # d2Mxydt_ax.grid()
        # # McrossBxy_ax.set_xlabel('time (s)')
        # d2Mxydt_ax.set_ylabel("")

        # d2Mzdt_ax.plot(
        #     self.timeStamp_s[0:lastIndx:plotIntv],
        #     self.d2Mdt2[0 : -1 : plotIntv, 2],
        #     label="$d^2 M_z /d t^2$",
        #     color="tab:purple",
        #     alpha=1,
        # )
        # d2Mzdt_ax.legend(loc="upper right")
        # d2Mzdt_ax.grid()
        # d2Mzdt_ax.set_xlabel("time (s)")
        # d2Mzdt_ax.set_ylabel("")

        fig.suptitle(f"T2={self.T2_s:.1e}s T1={self.T1_s:.1e}s")
        # gaNN={self.excField.gaNN:.0e} axion_nu={self.excField.nu:.1e}\nXe
        # print(f'TrajectoryMonitoring_gaNN={self.ALPwind.gaNN:.0e}_axion_nu={self.ALPwind.nu:.1e}_Xe_T2={self.T2:.1g}s_T1={self.T1:.1e}s')
        # plt.tight_layout()
        plt.show()

    def keepMeanStd(self):
        """
        Docstring for keepMeanStd

        :param self: Description
        """
        Mxy_magnitudes = np.sqrt(self.trjry[:, :, 0] ** 2 + self.trjry[:, :, 1] ** 2)
        Mx_magnitudes = np.abs(self.trjry[:, :, 0])
        My_magnitudes = np.abs(self.trjry[:, :, 1])
        Mz_magnitudes = np.abs(self.trjry[:, :, 2])

        # mean of root of squares
        self.Mxy_mrs = Mxy_magnitudes.mean(axis=0)
        self.Mxy_srs = Mxy_magnitudes.std(axis=0)

        self.Mxy_rms = np.sqrt((Mxy_magnitudes**2).mean(axis=0))
        self.Mxy_rss = np.sqrt((Mxy_magnitudes**2).std(axis=0))

        self.Mx_mrs = Mx_magnitudes.mean(axis=0)
        self.Mx_srs = Mx_magnitudes.std(axis=0)

        self.My_mrs = My_magnitudes.mean(axis=0)
        self.My_srs = My_magnitudes.std(axis=0)

        self.Mz_mrs = Mz_magnitudes.mean(axis=0)
        self.Mz_srs = Mz_magnitudes.std(axis=0)

        del self.trjry, Mxy_magnitudes, Mx_magnitudes, My_magnitudes, Mz_magnitudes

    def displayTrjry(
        self,
        plotRate_Hz: float = None,  #
        verbose: bool = False,
    ):
        if plotRate_Hz is None:
            plotRate_Hz = self.rate_Hz

        if plotRate_Hz > self.rate_Hz:
            print(
                "WARNING: samprate > self.simurate. samprate will be decreased to simurate"
            )
            plotRate_Hz = self.rate_Hz
            plotIntv = 1
        else:
            plotIntv = int(1.0 * self.rate_Hz / plotRate_Hz)

        timeStamp_s = np.linspace(
            start=0, stop=(self.timeLen) * self.timeStep_s, num=len(self.Mx_mrs)
        )
        fig = plt.figure(figsize=(15 * 0.8, 7 * 0.8), dpi=150)  #
        gs = gridspec.GridSpec(nrows=2, ncols=2)  #
        # fix the margins
        left = 0.1
        bottom = 0.1
        right = 0.985
        top = 0.924
        wspace = 0.313
        hspace = 0.260
        fig.subplots_adjust(
            left=left, top=top, right=right, bottom=bottom, wspace=wspace, hspace=hspace
        )
        Mx_ax = fig.add_subplot(gs[0, 0])
        My_ax = fig.add_subplot(gs[0, 1], sharex=Mx_ax, sharey=Mx_ax)
        Mxy_ax = fig.add_subplot(gs[1, 0], sharex=Mx_ax, sharey=Mx_ax)
        Mz_ax = fig.add_subplot(gs[1, 1], sharex=Mx_ax)
        axes = [Mx_ax, My_ax, Mxy_ax, Mz_ax]
        lastIndx = -1
        letters = ["x", "y", "xy", "z"]
        colorSets = np.array(
            [
                ["tab:orange", "tab:blue"],
                ["tab:red", "tab:blue"],
                ["tab:green", "tab:blue"],
                ["tab:pink", "tab:blue"],
            ]
        )
        # Sets = [["tab:orange", "tab:blue"],["tab:red", "tab:blue"],["tab:green", "tab:blue"],["tab:black", "tab:blue"]]
        mrsToDisplay = [self.Mx_mrs, self.My_mrs, self.Mxy_mrs, self.Mz_mrs]
        srsToDisplay = [self.Mx_srs, self.My_srs, self.Mxy_srs, self.Mz_srs]

        # Force scientific notation on ticks, not as offset
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))  # control when to switch to scientific

        for i, ax in enumerate(axes):
            ax.errorbar(
                x=timeStamp_s[0:lastIndx:plotIntv],
                y=mrsToDisplay[i][0:lastIndx:plotIntv],
                yerr=srsToDisplay[i][0:lastIndx:plotIntv],
                label="srs $M_" + letters[i] + "$",
                color=colorSets[i, 1],
                alpha=1,
            )
            ax.plot(
                timeStamp_s[0:lastIndx:plotIntv],
                mrsToDisplay[i][0:lastIndx:plotIntv],
                label="mrs $M_" + letters[i] + "$",
                color=colorSets[i, 0],
                alpha=1,
                zorder=3,
            )
            # ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))

        for ax in axes:
            ax.set_xlabel("time (s)")
            ax.set_ylabel("")
            ax.grid()
            ax.legend(loc="upper right")

        Mz_ax.set_ylim(0, 1.1)

        fig.suptitle(f"T2={self.T2_s:.1e}s; T1={self.T1_s:.1e}s")
        # plt.tight_layout()
        plt.show()
        return axes

    def visualizeTrajectory3D(
        self,
        plotrate: float,  # [Hz]
        # rotframe=True,
        verbose=False,
    ):
        if plotrate is None:
            plotrate = self.rate_Hz

        if plotrate > self.rate_Hz:
            print(
                "WARNING: plotrate > self.simurate. plotrate will be decreased to simurate"
            )
            # warnings.warn('plotrate > self.simurate. plotrate will be decreased to simurate', DeprecationWarning)
            plotrate = self.rate_Hz
            plotintv = 1
        else:
            plotintv = int(1.0 * self.rate_Hz / plotrate)

        # 3D plot for magnetization vector
        fig = plt.figure(figsize=(6, 5), dpi=150)
        gs = gridspec.GridSpec(nrows=1, ncols=1)
        # fig.subplots_adjust(left=left, top=top, right=right,
        #                             bottom=bottom, wspace=wspace, hspace=hspace)
        # threeD_ax:plt.Axes = fig.add_subplot(gs[0, 0], projection="3d")
        threeD_ax: Axes3D = fig.add_subplot(gs[0, 0], projection="3d")

        # verts = []
        # verts.append(list(zip(frequencies, spectrum_arr[i])))
        # print('verts.shape ', len(verts), len(verts[0]))
        # popt_arr = np.array(popt_arr)
        # confi95_arr = np.array(confi95_arr)
        # print('popt_arr.shape ', popt_arr.shape)
        # print('confi95_arr.shape ', popt_arr.shape)
        # poly = PolyCollection(verts)  # , facecolors=[cc('r'), cc('g'), cc('b'), cc('y')]
        # poly.set_alpha(0.9)
        # threeD_ax.add_collection3d(poly, zs=time_arr, zdir='y')
        # threeD_ax.set_xlabel('absolute frequency / ' + specxunit)
        # threeD_ax.set_xlim3d(np.amin(frequencies), np.amax(frequencies))
        # threeD_ax.set_zlabel('Flux PSD / $10^{-9}$' + specyunit, rotation=180)
        # threeD_ax.set_zlim3d(np.amin(spectrum_arr), np.amax(spectrum_arr))
        threeD_ax.plot(
            xs=self.trjry_mean[0:-1:plotintv, 0],
            ys=self.trjry_mean[0:-1:plotintv, 1],
            zs=self.trjry_mean[0:-1:plotintv, 2],
            zdir="z",
        )

        threeD_ax.zaxis._axinfo["juggled"] = (1, 2, 0)
        # threeD_ax.set_ylabel('Time / min')
        # threeD_ax.set_ylim3d(np.amin(time_arr)-5, np.amax(time_arr)+5)
        threeD_ax.grid(False)
        threeD_ax.w_xaxis.set_pane_color((1, 1, 1, 0.0))
        threeD_ax.w_yaxis.set_pane_color((1, 1, 1, 0.0))
        threeD_ax.w_zaxis.set_pane_color((1, 1, 1, 0.0))

        r = self.init_M_amp
        u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 20j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        threeD_ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r, alpha=0.4)
        XYZlim = [-1.2, 1.2]
        threeD_ax.set_xlim3d(XYZlim)
        threeD_ax.set_ylim3d(XYZlim)
        threeD_ax.set_zlim3d(XYZlim)
        try:
            threeD_ax.set_aspect("equal")
        except NotImplementedError:
            pass
        threeD_ax.set_box_aspect((1, 1, 1))

        threeD_ax.xaxis.set_label_text("x")  #
        threeD_ax.yaxis.set_label_text("y")  #
        threeD_ax.zaxis.set_label_text("z")  #

        fig.suptitle(f"T2={self.T2_s:.1g}s T1={self.T1_s:.1e}s")
        plt.tight_layout()
        plt.show()

    def statTrajectory(self, verbose=False):
        timestep = 1.0 / self.rate_Hz
        # xs=self.trjry_visual[0:-1:int(plotintv),0][0:plotlim], \
        # ys=self.trjry_visual[0:-1:int(plotintv),1][0:plotlim], \
        # zs=self.trjry_visual[0:-1:int(plotintv),0][0:plotlim]
        self.avgMxsq = np.mean(self.trjry[:, 0] ** 2, dtype=np.float64)
        self.avgMysq = np.mean(self.trjry[:, 1] ** 2, dtype=np.float64)
        self.avgMzsq = np.mean(self.trjry[:, 2] ** 2, dtype=np.float64)
        if verbose:
            check(self.avgMxsq)
            check(self.avgMysq)
            check(self.avgMzsq)
            check(np.sqrt(self.avgMxsq + self.avgMysq))

    def saveToH5(self, path: str = None, verbose: bool = False):
        """ """
        if path[-3:] != ".h5":
            suffix = ".h5"
        else:
            suffix = ""
        h5f = h5py.File(path + suffix, "w")
        # save_LIA = True
        save_simu = True
        save_axion = False
        save_station = False
        save_sample = True
        save_magnet = True

        if save_simu:
            simu_group = h5f.create_group("simulation")
            print("self.physicalQuantities = ", self.physicalQuantities)
            self.saveToH5group(group=simu_group, verbose=True)

            if self.trjry is not None:
                M_group = h5f.create_group("magnetization")
                M_mean = np.sqrt(
                    self.trjry[:, :, 0] ** 2 + self.trjry[:, :, 1] ** 2
                ).mean(axis=0)
                M_std = np.sqrt(
                    self.trjry[:, :, 0] ** 2 + self.trjry[:, :, 1] ** 2
                ).std(axis=0)
                save_phys_quantity(
                    group=M_group,
                    name="magnetization(mean)",
                    value=M_mean,
                    unit="dimensionless",
                )
                save_phys_quantity(
                    group=M_group,
                    name="magnetization(std)",
                    value=M_std,
                    unit="dimensionless",
                )

        if save_axion:
            axion_group = h5f.create_group("axion_wind")
            self.axion.saveToH5group(group=axion_group)

        # if save_station:
        #     h5station = h5f.create_group("Station")
        #     h5station.create_dataset("name", data=[self.station.name])

        if save_sample:
            sample_group = h5f.create_group("sample")
            self.sample.saveToH5group(group=sample_group)

        if save_magnet:
            magnet_group = h5f.create_group("magnet")
            self.magnet.saveToH5group(group=magnet_group)

        h5f.close()

    def loadFromH5(self, path: str = None, verbose: bool = False):
        # print(pathAndName)
        with h5py.File(path, "r", driver="core") as df:  # h5py loading method
            if verbose:
                check(df.keys())

    # def analyzeTrajectory(
    #     self,
    # ):
    #     type(Signal)
    #     print(Signal)

    #     self.trjryStream = Signal(
    #         # name="Simulation data",
    #         filelist=[],
    #         verbose=True,
    #     )
    #     # self.trjryStream.attenuation = 0
    #     # self.trjryStream.filterstatus = "off"
    #     # self.trjryStream.filter_TC = 0.0
    #     # self.trjryStream.filter_order = 0
    #     self.trjryStream.demodfreq = self.RCF_freq_Hz
    #     saveintv = 1
    #     self.trjryStream.samprate = self.rate_Hz / saveintv
    #     self.trjryStream.exptype = "Simulation"

    #     self.trjryStream.dataX = (
    #         1 * self.trjry[int(0 * self.rate_Hz) : -1 : saveintv, 0]
    #     )  # * \
    #     # np.cos(2 * np.pi * self.nu_rot * self.timestamp[0:-1:saveintv])
    #     self.trjryStream.dataY = (
    #         1 * self.trjry[int(0 * self.rate_Hz) : -1 : saveintv, 1]
    #     )  # * \
    #     # np.sin(2 * np.pi * self.nu_rot * self.timestamp[0:-1:saveintv])

    #     # self.liastream.dataX = 0.5 * 1 * \
    #     # 	np.cos(2 * np.pi * self.nu_rot * self.timestamp[0:-1:saveintv])
    #     # self.liastream.dataY = 0.5 * 1 * \
    #     # 	np.sin(2 * np.pi * self.nu_rot * self.timestamp[0:-1:saveintv])

    #     self.trjryStream.getNoPulsePSD(
    #         windowfunction="rectangle",
    #         # decayfactor=-10,
    #         chunksize=None,  # sec
    #         analysisrange=[0, -1],
    #         getstd=False,
    #         stddev_range=None,
    #         # polycorrparas=[],
    #         # interestingfreq_list=[],
    #         selectshots=[],
    #         verbose=False,
    #     )
    #     self.trjryStream.FitPSD(
    #         fitfunction="Lorentzian",  # 'Lorentzian' 'dualLorentzian' 'tribLorentzian' 'Gaussian 'dualGaussian' 'auto' 'Polyeven'
    #         inputfitparas=["auto", "auto", "auto", "auto"],
    #         smooth=False,
    #         smoothlevel=10,
    #         fitrange=["auto", "auto"],
    #         alpha=0.05,
    #         getresidual=False,
    #         getchisq=False,
    #         verbose=False,
    #     )

    # def analyzeB1(
    #     self,
    # ):
    #     self.B1Stream = Signal(
    #         name="Simulation data",
    #         # device="Simulation",
    #         # device_id="Simulation",
    #         filelist=[],
    #         verbose=True,
    #     )
    #     self.B1Stream.attenuation = 0
    #     self.B1Stream.filterstatus = "off"
    #     self.B1Stream.DTRC_TC = 0.0
    #     self.B1Stream.DTRC_order = 0
    #     self.B1Stream.demodfreq = self.RCF_freq_Hz
    #     saveintv = 1
    #     self.B1Stream.samprate = self.rate_Hz / saveintv
    #     # check(self.timestamp.shape)
    #     # check(self.trjry[0:-1:saveintv, 0].shape)

    #     self.B1Stream.dataX = (
    #         1 * self.excField.B_vec[int(0 * self.rate_Hz) : -1 : saveintv, 0]
    #     )  # * \
    #     # np.cos(2 * np.pi * self.nu_rot * self.timestamp[0:-1:saveintv])
    #     self.B1Stream.dataY = (
    #         1 * self.excField.B_vec[int(0 * self.rate_Hz) : -1 : saveintv, 1]
    #     )  # * \
    #     # np.sin(2 * np.pi * self.nu_rot * self.timestamp[0:-1:saveintv])

    #     # self.B1stream.dataX = 0.5 * 1 * \
    #     # 	np.cos(2 * np.pi * self.nu_rot * self.timestamp[0:-1:saveintv])
    #     # self.B1stream.dataY = 0.5 * 1 * \
    #     # 	np.sin(2 * np.pi * self.nu_rot * self.timestamp[0:-1:saveintv])

    #     self.B1Stream.getNoPulsePSD(
    #         windowfunction="rectangle",
    #         # decayfactor=-10,
    #         chunksize=None,  # sec
    #         analysisrange=[0, -1],
    #         getstd=False,
    #         stddev_range=None,
    #         # polycorrparas=[],
    #         # interestingfreq_list=[],
    #         selectshots=[],
    #         verbose=False,
    #     )
    #     self.B1Stream.FitPSD(
    #         fitfunction="Lorentzian",  # 'Lorentzian' 'dualLorentzian' 'tribLorentzian' 'Gaussian 'dualGaussian' 'auto' 'Polyeven'
    #         inputfitparas=["auto", "auto", "auto", "auto"],
    #         smooth=False,
    #         smoothlevel=1,
    #         fitrange=["auto", "auto"],
    #         alpha=0.05,
    #         getresidual=False,
    #         getchisq=False,
    #         verbose=False,
    #     )

    # def compareBandSig(self):
    #     self.analyzeTrajectory()

    #     specxaxis, ALP_signal_spec, specxunit, specyunit = self.trjryStream.getSpectrum(
    #         showtimedomain=True,
    #         showfit=True,
    #         showresidual=False,
    #         showlegend=True,  # !!!!!show or not to show legend
    #         spectype="PSD",  # in 'PSD', 'ASD', 'FLuxPSD', 'FluxASD'
    #         ampunit="V",
    #         specxunit="Hz",  # 'Hz' 'kHz' 'MHz' 'GHz' 'ppm' 'ppb'
    #         # specxlim=[self.demodfreq - 0 , self.demodfreq + 20],
    #         # specylim=[0, 4e-23],
    #         specyscale="linear",  # 'log', 'linear'
    #         showstd=False,
    #         showplt_opt=False,
    #         return_opt=True,
    #     )

    #     specxaxis, BALP_spec, specxunit, specyunit = self.excField.plotField(
    #         demodfreq=self.RCF_freq_Hz, samprate=self.rate_Hz, showplt_opt=False
    #     )

    #     fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
    #     gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures

    #     ax00 = fig.add_subplot(gs[0, 0])
    #     ax00.plot(
    #         specxaxis,
    #         BALP_spec / np.amax(BALP_spec),
    #         label="BALP_spec",
    #         linestyle="-",
    #         zorder=1,
    #     )
    #     ax00.plot(
    #         specxaxis,
    #         ALP_signal_spec / np.amax(ALP_signal_spec),  #
    #         label="ALP_signal_spec",
    #         linestyle="--",
    #     )
    #     ax00.plot(
    #         specxaxis,
    #         self.trjryStream.fitcurves[0] / np.amax(self.trjryStream.fitcurves[0]),
    #         label=self.trjryStream.fitreport,
    #         linestyle="--",
    #     )
    #     check(self.trjryStream.popt[1])
    #     check(self.trjryStream.popt[2])
    #     # print('fit linewidth = ', self.trjryStream.popt[1])
    #     ax00.set_xlabel("frequency" + specxunit)
    #     ax00.set_ylabel("PSD")
    #     # ax00.set_xscale('log')
    #     # ax00.set_yscale('log')
    #     ax00.legend()
    #     ax00.set_xlim(self.RCF_freq_Hz - 10, self.RCF_freq_Hz + 10)
    #     plt.tight_layout()
    #     # plt.savefig('example figure - one-column.png', transparent=False)
    #     plt.show()