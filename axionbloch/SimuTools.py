"""NMR / Bloch-equation simulation tools.

Core classes
------------
MagField
    Build time-domain magnetic-field arrays (excitation pulses, CPMG sequences,
    axion pseudo-magnetic fields) ready to pass to the C++ Bloch solver.

Simulation
    Wrapper around the compiled ``blochsimulation`` extension that runs the
    Bloch equations and stores the resulting magnetisation trajectory.

Module-level constants
----------------------
RECORD_RUNTIME : bool
    When ``True``, elapsed wall-clock times are printed after long operations.
T_SETFIELD_S, T_SIMUSTEP_S : float
    Per-step runtime estimates (seconds) used to project total run time.
"""

import os
import time
import warnings

from axionbloch.dependency import *

from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D  # for type hinting

from scipy.stats import uniform, expon
from scipy.fft import ifft as sp_ifft

# for saving data
import pickle
import h5py

from axionbloch.utils import (
    PhysicalObject,
    check,
    save_phys_quantity,
    getDateAndTime,
    sci_fmt,
)

# from axionbloch.DataAnalysis import Signal
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Magnet
from axionbloch.SimuTypes import SimuParams, SimuEntry
from axionbloch.Station import Station
from axionbloch.MilkyWayAxionHalo import MilkyWayAxionHalo
from axionbloch.FineGrainedAxionStream import FineGrainedAxionStream

import axionbloch.blochsimulation as bs

RECORD_RUNTIME = True

# estimated run time for a single step in setField
T_SETFIELD_S = 1.2e-7
# estimated run time for a single simulation step
T_SIMUSTEP_S = 1.2e-09


def gate(x: float | np.ndarray, start: float, stop: float) -> np.ndarray:
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
    """Time-domain (pseudo)magnetic field builder for Bloch-equation simulations.

    Constructs the effective magnetic-field array that is passed to the C++
    RK4 solver.  Supported field types include:

    - Continuous XY excitation pulses (CW NMR)
    - Calibrated 90° and 180° hard pulses (pulsed NMR)
    - CPMG refocusing pulse trains (spin-echo sequences)
    - Stochastic axion pseudomagnetic fields (Milky Way SHM or fine-grained streams)

    After calling one of the ``set*`` methods the field is stored in
    :attr:`B_vec`, a NumPy array of shape ``(timeLen-1, 3)`` with units of T.
    """

    def __init__(
        self,
        name="B field",
    ):
        super().__init__()
        self.name = name
        self.nu_Hz = None

    def setXYPulse_old(
        self,
        timeStamp: np.ndarray,
        B1: float,  # amplitude of the excitation pulse in (T)
        nu_rot: float,
        init_phase: float,
        duty_func,
        verbose: bool = False,
    ):
        """
        generate a pulse in the rotating frame
        """
        warnings.warn(
            "setXYPulse_old is deprecated and will be removed in a future version. Use setXYPulse instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        # self.dBdt_vec = dBxdt + dBydt
        # sanCheck(Bx, tag="Bx")
        # sanCheck(self.B_vec, tag="self.B_vec")
        # self.dBdt_vec = np.outer(dBxdt + dBydt, direction)
        # self.nu = nu_rot
        # def envelope(timeStamp):
        #     return duty_func(timeStamp) * B1 * np.sin(2 * np.pi * nu_e * timeStamp + init_phase)
        # return

    def setXYPulse(
        self,
        timeStep: Quantity,
        timeLen: int,
        B1: Quantity,
        nu_rot: Quantity,
        init_phase: Quantity = 0.0 * unit.rad,
        verbose: bool = False,
    ):
        """Generate a continuous XY excitation pulse in the rotating frame.

        Fills the entire time window with a constant-envelope circularly-
        polarised pulse at frequency ``nu_rot`` and amplitude ``B1``.

        Parameters
        ----------
        timeStep : Quantity
            Simulation time step (e.g. ``1 * unit.ms``).
        timeLen : int
            Total number of time steps (including the final boundary point).
        B1 : Quantity
            Peak field amplitude (e.g. ``1e-11 * unit.T``).
        nu_rot : Quantity
            Carrier frequency in the rotating frame (e.g. ``2 * unit.Hz``).
        init_phase : Quantity
            Initial phase offset (e.g. ``0 * unit.rad``).
        verbose : bool
            Unused; reserved for future diagnostic output.
        """
        timeStamp = timeStep * np.arange(timeLen - 1)
        envelope = np.zeros(timeStamp.shape) * unit.T
        envelope[:] = 0.5 * B1

        Bx_envelope = envelope * np.cos(2 * PI * nu_rot * timeStamp + init_phase)
        Bx = np.outer(Bx_envelope, np.array([1, 0, 0]))

        By_envelope = envelope * np.sin(2 * PI * nu_rot * timeStamp + init_phase)
        By = np.outer(By_envelope, np.array([0, 1, 0]))

        self.B_vec = Bx + By

    def set90DegPulse(
        self,
        timeStep: Quantity,
        timeLen: int,
        gamma: Quantity,
        t90: Quantity,
        nu_rot: Quantity,
        init_phase: Quantity = 0.0 * unit.rad,
        verbose: bool = False,
    ):
        """Generate a calibrated 90° (π/2) pulse in the rotating frame.

        The pulse amplitude is chosen so that ``gamma * B90 * t90 = π``, giving
        an exact 90° flip.  The pulse occupies the first ``t90_s`` seconds of
        the time window; the remainder is zero.

        Parameters
        ----------
        timeStep_s : float
            Simulation time step (s).
        timeLen : int
            Total number of time steps.
        gamma : float
            Gyromagnetic ratio in Hz/T (2π factor).
        t90_s : float
            Desired 90° pulse duration (s).
        nu_rot_Hz : float
            Carrier frequency of the rotating frame (Hz).
        init_phase : float
            Initial phase offset (rad).
        verbose : bool
            Unused; reserved for future diagnostic output.
        """
        startDelayLen = 0  # this should stay 0. It does not help in anything
        timeStamp_s = timeStep * np.arange(timeLen - 1)
        t90Len = int(np.round(t90 / timeStep))
        B90 = 2 * 0.5 * PI / (gamma * t90)
        envelope = np.zeros(timeStamp_s.shape) * unit.T
        t90Len = int(np.round(t90 / (timeStamp_s[1] - timeStamp_s[0])))

        envelope[startDelayLen : startDelayLen + t90Len] = 0.5 * B90

        # excitation along x-axis
        Bx_envelope = np.multiply(
            envelope, np.cos(2 * PI * nu_rot * timeStamp_s + init_phase)
        )
        Bx = np.outer(Bx_envelope, np.array([1, 0, 0]))

        # excitation along y-axis
        By_envelope = np.multiply(
            envelope, np.sin(2 * PI * nu_rot * timeStamp_s + init_phase)
        )
        By = np.outer(By_envelope, np.array([0, 1, 0]))

        self.B_vec = Bx + By

    def setCPMGPulseTrain(
        self,
        timeStep: Quantity,
        timeLen: int,
        gamma: Quantity,
        t90: Quantity,
        tau: Quantity,
        numEcho: int,
        nu_rot: Quantity,
        init_phase: Quantity = 0.0 * unit.rad,
        verbose: bool = False,
    ):
        """Generate a CPMG (Carr-Purcell-Meiboom-Gill) pulse train in the rotating frame.

        Produces a π/2 pulse followed by ``numEcho`` refocusing π pulses
        separated by free-precession intervals of length ``tau``.  The
        envelope schematic uses the pulse-timing notation from the original
        CPMG paper:
          90deg pulse       180deg pulse                     180deg pulse
                             ┌───────┐                       ┌───────┐
            ┌───┐            |       |                       |       |
            ┘   └────────────┘       └───────────────────────┘       └──────
            ↑   ↑            ↑       ↑                       ↑       ↑
            0  t90          tau   tau+t180                  3tau   3tau+t180

        Parameters
        ----------
        timeStep : Quantity
            Simulation time step (e.g. ``1 * unit.ms``).
        timeLen : int
            Total number of time steps.
        gamma : Quantity
            Gyromagnetic ratio (e.g. ``gamma_p`` in ``unit.rad * unit.Hz / unit.T``).
        t90 : Quantity
            Desired 90° pulse duration; 180° pulses use the same duration.
        tau : Quantity
            Half-echo spacing: free precession time between π/2 and first π pulse.
        numEcho : int
            Number of spin echoes (= number of π pulses).
        nu_rot : Quantity
            Carrier frequency offset in the rotating frame (e.g. ``2 * unit.Hz``).
        init_phase : Quantity
            Initial phase offset for all pulses (e.g. ``0 * unit.rad``).
        verbose : bool
            Print diagnostic information (e.g. ``t90Len``).
        """
        timeStamp = timeStep * np.arange(timeLen - 1)
        t90Len = int(np.round(t90 / timeStep))
        if verbose:
            print(f"t90Len = {t90Len} time steps")
        if t90Len < 2:
            warnings.warn(f"t90Len = {t90Len} < 3.", UserWarning, stacklevel=2)
        t180Len = t90Len

        t90 = t90Len * timeStep  # snap to grid

        B90 = PI / (gamma * t90)
        B180 = 2.0 * B90

        tauLen = int(np.round(tau / timeStep))
        if tauLen < 10 * t180Len:
            warnings.warn(
                f"tauLen = {tauLen} < 10 * t180Len = {10 * t180Len}. Too short!",
                UserWarning,
                stacklevel=2,
            )
        Bx = np.zeros(timeStamp.shape) * unit.T
        By = np.zeros(timeStamp.shape) * unit.T

        t90Stamp = 2 * PI * nu_rot * timeStamp[0:t90Len] + init_phase
        t180Stamp = 2 * PI * nu_rot * timeStamp[0:t180Len] + init_phase

        Bx_90pulse = 0.5 * B90 * np.cos(t90Stamp)
        By_90pulse = 0.5 * B90 * np.sin(t90Stamp)
        Bx_180pulse = 0.5 * B180 * np.cos(t180Stamp)
        By_180pulse = 0.5 * B180 * np.sin(t180Stamp)

        Bx[0:t90Len] = Bx_90pulse
        By[0:t90Len] = By_90pulse

        for i in range(numEcho):
            if (1 + i * 2) * tauLen + t180Len >= len(timeStamp):
                break
            Bx[(1 + i * 2) * tauLen : (1 + i * 2) * tauLen + t180Len] = Bx_180pulse
            By[(1 + i * 2) * tauLen : (1 + i * 2) * tauLen + t180Len] = By_180pulse

        self.B_vec = np.zeros((1, len(Bx), 3)) * unit.T
        self.B_vec[0, :, 0] = Bx
        self.B_vec[0, :, 1] = By

    def setNPulsesArbDelay(
        self,
        timeStep: Quantity,
        timeLen: int,
        gamma: Quantity[unit.rad * unit.Hz / unit.T],
        pulseDur: Quantity,
        tip_angles: list[Quantity[unit.rad]],
        delays: list[Quantity[unit.s]],
        nu_rot: Quantity[unit.Hz],
        phases: list[Quantity] | None = None,
        verbose: bool = False,
    ):
        """Generate N hard pulses at arbitrary times in the rotating frame.

        All pulses share the same duration ``t90_s`` (snapped to the time
        grid) but each has an independently chosen flip angle and phase.
        The amplitude of pulse *i* is scaled so that
        ``gamma * 0.5 * B_i * t90_s = flip_angles_deg[i] * pi/180``.

        Timing convention: ``delays_s[i]`` is the free-precession interval
        between the *end* of pulse *i-1* (or the simulation start for i=0)
        and the *start* of pulse *i*.

        Parameters
        ----------
        timeStep_s : float
            Simulation time step (s).
        timeLen : int
            Total number of time steps (including the final boundary point).
        gamma : Quantity[unit.rad * unit.Hz / unit.T]
            Gyromagnetic ratio in rad·Hz/T (sign is used to set pulse sense).
        t90_s : float
            Duration of a 90° pulse (s); all pulses share this duration.
        flip_angles_deg : list of float
            Flip angle of each pulse in degrees (e.g. ``[90, 180, 180]``).
        delays_s : list of float
            Free-precession time before each pulse (s). ``delays_s[0]`` is
            the initial delay before pulse 0 (usually 0). Must satisfy
            ``len(delays_s) == len(flip_angles_deg)``.
        nu_rot_Hz : float
            Carrier frequency in the rotating frame (Hz).
        phases_rad : list of float or None
            Phase offset of each pulse (rad), setting the rotation axis.
            ``None`` defaults to all-zero phases (all pulses along x).
        verbose : bool
            Print per-pulse diagnostics.
        """

        N = len(tip_angles)
        if len(delays) != N:
            raise ValueError("delays_s and flip_angles_deg must have the same length")
        if phases is None:
            phases = [0.0 * unit.rad] * N
        if len(phases) != N:
            raise ValueError("phases_rad and flip_angles_deg must have the same length")

        pulseLen = int(np.round(pulseDur / timeStep))
        if pulseLen < 2:
            print(f"WARNING: pulseLen = {pulseLen} < 2")
        pulseDur = pulseLen * timeStep  # snap to grid
        if verbose:
            print(f"pulseLen = {pulseLen}, pulseDur = {pulseDur:.4e} s")

        B90 = PI / (gamma * pulseDur)

        numSteps = timeLen - 1
        Bx = np.zeros(numSteps) * unit.T
        By = np.zeros(numSteps) * unit.T

        # Local time template for one pulse window (phase resets at each pulse start,
        # so the rotation axis is determined solely by phases_rad[i]).
        local_t = timeStep * np.arange(pulseLen)

        current_idx = 0
        for i in range(N):
            current_idx += int(np.round(delays[i] / timeStep))

            tip_angle = tip_angles[i]
            B_i = (2.0 * tip_angle / (PI)) * B90
            phase_i = phases[i]

            end_idx = min(current_idx + pulseLen, numSteps)
            actual_len = end_idx - current_idx
            if actual_len <= 0:
                if verbose:
                    print(f"Pulse {i}: starts beyond simulation end, skipping")
                break

            stamp = 2 * PI * nu_rot * local_t[:actual_len] + phase_i
            Bx[current_idx:end_idx] += 0.5 * B_i * np.cos(stamp)
            By[current_idx:end_idx] += 0.5 * B_i * np.sin(stamp)

            if verbose:
                t_start = current_idx * timeStep
                print(
                    f"Pulse {i}: {tip_angles[i]:.1f}° at t={t_start:.4f} s, "
                    f"phase={np.degrees(phase_i):.1f}°, B_i={B_i:.4e} T"
                )

            current_idx += pulseLen

        self.B_vec = np.zeros((1, numSteps, 3)) * unit.T
        self.B_vec[0, :, 0] = Bx
        self.B_vec[0, :, 1] = By

    def setAxionFields(
        self,
        # method: str,  # 'inverse-FFT'
        axion: MilkyWayAxionHalo | FineGrainedAxionStream,
        timeStep_s: float,
        timeLen: int,
        simuRate: Quantity,
        duration: Quantity,
        # nu_a_rot_Hz: float,  # axion effective frequency in RCF
        use_stoch: bool,
        RCF_freq_Hz: float,
        numFields: int,
        rand_seed: int = None,
        B_a_rms_T: float = None,  # amplitude of the pseudo-magnetic field in (T)
        makePlot: bool = False,
        verbose: bool = False,
    ):
        """
        generate a pseudo-magnetic field (ALP field gradient)
        """
        self.numFields = numFields
        numSteps = timeLen - 1

        def setByInvFFT():
            """
            generate Bx, By, dBxdt, dBydt
            """
            frequencies = np.linspace(
                -0.5 / timeStep_s, 0.5 / timeStep_s, num=numSteps, endpoint=True
            )

            ampSpectra = axion.getAmpSpectra(
                frequencies=frequencies + RCF_freq_Hz * unit.Hz,
                case="grad_perp",
                numSpectra=numFields,
                use_stoch=use_stoch,
                rand_seed=rand_seed,
                verbose=verbose,
            )
            # amplitue spectra of axion fieds
            ax_AS: np.ndarray = (
                B_a_rms_T * unit.T * simuRate * np.sqrt(duration) * ampSpectra
            )

            freq = np.fft.fftfreq(numSteps, timeStep_s)  # shape = (numSteps)
            check(ampSpectra.unit)
            check(ax_AS.unit)
            check(freq.unit)

            if makePlot:
                fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
                gs = gridspec.GridSpec(
                    nrows=1, ncols=1
                )  # create grid for multiple figures
                axPSD = fig.add_subplot(gs[0, 0])

                axPSD.scatter(
                    frequencies,
                    np.abs(ampSpectra) ** 2,
                    # label="Average ALP-field gradient PSD",
                    # linestyle="--",zorder=3,
                )
                if use_stoch:
                    rand_lineshapes = np.abs(ampSpectra) ** 2
                    axPSD.errorbar(
                        x=frequencies,
                        y=rand_lineshapes.mean(axis=0),
                        yerr=rand_lineshapes.std(axis=0),
                        label="Stochastic ALP-field gradient PSD",
                        linestyle="-",
                    )
                else:
                    axPSD.plot(
                        frequencies,
                        np.abs(ampSpectra) ** 2,
                        label="Average ALP-field gradient PSD",
                        color="tab:orange",
                        linestyle="--",
                        zorder=3,
                    )
                axPSD.set_xlabel(f"Frequency - {RCF_freq_Hz:.0g} (Hz)")
                axPSD.set_ylabel("")
                axPSD.legend()
                plt.tight_layout()
                plt.show()

            # ifft_runtimes = []
            if verbose:
                tic = time.perf_counter()

            ax_AS_pos_neg: np.ndarray = np.fft.fftshift(ax_AS, axes=1)

            B_t = (
                np.asarray(sp_ifft(ax_AS_pos_neg.value, axis=1)) * ax_AS_pos_neg.unit
            )  # batch IFFT along time axis

            dBdt_FD: np.ndarray = 1j * 2 * np.pi * freq * ax_AS_pos_neg

            dBdt: np.ndarray = np.asarray(sp_ifft(dBdt_FD.value, axis=1)) * dBdt_FD.unit
            check(ax_AS_pos_neg.unit)
            check(B_t.unit)
            check(dBdt.unit)
            if verbose:
                toc = time.perf_counter()
                timeConsumption = toc - tic
                print(f"ifft total time consumption = {timeConsumption:.3e} s")

            if verbose:
                tic = time.perf_counter()

            self.B_vec = np.zeros((numFields, numSteps, 3)) * B_t.unit
            # self.dBdt_vec = np.zeros((numFields, numSteps, 3)) * dBdt.unit

            self.B_vec[:, :, 0] = B_t.real
            self.B_vec[:, :, 1] = B_t.imag
            # self.dBdt_vec[:, :, 0] = dBdt.real
            # self.dBdt_vec[:, :, 1] = dBdt.imag

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

        setByInvFFT()

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
    """Collection manager for one or more Bloch-equation simulation runs.

    Accepts a list of :class:`~axionbloch.SimuTypes.SimuParams` dictionaries,
    instantiates a :class:`Simulation` for each, and executes them
    sequentially.  Results are stored in :attr:`pool` as a list of
    :class:`~axionbloch.SimuTypes.SimuEntry` objects.

    Attributes
    ----------
    pool : list of SimuEntry
        Completed simulation entries (populated after :meth:`run`).
    all_params : list of SimuParams
        Parameter dictionaries passed at construction.

    Examples
    --------
    >>> simulations = Simulations(all_params=[params])
    >>> simulations.run(verbose=True)
    >>> simulations.saveToPkl(dir="results/")
    """

    def __init__(
        self,
        name: str = "NMR simulationS",
        all_params: list = None,
        verbose=True,
    ):
        """ """
        self.name = name
        self.pool: list[SimuEntry] = []
        self.all_params: list[SimuParams] = all_params

    def setup(self, verbose: bool = False):
        logPrefix = f"[{self.__class__.__name__}.{self.setup.__name__}]"
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

            RCF_freq: Quantity = axion.nu_a_eff

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
            self.pool.append(SimuEntry(simu=simu, params=params))

            if verbose:
                print("", flush=True)
                for key, pq in params["key_info"].items():
                    print(key, "=", pq, flush=True)
                print(
                    logPrefix,
                    f"simu.magnet.numPt =",
                    simu.magnet.numPt,
                    flush=True,
                )
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
                print(
                    logPrefix,
                    "Estimated step runtime = "
                    + f"{(t_setFields_s + t_trjry_s) / 60:.2g} min",
                    flush=True,
                )
        return est_runtime, est_setFields_s, est_trjry_s

    def run(self, autoStart: bool = True, verbose: bool = False):
        logPrefix = f"[{self.__class__.__name__}.{self.run.__name__}]"

        actu_runtime = 0.0
        actu_setFields_s = 0.0
        actu_trjry_s = 0.0
        # initialization
        est_runtime, est_setFields_s, est_trjry_s = self.setup(verbose=verbose)

        if not autoStart:
            print(
                "# ---------------------------------------------------- #", flush=True
            )
            print(
                logPrefix,
                f"Estimated setFields time = {est_setFields_s / 60.0:.3g} min",
                flush=True,
            )
            print(
                logPrefix,
                f"Estimated trjry time = {est_trjry_s / 60.0:.3g} min",
                logPrefix,
                f"Estimated trjry time = {est_trjry_s / 60.0:.3g} min",
                flush=True,
            )
            print(
                logPrefix,
                f"Estimated total runtime = {est_runtime / 60.0:.3g} min",
                logPrefix,
                f"Estimated total runtime = {est_runtime / 60.0:.3g} min",
                flush=True,
            )
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
            simu: Simulation = self.pool[i].simu
            # set fields
            tic = time.perf_counter()
            simu.excField.setAxionFields(
                axion=params["axion"],
                timeStep_s=simu.timeStep,
                timeLen=simu.timeLen,
                simuRate=simu.rate,
                duration=simu.duration,
                use_stoch=True,
                RCF_freq_Hz=simu.RCF_freq_Hz,
                numFields=params["numFields"],
                rand_seed=params["rand_seed"],
                B_a_rms_T=params["B_a_rms"].to_value(
                    unit.T
                ),  # rms amplitude of the pseudo-magnetic field in [T]
                makePlot=False,
                verbose=False,
            )
            toc = time.perf_counter()
            timeConsumption = toc - tic
            actu_setFields_s += timeConsumption
            if verbose:
                print("", flush=True)
                # for key, pq in params["key_info"].items():
                #     print(key, "=", pq, flush=True)
                print(
                    logPrefix,
                    f"time consumption = {timeConsumption:.6f} s = {timeConsumption/60:.1g} min",
                    flush=True,
                )
                print(
                    logPrefix,
                    f"individual step time consumption = {timeConsumption/(simu.numSteps+1)/simu.excField.numFields:.3e} s",
                    flush=True,
                )
            # ------------------------------------
            # print("simu.numSteps =", simu.numSteps, flush=True)
            # print("len(simu.excField.B_vec) =", (simu.excField.B_vec.shape), flush=True)

            # simu.excType = "ALP"

            # ------------------------------------
            tic = time.perf_counter()
            simu.generateTrajectories(cleanup=False, verbose=False)
            toc = time.perf_counter()
            timeConsumption = toc - tic
            actu_trjry_s += timeConsumption
            if verbose:
                print(
                    f"[{simu.__class__.__name__}.{simu.generateTrajectories.__name__}] time consumption = {timeConsumption:.2g} s = {timeConsumption/60:.1g} min",
                    flush=True,
                )
                print(
                    f"[{simu.__class__.__name__}.{simu.generateTrajectories.__name__}] individual step time consumption = {timeConsumption/simu.numSteps/simu.magnet.numPt/simu.excField.numFields:.2e} s",
                    flush=True,
                )
            # ------------------------------------
            # simu.keepMeanStd(debug=verbose)

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
        ymd_hms = getDateAndTime()

        for time_str, pool_item in self.pool.items():
            data_file_name = dir + ymd_hms
            data_file_name += "-simulation"
            for key, pq in pool_item["params"]["key_info"].items():
                data_file_name += "-"
                data_file_name += key
                data_file_name += f"={pq.value:g}" + pq.unit
            pool_item.simu.saveToH5(path=data_file_name)

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
            if self.name is not None:
                fname = self.name + "_" + getDateAndTime()
            else:
                fname = "simulations_" + getDateAndTime()

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
        #     pool_item.simu.saveToH5(path=data_file_name)

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
    """Single Bloch-equation simulation run in the rotating coordinate frame (RCF).

    Wraps the C++ ``blochsimulation`` extension.  Generates stochastic axion
    pseudomagnetic fields via :meth:`~axionbloch.SimuTools.MagField.setAxionFields`,
    loops over ``numFields`` field realisations and ``Magnet.numPt`` spin
    packets, integrates the Bloch equations with RK4, and stores the
    resulting magnetisation trajectories in :attr:`trjries`.

    Key attributes
    --------------
    rate : Quantity [Hz]
        Simulation (output) sampling rate.
    duration : Quantity [s]
        Total simulation duration.
    numSteps : int
        Number of RK4 integration steps (= rate × duration).
    trjries : ndarray, shape (numFields, numSteps, 3)
        Magnetisation trajectories for each field realisation.
    """

    def __init__(
        self,
        name: str | None = "NMR simulation",
        axion: MilkyWayAxionHalo | None = None,
        sample: Sample | None = None,
        magnet: Magnet | None = None,
        excField: MagField | None = None,
        station: Station | None = None,
        init_M: Quantity | None = None,  # initial magnetization vector amplitude
        init_M_theta: Quantity | None = 0 * unit.rad,
        init_M_phi: Quantity | None = 0 * unit.rad,
        RCF_freq: Quantity[unit.Hz] | None = None,
        rate: Quantity | None = None,  # simulation rate
        duration: Quantity | None = None,  # simulation duration
        verbose: bool = False,
    ):
        # TODO: separate init and setup
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
        logPrefix = f"[{self.__class__.__name__}.{self.__init__.__name__}]"
        super().__init__()
        self.quantities = {
            "RCF_freq": "Hz",
            "rate": "Hz",
            "duration": "s",
            "init_M": "",
            "init_M_theta": "rad",
            "init_M_phi": "rad",
        }

        self.name = name

        # if None in [sample, magnet, excField]:
        #     raise ValueError("sample, magnet and excField must not be None")

        # Instances
        self.sample = sample
        self.magnet = magnet
        self.axion = axion
        self.station = station
        self.excField = excField

        # self.gamma_HzToT = np.abs(
        #     self.sample.gamma.to_value(unit.rad * unit.Hz / unit.T)
        # )

        # get the equilibrium magnetization M0
        self.M0eqb = self.sample.getM0eqb(B_pol=self.magnet.B0, verbose=verbose)
        # normalize the magnetization by the equilibrium magnetization M0, so that init_M is dimensionless and represents the initial polarization

        if self.sample.pol is not None:
            if init_M is not None:
                print(
                    logPrefix,
                    "WARNING: init_M is provided, but sample polarization is set. The provided init_M will be ignored and overwritten by the value determined from the sample polarization.",
                    flush=True,
                )
            # find the M0 by polarization
            # assert self.sample.pol is not None, "sample polarization is not set, cannot determine M0"
            M0_init = self.sample.getM0(verbose=verbose)
            init_M = M0_init / self.M0eqb
        elif init_M is None:
            init_M = (
                1.0 * unit.dimensionless_unscaled
            )  # default to fully polarized if neither init_M nor sample polarization is provided

        self.init_M = init_M
        self.init_M_theta = init_M_theta
        self.init_M_phi = init_M_phi

        self.init_M_amp = init_M.to_value(unit.dimensionless_unscaled)
        self.init_M_theta_rad = init_M_theta.to_value(unit.rad)
        self.init_M_phi_rad = init_M_phi.to_value(unit.rad)

        self.B0z_T = np.abs(self.magnet.B0).to_value(unit.T)

        # TODO：be more smart in setting rate and duration
        # set rotating frame frequency
        if RCF_freq is None or type(RCF_freq) != Quantity:
            self.RCF_freq = np.abs(self.sample.gamma / (2 * PI) * self.magnet.B0).to(
                unit.MHz
            )
        else:
            self.RCF_freq = RCF_freq.to(unit.MHz)
        self.RCF_freq_Hz = self.RCF_freq.to_value(unit.Hz)

        self.nu_L_Hz = (
            abs(self.sample.gamma.value * self.B0z_T / (2 * np.pi)) - self.RCF_freq_Hz
        )  # nuL_Hz is the Larmor frequency of the magnetization in the rotating frame

        self.T2_s = sample.T2.to_value(unit.s)
        self.T1_s = sample.T1.to_value(unit.s)

        self.trjry = None

        # ---------------------------- set duration ----------------------------#
        FWHM_freq: Quantity[unit.Hz] = np.abs(
            self.magnet.FWHM_B0 * sample.gamma / (2 * PI)
        )
        # find Tdelta
        self.Tdelta = 1 / (np.pi * FWHM_freq)
        self.T2star = (1.0 / (1.0 / self.Tdelta + 1 / self.sample.T2)).si
        if duration is None:
            if self.axion is not None:
                self.duration_s = np.amin(
                    [
                        3.2e3 * self.axion.tau_a_est.to_value(unit.s),
                        1.5e2 * self.T2star.to_value(unit.s),
                    ]
                )
            else:
                self.duration_s = 2e2 * self.T2star.to_value(unit.s)
            self.duration = self.duration_s * unit.s
        else:
            self.duration = duration
            self.duration_s = self.duration.to_value(unit.s)
        # ---------------------------------------------------------------------------- #

        # -----------------------------set magnet--------------------------------------- #
        # estimate the necessary data points for sampling the inhomogeneity
        numPt = abs(
            (self.duration * 2 * magnet.B0_nW * sample.gamma / (2 * PI)).to_value(
                unit.dimensionless_unscaled
            )
        )
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
        T2star_relaxation_rate = 1 / self.T2star
        if self.axion is not None:
            axion_decoherence_rate_Hz = 1 / self.axion.tau_a_est.to_value(unit.s)
        else:
            axion_decoherence_rate_Hz = 0.0
        nuL_vals_Hz = (
            np.abs(self.sample.gamma) / (2 * PI) * magnet.B_spread - self.RCF_freq
        ).to_value(unit.Hz)
        nuL_Hz_abs_max = np.amax(np.abs(nuL_vals_Hz))
        nu_Hz_range = np.amax(nuL_vals_Hz) - np.amin(nuL_vals_Hz)
        if rate is None:
            rate_Hz: float = np.amax(
                [
                    50.0 * nu_Hz_range,
                    50.0 * nuL_Hz_abs_max,
                    # 30.0 * nuL_Hz_abs_max * self.duration_s / (1e2 * self.T2star_s),
                    100.0 * T2star_relaxation_rate.to(unit.Hz),
                    100.0 * axion_decoherence_rate_Hz,
                ]
            )
            rate = rate_Hz * unit.Hz
        self.setRate(rate=rate)
        # -------------------------------------------------------------------------#

        # ----- check if parameter values are within reasonable range -----#
        assert self.numSteps >= 5, "number of steps < 5"
        # numPt >= 2 pi * Delta_nu * duration
        variation = (
            self.magnet.nFWHM
            * self.magnet.FWHM_B0
            * np.abs(self.sample.gamma)
            / (2 * PI)
            * self.duration
        )
        if self.magnet.numPt < 2 * variation.to_value(unit.one):
            warnings.warn(
                f"{self.__init__.__name__}: magnet_det.numPt may be too few.",
                UserWarning,
                stacklevel=2,
            )

        # simulation rate must be 20 times larger than maximum Larmor frequency
        # from experience, simulation rate should be 20 times greater than the max. of signal frequency in the rotating frame
        if self.rate_Hz < 20 * nuL_Hz_abs_max:
            warnings.warn(
                f"{logPrefix}: the simulation rate ({self.rate_Hz:g} Hz) might be too small compared to signal frequency ({nuL_Hz_abs_max:g} Hz).",
                UserWarning,
                stacklevel=2,
            )

        if self.sample.T2 > self.sample.T1:
            warnings.warn(
                f"{logPrefix}: T2 is larger than T1.", UserWarning, stacklevel=2
            )

        if self.rate <= 10 * (1.0 / self.sample.T2):
            warnings.warn(
                f"{logPrefix}: the simulation rate ({self.rate_Hz:g} Hz) might be too small compared to T2 relaxation rate ({1.0 / self.T2_s:g} Hz).",
                UserWarning,
                stacklevel=2,
            )
        if self.rate_Hz <= 1 / self.duration_s:
            warnings.warn(
                f"{logPrefix}: the simulation rate ({self.rate_Hz:g} Hz) might be too small so there are only {self.numSteps:d} step(s) in the duration of {self.duration_s:g} s.",
                UserWarning,
                stacklevel=2,
            )
        # ----- ----------------------------------------------------- -----#

    def setRate(self, rate: Quantity):
        logPrefix = f"[{self.__class__.__name__}.{self.setRate.__name__}]"
        assert rate is not None, f"{logPrefix} rate is None"
        self.rate = rate
        self.rate_Hz = float(rate.to_value(unit.Hz))
        self.timeStep = (
            1.0 / self.rate
        )  # the key parameter in setting simulation timing
        self.timeLen: int = int(np.ceil(self.duration_s * self.rate_Hz))
        self.numSteps: int = self.timeLen - 1
        if self.numSteps > 1e8:
            warnings.warn(
                f"{logPrefix}: Simulation.numSteps = {self.numSteps:.1e} > 1e8.",
                UserWarning,
                stacklevel=2,
            )
            # return
        # self.timeStamp_s = np.arange(start=0, stop=(self.timeLen) * self.timeStep_s, step=self.timeStep_s)

    def suggestRate(self, verbose: bool = False):
        # ----- check if parameter values are within reasonable range -----#

        # compute the maximum of (absolute) Larmor frequencies
        nuL_Hz_max = (
            self.sample.gamma.value() / (2 * np.pi) * self.magnet.B_spread.max()
            - self.RCF_freq_Hz
        )
        nuL_Hz_min = (
            self.sample.gamma.value() / (2 * np.pi) * self.magnet.B_spread.min()
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
        return rate_Hz * unit.Hz

    def getTimeStamp(self):
        return np.linspace(
            start=0, stop=(self.timeLen) * self.timeStep, num=self.timeLen
        )

    def estimateRuntime(self, verbose: bool = False) -> Quantity[unit.s]:
        logPrefix = f"[{self.__class__.__name__}.{self.estimateRuntime.__name__}]"
        t_trjry = (
            T_SIMUSTEP_S
            * self.numSteps
            * self.magnet.numPt
            * self.excField.B_vec.shape[0]
        ) * unit.s
        if verbose:
            print(
                logPrefix,
                "Estimated runtime = " + f"{t_trjry:g}",
                flush=True,
            )
        return t_trjry

    def generateTrajectories(
        self, cleanup: bool = False, integrator="RK4", verbose: bool = False
    ):
        """
        Generate trajectory of magnetization vector in Cartesian coordinate system
        based on kinetic simulation for Bloch equations.
        Parameters
        ----------
        cleanup : bool, optional
            whether to delete the intermediate variables after generating trajectories, by default False
        integrator : str, optional
            the numerical integrator to use, by default "RK4". Another option is "taylor"
        verbose : bool, optional
            whether to print time consumption, by default False
        Returns
        -------
        None
            The generated trajectories are stored in self.trjry with shape (numFields, numSteps
        """
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
        # normalized magnetization at equilibrium
        M0eqb_norm = 1.0

        # self.excField.B_vec and self.excField.dBdt_vec should have shape (numFields, numTimeSteps, 3)
        # if they are 1D arrays of shape (numSteps, 3),
        # reshape them to (1, numSteps, 3) so that they can be used in the loop without error
        if self.excField.B_vec.ndim == 2 and self.excField.B_vec.shape[1] == 3:
            self.excField.B_vec = self.excField.B_vec[np.newaxis, :, :]
        elif self.excField.B_vec.ndim != 3 or self.excField.B_vec.shape[2] != 3:
            raise ValueError(
                f"excField.B_vec has invalid shape {self.excField.B_vec.shape}, expected (numFields, numSteps, 3) or (numSteps, 3)"
            )

        # Use the kinetic simulation function from blochsimulation to generate trajectories
        self.trjry, self.dMdt, self.d2Mdt2 = bs.generateTrajectories(
            self.excField.B_vec.to_value(unit.T),
            # self.excField.dBdt_vec.to_value(unit.T / unit.s),
            self.magnet.B_spread.to_value(unit.T),
            self.magnet.ratios,
            np.abs(self.sample.gamma).to_value(unit.rad * unit.Hz / unit.T),
            self.timeStep.to_value(unit.s),
            self.sample.T1.to_value(unit.s),
            self.sample.T2.to_value(unit.s),
            self.RCF_freq_Hz,
            Mx0,
            My0,
            Mz0,
            M0eqb_norm,
            integrator,
        )
        if cleanup:
            del self.dMdt, self.d2Mdt2

    def cleanup(self):
        for attr in ["trjry", "dMdt", "d2Mdt2"]:
            if hasattr(self, attr):
                delattr(self, attr)

    def monitorTrajectories(
        self,
        plotRate: float = None,  #
        verbose: bool = False,
    ):
        warnings.warn(
            "monitorTrajectories is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        if numFields > 1 and len(self.trjry) > 1:
            self.trjry_mean = self.trjry.mean(axis=0)
            self.Mabs_mean = np.sqrt(
                self.trjry[:, :, 0] ** 2 + self.trjry[:, :, 1] ** 2
            ).mean(axis=0)
            self.M_std = np.sqrt(
                self.trjry[:, :, 0] ** 2 + self.trjry[:, :, 1] ** 2
            ).std(axis=0)
            magfield = np.concatenate(
                (
                    self.excField.B_vec.mean(axis=0),
                    [self.excField.B_vec.mean(axis=0)[-1]],
                ),
                axis=0,
            )
        else:
            self.trjry_mean = self.trjry[0]
            self.Mabs_mean = np.sqrt(
                self.trjry[0, :, 0] ** 2 + self.trjry[0, :, 1] ** 2
            )
            self.M_std = None
            magfield = np.concatenate(
                (self.excField.B_vec[0], [self.excField.B_vec[0][-1]]),
                axis=0,
            )
        BfieldTimeStamp_s = self.timeStep * np.arange(magfield.shape[0])
        timeStamp_s = self.timeStep * np.arange(self.Mabs_mean.shape[0])

        fig = plt.figure(figsize=(15 * 0.8, 7 * 0.8), dpi=150)  #
        gs = gridspec.GridSpec(nrows=2, ncols=4)  #
        # fix the margins
        left = 0.083
        bottom = 0.1
        right = 0.985
        top = 0.924
        wspace = 0.313
        hspace = 0.271
        fig.subplots_adjust(
            left=left, top=top, right=right, bottom=bottom, wspace=wspace, hspace=hspace
        )

        Ba_ax = fig.add_subplot(gs[0, 0])
        Mabs_ax = fig.add_subplot(gs[1, 0], sharex=Ba_ax)
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
            BfieldTimeStamp_s[0:lastIndx:plotIntv],
            magfield[0:lastIndx:plotIntv, 0],
            label="$B_{x}$",
            color="tab:blue",
            alpha=0.7,
        )
        Ba_ax.plot(
            BfieldTimeStamp_s[0:lastIndx:plotIntv],
            magfield[0:lastIndx:plotIntv, 1],
            label="$B_{y}$",
            color="tab:orange",
            alpha=0.7,
        )
        Ba_ax.plot(
            BfieldTimeStamp_s[0:lastIndx:plotIntv],
            magfield[0:lastIndx:plotIntv, 2],
            label="$B_{z}$",
            color="tab:green",
            alpha=0.7,
        )

        Ba_ax.set_ylabel("Magnetic field (T)")  # $B_\\mathrm{exc}$
        Ba_ax.set_xlabel("time (s)")
        Ba_ax.legend(loc="upper right")

        Mabs_ax.plot(
            timeStamp_s[0:lastIndx:plotIntv],
            self.Mabs_mean[0:lastIndx:plotIntv],
            label="mean",
            color="tab:orange",
            alpha=1,
            zorder=7,
        )
        if self.M_std is not None:
            Mabs_ax.errorbar(
                x=timeStamp_s[0:lastIndx:plotIntv],
                y=self.Mabs_mean[0:lastIndx:plotIntv],
                yerr=self.M_std[0:lastIndx:plotIntv],
                label="standard deviation",
                color="tab:blue",
                alpha=1,
            )

        Mabs_ax.set_ylabel("$M_{xy}$")
        Mabs_ax.set_xlabel("time (s)")
        Mabs_ax.legend(loc="upper right")

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
            self.Mabs_mean[0:lastIndx:plotIntv],
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
        # Mz_ax.plot(
        #     timeStamp_s[0:lastIndx:plotIntv],
        #     self.trjry_mean[0:lastIndx:plotIntv, 2]-np.cos(2 * np.pi * (1/0.08) * timeStamp_s[0:lastIndx:plotIntv]),
        #     label="cosine line - ",
        #     color="k",
        # )
        Mz_ax.scatter(
            timeStamp_s[0:lastIndx:plotIntv],
            self.trjry_mean[0:lastIndx:plotIntv, 2],
            # label="$M_z$",
            color="tab:pink",
        )
        Mz_ax.legend(loc="upper right")
        Mz_ax.grid()
        Mz_ax.set_xlabel("time (s)")
        Mz_ax.set_ylabel("")
        # Mz_ax.set_ylim(top=1.1)

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
        # g_aNN={self.excField.g_aNN:.0e} axion_nu={self.excField.nu:.1e}\nXe
        # print(f'TrajectoryMonitoring_g_aNN={self.ALPwind.g_aNN:.0e}_axion_nu={self.ALPwind.nu:.1e}_Xe_T2={self.T2:.1g}s_T1={self.T1:.1e}s')
        # plt.tight_layout()
        plt.show()

    def keepMeanStd(self, debug: bool = False):
        """
        Keep the mean values and standard deviations of the results
        """
        self.excField.B_vec_mean = self.excField.B_vec.mean(axis=0)
        self.excField.B_vec_std = self.excField.B_vec.std(axis=0)
        # self.excField.B_vec.shape = (numFields, numSteps, 3)
        # self.excField.B_vec_mean.shape = (numSteps, 3)

        Mxy_magnitudes: np.ndarray = np.sqrt(
            self.trjry[:, :, 0] ** 2 + self.trjry[:, :, 1] ** 2
        )
        # Mx_magnitudes: np.ndarray = np.abs(self.trjry[:, :, 0])
        # My_magnitudes: np.ndarray = np.abs(self.trjry[:, :, 1])
        # Mz_magnitudes: np.ndarray = np.abs(self.trjry[:, :, 2])
        self.M_mean = self.trjry.mean(axis=0)
        self.M_std = self.trjry.std(axis=0)

        Mx: np.ndarray = self.trjry[:, :, 0]
        My: np.ndarray = self.trjry[:, :, 1]
        Mz: np.ndarray = self.trjry[:, :, 2]

        # mean of root of squares
        self.Mxy_mrs = Mxy_magnitudes.mean(axis=0)
        self.Mxy_srs = Mxy_magnitudes.std(axis=0)

        # square root of mean of squares
        self.Mxy_rms = np.sqrt((Mxy_magnitudes**2).mean(axis=0))
        self.Mxy_rss = np.sqrt((Mxy_magnitudes**2).std(axis=0))

        # # mean and std of each component
        # self.Mx_mean = Mx.mean(axis=0)
        # self.Mx_std = Mx.std(axis=0)

        # self.My_mean = My.mean(axis=0)
        # self.My_std = My.std(axis=0)

        # self.Mz_mean = Mz.mean(axis=0)
        # self.Mz_std = Mz.std(axis=0)
        if not debug:
            del (
                self.excField.B_vec,
                self.trjry,
                Mxy_magnitudes,
                Mx,
                My,
                Mz,
            )

    def displayTrjries(
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

        timeStamp = np.linspace(
            start=0, stop=(self.timeLen) * self.timeStep.si, num=len(self.M_mean[:, 0])
        )
        fig = plt.figure(figsize=(17 / 2.54, 9 / 2.54), dpi=300)  #
        gs = gridspec.GridSpec(nrows=2, ncols=3)  #
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
        B_t_ax = fig.add_subplot(gs[0, 0])
        # B_f_ax = fig.add_subplot(gs[1, 0])
        Mx_ax = fig.add_subplot(gs[0, 1], sharex=B_t_ax)
        My_ax = fig.add_subplot(gs[0, 2], sharex=Mx_ax, sharey=Mx_ax)
        Mxy_ax = fig.add_subplot(gs[1, 1], sharex=Mx_ax, sharey=Mx_ax)
        Mz_ax = fig.add_subplot(gs[1, 2], sharex=Mx_ax)
        M_axes = [Mx_ax, My_ax, Mz_ax, Mxy_ax]
        all_axes = [B_t_ax, Mx_ax, My_ax, Mz_ax, Mxy_ax]
        lastIndx = -1
        B_ax_labels = ["x", "y", "z", "freq"]
        M_ax_labels = ["x", "y", "z", "xy"]
        colorSets = np.array(
            [
                ["tab:orange", "tab:blue"],
                ["tab:red", "tab:blue"],
                ["tab:green", "tab:blue"],
                ["tab:pink", "tab:blue"],
            ]
        )
        # Sets = [["tab:orange", "tab:blue"],["tab:red", "tab:blue"],["tab:green", "tab:blue"],["tab:black", "tab:blue"]]
        B_mean = [
            self.excField.B_vec_mean[:, 0],
            self.excField.B_vec_mean[:, 1],
            self.excField.B_vec_mean[:, 2],
        ]
        B_std = [
            self.excField.B_vec_std[:, 0],
            self.excField.B_vec_std[:, 1],
            self.excField.B_vec_std[:, 2],
        ]
        M_mean = [self.M_mean[:, 0], self.M_mean[:, 1], self.M_mean[:, 2], self.Mxy_mrs]
        M_std = [self.M_std[:, 0], self.M_std[:, 1], self.M_std[:, 2], self.Mxy_srs]

        # Force scientific notation on ticks, not as offset
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))  # control when to switch to scientific
        for i, axis in enumerate(["x", "y", "z"]):
            # check(timeStamp_s.shape)
            # check(B_mean[i].shape)
            # lastIndx = min(len(timeStamp_s), len(B_mean[i])) # I do not suggest to use this one..
            lastIndx = len(B_mean[i])
            B_t_ax.errorbar(
                x=timeStamp[0:lastIndx:plotIntv],
                y=B_mean[i][0:lastIndx:plotIntv],
                yerr=B_std[i][0:lastIndx:plotIntv],
                label="$B_{" + axis + "}$",
                color=colorSets[i, 1],
                alpha=1,
            )
            B_t_ax.plot(
                timeStamp[0:lastIndx:plotIntv],
                B_mean[i][0:lastIndx:plotIntv],
                label="$B_{" + axis + "}$",
                color=colorSets[i, 0],
                alpha=1,
                zorder=3,
            )
        B_t_ax.set_ylabel(
            f"Excitation field ({B_mean[0][0].unit.to_string('unicode')})"
        )
        # TODO plot frequency domain in B_f_ax

        for i, ax in enumerate(M_axes):
            ax.errorbar(
                x=timeStamp[0:lastIndx:plotIntv],
                y=M_mean[i][0:lastIndx:plotIntv],
                yerr=M_std[i][0:lastIndx:plotIntv],
                label="$M_{" + M_ax_labels[i] + "}$",
                color=colorSets[i, 1],
                alpha=1,
            )
            ax.plot(
                timeStamp[0:lastIndx:plotIntv],
                M_mean[i][0:lastIndx:plotIntv],
                label="$M_{" + M_ax_labels[i] + "}$",
                color=colorSets[i, 0],
                alpha=1,
                zorder=3,
            )
            # ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))

        for ax in all_axes:
            ax.set_xlabel(f"Time ({timeStamp.unit.to_string('unicode')})")
            ax.set_ylabel("")
            ax.grid()
            ax.legend(loc="upper right")

        Mz_ax.set_ylim(bottom=Mxy_ax.get_ylim()[0])

        fig.suptitle(
            f"Magnet {self.magnet.B0:g} {self.magnet.FWHM.to(ppm):g}"
            + f"\nSample {self.sample.name} T1={self.sample.T1:g} T2={self.sample.T2.si:g} T2*={self.T2star.si:g}"
        )
        plt.tight_layout()
        plt.show()
        return M_axes

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

        timeStamp = np.linspace(
            start=0, stop=(self.timeLen) * self.timeStep.si, num=len(self.M_mean[:, 0])
        )
        fig = plt.figure(figsize=(17 / 2.54, 9 / 2.54), dpi=300)  #
        gs = gridspec.GridSpec(nrows=2, ncols=3)  #
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
        B_t_ax = fig.add_subplot(gs[0, 0])
        # B_f_ax = fig.add_subplot(gs[1, 0])
        Mx_ax = fig.add_subplot(gs[0, 1], sharex=B_t_ax)
        My_ax = fig.add_subplot(gs[0, 2], sharex=Mx_ax, sharey=Mx_ax)
        Mxy_ax = fig.add_subplot(gs[1, 1], sharex=Mx_ax, sharey=Mx_ax)
        Mz_ax = fig.add_subplot(gs[1, 2], sharex=Mx_ax)
        M_axes = [Mx_ax, My_ax, Mz_ax, Mxy_ax]
        all_axes = [B_t_ax, Mx_ax, My_ax, Mz_ax, Mxy_ax]
        lastIndx = -1
        B_ax_labels = ["x", "y", "z", "freq"]
        M_ax_labels = ["x", "y", "z", "xy"]
        colorSets = np.array(
            [
                ["tab:orange", "tab:blue"],
                ["tab:red", "tab:blue"],
                ["tab:green", "tab:blue"],
                ["tab:pink", "tab:blue"],
            ]
        )
        # Sets = [["tab:orange", "tab:blue"],["tab:red", "tab:blue"],["tab:green", "tab:blue"],["tab:black", "tab:blue"]]
        B_mean = [
            self.excField.B_vec_mean[:, 0],
            self.excField.B_vec_mean[:, 1],
            self.excField.B_vec_mean[:, 2],
        ]
        B_std = [
            self.excField.B_vec_std[:, 0],
            self.excField.B_vec_std[:, 1],
            self.excField.B_vec_std[:, 2],
        ]
        M_mean = [self.M_mean[:, 0], self.M_mean[:, 1], self.M_mean[:, 2], self.Mxy_mrs]
        M_std = [self.M_std[:, 0], self.M_std[:, 1], self.M_std[:, 2], self.Mxy_srs]

        # Force scientific notation on ticks, not as offset
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))  # control when to switch to scientific
        for i, axis in enumerate(["x", "y", "z"]):
            lastIndx = len(B_mean[i])
            B_t_ax.plot(
                timeStamp[0:lastIndx:plotIntv],
                B_mean[i][0:lastIndx:plotIntv],
                label="$B_{" + axis + "}$",
                color=colorSets[i, 0],
                alpha=1,
                zorder=3,
            )
        B_t_ax.set_ylabel(
            f"Excitation field ({B_mean[0][0].unit.to_string('unicode')})"
        )

        # TODO plot frequency domain in B_f_ax

        for i, ax in enumerate(M_axes):
            ax.plot(
                timeStamp[0:lastIndx:plotIntv],
                M_mean[i][0:lastIndx:plotIntv],
                label="$M_{" + M_ax_labels[i] + "}$",
                color=colorSets[i, 0],
                alpha=1,
                zorder=3,
            )
            # ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))

        for ax in all_axes:
            ax.set_xlabel(f"Time ({timeStamp.unit.to_string('unicode')})")
            ax.set_ylabel("")
            ax.grid()
            ax.legend(loc="upper right")

        Mz_ax.set_ylim(bottom=Mxy_ax.get_ylim()[0])

        fig.suptitle(
            f"Magnet {self.magnet.B0:g} {self.magnet.FWHM.to(ppm):g}"
            + f"\nSample {self.sample.name} T1={self.sample.T1:g} T2={self.sample.T2.si:g} T2*={self.T2star.si:g}"
        )
        plt.tight_layout()
        plt.show()
        return M_axes

    def visualizeTrajectory3D(
        self,
        plotrate: float = None,  # [Hz]
        # rotframe=True,
        verbose=False,
    ):
        if plotrate is None:
            plotrate = self.rate_Hz

        if plotrate > self.rate_Hz:
            print(
                "WARNING: plotrate > self.simurate. plotrate will be decreased to simurate"
            )
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
        # TODO remove this
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

    # def statTrajectories(self, verbose=False):

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
            print("self.quantities = ", self.quantities)
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
