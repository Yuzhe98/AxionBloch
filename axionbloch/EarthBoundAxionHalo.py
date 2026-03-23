import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import correlate as correlate
from axionbloch.enphylope import PhysicalQuantity as PQ
from axionbloch.constants import c, h_Planck
from axionbloch.utils import PhysicalObject, check


class EarthBoundAxionHalo(PhysicalObject):
    # Create the "axion stream" (axion field) object
    # you can get properties of the axion field, computed based on the input information
    def __init__(
        self,
        name="axion stream",
    ):
        self.name = name

    def getBfield(self, rate_Hz: float, timeLen: int, rand_seed: int, numFields:int=1, verbose:bool=False):
        # 1 MHz axion
        eigenEnergies_eV = [
            8.185620266405553e-19,
            2.230884526115819e-18,
            3.17699654698912e-18,
            3.3025178098192375e-18,
            3.302688446107381e-18,
            3.8484585622919455e-18,
            4.1004249795459864e-18,
            4.1004251503701215e-18,
            4.341652193125044e-18,
            4.407871816010306e-18,
            4.408009935951216e-18,
            4.649824763113033e-18,
            4.793034098628449e-18,
            4.7930342283222715e-18,
            4.8831463914209464e-18,
            4.919259056702106e-18,
            4.919342580168352e-18,
            5.042907628381992e-18,
            5.124371776790224e-18,
            5.124371863603223e-18,
            5.1684784792208225e-18,
            5.1897014793057166e-18,
            5.189752636536869e-18,
            5.260382049853425e-18,
            5.310111086559351e-18,
            5.310111144839848e-18,
            5.335147509172691e-18,
            5.348511363425495e-18,
            5.348544297168566e-18,
            5.3924738060527425e-18,
            5.424802508063358e-18,
            5.4248025483118064e-18,
            5.440435917665003e-18,
            5.449341478496205e-18,
            5.449363729093159e-18,
            5.478480245818686e-18,
            5.500595123173712e-18,
            5.5005951518592556e-18,
            5.511033438087251e-18,
            5.517245349489185e-18,
            5.517261015401705e-18,
            5.537528824223334e-18,
            5.553289963907005e-18,
            5.553289984958548e-18,
            5.560615718245721e-18,
            5.565112331351988e-18,
            5.565123747711691e-18,
            5.579789494642934e-18,
            5.591403811323711e-18,
            5.591403827177387e-18,
        ]
        self.rate_Hz:float = rate_Hz
        self.timeLen:int = timeLen
        self.duration_s:float = self.timeLen / self.rate_Hz
        eigenEnergies_eV = np.asarray(eigenEnergies_eV)
        eigenFreqs_Hz:np.ndarray = eigenEnergies_eV / h_Planck.value_in("eV * s")
        if verbose:
            print(eigenFreqs_Hz.mean())
        B_rms_T = 1e-15
        self.Ba = np.zeros(timeLen)
        timeStamp = np.arange(timeLen) / rate_Hz
        rng = (
            np.random.default_rng(seed=rand_seed)
            if rand_seed is not None
            else np.random.default_rng()
        )

        # phases
        phases = (
            2 * np.pi * rng.random((len(eigenFreqs_Hz), numFields))
        )  # shape: (numFreqs, numFields)

        phase_time:np.ndarray = 2 * np.pi * eigenFreqs_Hz[:, None] * timeStamp[None, :]  # shape: (numFreqs, timeLen)

        total_phase = phase_time[:, :, None] + phases[:, None, :]
        # shape: (numFreqs, timeLen, numFields)

        self.Ba = np.exp(1j * total_phase).sum(axis=0) # shape: (timeLen, numFields)
        self.Ba *= B_rms_T
        if verbose:
            print("phases.shape", phases.shape)
            print("phase_time.shape", phase_time.shape)
            print("total_phase.shape", total_phase.shape)
            print("self.Ba.shape", self.Ba.shape)
        return self.Ba

    def coh_time_g1(self):
        """
        x : complex-valued time series
        dt: sampling interval
        method: "1e" or "integral"
        """
        x = self.Ba[:, 0] - np.mean(self.Ba[:, 0])
        dt = 1 / self.rate_Hz
        E = np.array(x)  # complex field

        N = len(E)

        # tic = time.time()
        corr= correlate(E, E.conj(), mode="full")
        # toc = time.time()
        # print(f"Time taken for correlation: {toc - tic:.3f} seconds")
        fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
        gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures
        ax00 = fig.add_subplot(gs[0, 0])
        ax00.plot(np.abs(corr), label="")
        ax00.set_xlabel("")
        ax00.set_ylabel("corr (arb. units)")
        ax00.legend()
        fig.suptitle("", wrap=True)
        plt.tight_layout()
        plt.show()

        check(len(corr) // 2)
        corr = corr[len(corr)//2 :]
        g1 = corr / corr[0]

        fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
        gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures
        ax00 = fig.add_subplot(gs[0, 0])
        ax00.plot(g1.real, label="real part")
        ax00.plot(g1.imag, label="imaginary part")
        ax00.set_xlabel("time (s)")
        ax00.set_ylabel("g1 (arb. units)")
        ax00.legend()
        fig.suptitle("", wrap=True)
        plt.tight_layout()
        plt.show()

        tau = 2 * np.sum(np.abs(g1)) * dt
        print("tau =", tau)
        print("duration =", self.duration_s)
        if tau > self.duration_s:
            print("WARNING: tau > self.duration_s")
        return tau

rate_Hz = 3e-2
duration_s = 1e6
timeLen = int(rate_Hz * duration_s)
axion = EarthBoundAxionHalo()
axion.getBfield(rate_Hz=rate_Hz, timeLen=timeLen, rand_seed=1, numFields=1, verbose=False)
axion.coh_time_g1()
