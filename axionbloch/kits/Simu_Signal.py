"""
TNMR signal classes for the CASPEr high-field (HF) NMR spectrometer.

This module provides Signal subclasses tailored to data acquired by the
TecMag TNMR spectrometer.  The raw FID is stored on disk as interleaved
float64 pairs (Re, Im) by TNMR's binary export; ``TNMR_Signal.loadBinData``
reads that format and converts it to a complex time-series compatible with
the rest of the CASPErG analysis pipeline.

Classes
-------
TNMR_Signal
    Single-shot FID acquired by the TNMR spectrometer.
TNMR_SignalEntry
    Thin dataclass that bundles a ``TNMR_Signal`` with an auxiliary
    parameter dictionary.
TNMR_Signals
    Ordered collection of ``TNMR_SignalEntry`` objects with batch
    analysis helpers inherited from ``Signals``.
"""

from axionbloch.dependency import *
from axionbloch.utils import check
from axionbloch.kits.DataAnalysis import Signal, Signals
from dataclasses import dataclass


class Simu_Signal(Signal):
    """NMR signal acquired by the TecMag TNMR high-field spectrometer.

    Extends ``Signal`` with TNMR-specific binary loading and an HDF5
    round-trip class method.

    Parameters
    ----------
    name : str
        Human-readable label for the signal.
    filePath : str or None
        Path to the TNMR binary export file (.bin).  Pass ``None`` to
        construct an empty signal and populate ``TS`` manually.
    device : str
        Device family identifier (default ``"Tecmag"``).
    deviceID : str
        Unique device string used as the HDF5 group key.
    verbose : bool
        Print debug information when ``True``.
    """

    def __init__(
        self,
        name="TNMR signal",
        filePath=None,
        # fileList=None,
        device="Tecmag",
        deviceID="HF_TNMR_LIA",
        verbose=False,
    ):
        logPrefix = f"[{self.__class__.__name__}.{self.__init__.__name__}] "
        super().__init__(
            name,
            filePath=filePath,
            device=device,
            deviceID=deviceID,
            verbose=verbose,
        )
        self.demodFreq = None

    @classmethod
    def loadFromH5(
        cls,
        filePath,
        deviceID="HF_TNMR_LIA",
    ):
        """Reconstruct a ``Simu_Signal`` from an HDF5 file written by
        ``Signal.saveToH5``.

        Parameters
        ----------
        filePath : str
            Path to the ``.h5`` file.
        deviceID : str
            HDF5 group key under which the signal was stored.

        Returns
        -------
        Simu_Signal
            Populated signal object.
        """
        obj = Signal.loadFromH5(filePath=filePath, deviceID=deviceID, demodIndex=0)
        return obj

    def getTSbyMask(self, mask):
        """

        Parameters
        ----------
        mask : array-like of bool
            Boolean array indexing the time points to keep.

        Returns
        -------
        np.ndarray
            Masked time-series data.
        """
        assert self.TS_raw is not None, "Raw time-series data (TS_raw) is not loaded."
        
        return self.TS_raw[mask]

@dataclass
class Simu_SignalEntry:
    """Container pairing a ``Simu_Signal`` with auxiliary metadata.

    Attributes
    ----------
    signal : Simu_Signal
        The acquired NMR signal.
    params : dict
        Experiment-specific parameters (e.g. pulse sequence settings,
        sample conditions) that do not belong in the signal itself.
    """

    signal: Simu_Signal
    params: dict


class Simu_Signals(Signals):
    """Ordered collection of ``Simu_SignalEntry`` objects.

    Inherits batch Fourier-transform, sorting, and HDF5 I/O helpers
    from ``Signals``.

    Parameters
    ----------
    name : str
        Label for the collection.
    pool : list[Simu_SignalEntry]
        Ordered list of signal entries.
    """

    def __init__(self, name: str, pool: list[Simu_SignalEntry]):
        super().__init__(name, pool)
