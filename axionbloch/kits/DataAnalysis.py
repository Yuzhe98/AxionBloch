##################################################
# for NMR data processing
##################################################

# import packages

# ------------------------------------------------------------
# # numerical computing
# import numpy as np

# # plotting
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec  # for creating subplots
# import matplotlib.ticker as mticker
# from matplotlib.axes import Axes

import os
import pickle
# monitor run time
from dataclasses import dataclass
# for handling time
from datetime import datetime, timezone
from functools import partial

# importing and processing hdf5 files
import h5py
# statistic tools
# for interpolation
# curve fitting (including calculating uncertainties)
# data processing
from scipy.stats import norm

# # physical units and constants
# from astropy import units as unit
# from astropy.units import Quantity, CompositeUnit
# from astropy.constants import codata2018 as const
from ..dependency import *
from .utils import (LIA_FFT, DTRC_filter, ExpCos, ExpCosiSinResidual, Gaussian,
                    Lorentzian, check, clear_lines, dualExpCos, dualGaussian,
                    dualLorentzian, estimatedualExpCos, estimatedualGaussFit,
                    estimatedualLorzfit, estimateExpCos, estimateExpCosiSin,
                    estimateGaussfit, estimateLorzfit, estimatetribLorzfit,
                    getDateAndTime, plotaxisfmt_Hz2ppm, plotaxisfmt_MHz2ppm,
                    plotaxisfmt_ppm2MHz, tribLorentzian)

# -----------------------------------------------------


# from uncertainties import ufloat


# from BaselineRemoval import BaselineRemoval

# np.random.seed(None)  # WARNING!
# BaselineRemoval will effect the randomness of the script.
# Better to set the random seed to None so as to restore the randomness


# include possible typos of "auto"
AUTO_LIST = ["AUTO", "AUTP", "AUT0"]
MANUAL_LIST = ["MANUAL"]

# Set this to False when you don't want to record runtime
RECORD_RUNTIME = False

# function dictionary for fitting and estimating fitting parameters
Function_dict = {
    "Lorentzian".upper(): ["Lorentzian", Lorentzian, estimateLorzfit],
    "dualLorentzian".upper(): ["dualLorentzian", dualLorentzian, estimatedualLorzfit],
    "tribLorentzian".upper(): ["tribLorentzian", tribLorentzian, estimatetribLorzfit],
    "Gaussian".upper(): ["Gaussian", Gaussian, estimateGaussfit],
    "dualGaussian".upper(): ["dualGaussian", dualGaussian, estimatedualGaussFit],
    "ExpCos".upper(): ["ExpCos", ExpCos, estimateExpCos],
    "ExpCosiSin".upper(): ["ExpCosiSin", ExpCosiSinResidual, estimateExpCosiSin],
    "dualExpCos".upper(): ["dualExpCos", dualExpCos, estimatedualExpCos],
    #'LIAFilterHomega'.upper(): ['LIAFilterHomega', LIAFilterHomega, estimateLIAFilterHomega],
}

# parameter names of functions
FunctionParas = {
    "Lorentzian": ["x", "center", "gamma", "area", "offset"],
    "dualLorentzian": [
        "x",
        "center0",
        "gamma0",
        "area0",
        "center1",
        "gamma1",
        "area1",
        "offset",
    ],
    "tribLorentzian": [
        "x",
        "center0",
        "gamma0",
        "area0",
        "center1",
        "gamma1",
        "area1",
        "center2",
        "gamma2",
        "area2",
        "offset",
    ],
    "Gaussian": ["x", "center", "sigma", "area", "offset"],
    "dualGaussian": [
        "x",
        "center0",
        "sigma0",
        "area0",
        "center1",
        "sigma1",
        "area1",
        "offset",
    ],
    "ExpCos": ["x", "Amp", "T2", "nu", "phi0", "offset"],
    "LIAFilterHomega": ["x", "taun", "order"],
}


# for save and load of classes associated with .pkl and .h5
# and for aliases in terminology
class NMRio:
    @classmethod
    def saveToPkl(
        cls,
        dir: str,
        fname: str | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ):
        """
        Save this instance to a pickle file.
        """
        logPrefix = f"[{cls.__class__.__name__}.{cls.saveToPkl.__name__}]"
        if dir is None:
            raise ValueError("dir must not be None")

        if fname is None:
            name = getattr(cls, "name", None)
            base = name if name is not None else cls.__class__.__name__.lower()
            fname = base + "_" + getDateAndTime()

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
            pickle.dump(cls, f, protocol=pickle.HIGHEST_PROTOCOL)

        if verbose:
            print(f"Saved object to {path}")

    @classmethod
    def loadFromPkl(cls, path: str, verbose: bool = False):
        """
        Load an instance of this class from a pickle file.
        """
        logPrefix = f"[{cls.__name__}.{cls.loadFromPkl.__name__}]"
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Pickle file not found: {path}")

        with open(path, "rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, cls):
            raise TypeError(f"Pickle contains {type(obj)}, expected {cls}")

        if verbose:
            print(f"Loaded object from {path}")

        return obj

    @staticmethod
    def saveQuantityToH5group(
        group: h5py.Group,
        name: str,
        quantity: Quantity,
        verbose=False,
    ):
        """
        Save a variable with its value and unit into an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            The parent HDF5 group where the dataset will be created.
        name : str
            The name of the subgroup to store this variable under.
        value : array-like
            The numerical data to store.
        unit : str, optional
            The unit of the data (default is "dimensionless").

        Returns
        -------
        subgroup : h5py.Group
            The created subgroup containing the datasets "value" and "unit".
        """
        logPrefix = f"[{NMRio.__name__}.{NMRio.saveQuantityToH5group.__name__}]"
        # print(logPrefix, quantity)
        subgroup = group.create_group(name)
        if type(quantity) == Quantity:
            q_value = quantity.value
            q_unit = quantity.unit.to_string()
            subgroup.create_dataset("value", data=[q_value])
            subgroup.create_dataset("unit", data=[q_unit])
            if verbose:
                print(logPrefix, "quantity.unit:", q_unit)
        else:
            q_value = quantity
            q_unit = ""
            subgroup.create_dataset("value", data=[q_value])
        return subgroup

    @staticmethod
    def _h5_decode_str(x) -> str:
        if isinstance(x, bytes):
            return x.decode("utf-8")
        arr = np.asarray(x)
        if arr.shape == ():
            item = arr.item()
            return item.decode("utf-8") if isinstance(item, bytes) else str(item)
        flat0 = arr.flat[0]
        return flat0.decode("utf-8") if isinstance(flat0, bytes) else str(flat0)

    @staticmethod
    def loadAttrFromH5group(subgroup: h5py.Group):
        """
        Read a Quantity written by saveQuantityToH5group (datasets 'value' and 'unit'),
        or a plain scalar (int/float) saved without a 'unit' dataset.
        """
        # value of the attribute (float / int)
        raw_val = np.asarray(subgroup["value"])

        # plain scalar saved without unit
        if "unit" not in subgroup:
            return raw_val[0]

        # Quantity saved with unit
        unit_str = Signal._h5_decode_str(subgroup["unit"][0])
        # if value is a scalar
        if raw_val.shape == ():
            return Quantity(raw_val, unit_str)
        else:
            return np.asanyarray(Quantity(raw_val.reshape(-1), unit_str))

    @staticmethod
    def loadValueFromH5group(subgroup: h5py.Group) -> Quantity:
        """
        Read a value written by saveQuantityToH5group (datasets 'value' and 'unit').
        """
        raw_val = np.asarray(subgroup["value"])
        return raw_val[0]

    # aliases allow users to specify dwell time using any of these properties, which all set the same internal attribute starting with _ in the name

    # dwell time aliases:
    # _dwell_time, dwell_time, dwelltime, dwellTime

    @property
    def dwell_time(self):
        return self._dwell_time

    @dwell_time.setter
    def dwell_time(self, value: Quantity[unit.microsecond]):
        self._dwell_time = value

    @property
    def dwelltime(self):
        return self._dwell_time

    @dwelltime.setter
    def dwelltime(self, value: Quantity[unit.microsecond]):
        self._dwell_time = value

    @property
    def dwellTime(self):
        return self._dwell_time

    @dwellTime.setter
    def dwellTime(self, value: Quantity[unit.microsecond]):
        self._dwell_time = value

    # pulse amplitude aliases:
    # _pulse_amplitude, pulse_amplitude, pulseamplitude, pulse_amp, pulseamp, pulseAmp, pa

    @property
    def pulse_amplitude(self):
        return self._pulse_amplitude

    @pulse_amplitude.setter
    def pulse_amplitude(self, value: Quantity):

        self._pulse_amplitude = value

    @property
    def pulseamplitude(self):
        return self._pulse_amplitude

    @pulseamplitude.setter
    def pulseamplitude(self, value: Quantity):
        self._pulse_amplitude = value

    @property
    def pulse_amp(self):
        return self._pulse_amplitude

    @pulse_amp.setter
    def pulse_amp(self, value: Quantity):
        self._pulse_amplitude = value

    @property
    def pulseamp(self):
        return self._pulse_amplitude

    @pulseamp.setter
    def pulseamp(self, value: Quantity):
        self._pulse_amplitude = value

    @property
    def pulseAmp(self):
        return self._pulse_amplitude

    @pulseAmp.setter
    def pulseAmp(self, value: Quantity):
        self._pulse_amplitude = value

    @property
    def pa(self):
        return self._pulse_amplitude

    @pa.setter
    def pa(self, value: Quantity):
        self._pulse_amplitude = value

    # pulse duration aliases:
    # _pulse_duration, pulse_duration, pulseduration, pulse_dur, pulsedur, pulseDuration, pulseDur, pd, _pulse_width, pulse_width, pulsewidth, pulseWidth, pw

    @property
    def pulseDur(self):
        return getattr(self, "_pulse_duration", None)

    @pulseDur.setter
    def pulseDur(self, value: Quantity[unit.microsecond]):
        self._pulse_duration = value

    @property
    def pulse_duration(self):
        return self._pulse_duration

    @pulse_duration.setter
    def pulse_duration(self, value: Quantity[unit.microsecond]):
        self._pulse_duration = value

    @property
    def pulse_dur(self):
        return self._pulse_duration

    @pulse_dur.setter
    def pulse_dur(self, value: Quantity[unit.microsecond]):
        self._pulse_duration = value

    @property
    def pulsedur(self):
        return self._pulse_duration

    @pulsedur.setter
    def pulsedur(self, value: Quantity[unit.microsecond]):
        self._pulse_duration = value

    @property
    def pulseduration(self):
        return self._pulse_duration

    @pulseduration.setter
    def pulseduration(self, value: Quantity[unit.microsecond]):
        self._pulse_duration = value

    @property
    def pd(self):
        return self._pulse_duration

    @pd.setter
    def pd(self, value: Quantity[unit.microsecond]):
        self._pulse_duration = value

    @property
    def pulseDuration(self):
        return self._pulse_duration

    @pulseDuration.setter
    def pulseDuration(self, value: Quantity[unit.microsecond]):
        self._pulse_duration = value

    @property
    def pulsewidth(self):
        return self._pulse_duration

    @pulsewidth.setter
    def pulsewidth(self, value: Quantity[unit.microsecond]):
        self._pulse_duration = value

    # sampling rate aliases:
    # _sampling_rate, sampling_rate, samplingrate, samplingRate, samp_rate, samprate, sampRate, sr

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: Quantity[unit.Hz]):
        self._sampling_rate = value

    @property
    def samplingrate(self):
        return self._sampling_rate

    @samplingrate.setter
    def samplingrate(self, value: Quantity[unit.Hz]):
        self._sampling_rate = value

    @property
    def samplingRate(self):
        return self._sampling_rate

    @samplingRate.setter
    def samplingRate(self, value: Quantity[unit.Hz]):
        self._sampling_rate = value

    @property
    def samp_rate(self):
        return self._sampling_rate

    @samp_rate.setter
    def samp_rate(self, value: Quantity[unit.Hz]):
        self._sampling_rate = value

    @property
    def samprate(self):
        return self._sampling_rate

    @samprate.setter
    def samprate(self, value: Quantity[unit.Hz]):
        self._sampling_rate = value

    @property
    def sampRate(self):
        return self._sampling_rate

    @sampRate.setter
    def sampRate(self, value: Quantity[unit.Hz]):
        self._sampling_rate = value

    # demodulator frequency (Observe Freq. in Tec Mag) aliases:
    # _demod_freq, demod_freq, demodfreq, demodFreq
    @property
    def demod_freq(self):
        return self._demod_freq

    @demod_freq.setter
    def demod_freq(self, value: Quantity[unit.MHz]):
        self._demod_freq = value

    @property
    def demodfreq(self):
        return self._demod_freq

    @demodfreq.setter
    def demodfreq(self, value: Quantity[unit.MHz]):
        self._demod_freq = value

    @property
    def demodFreq(self):
        return self._demod_freq

    @demodFreq.setter
    def demodFreq(self, value: Quantity[unit.MHz]):
        self._demod_freq = value

    # acquisition duration aliases:
    # _acq_duration, acq_duration, acqduration, acq_dur, acqdur, acqDur,

    @property
    def acq_duration(self):
        return self._acq_duration

    @acq_duration.setter
    def acq_duration(self, value: Quantity[unit.millisecond]):
        assert type(value) == Quantity
        self._acq_duration = value

    @property
    def acqduration(self):
        return self._acq_duration

    @acqduration.setter
    def acqduration(self, value: Quantity[unit.millisecond]):
        self._acq_duration = value

    @property
    def acq_dur(self):
        return self._acq_duration

    @acq_dur.setter
    def acq_dur(self, value: Quantity[unit.millisecond]):
        self._acq_duration = value

    @property
    def acqdur(self):
        return self._acq_duration

    @acqdur.setter
    def acqdur(self, value: Quantity[unit.millisecond]):
        self._acq_duration = value

    @property
    def acqDur(self):
        return self._acq_duration

    @acqDur.setter
    def acqDur(self, value: Quantity[unit.millisecond]):
        self._acq_duration = value

    @property
    def acqPts(self):
        return self._acq_points

    @acqPts.setter
    def acqPts(self, value: int):
        self._acq_points = value


class Signal(NMRio):
    name: str | None = None
    device: str
    deviceID: str
    filePath: str | None
    fileList: list[str] | list[None] | None
    demodIndex: int
    TS_raw: Quantity
    TS: Quantity
    sampRate: Quantity | float
    freqs: Quantity
    FS: Quantity
    PSD: Quantity
    creation_times: list[Time] | None
    creation_time: Time | None
    acq_start_time: Time | None

    # acq_start_time_stamp : str | None
    def __init__(
        self,
        name="LIA_signal",
        device: str = "unknown_device",
        deviceID: str = "unknown_device_ID",
        demodIndex: int = 0,
        filePath: str | None = None,
        fileList: list[str] | list[None] | None = None,
        verbose=False,
    ):
        """
        Initialize the (class) Signal object

        Parameters
        ----------
        self : (class) Signal
            Signal from the lock-in amplifier and also its processed data
        name : string
            name of the signal. Defaults to 'LIA signal'.
        filePath : string
            path + filename of the data file.
            e.g. 'C:/Users/Rackyboy/Documents/SQUID data/stream_008/stream_00000.h5'
            it is suggested to use absolute path
        fileList : array_like
            List of path + filename of the data files if the data is stored in multiple files.
            e.g. ['C:/Users/Rackyboy/Documents/SQUID data/stream_008/stream_00000.h5',
            'C:/Users/Rackyboy/Documents/SQUID data/stream_008/stream_00001.h5',
            'C:/Users/Rackyboy/Documents/SQUID data/stream_008/stream_00002.h5']
        device : string
            Name of the device. Defaults to 'dev4434'.
        device_id : string
            ID of the device. Defaults to 'dev4434'
        demod_index : int
            index of the demodulator. Defaults to 0.
        verbose : bool
            Choose True to display processing information. Defaults to False.

        Returns
        -------
        Null

        Examples
        --------
        >>>

        Referrence
        --------
        Null
        """
        # initialize some necessary variables
        self.name = name
        self.device = device
        self.deviceID = deviceID
        self.demodIndex = demodIndex
        self.filePath = filePath
        self.fitflag = False  # initialize fit flag to False (no fitting or fitting is unsuccessful)
        self.exptype = "Experiment type not specified"  # experiment type.

        # store date file list
        # LoadStream will use self.filelist only, not self.file
        if fileList is None:
            if filePath is None:
                self.fileList = []  # initialize an empty filelist
            else:
                self.fileList = [filePath]  # make filelist include file
        else:
            self.fileList = fileList  # store filelist
        if len(self.fileList) > 0:
            self.sortFiles(verbose=False)

        self.timeStamp = None  # initialize timestamp for the data

        self.AvgFIDsq = None  # initialize average FID square to 0.
        # It is not recommended to use AvgFIDsq now, because the time-series is not necessarily FID.
        # Instead, it is recommended to use AvgTSsq
        self.avgTSsq = None  # initialize time-series mean square to 0.
        self.freq_resol = None
        self.timeLen = None
        self.dataXstd, self.dataXmean, self.dataYstd, self.dataYmean = (
            None,
            None,
            None,
            None,
        )
        self.acqDelay = None
        self.acqTime = None

    def __enter__(self):
        print("Inside __enter__")
        return self

    def sortFiles(self, verbose: bool = False):
        if self.fileList is None or len(self.fileList) == 0:
            return
        self.getCreationTimes()

        # print filelist before sorting
        if verbose:
            print(
                f"[{self.sortFiles.__name__}] filelist and creation_times before sorting:"
            )
            print((self.fileList, self.creation_times))

        # Sort data files and creation times by creation time
        sorted_files_and_times = sorted(
            zip(self.fileList, self.creation_times), key=lambda x: x[1]
        )

        self.fileList, self.creation_times = zip(*sorted_files_and_times)
        # self.filelist = list(self.filelist)
        # self.creation_times = list(self.creation_times)

        # print filelist after sorting
        if verbose:
            print(f"[{self.sortFiles.__name__}] filelist after sorting:")
            print((self.fileList, self.creation_times))

    def getCreationTime(self, verbose: bool = False):
        """
        Get the signal creation time from underlying files.

        For multi-file signals, the earliest file creation time is used.
        """
        logPrefix = f"[{self.__class__.__name__}.{self.getCreationTime.__name__}]"
        candidate_files = []
        if hasattr(self, "fileList") and self.fileList is not None:
            candidate_files.extend([f for f in self.fileList if f is not None])
        elif hasattr(self, "filePath") and self.filePath is not None:
            candidate_files.append(self.filePath)

        if len(candidate_files) == 0:
            raise ValueError("No files available to determine signal creation time.")

        self.creation_times = []
        for file in candidate_files:
            if not os.path.isfile(file):
                if verbose:
                    print(logPrefix, f"file not found: {file}")
                continue

            creation_time = os.path.getctime(file)
            creation_time_utc = datetime.fromtimestamp(creation_time, tz=timezone.utc)
            # append time converted to astropy format by astropy.time.Time
            self.creation_times.append(Time(creation_time_utc))

        if len(self.creation_times) == 0:
            raise ValueError("Could not determine creation time from available files.")

        self.creation_time = min(self.creation_times)

        return self.creation_time

    def getCreationTimes(self, verbose: bool = False):
        """
        get creation times of data files.
        """
        logPrefix = f"[{self.__class__.__name__}.{self.getCreationTimes.__name__}]"
        return self.getCreationTime(verbose=verbose)

    def LoadStream(
        self,
        Keadevice=None,  # Load related information to Keadevice
        SQDsensor=None,  # Load related information to SQDsensor
        Expinfo=None,  # Load related information to Expinfo
        skip_pulsedata=False,
        skip_timestamp=False,
        skiprows=3,
        skip_dataXY=False,
        verbose=False,
    ):
        """
        Load lock-in amplifier data from single / multiple file(s).

        Parameters
        ----------
        self : (class) LIAsignal
            Signal from lock-in amplifier and also the processed data

        Keadevice : (class) Kea
            in KeaControl

        SQDsensor : (class) SQUID
            in DataAnalysis

        Expinfo : (class) Experiment
            in LIAControl

        verbose : bool
            Choose True to display processing information. Defaults to False.

        Returns
        -------
        Null

        Examples
        --------
        >>>

        Referrence
        --------
        Null
        """

        self.dataX = []
        self.dataY = []
        self.pulsedata = []

        def loadstream_DAQ(dataFile):
            # with h5py.File(singlefile, 'r', driver='core') as dataFile:
            self.demodFreq = dataFile[
                f"{self.deviceID:s}/demods/{self.demodIndex:d}/demodFreq"
            ][0]
            self.sampRate = dataFile[
                f"{self.deviceID:s}/demods/{self.demodIndex:d}/samprate"
            ][0]
            self.filter_TC = dataFile[
                f"{self.deviceID:s}/demods/{self.demodIndex:d}/filter_TC"
            ][0]
            self.filter_order = dataFile[
                f"{self.deviceID:s}/demods/{self.demodIndex:d}/filter_order"
            ][0]
            if not skip_dataXY:
                self.dataX += list(
                    dataFile[f"{self.deviceID:s}/demods/{self.demodIndex:d}/samplex"]
                )
                self.dataY += list(
                    dataFile[f"{self.deviceID:s}/demods/{self.demodIndex:d}/sampley"]
                )
                # check(sys.getsizeof(self.dataX))
                # check(sys.getsizeof(self.dataY))

            if not skip_pulsedata:
                self.pulsedata += list(
                    dataFile[f"{self.deviceID:s}/demods/{self.demodIndex:d}/auxin0"]
                )

            if not skip_timestamp:
                self.timeStamp = np.linspace(
                    start=0,
                    stop=len(self.dataX) / self.sampRate,
                    num=len(self.dataX),
                    endpoint=False,
                    dtype=float,
                )

            if "Expinfo" in dataFile.keys() and Expinfo is not None:
                Expinfo.name = dataFile["Expinfo/name"][0]
                Expinfo.dateandtime = dataFile["Expinfo/dateandtime"][0]
                Expinfo.exptype = dataFile["Expinfo/exptype"][0]
                self.exptype = dataFile["Expinfo/exptype"][0]

            if "Keadevice" in dataFile.keys() and Keadevice is not None:
                Keadevice.name = dataFile["Keadevice/name"][0]
                Keadevice.B1freq = dataFile["Keadevice/B1freq"][0]
                Keadevice.pulseamp = dataFile["Keadevice/pulseamp"][0]
                Keadevice.pulsedur = dataFile["Keadevice/pulsedur"][0]

            if "SQDsensor" in dataFile.keys():
                if SQDsensor is not None:
                    SQDsensor.name = dataFile["SQDsensor/name"][0]
                    SQDsensor.Mf = dataFile["SQDsensor/Mf"][0]
                    SQDsensor.Rf = dataFile["SQDsensor/Rf"][0]
                    SQDsensor.attenuation = dataFile["SQDsensor/attenuation"][0]
                self.y = dataFile["SQDsensor/Mf"][0]
                self.SQD_Rf = dataFile["SQDsensor/Rf"][0]
                self.attenuation = dataFile["SQDsensor/attenuation"][0]

        def loadstream_UI(dataFile):
            self.dataX += list(
                dataFile[f"000/{self.deviceID:s}/demods/{self.demodIndex:d}/sample/x"][
                    :
                ]
            )
            self.dataY += list(
                dataFile[f"000/{self.deviceID:s}/demods/{self.demodIndex:d}/sample/y"][
                    :
                ]
            )
            # self.pulsedata += list(dataFile[f'000/{self.device_id:s}/demods/{self.demod_index:d}/sample/auxin0'][:])

            self.demodFreq = dataFile[
                f"000/{self.deviceID:s}/demods/{self.demodIndex:d}/sample/frequency"
            ][0]
            self.sampRate = dataFile[
                f"000/{self.deviceID:s}/demods/{self.demodIndex:d}/rate/value"
            ][0]
            self.filter_TC = dataFile[
                f"000/{self.deviceID:s}/demods/{self.demodIndex:d}/timeconstant/value"
            ][0]
            self.filter_order = dataFile[
                f"000/{self.deviceID:s}/demods/{self.demodIndex:d}/order/value"
            ][0]
            if not skip_timestamp:
                self.timeStamp = np.linspace(
                    start=0,
                    stop=len(self.dataX) / self.sampRate,
                    num=len(self.dataX),
                    endpoint=False,
                    dtype=float,
                )

        def loadstream_DAQ_continuous(dataFile):
            self.dataX += list(
                dataFile[
                    f"000/{self.deviceID:s}/demods/{self.demodIndex:d}/sample.x/value"
                ][:]
            )
            self.dataY += list(
                dataFile[
                    f"000/{self.deviceID:s}/demods/{self.demodIndex:d}/sample.y/value"
                ][:]
            )
            self.pulsedata += list(
                dataFile[
                    f"000/{self.deviceID:s}/demods/{self.demodIndex:d}/sample.auxin0/value"
                ][:]
            )

        def loadstream_NMRKineticSimu(dataFile):
            self.testtimestamp = dataFile["NMRKineticSimu/demods/0/timestamp"]
            # check(self.testtimestamp)
            # check(self.testtimestamp[0])
            self.demodFreq = dataFile["NMRKineticSimu/demods/0/demodFreq"][0]
            self.sampRate = dataFile["NMRKineticSimu/demods/0/samprate"][0]
            self.filter_TC = dataFile["NMRKineticSimu/demods/0/filter_TC"][0]
            self.filter_order = dataFile["NMRKineticSimu/demods/0/filter_order"][0]
            # check(self.demodFreq)
            # check(self.samprate)
            self.dataX += list(dataFile["NMRKineticSimu/demods/0/samplex"])
            self.dataY += list(dataFile["NMRKineticSimu/demods/0/sampley"])

        def loadstream_csv():
            self.exptype = "Pulsed-NMR"
            data = np.loadtxt(singlefile, delimiter=",")
            self.timeStamp = 1e-6 * np.array(data[:, 0], dtype=np.float64)
            self.dataX += list(1e-6 * data[:, 1])
            self.dataY += list(1e-6 * data[:, 2])
            del data

            self.demodFreq = 0.0
            self.sampRate = 1.0 / abs(self.timeStamp[1] - self.timeStamp[0])
            self.filter_status = "off"
            self.filter_TC = 0.0
            self.filter_order = 0.0
            self.attenuation = 0.0
            self.acq_arr = np.array(
                [[0], [len(self.dataY)]], dtype=np.int64
            ).transpose()

        def loadstream_txt():
            self.exptype = "Pulsed-NMR"
            # TODO add the function to count text lines in the data file
            data = np.loadtxt(singlefile, skiprows=skiprows)
            self.dataX = data[:, 0]
            self.dataY = data[:, 1]
            del data
            self.pulsedata = np.zeros(len(self.dataY))

            self.demodFreq = 0.0
            # self.samprate = 1. / abs(self.timestamp[1]-self.timestamp[0])
            self.filter_status = "off"
            self.filter_TC = 0.0
            self.filter_order = 0.0
            self.attenuation = 0.0
            self.acq_arr = np.array(
                [[0], [len(self.dataY)]], dtype=np.int64
            ).transpose()

        loadstream_methods = {
            "DAQ_record": loadstream_DAQ,
            "UI": loadstream_UI,
            "DAQ_continuous": loadstream_DAQ_continuous,
            "NMRKineticSimu": loadstream_NMRKineticSimu,
            "Kea_csv": loadstream_csv,
            "TecMag_txt": loadstream_txt,
        }  #

        for singlefile in self.fileList:
            if verbose:
                print("Loading data from " + singlefile)
            if singlefile[-3:] == ".h5":
                with h5py.File(
                    singlefile, "r", driver="core"
                ) as dataFile:  # h5py loading method
                    # check recording method
                    if verbose:
                        check(dataFile.keys())
                    # check(dataFile)
                    # check(sys.getsizeof(dataFile))
                    if (
                        f"{self.deviceID:s}/demods/{self.demodIndex:d}/samplex"
                        in dataFile.keys()
                    ):
                        recordmethod = "DAQ_record"
                    elif (
                        f"000/{self.deviceID:s}/demods/{self.demodIndex:d}/sample/x"
                        in dataFile.keys()
                    ):
                        recordmethod = "UI"
                    elif (
                        f"000/{self.deviceID:s}/demods/{self.demodIndex:d}/sample.x/value"
                        in dataFile.keys()
                    ):
                        recordmethod = "DAQ_continuous"
                    elif "NMRKineticSimu/demods/0" in dataFile.keys():
                        recordmethod = "NMRKineticSimu"
                    else:
                        raise ValueError(
                            "LoadStream() cannot figure out the recording method"
                        )
                    if verbose:
                        print("recordmethod: ", recordmethod)
                    # load stream
                    loadstream_methods[recordmethod](dataFile)
                    if verbose:
                        print("Loading Finished")
            elif singlefile[-4:] == ".csv":
                recordmethod = "Kea_csv"
                if verbose:
                    print("recordmethod: ", recordmethod)
                loadstream_methods[recordmethod]()
            elif singlefile[-4:] == ".tnt":
                recordmethod = "TecMag_tnt"
                # tnt file is not supported for now
            elif singlefile[-4:] == ".txt":
                recordmethod = "TecMag_txt"
                if verbose:
                    print("Reading txt file")
                # loadstream_txt()
                loadstream_methods[recordmethod]()
            else:
                raise ValueError("file type not in .h5, .csv, .tnt nor .txt")

        self.dataX = np.array(self.dataX, dtype=np.float64).flatten()
        self.dataY = np.array(self.dataY, dtype=np.float64).flatten()
        self.pulsedata = np.array(self.pulsedata, dtype=np.float64).flatten()
        self.chanstd_flag = False

        if self.attenuation is None:
            print(
                "Please remember to specify attenuation of the signal before it went to LIA. "
            )

        # self.GetMeas
        # if verbose:
        #     print('after flattening')
        #     print('self.dataX.shape', self.dataX.shape)
        #     print('self.dataY.shape', self.dataY.shape)
        #     print('self.pulsedata.shape', self.pulsedata.shape)

    def createArtificialTS(
        self,
        demodFreq: Quantity = 1 * unit.MHz,
        sampRate: Quantity = 13 * unit.kHz,
        acqDur: Quantity = 10 * unit.s,
        tsUnit: CompositeUnit = unit.one,
        creation_time: Time = None,
        randSeed: int | None = None,
        verbose: bool = False,
    ):
        """
        Create an artificial lock-in amplifier signal stream for, e.g. Monte-Carlo simulation.
        The signal is random noise ~ N(0, 1), normally distributed.


        Parameters
        ----------
        demodFreq: Quantity
            The demodulation frequency of the artificial time-series data.
            Default to 1 MHz.
        sampRate: Quantity
            The sampling rate of the artificial time-series data.
            Default to 13 kHz.
        totalDur: Quantity
            The total duration of the artificial time-series data.
            Default to 10 seconds.
        tsUnit: CompositeUnit
            The unit of the time-series data.
            Default to unit.one.
        creation_time: Time
            The creation time of the artificial time-series data.
            If None, the current time will be used.
            Default to None.
        randSeed: int | None
            The random seed for the random number generator.
            If None, a random seed will be generated.
            Default to None.
        verbose: bool
            If True, print verbose output.
            Default to False.
        """
        logPrefix = f"[{self.__class__.__name__}.{self.createArtificialTS.__name__}]"
        self.demodFreq = demodFreq
        self.sampRate = sampRate
        self.acqDur = acqDur
        # self.freq_resol = 1.0 / self.totalDur
        self.timeLen = int(sampRate * acqDur)

        array = norm.rvs(
            loc=0.0,
            scale=1.0 / np.sqrt(2.0),
            size=[2, self.timeLen],
            random_state=randSeed,
        )
        complexArray = np.asarray(array[0] + 1j * array[1])
        self.TS_raw = complexArray * tsUnit
        self.TS = self.TS_raw.reshape(1, len(array[0]))

        # Use astropy UTC timestamp for synthetic signals.
        self.creation_time = creation_time if creation_time is not None else Time.now()

        # self.filter_status = "off"
        # self.filter_TC = 1
        # self.filter_order = 0
        # if not skip_timestamp:
        #     self.timeStamp = np.linspace(
        #         start=0.0,
        #         stop=len(self.dataX) / self.sampRate,
        #         num=len(self.dataX),
        #         endpoint=False,
        #         dtype=float,
        #     )
        # self.SQD_Mf = 1.0
        # self.SQD_Rf = 1.0
        # self.attenuation = 0.0
        # self.chanstd_flag = False

    def addArtificialTS(
        self,
        add_TS_raw: Quantity,
        verbose: bool = False,
    ):
        """
        Create an artificial lock-in amplifier signal stream for, e.g. Monte-Carlo simulation.
        The signal is random noise ~ N(0, 1), normally distributed.


        Parameters
        ----------

        """

        assert add_TS_raw is not None

        self.TS_raw += np.asanyarray(add_TS_raw)
        self.TS = self.TS_raw.reshape(self.TS.shape)

    def DTRC_filter(
        self,
        TC: float,
        order: int,
    ):
        if not hasattr(self, "dataX") or not hasattr(self, "dataY"):
            if not hasattr(self, "TS"):
                raise ValueError("Signal data not found")
        self.dataX = DTRC_filter(self.dataX, samprate=self.sampRate, TC=TC, order=order)
        self.dataY = DTRC_filter(self.dataX, samprate=self.sampRate, TC=TC, order=order)

        if hasattr(self, "timestamp") and self.timeStamp is not None:
            self.timeStamp = self.timeStamp[: (-1) * order]

        self.filter_status = "on"
        self.filter_TC = TC
        self.filter_order = order

    # def GetStreamDurs(self, verbose: bool = False):
    #     """
    #     Get the durations and the total duration of streams in the LIASignal.
    #     (self.MeasDur_list and self.total_dur)
    #     Usually it is not necessary to remember execute this function
    #     since it is always executed by GetStreamTimes().
    #     """
    #     self.MeasDur_list = []  # measurement durations in [s]
    #     for singlefile in self.filelist:
    #         if verbose:
    #             print("Loading measurement duration from " + singlefile)
    #         if singlefile[-3:] == ".h5":
    #             with h5py.File(
    #                 singlefile, "r", driver="core"
    #             ) as dataFile:  # h5py loading method
    #                 # check recording method
    #                 # if verbose:
    #                 #     check(dataFile.keys())
    #                 # check(dataFile)
    #                 if (
    #                     f"{self.device_id:s}/demods/{self.demod_index:d}/samplex"
    #                     in dataFile.keys()
    #                 ):
    #                     recordmethod = "DAQ_record"
    #                 elif (
    #                     f"000/{self.device_id:s}/demods/{self.demod_index:d}/sample/x"
    #                     in dataFile.keys()
    #                 ):
    #                     recordmethod = "UI"
    #                 elif (
    #                     f"000/{self.device_id:s}/demods/{self.demod_index:d}/sample.x/value"
    #                     in dataFile.keys()
    #                 ):
    #                     recordmethod = "DAQ_continuous"
    #                 elif "NMRKineticSimu/demods/0" in dataFile.keys():
    #                     recordmethod = "NMRKineticSimu"
    #                 else:
    #                     raise ValueError(
    #                         "LoadStream() cannot figure out the recording method"
    #                     )
    #                 # if verbose:
    #                 #     print('recordmethod: ', recordmethod)

    #                 if recordmethod == "DAQ_record":
    #                     self.sampRate = dataFile[
    #                         f"{self.device_id:s}/demods/{self.demod_index:d}/samprate"
    #                     ][0]
    #                     self.MeasDur_list.append(
    #                         len(
    #                             (
    #                                 dataFile[
    #                                     f"{self.device_id:s}/demods/{self.demod_index:d}/samplex"
    #                                 ]
    #                             )
    #                         )
    #                         / self.sampRate
    #                     )
    #                 elif recordmethod == "UI":
    #                     self.sampRate = dataFile[
    #                         f"000/{self.device_id:s}/demods/{self.demod_index:d}/rate/value"
    #                     ][0]
    #                     self.MeasDur_list.append(len(self.dataX) / self.sampRate)
    #                 elif recordmethod == "NMRKineticSimu":
    #                     self.sampRate = dataFile["NMRKineticSimu/demods/0/samprate"][0]
    #                     self.MeasDur_list.append(
    #                         len(dataFile["NMRKineticSimu/demods/0/samplex"])
    #                         / self.sampRate
    #                     )
    #                 else:
    #                     raise ValueError("cannot find recording method")
    #                 if verbose:
    #                     print("Loading Finished")
    #         elif singlefile[-4:] == ".csv":
    #             dataFile = np.loadtxt(singlefile, delimiter=",")
    #             self.timeStamp = np.array(dataFile[:, 0], dtype=np.float64)
    #             self.sampRate = 1e6 / abs(self.timeStamp[1] - self.timeStamp[0])
    #             self.MeasDur_list.append(len(dataFile[:, 2]) / self.sampRate)
    #         else:
    #             raise ValueError("file type not in .h5 nor .csv")

    #     self.total_dur = 0.0
    #     for dur in self.MeasDur_list:
    #         self.total_dur += dur

    #     if verbose:
    #         print(f"[{self.GetStreamDurs.__name__}] Finished. ")
    #         check(self.MeasDur_list)

    # # @record_runtime_YorN(RECORD_RUNTIME)
    # def GetStreamTimes(self, verbose: bool = False):
    #     # self.SortFiles(verbose=verbose)

    #     self.GetStreamDurs(verbose)
    #     # Calculate stop times
    #     stop_times = [
    #         creation_time + timedelta(seconds=meas_duration)
    #         for creation_time, meas_duration in zip(
    #             self.creation_times, self.MeasDur_list
    #         )
    #     ]

    #     # # Convert stop times to astropy Time objects
    #     self.stop_times = [
    #         Time(stop_time, format="datetime", scale="utc") for stop_time in stop_times
    #     ]
    #     del stop_times

    #     # Print results
    #     if verbose:
    #         print("Stop Times: ", self.stop_times)

    #     self.StreamStart = self.creation_times[0]
    #     self.StreamStop = self.stop_times[-1]
    #     if verbose:
    #         check(self.StreamStart)
    #         check(self.StreamStop)
    #         print(f"total measurement duration: {self.total_dur}. ")

    #     self.timeastro = self.creation_times[0]

    def displayTS(
        self,
        showpulsedata=False,
        showchanX=True,
        showchanY=True,
        plotrate=None,  # in [Hz]. Default to None
        scatter_X=[],
        scatter_Y=[],
        maxlen=int(1e7),
        # movingavg_len=1,
        verbose=False,
    ):
        """
        Display the time-series data

        Parameters
        ----------
        showpulsedata, showchanX, showchanY : bool
            choose to display these channels or not
        plotrate : float
            The length of displayed data in one second.
            It determines the length of ploted data.
            The unit of this rate is [Hz] or [s^-1]
            Default to None, which means use plotrate = sampling rate
        maxlen : int
            maximum length for the displayed data.
            A warning will be triggered if the actually-ploted data exceed this length.
            Default to 1e6.

        """
        if showpulsedata:
            assert self.pulsedata is not None
        assert self.dataX is not None
        assert self.dataY is not None
        assert self.sampRate is not None
        assert plotrate is None or plotrate > 0
        lenofdata = len(self.dataX)
        if plotrate is None:
            plotrate = self.sampRate
            plotintv = 1
            if lenofdata > maxlen:
                clear_lines()
                print(
                    f"Warning! length of data = {len(self.dataX):e} > max length {maxlen:e}. "
                    "This may result in long runtime. "
                )
        elif plotrate > self.sampRate:
            clear_lines()
            print(
                "WARNING: plotrate > self.samprate. plotrate will be reassigned the value of self.samprate"
            )
            plotrate = self.sampRate
            plotintv = 1
        elif plotrate <= self.sampRate and plotrate > 0:
            plotintv = max(int(1.0 * self.sampRate / plotrate), 1)
        else:
            raise ValueError("plotrate < 0. ")

        fig = plt.figure(figsize=(10, 4), dpi=150)  #
        gs = gridspec.GridSpec(nrows=3, ncols=1)  #
        pulse_ax = fig.add_subplot(gs[0, 0])
        dataX_ax = fig.add_subplot(
            gs[1, 0], sharex=pulse_ax
        )  # share x-axis with pulse_ax
        dataY_ax = fig.add_subplot(
            gs[2, 0], sharex=pulse_ax, sharey=dataX_ax
        )  # share x-axis with pulse_ax, share y-axis with dataX_ax
        if showpulsedata:
            pulse_ax.plot(
                self.timeStamp[0:-1:plotintv],
                self.pulsedata[0:-1:plotintv],
                label="pulse sequence",
                c="tab:purple",
            )
            pulse_ax.set_ylabel("Voltage [V]")
            pulse_ax.legend(loc="upper right")  # adjust the location of the legend

        def plotTS(ax, data, label, color, scatter):
            ax.plot(
                self.timeStamp[0:-1:plotintv], data[0:-1:plotintv], label=label, c=color
            )
            if scatter is not None and len(scatter) > 0:
                for idx in scatter[0:]:
                    ax.scatter(
                        self.timeStamp[0:-1:plotintv][idx // plotintv],
                        data[0:-1:plotintv][idx // plotintv],
                        marker="*",
                        c="tab:red",
                    )
            ax.set_ylabel("Voltage [V]")
            ax.legend(loc="upper right")  # adjust the location of the legend

        # X channel of LIA
        if showchanX:
            plotTS(dataX_ax, self.dataX, "LIA X", "tab:green", scatter_X)

        # Y channel of LIA
        if showchanY:
            plotTS(dataY_ax, self.dataY, "LIA Y", "tab:brown", scatter_Y)

        pulse_ax.set_xlim(self.timeStamp[0], self.timeStamp[-1])
        dataX_ax.set_xlim(self.timeStamp[0], self.timeStamp[-1])
        dataY_ax.set_xlim(self.timeStamp[0], self.timeStamp[-1])
        dataY_ax.set_xlabel("time [s]")

        # adjust tick parameters
        pulse_ax.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off

        dataX_ax.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off

        # Set the plot title
        titletext = self.exptype
        for singlefile in self.filelist:
            titletext += "\n" + singlefile  # include the data file name(s) in the title
        fig.suptitle(titletext, wrap=True)
        plt.tight_layout()
        # plt.grid()
        # plt.show()

        return 0

    # def displayHist(
    #     self,
    #     showchanXhist: bool = False,
    #     showchanYhist: bool = False,
    #     showFFTrealhist: bool = False,
    #     showFFTimghist: bool = False,
    #     showPSDhist: bool = False,
    #     showExcess: bool = False,
    #     excessThres: float = 3.4,
    #     freqRangeforHist=None,
    #     scale: str = "log",  # 'log' or 'linear'
    #     # normalizebysigma=True,
    #     showplt: bool = True,
    #     verbose=False,
    # ):
    #     """
    #     Display the histograms

    #     Parameters
    #     ----------
    #     showchanXhist, showchanYhist, showPSDhist : bool
    #         choose to show the histograms of these channels / spectrum

    #     """
    #     # print(f'{self.displayHist.__name__} needs documentations. -- Yuzhe 2024-07-10')
    #     fig = plt.figure(figsize=(11 * 0.8, 7 * 0.8), dpi=150)
    #     width_ratios = [1, 1, 1]
    #     height_ratios = [1, 1]
    #     gs = gridspec.GridSpec(
    #         nrows=2, ncols=3, width_ratios=width_ratios, height_ratios=height_ratios
    #     )
    #     # fig.subplots_adjust(
    #     #     top=,
    #     #     bottom=,
    #     #     left=,
    #     #     right=,
    #     #     hspace=,
    #     #     wspace=)
    #     dataX_hist_ax = fig.add_subplot(gs[0, 0])  #
    #     dataY_hist_ax = fig.add_subplot(gs[0, 1])  #
    #     FFTreal_hist_ax = fig.add_subplot(gs[1, 0])  #
    #     FFTimag_hist_ax = fig.add_subplot(gs[1, 1])  #
    #     PSD_ax = fig.add_subplot(gs[0, 2])  #
    #     PSD_hist_ax = fig.add_subplot(gs[1, 2])  #

    #     #
    #     def GaussianHistPlot(data, title, ax):
    #         mean = np.mean(data)
    #         std = np.std(data)
    #         hist, bin_edges = np.histogram(
    #             data / std,
    #             bins=max(10, 7 * int(np.floor(np.amax(abs(data) / std)))),
    #             density=False,
    #         )
    #         binwidth = abs(bin_edges[1] - bin_edges[0])
    #         sumofcounts = len(data)
    #         hist_info = (
    #             f"sum of counts={sumofcounts:d}\nmean={mean:.1e} [V] std={std:.1e} [V]"
    #         )
    #         ax.set_xlabel("signal / $\\sigma$")

    #         hist_x = []
    #         hist_y = []
    #         for i, count in enumerate(hist):
    #             if count > 0:
    #                 hist_x.append((bin_edges[i] + bin_edges[i + 1]) / 2.0)
    #                 hist_y.append(count)
    #         ax.scatter(
    #             hist_x,
    #             hist_y,
    #             color="goldenrod",
    #             edgecolors="k",
    #             linewidths=1,
    #             marker="o",
    #             s=6,
    #             zorder=6,
    #             label=f"histogram",
    #         )
    #         # plot pdf of gaussian distribution (mean = 0, std = 1)
    #         xstamp = np.linspace(
    #             start=np.amin(bin_edges),
    #             stop=np.amax(bin_edges),
    #             num=max(100, len(bin_edges)),
    #             endpoint=True,
    #         )
    #         # ax.plot(xstamp, norm.pdf(xstamp, mean, 1), label='Gaussian pdf', linestyle='--')
    #         ax.plot(
    #             xstamp,
    #             sumofcounts * binwidth * norm.pdf(xstamp, 0, 1),
    #             label="Gaussian pdf",
    #             linestyle="--",
    #         )
    #         # sumofcounts * binwidth *
    #         ax.set_ylabel("count")
    #         ax.set_title(title + "\n" + hist_info, size=fontsize)
    #         ax.set_yscale(scale)
    #         ax.set_ylim(bottom=0.1)
    #         ax.legend(loc="best")
    #         ax.grid(True)
    #         del (
    #             mean,
    #             std,
    #             hist,
    #             bin_edges,
    #             binwidth,
    #             sumofcounts,
    #             hist_info,
    #             hist_x,
    #             hist_y,
    #             xstamp,
    #         )

    #     if showchanXhist:
    #         GaussianHistPlot(self.dataX, title="LIA channel X", ax=dataX_hist_ax)
    #     if showchanYhist:
    #         GaussianHistPlot(self.dataY, title="LIA channel Y", ax=dataY_hist_ax)

    #     if showFFTrealhist or showFFTimghist:
    #         if not hasattr(self, "avgFFT"):
    #             print(
    #                 "Warning: LIASignal object does not have the attribute 'avgFFT'. "
    #                 "Now self.GetFFT() is executed for generating FFT "
    #                 "for the histogram. "
    #             )
    #             self.GetNoPulseFFT()

    #     if showFFTrealhist:
    #         GaussianHistPlot(self.avgFFT.real, title="FFT real", ax=FFTreal_hist_ax)
    #     if showFFTimghist:
    #         GaussianHistPlot(self.avgFFT.imag, title="FFT imag", ax=FFTimag_hist_ax)

    #     def plotPSD(x, y, title, ax: Axes, freqRangeforHist):
    #         if freqRangeforHist is None:
    #             r0 = 0
    #             r1 = len(self.avgPSD)
    #         else:
    #             r0 = np.argmin(abs(self.frequencies - freqRangeforHist[0]))
    #             r1 = np.argmin(abs(self.frequencies - freqRangeforHist[1]))
    #         if abs(r1 - r0) < 100:
    #             print(
    #                 "Warning: abs(r1-r0) < 100. PSD bins in freqRangeforHist may be too less for histogram"
    #             )
    #         ax.plot(x, y, color="tab:green", label=f"normalized PSD", linewidth=1)
    #         ax.plot(
    #             x[r0:r1],
    #             y[r0:r1],
    #             color="tab:red",
    #             label=f"data for histogram",
    #             alpha=0.2,
    #             linewidth=2,
    #         )

    #         check((r0, r1))
    #         # ax.fill_between(x[r0:r1], np.amax(y), np.amin(y),
    #         #  color = 'r', alpha=0.1, zorder=6)
    #         # ax.plot(xstamp, chi2.pdf(xstamp, df=2), label='Chi-sq pdf\ndof=2', linestyle='--')
    #         # ax.plot(xstamp, sumofcounts * binwidth * chi2.pdf(xstamp, df=1), label='Chi-sq pdf dof=1', linestyle='--')
    #         ax.set_ylabel("normalized PSD")
    #         ax.set_xlabel("frequency [Hz]")
    #         ax.set_title(title + "\n", size=fontsize)
    #         ax.legend(loc="best")
    #         ax.grid(True)

    #     def ChisqHistPlot(ax, data, title, thres: float = None):
    #         """
    #         data is normalize by being dividing std
    #         """
    #         mean = np.mean(data)
    #         std = np.std(data)
    #         check(np.amax(data))
    #         hist, bin_edges = np.histogram(
    #             data, bins=max(10 * int(np.amax(data)), 100), density=False
    #         )
    #         binwidth = abs(bin_edges[1] - bin_edges[0])
    #         sumofcounts = len(data)
    #         hist_info = f"sum of counts={sumofcounts:d}\nmean={mean:.1e} [V^2/Hz] std={std:.1e} [V^2/Hz]"
    #         # if normalizebysigma:
    #         # bin_edges /= std
    #         ax.set_xlabel("PSD / $\\sigma_{\mathrm{PSD}}$")
    #         # else:
    #         #     ax.set_xlabel('signal [V]')
    #         hist_x = []
    #         hist_y = []
    #         for i, count in enumerate(hist):
    #             if count > 0:
    #                 hist_x.append((bin_edges[i] + bin_edges[i + 1]) / 2.0)
    #                 hist_y.append(count)
    #         # TODO add this to every hist_x and hist_y
    #         hist_x = np.array(hist_x)
    #         hist_y = np.array(hist_y)
    #         if thres is not None:
    #             thresIndex = np.argmin(abs(hist_x - thres))
    #             check(np.sum(hist_y[thresIndex:]))
    #             check(np.sum(hist_y[thresIndex:]) / sumofcounts)
    #         ax.scatter(
    #             hist_x,
    #             hist_y,
    #             color="tab:red",
    #             edgecolors="k",
    #             linewidths=1,
    #             marker="o",
    #             s=6,
    #             zorder=6,
    #             label=f"histogram",
    #         )

    #         # plot pdf of gaussian distribution (mean = 0, std = 1)
    #         xstamp = np.linspace(
    #             start=0,
    #             stop=np.amax(bin_edges),
    #             num=max(100, len(bin_edges)),
    #             endpoint=True,
    #         )
    #         # ax.plot(xstamp, norm.pdf(xstamp, mean, 1), label='Gaussian pdf', linestyle='--')
    #         ax.plot(
    #             xstamp,
    #             2 * sumofcounts * binwidth * chi2.pdf(2.0 * xstamp, df=2),
    #             label="Chi-sq pdf\ndof=2",
    #             linestyle="--",
    #         )
    #         # sumofcounts * binwidth *
    #         # ax.plot(xstamp, sumofcounts * binwidth * chi2.pdf(xstamp, df=1), label='Chi-sq pdf dof=1', linestyle='--')
    #         ax.set_ylabel("count")
    #         ax.set_title(title + "\n" + hist_info, size=fontsize)
    #         ax.set_xscale("linear")
    #         ax.set_yscale(scale)
    #         ax.set_ylim(bottom=0.1)
    #         ax.legend(loc="best")
    #         ax.grid(True)
    #         del (
    #             mean,
    #             std,
    #             hist,
    #             bin_edges,
    #             sumofcounts,
    #             hist_info,
    #             hist_x,
    #             hist_y,
    #             xstamp,
    #         )

    #     if showPSDhist:
    #         # if not hasattr(self, 'avgFFT'):
    #         #     print("Warning: LIASignal object does not have the attribute 'avgFFT'. "\
    #         #           "Now self.GetFFT() is executed for generating FFT "\
    #         #           "for the histogram. ")
    #         #     self.GetNoPulseFFT()
    #         if not hasattr(self, "avgPSD"):
    #             print(
    #                 "Warning: LIASignal object does not have the attribute 'avgPSD'. "
    #                 "Now self.GetNoPulsePSD() is executed for generating a power spectrum "
    #                 "for the histogram. "
    #             )
    #             self.GetNoPulsePSD()

    #         plotPSD(
    #             x=self.frequencies,
    #             y=self.avgPSD,
    #             title="PSD",
    #             ax=PSD_ax,
    #             freqRangeforHist=freqRangeforHist,
    #         )

    #         if freqRangeforHist is None:
    #             r0 = 0
    #             r1 = len(self.avgPSD)
    #         else:
    #             r0 = np.argmin(abs(self.frequencies - freqRangeforHist[0]))
    #             r1 = np.argmin(abs(self.frequencies - freqRangeforHist[1]))
    #         ChisqHistPlot(
    #             ax=PSD_hist_ax,
    #             data=self.avgPSD[r0:r1] / np.std(self.avgPSD[r0:r1]),
    #             title="Histogram of the normalized PSD",
    #             thres=excessThres,
    #         )
    #         PSD_hist_ax.grid(True)
    #         del r0, r1

    #     titletext = "Histograms of data"
    #     for singlefile in self.filelist:
    #         titletext += "\n" + singlefile  # include the data file name(s) in the title
    #     fig.suptitle(titletext, wrap=True)
    #     plt.tight_layout()
    #     # plt.grid()
    #     if showplt:
    #         plt.show()
    #     return (
    #         dataX_hist_ax,
    #         dataY_hist_ax,
    #         FFTreal_hist_ax,
    #         FFTimag_hist_ax,
    #         PSD_ax,
    #         PSD_hist_ax,
    #     )

    # def tsCheckDrift(
    #     self,
    #     checkX: bool = True,
    #     checkY: bool = True,
    #     # removeDrift:bool=False,
    #     makeplot: bool = False,
    #     verbose: bool = True,
    # ):
    #     """
    #     check difts in the time-series
    #     """
    #     if len(self.dataX) < 100:
    #         clear_lines()
    #         print(
    #             f"[{self.tsCheckDrift.__name__}] Warning: len(self.dataX) = {len(self.dataX)}. array may be too short for drift diagnostics. "
    #         )

    #     self.tsHasBeenModified_flag = False

    #     chunklen = len(self.dataX) // 4
    #     chunk_list = []
    #     dataset_drift_list = []
    #     for i in range(4):
    #         chunk_list.append(
    #             [(i) * chunklen, min((i + 1) * chunklen, len(self.dataX))]
    #         )

    #     def check_iter(data):
    #         drift = False
    #         for i, chunk_i in enumerate(chunk_list):
    #             for j, chunk_j in enumerate(chunk_list):
    #                 if i < j:
    #                     dataset_drift = checkDrift(
    #                         data[chunk_i[0] : chunk_i[1]], data[chunk_j[0] : chunk_j[1]]
    #                     )
    #                     # if dataset_drift and makeplot:
    #                     #     checkDrift(
    #                     #         data[chunk_i[0]:chunk_i[1]],
    #                     #         data[chunk_j[0]:chunk_j[1]],
    #                     #         makeplot=makeplot)
    #                     dataset_drift_list.append(dataset_drift)
    #                     drift = drift or dataset_drift
    #         return drift

    #     drift_X, drift_Y = False, False
    #     if checkX:
    #         drift_X = check_iter(self.dataX)
    #     if checkY:
    #         drift_Y = check_iter(self.dataY)

    #     if True in dataset_drift_list:  # if there is a drift
    #         self.tsHasDrift_flag = True
    #         clear_lines()
    #         print(f"data {self.filelist} has drift. ")
    #         # sys.stdout.flush()
    #     else:  # if there is no drift
    #         self.tsHasDrift_flag = False

    #     if makeplot:
    #         # self.GetSpectrum(showtimedomain=True, showfreqdomain=False)
    #         self.displayTS(
    #             showpulsedata=False,
    #             showchanX=True,
    #             showchanY=True,
    #             plotrate=None,  # in [Hz]. Default to None
    #             # maxlen=int(1e7),
    #             verbose=False,
    #         )

    #     # if self.tsHasDrift_flag and removeDrift:
    #     #     if verbose:
    #     #         print(f'{self.??.__name__} is removing the drift in the time series. ')
    #     #     self.tsHasDrift_flag = False

    #     if makeplot:
    #         pass

    #     if checkX and checkY:
    #         return [drift_X, drift_Y]

    # def tsCheckJump(
    #     self,
    #     threshold_in_std: float = 5,
    #     checkX: bool = True,
    #     checkY: bool = True,
    #     makeplot: bool = False,
    #     verbose: bool = False,
    # ):
    #     """
    #     check jumps in the time series
    #     """
    #     # better to check drifts before checking jumps so that the standard deviation
    #     # is of value of reference
    #     # if verbose:
    #     #     print(f'{self.tsCheckJump.__name__} is working. ')
    #     self.dataXmean, self.dataXstd = np.mean(self.dataX), np.std(self.dataX)
    #     self.dataYmean, self.dataYstd = np.mean(self.dataY), np.std(self.dataY)

    #     def checkJump(array, mean, std):
    #         std_list = []
    #         if not isinstance(array, np.ndarray):
    #             array = np.array(array)
    #         Jumps_indices = list(
    #             np.where(np.abs(array - mean) > threshold_in_std * std)[0]
    #         )
    #         if verbose and len(Jumps_indices) > 0:
    #             # clear_lines()
    #             # print(f'{self.tsCheckJump.__name__}: ' +
    #             #       f'data {self.filelist} {len(Jumps_indices):d} jump(s) exceeding '
    #             #         + f'{threshold_in_std:.2g} std found. ')
    #             # if len(Jumps_indices) > 0:
    #             # clear_lines()
    #             # print(f'std values: ')
    #             std_list = list(array[Jumps_indices] / std)
    #             # print(array[Jumps_indices] / std)
    #         return Jumps_indices, std_list

    #     self.Jumps_X, self.Jumps_Y = [], []
    #     std_X_list, std_Y_list = [], []
    #     if checkX:
    #         self.Jumps_X, std_X_list = checkJump(
    #             self.dataX, self.dataXmean, self.dataXstd
    #         )
    #     if checkY:
    #         self.Jumps_Y, std_Y_list = checkJump(
    #             self.dataY, self.dataYmean, self.dataYstd
    #         )

    #     Jumps_indices = self.Jumps_X + self.Jumps_Y
    #     if verbose:
    #         clear_lines()
    #         print(
    #             f"{self.tsCheckJump.__name__}: "
    #             + f"data {self.filelist} \n{len(Jumps_indices):d} jump(s) exceeding "
    #             + f"{threshold_in_std:.2g} std found. "
    #         )
    #         if len(Jumps_indices) > 0:
    #             clear_lines()
    #             print(f"std values: ")
    #             print([round(std_sigma, 3) for std_sigma in std_X_list + std_Y_list])
    #     if makeplot:
    #         self.displayTS(
    #             showpulsedata=False,
    #             showchanX=True,
    #             showchanY=True,
    #             plotrate=None,  # in [Hz]. Default to None
    #             scatter_X=self.Jumps_X,
    #             scatter_Y=self.Jumps_Y,
    #             # maxlen=int(1e7),
    #             verbose=False,
    #         )

    #     if checkX and checkY:
    #         return [(len(self.Jumps_X) > 0), (len(self.Jumps_Y) > 0)]

    # def tsCheckNorm(
    #     self,
    #     checkX: bool = True,
    #     checkY: bool = True,
    #     alpha: float = 0.05,
    #     makeplot: bool = False,
    #     verbose: bool = False,
    # ):
    #     """
    #     Perform the Shapiro-Wilk test for normality.

    #     The Shapiro-Wilk test tests the null hypothesis that the
    #     data was drawn from a normal distribution.

    #     return
    #     ------
    #     Null hypothesis H0: The time-series is normally distributed.
    #     Alternative hypothesis H1: The time-series is not normally distributed.

    #     return False if the null hypothesis is NOT rejected (TS is normally distributed).
    #     return True if the null hypothesis is rejected (TS is NOT normally distributed).

    #     False (negative) is favored, like in a COVID-19 PCR test report.

    #     """
    #     if checkX:
    #         stat_X, p_X = shapiro(self.dataX)
    #     if checkY:
    #         stat_Y, p_Y = shapiro(self.dataY)
    #     if makeplot:
    #         # dataX_hist_ax, dataY_hist_ax, \
    #         # FFTreal_hist_ax, FFTimag_hist_ax, \
    #         # PSD_ax, PSD_hist_ax =
    #         self.displayHist(
    #             showchanXhist=True,
    #             showchanYhist=True,
    #             showFFTrealhist=False,
    #             showFFTimghist=False,
    #             showPSDhist=False,
    #             freqRangeforHist=None,
    #             scale="log",  # or 'log'
    #             # normalizebysigma=True,
    #             showplt=True,
    #             verbose=False,
    #         )
    #     if checkX and checkY and verbose:
    #         if p_X < alpha:
    #             clear_lines()
    #             print(
    #                 f"[{self.tsCheckNorm.__name__}] Warning: dataX of {self.filelist} failed to pass normality check. "
    #                 + f"stat_X={stat_X:.3f}, p_X={p_X:.3e}. "
    #             )
    #         if p_Y < alpha:
    #             clear_lines()
    #             print(
    #                 f"[{self.tsCheckNorm.__name__}] Warning: dataY of {self.filelist} failed to pass normality check. "
    #                 + f"stat_Y={stat_Y:.3f}, p_Y={p_Y:.3e}. "
    #             )

    #     if checkX and checkY:
    #         return [(p_X < alpha), (p_Y < alpha)]

    # def tsCheckSanity(
    #     self,
    #     # checkXandYconsit:bool=True,
    #     plotIfInsane: bool = False,
    #     verbose=False,
    # ):
    #     """
    #     check drifts jumps, and normality in the time-series
    #     """
    #     func_list = [self.tsCheckDrift, self.tsCheckJump, self.tsCheckNorm]
    #     report = {"drift": [], "jump": [], "normality": []}
    #     # report['drift'] = [self.tsCheckDrift(makeplot=plotIfInsane, verbose=verbose)]
    #     # report['jump'] = [self.tsCheckJump(threshold_in_std=5, makeplot=plotIfInsane, verbose=verbose)]
    #     # report['normality'] = [self.tsCheckNorm(makeplot=False, verbose=verbose)]

    #     for i, key in enumerate(report.keys()):
    #         report[key] = func_list[i](makeplot=False, verbose=verbose)
    #         if True in report[key]:
    #             func_list[i](makeplot=plotIfInsane, verbose=verbose)
    #     if verbose:
    #         print("Sanity check report")
    #         print("drift")
    #         print(report["drift"])
    #         print("jump")
    #         print(report["jump"])
    #         print("normality")
    #         print(report["normality"])
    #     return report

    # # @record_runtime_YorN(RECORD_RUNTIME)
    # def psdFindBaseline(
    #     self,
    #     HorCombn_opt: bool = False,
    #     HorCombn_step: int = None,
    #     poly_degree: int = 4,
    #     removeBaseline: bool = False,
    #     showStats: bool = False,
    #     freqRangeForStats: list = None,
    #     makeplot: bool = False,
    #     return_Baseline: bool = False,
    #     verbose: bool = False,
    # ):
    #     """
    #     find baseline in the power spectrum
    #     """
    #     self.psdBaselineRemoval_flag = False
    #     if not hasattr(self, "avgPSD"):
    #         # clear_lines()
    #         print(
    #             f"[{self.psdFindBaseline.__name__}] Warning: object "
    #             "does not have the attribute avgPSD. "
    #             "Now self.GetNoPulsePSD() is executed for generating a power spectrum "
    #             "for the histogram. "
    #         )
    #         self.GetNoPulsePSD()

    #     if len(self.avgPSD) < 100:
    #         # clear_lines()
    #         print(
    #             f"[{self.psdFindBaseline.__name__}] len(self.avgPSD) < 100. "
    #             "array may be too short for baseline diagnostics. "
    #         )

    #     if HorCombn_opt:
    #         assert HorCombn_step is not None
    #         if HorCombn_step == 1:
    #             # clear_lines()
    #             print(
    #                 f"[{self.psdFindBaseline.__name__}] Warning: HorCombn_step == 1. "
    #                 "No horizontal combination will be executed. "
    #             )
    #             freq = self.frequencies
    #             PSD = self.avgPSD
    #         else:
    #             num = len(self.avgPSD) // HorCombn_step
    #             HorLen = int(num * HorCombn_step)
    #             freq = self.frequencies[
    #                 HorCombn_step // 2 : HorLen + HorCombn_step // 2 : HorCombn_step
    #             ]
    #             PSD = self.avgPSD[0:HorLen].reshape((num, HorCombn_step))
    #             PSD = np.mean(PSD, axis=1)
    #             assert freq.shape == PSD.shape
    #     else:
    #         freq = self.frequencies
    #         PSD = self.avgPSD

    #     if len(PSD) > 1000000:
    #         # clear_lines()
    #         print(
    #             f"[{self.psdFindBaseline.__name__}] Warning: array too long. "
    #             + f"len(PSD)={len(PSD):.2e} > 1 000 000. "
    #             + "The computation time for finding the baseline could be long. "
    #         )

    #     # find baseline
    #     baseObj = BaselineRemoval(input_array=PSD)
    #     tic = time.time()
    #     # IModpoly method for removing baseline
    #     # default param: degree=2,repitition=100,gradient=0.001
    #     output = baseObj.IModPoly(degree=poly_degree)
    #     toc = time.time()
    #     # baselineFit_time = abs(toc - tic)

    #     if verbose:
    #         print(
    #             f"[{self.psdFindBaseline.__name__}] "
    #             + "BaselineRemoval time consumption: "
    #             + f"{abs(toc - tic):.2g} [s]"
    #         )
    #     del baseObj

    #     base_f = interpolate.interp1d(freq, (PSD - output), fill_value="extrapolate")
    #     baseline = base_f(self.frequencies)

    #     iter = 0
    #     while np.any(baseline == 0) and iter < 100:
    #         zero_indices = np.where(baseline == 0)[0]
    #         for idx in zero_indices:
    #             if idx == 0:
    #                 # If the zero is at the start, use the next element
    #                 baseline[idx] = baseline[idx + 1]
    #             elif idx == len(baseline) - 1:
    #                 # If the zero is at the end, use the previous element
    #                 baseline[idx] = baseline[idx - 1]
    #             else:
    #                 # Use the average of the neighbors
    #                 baseline[idx] = (baseline[idx - 1] + baseline[idx + 1]) / 2.0
    #         iter += 1

    #     if np.any(baseline == 0):
    #         raise ValueError(
    #             f"[{self.psdFindBaseline.__name__}] np.any(baseline == 0) == True. "
    #         )

    #     PSDlabel = "PSD"

    #     normPSD = self.avgPSD / baseline - 1.0
    #     normPSDlabel = "normalized PSD (baseline removed)"

    #     if np.any(normPSD > 1e3):
    #         print(
    #             f"[{self.psdFindBaseline.__name__}] Warning: np.any(normPSD > 1e3) == True. "
    #         )

    #     if showStats:
    #         if freqRangeForStats is None:
    #             r0 = 0
    #             r1 = len(self.avgPSD)
    #         else:
    #             r0 = np.argmin(abs(self.frequencies - freqRangeForStats[0]))
    #             r1 = np.argmin(abs(self.frequencies - freqRangeForStats[1]))

    #         PSDlabel += f"\nmean={np.mean(self.avgPSD[r0:r1]):.2g}"
    #         PSDlabel += f"\nstd={np.std(self.avgPSD[r0:r1]):.2g}"
    #         normPSDlabel += f"\nmean={np.mean(normPSD[r0:r1]):.2g}"
    #         normPSDlabel += f"\nstd={np.std(normPSD[r0:r1]):.2g}"
    #         print(f"[{self.psdFindBaseline.__name__}] stats:")
    #         print(PSDlabel)
    #         print(normPSDlabel)
    #         print("\n")

    #     if makeplot:
    #         fig = plt.figure(
    #             figsize=(11 * 0.8, 7 * 0.8), dpi=150
    #         )  # initialize a figure
    #         gs = gridspec.GridSpec(nrows=2, ncols=1)  # create grid for multiple figures
    #         ax00 = fig.add_subplot(gs[0, 0])
    #         ax10 = fig.add_subplot(gs[1, 0])

    #         if showStats:
    #             ax00.fill_between(
    #                 self.frequencies[r0:r1],
    #                 np.amax(self.avgPSD),
    #                 np.amin(self.avgPSD),
    #                 color="r",
    #                 alpha=0.1,
    #                 zorder=6,
    #                 label="Stats Range",
    #             )
    #         ax00.plot(self.frequencies, self.avgPSD, label=PSDlabel)
    #         if HorCombn_opt:
    #             ax00.plot(freq, PSD, label="PSD_HC", linestyle="-", c="g")
    #         ax00.plot(
    #             self.frequencies, baseline, label="baseline", linestyle="--", c="r"
    #         )
    #         ax00.set_ylabel("PSD [V^2/Hz]")

    #         if showStats:
    #             ax10.fill_between(
    #                 self.frequencies[r0:r1],
    #                 np.amax(normPSD),
    #                 np.amin(normPSD),
    #                 color="r",
    #                 alpha=0.1,
    #                 zorder=6,
    #                 label="Stats Range",
    #             )

    #         ax10.plot(self.frequencies, normPSD, label=normPSDlabel)

    #         for ax in [ax00, ax10]:
    #             ax.set_xlabel("frequency [Hz]")
    #             ax.legend(loc="best")
    #             ax.grid()

    #         plt.show()

    #     if removeBaseline:
    #         if verbose:
    #             print(
    #                 f"{self.psdFindBaseline.__name__} is removing the baseline in the PSD. "
    #             )
    #         self.avgPSD = normPSD
    #         del output
    #         self.psdBaselineRemoval_flag = True

    #     if return_Baseline:
    #         return baseline

    # def FindPulse(
    #     self,
    #     trigger_mod=None,
    #     trigger_value=None,
    #     search_mod=None,
    #     pulseduration=None,
    #     verbose=False,
    # ):
    #     """
    #     Find the start and end of excitation pulse in the Aux channel recording.

    #     (If you are reading in the small window, please go to the source for correct schematics. )

    #     Parameters
    #     ----------
    #     self : (class) LIAsignal
    #         Signal from lock-in amplifier and also the processed data

    #     trigger_mod : str
    #         Trigger mod. It must be either 'auto' or 'manual' (case-insensitive). Defaults to None.
    #         In auto mod, you do not need to specify a trigger_value, and trigger_value would be np.amax(self.pulsedata) / 2.
    #         In manual mod, you need to specify a trigger_value.

    #     trigger_value : float
    #         Value for detecting rising and falling edge of a pulse. Defaults to None.
    #         If trigger_mod == 'auto', trigger_value would be automatically chosen as np.amax(self.pulsedata) / 2.
    #         Otherwise, you need to specify a trigger_value.

    #     search_mod : str
    #         the mod of searching for pulses. It must be either 'auto' or 'manual' (case-insensitive). Defaults to None.
    #         If search_mod == 'auto', the function will automatically find the rising (start) and falling (end) edges of pulses.
    #         Otherwise, the function would automatically find the rising edges, and determine the falling edges by pulseduation.

    #     pulseduration : float
    #         the duration of pulse in [s]. Defaults to None.
    #         Only when search_mod == 'manual', it is necessary to specify pulseduration,
    #         so that the function can determine the falling edges of pulses.

    #     verbose : bool
    #         Choose True to display processing information. Defaults to False.

    #     schematics on the algorithm of FindPulse

    #     Concerning the end of acquisition, please refer to scenario (1) and (2)
    #     Concerning the start of acquisition, please refer to scenario (1) to (4)

    #     scenario (1) in either auto or manual mod
    #     when start[2] - end[2] >= self.acqTime + self.acqDelay
    #                 ┌───────┐            ┌───────┐            ┌───────┐            ┌───────┐
    #                 |       |            |       |            |       |            |       |
    #     ────────────┘       └────────────┘       └────────────┘       └────────────┘       └───────────
    #                 ↑       ↑            ↑       ↑            ↑       ↑            ↑       ↑          ↑
    #             discard   end[0]      start[0] end[1]      start[1] end[2]      start[2] end[2]     start[2]
    #     acquied data:
    #                         └────────────┘       └────────────┘       └────────────┘       └───────────

    #     scenario (2) in either auto or manual mod
    #     when start[2] - end[2] < self.acqTime + self.acqDelay
    #                 ┌───────┐            ┌───────┐            ┌───────┐            ┌───────┐
    #                 |       |            |       |            |       |            |       |
    #     ────────────┘       └────────────┘       └────────────┘       └────────────┘       └──
    #                 ↑       ↑            ↑       ↑            ↑       ↑            ↑       ↑
    #             discard   end[0]      start[0] end[1]      start[1] end[2]      start[2] discard
    #     acquied data:
    #                         └────────────┘       └────────────┘       └────────────┘

    #     scenario (3) in auto mod
    #     ────┐            ┌───────┐            ┌───────┐            ┌───
    #         |            |       |            |       |            |
    #         └────────────┘       └────────────┘       └────────────┘
    #         ↑            ↑       ↑            ↑       ↑            ↑
    #       end[0]      start[0] end[1]      start[1] end[2]      start[2]
    #     acquied data:
    #         └────────────┘       └────────────┘       └────────────┘

    #     scenario (4) in manual mod
    #     ────┐            ┌───────┐            ┌───────┐            ┌───
    #         |            |       |            |       |            |
    #         └────────────┘       └────────────┘       └────────────┘
    #         ↑            ↑       ↑            ↑       ↑            ↑
    #      discard      discard end[0]      start[0] end[1]      start[1]
    #     acquied data:
    #                              └────────────┘       └────────────┘

    #     Returns
    #     -------
    #     Null

    #     Examples
    #     --------
    #     >>>

    #     Referrence
    #     --------
    #     Null
    #     """
    #     self.exptype = "Pulsed-NMR"  # FindPulse is only for pulsed-NMR experiment

    #     # determine the trigger mode
    #     assert trigger_mod is not None
    #     if trigger_mod.upper() in AUTO_LIST:
    #         trigger_value = np.amax(self.pulsedata) / 2.0
    #     elif trigger_mod.upper() in MANUAL_LIST:
    #         assert trigger_value is not None

    #     # determine the length of acquisition delay and acquisition time
    #     assert self.acqDelay is not None
    #     assert self.acqTime is not None
    #     acqDelayLen = int(np.ceil(self.acqDelay * self.sampRate))
    #     if acqDelayLen <= 3:
    #         print(
    #             "!!WARNING!! acqDelay * samprate = %d. This could be too short for PSD. "
    #             % acqDelayLen
    #         )
    #     acqTimeLen = int(np.ceil(self.acqTime * self.sampRate))

    #     # start to search for pulses
    #     # it is recommended to use manual mode instead of Auto mode
    #     assert search_mod is not None
    #     assert (search_mod.upper() in AUTO_LIST) or (search_mod.upper() in MANUAL_LIST)

    #     # search for pulses in auto mode
    #     if search_mod.upper() in AUTO_LIST:
    #         startofpulse = np.flatnonzero(
    #             (self.pulsedata[1:] > trigger_value)
    #             & (self.pulsedata[:-1] < trigger_value)
    #         )
    #         endofpulse = np.flatnonzero(
    #             (self.pulsedata[1:] < trigger_value)
    #             & (self.pulsedata[:-1] > trigger_value)
    #         )

    #         # check the length of startofpulse and endofpulse
    #         if len(startofpulse) < 1 or len(endofpulse) < 1:
    #             raise ValueError(
    #                 "AUTO mode. Number of pulses smaller than 1. \nlen(startofpulse)<1 or len(endofpulse)<1"
    #             )

    #         if verbose:
    #             print("startofpulse.shape before adjustment ", startofpulse.shape)
    #             print("endofpulse.shape before adjustment ", endofpulse.shape)

    #         # discard the first element in the startofpulse
    #         # if it is ahead of the first element in the endofpulse
    #         if startofpulse[0] < endofpulse[0]:
    #             startofpulse = startofpulse[1:]

    #         # check again the length of startofpulse and endofpulse
    #         if len(startofpulse) < 1 or len(endofpulse) < 1:
    #             raise ValueError("len(startofpulse)<=1 or len(endofpulse)<=1")

    #         # discard the last element in the startofpulse
    #         # if it is ahead of the first element in the endofpulse
    #         if startofpulse[-1] < endofpulse[-1]:
    #             #        ┌────────────────┐
    #             #        |                |
    #             # ───────┘                └───────────
    #             #        ↑                ↑
    #             #  startofpulse[-1] endofpulse[-1]

    #             # # When there is usable signal after the last pulse start, like in scenario (1):
    #             # if (len(self.pulsedata) - 1) >= startofpulse[-1] + pulselen + (acqdelaylen + acqtimelen):
    #             #     startofpulse = np.append(startofpulse, [len(self.pulsedata)-1]) # add the final data point to startofpulse
    #             # else:
    #             #     endofpulse = endofpulse[0:-1]
    #             pass
    #         if verbose:
    #             print("startofpulse.shape after adjustment ", startofpulse.shape)
    #             print("endofpulse.shape after adjustment ", endofpulse.shape)

    #     # search for pulses in manual mode
    #     if search_mod.upper() in MANUAL_LIST:
    #         assert pulseduration is not None
    #         pulselen = int(pulseduration * self.sampRate) + 1
    #         if verbose:
    #             print("pulseduration ", pulseduration)
    #             print("self.samprate ", self.sampRate)
    #             print("pulselen ", pulselen)

    #         # find the start of pulses
    #         startofpulse = np.flatnonzero(
    #             (self.pulsedata[1:] > trigger_value)
    #             & (self.pulsedata[:-1] < trigger_value)
    #         )

    #         assert len(startofpulse) >= 1

    #         # When there is usable signal after the last pulse start, like in scenario (1)
    #         if (len(self.pulsedata) - 1) >= startofpulse[-1] + pulselen + (
    #             acqDelayLen + acqTimeLen
    #         ):
    #             endofpulse = startofpulse + pulselen  # keep the last endofpulse
    #             startofpulse = np.append(
    #                 startofpulse, [len(self.pulsedata) - 1]
    #             )  # add the final data point to startofpulse
    #         # When there is no usable signal after the last pulse start, like in scenario (2)
    #         else:
    #             if len(startofpulse) == 1:
    #                 raise ValueError("no useable data. ")
    #             endofpulse = (
    #                 startofpulse[:-1] + pulselen
    #             )  # create endofpulse without the last pulse start

    #         startofpulse = startofpulse[1:]  # discard the first pulse start

    #     # save startofpulse and endofpulse
    #     self.startofpulse = startofpulse
    #     self.endofpulse = endofpulse
    #     del startofpulse, endofpulse

    #     # check if the lengths of data are too short
    #     if (acqDelayLen + acqTimeLen) > np.amin(self.startofpulse - self.endofpulse):
    #         print(f'acqdelaylen + acqtimelen = {acqDelayLen + acqTimeLen}')
    #         print(f'np.amin(self.startofpulse - self.endofpulse) = {np.amin(self.startofpulse - self.endofpulse)}')
    #         print('Warning: (acqdelaylen + acqtimelen) > np.amin(self.startofpulse - self.endofpulse)')
    #         check(self.startofpulse.shape)
    #         check(self.endofpulse.shape)
    #         # if verbose:
    #         self.GetSpectrum(showfreqdomain=False)

    #         # raise ValueError('(points of acqDelay + points of acqTime) > min(endofpulse-startofpulse)')

    #     # save parameters
    #     self.acqDelayLen = acqDelayLen
    #     self.acqTimeLen = acqTimeLen
    #     startofacq = self.endofpulse + acqDelayLen
    #     self.acq_arr = np.array([startofacq, startofacq + acqTimeLen]).transpose()
    #     self.selectPulse_list = range(len(self.acq_arr))
    #     # an example of self.acq_arr
    #     # self.acq_arr : ndarray(array([
    #     #    [ 15964,  18643],
    #     #    [ 42748,  45427],
    #     #    [ 69538,  72217],
    #     #    [ 96333,  99012],
    #     #    [123112, 125791],
    #     #    [149893, 152572],
    #     #    [176686, 179365],
    #     #    [203463, 206142],
    #     #    [230256, 232935],
    #     #    [257035, 259714],
    #     #    [283826, 286505],
    #     #    [310601, 313280],
    #     #    [337386, 340065],
    #     #    [364175, 366854],
    #     #    [390960, 393639]
    #     #                     ], dtype=int64)) [shape=(15, 2)]
    #     # here we have 15 chunks of acquired data
    #     del acqDelayLen, acqTimeLen, startofacq

    #     if verbose:
    #         check(self.startofpulse)
    #         check(self.endofpulse)
    #         check(self.acq_arr)
    #         print("\n")
    #     # return False

    # def FindSpinEcho_BACKUP_16_01_24(
    #     self,
    #     search_mod: str,
    #     trigger_value,
    #     pulseduration,
    #     echonumber: int,
    #     echotime: float,
    #     verbose: bool,
    #     plot: bool,
    #     savepath: str,
    # ):
    #     """
    #     Find the start and end of excitation pulse in the Aux channel recording.

    #                 90° pulse      180° pulse           180° pulse           180° pulse
    #                                ┌───────┐            ┌───────┐            ┌───────┐
    #                                |       |            |       |            |       |
    #                 ┌───────┐      |       |            |       |            |       |
    #                 |       |      |       |            |       |            |       |
    #     ────────────┘       └──────┘       └────────────┘       └────────────┘       └───────────
    #                 ↑       ↑      ↑       ↑            ↑       ↑            ↑       ↑
    #             discard   discard start[0] end[1]      start[1] end[2]     start[2] discard

    #     Parameters
    #     ----------
    #     self : (class) LIAsignal
    #         Signal from lock-in amplifier and also the processed data
    #     trigger_value : float
    #         same as for FindPulse()
    #     search_mod : str
    #         same as for FindPulse()
    #     pulseduration : float
    #         Duration of a single pulse in
    #     echonumber : float
    #         Number of 180° pulses
    #     echotime : float
    #         Time between the starting points of 2 180° pulses

    #     Returns
    #     -------
    #     Null

    #     Examples
    #     --------
    #     >>>

    #     Referrence
    #     --------
    #     Null
    #     """
    #     self.exptype = "CPMG"

    #     # determine the trigger mode
    #     assert search_mod is not None
    #     if search_mod.upper() in AUTO_LIST:
    #         trigger_value = np.amax(self.pulsedata) / 2.0
    #     elif search_mod.upper() in MANUAL_LIST:
    #         assert trigger_value is not None

    #     if trigger_value == 0:
    #         trigger_value = np.amax(self.pulsedata) / 2.0
    #         if verbose:
    #             check(trigger_value)

    #     # determine the length of acquisition delay and acquisition time
    #     assert self.acqDelay is not None
    #     assert self.acqTime is not None
    #     # acqdelaylen = int(self.acqDelay * self.samprate)
    #     acqdelaylen = int(np.ceil(self.acqDelay * self.sampRate))
    #     acqtimelen = int(np.ceil(self.acqTime * self.sampRate))
    #     self.acqTimeLen = acqtimelen
    #     if acqdelaylen <= 3:
    #         print(
    #             f"!!WARNING!! acqDelay * samprate = {acqdelaylen}. This could be too short for PSD. "
    #         )

    #     if search_mod.upper() in AUTO_LIST:
    #         # print("search mod auto")
    #         startofpulse = np.flatnonzero(
    #             (self.pulsedata[1:] > trigger_value)
    #             & (self.pulsedata[:-1] < trigger_value)
    #         )
    #         endofpulse = np.flatnonzero(
    #             (self.pulsedata[1:] < trigger_value)
    #             & (self.pulsedata[:-1] > trigger_value)
    #         )
    #         # check('Auto mode')
    #         # check(startofpulse)
    #         # check(endofpulse)

    #         if verbose:
    #             print(
    #                 "removing first entry (90° pulse) from pulse arrays for spin echo analysis"
    #             )
    #         startofpulse = startofpulse[1:]
    #         endofpulse = endofpulse[1:]

    #         if verbose:
    #             print("startofpulse.shape before adjustment ", startofpulse.shape)
    #             print("endofpulse.shape before adjustment ", endofpulse.shape)

    #         if startofpulse[0] < endofpulse[0]:
    #             startofpulse = startofpulse[1:]

    #         if len(startofpulse) < 1 or len(endofpulse) < 1:
    #             raise ValueError("len(startofpulse)<=1 or len(endofpulse)<=1")

    #         if startofpulse[-1] < endofpulse[-1]:
    #             endofpulse = endofpulse[0:-1]

    #         if verbose:
    #             print("startofpulse.shape after adjustment ", startofpulse.shape)
    #             print("endofpulse.shape after adjustment ", endofpulse.shape)

    #         #########################
    #         # # check mismatch
    #         # diff_default = startofpulse[0] - endofpulse[0]
    #         # if verbose: check(diff_default)
    #         # mismatches = np.ones(len(startofpulse))
    #         # mismatches[0] = 0
    #         # while 1 in mismatches:
    #         #     for i in range(1, len(startofpulse)):
    #         #         diff = startofpulse[i] - endofpulse[i]
    #         #         if diff != diff_default:
    #         #             endofpulse[i] += 1
    #         #             mismatches[i] = 1
    #         #             print("chunk mismatch ALERT")
    #         #             check(diff)
    #         #         else:
    #         #             mismatches[i] = 0
    #         # #check(mismatches)
    #         #########################

    #         self.startofpulse = startofpulse
    #         self.endofpulse = endofpulse

    #         startofacq = self.endofpulse + acqdelaylen
    #         # endofacq = self.startofpulse - acqdelaylen

    #         acqlength = self.startofpulse[0] - self.endofpulse[0] - acqdelaylen
    #         endofacq = self.endofpulse + acqlength

    #         self.acq_arr = np.array([startofacq, endofacq]).transpose()
    #         # if verbose: check(self.acq_arr)
    #         if len(self.acq_arr) <= 3:
    #             print("WARNING! len(self.acq_arr) <= 3. Too less pulses?")
    #         # self.acq_arr = self.acq_arr[1:]

    #         del (
    #             startofpulse,
    #             endofpulse,
    #             startofacq,
    #             endofacq,
    #             acqdelaylen,
    #         )  # , mismatches

    #     elif search_mod in ["manual", "Manual"]:
    #         # print("search mod manual")

    #         pulselen = int(np.ceil(pulseduration * self.sampRate))
    #         if verbose:
    #             print("pulseduration ", pulseduration)
    #             print("self.samprate ", self.sampRate)
    #             print("pulselen ", pulselen)
    #         if pulseduration is None:
    #             pulseduration = self.pulseduration
    #         if pulseduration is None:
    #             raise ValueError("pulseduration and self.pulseduration is None")

    #         startofpulse = np.flatnonzero(
    #             (self.pulsedata[1:] > trigger_value)
    #             & (self.pulsedata[:-1] < trigger_value)
    #         )
    #         endofpulse = startofpulse + pulselen

    #         if verbose:
    #             print(
    #                 "remove first entry (90° pulse) from pulse arrays for spin echo analysis"
    #             )
    #         startofpulse = startofpulse[1:]
    #         endofpulse = endofpulse[1:]

    #         if len(startofpulse) < 1 or len(endofpulse) < 1:
    #             return True
    #             # raise ValueError('number of pulses less than 1.')

    #         if verbose:
    #             print("startofpulse.shape before adjustment ", startofpulse.shape)
    #             print("endofpulse.shape before adjustment ", endofpulse.shape)

    #         if startofpulse[0] < endofpulse[0]:
    #             print("fixing first pulse start")
    #             startofpulse = startofpulse[1:]

    #         if startofpulse[-1] < endofpulse[-1]:
    #             # if IndexError: # unsafe edit just to get it working, we have to resolve it later
    #             #    pass
    #             print("fixing last pulse start")
    #             endofpulse = endofpulse[0:-1]

    #         if endofpulse[-1] > (len(self.pulsedata) - 1):
    #             print("fixing last pulse end")
    #             endofpulse = endofpulse[:-1]

    #         if verbose:
    #             print("startofpulse.shape after adjustment ", startofpulse.shape)
    #             print("endofpulse.shape after adjustment ", endofpulse.shape)

    #         if verbose:
    #             check(startofpulse)
    #             check(endofpulse)

    #         self.startofpulse = startofpulse
    #         self.endofpulse = endofpulse
    #         del startofpulse, endofpulse

    #         # acqdelaylen = int(self.acqDelay * self.samprate)
    #         # if acqdelaylen <=3:
    #         #     print(f'!!WARNING!! acqDelay * samprate = {acqdelaylen}. ')

    #         echotimelen = int(echotime * self.sampRate)
    #         pulse_diff = np.amin(self.startofpulse - self.endofpulse)
    #         if verbose:
    #             check(echotimelen)
    #             check(pulse_diff)
    #         # these 2 above should be the same I guess? when testing: difference 10

    #         pulse_interval = np.amax(self.endofpulse) - np.amin(self.endofpulse)
    #         echotime_total = echotimelen * echonumber
    #         if verbose:
    #             check(pulse_interval)
    #             check(echotime_total)
    #         # same for these 2, difference 1595

    #         if acqdelaylen > pulse_diff:
    #             print("acqdelaylen ", acqdelaylen)
    #             print("np.amin(self.startofpulse - self.endofpulse) ", pulse_diff)
    #             raise ValueError("points of acqDelay > min(endofpulse-startofpulse)")

    #         startofacq = self.endofpulse + acqdelaylen
    #         endofacq = self.endofpulse + echotimelen - pulselen
    #         self.acq_arr = np.array([startofacq, endofacq]).transpose()
    #         if verbose:
    #             check(self.acq_arr)
    #         if len(self.acq_arr) <= 3:
    #             print("WARNING! len(self.acq_arr) <= 3. Too less pulses?")

    #         # del pulselen, acqdelaylen, echotimelen, pulse_diff, pulse_interval, echotime_total, startofacq, endofacq

    #     else:
    #         raise ValueError("search_mod not found")

    #     if plot:
    #         fig, ax1 = plt.subplots(figsize=(16, 9), dpi=100)
    #         ax2 = ax1.twinx()
    #         # ax1.plot(self.timestamp, self.pulsedata, color="g")
    #         # ax2.plot(self.timestamp, (self.dataX), color="b")

    #         FIDall = []
    #         alltimes = []
    #         for i in range(len(self.acq_arr)):

    #             timedomain = self.timeStamp[self.acq_arr[i, 0] : self.acq_arr[i, 1]]
    #             alltimes.append(timedomain)

    #             pulsedata = self.pulsedata[self.acq_arr[i, 0] : self.acq_arr[i, 1]]

    #             FIDhere = (
    #                 self.dataX[self.acq_arr[i, 0] : self.acq_arr[i, 1]]
    #                 + 1j * self.dataY[self.acq_arr[i, 0] : self.acq_arr[i, 1]]
    #             )
    #             FIDall.append(FIDhere)

    #             # ax1.plot(timedomain, pulsedata, color="g")
    #             ax2.plot(timedomain, FIDhere, color="b")

    #         plt.show()

    #         # np.savetxt(savepath[:-4]+'_timeseries.txt', np.array(alltimes), delimiter=' ', newline='\n')
    #         # np.savetxt(savepath[:-4]+'_FIDseries.txt', np.array(FIDall), delimiter=' ', newline='\n')
    #         fig.savefig(savepath[:-4] + "_timeseries.png")

    # def FindSpinEcho(
    #     self,
    #     search_mod: str,
    #     trigger_value,
    #     pulseduration: float,
    #     echonumber: int,
    #     echotime: float,
    #     plot: bool,
    #     savepath: str,
    #     verbose=False,
    # ):
    #     """
    #     Find the start and end of pi pulses in a spin-echo measurement.
    #     The first pulse, which is a pi/2 pulse, is discarded.

    #                 90° pulse      180° pulse           180° pulse           180° pulse
    #                                ┌───────┐            ┌───────┐            ┌───────┐
    #                                |       |            |       |            |       |
    #                 ┌───────┐      |       |            |       |            |       |
    #                 |       |      |       |            |       |            |       |
    #     ────────────┘       └──────┘       └────────────┘       └────────────┘       └───────────
    #                 ↑       ↑      ↑       ↑            ↑       ↑            ↑       ↑
    #             discard   discard discard end[0]      start[0] end[1]     start[1] discard

    #     Parameters
    #     ----------
    #     self : (class) LIAsignal
    #         Signal from lock-in amplifier and also the processed data

    #     search_mod : str

    #     trigger_value : float
    #         Value for detecting rising and falling edge of a pulse. Defaults to None.
    #         If trigger_mod == 'auto', trigger_value would be automatically chosen as np.amax(self.pulsedata) / 2.
    #         Otherwise, you need to specify a trigger_value.

    #     search_mod : str
    #         same as for FindPulse()

    #     pulseduration : float
    #         Duration of a single pulse in

    #     echonumber : float
    #         Number of 180° pulses

    #     echotime : float
    #         Time between the starting points of 2 180° pulses

    #     Returns
    #     -------
    #     Null

    #     Examples
    #     --------
    #     >>>

    #     Referrence
    #     --------
    #     Null
    #     """
    #     self.exptype = "CPMG"

    #     # determine the trigger mode
    #     assert search_mod is not None
    #     if search_mod.upper() in AUTO_LIST:
    #         trigger_value = np.amax(self.pulsedata) / 2.0
    #         check(trigger_value)
    #     elif search_mod.upper() in MANUAL_LIST:
    #         assert trigger_value is not None

    #     # if trigger_value == 0:
    #     #    trigger_value = np.amax(self.pulsedata) / 2.
    #     #    if verbose:
    #     #        check(trigger_value)

    #     # determine the length of acquisition delay and acquisition time
    #     assert self.acqDelay is not None
    #     assert self.acqTime is not None
    #     acqdelaylen = int(np.ceil(self.acqDelay * self.sampRate))

    #     if acqdelaylen <= 3:
    #         print(
    #             f"!!WARNING!! acqDelay * samprate = {acqdelaylen}. This could be too short. "
    #         )

    #     acqtimelen = int(np.ceil(self.acqTime * self.sampRate))
    #     self.acqTimeLen = acqtimelen

    #     # parameter sanity check
    #     assert search_mod is not None
    #     assert (search_mod.upper() in AUTO_LIST) or (search_mod.upper() in MANUAL_LIST)

    #     # start to search for pulses
    #     # search for pulses in auto mode
    #     if search_mod.upper() in AUTO_LIST:
    #         startofpulse = np.flatnonzero(
    #             (self.pulsedata[1:] > trigger_value)
    #             & (self.pulsedata[:-1] < trigger_value)
    #         )
    #         endofpulse = np.flatnonzero(
    #             (self.pulsedata[1:] < trigger_value)
    #             & (self.pulsedata[:-1] > trigger_value)
    #         )

    #         # check the length of startofpulse and endofpulse
    #         if len(startofpulse) < 1:
    #             raise ValueError(
    #                 "AUTO mode. Number of pulses smaller than 1. \nlen(startofpulse)<1"
    #             )
    #         if len(endofpulse) < 1:
    #             raise ValueError(
    #                 "AUTO mode. Number of pulses smaller than 1. \nlen(endofpulse)<1"
    #             )
    #         if verbose:
    #             print("startofpulse.shape before adjustment ", startofpulse.shape)
    #             print("endofpulse.shape before adjustment ", endofpulse.shape)

    #         # discard the first element because it is the pi/2 pulse
    #         if verbose:
    #             print(
    #                 "removing first entry (90° pulse) from pulse arrays for spin echo analysis"
    #             )
    #         startofpulse = startofpulse[1:]
    #         endofpulse = endofpulse[1:]

    #         # if the first element in the startofpulses is ahead of that in the endofpulse
    #         # discard the first element in the startofpulse
    #         if startofpulse[0] < endofpulse[0]:
    #             startofpulse = startofpulse[1:]

    #         # check again the length of startofpulse and endofpulse
    #         if len(startofpulse) < 1 or len(endofpulse) < 1:
    #             raise ValueError("len(startofpulse)<=1 or len(endofpulse)<=1")

    #         # discard the last element in the endofpulse
    #         # if it is behind the last element in the startofpulse
    #         if startofpulse[-1] < endofpulse[-1]:
    #             #        ┌────────────────┐
    #             #        |                |
    #             # ───────┘                └───────────
    #             #        ↑                ↑
    #             #  startofpulse[-1] endofpulse[-1]
    #             endofpulse = endofpulse[0:-1]

    #         if verbose:
    #             print("startofpulse.shape after adjustment ", startofpulse.shape)
    #             print("endofpulse.shape after adjustment ", endofpulse.shape)

    #         #########################
    #         # # check mismatch
    #         # diff_default = startofpulse[0] - endofpulse[0]
    #         # if verbose: check(diff_default)
    #         # mismatches = np.ones(len(startofpulse))
    #         # mismatches[0] = 0
    #         # while 1 in mismatches:
    #         #     for i in range(1, len(startofpulse)):
    #         #         diff = startofpulse[i] - endofpulse[i]
    #         #         if diff != diff_default:
    #         #             endofpulse[i] += 1
    #         #             mismatches[i] = 1
    #         #             print("chunk mismatch ALERT")
    #         #             check(diff)
    #         #         else:
    #         #             mismatches[i] = 0
    #         # #check(mismatches)
    #         #########################

    #         # save variables
    #         self.startofpulse = startofpulse
    #         self.endofpulse = endofpulse

    #         # determine the acquisition array and the acquisition length
    #         startofacq = self.endofpulse + acqdelaylen
    #         acqlength = self.startofpulse[0] - self.endofpulse[0] - acqdelaylen
    #         endofacq = self.endofpulse + acqlength

    #         self.acq_arr = np.array([startofacq, endofacq]).transpose()
    #         # if verbose: check(self.acq_arr)
    #         if len(self.acq_arr) <= 3:
    #             print("WARNING! len(self.acq_arr) <= 3. Too less pulses?")
    #         # self.acq_arr = self.acq_arr[1:]

    #         del (
    #             startofpulse,
    #             endofpulse,
    #             startofacq,
    #             endofacq,
    #             acqdelaylen,
    #         )  # , mismatches

    #     elif search_mod.upper() in MANUAL_LIST:
    #         if pulseduration is None:
    #             pulseduration = self.pulseduration
    #         if pulseduration is None:
    #             raise ValueError("pulseduration and self.pulseduration is None")
    #         pulselen = int(pulseduration * self.sampRate) + 1
    #         if verbose:
    #             print("pulseduration ", pulseduration)
    #             print("self.samprate ", self.sampRate)
    #             print("pulselen ", pulselen)

    #         # find the start of pulses
    #         startofpulse = np.flatnonzero(
    #             (self.pulsedata[1:] > trigger_value)
    #             & (self.pulsedata[:-1] < trigger_value)
    #         )
    #         if verbose:
    #             check(self.pulsedata[1:])
    #             check(self.pulsedata[:-1])
    #             check(trigger_value)
    #         assert len(startofpulse) >= 1

    #         # When there is usable signal after the last pulse start,
    #         # like in FindPulse() documentation scenario (1)
    #         if (len(self.pulsedata) - 1) >= startofpulse[-1] + pulselen + (
    #             acqdelaylen + acqtimelen
    #         ):
    #             endofpulse = startofpulse + pulselen  # keep the last endofpulse
    #             startofpulse = np.append(
    #                 startofpulse, [len(self.pulsedata) - 1]
    #             )  # add the final data point to startofpulse
    #         # When there is no usable signal after the last pulse start, like in FindPulse() scenario (2)
    #         else:
    #             if len(startofpulse) == 1:
    #                 raise ValueError("no useable data. ")
    #             endofpulse = (
    #                 startofpulse[:-1] + pulselen
    #             )  # create endofpulse without the last pulse start

    #         if verbose:
    #             print(
    #                 "remove first entry (90° pulse) from pulse arrays for spin echo analysis"
    #             )
    #         startofpulse = startofpulse[1:]  # discard the first pulse start

    #         #################################################################
    #         # do a number of corrections which apparently worked a long time ago

    #         endofpulse = endofpulse[1:]  # why?  # Yuzhe bookmark

    #         if len(startofpulse) < 1 or len(endofpulse) < 1:
    #             return True
    #             # raise ValueError('number of pulses less than 1.')

    #         if verbose:
    #             print("startofpulse.shape before adjustment ", startofpulse.shape)
    #             print("endofpulse.shape before adjustment ", endofpulse.shape)

    #         if startofpulse[0] < endofpulse[0]:
    #             print("fixing first pulse start")
    #             startofpulse = startofpulse[1:]

    #         if startofpulse[-1] < endofpulse[-1]:
    #             # if IndexError: # unsafe edit just to get it working, we have to resolve it later
    #             #    pass
    #             print("fixing last pulse start")
    #             endofpulse = endofpulse[0:-1]

    #         if endofpulse[-1] > (len(self.pulsedata) - 1):
    #             print("fixing last pulse end")
    #             endofpulse = endofpulse[:-1]

    #         if verbose:
    #             print("startofpulse.shape after adjustment ", startofpulse.shape)
    #             print("endofpulse.shape after adjustment ", endofpulse.shape)

    #         #################################################################
    #         # end of corrections

    #         if verbose:
    #             check(startofpulse)
    #             check(endofpulse)

    #         # save startofpulse and endofpulse
    #         self.startofpulse = startofpulse
    #         self.endofpulse = endofpulse
    #         del startofpulse, endofpulse

    #         echotimelen = int(echotime * self.sampRate)

    #         startofacq = self.endofpulse + acqdelaylen
    #         endofacq = self.endofpulse + echotimelen - pulselen
    #         self.acq_arr = np.array([startofacq, endofacq]).transpose()
    #         if verbose:
    #             check(self.acq_arr)
    #         if len(self.acq_arr) <= 3:
    #             print("WARNING! len(self.acq_arr) <= 3. Too less pulses?")

    #         del pulselen, startofacq, endofacq

    #     else:
    #         raise ValueError("search_mod not found")

    #     # do some final checks

    #     acqdelaylen = int(self.acqDelay * self.sampRate)
    #     if acqdelaylen <= 3:
    #         print(
    #             f"!!WARNING!! acqDelay * samprate = {acqdelaylen} could be too small. "
    #         )

    #     echotimelen = int(echotime * self.sampRate)
    #     pulse_diff = np.amin(self.startofpulse - self.endofpulse)
    #     if verbose:
    #         check(echotimelen)
    #         check(pulse_diff)
    #     # these 2 above should be the same I guess? when testing: difference 10

    #     pulse_interval = np.amax(self.endofpulse) - np.amin(self.endofpulse)
    #     echotime_total = echotimelen * echonumber
    #     if verbose:
    #         check(pulse_interval)
    #         check(echotime_total)
    #     # same for these 2, difference 1595

    #     if acqdelaylen > pulse_diff:
    #         print("acqdelaylen ", acqdelaylen)
    #         print("np.amin(self.startofpulse - self.endofpulse) ", pulse_diff)
    #         raise ValueError("points of acqDelay > min(endofpulse-startofpulse)")

    #     # check if the lengths of data are too short
    #     if (acqdelaylen + acqtimelen) > np.amin(self.startofpulse - self.endofpulse):
    #         print(f"acqdelaylen + acqtimelen = {acqdelaylen + acqtimelen}")
    #         print(
    #             f"np.amin(self.startofpulse - self.endofpulse) = {np.amin(self.startofpulse - self.endofpulse)}"
    #         )
    #         check(self.startofpulse)
    #         check(self.endofpulse)
    #         raise ValueError(
    #             "(points of acqDelay + points of acqTime) > min(endofpulse-startofpulse)"
    #         )

    #     del acqdelaylen, echotimelen, pulse_diff, pulse_interval, echotime_total

    #     # plot only data within acquisition windows
    #     if plot:
    #         fig, ax1 = plt.subplots(figsize=(16, 9), dpi=100)
    #         ax2 = ax1.twinx()
    #         # ax1.plot(self.timestamp, self.pulsedata, color="g")
    #         # ax2.plot(self.timestamp, (self.dataX), color="b")

    #         FIDall = []
    #         alltimes = []
    #         for i in range(len(self.acq_arr)):

    #             timedomain = self.timeStamp[self.acq_arr[i, 0] : self.acq_arr[i, 1]]
    #             alltimes.append(timedomain)

    #             pulsedata = self.pulsedata[self.acq_arr[i, 0] : self.acq_arr[i, 1]]

    #             FIDhere = (
    #                 self.dataX[self.acq_arr[i, 0] : self.acq_arr[i, 1]]
    #                 + 1j * self.dataY[self.acq_arr[i, 0] : self.acq_arr[i, 1]]
    #             )
    #             FIDall.append(FIDhere)

    #             # ax1.plot(timedomain, pulsedata, color="g")
    #             ax2.plot(timedomain, FIDhere, color="b")
    #         fig.savefig(savepath[:-4] + "_timeseries.png")

    #     # plt.figure()
    #     # plt.plot(self.startofpulse)
    #     # plt.plot(self.endofpulse)
    #     # plt.grid()
    #     # plt.title('p')
    #     # plt.show()
    #     # plt.close()

    #     # plt.figure()
    #     # plt.plot(self.startofpulse-self.endofpulse)
    #     # plt.grid()
    #     # plt.title('self.startofpulse-self.endofpulse')
    #     # plt.show()
    #     # plt.close()

    # def GetAvgPSD(
    #     self,
    #     AVARPopt: bool = False,
    #     windowfunction="rectangle",
    #     decayfactor=-10,
    #     selectshots=[],
    #     verbose=False,
    # ):
    #     """
    #     Compute average power spectrum.

    #     Parameters
    #     ----------
    #     (not implemented yet)AVARPopt : bool
    #         Option for computing the Allan variance of power spectra. Defaults to False.

    #     windowfunction : str
    #         window function for FFT.
    #         Available choices:
    #             'rectangle'
    #             'expdecay'
    #             'Hanning' or 'hanning' or 'Han' or 'han'
    #             'Hamming' or 'hamming'
    #             'Blackman'
    #         Defaults to 'rectangle'.

    #     decayfactor : float
    #         parameter for window function 'expdecay'.
    #         The 'expdecay' window function generates an array of [exp(-df*0), exp(-df*1), ... , exp(-df*(num-1))],
    #         where df stands for decayfactor.
    #         Defaults to -10.

    #     selectshots : list
    #         List of selected shots / chunks of acquired time-series for power spectra.
    #         e.g. [0] means that you only use the first shot.
    #         Defaults to [], which means all shots / chunks are included.

    #     verbose : bool
    #         Choose True to display processing information. Defaults to False.

    #     Returns
    #     -------
    #     Null

    #     Examples
    #     --------
    #     >>>

    #     Referrence
    #     --------
    #     https://holometer.fnal.gov/GH_FFT.pdf
    #     For details, see functioncache - stdPSD and stdLIAPSD
    #     """
    #     # initialize an all-zero PSD array
    #     PSD = np.zeros(abs(self.acq_arr[0, 1] - self.acq_arr[0, 0]))

    #     # select the shots / chunks for PSD
    #     if len(selectshots) == 0 or selectshots is None:
    #         # when all shots / chunks are included
    #         selectPulse_list = range(len(self.acq_arr[:, 0]))
    #     else:
    #         # when the user selects the shots / chunks
    #         selectPulse_list = selectshots

    #     # computed accumulated PSD of selected shots / chunks
    #     for i in selectPulse_list:
    #         frequencies, singlePSD = stdLIAPSD(
    #             data_x=self.dataX[self.acq_arr[i, 0] : self.acq_arr[i, 1]],
    #             data_y=self.dataY[self.acq_arr[i, 0] : self.acq_arr[i, 1]],
    #             samprate=self.sampRate,
    #             demodFreq=self.demodFreq,
    #             attenuation=self.attenuation,
    #             windowfunction=windowfunction,
    #             decayfactor=decayfactor,
    #             showwindow=False,
    #             DTRCfilter=self.filterstatus,
    #             DTRCfilter_TC=self.filter_TC,
    #             DTRCfilter_order=self.filter_order,
    #             verbose=verbose,
    #         )
    #         PSD += singlePSD  # accumulate PSD
    #         del singlePSD  # delete singlePSD before going to the next iteration

    #     self.frequencies = frequencies  # store frequency axis
    #     self.avgPSD = PSD / len(selectPulse_list)  # store the average PSD
    #     self.selectPulse_list = (
    #         selectPulse_list  # This is for plotting in self.GetSpectrum()
    #     )
    #     del frequencies, PSD  # delete useless variables

    # def GetAvgPSD_NoiseReduction(
    #     self,
    #     windowfunction="rectangle",
    #     decayfactor=-10,
    #     selectshots=[],
    #     verbose=False,
    # ):
    #     """
    #     Compute average power spectrum, and substract the nosie background from the signal spectrum.
    #     This function is specially designed for pulsed-NMR data.
    #     The noise background is generated by the latter part of the acquired time-series,
    #     where the decay signal is weak, while the noise is the same.
    #     A schematic is included here:
    #         ┌───────┐                                                   ┌───────┐
    #         |       |                                                   |       |
    #     ────┘       └───────────────────────────────────────────────────┘
    #                 |←   →|←             →|       |←             →|←   →|
    #                   (a)       (b)                      (c)        (d)
    #     caption: (a) acqDelay. (b) acqTime for signal PSD. (c) acqTime for noise PSD.
    #     (d) safe space before the next pulse so that the noise PSD is not
    #     influenced by the next pulse. The length is chosen to be the same as acqDelay.

    #     This method can reduce or elimate noise peaks that are independent of excitation pulses,
    #     while keeping the decay signal.

    #     Parameters
    #     ----------
    #     windowfunction : str
    #         window function for FFT.
    #         Available choices:
    #             'rectangle'
    #             'expdecay'
    #             'Hanning' or 'hanning' or 'Han' or 'han'
    #             'Hamming' or 'hamming'
    #             'Blackman'
    #         Defaults to 'rectangle'.

    #     decayfactor : float
    #         parameter for window function 'expdecay'. Defaults to -10.
    #         The 'expdecay' window function generates an array of [exp(-df*0), exp(-df*1), ... , exp(-df*(num-1))],
    #         where df stands for decayfactor.

    #     selectshots : list
    #         List of selected shots / chunks of acquired time-series for power spectra.
    #         e.g. [0] means that you only use the first shot.
    #         Defaults to [], which means all shots / chunks are included.

    #     verbose : bool
    #         Choose True to display processing information. Defaults to False.

    #     Returns
    #     -------
    #     Null

    #     Examples
    #     --------
    #     >>>

    #     Referrence
    #     --------
    #     https://holometer.fnal.gov/GH_FFT.pdf
    #     For details, see functioncache - stdPSD and stdLIAPSD
    #     """

    #     # select the shots / chunks for PSD
    #     if len(selectshots) == 0 or selectshots is None:
    #         # when all shots / chunks are included
    #         selectPulse_list = range(len(self.acq_arr[:, 0]))
    #     else:
    #         # when the user selects the shots / chunks
    #         selectPulse_list = selectshots

    #     # get the length of reptime (intervals between pulses)
    #     reptimelen_list = []
    #     for i in selectPulse_list:
    #         reptimelen_list.append(self.startofpulse[i] - self.endofpulse[i])

    #     # check if the reptime's / intervals between pulses are large enough
    #     if np.amin(reptimelen_list) < 2 * (self.acqDelayLen + self.acqTimeLen):
    #         print(
    #             f"!!WARNING!! minimum reptimelen < 2 * (acqdelaylen + acqtimelen). You may need to increase repetiton time. Otherwise you could weaken the signal. "
    #         )

    #     del selectPulse_list, reptimelen_list

    #     # extract noise spectrum (noise background)
    #     startofacq = self.startofpulse - (self.acqDelayLen + self.acqTimeLen)
    #     self.acq_arr = np.array([startofacq, startofacq + self.acqTimeLen]).transpose()
    #     self.GetAvgPSD(
    #         windowfunction=windowfunction,
    #         decayfactor=decayfactor,
    #         selectshots=selectshots,
    #         verbose=verbose,
    #     )
    #     noisespectrum = self.avgPSD
    #     del self.avgPSD

    #     # extract signal spectrum
    #     startofacq = self.endofpulse + self.acqDelayLen
    #     self.acq_arr = np.array([startofacq, startofacq + self.acqTimeLen]).transpose()
    #     self.GetAvgPSD(
    #         windowfunction=windowfunction,
    #         decayfactor=decayfactor,
    #         selectshots=selectshots,
    #         verbose=verbose,
    #     )

    #     # substract the nosie background from the signal spectrum
    #     self.avgPSD -= noisespectrum

    #     del startofacq, noisespectrum  # delete useless variables

    # def Get2Domain_chanstd(
    #     self,
    #     windowfunction="rectangle",
    #     decayfactor=-10,
    #     selectshots=[],
    #     plot_opt=False,
    #     verbose=False,
    # ):
    #     """
    #     This function is designed for searching for the NMR signal in pulsed-NMR scheme.
    #     The NMR signal decays after the excitation pulse. We should observe this dacay
    #     not only in time-series, but also in the frequency channels of the Larmor
    #     frequency if we generate multiple power spectra from slices of the time-series.
    #     Such decay can be found by computing the standard deviation of the values in a
    #     frequency bin of different power spectra.

    #     Parameters:
    #     --------------
    #     windowfunction : str
    #         optional Defaults to 'rectangle'.
    #     decayfactor : int, optional
    #         Defaults to -10.
    #     selectshots : list, optional
    #         Defaults to [].
    #     verbose : bool, optional
    #         Defaults to False.
    #     """
    #     self.chanstd_flag = False
    #     if len(self.endofpulse) == 1:
    #         return
    #     if selectshots is None or len(selectshots) == 0:
    #         selectshots = range(len(self.endofpulse))

    #     # else:
    #     #     selectPulse_list = selectshots
    #     # numofselectshots = len(selectPulse_list)

    #     # if self.startofpulse[0] < self.endofpulse[0]:
    #     #     interval = self.startofpulse[1] - self.endofpulse[0]
    #     # else:
    #     #     interval = self.endofpulse[0] - self.startofpulse[0]

    #     chanstd_list = []  # what is this?

    #     # find the longest pulse interval
    #     # pulseInterval_list = []
    #     numofPSD_list = []
    #     for i in selectshots:
    #         # pulseInterval_list.append(interval)
    #         interval = self.startofpulse[i] - self.endofpulse[i]
    #         numofPSD = (interval - 2 * self.acqDelayLen) // self.acqTimeLen
    #         if numofPSD < 2:
    #             print(
    #                 f"selected shot: {i}. numofPSD < 2. Insufficient amount of power spectra for computing standard deviation. "
    #             )
    #             return
    #         if numofPSD < 3:
    #             print("Warning: numofPSD < 3. ")
    #         numofPSD_list.append(numofPSD)
    #     # check(np.amax(numofPSD_list))

    #     for i in selectshots:
    #         PSD2D_list = []
    #         for j in range(numofPSD_list[i]):
    #             frequencies, singlePSD = stdLIAPSD(
    #                 data_x=self.dataX[
    #                     self.acq_arr[i, 0]
    #                     + j * self.acqTimeLen : self.acq_arr[i, 1]
    #                     + j * self.acqTimeLen
    #                 ],
    #                 data_y=self.dataY[
    #                     self.acq_arr[i, 0]
    #                     + j * self.acqTimeLen : self.acq_arr[i, 1]
    #                     + j * self.acqTimeLen
    #                 ],
    #                 samprate=self.sampRate,  # in Hz
    #                 demodFreq=self.demodFreq,  # in Hz
    #                 attenuation=self.attenuation,  # in dB. Power ratio (10^(attenuation/10))
    #                 windowfunction=windowfunction,  # Hanning, Hamming, Blackman
    #                 decayfactor=decayfactor,
    #                 showwindow=False,
    #                 DTRCfilter=self.filterstatus,
    #                 DTRCfilter_TC=self.filter_TC,
    #                 DTRCfilter_order=self.filter_order,
    #                 verbose=verbose,
    #             )
    #             PSD2D_list.append(singlePSD)
    #             del singlePSD
    #         PSD2D_arr = np.array(PSD2D_list)
    #         # PSD2D_chanmean = np.mean(PSD2D_arr,axis=0)
    #         chanstd_list.append(np.std(PSD2D_arr, axis=0) / np.mean(PSD2D_arr, axis=0))
    #         del PSD2D_list, PSD2D_arr
    #     self.chanstd_arr = np.array(chanstd_list)
    #     self.chanstd_avg = np.mean(self.chanstd_arr, axis=0)
    #     # check(frequencies.shape)
    #     # check(self.chanstd_arr.shape)
    #     # check(self.chanstd_avg.shape)
    #     self.chanstd_flag = True
    #     if plot_opt == True:
    #         print("This function is not ready yet. ")

    # def GetAvgFIDsq(self, verbose=False):
    #     print(
    #         "It is not recommended to use AvgFIDsq now, because the time-series is not necessarily FID. "
    #         "Instead, it is recommended to use AvgTSsq"
    #     )
    #     sumofFID = 0.0
    #     self.AvgFIDsq = 0.0
    #     numofpulses = len(self.acq_arr[:, 0])
    #     for i in range(numofpulses):
    #         FIDHere = (
    #             self.dataX[self.acq_arr[i, 0] : self.acq_arr[i, 1]]
    #             + 1j * self.dataY[self.acq_arr[i, 0] : self.acq_arr[i, 1]]
    #         )
    #         sumofFID += np.sum(abs(FIDHere) ** 2)
    #     if verbose:
    #         plt.figure()
    #         plt.plot(np.real(sumofFID) / numofpulses, label="np.real(sumofFID)")
    #         plt.plot(np.imag(sumofFID) / numofpulses, label="np.imag(sumofFID)")
    #         # plt.scatter(np.real(sumofFID))
    #         # plt.scatter(np.imag(sumofFID))
    #         plt.grid()
    #         plt.title("Averaged FID")
    #         plt.show()
    #     self.AvgFIDsq = np.real_if_close(sumofFID) / numofpulses
    #     self.avgTSsq = self.AvgFIDsq
    #     del sumofFID, numofpulses

    # def GetAvgTSsq(self, verbose=False):
    #     """
    #     Compute the time-series mean square. The result is stored in self.AvgTSsq

    #     Parameters
    #     ----------
    #     verbose : bool, optional
    #         Defaults to False.

    #     Returns
    #     -------
    #     Null

    #     """
    #     sumofTSsq = 0.0
    #     TSsq_vals = []
    #     self.avgTSsq = 0.0
    #     numofpulses = len(self.acq_arr[:, 0])
    #     for i in range(numofpulses):
    #         TSHere = (
    #             self.dataX[self.acq_arr[i, 0] : self.acq_arr[i, 1]]
    #             + 1j * self.dataY[self.acq_arr[i, 0] : self.acq_arr[i, 1]]
    #         )
    #         TSsq = np.sum(abs(TSHere) ** 2)
    #         sumofTSsq += np.sum(abs(TSHere) ** 2)
    #         TSsq_vals.append(TSsq)
    #     if verbose:
    #         plt.figure()
    #         plt.plot(np.real(sumofTSsq) / numofpulses, label="np.real(sumofTS)")
    #         plt.plot(np.imag(sumofTSsq) / numofpulses, label="np.imag(sumofTS)")
    #         # plt.scatter(np.real(sumofTS))
    #         # plt.scatter(np.imag(sumofTS))
    #         plt.grid()
    #         plt.title("Averaged TS square")
    #         plt.show()
    #     self.avgTSsq = np.real_if_close(sumofTSsq) / numofpulses
    #     self.TSsq_std = np.std(TSsq_vals)
    #     self.AvgFIDsq = self.avgTSsq
    #     del sumofTSsq, numofpulses
    #     return self.avgTSsq

    # def GetTSpower(self, selectshots=[], verbose: bool = False):
    #     """
    #     Compute the power of time-series. The result is stored in self.TSPower

    #     Parameters
    #     ----------
    #     selectshots : list, optional
    #         List of selected shots / chunks of acquired time-series for power spectra.
    #         e.g. [0] means that you only use the first shot.
    #         Defaults to [], which means all shots / chunks are included.
    #     verbose : bool, optional
    #         Defaults to False.

    #     Returns
    #     -------
    #     self.TSPower

    #     """
    #     # select the shots / chunks for PSD
    #     if len(selectshots) == 0 or selectshots is None:
    #         # when all shots / chunks are included
    #         selectPulse_list = range(len(self.acq_arr[:, 0]))
    #     else:
    #         # when the user selects the shots / chunks
    #         selectPulse_list = selectshots

    #     # initialize sum of time-series square
    #     meanofTSsq = 0.0
    #     # loop over selectPulse_list to sum up the time-series square
    #     for i in selectPulse_list:
    #         TSHere = (
    #             self.dataX[self.acq_arr[i, 0] : self.acq_arr[i, 1]]
    #             + 1j * self.dataY[self.acq_arr[i, 0] : self.acq_arr[i, 1]]
    #         )
    #         meanofTSsq += np.mean(abs(TSHere) ** 2)

    #     # Compute the power of time-series
    #     # Power = 1 / T * Integrate_{0}^{T} s^2(t) dt
    #     #       = 1 / (N * 1 / SR) * Sum_{i=0}^{i=N-1} s^2_i * (1 / SR)
    #     #       = 1 / N * Sum_{i=0}^{i=N-1} s^2_i , where SR stands for sampling rate
    #     self.TSpower = np.real_if_close(meanofTSsq) / len(selectPulse_list)
    #     if verbose:
    #         check(self.TSpower)
    #     del meanofTSsq
    #     return self.TSpower

    # def GetFFTpower(self, verbose: bool = False):
    #     """
    #     Compute the power of FT spectrum. The result is stored in self.FFTpower

    #     Parameters
    #     ----------
    #     verbose : bool, optional
    #         Defaults to False.

    #     Returns
    #     -------
    #     self.FFTpower

    #     """
    #     assert hasattr(self, "avgFFT")
    #     assert hasattr(self, "freq_resol")
    #     self.FFTpower = np.sum(np.abs(self.avgFFT) ** 2.0) * self.freq_resol
    #     return self.FFTpower

    # def GetPSDpower(self, verbose: bool = False):
    #     """
    #     Compute the power of PSD. The result is stored in self.PSDpower

    #     Parameters
    #     ----------
    #     verbose : bool, optional
    #         Defaults to False.

    #     Returns
    #     -------
    #     self.FFTpower

    #     """
    #     assert hasattr(self, "avgPSD")
    #     assert hasattr(self, "freq_resol")
    #     self.PSDpower = np.sum(self.avgPSD) * self.freq_resol
    #     self.PSDpower

    # # dealing with CPMG measurements
    # def GetEchoAmplitudes(
    #     self,
    #     pulseduration: float,
    #     echonumber: int,
    #     echotime: float,
    #     pulseampli: float,
    #     PeriodN: int,
    #     WinPercent: float,
    #     savepath: str,
    #     method="NPeriodMaximum",
    #     fitmethod="exp",
    #     showlegend=True,
    #     figsize=(16, 9),
    #     fontsize=22,
    #     tightlayout_opt=True,
    #     showplot=False,
    #     saveplot=False,
    #     verbose=False,
    # ):

    #     numofechoes = int(len(self.acq_arr[:, 0]))
    #     if verbose:
    #         check(numofechoes)

    #     def linear(x, a, b):
    #         return a * x + b

    #     def exp_decay(x, A, T, B):
    #         return A * np.exp(-x / T) + B

    #     def exp_decay2(x, A, T):
    #         return A * np.exp(-x / T)

    #     def exp_decay_osc(x, A, T, B, C, w, phase):
    #         return A * np.exp(-x / T) + B + C * np.cos(2 * np.pi * w * x + phase)

    #     def CPMG_fixed_phase(echo_num, T2, T1, alpha):

    #         gyro = 42.5774825e6 * 2 * np.pi  # Hz/T
    #         B1freq = 1348570.0  # Hz
    #         B1amp = pulseampli
    #         # T1 = 5 # s

    #         # alpha = gyro * B1amp * pulseduration / 360*2*np.pi
    #         # alpha = 160 / 360*2*np.pi
    #         # print(alpha)

    #         kappa = echotime * (1 / T2 - 1 / T1) * 1 / 2
    #         kappa_0 = echotime * (1 / T1 + 1 / T2) * 1 / 2
    #         kappa_s = kappa_0 / echotime  # = (1/T1 + 1/T2) / 2
    #         kappa_sin = kappa * np.sin(alpha / 2) ** 2
    #         kappa_cos = kappa * np.cos(alpha / 2) ** 2

    #         # Bessel functions
    #         def J0(x):
    #             # return np.sqrt(2 / (np.pi*x)) * np.sin(np.pi/4 + x)
    #             return scipy.special.j0(x)

    #         def J1(x):
    #             # return -np.sqrt(2 / (np.pi*x)) * np.cos(np.pi/4 + x)
    #             return scipy.special.j1(x)

    #         def I0(x):
    #             # return np.exp(x) / (np.sqrt(2*np.pi*x)) * (1 + 1/(8*x))
    #             return scipy.special.i0(x)

    #         def I1(x):
    #             # return np.exp(x) / (np.sqrt(2*np.pi*x)) * (1 - 3/(8*x))
    #             return scipy.special.i0(x)

    #         # Sigmoid function(n*(pi-alpha))
    #         sigma = 1 / (1 + np.exp(10 * (1 - (np.pi - alpha) * echo_num)))
    #         # sigma = scipy.special.expit(-10*(1-(np.pi-alpha)*echo_num))

    #         # magnetization factors
    #         M0 = (
    #             np.sin(alpha / 2)
    #             * np.exp(-kappa * echo_num)
    #             * J0(echo_num * (np.pi - alpha))
    #         )

    #         M1 = (
    #             (-1) ** (echo_num - 1)
    #             * np.sqrt(alpha * np.tan(alpha / 2))
    #             * J1(echo_num * alpha)
    #             / (2 * np.sqrt(2))
    #         )

    #         M2 = (
    #             -kappa_s
    #             / (2 * np.cos(alpha / 2))
    #             * np.exp(-kappa_cos * echo_num)
    #             * (I0(kappa_s * echo_num) - I1(kappa_s * echo_num))
    #         )

    #         Ampfactor = np.exp(-kappa_0 * echo_num)
    #         AmpFinal = (1 - sigma) * Ampfactor * M0 + sigma * Ampfactor * (M1 + M2)
    #         # AmpFinal = Ampfactor*M1 - Ampfactor*M2
    #         return AmpFinal

    #     def CPMG_alternat_phase(echo_num, T2, T1, alpha):

    #         kappa = echotime * (1 / T2 - 1 / T1) * 1 / 2
    #         kappa_0 = echotime * (1 / T1 + 1 / T2) * 1 / 2
    #         kappa_s = kappa_0 / echotime  # = (1/T1 + 1/T2) / 2
    #         kappa_sin = kappa * np.sin(alpha / 2) ** 2
    #         kappa_cos = kappa * np.cos(alpha / 2) ** 2

    #         # Bessel functions
    #         def J0(x):
    #             return scipy.special.j0(x)

    #         def J1(x):
    #             return scipy.special.j1(x)

    #         def I0(x):
    #             return scipy.special.i0(x)

    #         def I1(x):
    #             return scipy.special.i0(x)

    #         # Sigmoid function(n*alpha)
    #         sigma = 1 / (1 + np.exp(5 * (3 / 2 - alpha * echo_num)))

    #         # magnetization factors
    #         M0 = np.sin(alpha) * np.exp(-kappa * echo_num) * J1(alpha * echo_num / 2)

    #         M1 = (
    #             np.sin(alpha / 2)
    #             * np.exp(-kappa_s * echo_num)
    #             * I0(kappa_cos * echo_num)
    #         )

    #         M2 = (
    #             (-1) ** echo_num
    #             * (kappa / 4)
    #             * np.sin(alpha / 2)
    #             * np.sin(alpha)
    #             * np.exp(-kappa_cos * echo_num)
    #             * (I0(kappa_s * echo_num) + I1(kappa_s * echo_num))
    #         )

    #         M3 = (
    #             -J0(alpha * echo_num)
    #             / (2 * np.sqrt(2) * echo_num)
    #             * np.sqrt(alpha / np.tan(alpha / 2))
    #         )

    #         Ampfactor = np.exp(-kappa_0 * echo_num)
    #         AmpFinal = (1 - sigma) * Ampfactor * M0 + sigma * Ampfactor * (M1 + M2 + M3)
    #         return AmpFinal

    #     amplitudes = []

    #     if method == "NPeriodMaximum":

    #         # frequency of oscillations of time domain signal = difference between larmor freq and demod freq
    #         # we obtain resonance freq from FitPSD() parameters
    #         # check(self.demodFreq)
    #         # check(self.popt[0])
    #         tau = 1.0 / abs(self.popt[0] - self.demodFreq)  # [s]
    #         if verbose:
    #             check(tau)
    #         tau = int(tau * self.sampRate)  # convert to bins
    #         check(tau)

    #         for i in range(0, numofechoes):

    #             acqinterval = range(self.acq_arr[i, 0], self.acq_arr[i, 1])
    #             # check(acqinterval)
    #             if tau * PeriodN >= len(acqinterval):
    #                 raise ValueError("interval to look for maximum > acquisition time")
    #             echo = (
    #                 self.dataX[
    #                     int(np.average(acqinterval))
    #                     - tau * PeriodN : int(np.average(acqinterval))
    #                     + tau * PeriodN
    #                 ]
    #                 + 1j
    #                 * self.dataY[
    #                     int(np.average(acqinterval))
    #                     - tau * PeriodN : int(np.average(acqinterval))
    #                     + tau * PeriodN
    #                 ]
    #             )
    #             # echo = abs(echo)**2
    #             # check(echo)
    #             # check(np.amax(echo))
    #             # if i > 10:
    #             #    plt.figure()
    #             #    plt.plot(np.real(echo))
    #             #    plt.grid()
    #             #    plt.title('echo')
    #             #    plt.show()
    #             amplitudes.append(np.sqrt(np.sum(abs(echo) ** 2)))
    #         # plt.figure()
    #         # plt.plot(amplitudes)
    #         # plt.grid()
    #         # plt.title('amplitudes')
    #         # plt.show()

    #         del numofechoes, acqinterval, echo, tau

    #     elif method == "FitDecay":

    #         # print(numofechoes)
    #         for i in range(numofechoes):

    #             acqinterval = range(self.acq_arr[i, 0], self.acq_arr[i, 1])
    #             # check(len(acqinterval))
    #             acqtimedomain = self.timeStamp[
    #                 int(np.average(acqinterval)) : self.acq_arr[i, 1]
    #             ]
    #             echoFID = (
    #                 self.dataX[int(np.average(acqinterval)) : self.acq_arr[i, 1]]
    #                 + 1j * self.dataY[int(np.average(acqinterval)) : self.acq_arr[i, 1]]
    #             )
    #             echoFID = abs(echoFID) ** 2

    #             # plt.figure()
    #             # plt.plot(acqtimedomain, echoFID)
    #             # plt.show()
    #             # check(len(acqtimedomain))

    #             x_data = acqtimedomain
    #             y_data = echoFID
    #             fitfunc = exp_decay  # does not converge
    #             initial_guess = [1, 0.1, 1, 0, 0]
    #             popt, pcov = curve_fit(fitfunc, x_data, y_data)  # , p0=initial_guess)

    #             showfit = True
    #             if showfit:
    #                 plt.scatter(x_data, y_data, label="Original Data")
    #                 plt.plot(x_data, fitfunc(x_data, *popt), "r-", label="Fitted Curve")
    #                 plt.legend()
    #                 plt.xlabel("x")
    #                 plt.ylabel("y")
    #                 plt.title("Oscillating Exponential Decay Curve Fitting")
    #                 plt.show()
    #                 plt.figure()

    #             check(popt)
    #             amplitudes.append(popt[0])  # amplitude A, hopefully

    #         del acqinterval, numofechoes, acqtimedomain, echoFID, popt

    #     elif method == "WindowFraction":

    #         for i in range(0, numofechoes):

    #             # we go from the center of the acquisition window and take all the data within 5% above and below it
    #             # this way we have captured the part of the echo that sits within 10% between the middle of
    #             # 2 subsequent pulses, and we are less likely to capture noise which might dominate closer to the edges

    #             acqinterval = range(self.acq_arr[i, 0], self.acq_arr[i, 1])
    #             # check(acqinterval)
    #             WindowFraction = int(len(acqinterval) * WinPercent)
    #             # check(WindowFraction)

    #             echo = (
    #                 self.dataX[
    #                     int(np.average(acqinterval))
    #                     - WindowFraction : int(np.average(acqinterval))
    #                     + WindowFraction
    #                 ]
    #                 + 1j
    #                 * self.dataY[
    #                     int(np.average(acqinterval))
    #                     - WindowFraction : int(np.average(acqinterval))
    #                     + WindowFraction
    #                 ]
    #             )

    #             # if i > 10:
    #             #    plt.figure()
    #             #    plt.plot(np.real(echo))
    #             #    plt.grid()
    #             #    plt.title('echo')
    #             #    plt.show()

    #             amplitudes.append(np.sqrt(np.sum(abs(echo) ** 2)))

    #         del numofechoes, acqinterval, WindowFraction, echo

    #     else:
    #         raise OSError(
    #             "define method for obtaining echo amplitude: either NPeriodMaximum or FitDecay"
    #         )

    #     if verbose:
    #         check(amplitudes)

    #     measuretime = (
    #         self.timeStamp[self.acq_arr[:, 0]] - self.timeStamp[self.endofpulse[0]]
    #     )
    #     # check(len(measuretime))
    #     echo_numbers = np.arange(1, len(measuretime) + 1)
    #     # check(len(measuretime))

    #     # popt = curve_fit(linear, measuretime, amplitudes)
    #     try:

    #         y_data = amplitudes

    #         if fitmethod == "exp":
    #             x_data = measuretime
    #             fitfunc = exp_decay
    #             fitparas = []
    #             popt, pcov = curve_fit(fitfunc, x_data, y_data)
    #             self.T2 = popt[1]
    #             self.T2Uncertainty = np.sqrt(pcov[1, 1])
    #         elif fitmethod == "exp_osc":
    #             x_data = measuretime
    #             # freq = abs(self.popt[0] - self.demodFreq)
    #             fitfunc = exp_decay_osc
    #             popt0, pcov0 = curve_fit(exp_decay, x_data, y_data)
    #             fitparas = [popt0[0], popt0[1], popt0[2], 1e-4, 5.0, np.pi / 2]
    #             popt, pcov = curve_fit(fitfunc, x_data, y_data, p0=fitparas)
    #             self.T2 = popt[1]
    #             self.T2Uncertainty = np.sqrt(pcov[1, 1])
    #         elif fitmethod == "CPMG_fix":
    #             x_data = echo_numbers
    #             fitfunc = CPMG_fixed_phase
    #             popt0, pcov0 = curve_fit(exp_decay, x_data, y_data)
    #             fitparas = [popt0[1], 5.0, 170]  # /360*2*np.pi
    #             popt, pcov = curve_fit(fitfunc, x_data, y_data, p0=fitparas)
    #             self.T2 = popt[0]
    #             errors = np.sqrt(np.diag(pcov))
    #             self.T2Uncertainty = errors[0]
    #             if np.isnan(self.T2Uncertainty) or np.isinf(self.T2Uncertainty):
    #                 self.T2Uncertainty = 0
    #         elif fitmethod == "CPMG_alt":
    #             x_data = echo_numbers
    #             fitfunc = CPMG_alternat_phase
    #             popt0, pcov0 = curve_fit(exp_decay, x_data, y_data)
    #             fitparas = [popt0[1], 5.0, 170]  # /360*2*np.pi
    #             popt, pcov = curve_fit(fitfunc, x_data, y_data, p0=fitparas)
    #             self.T2 = popt[0]
    #             errors = np.sqrt(np.diag(pcov))
    #             self.T2Uncertainty = errors[0]
    #             if np.isnan(self.T2Uncertainty) or np.isinf(self.T2Uncertainty):
    #                 self.T2Uncertainty = 0

    #         plt.rcParams["font.family"] = "Times New Roman"
    #         plt.figure(figsize=figsize, dpi=100)

    #         plt.scatter(x_data, amplitudes / np.max(amplitudes))
    #         plt.grid()
    #         # plt.title(rf"Spin Echo amplitude in a ${PeriodN}\cdot T $ interval around the echotime center")#\n + self.file
    #         plt.xlabel("Measurement time [s]")
    #         plt.ylabel(f"Spin echo amplitudes [arbitrary units]")

    #         # plt.yticks(np.arange(0, np.max(amplitudes), 0.1))
    #         plt.gca().set_ylim(bottom=0)

    #         # plt.plot(x_data, fitfunc(x_data,*fitparas), color="green", label=f"Exp fit initial guess")
    #         plt.plot(
    #             x_data,
    #             fitfunc(x_data, *popt) / np.max(amplitudes),
    #             color="red",
    #             label=rf"Fitted $T_2 = {self.T2:.3f}~\\pm~{self.T2Uncertainty:.3f}~s$",
    #         )

    #         # plt.title(self.file)
    #         if showlegend:
    #             plt.legend()
    #         if tightlayout_opt:
    #             plt.tight_layout()
    #         if saveplot:
    #             plt.savefig(savepath)
    #         if showplot:
    #             plt.show()
    #         # del plt
    #         plt.close()

    #     except RuntimeError:
    #         print("fit did not converge :(")
    #         self.T2 = 0
    #         self.T2Uncertainty = 0

    #     return x_data, amplitudes, popt, pcov

    # def GetEchoSqInt(self, selectshots=[], verbose=False):

    #     if selectshots is None or len(selectshots) == 0:
    #         selectshots = range(len(self.acq_arr[:, 0]))
    #     sumofechoes = 0.0
    #     for i in selectshots:
    #         echo = (
    #             self.dataX[self.acq_arr[i, 0] : self.acq_arr[i, 1]]
    #             + 1j * self.dataY[self.acq_arr[i, 0] : self.acq_arr[i, 1]]
    #         )
    #         sumofechoes += np.sum((abs(echo)) ** 2)
    #     self.EchoSqInt = np.real_if_close(sumofechoes) / self.sampRate
    #     if verbose:
    #         check(self.EchoSqInt)

    # def FitPSD(
    #     self,
    #     fitfunction="Lorentzian",  # 'Lorentzian' 'dualLorentzian' 'tribLorentzian' 'Gaussian 'dualGaussian'
    #     inputfitparas=["auto", "auto", "auto", "auto"],
    #     smooth=False,
    #     smoothlevel=1,
    #     fitrange=[0, -1],
    #     alpha=0.05,
    #     getresidual=False,
    #     getchisq=False,
    #     verbose=False,
    # ):
    #     """
    #     Fit on power spectral line.

    #     Parameters
    #     ----------
    #     fitfunction : str
    #         Name of the fit function.
    #         All options: 'Lorentzian' 'dualLorentzian' 'tribLorentzian' 'Gaussian 'dualGaussian'
    #         'Polyeven' and 'auto' (all the options in FunctionParas.keys()).
    #         Defaults to 'Lorentzian'

    #     inputfitparas : list
    #         The initial guess for parameters.
    #         Defaults to ['auto','auto','auto','auto'],

    #     smooth : bool, optional
    #         Option to smooth the spectral line by moving average before fitting.
    #         Defaults to False.

    #     smoothlevel : int, optional
    #         The extend to which the spectral line is smoothed.
    #         Higher the value, smoother the line.
    #         Defaults to 1.
    #         To know more about the effects of it, check figures in
    #         CASPEr Code\Supplementary\Smoothforautofitting

    #     fitrange : list, optional
    #         The index range for the fitting.
    #         Defaults to [0, -1], which means the whole spectrum would be fed for fitting.

    #     alpha : float, optional
    #         The parameter for confidence interval (CI).
    #         Refer to https://en.wikipedia.org/wiki/Confidence_interval
    #         Defaults to 0.05.

    #     getresidual : bool, optional
    #         The option to get residual values after fitting.
    #         Defaults to False.

    #     getchisq : bool, optional
    #         The option to conduct chi-square test.
    #         Refer to https://en.wikipedia.org/wiki/Chi-squared_test
    #         Defaults to False.

    #     verbose : bool, optional
    #         Defaults to False.

    #     Returns
    #     --------
    #     if the fitting is unsuccessful, returns nothing
    #     else, returns
    #     self.popt : array
    #         result of fir parameters

    #     tval*self.perr : array
    #         The error in fit parameters under the specified confidence level
    #     """
    #     fitfunction = fitfunction.upper()
    #     # If the fitfunction is 'auto', all possible fitting would be tried.
    #     if fitfunction in AUTO_LIST:
    #         FitResult_arr = []
    #         fitfunction_list = list(FunctionParas.keys())
    #         for i in range(len(fitfunction_list)):
    #             self.FitPSD(
    #                 fitfunction=fitfunction_list[i],
    #                 smooth=smooth,
    #                 smoothlevel=smoothlevel,
    #                 fitrange=fitrange,
    #                 alpha=alpha,
    #                 getresidual=True,
    #                 getchisq=getchisq,
    #                 verbose=False,
    #             )
    #             FitResult_arr.append(
    #                 [self.residualval, self.chisq]
    #             )  # be careful with None here
    #         FitResult_arr = np.array(FitResult_arr)

    #         # The fitting with least residualval wins
    #         fitfunction = Function_dict[np.argmin(FitResult_arr[:, 0])][0]
    #         if verbose:
    #             print("Best fit function found: ", fitfunction)

    #     # initialize variables
    #     self.fitflag = False
    #     self.popt = None
    #     self.pcov = None
    #     self.residual = None
    #     self.residualval = None
    #     self.chisq = None
    #     self.fitcurve = []

    #     # analysis range
    #     ar = [0, len(self.avgPSD)]
    #     if fitrange is not None:
    #         for i, freqindex in enumerate(fitrange):
    #             if type(freqindex) == int:
    #                 ar[i] = freqindex
    #             if type(freqindex) == float:
    #                 print(
    #                     "WARNING!! fitrange should be index range (int) instead of float. The input fitrange may not be accepted. "
    #                 )

    #     # get the curve function from the dictionary
    #     curvefunction = Function_dict[fitfunction][1]

    #     # estimate fit parameters
    #     fitparas = Function_dict[fitfunction][2](
    #         datax=self.frequencies[ar[0] : ar[1]],
    #         datay=self.avgPSD[ar[0] : ar[1]],
    #         smooth=smooth,
    #         smoothlevel=smoothlevel,
    #         verbose=verbose,
    #     )

    #     # update fit parameters if the user specifies
    #     for i, input in enumerate(inputfitparas):
    #         if type(input) == int or type(input) == float:
    #             fitparas[i] = 1.0 * input

    #     # try to fit
    #     # check(fitparas)
    #     try:
    #         if verbose:
    #             print("start " + Function_dict[fitfunction][0] + " fitting")
    #         self.popt, self.pcov = scipy.optimize.curve_fit(
    #             curvefunction,
    #             self.frequencies[ar[0] : ar[1]],
    #             self.avgPSD[ar[0] : ar[1]],
    #             p0=fitparas,
    #             absolute_sigma=False,
    #         )
    #         self.fitflag = True
    #         if verbose:
    #             print(
    #                 "Obtained " + Function_dict[fitfunction][0] + " fit parameters ",
    #                 [x for x in self.popt],
    #             )
    #     except RuntimeError:
    #         print("fit RuntimeError")
    #     else:
    #         pass
    #     # del fitparas

    #     # If the fitting is unsuccessful, terminate the program.
    #     if self.fitflag is False:
    #         return

    #     # If the fitting is successful, generate a fit results, analysis and report.
    #     self.perr = np.sqrt(np.diag(self.pcov))
    #     dof = max(
    #         0, len(self.frequencies[ar[0] : ar[1]]) - len(self.popt)
    #     )  # degree of freedom
    #     # student-t value for the dof and confidence level
    #     tval = scipy.stats.distributions.t.ppf(1.0 - alpha / 2.0, dof)

    #     # Generate the array of fit curves
    #     if Function_dict[fitfunction][0] == "Lorentzian":
    #         self.popt[1] = abs(self.popt[1])
    #         self.fitcurve.append(
    #             Lorentzian(
    #                 self.frequencies[ar[0] : ar[1]],
    #                 self.popt[0],
    #                 self.popt[1],
    #                 self.popt[2],
    #                 self.popt[3],
    #             )
    #         )
    #     elif Function_dict[fitfunction][0] == "dualLorentzian":
    #         self.popt[1] = abs(self.popt[1])
    #         self.popt[4] = abs(self.popt[4])
    #         self.fitcurve.append(
    #             Lorentzian(
    #                 self.frequencies[ar[0] : ar[1]],
    #                 self.popt[0],
    #                 self.popt[1],
    #                 self.popt[2],
    #                 self.popt[6] / 2,
    #             )
    #         )
    #         self.fitcurve.append(
    #             Lorentzian(
    #                 self.frequencies[ar[0] : ar[1]],
    #                 self.popt[3],
    #                 self.popt[4],
    #                 self.popt[5],
    #                 self.popt[6] / 2,
    #             )
    #         )
    #         self.fitcurve.append(
    #             dualLorentzian(
    #                 self.frequencies[ar[0] : ar[1]],
    #                 self.popt[0],
    #                 self.popt[1],
    #                 self.popt[2],
    #                 self.popt[3],
    #                 self.popt[4],
    #                 self.popt[5],
    #                 self.popt[6],
    #             )
    #         )
    #     elif Function_dict[fitfunction][0] == "tribLorentzian":
    #         self.popt[1] = abs(self.popt[1])
    #         self.popt[4] = abs(self.popt[4])
    #         self.popt[7] = abs(self.popt[7])
    #         self.fitcurve.append(
    #             Lorentzian(
    #                 self.frequencies[ar[0] : ar[1]],
    #                 self.popt[0],
    #                 self.popt[1],
    #                 self.popt[2],
    #                 self.popt[9],
    #             )
    #         )
    #         self.fitcurve.append(
    #             Lorentzian(
    #                 self.frequencies[ar[0] : ar[1]],
    #                 self.popt[3],
    #                 self.popt[4],
    #                 self.popt[5],
    #                 self.popt[9],
    #             )
    #         )
    #         self.fitcurve.append(
    #             Lorentzian(
    #                 self.frequencies[ar[0] : ar[1]],
    #                 self.popt[6],
    #                 self.popt[7],
    #                 self.popt[8],
    #                 self.popt[9],
    #             )
    #         )
    #         self.fitcurve.append(
    #             tribLorentzian(
    #                 self.frequencies[ar[0] : ar[1]],
    #                 self.popt[0],
    #                 self.popt[1],
    #                 self.popt[2],
    #                 self.popt[3],
    #                 self.popt[4],
    #                 self.popt[5],
    #                 self.popt[6],
    #                 self.popt[7],
    #                 self.popt[8],
    #                 self.popt[9],
    #             )
    #         )
    #     elif Function_dict[fitfunction][0] == "Gaussian":
    #         self.fitcurve.append(
    #             Gaussian(
    #                 self.frequencies[ar[0] : ar[1]],
    #                 self.popt[0],
    #                 self.popt[1],
    #                 self.popt[2],
    #                 self.popt[3],
    #             )
    #         )
    #     elif Function_dict[fitfunction][0] == "dualGaussian":
    #         self.fitcurve.append(
    #             Gaussian(
    #                 self.frequencies[ar[0] : ar[1]],
    #                 self.popt[0],
    #                 self.popt[1],
    #                 self.popt[2],
    #                 self.popt[6] / 2,
    #             )
    #         )
    #         self.fitcurve.append(
    #             Gaussian(
    #                 self.frequencies[ar[0] : ar[1]],
    #                 self.popt[3],
    #                 self.popt[4],
    #                 self.popt[5],
    #                 self.popt[6] / 2,
    #             )
    #         )
    #         self.fitcurve.append(
    #             dualGaussian(
    #                 self.frequencies[ar[0] : ar[1]],
    #                 self.popt[0],
    #                 self.popt[1],
    #                 self.popt[2],
    #                 self.popt[3],
    #                 self.popt[4],
    #                 self.popt[5],
    #                 self.popt[6],
    #             )
    #         )

    #     # Compute residual and residual value.
    #     if getresidual:
    #         self.residual = self.avgPSD[ar[0] : ar[1]] - self.fitcurve[-1]
    #         self.residualval = np.sum(abs(self.residual))

    #     # Compute chi square
    #     # Not sure if this is the correct way to do so, but self.chisq is certainly helpful.
    #     if getchisq:
    #         self.chisq = np.sum(
    #             (self.avgPSD[ar[0] : ar[1]] - self.fitcurve[-1]) ** 2
    #             / abs(self.fitcurve[-1])
    #         )

    #     ############################################################################################
    #     # Start of generating fitreport
    #     self.fitreport = (
    #         Function_dict[fitfunction][0]
    #         + f" Fit ({(100 - 100 * alpha):.0f} % confidence level)\n"
    #     )
    #     fitresultlist = []
    #     for i in range(len(self.popt)):
    #         fitresultlist.append(ufloat(self.popt[i], tval * self.perr[i]))

    #     # some example results:

    #     # G:/SQUID NMR/Shim Run/BayesAutoShim_20221123_111042/runnumber_0.h5
    #     # "Center frequency = 1.347654(4) MHz\n"
    #     # "Linewidth = 212(11) Hz\n"
    #     # "Area = 1.25(5)e-8 $\\phi_0^2$\n"
    #     # "Offset = 2.1(1)e-12 $\\phi_0^2$/Hz\n"

    #     # G:/SQUID NMR/Shim Run/BayesAutoShim_20221123_111042/runnumber_14.h5
    #     # "Center frequency = 1.3489973(3) MHz\n"
    #     # "Linewidth = 29.7(7) Hz\n"
    #     # "Area = 2.04(3)e-8 $\\phi_0^2$\n"
    #     # "Offset = 2.7(2)e-12 $\\phi_0^2$/Hz\n"

    #     # G:/SQUID NMR/Shim Run/BayesAutoShim_20221221_162415/runnumber_42.h5
    #     # "Center frequency = 1.34838796(7) MHz\n"
    #     # "Linewidth = 7.8(2) Hz\n"
    #     # "Area = 4.80(8)e-9 $\\phi_0^2$\n"
    #     # "Offset = 4.6(1)e-12 $\\phi_0^2$/Hz\n"

    #     if Function_dict[fitfunction][0] == "Lorentzian":
    #         # self.fitreport += "Center frequency = 1.34838796(7) MHz\n"#f'center = {fitresultlist[0]:.6ue} Hz\n' #:.2f #:.6ue
    #         # self.fitreport += "Linewidth = 7.8(2) Hz\n"#f'linewidth = {abs(fitresultlist[1]):.2f} Hz\n' # linewidth
    #         # self.fitreport += "Area = 4.80(8)e-9 $\\phi_0^2$\n"#farea = {fitresultlist[2]:.2e}\n'
    #         # self.fitreport += "Offset = 4.6(1)e-12 $\\phi_0^2$/Hz\n"#f'offset = {fitresultlist[3]:.2e}\n'
    #         self.fitreport += "center = {:.3f} Hz\n".format(fitresultlist[0])
    #         self.fitreport += "linewidth = {:.3f} Hz\n".format(fitresultlist[1])
    #         self.fitreport += "area = {:.3e}\n".format(fitresultlist[2])
    #         self.fitreport += "offset = {:.2e}\n".format(fitresultlist[3])
    #     elif Function_dict[fitfunction][0] == "dualLorentzian":
    #         self.fitreport += "center0 = {:.2f} Hz\n".format(fitresultlist[0])
    #         self.fitreport += "linewidth0 = {:.2f} Hz\n".format(fitresultlist[1])
    #         self.fitreport += "area0 = {:.2e}\n".format(fitresultlist[2])
    #         self.fitreport += "center1 = {:.2f} Hz\n".format(fitresultlist[3])
    #         self.fitreport += "linewidth1 = {:.2f} Hz\n".format(fitresultlist[4])
    #         self.fitreport += "area1 = {:.2e}\n".format(fitresultlist[5])
    #         self.fitreport += "offset = {:.2e}\n".format(fitresultlist[6])
    #     elif Function_dict[fitfunction][0] == "tribLorentzian":
    #         self.fitreport += "center0 = {:.2f} Hz\n".format(fitresultlist[0])
    #         self.fitreport += "linewidth0 = {:.2f} Hz\n".format(fitresultlist[1])
    #         self.fitreport += "area0 = {:.2e}\n".format(fitresultlist[2])
    #         self.fitreport += "center1 = {:.2f} Hz\n".format(fitresultlist[3])
    #         self.fitreport += "linewidth1 = {:.2f} Hz\n".format(fitresultlist[4])
    #         self.fitreport += "area1 = {:.2e}\n".format(fitresultlist[5])
    #         self.fitreport += "center2 = {:.2f} Hz\n".format(fitresultlist[6])
    #         self.fitreport += "linewidth2 = {:.2f} Hz\n".format(fitresultlist[7])
    #         self.fitreport += "area2 = {:.2e}\n".format(fitresultlist[8])
    #         self.fitreport += "offset = {:.2e}\n".format(fitresultlist[9])
    #     elif Function_dict[fitfunction][0] == "Gaussian":
    #         self.fitreport += "center = {:.2f} Hz\n".format(fitresultlist[0])
    #         # self.fitreport += 'sigma = {:.2f} Hz\n'.format(fitresultlist[1])
    #         self.fitreport += "FWHM = {:.2f} Hz\n".format(2.35482 * fitresultlist[1])
    #         self.fitreport += "area = {:.2e}\n".format(fitresultlist[2])
    #         self.fitreport += "offset = {:.2e}\n".format(fitresultlist[3])
    #     elif Function_dict[fitfunction][0] == "dualGaussian":
    #         self.fitreport += "center0 = {:.2f} Hz\n".format(fitresultlist[0])
    #         self.fitreport += "FWHM0 = {:.2f} Hz\n".format(2.35482 * fitresultlist[1])
    #         self.fitreport += "area0 = {:.2e}\n".format(fitresultlist[2])
    #         self.fitreport += "center1 = {:.2f} Hz\n".format(fitresultlist[3])
    #         self.fitreport += "FWHM1 = {:.2f} Hz\n".format(2.35482 * fitresultlist[4])
    #         self.fitreport += "area1 = {:.2e}\n".format(fitresultlist[5])
    #         self.fitreport += "offset = {:.2e}\n".format(fitresultlist[6])

    #     if getresidual:
    #         self.fitreport += "sum of absolute residual = {:.2e}\n".format(
    #             self.residualval
    #         )
    #     if getchisq:
    #         self.fitreport += "$\chi^2$ = {:.2e}\n".format(self.chisq)

    #     # self.fitreport = self.fitreport[:-1]
    #     self.fitrange = [ar[0], ar[1]]
    #     del ar, fitresultlist
    #     # End of generating fitreport
    #     ########################################################################

    #     if verbose:
    #         print(self.fitreport)
    #     return self.popt, tval * self.perr

    # def FitPSDnoDict(
    #     self,
    #     fitfunction=None,  # 'Lorentzian' 'dualLorentzian' 'tribLorentzian' 'Gaussian 'dualGaussian' 'auto' 'Polyeven'
    #     inputfitparas=[],
    #     smooth=False,
    #     smoothlevel=1,
    #     fitrange=["auto", "auto"],
    #     alpha=0.05,
    #     getresidual=False,
    #     getchisq=False,
    #     verbose=False,
    # ):

    #     # fitfunction = fitfunction.upper()

    #     self.fitflag = False
    #     self.popt = None
    #     self.pcov = None
    #     self.residual = None
    #     self.residualval = None
    #     self.chisq = None
    #     self.fitcurve = []
    #     ar = [0, len(self.avgPSD)]
    #     for i in range(len(fitrange)):
    #         if type(fitrange[i]) == int or type(fitrange[i]) == float:
    #             ar[i] = fitrange[i]

    #     fitparas = inputfitparas
    #     # numofparas = len(fitparas)
    #     # for i in range(len(inputfitparas)):
    #     #    if type(inputfitparas[i]) == int or type(inputfitparas[i]) == float:
    #     #        fitparas[i] = inputfitparas[i]

    #     try:
    #         if verbose:
    #             print("start fit")
    #         self.popt, self.pcov = scipy.optimize.curve_fit(
    #             f=fitfunction,
    #             xdata=self.frequencies[ar[0] : ar[1]],
    #             ydata=self.avgPSD[ar[0] : ar[1]],
    #             p0=fitparas,
    #             absolute_sigma=False,
    #         )
    #         self.fitflag = True
    #         if verbose:
    #             print("Obtained fit parameters ", [x for x in self.popt])
    #     except RuntimeError:
    #         print("fit RuntimeError")
    #     else:
    #         pass

    #     if self.fitflag:
    #         self.perr = np.sqrt(np.diag(self.pcov))
    #         dof = max(
    #             0, len(self.frequencies[ar[0] : ar[1]]) - len(self.popt)
    #         )  # degree of freedom
    #         # student-t value for the dof and confidence level
    #         tval = scipy.stats.distributions.t.ppf(1.0 - alpha / 2.0, dof)

    #         self.fitcurve.append(
    #             fitfunction(self.frequencies[ar[0] : ar[1]], self.popt[0])
    #         )  # self.popt[1]

    #         if getresidual:
    #             self.residual = self.avgPSD[ar[0] : ar[1]] - self.fitcurve[-1]
    #             self.residualval = np.sum(abs(self.residual))
    #         if getchisq:
    #             self.chisq = np.sum(
    #                 (self.avgPSD[ar[0] : ar[1]] - self.fitcurve[-1]) ** 2
    #                 / abs(self.fitcurve[-1])
    #             )
    #         self.fitreport = f" Fit ({(100 - 100 * alpha):.0f} % confidence level)\n"
    #         fitresultlist = []
    #         for i in range(len(self.popt)):
    #             fitresultlist.append(ufloat(self.popt[i], tval * self.perr[i]))
    #             self.fitreport += f"param{i} = {abs(fitresultlist[i]):.2f}"

    #         if getresidual:
    #             self.fitreport += "sum of absolute residual = {:.2e}\n".format(
    #                 self.residualval
    #             )
    #         if getchisq:
    #             self.fitreport += "chi-sq = {:.2e}\n".format(self.chisq)

    #         self.fitreport = self.fitreport[:-1]
    #         self.fitrange = [ar[0], ar[1]]
    #         del ar, fitresultlist
    #         if verbose:
    #             print(self.fitreport)
    #         return self.popt, tval * self.perr

    # def FitFID(
    #     self,
    #     functionname="ExpCos",  # 'ExpCos','dualExpCos','tribExpCos'. 'auto' is not available yet
    #     inputfitparas=["auto", "auto", "auto", "auto", "auto"],
    #     smooth=False,
    #     smoothlevel=1,
    #     fitlen="auto",
    #     alpha=0.05,
    #     selectshot=[0],
    #     getresidual=False,
    #     getchisq=False,
    #     showplt=False,
    #     verbose=False,
    # ):
    #     functionname = functionname.upper()

    #     self.FIDfitflag = False
    #     self.FIDpopt = None
    #     self.FIDpcov = None
    #     self.FIDresidual = None
    #     self.FIDresidualval = None
    #     self.FIDchisq = None
    #     self.FIDfitcurve = []

    #     self.timeStamp = np.linspace(
    #         start=0,
    #         stop=len(self.dataX) / self.sampRate,
    #         num=len(self.dataX),
    #         endpoint=False,
    #         dtype=float,
    #     )

    #     ar = self.acq_arr[selectshot[0]].copy()
    #     ar[1] = ar[0] + int(self.sampRate * 50e-3)
    #     # for i in range(len(fitrange)):
    #     if type(fitlen) == int:
    #         ar[1] = ar[0] + fitlen
    #     # if verbose:
    #     #     check(ar)
    #     # check(ar)
    #     # check(Functionname_dict[functionname][1])
    #     curvefunction = Function_dict[functionname][1]
    #     fitparas = Function_dict[functionname][2](
    #         s=self.dataX[ar[0] : ar[1]] + 1j * self.dataY[ar[0] : ar[1]],
    #         Lorpopt=self.popt,
    #         demodFreq=self.demodFreq,
    #     )
    #     for i, inputfitpara in enumerate(inputfitparas):
    #         if type(inputfitpara) == int or type(inputfitpara) == float:
    #             fitparas[i] = inputfitparas[i]

    #     # check(fitparas)
    #     try:
    #         if verbose:
    #             print("start " + Function_dict[functionname][0] + " fit")
    #         if Function_dict[functionname][0] in ["ExpCos", "dualExpCos", "tribExpCos"]:
    #             self.FIDpopt, self.FIDpcov = scipy.optimize.curve_fit(
    #                 curvefunction,
    #                 self.timeStamp[ar[0] : ar[1] : 20],
    #                 self.dataX[ar[0] : ar[1] : 20],
    #                 p0=fitparas,
    #                 absolute_sigma=False,
    #             )

    #         elif Function_dict[functionname][0] in [
    #             "ExpCosiSin",
    #             "dualExpCosiSin",
    #             "tribExpCosiSin",
    #         ]:
    #             # func = partial(curvefunction, t=self.timestamp[ar[0]:ar[1]], s=self.dataX[ar[0]:ar[1]]+1j*self.dataY[ar[0]:ar[1]])
    #             # check(list(fitparas))
    #             # self.FIDpopt, self.FIDpcov = scipy.optimize.curve_fit(
    #             #     curvefunction, self.timestamp[ar[0]:ar[1]], self.dataX[ar[0]:ar[1]]+1j*self.dataY[ar[0]:ar[1]] , p0=fitparas,
    #             #     absolute_sigma = False, method='lm')
    #             step = 4
    #             result = scipy.optimize.leastsq(
    #                 curvefunction,
    #                 x0=fitparas,
    #                 args=(
    #                     self.timeStamp[ar[0] : ar[1] : step],
    #                     np.array(
    #                         self.dataX[ar[0] : ar[1] : step]
    #                         + 1j * self.dataY[ar[0] : ar[1] : step]
    #                     ),
    #                 ),
    #                 maxfev=10000,
    #                 full_output=True,
    #             )
    #             self.FIDpopt = result[0]
    #             # check(self.timestamp[ar[0]:ar[1]:step].shape)
    #             check(self.FIDpopt)

    #         self.FIDfitflag = True
    #         if verbose:
    #             print(
    #                 "Obtained " + Function_dict[functionname][0] + " fit parameters ",
    #                 [x for x in self.popt],
    #             )
    #     except RuntimeError:
    #         print("fit RuntimeError")
    #     else:
    #         pass
    #     # del fitparas

    #     if self.FIDfitflag:

    #         if Function_dict[functionname][0] == "ExpCos":
    #             self.FIDfitcurve.append(
    #                 curvefunction(
    #                     self.timeStamp[ar[0] : ar[1]],
    #                     self.FIDpopt[0],
    #                     self.FIDpopt[1],
    #                     self.FIDpopt[2],
    #                     self.FIDpopt[3],
    #                     self.FIDpopt[-1],
    #                 )
    #             )
    #         elif Function_dict[functionname][0] == "ExpCosiSin":
    #             self.FIDfitcurve.append(
    #                 ExpCosiSin(
    #                     self.timeStamp[ar[0] : ar[1]],
    #                     self.FIDpopt[0],
    #                     self.FIDpopt[1],
    #                     self.FIDpopt[2],
    #                     self.FIDpopt[3],
    #                     self.FIDpopt[4],
    #                     self.FIDpopt[-1],
    #                 )
    #             )
    #         elif Function_dict[functionname][0] == "dualExpCos":
    #             self.FIDfitcurve.append(
    #                 ExpCos(
    #                     self.timeStamp[ar[0] : ar[1]],
    #                     self.FIDpopt[0],
    #                     self.FIDpopt[1],
    #                     self.FIDpopt[2],
    #                     self.FIDpopt[3],
    #                     self.FIDpopt[-1] / 2,
    #                 )
    #             )
    #             self.FIDfitcurve.append(
    #                 ExpCos(
    #                     self.timeStamp[ar[0] : ar[1]],
    #                     self.FIDpopt[4],
    #                     self.FIDpopt[5],
    #                     self.FIDpopt[6],
    #                     self.FIDpopt[7],
    #                     self.FIDpopt[-1] / 2,
    #                 )
    #             )
    #             self.FIDfitcurve.append(
    #                 dualExpCos(
    #                     self.timeStamp[ar[0] : ar[1]],
    #                     self.FIDpopt[0],
    #                     self.FIDpopt[1],
    #                     self.FIDpopt[2],
    #                     self.FIDpopt[3],
    #                     self.FIDpopt[4],
    #                     self.FIDpopt[5],
    #                     self.FIDpopt[6],
    #                     self.FIDpopt[7],
    #                     self.FIDpopt[8],
    #                 )
    #             )

    #         if getresidual or showplt:
    #             if "i" in Function_dict[functionname][0]:
    #                 self.FIDresidual = (
    #                     self.dataX[ar[0] : ar[1]]
    #                     + 1j * self.dataY[ar[0] : ar[1]]
    #                     - self.FIDfitcurve[-1]
    #                 )
    #                 self.FIDresidualval = np.sum(abs(self.FIDresidual))
    #                 self.FIDpcov = result[1] * np.var(
    #                     ExpCosiSinResidual(
    #                         params=self.FIDpopt,
    #                         t=self.timeStamp[ar[0] : ar[1]],
    #                         s=self.dataX[ar[0] : ar[1]]
    #                         + 1j * self.dataY[ar[0] : ar[1]],
    #                     )
    #                 )
    #             else:
    #                 self.FIDresidual = self.dataX[ar[0] : ar[1]] - self.FIDfitcurve[-1]
    #                 self.FIDresidualval = np.sum(abs(self.FIDresidual))

    #         if getchisq:
    #             self.FIDchisq = np.sum(
    #                 abs(self.dataX[ar[0] : ar[1]] - self.FIDfitcurve[-1]) ** 2
    #                 / abs(self.FIDfitcurve[-1])
    #             )

    #         self.FIDperr = np.sqrt(np.diag(self.FIDpcov))
    #         dof = max(0, abs(ar[1] - ar[0]) - len(self.FIDpopt))  # degree of freedom
    #         # student-t value for the dof and confidence level
    #         tval = scipy.stats.distributions.t.ppf(1.0 - alpha / 2.0, dof)

    #         # write fit report
    #         self.FIDfitreport = (
    #             Function_dict[functionname][0]
    #             + f" Fit ({100 - 100 * alpha:.0f} % confidence level)\n"
    #         )
    #         FIDfitresultlist = []
    #         for i, FIDpopti in enumerate(self.FIDpopt):
    #             FIDfitresultlist.append(ufloat(self.FIDpopt[i], tval * self.FIDperr[i]))

    #         FIDfitresultlist.append(
    #             ufloat(
    #                 1 / (np.pi * self.FIDpopt[1]),
    #                 abs(
    #                     1 / (np.pi * (self.FIDpopt[1] - tval * self.FIDperr[1]))
    #                     - 1 / (np.pi * self.FIDpopt[1])
    #                 ),
    #             )
    #         )
    #         if Function_dict[functionname][0] in ["ExpCos", "ExpCosiSin"]:
    #             self.FIDfitreport += f"Amp = {FIDfitresultlist[0]:.2e} V\n"
    #             self.FIDfitreport += f"T2* = {FIDfitresultlist[1]:.6f} s\n"
    #             self.FIDfitreport += f"linewidth = {FIDfitresultlist[-1]:.2f} Hz\n"
    #             self.FIDfitreport += f"nu = {FIDfitresultlist[2]:.3f} Hz\n"
    #             self.FIDfitreport += f"phi0 = {FIDfitresultlist[3]:.2f} \n"
    #             self.FIDfitreport += f"offset = {FIDfitresultlist[-1]:.2e} V\n"
    #         elif Function_dict[functionname][0] == "dualExpCos":
    #             self.FIDfitreport += f"Amp_0 = {FIDfitresultlist[0]:.2e} V\n"
    #             self.FIDfitreport += f"T2*_0 = {FIDfitresultlist[1]:.6f} s\n"
    #             self.FIDfitreport += (
    #                 f"linewidth_0 = {1/(np.pi*self.FIDpopt[1]):.2f} Hz\n"
    #             )
    #             self.FIDfitreport += f"nu_0 = {FIDfitresultlist[2]:.3f} Hz\n"
    #             self.FIDfitreport += f"phi0_0 = {FIDfitresultlist[3]:.2f} \n\n"
    #             self.FIDfitreport += f"Amp_1 = {FIDfitresultlist[4]:.2e} V\n"
    #             self.FIDfitreport += f"T2*_1 = {FIDfitresultlist[5]:.6f} s\n"
    #             self.FIDfitreport += (
    #                 f"linewidth_1 = {1/(np.pi*self.FIDpopt[5]):.2f} Hz\n"
    #             )
    #             self.FIDfitreport += f"nu_1 = {FIDfitresultlist[6]:.3f} Hz\n"
    #             self.FIDfitreport += f"phi0_1 = {FIDfitresultlist[7]:.2f} \n"
    #             self.FIDfitreport += f"offset = {FIDfitresultlist[-1]:.2e} V\n"

    #         if getresidual:
    #             self.FIDfitreport += (
    #                 f"sum of absolute residual = {self.FIDresidualval:.2e}\n"
    #             )
    #         if getchisq:
    #             self.FIDfitreport += f"chi-sq = {self.FIDchisq:.2e}\n"

    #         self.FIDfitreport = self.FIDfitreport[:-1]  # get rid of the last \n
    #         self.FIDfitrange = ar
    #         del ar, FIDfitresultlist
    #         if verbose:
    #             print(self.FIDfitreport)
    #         if showplt:
    #             fig = plt.figure(figsize=(16 * 0.8, 8 * 0.8))  #
    #             gs = gridspec.GridSpec(nrows=4, ncols=1)  #
    #             # fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
    #             #                     bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
    #             # make plot for time domain signal
    #             timedm_ax = fig.add_subplot(gs[0:3, 0])

    #             if "i" in Function_dict[functionname][0]:
    #                 timedm_ax.plot(
    #                     self.timeStamp[self.FIDfitrange[0] : self.FIDfitrange[1]],
    #                     np.real(self.FIDfitcurve[-1]),
    #                     c="tab:red",
    #                     label="fitcurve.real",
    #                 )
    #                 timedm_ax.plot(
    #                     self.timeStamp[self.FIDfitrange[0] : self.FIDfitrange[1]],
    #                     np.imag(self.FIDfitcurve[-1]),
    #                     c="tab:green",
    #                     label="fitcurve.imag\n" + self.FIDfitreport,
    #                 )
    #             else:
    #                 timedm_ax.plot(
    #                     self.timeStamp[self.FIDfitrange[0] : self.FIDfitrange[1]],
    #                     self.FIDfitcurve[-1],
    #                     c="tab:red",
    #                     label=self.FIDfitreport,
    #                 )
    #                 if Function_dict[functionname][0] == "dualExpCos":
    #                     timedm_ax.plot(
    #                         self.timeStamp[self.FIDfitrange[0] : self.FIDfitrange[1]],
    #                         self.FIDfitcurve[0],
    #                         "--",
    #                         c="tab:orange",
    #                         alpha=0.9,
    #                         label="ExpCos0",
    #                     )
    #                     timedm_ax.plot(
    #                         self.timeStamp[self.FIDfitrange[0] : self.FIDfitrange[1]],
    #                         self.FIDfitcurve[1],
    #                         "--",
    #                         c="tab:green",
    #                         alpha=0.9,
    #                         label="ExpCos1",
    #                     )

    #             # timedm_ax.plot(self.timestamp[self.FIDfitrange[0]:self.FIDfitrange[1]],
    #             #                curvefunction(
    #             #                    t=self.timestamp[self.FIDfitrange[0]:self.FIDfitrange[1]],
    #             #                    Amp=fitparas[0],
    #             #                    T2=fitparas[1],
    #             #                    nu=fitparas[2],
    #             #                    phi0=fitparas[3],
    #             #                     offset=fitparas[4],
    #             #                    ),
    #             #                c='tab:orange', label='estimated curve')
    #             timedm_ax.scatter(
    #                 self.timeStamp[self.FIDfitrange[0] : self.FIDfitrange[1]],
    #                 self.dataX[self.FIDfitrange[0] : self.FIDfitrange[1]],
    #                 c="tab:blue",
    #                 s=3,
    #                 label="dataX",
    #             )
    #             if "i" in Function_dict[functionname][0]:
    #                 timedm_ax.scatter(
    #                     self.timeStamp[self.FIDfitrange[0] : self.FIDfitrange[1]],
    #                     self.dataY[self.FIDfitrange[0] : self.FIDfitrange[1]],
    #                     c="tab:pink",
    #                     s=3,
    #                     label="dataY",
    #                 )
    #             timedm_ax.set_xlim(right=self.timeStamp[self.FIDfitrange[1]] + 0.03)
    #             timedm_ax.set_xlabel("time / s")
    #             timedm_ax.set_ylabel("Signal / V")
    #             timedm_ax.legend(loc="best")
    #             timedm_ax.grid()

    #             resi_ax = fig.add_subplot(gs[-1, 0])
    #             if "i" in Function_dict[functionname][0]:
    #                 resi_ax.plot(
    #                     self.timeStamp[self.FIDfitrange[0] : self.FIDfitrange[1]],
    #                     np.real(self.FIDresidual),
    #                     c="tab:blue",
    #                     label="residual.real",
    #                 )
    #                 resi_ax.plot(
    #                     self.timeStamp[self.FIDfitrange[0] : self.FIDfitrange[1]],
    #                     np.imag(self.FIDresidual),
    #                     c="tab:pink",
    #                     label="residual.imag",
    #                 )
    #             else:
    #                 resi_ax.plot(
    #                     self.timeStamp[self.FIDfitrange[0] : self.FIDfitrange[1]],
    #                     self.FIDresidual,
    #                     c="tab:blue",
    #                     label="residual",
    #                 )
    #             resi_ax.set_xlim(right=self.timeStamp[self.FIDfitrange[1]] + 0.05)
    #             resi_ax.set_xlabel("time / s")
    #             resi_ax.set_ylabel("Residual / V")
    #             resi_ax.legend(loc="best")
    #             resi_ax.set_xlim(right=self.timeStamp[self.FIDfitrange[1]] + 0.03)
    #             plt.show()

    #         return self.FIDpopt, tval * self.FIDperr

    # def Hz2Index(self, freq_Hz):
    #     """
    #     Parameters
    #     ----------
    #     freq_Hz: float
    #         frequency in [Hz]

    #     Returns
    #     -------
    #     freq_index : int
    #         the index of the input frequency [Hz]

    #     raises ValueError if the input frequency value or the output index
    #     is out of range

    #     Examples
    #     -------
    #     Example of 7 elements
    #         index            0       1       2       3       4       5        6
    #         frequency [Hz]  0.1     0.2     0.3     0.4     0.5     0.6      0.7

    #         lenofdata = 7
    #         center_freq = self.frequencies[lenofdata//2] = 0.4 [Hz]
    #         self.freq_resol = 0.1 [Hz]

    #         input frequency: 0.6
    #         output index = lenofdata // 2 + int(np.round((freq_Hz - center_freq) / self.freq_resol))
    #                      = 7 // 2 + int( np.round( (0.6 - 0.4)/0.1 ) )
    #                      = 3 + 2
    #                      = 5

    #         input frequency: 0.66
    #         output index = lenofdata // 2 + int(np.round((freq_Hz - center_freq) / self.freq_resol))
    #                      = 7 // 2 + int( np.round( (0.66 - 0.4)/0.1 ) )
    #                      = 3 + 3
    #                      = 6

    #     Example of 8 elements
    #         index           0   1       2       3       4       5        6      7
    #         frequency[Hz]  0.1  0.2     0.3     0.4     0.5     0.6     0.7    0.8

    #         lenofdata = 8
    #         center_freq = self.frequencies[lenofdata//2] = 0.5 [Hz]
    #         self.freq_resol = 0.1 [Hz]

    #         input frequency: 0.6
    #         output index = lenofdata // 2 + int(np.round((freq_Hz - center_freq) / self.freq_resol))
    #                      = 8 // 2 + int( np.round( (0.6 - 0.5)/0.1 ) )
    #                      = 4 + 1
    #                      = 5

    #         input frequency: 0.66
    #         output index = lenofdata // 2 + int(np.round((freq_Hz - center_freq) / self.freq_resol))
    #                      = 8 // 2 + int( np.round( (0.66 - 0.5)/0.1 ) )
    #                      = 4 + 2
    #                      = 6

    #     """
    #     if freq_Hz < np.amin(self.frequencies) or freq_Hz > np.amax(self.frequencies):
    #         raise ValueError(
    #             f"[{self.Hz2Index.__name__}] frequency {freq_Hz:g} "
    #             + "out of range "
    #             + f"[{np.amin(self.frequencies):g}, {np.amax(self.frequencies):g}]. "
    #         )
    #     lenofdata = len(self.frequencies)
    #     center_freq = self.frequencies[lenofdata // 2]
    #     self.freq_resol = abs(
    #         self.frequencies[0] - self.frequencies[1]
    #     )  # frequency resolution
    #     freq_index = lenofdata // 2 + int(
    #         np.round((freq_Hz - center_freq) / self.freq_resol)
    #     )
    #     if freq_index < 0 or freq_index > (lenofdata - 1):
    #         raise ValueError(
    #             f"[{self.Hz2Index.__name__}] index out of range "
    #             + f"[0, {len(self.frequencies)-1:d}].  "
    #         )
    #     return freq_index

    # def Index2Hz(self, freq_index):
    #     """
    #     Parameters
    #     ----------
    #     freq_index : int
    #         the index of the input frequency [Hz]

    #     Returns
    #     -------
    #     freq_index: float
    #         frequency in [Hz]

    #     raises ValueError if the inoput index or the output frequency value
    #     is out of range

    #     Examples
    #     -------
    #     Example of 7 elements
    #         index            0       1       2       3       4       5        6
    #         frequency [Hz]  0.1     0.2     0.3     0.4     0.5     0.6      0.7

    #         lenofdata = 7
    #         center_freq = self.frequencies[lenofdata//2] = 0.4 [Hz]
    #         self.freq_resol = 0.1 [Hz]freq_resol = 0.1 [Hz]

    #         input index: 3
    #         output frequency = center_freq + self.freq_resol * (freq_index - lenofdata // 2)
    #                      = 0.4 + 0.1 * (3 - 3)
    #                      = 0.4 + 0.0
    #                      = 0.4

    #     Example of 8 elements
    #         index           0   1       2       3       4       5        6      7
    #         frequency[Hz]  0.1  0.2     0.3     0.4     0.5     0.6     0.7    0.8

    #         lenofdata = 8
    #         center_freq = self.frequencies[lenofdata//2] = 0.5 [Hz]
    #         self.freq_resol = 0.1 [Hz]freq_resol = 0.1 [Hz]

    #         input index: 3
    #         output frequency = center_freq + self.freq_resol * (freq_index - lenofdata // 2)
    #                      = 0.5 + 0.1 * (3 - 4)
    #                      = 0.5 - 1.0
    #                      = 0.4

    #     """
    #     if freq_index < 0 or freq_index > (len(self.frequencies) - 1):
    #         raise ValueError(
    #             f"[{self.Index2Hz.__name__}] frequency index {freq_index:d} "
    #             + "out of range "
    #             + f"[0, {len(self.frequencies)-1:d}]. "
    #         )
    #     lenofdata = len(self.frequencies)
    #     center_freq = self.frequencies[lenofdata // 2]
    #     self.freq_resol = abs(
    #         self.frequencies[0] - self.frequencies[1]
    #     )  # frequency resolution
    #     freq_Hz = center_freq + self.freq_resol * (freq_index - lenofdata // 2)
    #     if freq_Hz < np.amin(self.frequencies) or freq_Hz > np.amax(self.frequencies):
    #         raise ValueError(
    #             f"[{self.Hz2Index.__name__}] frequency {freq_Hz:g} "
    #             + "out of range "
    #             + f"[{np.amin(self.frequencies):g}, {np.amax(self.frequencies):g}]. "
    #         )
    #     return freq_Hz

    # def sgFilterPSD(
    #     self,
    #     window_length: int = 100,
    #     polyorder: int = 2,
    #     makeplot: bool = False,
    #     verbose: bool = False,
    # ):
    #     assert hasattr(self, "avgPSD")
    #     sg = savgol_filter(
    #         self.avgPSD, window_length=window_length, polyorder=polyorder
    #     )
    #     if makeplot:
    #         fig = plt.figure(figsize=(10.0, 8.0), dpi=150)  # initialize a figure
    #         width_ratios = [1, 1]
    #         height_ratios = [1, 1, 2]
    #         gs = gridspec.GridSpec(
    #             nrows=3, ncols=2, width_ratios=width_ratios, height_ratios=height_ratios
    #         )  #
    #         # gs = gridspec.GridSpec(nrows=3, ncols=2)  #
    #         PSD_ax = fig.add_subplot(gs[0, :])
    #         # SG_ax = fig.add_subplot(gs[1,0], sharex=PSD_ax, sharey=PSD_ax)
    #         PSDsubSG_ax = fig.add_subplot(gs[1, :], sharex=PSD_ax, sharey=PSD_ax)
    #         PSD_hist_ax = fig.add_subplot(gs[2, 0])
    #         PSDsubSG_hist_ax = fig.add_subplot(gs[2, 1])

    #         PSD_ax.plot(self.frequencies, self.avgPSD, label="raw PSD")
    #         PSD_ax.plot(self.frequencies, sg, label="SG output")
    #         # SG_ax.plot(self.frequencies, sg, label='SG output')
    #         PSDsubSG_ax.plot(
    #             self.frequencies, self.avgPSD - sg, c="g", label="raw PSD - SG output"
    #         )

    #         for ax in [PSD_ax, PSDsubSG_ax]:  # SG_ax,
    #             ax.legend()
    #             ax.set_ylabel("PSD [$\mathrm{V}^2/\mathrm{Hz}$]")

    #         allmin = np.amin(
    #             [np.amin(self.avgPSD), np.amin(sg), np.amin(self.avgPSD - sg)]
    #         )
    #         allmax = np.amax(
    #             [np.amax(self.avgPSD), np.amax(sg), np.amax(self.avgPSD - sg)]
    #         )

    #         PSD_ax.set_ylim(min(-allmax * 0.1, allmin * 1.5), max(0, allmax * 1.5))
    #         PSDsubSG_ax.set_xlabel("Frequency [Hz]")

    #         def ChisqHistPlot(ax, data, title):
    #             """ """
    #             mean = np.mean(data)
    #             std = np.std(data)
    #             # check(np.amax(data))
    #             check(std)
    #             hist, bin_edges = np.histogram(data / std, bins="auto", density=False)
    #             binwidth = abs(bin_edges[1] - bin_edges[0])
    #             sumofcounts = len(data)
    #             hist_info = f"sum of counts={sumofcounts:d}\nmean={mean:.1e} [V^2/Hz] std={std:.1e} [V^2/Hz]"
    #             # if normalizebysigma:
    #             # bin_edges /= std
    #             ax.set_xlabel("PSD / $\\sigma_{\mathrm{PSD}}$")
    #             # else:
    #             #     ax.set_xlabel('signal [V]')
    #             hist_x = []
    #             hist_y = []
    #             for i, count in enumerate(hist):
    #                 if count > 0:
    #                     hist_x.append((bin_edges[i] + bin_edges[i + 1]) / 2.0)
    #                     hist_y.append(count)
    #             ax.scatter(
    #                 hist_x,
    #                 hist_y,
    #                 color="tab:red",
    #                 edgecolors="k",
    #                 linewidths=1,
    #                 marker="o",
    #                 s=6,
    #                 zorder=6,
    #                 label=f"histogram",
    #             )

    #             # plot pdf of gaussian distribution (mean = 0, std = 1)
    #             xstamp = np.linspace(
    #                 start=0,
    #                 stop=np.amax(bin_edges),
    #                 num=max(100, len(bin_edges)),
    #                 endpoint=True,
    #             )
    #             # ax.plot(xstamp, norm.pdf(xstamp, mean, 1), label='Gaussian pdf', linestyle='--')
    #             ax.plot(
    #                 xstamp,
    #                 2 * sumofcounts * binwidth * chi2.pdf(2 * xstamp, df=2),
    #                 label="$\\chi^2$ Chi-sq pdf\ndof=2",
    #                 linestyle="--",
    #             )
    #             # sumofcounts * binwidth *
    #             # ax.plot(xstamp, sumofcounts * binwidth * chi2.pdf(xstamp, df=1), label='Chi-sq pdf dof=1', linestyle='--')
    #             ax.set_ylabel("count")
    #             ax.set_title(title + "\n" + hist_info)
    #             ax.set_xscale("linear")
    #             ax.set_yscale("log")
    #             ax.set_ylim(bottom=0.1)
    #             ax.legend(loc="best")
    #             ax.grid(True)
    #             del (
    #                 mean,
    #                 std,
    #                 hist,
    #                 bin_edges,
    #                 sumofcounts,
    #                 hist_info,
    #                 hist_x,
    #                 hist_y,
    #                 xstamp,
    #             )

    #         ChisqHistPlot(PSD_hist_ax, data=self.avgPSD, title="raw PSD histogram")
    #         ChisqHistPlot(
    #             PSDsubSG_hist_ax,
    #             data=self.avgPSD - sg,
    #             title="(raw PSD - SG) histogram",
    #         )

    #         fig.tight_layout()
    #         plt.show()

    #     return sg

    # def psdMovAvgByStep(self, weights=None, step_len: int = 1, verbose: bool = False):
    #     self.frequencies, self.avgPSD = MovAvgByStep(
    #         xstamp=self.frequencies,
    #         rawsignal=self.avgPSD,
    #         weights=weights,
    #         step_len=step_len,
    #         verbose=verbose,
    #     )
    #     # self.avgPSD /= np.vdot(weights, weights)
    #     self.freq_resol = step_len * self.freq_resol

    # def GetSpinNoisePSDsub(
    #     self,
    #     windowfunction="rectangle",  # Hanning, Hamming, Blackman
    #     chunksize=33e-3,  # in second
    #     analysisrange=[0, -1],
    #     showstd=True,
    #     stddev_range=[35e3, 37e3],
    #     ploycorrparas=[],
    #     interestingfreq_list=[],
    #     selectshot=[],
    #     verbose=False,
    # ):
    #     print("This function is out of date. Please use GetNoPulsePSD instead. ")
    #     self.chunksize = chunksize
    #     chunklen = int(chunksize * self.sampRate)
    #     data_start = analysisrange[0]
    #     data_end = len(self.dataX) + analysisrange[1]
    #     numofchunk = (data_end - data_start) // chunklen

    #     acq_arr0 = data_start + chunklen * np.linspace(
    #         start=0, stop=numofchunk, num=numofchunk, endpoint=False, dtype=int
    #     )
    #     acq_arr1 = acq_arr0 + chunklen
    #     acq_arr = np.array([acq_arr0, acq_arr1]).transpose()
    #     del data_start, data_end, acq_arr0, acq_arr1
    #     if verbose:
    #         print("acq_arr.shape", acq_arr.shape)
    #     PSD = np.zeros(chunklen)
    #     # stdofPSD = np.array([np.linspace(1, numofpulse, numofpulse, endpoint=True), np.zeros(numofpulse)])
    #     i = 0
    #     frequencies, singlePSD = stdLIAPSD(
    #         data_x=self.dataX[acq_arr[i, 0] : acq_arr[i, 1]],
    #         data_y=self.dataY[acq_arr[i, 0] : acq_arr[i, 1]],
    #         samprate=self.sampRate,  # in Hz
    #         demodFreq=self.demodFreq,  # in Hz
    #         attenuation=self.attenuation,  # in dB. Power ratio (10^(attenuation/10))
    #         windowfunction=windowfunction,  # Hanning, Hamming, Blackman
    #         DTRCfilter=self.filterstatus,
    #         DTRCfilter_TC=self.filter_TC,
    #         DTRCfilter_order=self.filter_order,
    #     )
    #     correction = np.zeros(len(frequencies))
    #     # print('np.average(correction) ', np.average(correction))
    #     if len(ploycorrparas) != 0:
    #         correction = PolyEven(
    #             frequencies,
    #             ploycorrparas[0],
    #             ploycorrparas[1],
    #             ploycorrparas[2],
    #             ploycorrparas[3],
    #             ploycorrparas[4],
    #             ploycorrparas[5],
    #             self.demodFreq,
    #         )
    #         # print('np.average(correction) ', np.average(correction))
    #     self.interestingfreqvalue_list = []
    #     self.interestingfreqindex_list = []
    #     self.interestingPSDvalue_list = []

    #     if interestingfreq_list is not None and len(interestingfreq_list) != 0:
    #         numofintfreq = range(len(interestingfreq_list))
    #         if verbose:
    #             print("Analysis of single-frequency-channel is activated. ")
    #         freqresol = abs(frequencies[0] - frequencies[1])  # frequency resolution
    #         for targetfreq in interestingfreq_list:
    #             if verbose:
    #                 print("frequency resolution %g Hz" % freqresol)
    #                 print(
    #                     "np.amin(abs(frequencies-targetfreq)) ",
    #                     np.amin(abs(frequencies - targetfreq)),
    #                 )
    #             if np.amin(abs(frequencies - targetfreq)) > freqresol:
    #                 raise ValueError(
    #                     "interesting frequency %g Hz is not included in the spectrum"
    #                     % (targetfreq)
    #                 )
    #             index = np.argmin(abs(frequencies - targetfreq))
    #             self.interestingfreqindex_list.append(index)
    #             self.interestingfreqvalue_list.append(frequencies[index])
    #             self.interestingPSDvalue_list.append([])
    #         del singlePSD, freqresol, index

    #     if showstd:
    #         stdstart = np.argmin(abs(frequencies - stddev_range[0]))
    #         stdend = np.argmin(abs(frequencies - stddev_range[1]))
    #         if abs(stdstart - stdend) <= 10:
    #             print(
    #                 "abs(stdstart-stdend)<=10\ntoo less data for computing standard deviation"
    #             )
    #     self.stdofPSD_list = []
    #     # self.stdofASD = []
    #     if len(selectshot) == 0:
    #         chunk_list = range(numofchunk)
    #     else:
    #         chunk_list = selectshot

    #     numofchunk = len(chunk_list)
    #     for i in chunk_list:
    #         frequencies, singlePSD = stdLIAPSD(
    #             data_x=self.dataX[acq_arr[i, 0] : acq_arr[i, 1]],
    #             data_y=self.dataY[acq_arr[i, 0] : acq_arr[i, 1]],
    #             samprate=self.sampRate,  # in Hz
    #             demodFreq=self.demodFreq,  # in Hz
    #             attenuation=self.attenuation,  # in dB. Power ratio (10^(attenuation/10))
    #             windowfunction=windowfunction,  # Hanning, Hamming, Blackman
    #             DTRCfilter=self.filterstatus,
    #             DTRCfilter_TC=self.filter_TC,
    #             DTRCfilter_order=self.filter_order,
    #         )

    #         PSD += singlePSD - correction

    #         if showstd:
    #             self.stdofPSD_list.append(np.std(PSD[stdstart:stdend]) / (i + 1))
    #             # self.stdofASD.append(np.std(np.sqrt(PSD[stdstart:stdend]/ (i + 1)) ) )
    #         if interestingfreq_list is not None and len(interestingfreq_list) != 0:
    #             for i in numofintfreq:
    #                 (self.interestingPSDvalue_list[i]).append(
    #                     singlePSD[self.interestingfreqindex_list[i]]
    #                 )
    #     del singlePSD
    #     self.exptype = "Spin Noise Measurement"
    #     self.frequencies = frequencies
    #     self.avgPSD = PSD / numofchunk  # averaged PSD in V^2/Hz
    #     del acq_arr, frequencies, PSD
    #     self.interestingfreqindex_list = np.array(self.interestingfreqindex_list)
    #     self.interestingfreqvalue_list = np.array(self.interestingfreqvalue_list)
    #     self.interestingPSDvalue_list = np.array(self.interestingPSDvalue_list)
    #     self.correction = correction
    #     if verbose:
    #         print("self.interestingfreqindex_arr ", self.interestingfreqindex_list)
    #         print("self.interestingfreqvalue_arr ", self.interestingfreqvalue_list)
    #         print(
    #             "shape of self.interestingfreqindex_arr",
    #             (self.interestingPSDvalue_list).shape,
    #         )

    # @record_runtime_YorN(RECORD_RUNTIME)
    # def GetNoPulsePSD(
    #     self,
    #     windowfunction="rectangle",
    #     decayfactor=-10,
    #     chunksize=None,  # sec
    #     analysisrange=[0, -1],
    #     getstd=False,
    #     stddev_range=None,
    #     polycorrparas=[],
    #     interestingfreq_list=[],
    #     selectshots=[],
    #     verbose=False,
    # ):
    #     """
    #     Compute PSD of a no-pulse measurement.

    #     Parameters
    #     ----------
    #     windowfunction : str, optional
    #         window function for FFT.
    #         Available choices:
    #             'rectangle'
    #             'expdecay'
    #             'Hanning' or 'hanning' or 'Han' or 'han'
    #             'Hamming' or 'hamming'
    #             'Blackman'
    #         Defaults to 'rectangle'.

    #     decayfactor : float, optional
    #         parameter for window function 'expdecay'.
    #         The 'expdecay' window function generates an array of [exp(-df*0), exp(-df*1), ... , exp(-df*(num-1))],
    #         where df stands for decayfactor.
    #         Defaults to -10.

    #     chunksize : float, optional
    #         The size of a single chunk for computing one power spectrum in [s].
    #         This size determines how the time-series is sliced.
    #         You can set it to None or a very large chunksize so that you use the whole time-series.
    #         Defaults to None.

    #     analysisrange
    #         The range of time-series index that are analyzed.
    #         Defaults to [0,-1], which means all time-series ([0:-1]) are included.

    #     getstd : bool, optional
    #         The option whether to get standard deviation in stddev_range.
    #         Defaults to False.

    #     stddev_range : list, optional
    #         The frequency range in [Hz] from which the standard deviation is computed.
    #         Notice that this range means the absolute frequency range. e.g. [1.346e6 - 100, 1.346e6 + 200]
    #         Defaults to None. When it is None, the standard deviation would not be computed.

    #     polycorrparas : list, optional
    #         Parameters for a even-order polynomial correcting (subtracting) the power spectrum.
    #         check functioncache - PolyEven() for more details.
    #         Defaults to [], which means no correction would be made.

    #     interestingfreq_list : list, optional
    #         List of interesting frequencies, in [Hz].
    #         Usually users choose the frequencies near the Larmore frequency to see its amplitude over time.
    #         Defaults to [].

    #     selectshots : list, optional
    #         List of selected shots / chunks of acquired time-series for power spectra.
    #         e.g. [0] means that you only use the first shot.
    #         Defaults to [], which means all shots / chunks are included.

    #     verbose : bool, optional
    #         Defaults to False.

    #     Returns
    #     -------
    #     Null

    #     """

    #     # determine the chunksize
    #     # if the input chunksize is None or chunksize is larger than the whole time-series
    #     if chunksize is None or chunksize >= len(self.dataX) / self.sampRate:
    #         chunklen = len(self.dataX)
    #         self.chunksize = chunklen / self.sampRate
    #     else:
    #         self.chunksize = chunksize
    #         chunklen = int(chunksize * self.sampRate)
    #     # check(chunklen)
    #     # determine the range of time-series for analysis and number of chunks
    #     data_start = analysisrange[0]
    #     data_end = len(self.dataX) + analysisrange[1]
    #     numofchunk = (data_end - data_start + 1) // chunklen
    #     # Notice that '//' means there could be some data cannot be used after the last chunk.
    #     # check(numofchunk)

    #     # determine the acquisition (chunks)
    #     acq_arr0 = data_start + chunklen * np.linspace(
    #         start=0, stop=numofchunk, num=numofchunk, endpoint=False, dtype=int
    #     )
    #     acq_arr1 = acq_arr0 + chunklen
    #     acq_arr = np.array([acq_arr0, acq_arr1]).transpose()
    #     del data_start, data_end, acq_arr0, acq_arr1
    #     if verbose:
    #         print("acq_arr.shape", acq_arr.shape)

    #     # Initialize PSD array. the length of PSD is the length of a chunk.
    #     PSD = np.zeros(chunklen)

    #     # Generate the frequency axis of the power spectrum
    #     frequencies = np.sort(
    #         np.fft.fftfreq(chunklen, d=1.0 / self.sampRate) + self.demodFreq
    #     )

    #     # determine the correction (substraction) for PSD
    #     correction = np.zeros(len(frequencies))
    #     if polycorrparas is not None and len(polycorrparas) == 6:
    #         correction = PolyEven(
    #             frequencies,
    #             polycorrparas[0],
    #             polycorrparas[1],
    #             polycorrparas[2],
    #             polycorrparas[3],
    #             polycorrparas[4],
    #             polycorrparas[5],
    #             self.demodFreq,
    #         )

    #     # initialize lists of
    #     self.interestingfreqvalue_list = []  # interesting frequency values in [Hz]
    #     self.interestingfreqindex_list = []  # interesting frequency indexes
    #     self.interestingPSDvalue_list = []  # PSD of interesting frequencies

    #     # Store information of interesting frequencies
    #     if interestingfreq_list is not None and len(interestingfreq_list) != 0:
    #         numofintfreq = range(len(interestingfreq_list))
    #         if verbose:
    #             print("Analysis of single-frequency-bin is activated. ")
    #         freqresol = abs(frequencies[0] - frequencies[1])  # frequency resolution
    #         if verbose:
    #             check(freqresol)
    #         for intfreq in interestingfreq_list:
    #             if verbose:
    #                 check(np.amin(abs(frequencies - intfreq)))
    #             if np.amin(abs(frequencies - intfreq)) > freqresol:
    #                 raise ValueError(
    #                     f"interesting frequency {intfreq:g} [Hz] is not included in the spectrum"
    #                 )
    #             index = np.argmin(abs(frequencies - intfreq))
    #             self.interestingfreqindex_list.append(index)
    #             self.interestingfreqvalue_list.append(frequencies[index])
    #             self.interestingPSDvalue_list.append([])
    #         del freqresol, index

    #     # determine the range for computing the standard deviation
    #     if getstd:
    #         if stddev_range == []:
    #             stdstart = np.argmin(abs(frequencies - frequencies[0]))
    #             stdend = np.argmin(abs(frequencies - frequencies[-1]))
    #         else:
    #             stdstart = np.argmin(abs(frequencies - stddev_range[0]))
    #             stdend = np.argmin(abs(frequencies - stddev_range[-1]))
    #         if abs(stdstart - stdend) <= 10:
    #             print(
    #                 "WARNING!! abs(stdstart-stdend)<=10\ntoo less data for computing standard deviation"
    #             )

    #     # initialize the list of standard deviations
    #     # This list includes standard deviations computed during the accumulation of power spectra.
    #     # The process is like:
    #     # 1. Get the first power spectrum, then get one standard deviation.
    #     # 2. Get the second power spectrum, accumulate it onto the first one (plus).
    #     # Then compute the (standard deviation of the accumulated power spectrum) / 2.
    #     # Append this value to the list.
    #     # 3. Repeat step 2, until the end.
    #     self.stdofPSD_list = []

    #     # find the selected shots that are going to be used.
    #     if selectshots is None or len(selectshots) == 0:
    #         chunk_list = range(numofchunk)
    #     else:
    #         if np.amax(selectshots) + 1 > numofchunk:
    #             raise ValueError("np.amax(selectshots) + 1 > numofchunk. ")
    #         chunk_list = selectshots
    #     check(chunk_list)
    #     # Compute power spectra for selected chunks.
    #     for i in chunk_list:
    #         frequencies, singlePSD = stdLIAPSD(
    #             data_x=self.dataX[acq_arr[i, 0] : acq_arr[i, 1]],
    #             data_y=self.dataY[acq_arr[i, 0] : acq_arr[i, 1]],
    #             samprate=self.sampRate,
    #             demodFreq=self.demodFreq,
    #             attenuation=self.attenuation,
    #             windowfunction=windowfunction,
    #             decayfactor=decayfactor,
    #             DTRCfilter=self.filterstatus,
    #             DTRCfilter_TC=self.filter_TC,
    #             DTRCfilter_order=self.filter_order,
    #         )
    #         PSD += singlePSD - correction

    #         # Compute standard deviation
    #         if getstd:
    #             # self.stdofPSD_list.append(np.std(PSD[stdstart:stdend]) / (i + 1))
    #             self.stdofPSD_list.append(np.std(singlePSD[stdstart:stdend]))
    #         # Store PSD values of interesting frequencies
    #         if interestingfreq_list is not None and len(interestingfreq_list) != 0:
    #             for freqindex in self.interestingfreqindex_list:
    #                 (self.interestingPSDvalue_list[i]).append(singlePSD[freqindex])
    #         # delete singlePSD before going into the next iteration
    #         del singlePSD

    #     # store information
    #     self.frequencies = frequencies
    #     self.freq_resol = abs(frequencies[0] - frequencies[1])  # frequency resolution
    #     self.avgPSD = PSD / len(chunk_list)  # averaged PSD in V^2/Hz
    #     self.acq_arr = acq_arr
    #     del acq_arr, frequencies, PSD
    #     self.correction = correction
    #     self.selectPulse_list = chunk_list  # This is for plotting in self.GetSpectrum()
    #     # if verbose:
    #     #     self.interestingfreqindex_arr=np.array(self.interestingfreqindex_list)
    #     #     self.interestingfreqvalue_arr=np.array(self.interestingfreqvalue_list)
    #     #     self.interestingPSDvalue_arr=np.array(self.interestingPSDvalue_list)
    #     #     print('self.interestingfreqindex_arr ', self.interestingfreqindex_arr)
    #     #     print('self.interestingfreqvalue_arr ', self.interestingfreqvalue_arr)
    #     #     print('shape of self.interestingfreqindex_arr', (self.interestingPSDvalue_arr).shape)

    # @record_runtime_YorN(RECORD_RUNTIME)
    # def GetNoPulseFFTandPSD(
    #     self,
    #     windowfunction="rectangle",
    #     decayfactor=-10,
    #     chunksize=None,
    #     analysisrange=[0, -1],
    #     getstd=False,
    #     stddev_range=None,
    #     polycorrparas=[],
    #     interestingfreq_list=[],
    #     selectshots=[],
    #     verbose=False,
    # ):
    #     """
    #     Compute PSD of a no-pulse measurement.

    #     Parameters
    #     ----------
    #     windowfunction : str, optional
    #         window function for FFT.
    #         Available choices:
    #             'rectangle'
    #             'expdecay'
    #             'Hanning' or 'hanning' or 'Han' or 'han'
    #             'Hamming' or 'hamming'
    #             'Blackman'
    #         Defaults to 'rectangle'.

    #     decayfactor : float, optional
    #         parameter for window function 'expdecay'.
    #         The 'expdecay' window function generates an array of [exp(-df*0), exp(-df*1), ... , exp(-df*(num-1))],
    #         where df stands for decayfactor.
    #         Defaults to -10.

    #     chunksize : float, optional
    #         The size of a single chunk for computing one power spectrum in [s].
    #         This size determines how the time-series is sliced.
    #         You can set it to None or a very large chunksize so that you use the whole time-series.
    #         Defaults to None.

    #     analysisrange
    #         The range of time-series index that are analyzed.
    #         Defaults to [0,-1], which means all time-series ([0:-1]) are included.

    #     getstd : bool, optional
    #         The option whether to get standard deviation in stddev_range.
    #         Defaults to False.

    #     stddev_range : list, optional
    #         The frequency range in [Hz] from which the standard deviation is computed.
    #         Notice that this range means the absolute frequency range. e.g. [1.346e6 - 100, 1.346e6 + 200]
    #         Defaults to None. When it is None, the standard deviation would not be computed.

    #     polycorrparas : list, optional
    #         Parameters for a even-order polynomial correcting (subtracting) the power spectrum.
    #         check functioncache - PolyEven() for more details.
    #         Defaults to [], which means no correction would be made.

    #     interestingfreq_list : list, optional
    #         List of interesting frequencies, in [Hz].
    #         Usually users choose the frequencies near the Larmore frequency to see its amplitude over time.
    #         Defaults to [].

    #     selectshots : list, optional
    #         List of selected shots / chunks of acquired time-series for power spectra.
    #         e.g. [0] means that you only use the first shot.
    #         Defaults to [], which means all shots / chunks are included.

    #     verbose : bool, optional
    #         Defaults to False.

    #     Returns
    #     -------
    #     Null

    #     """
    #     if not hasattr(self, "avgFFT"):
    #         self.GetNoPulseFFT(
    #             windowfunction=windowfunction,
    #             decayfactor=decayfactor,
    #             chunksize=chunksize,
    #             analysisrange=analysisrange,
    #             getstd=getstd,
    #             stddev_range=stddev_range,
    #             polycorrparas=polycorrparas,
    #             interestingfreq_list=interestingfreq_list,
    #             selectshots=selectshots,
    #             verbose=verbose,
    #         )
    #     self.avgPSD = np.abs(self.avgFFT) ** 2.0

    # @record_runtime_YorN(RECORD_RUNTIME)
    # def GetNoPulseFFT(
    #     self,
    #     windowfunction: str = "rectangle",
    #     decayfactor: float = -10,
    #     chunksize: float = None,
    #     analysisrange: list = [0, -1],
    #     getstd: bool = False,
    #     stddev_range: list = None,
    #     polycorrparas: list = [],
    #     interestingfreq_list: list = [],
    #     selectshots: list = [],
    #     verbose: bool = False,
    # ):
    #     """
    #     Compute FFT of a (usually no-pulse) measurement.

    #     Parameters
    #     ----------
    #     windowfunction : str, optional
    #         window function for FFT.
    #         Available choices:
    #             'rectangle'
    #             'expdecay'
    #             'Hanning' or 'hanning' or 'Han' or 'han'
    #             'Hamming' or 'hamming'
    #             'Blackman'
    #         Defaults to 'rectangle'.

    #     decayfactor : float, optional
    #         parameter for window function 'expdecay'.
    #         The 'expdecay' window function generates an array of [exp(-df*0), exp(-df*1), ... , exp(-df*(num-1))],
    #         where df stands for decayfactor.
    #         Defaults to -10.

    #     chunksize : float, optional
    #         The size of a single chunk for computing one power spectrum in [s].
    #         This size determines how the time-series is sliced.
    #         You can set it to None or a very large chunksize so that you use the whole time-series.
    #         Defaults to None.

    #     analysisrange
    #         The range of time-series index that are analyzed.
    #         Defaults to [0,-1], which means all time-series ([0:-1]) are included.

    #     getstd : bool, optional
    #         The option whether to get standard deviation in stddev_range.
    #         Defaults to False.

    #     stddev_range : list, optional
    #         The frequency range in [Hz] from which the standard deviation is computed.
    #         Notice that this range means the absolute frequency range. e.g. [1.346e6 - 100, 1.346e6 + 200]
    #         Defaults to None. When it is None, the standard deviation would not be computed.

    #     polycorrparas : list, optional
    #         Parameters for a even-order polynomial correcting (subtracting) the power spectrum.
    #         check functioncache - PolyEven() for more details.
    #         Defaults to [], which means no correction would be made.

    #     interestingfreq_list : list, optional
    #         List of interesting frequencies, in [Hz].
    #         Usually users choose the frequencies near the Larmore frequency to see its amplitude over time.
    #         Defaults to [].

    #     selectshots : list, optional
    #         List of selected shots / chunks of acquired time-series for power spectra.
    #         e.g. [0] means that you only use the first shot.
    #         Defaults to [], which means all shots / chunks are included.

    #     verbose : bool, optional
    #         Defaults to False.

    #     Returns
    #     -------
    #     Null

    #     """

    #     # determine the chunksize
    #     # if the input chunksize is None or chunksize is larger than the whole time-series
    #     if chunksize is None or chunksize >= len(self.dataX) / self.sampRate:
    #         chunklen = len(self.dataX)
    #         self.chunksize = chunklen / self.sampRate
    #     else:
    #         self.chunksize = chunksize
    #         chunklen = int(chunksize * self.sampRate)
    #     # check(chunklen)
    #     # determine the range of time-series for analysis and number of chunks
    #     data_start = analysisrange[0]
    #     data_end = len(self.dataX) + analysisrange[1]
    #     numofchunk = (data_end - data_start + 1) // chunklen
    #     # Notice that '//' means there could be some data cannot be used after the last chunk.
    #     # check(numofchunk)

    #     # determine the acquisition (chunks)
    #     acq_arr0 = data_start + chunklen * np.linspace(
    #         start=0, stop=numofchunk, num=numofchunk, endpoint=False, dtype=int
    #     )
    #     acq_arr1 = acq_arr0 + chunklen
    #     acq_arr = np.array([acq_arr0, acq_arr1]).transpose()
    #     del data_start, data_end, acq_arr0, acq_arr1
    #     if verbose:
    #         print("acq_arr.shape", acq_arr.shape)

    #     # Initialize FFT array. The length of FFT is the length of a chunk.
    #     FFT = np.zeros(chunklen, dtype=complex)

    #     # Generate the frequency axis of the power spectrum
    #     frequencies = np.sort(
    #         np.fft.fftfreq(chunklen, d=1.0 / self.sampRate) + self.demodFreq
    #     )
    #     self.freq_resol = abs(frequencies[0] - frequencies[1])  # frequency resolution

    #     # determine the correction (substraction) for FFT
    #     correction = np.zeros(len(frequencies))
    #     if polycorrparas is not None and len(polycorrparas) == 6:
    #         correction = PolyEven(
    #             frequencies,
    #             polycorrparas[0],
    #             polycorrparas[1],
    #             polycorrparas[2],
    #             polycorrparas[3],
    #             polycorrparas[4],
    #             polycorrparas[5],
    #             self.demodFreq,
    #         )

    #     # initialize lists of
    #     self.interestingfreqvalue_list = []  # interesting frequency values in [Hz]
    #     self.interestingfreqindex_list = []  # interesting frequency indexes
    #     self.interestingFFTvalue_list = []  # FFT of interesting frequencies

    #     # Store information of interesting frequencies
    #     if interestingfreq_list is not None and len(interestingfreq_list) != 0:
    #         numofintfreq = range(len(interestingfreq_list))
    #         if verbose:
    #             print("Analysis of single-frequency-bin is activated. ")
    #         freqresol = abs(frequencies[0] - frequencies[1])  # frequency resolution
    #         if verbose:
    #             check(freqresol)
    #         for intfreq in interestingfreq_list:
    #             if verbose:
    #                 check(np.amin(abs(frequencies - intfreq)))
    #             if np.amin(abs(frequencies - intfreq)) > freqresol:
    #                 raise ValueError(
    #                     f"interesting frequency {intfreq:g} [Hz] is not included in the spectrum"
    #                 )
    #             index = np.argmin(abs(frequencies - intfreq))
    #             self.interestingfreqindex_list.append(index)
    #             self.interestingfreqvalue_list.append(frequencies[index])
    #             self.interestingFFTvalue_list.append([])
    #         del freqresol, index

    #     # determine the range for computing the standard deviation
    #     if getstd:
    #         stdstart = np.argmin(abs(frequencies - stddev_range[0]))
    #         stdend = np.argmin(abs(frequencies - stddev_range[1]))
    #         if abs(stdstart - stdend) <= 10:
    #             print(
    #                 "WARNING!! abs(stdstart-stdend)<=10\ntoo less data for computing standard deviation"
    #             )

    #     # initialize the list of standard deviations
    #     # This list includes standard deviations computed during the accumulation of power spectra.
    #     # The process is like:
    #     # 1. Get the first power spectrum, then get one standard deviation.
    #     # 2. Get the second power spectrum, accumulate it onto the first one (plus).
    #     # Then compute the (standard deviation of the accumulated power spectrum) / 2.
    #     # Append this value to the list.
    #     # 3. Repeat step 2, until the end.
    #     self.stdofFFT_list = []

    #     # find the selected shots that are going to be used.
    #     if selectshots is None or len(selectshots) == 0:
    #         chunk_list = range(numofchunk)
    #     else:
    #         if np.amax(selectshots) + 1 > numofchunk:
    #             raise ValueError("np.amax(selectshots) + 1 > numofchunk. ")
    #         chunk_list = selectshots

    #     # Compute power spectra for selected chunks.
    #     for i in chunk_list:
    #         frequencies, singleFFT = LIA_FFT_legacy(
    #             data_x=self.dataX[acq_arr[i, 0] : acq_arr[i, 1]],
    #             data_y=self.dataY[acq_arr[i, 0] : acq_arr[i, 1]],
    #             samprate=self.sampRate,
    #             dfreq=self.demodFreq,
    #             attenuation=self.attenuation,
    #             windowfunction=windowfunction,
    #             decayfactor=decayfactor,
    #             DTRCfilter=self.filterstatus,
    #             DTRCfilter_TC=self.filter_TC,
    #             DTRCfilter_order=self.filter_order,
    #         )
    #         FFT += singleFFT  # -correction

    #         # Compute standard deviation
    #         if getstd:
    #             self.stdofFFT_list.append(np.std(FFT[stdstart:stdend]) / (i + 1))
    #         # Store FFT values of interesting frequencies
    #         if interestingfreq_list is not None and len(interestingfreq_list) != 0:
    #             for freqindex in self.interestingfreqindex_list:
    #                 (self.interestingFFTvalue_list[i]).append(singleFFT[freqindex])
    #         # delete singleFFT before going into the next iteration
    #         del singleFFT

    #     # store information
    #     self.frequencies = frequencies
    #     self.avgFFT = FFT / len(chunk_list)  # averaged FFT in V/sqrt(Hz)
    #     self.acq_arr = acq_arr
    #     del acq_arr, frequencies, FFT
    #     self.correction = correction
    #     self.selectPulse_list = chunk_list  # This is for plotting in self.GetSpectrum()

    # def GetScopeFFT(
    #     self,
    #     windowfunction: str = "rectangle",
    #     decayfactor: float = -10,
    #     chunksize: float = None,
    #     analysisrange: list = [0, -1],
    #     getstd: bool = False,
    #     stddev_range: list = None,
    #     polycorrparas: list = [],
    #     interestingfreq_list: list = [],
    #     selectshots: list = [],
    #     verbose: bool = False,
    # ):
    #     """
    #     Compute FFT of a (usually no-pulse) measurement.

    #     Parameters
    #     ----------
    #     windowfunction : str, optional
    #         window function for FFT.
    #         Available choices:
    #             'rectangle'
    #             'expdecay'
    #             'Hanning' or 'hanning' or 'Han' or 'han'
    #             'Hamming' or 'hamming'
    #             'Blackman'
    #         Defaults to 'rectangle'.

    #     decayfactor : float, optional
    #         parameter for window function 'expdecay'.
    #         The 'expdecay' window function generates an array of [exp(-df*0), exp(-df*1), ... , exp(-df*(num-1))],
    #         where df stands for decayfactor.
    #         Defaults to -10.

    #     chunksize : float, optional
    #         The size of a single chunk for computing one power spectrum in [s].
    #         This size determines how the time-series is sliced.
    #         You can set it to None or a very large chunksize so that you use the whole time-series.
    #         Defaults to None.

    #     analysisrange
    #         The range of time-series index that are analyzed.
    #         Defaults to [0,-1], which means all time-series ([0:-1]) are included.

    #     getstd : bool, optional
    #         The option whether to get standard deviation in stddev_range.
    #         Defaults to False.

    #     stddev_range : list, optional
    #         The frequency range in [Hz] from which the standard deviation is computed.
    #         Notice that this range means the absolute frequency range. e.g. [1.346e6 - 100, 1.346e6 + 200]
    #         Defaults to None. When it is None, the standard deviation would not be computed.

    #     polycorrparas : list, optional
    #         Parameters for a even-order polynomial correcting (subtracting) the power spectrum.
    #         check functioncache - PolyEven() for more details.
    #         Defaults to [], which means no correction would be made.

    #     interestingfreq_list : list, optional
    #         List of interesting frequencies, in [Hz].
    #         Usually users choose the frequencies near the Larmore frequency to see its amplitude over time.
    #         Defaults to [].

    #     selectshots : list, optional
    #         List of selected shots / chunks of acquired time-series for power spectra.
    #         e.g. [0] means that you only use the first shot.
    #         Defaults to [], which means all shots / chunks are included.

    #     verbose : bool, optional
    #         Defaults to False.

    #     Returns
    #     -------
    #     Null

    #     """
    #     return

    # def plotFFT(self, specxlim=None, verbose=False):
    #     """
    #     Plot FFT.
    #     """
    #     assert hasattr(self, "avgFFT")
    #     fig = plt.figure(figsize=(20, 8))  #
    #     gs = gridspec.GridSpec(nrows=2, ncols=2)
    #     fig.subplots_adjust(
    #         top=0.91, bottom=0.11, left=0.08, right=0.96, hspace=0.0, wspace=0.25
    #     )
    #     real_ax = fig.add_subplot(gs[0, 0])
    #     imag_ax = fig.add_subplot(gs[1, 0])
    #     amp_ax = fig.add_subplot(gs[0, 1])
    #     phase_ax = fig.add_subplot(gs[1, 1])
    #     real_ax.plot(
    #         self.frequencies,
    #         self.avgFFT.real,
    #         label="Real part",
    #         color="tab:blue",
    #     )
    #     real_ax.set_ylabel("Amplitude")
    #     real_ax.grid(True)
    #     real_ax.legend(loc="upper right")
    #     # real_ax.tick_params(axis='y', left=False, labelleft=False)
    #     real_ax.tick_params(axis="x", bottom=False, labelbottom=False)
    #     # bottom, top, left, right : bool : Whether to draw the respective ticks
    #     # labelbottom, labeltop, labelleft, labelright : bool : Whether to draw the respective tick labels.
    #     imag_ax.plot(
    #         self.frequencies,
    #         self.avgFFT.imag,
    #         label="Imaginary part",
    #         color="tab:orange",
    #     )
    #     imag_ax.set_ylabel("Amplitude")
    #     imag_ax.set_xlabel("Frequency [Hz]")
    #     imag_ax.grid(True)
    #     # imag_ax.tick_params(axis='y', left=False, labelleft=False)
    #     imag_ax.legend(loc="upper right")
    #     amp_ax.plot(
    #         self.frequencies,
    #         np.abs(self.avgFFT) ** 2,
    #         label="Amplitude of FFT^2",
    #         color="tab:purple",
    #     )
    #     amp_ax.set_ylabel("Amplitude")
    #     amp_ax.grid(True)
    #     amp_ax.legend(loc="upper right")
    #     # amp_ax.tick_params(axis='y', left=False, labelleft=False)
    #     amp_ax.tick_params(axis="x", bottom=False, labelbottom=False)

    #     phase_ax.plot(
    #         self.frequencies,
    #         np.angle(self.avgFFT, deg=True),
    #         label="Phase of  FFT",
    #         color="tab:cyan",
    #     )
    #     phase_ax.set_ylabel("Phase / $\degree$")
    #     phase_ax.set_xlabel("Frequency [Hz]")
    #     phase_ax.grid(True)
    #     phase_ax.legend(loc="upper right")
    #     if specxlim is not None:
    #         real_ax.set_xlim(specxlim[0], specxlim[1])
    #         imag_ax.set_xlim(specxlim[0], specxlim[1])
    #         amp_ax.set_xlim(specxlim[0], specxlim[1])
    #         phase_ax.set_xlim(specxlim[0], specxlim[1])

    #     titletext = self.exptype  #
    #     for singlefile in self.filelist:
    #         titletext += "\n" + singlefile
    #     fig.suptitle(titletext, wrap=True)  # , fontsize=8

    #     plt.show()

    # def MonoFreqAnalysis(
    #     self,
    #     showtimedomain=True,
    #     showfreqdomain=True,
    #     specscale="log",
    #     left_spc=0.1,
    #     top_spc=1 - 0.1,
    #     right_spc=1 - 0.05,
    #     bottom_spc=0.1,
    #     xgrid_spc=0.3,
    #     ygrid_spc=0.2,
    #     # vlinex = None,
    #     showplt_opt=True,
    #     saveplt_opt=False,
    #     saveplt_path=None,
    #     pltmsg="",
    #     verbose=False,
    # ):
    #     """
    #     Not in use

    #     Args:
    #         showtimedomain : bool, optional Defaults to True.
    #         showfreqdomain : bool, optional Defaults to True.
    #         specscale : str, optional Defaults to 'log'.
    #         left_spc (float, optional Defaults to 0.1.
    #         top_spc : type_, optional Defaults to 1-0.1.
    #         right_spc : type_, optional Defaults to 1-.05.
    #         bottom_spc (float, optional Defaults to .1.
    #         xgrid_spc (float, optional Defaults to .3.
    #         ygrid_spc (float, optional Defaults to .2.
    #         showplt_opt : bool, optional Defaults to True.
    #         saveplt_opt : bool, optional Defaults to False.
    #         saveplt_path : type_, optional Defaults to None.
    #         pltmsg : str, optional Defaults to ''.
    #         verbose : bool, optional Defaults to False.

    #     Raises:
    #         ValueError: _description_

    #     Returns:
    #         _type_: _description_
    #     """
    #     for i in range(len(self.interestingfreqvalue_list)):
    #         if verbose:
    #             # print('timestamp.shape ', timestamp.shape)
    #             print(
    #                 "self.interestingPSDvalue_arr[i].shape",
    #                 self.interestingPSDvalue_list[i].shape,
    #             )
    #         if showtimedomain and (not showfreqdomain):
    #             fig = plt.figure(figsize=(14, 9))  #
    #             gs = gridspec.GridSpec(nrows=1, ncols=1)  #
    #             fig.subplots_adjust(
    #                 left=left_spc,
    #                 top=top_spc,
    #                 right=right_spc,
    #                 bottom=bottom_spc,
    #                 wspace=xgrid_spc,
    #                 hspace=ygrid_spc,
    #             )
    #             # make plot for time domain signal
    #             timedm_ax = fig.add_subplot(gs[0, 0])
    #             timestamp = np.linspace(
    #                 start=0,
    #                 stop=len(self.interestingPSDvalue_list[i]) * self.chunksize,
    #                 num=len(self.interestingPSDvalue_list[i]),
    #                 endpoint=False,
    #             )
    #             timedm_ax.plot(
    #                 timestamp,
    #                 self.interestingPSDvalue_list[i],
    #                 label="%5.7g MHz" % (1e-6 * self.interestingfreqvalue_list[i]),
    #                 c="tab:green",
    #             )
    #             timedm_ax.set_xlabel("time / s")
    #             timedm_ax.set_ylabel("Power spectral density / $V^2/Hz$")
    #             # timedm_ax.set_xlim(timestamp[0], timestamp[-1])
    #             timedm_ax.legend(loc="upper right")
    #         elif (not showtimedomain) and showfreqdomain:
    #             fig = plt.figure(figsize=(14, 9))  #
    #             gs = gridspec.GridSpec(nrows=1, ncols=1)  #
    #             fig.subplots_adjust(
    #                 left=left_spc,
    #                 top=top_spc,
    #                 right=right_spc,
    #                 bottom=bottom_spc,
    #                 wspace=xgrid_spc,
    #                 hspace=ygrid_spc,
    #             )
    #             # make plot for frequency domain signal
    #             freqdm_ax = fig.add_subplot(gs[0, 0])
    #             frequencies, PSD = stdPSD(
    #                 data=self.interestingPSDvalue_list[i],
    #                 samprate=1 / self.chunksize,
    #                 windowfunction="rectangle",  # Hanning, Hamming, Blackman
    #                 verbose=False,
    #             )
    #             freqdm_ax.plot(
    #                 frequencies,
    #                 PSD,
    #                 label="%5.7g MHz" % (1e-6 * self.interestingfreqvalue_list[i]),
    #                 c="tab:blue",
    #             )
    #             freqdm_ax.set_xscale(specscale)
    #             freqdm_ax.set_yscale(specscale)
    #             freqdm_ax.set_xlabel("frequency (Hz)")
    #             freqdm_ax.set_ylabel("PSD of PSD amplitude / $V^{4}/Hz^{3}$")
    #             freqdm_ax.legend(loc="upper right")
    #         elif showtimedomain and showfreqdomain:
    #             fig = plt.figure(figsize=(21, 9))  #
    #             gs = gridspec.GridSpec(nrows=1, ncols=2)  #
    #             fig.subplots_adjust(
    #                 left=left_spc,
    #                 top=top_spc,
    #                 right=right_spc,
    #                 bottom=bottom_spc,
    #                 wspace=xgrid_spc,
    #                 hspace=ygrid_spc,
    #             )
    #             # make plot for time domain signal
    #             timedm_ax = fig.add_subplot(gs[0, 0])
    #             timestamp = np.linspace(
    #                 start=0,
    #                 stop=len(self.interestingPSDvalue_list[i]) * self.chunksize,
    #                 num=len(self.interestingPSDvalue_list[i]),
    #                 endpoint=False,
    #             )
    #             timedm_ax.plot(
    #                 timestamp,
    #                 self.interestingPSDvalue_list[i],
    #                 label="%5.7g MHz" % (1e-6 * self.interestingfreqvalue_list[i]),
    #                 c="tab:green",
    #             )
    #             timedm_ax.set_xlabel("time / s")
    #             timedm_ax.set_ylabel("Power spectral density / $V^2/Hz$")
    #             # timedm_ax.set_xlim(timestamp[0], timestamp[-1])
    #             timedm_ax.legend(loc="upper right")
    #             # make plot for frequency domain signal
    #             freqdm_ax = fig.add_subplot(gs[0, 1])
    #             frequencies, PSD = stdPSD(
    #                 data=self.interestingPSDvalue_list[i],
    #                 samprate=1 / self.chunksize,
    #                 windowfunction="rectangle",  # Hanning, Hamming, Blackman
    #                 verbose=False,
    #             )
    #             freqdm_ax.plot(
    #                 frequencies,
    #                 PSD,
    #                 label="%5.7g MHz" % (1e-6 * self.interestingfreqvalue_list[i]),
    #                 c="tab:blue",
    #             )
    #             freqdm_ax.set_xscale(specscale)
    #             freqdm_ax.set_yscale(specscale)
    #             freqdm_ax.set_xlabel("frequency (Hz)")
    #             freqdm_ax.set_ylabel("PSD of PSD amplitude / $V^{4}/Hz^{3}$")
    #             freqdm_ax.legend(loc="upper right")
    #         else:
    #             raise ValueError("showtimedomain and showfreqdomain both False")

    #         titletext = "Mono-Frequency Channle Analysis"
    #         fig.suptitle(titletext, wrap=True)
    #         plt.grid()
    #         if saveplt_opt:
    #             plt.savefig(
    #                 saveplt_path + self.exptype + " " + titletext + pltmsg + ".png"
    #             )
    #         if showplt_opt:
    #             plt.show()
    #         plt.close()
    #         del fig, gs
    #     return 0

    # def GetAvgFFT(
    #     self,
    #     # AVARPopt: bool = False,
    #     windowfunction="rectangle",
    #     selectshots=[],
    #     showplt=False,
    #     specxlim=None,
    #     verbose=False,
    # ):
    #     """
    #     Has not been maintained for a while. If you want to use it, please let Yuzhe check it.

    #     Args:
    #         windowfunction : str, optional Defaults to 'rectangle'.
    #         selectshots : list, optional Defaults to [].
    #         showplt : bool, optional Defaults to False.
    #         specxlim : type_, optional Defaults to None.
    #         verbose : bool, optional Defaults to False.
    #     """
    #     if not hasattr(self, "acqTime"):
    #         raise AttributeError(
    #             "'LIASignal' object has no attribute 'acqTime'. Maybe the stream is a no-pulse stream? "
    #         )
    #     acqtimelen = int(np.abs(self.acq_arr[0, 0] - self.acq_arr[0, 1]))
    #     FFT = np.zeros(acqtimelen, dtype=complex)

    #     if len(selectshots) == 0:
    #         selectFFTindex = range(len(self.endofpulse))
    #     else:
    #         selectFFTindex = selectshots
    #     # numofpulses = len(self.endofpulse)
    #     for i in selectFFTindex:
    #         frequencies, singleFFT = LIA_FFT_legacy(
    #             data_x=self.dataX[self.acq_arr[i, 0] : self.acq_arr[i, 1]],
    #             data_y=self.dataY[self.acq_arr[i, 0] : self.acq_arr[i, 1]],
    #             samprate=self.sampRate,  # in Hz
    #             dfreq=self.demodFreq,  # in Hz
    #             attenuation=self.attenuation,  # in dB. Power ratio (10^(attenuation/10))
    #             windowfunction=windowfunction,  # Hanning, Hamming, Blackman
    #             DTRCfilter=self.filterstatus,
    #             DTRCfilter_TC=self.filter_TC,
    #             DTRCfilter_order=self.filter_order,
    #             verbose=verbose,
    #         )
    #         print(singleFFT.shape)
    #         FFT += singleFFT
    #         del singleFFT

    #     self.frequencies = frequencies
    #     self.avgFFT = FFT / len(self.endofpulse)
    #     self.avgPSD = abs(self.avgFFT) ** 2
    #     del frequencies, FFT
    #     if showplt:
    #         fig = plt.figure(figsize=(20, 8))  #
    #         gs = gridspec.GridSpec(nrows=2, ncols=2)
    #         fig.subplots_adjust(
    #             top=0.91, bottom=0.11, left=0.08, right=0.96, hspace=0.0, wspace=0.25
    #         )
    #         real_ax = fig.add_subplot(gs[0, 0])
    #         imag_ax = fig.add_subplot(gs[1, 0], sharex=real_ax)
    #         amp_ax = fig.add_subplot(gs[0, 1], sharex=real_ax)
    #         phase_ax = fig.add_subplot(gs[1, 1], sharex=real_ax)
    #         real_ax.plot(
    #             self.frequencies,
    #             self.avgFFT.real,
    #             label="Real part",
    #             color="tab:blue",
    #         )
    #         real_ax.set_ylabel("Amplitude (Arb. Units)")
    #         real_ax.grid(True)
    #         real_ax.legend(loc="upper right")
    #         # real_ax.tick_params(axis='y', left=False, labelleft=False)
    #         real_ax.tick_params(axis="x", bottom=False, labelbottom=False)
    #         # bottom, top, left, right : bool : Whether to draw the respective ticks
    #         # labelbottom, labeltop, labelleft, labelright : bool : Whether to draw the respective tick labels.
    #         imag_ax.plot(
    #             self.frequencies,
    #             self.avgFFT.imag,
    #             label="Imaginary part",
    #             color="tab:orange",
    #         )
    #         imag_ax.set_ylabel("Amplitude (Arb. Units)")
    #         imag_ax.set_xlabel("Frequency (Hz)")
    #         imag_ax.grid(True)
    #         # imag_ax.tick_params(axis='y', left=False, labelleft=False)
    #         imag_ax.legend(loc="upper right")
    #         amp_ax.plot(
    #             self.frequencies,
    #             np.abs(self.avgFFT) ** 2,
    #             label="Amplitude of FFT^2",
    #             color="tab:purple",
    #         )
    #         amp_ax.set_ylabel("Amplitude (Arb. Units)")
    #         amp_ax.grid(True)
    #         amp_ax.legend(loc="upper right")
    #         # amp_ax.tick_params(axis='y', left=False, labelleft=False)
    #         amp_ax.tick_params(axis="x", bottom=False, labelbottom=False)

    #         phase_ax.plot(
    #             self.frequencies,
    #             np.angle(self.avgFFT, deg=True),
    #             label="Phase of  FFT",
    #             color="tab:cyan",
    #         )
    #         phase_ax.set_ylabel("Phase / $\degree$")
    #         phase_ax.set_xlabel("Frequency (Hz)")
    #         phase_ax.grid(True)
    #         phase_ax.legend(loc="upper right")
    #         if specxlim is not None:
    #             real_ax.set_xlim(specxlim[0], specxlim[1])
    #             imag_ax.set_xlim(specxlim[0], specxlim[1])
    #             amp_ax.set_xlim(specxlim[0], specxlim[1])
    #             phase_ax.set_xlim(specxlim[0], specxlim[1])

    #         titletext = self.exptype  # 'All shots of '+
    #         for singlefile in self.filelist:
    #             titletext += "\n" + singlefile
    #         fig.suptitle(titletext, wrap=True)  # , fontsize=8

    #         plt.show()

    # def GetSpectrogram(
    #     self,
    #     acq_time: list = [],
    #     FreqRange: list = [],
    #     delta_t: float = 1.0,
    #     delta_f: float = 1.0,
    #     showplt: bool = True,
    #     verbose: bool = False,
    # ):
    #     # if not hasattr(self, 'timestamp'):
    #     #     self.timestamp = np.linspace(start=0, stop=len(self.dataX) / self.samprate, \
    #     #         num=len(self.dataX), endpoint=False, dtype=float)
    #     hop = int(delta_t * self.sampRate)
    #     mfft = int(self.sampRate / delta_f)  # length of data for a fft
    #     if len(acq_time) == 0:
    #         r0, r1 = 0, -1
    #         N = len(self.dataX)
    #     SFT = ShortTimeFFT(
    #         np.hanning(mfft),
    #         hop=hop,
    #         fs=self.sampRate,
    #         mfft=mfft,
    #         scale_to="psd",
    #         fft_mode="centered",
    #     )
    #     Sf = SFT.stft(self.dataX + 1j * self.dataY)  # perform the STFT
    #     check(Sf.shape)
    #     check(np.mean(Sf))
    #     check(np.std(Sf))
    #     # In the plot, the time extent of the signal x is marked by
    #     # vertical dashed lines. Note that the SFT produces values
    #     # outside the time range of x. The shaded areas on the left
    #     # and the right indicate border effects caused by the window
    #     # slices in that area not fully being inside time range of x:

    #     fig = plt.figure(figsize=(8.0, 6.0), dpi=150)  # initialize a figure
    #     # to specify heights and widths of subfigures
    #     width_ratios = [1, 0.5]
    #     height_ratios = [0.25, 0.25, 1]
    #     gs = gridspec.GridSpec(
    #         nrows=3, ncols=2, width_ratios=width_ratios, height_ratios=height_ratios
    #     )  #
    #     dataX_ax = fig.add_subplot(gs[0, 0])
    #     dataY_ax = fig.add_subplot(gs[1, 0], sharex=dataX_ax, sharey=dataX_ax)
    #     specgm_ax = fig.add_subplot(gs[2, 0], sharex=dataX_ax)

    #     def plotTS(ax, data, label, color, scatter):
    #         ax.plot(self.timeStamp[0:-1], data[0:-1], label=label, c=color)
    #         if scatter is not None and len(scatter) > 0:
    #             for idx in scatter[0:]:
    #                 ax.scatter(
    #                     self.timeStamp[0:-1][idx],
    #                     data[0:-1][idx],
    #                     marker="*",
    #                     c="tab:red",
    #                 )
    #         ax.set_ylabel("Voltage [V]")
    #         ax.legend(loc="upper right")  # adjust the location of the legend

    #     if not hasattr(self, "Jumps_X"):
    #         self.tsCheckJump()
    #     plotTS(dataX_ax, self.dataX, "LIA X", "tab:green", self.Jumps_X)
    #     plotTS(dataY_ax, self.dataY, "LIA Y", "tab:brown", self.Jumps_Y)

    #     # fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit

    #     t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
    #     specgm_ax.set_title(
    #         rf"ShortTime FFT ({SFT.m_num*SFT.T:.1g}$\,s$ hanning window)"
    #     )
    #     specgm_ax.set(
    #         xlabel=f"time [sec] ({SFT.p_num(N)} slices, "
    #         + rf"$\Delta t = {SFT.delta_t:.1g}\,$s)",
    #         ylabel=f"Frequency [Hz] ({SFT.f_pts} bins, "
    #         + rf"$\Delta f = {SFT.delta_f:.1g}\,$Hz)",
    #         xlim=(t_lo, t_hi),
    #     )

    #     im1 = specgm_ax.imshow(
    #         abs(Sf), origin="lower", aspect="auto", extent=SFT.extent(N), cmap="viridis"
    #     )
    #     # ax1.plot(t_x, f_i, 'r--', alpha=.5, label='$f_i(t)$')
    #     fig.colorbar(im1, label="Magnitude $|S_x(t, f)|$")

    #     # Shade areas where window slices stick out to the side:
    #     for t0_, t1_ in [
    #         (t_lo, SFT.lower_border_end[0] * SFT.T),
    #         (SFT.upper_border_begin(N)[0] * SFT.T, t_hi),
    #     ]:
    #         specgm_ax.axvspan(t0_, t1_, color="w", linewidth=0, alpha=0.2)
    #     for t_ in [0, N * SFT.T]:  # mark signal borders with vertical line:
    #         specgm_ax.axvline(t_, color="y", linestyle="--", alpha=0.5)
    #     # specgm_ax.legend()
    #     fig.tight_layout()
    #     if showplt:
    #         plt.show()
    #     plt.close()
    #     return fig, specgm_ax

    # def InsertSignalFFT(self, func, verbose: bool = False):
    #     if not hasattr(self, "avgFFT"):
    #         self.GetNoPulseFFT()
    #     self.avgFFT += func(self.frequencies)

    # def GetT1(
    #     self,
    #     # acqDelay: float,
    #     # acqTime: float,
    #     windowfunction="rectangle",
    #     # decayfactor=-10,
    #     selectshots=[],
    #     verbose=False,
    # ):
    #     """

    #     Args:
    #         windowfunction : str, optional Defaults to 'rectangle'.
    #         selectshots : list, optional Defaults to [].
    #         verbose : bool, optional Defaults to False.
    #     """
    #     # acqtimelen = int(self.acqTime * self.samprate)
    #     # check(self.acq_arr)
    #     raise ValueError(
    #         f"[{self.GetT1.__name__}] has not been maintained for a while. If you want to use it, please let Yuzhe check it. "
    #     )
    #     PSD = np.zeros(abs(self.acq_arr[0, 1] - self.acq_arr[0, 0]))
    #     PSD_list = []
    #     if len(selectshots) == 0 or selectshots is None:
    #         selectPSDindex = range(len(self.acq_arr[:, 0]))
    #     else:
    #         selectPSDindex = selectshots
    #     # numofpulses = len(self.endofpulse)
    #     for i in selectPSDindex:
    #         frequencies, singlePSD = stdLIAPSD(
    #             data_x=self.dataX[self.acq_arr[i, 0] : self.acq_arr[i, 1]],
    #             data_y=self.dataY[self.acq_arr[i, 0] : self.acq_arr[i, 1]],
    #             samprate=self.sampRate,  # in Hz
    #             demodFreq=self.demodFreq,  # in Hz
    #             attenuation=self.attenuation,  # in dB. Power ratio (10^(attenuation/10))
    #             windowfunction=windowfunction,  # Hanning, Hamming, Blackman
    #             decayfactor=-10,
    #             showwindow=False,
    #             DTRCfilter=self.filterstatus,
    #             DTRCfilter_TC=self.filter_TC,
    #             DTRCfilter_order=self.filter_order,
    #             verbose=verbose,
    #         )
    #         PSD_list.append(singlePSD)
    #         PSD += singlePSD
    #         del singlePSD

    #     # gc.collect()

    #     self.frequencies = frequencies
    #     self.avgPSD = PSD / len(selectPSDindex)
    #     del frequencies, PSD
    #     # gc.collect()

    def GetSpectrum(
        self,
        showtimedomain=True,
        showacqdata=False,
        showfreqdomain=True,
        showfit=False,
        showresidual=False,
        showchanstd=False,
        spectype="FluxPSD",  # 'PSD', 'ASD', 'FluxPSD', 'FluxASD'
        Mf=None,  # feedback sensitivity
        Rf=None,  # in Ohm
        specxunit="Hz",  # 'Hz' 'kHz' 'MHz' 'GHz' 'ppm' 'ppb'
        referFreq=None,
        specxunit2=None,  # 'Hz' 'kHz' 'MHz' 'GHz' 'ppm' 'ppb'
        specx2format="{:.3f}",
        ampunit="expPhi",  #  'muV' 'V' 'muPhi' 'Phi' 'expPhi'
        amppow=None,
        specyscale="linear",  # 'log', 'linear'
        showstd=False,
        stddev_range: list = None,
        specxlim=[],
        specylim=[],
        showlegend=True,
        legendloc="upper right",  # 'best' 'upper right' 'upper left' 'lower left' 'lower right'
        figsize=(16 * 0.8, 9 * 0.8),
        top=None,
        bottom=None,
        left=None,
        right=None,
        hspace=None,
        wspace=None,
        vlinex=None,
        tightlayout_opt=False,
        showplt_opt=True,
        saveplt_opt=False,
        saveSpecData_opt=False,
        save_path=None,
        save_msg="",
        return_opt=False,
        verbose=False,
    ):
        """
        Generate the spectrum for visulization of the analysis.

        Parameters
        ----------
        showtimedomain : bool, optional
            Option to show time-series signals.
            Defaults to True.

        showacqdata : bool, optional
            Option to show the time-series acquired for FFT.
            Only effective when showtimedomain == True.
            Defaults to False.

        showfreqdomain : bool, optional
            Option to show frequency-domain signals.
            Defaults to True.

        showfit : bool, optional
            Option to show fit result.
            Defaults to False.

        showresidual : bool, optional
            Option to show the residual of fitting.
            Defaults to False.

        showchanstd : bool, optional
            Option to show the standard deviations of frequency channels.
            Defaults to False.

        spectype : str, optional
            Options are 'PSD', 'ASD', 'FluxPSD' and 'FluxASD', in units of
            'V^2 (Hz)', 'V / sqrt(Hz)', 'Phi0^2 (Hz)' and 'Phi0 / sqrt(Hz)' respectively.
            Defaults to 'FluxPSD'.

        Mf : float, optional
            Feedback sensitivity which can be found in the SQUID specifications.
            For the SQUID we usually use, M_f = 31 706 \\phi_0 / A???? Is this True?
            Defaults to 1. / (44.12e-6).

        Rf : float, optional
            Defaults to 10e3 [Ohm].

        specxunit2 : str, optional
            The unit for the second x-axis of the spectrum.
            Defaults to None.

        specx2format : str, optional
            Defaults to '{:.3f}'.

        ampunit : str, optional
            The unit for the vertical axis of the spectrum.
            Defaults to 'expPhi'.

        specyscale : str, optional
            The y-scale of the spectrum, which can be 'linear' or 'log'.
            Defaults to 'linear'.

        stddev_range : list, optional
            The absolute frequency range for computing the standard deviation,
            which is usually the noise.
            Defaults to [1.346e6 - 500, 1.346e6 + 500].

        specxlim, specylim : list, optional
            The absolute range for plot x- or y-axis display limit.
            e.g. specxlim = [1.346e6 - 500, 1.346e6 + 500], specylim = [1e-13, 1e-8]
            The units of x-axis and y-axis are determined by specxunit and ampunit.
            Default to [], which means the plot automatically decides its x or y limits.

        showlegend : bool, optional
            Option to show the plot legend.
            Defaults to True.

        legendloc : str, optional
            Location of the plot legend.
            Other options are 'upper right', 'upper left', 'lower left' and 'lower right'.
            Defaults to 'best'.

        top, bottom : float, optional
            The position of the top / bottom edge of the subplots, as a fraction of the
            figure height.
            Defaults to None, which means the parameters will be automatically chosen.

        left, right : float, optional
            The position of the left / right edge of the subplots, as a fraction of the
            figure width.
            Defaults to None, which means the parameters will be automatically chosen.

        hspace, wspace : float, optional
            The height / width of the padding between subplots, as a fraction of the
            average Axes width.
            Defaults to None, which means the parameters will be automatically chosen.

        vlinex : float, optional
            This parameter is not in use at the moment.
            Defaults to None.

        tightlayout_opt : bool, optional
            Decide whether to have tightlayout.
            Defaults to False.

        showplt_opt : bool, optional
            Option to show the plot.
            Defaults to True.

        saveplt_opt : bool, optional
            Option to save the plot.
            Defaults to False.

        saveSpecData_opt : bool, optional
            Option to save spectrum data, including specxaxis, self.spectrum, specxunit, specyunit
            Defaults to False.

        save_path : str, optional
            The path for saving plot or data.
            Defaults to None.

        save_msg : str, optional
            a message that is added as the suffix of the file name.
            Defaults to ''.

        return_opt : bool, optional
            Option to return the spectral data.
            If True, the function returns specxaxis, self.spectrum, specxunit, specyunit.
            Defaults to False.

        verbose : bool, optional
            Defaults to False.


        Returns
        ---------
        if return_opt:
            return specxaxis, self.spectrum, specxunit, specyunit
        """

        # plt.rcParams["font.family"] = "Times New Roman"
        # plt.rcParams["mathtext.fontset"] = 'cm'  # 'dejavuserif'

        # If show standard deviation in the plot
        if showstd:
            # The start and the end of a range for computing the standard deviation
            # stdstart and stdend are the indice of the array
            if stddev_range is None:
                stdstart = 0
                stdend = len(self.freqs)
            else:
                stdstart = np.argmin(abs(self.freqs - stddev_range[0]))
                stdend = np.argmin(abs(self.freqs - stddev_range[1]))

            # If there are not many data points in this range, a warning will be raised.
            if abs(stdstart - stdend) <= 10:
                print(
                    "abs(stdstart-stdend)<=10\ntoo little data for computing standard deviation"
                )

            # Compute the standard deviation
            self.avgPSDstd = np.std(self.avgPSD[stdstart:stdend])

        # The ratios of the heights of subplots
        height_ratios = [1, 1, 1, 0, 1]

        # If show fitting, and fitting was done, and show residual of the fitting
        if showfit and self.fitflag and showresidual:
            # decide the height of the subplot of the residual so that the scale can be
            # approximately consistent with the spectrum
            height_ratios[3] = (
                3
                * 1.05
                * (np.amax(self.residual) - np.amin(self.residual))
                / (np.amax(self.avgPSD) - np.amin(self.avgPSD))
            )

        # decide the height of the chanstd plot
        if showchanstd is not True:
            height_ratios[4] = 0

        # Currently the function cannot plot Kea time-series
        if showtimedomain and self.exptype == "Kea Pulsed-NMR":
            print("WARNING! Kea data has no pulse sequence information")
            showtimedomain = False

        # Currently the function cannot plot time-series of no-pulse measurements
        # Often, no-pulse measurements last so long that it would be very difficult
        # to plot
        if showtimedomain and self.exptype == "No-pulse NMR Measurement":
            print("WARNING! This is a No-pulse NMR Measurement")
            showtimedomain = False

        # Initialize the figure
        fig = plt.figure(figsize=figsize, dpi=100)  #

        # Initialize the grid structure for subplots
        gs = gridspec.GridSpec(nrows=5, ncols=2, height_ratios=height_ratios)  #
        # gs = gridspec.GridSpec(nrows=rownum, ncols=numcols, width_ratios=widths, height_ratios=heights)

        # If every parameter for the layout is specified
        if None not in [top, bottom, left, right, hspace, wspace]:
            fig.subplots_adjust(
                top=top,
                bottom=bottom,
                left=left,
                right=right,
                wspace=wspace,
                hspace=hspace,
            )
        # elsewise, adjust the layout by specific parameters
        else:
            if top is not None:
                fig.subplots_adjust(top=top)
            if bottom is not None:
                fig.subplots_adjust(bottom=bottom)
            if left is not None:
                fig.subplots_adjust(left=left)
            if right is not None:
                fig.subplots_adjust(right=right)
            if wspace is not None:
                fig.subplots_adjust(wspace=wspace)
            if hspace is not None:
                fig.subplots_adjust(hspace=hspace)

        # if show time domain (time-series) and frequency domain (spectrum)
        if showtimedomain and showfreqdomain:
            # if there is no timestamp for the x-axis of the time-series,
            # a timestamp will be generated
            if self.timeStamp is None:
                self.timeStamp = np.linspace(
                    start=0,
                    stop=len(self.dataX) / self.sampRate,
                    num=len(self.dataX),
                    endpoint=False,
                    dtype=float,
                )
            # add subplots of pulse data, X and Y channel of the lock-in.
            pulse_ax = fig.add_subplot(gs[0, 0])
            dataX_ax = fig.add_subplot(
                gs[1, 0], sharex=pulse_ax
            )  # share x-axis with pulse_ax
            dataY_ax = fig.add_subplot(
                gs[2, 0], sharex=pulse_ax, sharey=dataX_ax
            )  # share x-axis with dataX_ax
            # add the subplot of the spectrum
            spec_ax = fig.add_subplot(gs[0:3, -1])

            # add the subplot for the fit residual
            if showresidual:
                resi_ax = fig.add_subplot(gs[3, -1], sharex=spec_ax)
            # add the subplot for the channel standard deviation
            if showchanstd:
                chanstd_ax = fig.add_subplot(gs[4, -1], sharex=spec_ax)
        elif (not showtimedomain) and (showfreqdomain):
            spec_ax = fig.add_subplot(gs[0:3, :])
            # add the subplot for the fit residual
            if showresidual:
                resi_ax = fig.add_subplot(gs[3, :], sharex=spec_ax)
            # add the subplot for the channel standard deviation
            if showchanstd:
                chanstd_ax = fig.add_subplot(gs[4, :], sharex=spec_ax)
        elif (showtimedomain) and (not showfreqdomain):
            # if there is no timestamp for the x-axis of the time-series,
            # a timestamp will be generated
            if self.timeStamp is None:
                self.timeStamp = np.linspace(
                    start=0,
                    stop=len(self.dataX) / self.sampRate,
                    num=len(self.dataX),
                    endpoint=False,
                    dtype=float,
                )
            # add subplots of pulse data, X and Y channel of the lock-in.
            pulse_ax = fig.add_subplot(gs[0, :])
            dataX_ax = fig.add_subplot(
                gs[1, :], sharex=pulse_ax
            )  # share x-axis with pulse_ax
            dataY_ax = fig.add_subplot(
                gs[2, :], sharex=pulse_ax, sharey=dataX_ax
            )  # share x-axis with dataX_ax
        else:
            raise ValueError("no plot")

        # plot time-series
        if showtimedomain:
            # TTL signal of pulses
            pulse_ax.plot(
                self.timeStamp, self.pulsedata, label="pulse sequence", c="tab:purple"
            )
            pulse_ax.set_ylabel("Voltage [V]")
            pulse_ax.set_xlim(self.timeStamp[0], self.timeStamp[-1])

            # X channel of LIA
            dataX_ax.plot(self.timeStamp, self.dataX, label="LIA X", c="tab:green")
            dataX_ax.set_ylabel("Voltage [V]")
            dataX_ax.set_xlim(self.timeStamp[0], self.timeStamp[-1])

            # Y channel of LIA
            dataY_ax.plot(self.timeStamp, self.dataY, label="LIA Y", c="tab:brown")
            dataY_ax.set_ylabel("Voltage [V]")
            dataY_ax.set_xlim(self.timeStamp[0], self.timeStamp[-1])
            dataY_ax.set_xlabel("time [s]")

            # adjust tick parameters
            pulse_ax.tick_params(
                axis="x",  # changes apply to the x-axis
                which="both",  # both major and minor ticks are affected
                bottom=True,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
            )  # labels along the bottom edge are off

            dataX_ax.tick_params(
                axis="x",  # changes apply to the x-axis
                which="both",  # both major and minor ticks are affected
                bottom=True,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
            )  # labels along the bottom edge are off

            # plot acquisition data, which is recorded in the self.selectPulse_list
            if showacqdata:
                for i in self.selectPulse_list[0:-1]:
                    # use orange to mark the acquisition data
                    dataX_ax.plot(
                        self.timeStamp[self.acq_arr[i, 0] : self.acq_arr[i, 1] + 1],
                        self.dataX[self.acq_arr[i, 0] : self.acq_arr[i, 1] + 1],
                        c="tab:orange",
                        alpha=0.9,
                    )
                    dataY_ax.plot(
                        self.timeStamp[self.acq_arr[i, 0] : self.acq_arr[i, 1] + 1],
                        self.dataY[self.acq_arr[i, 0] : self.acq_arr[i, 1] + 1],
                        c="tab:orange",
                        alpha=0.9,
                    )
                i = self.selectPulse_list[-1]
                dataX_ax.plot(
                    self.timeStamp[self.acq_arr[i, 0] : self.acq_arr[i, 1] + 1],
                    self.dataX[self.acq_arr[i, 0] : self.acq_arr[i, 1] + 1],
                    label="LIA X data for PSD",
                    c="tab:orange",
                    alpha=0.9,
                )
                dataY_ax.plot(
                    self.timeStamp[self.acq_arr[i, 0] : self.acq_arr[i, 1] + 1],
                    self.dataY[self.acq_arr[i, 0] : self.acq_arr[i, 1] + 1],
                    label="LIA Y data for PSD",
                    c="tab:orange",
                    alpha=0.9,
                )
            # adjust the location of the legend
            pulse_ax.legend(loc=legendloc)
            dataX_ax.legend(loc=legendloc)
            dataY_ax.legend(loc=legendloc)

        # plot frequency domain
        if showfreqdomain:
            # determine the reference frequency if it has not been specified
            # in case the x unit is ppm or ppb
            if referFreq is None:
                # if there has been a successful fitting
                if self.fitflag:
                    referFreq = self.popt[0]
                # otherwise, choose the frequency of the data point with the maximum amplitude
                else:
                    referFreq = self.freqs[np.argmax(self.avgPSD)]
            specx_dict = {
                "HZ": ["Hz", 1],
                "KHZ": ["kHz", 1e-3],
                "MHZ": ["MHz", 1e-6],
                "GHZ": ["GHz", 1e-9],
                "THZ": ["THz", 1e-12],
                "PHZ": ["PHz", 1e-15],
            }

            # Set x-axis unit
            if specxunit.upper() in specx_dict.keys():
                specxlabel = "Frequency"
                specxaxis = specx_dict[specxunit.upper()][1] * self.freqs
                specxunit = specx_dict[specxunit.upper()][0]
            elif specxunit == "ppm":
                specxaxis = 1e6 * (self.freqs / referFreq - 1.0)
                specxlabel = "Separation"
            elif specxunit == "ppb":
                specxaxis = 1e9 * (self.freqs / referFreq - 1.0)
                specxlabel = "Separation"
            else:
                raise ValueError(
                    "specxunit cannot find match. Check spelling, especially letTeR cAsIng. "
                )

            # Set the second x-axis
            if specxunit2 is None:
                pass
            elif specxunit2.upper() in specx_dict.keys():
                specxlabel2 = "Frequency"
            elif specxunit2 in ["ppm", "ppb"]:
                specxlabel2 = "Separation"
            else:
                raise ValueError(
                    "specxunit2 cannot find match. Check spelling, especially leTTeR cAsIng. "
                )

            if Mf is None and (ampunit == "Phi" or ampunit == "expPhi"):
                if hasattr(self, "SQD_Mf") and self.SQD_Mf is not None:
                    Mf = self.SQD_Mf
                else:
                    print(
                        f"[{self.GetSpectrum.__name__}] Warning: Mf is not specified. "
                        "Force spectrum 'ampunit' to be 'V'. "
                    )
                    ampunit = "V"

            if Rf is None and (ampunit == "Phi" or ampunit == "expPhi"):
                if hasattr(self, "SQD_Rf") and self.SQD_Rf is not None:
                    Rf = self.SQD_Rf
                else:
                    print(
                        f"[{self.GetSpectrum.__name__}] Warning: Rf is not specified. "
                        "Force spectrum 'ampunit' to be 'V'. "
                    )
                    ampunit = "V"

            spectype = spectype.upper()
            ampunit = ampunit.upper()
            # Set y-axis unit
            # if ampunit.upper() == 'V':
            #     ampfactor = 1.
            if spectype == "PSD".upper():
                self.spectrum = self.avgPSD
                ampfactor = 1.0
                specyunit = "$V^2/\\mathrm{Hz}$"
            elif spectype == "ASD".upper():
                self.spectrum = np.sqrt(self.avgPSD)
                ampfactor = 1.0
                specyunit = "$V/\\sqrt{\\mathrm{Hz}}$"
            elif spectype == "FluxPSD".upper() and ampunit != "expPhi".upper():
                ampfactor = (Mf / Rf) ** 2
                self.spectrum = self.avgPSD * ampfactor
                specyunit = "$\\Phi_{0}^{2}/\\mathrm{Hz}$"
            elif spectype == "FluxASD".upper() and ampunit != "expPhi".upper():
                ampfactor = Mf / Rf
                self.spectrum = np.sqrt(self.avgPSD) * ampfactor
                specyunit = "$\\Phi_{0}/\\sqrt{\\mathrm{Hz}}$"
            # elif ampunit in ['muV'.upper(),'microV'.upper(),'uV'.upper(),'muv'.upper(),'microv'.upper(),'uv'.upper()]:
            #     if spectype == 'PSD'.upper():e12
            #         self.spectrum = self.avgPSD * ampfactor
            #         specyunit = '$\mu V^2/\\mathrm{Hz}$'
            #     elif spectype == 'ASD'.upper():
            #         ampfactor = 1e6
            #         self.spectrum = np.sqrt(self.avgPSD) * ampfactor
            #         specyunit = '$\mu V/\\sqrt{\\mathrm{Hz}}$'
            #     else:
            #         raise ValueError('spectype wrong 313')
            # elif ampunit in ['muPhi'.upper(), 'microPhi'.upper(), 'uPhi'.upper(), 'muphi'.upper(), 'microphi'.upper(), 'uphi'.upper()]:
            #     if spectype == 'FluxPSD'.upper():
            #         ampfactor = (Mf / Rf * 1e6) ** 2
            #         self.spectrum = self.avgPSD * ampfactor
            #         specyunit = '$\mu \Phi_{0}^{2}/\\mathrm{Hz}$'
            #     elif spectype == 'FluxASD'.upper():
            #         ampfactor = Mf / Rf * 1e6
            #         self.spectrum = np.sqrt(self.avgPSD) * ampfactor
            #         specyunit = '$\mu \Phi_{0}/\\sqrt{\\mathrm{Hz}}$'
            #     else:
            #         raise ValueError('spectype wrong 314')
            # the values of y-axis is diplayed in scientific notation
            elif ampunit == "expPhi".upper():
                ampfactor = (Mf / Rf) ** 2
                if spectype == "FluxPSD".upper():
                    if amppow is None:
                        amppow = int(np.log10(np.amax(self.avgPSD * ampfactor)) - 1)
                    self.spectrum = 10 ** (-amppow) * ampfactor * self.avgPSD
                    # determine the unit of the spectrum
                    if amppow == 0:
                        specyunit = "$\\Phi_{0}^{2}/\\mathrm{Hz}$"
                    else:
                        specyunit = "$10^{%d}\\, \\Phi_{0}^{2}/\\mathrm{Hz}$" % (amppow)
                elif spectype == "FluxASD".upper():
                    if amppow is None:
                        self.spectrum = np.sqrt(self.avgPSD * ampfactor)
                        amppow = int(np.log10(np.amax(self.spectrum)) - 1)
                    self.spectrum = 10 ** (-amppow) * np.sqrt(self.avgPSD * ampfactor)
                    if amppow == 0:
                        specyunit = "$\\Phi_{0}/\\sqrt{\\mathrm{Hz}}$"
                    else:
                        specyunit = "$10^{%d}\\, \\Phi_{0}/\\sqrt{\\mathrm{Hz}}$" % (
                            amppow
                        )
                else:
                    raise ValueError("spectype wrong 315")
            # elif ampunit == '1':
            #     if spectype == 'PSD'.upper():
            #         ampfactor = 1
            #         self.spectrum = self.avgPSD * ampfactor ** 2
            #         specyunit = '$1/\\mathrm{Hz}$'
            #     elif spectype == 'ASD'.upper():
            #         ampfactor = 1
            #         self.spectrum = np.sqrt(self.avgPSD) * ampfactor
            #         specyunit = '$1/\sqrt{\\mathrm{Hz}}$'
            #     else:
            #         raise ValueError('spectype must be \'PSD\' or \'ASD\'. ')
            else:
                raise ValueError("cannot find ampunit")

            # assign 0 to amppow if it has not been specified
            if amppow is None:
                amppow = 0

            # show frequency domain
            if showfit and self.fitflag:
                spec_ax.plot(
                    specxaxis, self.spectrum, label=spectype, c="tab:blue", alpha=1
                )
                # if to show data points in scatter plot
                # spec_ax.scatter(specxaxis, self.spectrum, label=spectype, c='tab:blue', s=3)
            else:
                spec_ax.plot(specxaxis, self.spectrum, label=spectype, c="tab:blue")
                # if to show data points in scatter plot
                # spec_ax.scatter(specxaxis, self.spectrum, label=spectype, c='tab:blue', s=3)

            # plot standard deviation of part of the spectrum
            if showstd:
                spec_ax.plot(
                    specxaxis[stdstart:stdend],
                    self.spectrum[stdstart:stdend],
                    label=f"Standard deviation = {(ampfactor) * self.avgPSDstd:.3e} "
                    + specyunit,
                    c="tab:green",
                )

            # if to show fit curve and there has been a successful fitting
            if showfit and self.fitflag:
                # in case there are several fit curves, like dualGaussian
                for i in range(len(self.fitcurve) - 1):
                    spec_ax.plot(
                        specxaxis[self.fitrange[0] : self.fitrange[1]],
                        10 ** (-amppow) * ampfactor * self.fitcurve[i],
                        "--",
                        c="tab:red",
                        alpha=0.6,
                    )
                # plot the fit curve
                spec_ax.plot(
                    specxaxis[self.fitrange[0] : self.fitrange[1]],
                    10 ** (-amppow) * ampfactor * self.fitcurve[-1],
                    c="tab:red",
                    alpha=0.7,
                    label=self.fitreport,
                )
                if verbose:
                    print("self.fitreport ", self.fitreport)

            # Set x and y labels
            spec_ax.set_xlabel(specxlabel + " [" + specxunit + "] ")
            spec_ax.set_ylabel(spectype + " [" + specyunit + "] ")

            # if spec x or y limit was specified by the user
            # then follow the specfied limit
            if specxlim is not None and len(specxlim) == 2:
                spec_ax.set_xlim(specxlim[0], specxlim[1])
            if specylim is not None and len(specylim) == 2:
                spec_ax.set_ylim(specylim[0], specylim[1])

            # Set the second axis for horizontal axis
            if specxunit2 is not None:
                spec_ax2 = spec_ax.twiny()
                spec_ax2.set_xlabel(specxlabel2 + " [" + specxunit2 + "] ")
                spec_ax2.set_xlim(spec_ax.get_xlim())
                if specxunit == "ppm":
                    if specxunit2 == "MHz":
                        plotaxisfmt_partial = partial(
                            plotaxisfmt_ppm2MHz,
                            format_string=specx2format,
                            referfreq=referFreq,
                        )
                        # formatter = mticker.FuncFormatter(lambda x, pos: '{:.3f}'.format(1e-6*self.frequencies))
                        formatter = mticker.FuncFormatter(plotaxisfmt_partial)
                if specxunit == "Hz":
                    if specxunit2 == "ppm":
                        plotaxisfmt_partial = partial(
                            plotaxisfmt_Hz2ppm,
                            format_string=specx2format,
                            referfreq=referFreq,
                        )
                        # formatter = mticker.FuncFormatter(lambda x, pos: '{:.3f}'.format(1e-6*self.frequencies))
                        formatter = mticker.FuncFormatter(plotaxisfmt_partial)
                if specxunit == "MHz":
                    if specxunit2 == "ppm":
                        plotaxisfmt_partial = partial(
                            plotaxisfmt_MHz2ppm,
                            format_string=specx2format,
                            referfreq=referFreq,
                        )
                        # formatter = mticker.FuncFormatter(lambda x, pos: '{:.3f}'.format(1e-6*self.frequencies))
                        formatter = mticker.FuncFormatter(plotaxisfmt_partial)
                spec_ax2.xaxis.set_major_formatter(formatter)

            # Set y scale
            if specyscale == "log":
                spec_ax.set_yscale("log")

            spec_ax.grid(visible=True, axis="both")

            if showlegend:
                spec_ax.legend(loc=legendloc)

            # show residual plot
            if showfit and self.fitflag and showresidual:
                resi_ax.plot(
                    specxaxis,
                    self.spectrum - 10 ** (-amppow) * ampfactor * self.fitcurve[-1],
                    label="residual",
                    color="tab:purple",
                )

                resi_ax.set_xlim(spec_ax.get_xlim())

                specybottom, specytop = spec_ax.get_ylim()
                specyrange = abs(specybottom - specytop)
                resiybottom, resiytop = resi_ax.get_ylim()
                resiyrange = abs(resiybottom - resiytop)
                xxme = (
                    height_ratios[3]
                    / np.sum(height_ratios[0:3])
                    * specyrange
                    / resiyrange
                )
                resi_ax.set_ylim(resiybottom * xxme, resiytop * xxme)
                del (
                    specybottom,
                    specytop,
                    specyrange,
                    resiybottom,
                    resiytop,
                    resiyrange,
                    xxme,
                )

                resi_ax.set_xlabel(specxlabel + " [" + specxunit + "]")
                resi_ax.set_ylabel(spectype + " [" + specyunit + "]")
                resi_ax.legend(loc=legendloc)
                resi_ax.grid(visible=True, axis="both")
                spec_ax.tick_params(
                    axis="x",  # changes apply to the x-axis
                    which="both",  # both major and minor ticks are affected
                    bottom=True,  # ticks along the bottom edge are off
                    top=True,  # ticks along the top edge are off
                    labelbottom=False,
                )  # labels along the bottom edge are off
                spec_ax.set_xlabel("")

            if showchanstd and self.chanstd_flag:
                chanstd_ax.plot(
                    specxaxis,
                    100 * self.chanstd_avg,
                    label="channel StD",
                    color="tab:green",
                )  # '#ff4c7c'
                chanstd_ax.set_ylim(bottom=0)
                chanstd_ax.set_xlabel(specxlabel + " [" + specxunit + "]")
                chanstd_ax.set_ylabel("norm StD [%]")
                # chanstd_ax.legend(loc=legendloc)
                chanstd_ax.grid(visible=True, axis="both")
                spec_ax.tick_params(
                    axis="x",  # changes apply to the x-axis
                    which="both",  # both major and minor ticks are affected
                    bottom=True,  # ticks along the bottom edge are off
                    top=True,  # ticks along the top edge are off
                    labelbottom=False,
                )  # labels along the bottom edge are off
                spec_ax.set_xlabel("")
                if showresidual:
                    resi_ax.tick_params(
                        axis="x",  # changes apply to the x-axis
                        which="both",  # both major and minor ticks are affected
                        bottom=True,  # ticks along the bottom edge are off
                        top=True,  # ticks along the top edge are off
                        labelbottom=False,
                    )  # labels along the bottom edge are off
                    resi_ax.set_xlabel("")

        # Set the plot title
        titletext = self.exptype
        for singlefile in self.fileList:
            titletext += "\n" + singlefile  # include the data file name(s) in the title
        fig.suptitle(titletext, fontsize=12)

        plt.grid()

        # save plot
        if saveplt_opt:
            if save_msg is None or len(save_msg) == 0:
                if showtimedomain and showfreqdomain:
                    plt.savefig(
                        save_path + self.exptype + " time and frequency domain" + ".png"
                    )
                elif (not showtimedomain) and showfreqdomain:
                    plt.savefig(save_path + self.exptype + " frequency domain" + ".png")
            else:
                plt.savefig(save_path + save_msg + ".png")

        if tightlayout_opt:
            plt.tight_layout()

        if showplt_opt:
            plt.show()

        if saveSpecData_opt:
            np.savetxt(
                save_path + self.exptype + save_msg + ".txt",
                np.transpose([specxaxis, self.spectrum]),
                header=specxunit + "\n" + specyunit,
            )

        plt.clf()
        plt.close()
        del fig, gs

        if return_opt:
            return specxaxis, self.spectrum, specxunit, specyunit

    @classmethod
    def getStackedSpectra(
        cls, freqs, specStack, labels=None, specxunit=None, specyunit=None, offset=None
    ):
        print(plt.rcParams["font.size"])
        # Auto offset = mean of maxima
        if offset is None:
            offset = np.mean(np.amax(np.abs(specStack), axis=1))
            check(offset)
        cm = 1 / 2.54  # convert cm to inch
        figHeight = min(5 * cm, specStack.shape[0] * 1.2 * cm)
        figHeight = max(8 * cm, figHeight)
        fig = plt.figure(figsize=(8.5 * cm, figHeight), dpi=300)
        gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures
        # # fix the margins
        # fig.subplots_adjust(left=left, top=top, right=right,
        #                     bottom=bottom, wspace=wspace, hspace=hspace)
        # left=
        # bottom=
        # right=
        # top=
        # wspace=
        # hspace=
        ax = fig.add_subplot(gs[0, 0])
        for i, spec in enumerate(specStack):
            cls._plotPSD_(ax=ax, x=freqs, y=specStack[i] + i * offset)

        # ax00.set_xlabel("Frequency (" + specxunit + ")")
        # ax00.set_ylabel("PSD (" + specyunit + ")")

        # ax.set_xlabel("Frequency ("  + ")")
        # ax.set_ylabel("PSD ("  + ")")
        # ax00.grid()  # set gird color, linewidth and etc.
        # ax.legend(loc="upper right")
        # #############################################################################
        # # put a mark of script information on the figure
        # # Get the script name and path automatically
        # script_path = os.path.abspath(__file__)

        # # Add the annotation to the figure
        # plt.annotate(
        #     f"Generated by: {script_path}",
        #     xy=(0.02, 0.02),
        #     xycoords="figure fraction",
        #     fontsize=3,
        #     color="gray",
        # )
        # # #############################################################################

        plt.tight_layout()
        # plt.savefig('example figure - one-column.png', transparent=False)
        plt.show()

    # @classmethod
    def getPSDIntegrals(self, freq_center: Quantity, halfwidth: Quantity):
        """Integrate over [freq_center-halfwidth, freq_center+halfwidth] of the PSD and return the integral. Note that cls.PSD is a 2-D arry with shape (Npulses, acqPts). the returned array has shape (Npulses)"""
        #  freq_center= cls.freqs.to(.unit)
        mask = (self.freqs >= freq_center - halfwidth) & (
            self.freqs <= freq_center + halfwidth
        )
        return np.trapezoid(self.PSD[:, mask], self.freqs[mask], axis=1)

    def getFSIntegrals(self, freq_center: Quantity, halfwidth: Quantity):
        """Integrate over [freq_center-halfwidth, freq_center+halfwidth] of the FS and return the integral. Note that cls.PSD is a 2-D arry with shape (Npulses, acqPts). the returned array has shape (Npulses)"""
        #  freq_center= cls.freqs.to(.unit)
        mask = (self.freqs >= freq_center - halfwidth) & (
            self.freqs <= freq_center + halfwidth
        )
        return np.trapezoid(self.FS[:, mask], self.freqs[mask], axis=1)

    def getFS(
        self,
        winInfo={"name": "rectangle", "param": 100},
        zeroPadding=False,
        selectShots=[],
        showPlot=False,
        specxlim=None,
        verbose=False,
    ):
        """
        Get Fourier spectrum.

        Args:
            window_info : dict, optional Defaults to {"name": "rectangle", "param": 100}.
            selectShots : list, optional Defaults to [].
            showPlot : bool, optional Defaults to False.
            specxlim : type_, optional Defaults to None.
            verbose : bool, optional Defaults to False.
        """
        if not hasattr(self, "TS"):
            raise AttributeError(
                "'Signal' object has no attribute 'TS'. Maybe the stream is a no-pulse stream? "
            )

        if len(selectShots) == 0:
            select = range(self.TS.shape[0])
        else:
            select = selectShots
        self.freqs, self.FS, self.window_arr = LIA_FFT(
            TS=self.TS[select],
            sampRate=self.sampRate,
            demodFreq=self.demodFreq,
            # attenuation=attenuation,
            winInfo=winInfo,
            zeroPadding=zeroPadding,
            showWin=verbose,
            verbose=verbose,
        )

        self.PSD: Quantity = np.abs(self.FS) ** 2

        # if showPlot==True:
        # self.plotFS(showPlot=showPlot)

    def plotFS(self, xUnit=None, yUnit=None, savePlot=False, showPlot=False):
        print("i am the function plotFS from DataAnalysis.py")
        self.freqs = self.freqs.to(unit.Hz)
        fig = plt.figure(figsize=(1 * 8.5 / 2.54, 6.5 / 2.54), dpi=300)  #
        gs = gridspec.GridSpec(nrows=2, ncols=2)
        left = 0.173
        bottom = 0.16
        right = 0.9
        top = 0.895
        wspace = 0.613
        hspace = 0.475
        fig.subplots_adjust(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            wspace=wspace,
            hspace=hspace,
        )
        real_ax = fig.add_subplot(gs[0, 0])
        imag_ax = fig.add_subplot(gs[1, 0], sharex=real_ax)
        amp_ax = fig.add_subplot(gs[0, 1], sharex=real_ax)
        phase_ax = fig.add_subplot(gs[1, 1], sharex=real_ax)
        (real_line,) = real_ax.plot(
            self.freqs,
            self.FS.mean(axis=0).real,
            marker="o",
            label="Spectrum.real",
            color="tab:blue",
        )
        real_ax.set_ylabel("Amplitude (Arb. Units)")
        real_ax.grid(True)
        # real_ax.legend(loc="upper right")
        # real_ax.tick_params(axis='y', left=False, labelleft=False)
        real_ax.tick_params(axis="x", bottom=False, labelbottom=False)
        # bottom, top, left, right : bool : Whether to draw the respective ticks
        # labelbottom, labeltop, labelleft, labelright : bool : Whether to draw the respective tick labels.
        (imag_line,) = imag_ax.plot(
            self.freqs,
            self.FS.mean(axis=0).imag,
            label="Spectrum.imag",
            marker="o",
            color="tab:orange",
        )
        imag_ax.set_ylabel("Amplitude (Arb. Units)")
        imag_ax.set_xlabel("Frequency (Hz)")
        imag_ax.grid(True)
        # imag_ax.tick_params(axis='y', left=False, labelleft=False)
        # imag_ax.legend(loc="upper right")
        (amp_line,) = amp_ax.plot(
            self.freqs,
            (self.PSD.mean(axis=0)) ** 0.5,
            marker="o",
            label="ASD",
            color="tab:purple",
        )
        amp_ax.set_ylabel("Amplitude (Arb. Units)")
        amp_ax.grid(True)
        # amp_ax.legend(loc="upper right")
        # amp_ax.tick_params(axis='y', left=False, labelleft=False)
        amp_ax.tick_params(axis="x", bottom=False, labelbottom=False)

        (phase_line,) = phase_ax.plot(
            self.freqs,
            np.angle(self.FS.mean(axis=0), deg=True),
            marker="o",
            label="Phase of spectrum",
            color="tab:cyan",
        )
        phase_ax.set_ylabel("Phase ($\\degree$)")
        phase_ax.set_xlabel("Frequency (Hz)")
        phase_ax.grid(True)
        # phase_ax.legend(loc="upper right")
        # if specxlim is not None:
        #     real_ax.set_xlim(specxlim[0], specxlim[1])
        #     imag_ax.set_xlim(specxlim[0], specxlim[1])
        #     amp_ax.set_xlim(specxlim[0], specxlim[1])
        #     phase_ax.set_xlim(specxlim[0], specxlim[1])

        # real_ax.legend(
        #     handles=[real_line, imag_line, amp_line, phase_line],
        #     bbox_to_anchor=(-0.4, 1.0),
        #     loc="lower left",
        #     ncol=2
        # )
        real_ax.set_title("real")
        imag_ax.set_title("imag")
        amp_ax.set_title("ASD")
        phase_ax.set_title("Phase")
        # titletext = self.exptype  # 'All shots of '+
        # for singlefile in self.filelist:
        #     titletext += "\n" + singlefile
        # fig.suptitle(titletext, wrap=True)  # , fontsize=8
        # fig.tight_layout()
        if savePlot:
            plt.savefig()  # TODO complete this
        # if showPlot==True:
        # plt.show()
        plt.close("all")

    @classmethod
    def _plotPulseTrain_(
        cls,
    ):
        pass

    @classmethod
    def _plotTS_(cls, ax: Axes, x: Quantity, y: Quantity, color: str, label: str = ""):
        yUnit = (
            "Arb. Units"
            if y.unit.is_equivalent(unit.one)
            else y.unit.to_string("unicode")
        )
        # plot real array
        ax.plot(x, y, color=color, label=label, alpha=1)
        ax.set_xlabel(f"Time ({x.unit.to_string('unicode')})")
        ax.set_ylabel(f"Signal ({yUnit})")
        ax.grid()

    def plotTS(
        self,
        showPlot=True,
        verbose=False,
        watermark: str | None = None,
    ):
        cm = 1 / 2.54
        figsize = (
            (1.0 * 10.0 * cm),
            (6.5 * cm),
        )
        fig = plt.figure(figsize=figsize, dpi=300)

        # Initialize the grid structure for subplots
        gs = gridspec.GridSpec(nrows=2, ncols=1, width_ratios=[1], height_ratios=[1, 1])
        left = 0.13
        bottom = 0.173
        right = 0.973
        top = 0.94
        wspace = 0.472
        hspace = 0.516
        fig.subplots_adjust(
            left=left, top=top, right=right, bottom=bottom, wspace=wspace, hspace=hspace
        )
        dataX_ax = fig.add_subplot(gs[0, 0])
        dataY_ax = fig.add_subplot(gs[1, 0], sharex=dataX_ax)
        # spec_ax = fig.add_subplot(gs[:, 1])
        # if not hasattr(self, "acqDur"):
        #     self.acqDur = self.TS_raw.shape[0] / self.samp_rate
        timeStamp = np.linspace(
            0,
            (self.TS_raw.shape[0] / self.sampRate).to(unit.s),
            num=self.TS_raw.shape[0],
        )

        # plot raw time series
        self._plotTS_(
            ax=dataX_ax, x=timeStamp, y=self.TS_raw.real, color="tab:green", label="raw"
        )
        self._plotTS_(
            ax=dataY_ax, x=timeStamp, y=self.TS_raw.imag, color="tab:brown", label="raw"
        )

        # self._plotTS_(
        #     ax=dataX_ax, x=timeStamp, y=(self.TS.real * self.window_arr).reshape(-1), color="tab:orange", label="windowed"
        # )
        # self._plotTS_(
        #     ax=dataY_ax, x=timeStamp, y=(self.TS.imag * self.window_arr).reshape(-1), color="blue", label="windowed"
        # )

        # TODO add window here
        # plot PSD
        # self._plotPSD_(
        #     ax=spec_ax, x=self.freqs, y=self.PSD.mean(axis=0), color="tab:blue"
        # )

        titletext = ""
        if self.fileList is None or self.fileList == []:
            if self.filePath is None or type(self.filePath) != str:
                pass
            else:
                titletext = self.filePath
        else:
            for singlefile in self.fileList:
                titletext += (
                    "\n" + singlefile
                )  # include the data file name(s) in the title

        fig.suptitle(textwrap.fill(titletext, width=80), fontsize=4)

        dataX_ax.legend()
        dataY_ax.legend()

        fig.tight_layout()
        self.addWaterMark(watermark=watermark)
        if showPlot:
            plt.show()

    @classmethod
    def _plotFSreal_(cls, ax: Axes):
        pass

    @classmethod
    def _plotPSD_(cls, ax: Axes, x: Quantity, y: Quantity, color: str = "k"):
        freq_center = np.mean(x)
        yUnit = (
            "Arb. Units"
            if y.unit.is_equivalent(unit.one)
            else y.unit.to_string("unicode")
        )
        ax.plot(x - freq_center, y, color=color, alpha=1)
        ax.set_xlabel(
            f"Frequency - {freq_center.to_value(x.unit):g} ({x.unit.to_string('unicode')})"
        )
        ax.set_ylabel(f"PSD ({yUnit})")
        ax.grid()

    def plotSpectrum(
        self,
    ):

        pass

    @staticmethod
    def addWaterMark(watermark: str | None = None, fontsize: int | None = None):
        if watermark is None:
            return
        plt.annotate(
            watermark,
            xy=(0.02, 0.02),
            xycoords="figure fraction",
            fontsize=fontsize if fontsize is not None else 3,
            color="gray",
        )

    def plotTSandSpectrum(
        self,
        showPlot=True,
        verbose=False,
        watermark: str | None = None,
    ):
        """
        plot time series and
        """
        # Initialize the figure
        # figsize = (
        #     (1. * 8.5 * unit.cm).to_value(unit.imperial.inch),
        #     (4.5 * unit.cm).to_value(unit.imperial.inch),
        # )
        cm = 1 / 2.54
        figsize = (
            (1.0 * 10.0 * cm),
            (6.5 * cm),
        )
        fig = plt.figure(figsize=figsize, dpi=300)

        # Initialize the grid structure for subplots
        gs = gridspec.GridSpec(
            nrows=2, ncols=2, width_ratios=[1, 1], height_ratios=[1, 1]
        )
        left = 0.13
        bottom = 0.173
        right = 0.973
        top = 0.94
        wspace = 0.472
        hspace = 0.516
        fig.subplots_adjust(
            left=left, top=top, right=right, bottom=bottom, wspace=wspace, hspace=hspace
        )
        dataX_ax = fig.add_subplot(gs[0, 0])
        dataY_ax = fig.add_subplot(gs[1, 0], sharex=dataX_ax)
        spec_ax = fig.add_subplot(gs[:, 1])
        if not hasattr(self, "acqDur"):
            self.acqDur = self.TS_raw.shape[0] / self.samp_rate
        timeStamp = np.linspace(0, self.acqDur.to(unit.s), num=self.TS_raw.shape[0])

        # plot raw time series
        self._plotTS_(
            ax=dataX_ax, x=timeStamp, y=self.TS_raw.real, color="tab:green", label="raw"
        )
        self._plotTS_(
            ax=dataY_ax, x=timeStamp, y=self.TS_raw.imag, color="tab:brown", label="raw"
        )

        self._plotTS_(
            ax=dataX_ax,
            x=timeStamp,
            y=(self.TS.real * self.window_arr).reshape(-1),
            color="tab:orange",
            label="windowed",
        )
        self._plotTS_(
            ax=dataY_ax,
            x=timeStamp,
            y=(self.TS.imag * self.window_arr).reshape(-1),
            color="blue",
            label="windowed",
        )

        # TODO add window here
        # plot PSD
        self._plotPSD_(
            ax=spec_ax, x=self.freqs, y=self.PSD.mean(axis=0), color="tab:blue"
        )

        titletext = ""
        if self.fileList is None or self.fileList == []:
            if self.filePath is None or type(self.filePath) != str:
                pass
            else:
                titletext = self.filePath
        else:
            for singlefile in self.fileList:
                titletext += (
                    "\n" + singlefile
                )  # include the data file name(s) in the title

        fig.suptitle(textwrap.fill(titletext, width=80), fontsize=4)

        dataX_ax.legend()
        dataY_ax.legend()

        fig.tight_layout()
        self.addWaterMark(watermark=watermark)
        if showPlot:
            plt.show()

    def getMeanPower(self):
        assert hasattr(self, "freqs")
        assert hasattr(self, "PSD")
        self.meanPower = np.trapezoid(self.PSD.mean(axis=0), self.freqs)
        return self.meanPower

    def exportPSD(self, filePath: str):
        assert hasattr(self, "freqs") and hasattr(
            self, "PSD"
        ), "Run getFS() before exportSpectrum()"
        freqs_Hz = self.freqs.to(unit.Hz).value
        psd_mean = self.PSD.mean(axis=0)
        psd_unit_str = (
            "Arb. Units"
            if psd_mean.unit.is_equivalent(unit.one)
            else psd_mean.unit.to_string("unicode")
        )
        header = f"Frequency (Hz)\tMean PSD ({psd_unit_str})"
        np.savetxt(filePath, np.column_stack([freqs_Hz, psd_mean.value]), header=header)

    def saveToH5(
        self,
        saveDir,
        fileName=None,
        filePath: str = None,
        h5_group_name: str | None = None,
        verbose: bool = False,
    ):
        """
        Save this object to an HDF5 file.
        The file structure:
            OnePulse.h5
                ...:group  # other groups, like SQUID info group
                LIA:group  # level-1 group. This class will be saved here
                    demods:group  # level-2 groups for demodulators. There could be multiple demodulatos
                        demodID:group  # demodulator ID
                            auxin0:group  # aux channel 0
                                [array]:value  # array value
                                [unit]:value  # unit of the array
                            demodfreq:group  # demodulator frequency
                                [array]:value
                                [unit]:value
                            filter_TC:group  # filter time constant
                                [array]:value
                                [unit]:value
                            filter_order:group  # filter order
                                [array]:value
                                [unit]:value
                            samplex:group  # LIA channel ouput (real)
                                [array]:value
                                [unit]:value
                            sampley:group  # LIA channel ouput (imag)
                                [array]:value
                                [unit]:value
                            samprate:group  # sampling rate
                                [array]:value
                                [unit]:value
        """
        logPrefix = f"[{self.__class__.__name__}.{self.saveToH5.__name__}]"

        # use device ID as the level-1 group name
        if h5_group_name is None:
            h5_group_name = self.deviceID

        # check if .h5 is in the file name
        if not str(filePath).endswith(".h5"):
            raise ValueError("filePath should end with filename and '.h5'")
        # filePath = filePath if str(filePath).endswith(".h5") else f"{filePath}.h5"
        # write to disk
        with h5py.File(filePath, "w") as h5f:
            demods_group = h5f.create_group(f"{h5_group_name}/demods/{self.demodIndex}")
            self.saveToH5group(group=demods_group, verbose=verbose)

        if verbose:
            print(logPrefix, f"Saved to {filePath}")

    def saveToH5group(
        self,
        group: h5py.Group,
        verbose: bool = False,
    ):
        """
        Save lock-in amplifier data to the HDF5 group.
        The file structure:
            OnePulse.h5
                ...:group  # other groups, like SQUID info group
                LIA:group  # This class will be saved here
                    demods:group  # demodulators. There could be multiple demodulatos
                        demodID:group  # demodulator ID
                            auxin0:group  # aux channel 0
                                [array]:value  # array value
                                [unit]:value  # unit of the array
                            demodfreq:group  # demodulator frequency
                                [array]:value
                                [unit]:value
                            filter_TC:group  # filter time constant
                                [array]:value
                                [unit]:value
                            filter_order:group  # filter order
                                [array]:value
                                [unit]:value
                            samplex:group  # LIA channel ouput (real)
                                [array]:value
                                [unit]:value
                            sampley:group  # LIA channel ouput (imag)
                                [array]:value
                                [unit]:value
                            samprate:group  # sampling rate
                                [array]:value
                                [unit]:value
        """
        logPrefix = f"[{self.__class__.__name__}.{self.saveToH5group.__name__}]"

        # Save name if exists
        if hasattr(self, "name"):
            group.create_dataset(
                "name", data=["nameless" if self.name is None else self.name]
            )
        # print(logPrefix, "sampleX" )
        Signal.saveQuantityToH5group(
            group, "sampleX", self.TS_raw.real, verbose=verbose
        )
        # print(logPrefix, "sampleY" )
        Signal.saveQuantityToH5group(
            group, "sampleY", self.TS_raw.imag, verbose=verbose
        )
        for attr in ["demodFreq", "sampRate", "acqDur", "acqPts"]:
            print(logPrefix, attr)
            Signal.saveQuantityToH5group(
                group, attr, getattr(self, attr), verbose=verbose
            )
        # saveQuantityToH5group(group, "filter_TC", getattr(self, "filter_TC"))
        # saveQuantityToH5group(group, "filter_order", getattr(self, "filter_order"))
        # saveQuantityToH5group(group, "auxin0", getattr(self, "auxin0"))
        # saveQuantityToH5group(group, "auxin1", getattr(self, "auxin1"))

    @classmethod
    def loadFromH5group(cls, group, deviceID="dev4434", demodIndex=0):
        """
        Load Signal fields written by saveToH5group.

        Expects subgroups 'demodFreq', 'sampRate', 'sampleX', 'sampleY' from
        saveQuantityToH5group, optional dataset 'name'.
        """
        logPrefix = f"[{cls.__name__}.{cls.loadFromH5group.__name__}]"
        obj = cls(
            name="LIA signal",
            device=deviceID,
            deviceID=deviceID,
            demodIndex=demodIndex,
            filePath=None,
            fileList=[],
        )

        if "name" in group:
            raw = group["name"][()]
            arr = np.asarray(raw)
            name_val = arr.flat[0]
            if isinstance(name_val, bytes):
                name_val = name_val.decode("utf-8")
            else:
                name_val = str(name_val)
            obj.name = None if name_val == "nameless" else name_val

        for key in ("demodFreq", "sampRate", "sampleX", "sampleY"):
            if key not in group:
                raise KeyError(f"{logPrefix} missing subgroup {key!r} in HDF5 group")

        obj.demodFreq = cls.loadAttrFromH5group(group["demodFreq"])
        obj.sampRate = cls.loadAttrFromH5group(group["sampRate"])
        sampleX = cls.loadAttrFromH5group(group["sampleX"])
        sampleY = cls.loadAttrFromH5group(group["sampleY"])
        # obj.TS_raw = Quantity(
        #     np.asarray(sample_x.value) + 1j * np.asarray(sample_y.value),
        #     sample_x.unit,
        # )
        obj.TS_raw = np.asanyarray(sampleX + 1j * sampleY)
        try:
            obj.acqPts = cls.loadAttrFromH5group(group["acqPts"])
        except Exception:
            print("missing acqPts")
        if obj.TS_raw.ndim != 1:
            print(logPrefix, "WARNING: obj.TS_raw.ndim != 1")

        return obj

    @classmethod
    def loadFromH5(cls, filePath, deviceID, demodIndex):
        # prefix for log messages
        logPrefix = f"[{cls.__name__}.{cls.loadFromH5.__name__}]"

        # add .h5 extension if needed
        path = filePath if str(filePath).endswith(".h5") else f"{filePath}.h5"

        # if no such file
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{logPrefix} file not found: {path}")
        grp_key = f"{deviceID}/demods/{demodIndex}"

        # load from a h5 group
        with h5py.File(path, "r") as h5f:
            if grp_key not in h5f:
                raise KeyError(f"{logPrefix} missing HDF5 group {grp_key!r} in {path}")
            obj = cls.loadFromH5group(
                h5f[grp_key], deviceID=deviceID, demodIndex=demodIndex
            )
        obj.filePath = path
        obj.fileList = [obj.filePath]
        obj.deviceID = deviceID
        obj.demodIndex = demodIndex
        return obj


# -------------------------------------------------------------------
# Dataclass to store an NMR signal instance and its parameters
# -------------------------------------------------------------------
@dataclass
class SignalEntry:
    signal: Signal  # NMR signal
    params: dict  # Additional parameters


class Signals(NMRio):
    name: str | None = None
    pool: list[SignalEntry] | None = None

    def __init__(self, name: str | None = None, pool: list[SignalEntry] | None = None):
        """ "
        Initialize a Signals
        """
        self.name = name
        self.pool = pool

    def sortByAttrs(self, *attrs, reverse: bool = False, verbose=False):
        logPrefix = f"[{self.__class__.__name__}.{self.sortByAttrs.__name__}]"
        if not attrs:
            raise ValueError("At least one attribute must be specified")

        # # check if attrs are valid
        # for attr in attrs:
        #     for signal in self.pool.values():
        #         if not hasattr(signal, attr):
        #             raise AttributeError(f"Signal has no attribute '{attr}'")
        # sort the pool by attributes on the signal first, then by stored params
        def _sort_key(entry: SignalEntry):
            key_values = []
            for attr in attrs:
                if hasattr(entry.signal, attr):
                    key_values.append(getattr(entry.signal, attr))
                elif attr in entry.params:
                    key_values.append(entry.params[attr])
                else:
                    raise AttributeError(
                        f"SignalEntry has no attribute or param named '{attr}'"
                    )
            return tuple(key_values)

        self.pool = sorted(self.pool, key=_sort_key, reverse=reverse)

        if verbose:
            for entry in self.pool:
                print("")
                print(entry)

    def sortByCreationTime(
        self,
        verbose: bool = False,
    ):
        logPrefix = f"[{self.__class__.__name__}.{self.sortByCreationTime.__name__}]"
        if len(self.pool) == 0:
            raise ValueError("'Signals.pool' is empty.")

        # First pass: compute missing creation times.
        for i, entry in enumerate(self.pool):
            if not hasattr(entry, "signal"):
                raise AttributeError(f"Entry {i} does not have a 'signal' attribute.")
            if entry.signal.creation_time is None:
                entry.signal.getCreationTimes(verbose=False)

        # Second pass: ensure all creation times are available.
        for i, entry in enumerate(self.pool):
            if entry.signal.creation_time is None:
                raise ValueError(
                    f"Creation time could not be obtained for signal at index {i}."
                )

        self.pool = sorted(self.pool, key=lambda entry: entry.signal.creation_time)
        if verbose:
            print(f"{logPrefix} pool sorted by creation time:")
            for entry in self.pool:
                print(f"  {entry.signal.creation_time}: {entry.signal.name}")

    def getFS(
        self,
        winInfo={"name": "rectangle", "param": 100},
        zeroPadding=False,
        selectShots=[],
        showPlot=False,
        specxlim=None,
        verbose=False,
    ):
        """
        Get Fourier spectra for all signals in this Signals pool.

        Args:
            winInfo : dict, optional Defaults to {"name": "rectangle", "param": 100}.
            zeroPadding : bool, optional Defaults to False.
            selectShots : list, optional Defaults to [].
            showPlot : bool, optional Defaults to False.
            specxlim : type_, optional Defaults to None.
            verbose : bool, optional Defaults to False.
        """
        logPrefix = f"[{self.__class__.__name__}.{self.getFS.__name__}]"
        if len(self.pool) == 0:
            raise ValueError("'Signals.pool' is empty.")

        for i, entry in enumerate(self.pool):
            if not hasattr(entry, "signal"):
                raise AttributeError(f"Entry {i} does not have a 'signal' attribute.")
            if not hasattr(entry.signal, "getFS"):
                raise AttributeError(f"Signal in entry {i} does not implement 'getFS'.")

            entry.signal.getFS(
                winInfo=winInfo,
                zeroPadding=zeroPadding,
                selectShots=selectShots,
                showPlot=showPlot,
                specxlim=specxlim,
                verbose=verbose,
            )

    def getMeanPower(self):
        logPrefix = f"[{self.__class__.__name__}.{self.getMeanPower.__name__}]"
        if len(self.pool) == 0:
            raise ValueError("'Signals.pool' is empty.")

        for i, entry in enumerate(self.pool):
            if not hasattr(entry, "signal"):
                raise AttributeError(f"Entry {i} does not have a 'signal' attribute.")
            entry.signal.getMeanPower()

        return self.pool[0].signal.meanPower

    def _stackSpectra(
        self,
        ax: Axes,
        # labels: list = None,
        xUnit: CompositeUnit,
        yUnit: CompositeUnit,
        offset: Quantity,
        linewidth: float = 0.5,
    ):
        for i, entry in enumerate(self.pool):
            ax.plot(
                entry.signal.freqs - entry.signal.demodFreq,
                entry.signal.PSD.mean(axis=0) + i * offset,
                # label=(labels[i] if labels is not None else None),
                color="k",
                alpha=1,
                linestyle="-",
                linewidth=linewidth,
            )

        ax.set_xlabel(
            f"Frequency - {entry.signal.demodFreq.to_value(xUnit)} ({str(xUnit)})"
        )
        ylabel = str(yUnit) if not yUnit.is_equivalent(unit.one) else "Arb. Units"
        ax.set_ylabel("PSD" + " (" + ylabel + ")")

        ax.grid()  # set gird color, linewidth and etc.
        # if labels is not None:
        #     ax.legend(loc="upper right")

    def getStackedSpectra(
        self,
        # labels: list = None,
        xUnit: CompositeUnit = None,
        yUnit: CompositeUnit = None,
        offset=None,
        linewidth=0.5,  # line width in points
    ):
        logPrefix = f"[{self.__class__.__name__}.{self.getStackedSpectra.__name__}]"

        if xUnit is None:
            xUnit = self.pool[0].signal.freqs.unit
        if yUnit is None:
            yUnit = self.pool[0].signal.PSD.unit

        spectra_amax = []
        for i, entry in enumerate(self.pool):
            entry.signal.freqs = entry.signal.freqs.to(xUnit)
            entry.signal.PSD = entry.signal.PSD.to(yUnit)
            spectra_amax.append(np.amax(entry.signal.PSD))

        if offset is None:
            offset = np.mean(spectra_amax)

        cm = 1 / 2.54  # convert cm to inch
        figHeight = min(10 * cm, len(self.pool) * 1.2 * cm)
        figHeight = max(6 * cm, figHeight)
        check(figHeight)
        fig = plt.figure(figsize=(8.5 * cm, figHeight), dpi=300)

        gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures
        # # fix the margins
        # fig.subplots_adjust(left=left, top=top, right=right,
        #                     bottom=bottom, wspace=wspace, hspace=hspace)
        # left=
        # bottom=
        # right=
        # top=
        # wspace=
        # hspace=

        ax = fig.add_subplot(gs[0, 0])

        self._stackSpectra(
            ax=ax,
            # labels= labels,
            xUnit=xUnit,
            yUnit=yUnit,
            offset=offset,
            linewidth=linewidth,
        )
        #############################################################################
        # put a mark of script information on the figure
        # Get the script name and path automatically
        script_path = os.path.abspath(__file__)

        # Add the annotation to the figure
        plt.annotate(
            f"Generated by: {script_path} [{self.__class__.__name__}.{self.getStackedSpectra.__name__}] ",
            xy=(0.02, 0.02),
            xycoords="figure fraction",
            fontsize=3,
            color="gray",
        )
        # #############################################################################

        plt.tight_layout()
        # plt.savefig('example figure - one-column.png', transparent=False)
        plt.show()


# TODO
# pahse correction
# T1 measurement analysis
# T2 analsysis
# are we really robust with different units?
