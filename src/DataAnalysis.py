##################################################
# for NMR data processing
##################################################

import os


from src.utils import (
    check,
    record_runtime_YorN,
    Lorentzian,
    estimateLorzfit,
    dualLorentzian,
    estimatedualLorzfit,
    tribLorentzian,
    estimatetribLorzfit,
    Gaussian,
    estimateGaussfit,
    dualGaussian,
    estimatedualGaussFit,
    ExpCos,
    estimateExpCos,
    ExpCosiSin,
    ExpCosiSinResidual,
    estimateExpCosiSin,
    dualExpCos,
    estimatedualExpCos,
    stdLIAFFT,
    stdPSD,
    stdLIAPSD,
    PolyEven,
    plotaxisfmt_ppm2MHz,
    plotaxisfmt_Hz2ppm,
    plotaxisfmt_MHz2ppm,
    MovAvgByStep,
    clear_lines,
)

import numpy as np

# plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # for creating subplots
import matplotlib.ticker as mticker

# curve fitting (including calculating uncertainties)
from scipy.optimize import curve_fit
import scipy.stats.distributions
from uncertainties import ufloat

# for interpolation
from scipy import interpolate

from scipy.stats import norm, chi2

from scipy.signal import ShortTimeFFT, savgol_filter

# importing and processing hdf5 files

# monitor run time
import time

from functools import partial

from BaselineRemoval import BaselineRemoval

np.random.seed(None)  # WARNING!
# BaselineRemoval will effect the randomness of the script.
# Better to set the random seed to None so as to restore the randomness

from astropy.time import Time

from datetime import datetime, timezone


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


# Signal class containing 2-channel data, data-processing methods and processed data
# @dataclass
class Signal:
    def __init__(
        self,
        name="Dual Channel Signal",
        file=None,
        filelist=None,
        verbose=False,
    ):
        """
        Initialize the (class) LIAsignal object

        Parameters
        ----------
        self : (class) LIAsignal
            Signal from the lock-in amplifier and also its processed data
        name : string
            name of the signal. Defaults to 'LIA signal'.
        file : string
            path + filename of the data file.
            e.g. 'C:/Users/Rackyboy/Documents/SQUID data/stream_008/stream_00000.h5'
            it is suggested to use absolute path
        filelist : array_like
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
        self.file = file
        self.fitflag = False  # initialize fit flag to False (no fitting or fitting is unsuccessful)
        self.exptype = "Experiment type not specified"  # experiment type.
        self.filterstatus = "on"
        self.attenuation = 0
        self.DTRC_TC = 0
        self.DTRC_order = 0

        # store date file list
        # LoadStream will use self.filelist only, not self.file
        if filelist is None:
            if file is None:
                self.fileList = []  # initialize an empty filelist
            else:
                self.fileList = [file]  # make filelist include file
        else:
            self.fileList = filelist  # store filelist
        if len(self.fileList) > 0:
            self.SortFiles(verbose)

        self.timeStamp_s = None  # initialize timestamp for the data

        self.avgFIDsq = None  # initialize average FID square to 0.
        # It is not recommended to use AvgFIDsq now, because the time-series is not necessarily FID.
        # Instead, it is recommended to use AvgTSsq
        self.avgTSsq = None  # initialize time-series mean square to 0.
        self.freq_resol_Hz = None
        self.dataLen = None
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

    def SortFiles(self, verbose: bool = False):
        if self.fileList is None or len(self.fileList) == 0:
            return
        self.creation_times = []

        # get stream start times
        for file in self.fileList:
            # Get the file creation time
            creation_time = os.path.getctime(file)
            # Convert to UTC time
            creation_time_utc = datetime.fromtimestamp(creation_time, tz=timezone.utc)
            # print(creation_time_utc)
            # Convert to a format readable by astropy
            astropy_time = Time(creation_time_utc)
            # append astro-time to the list
            self.creation_times.append(astropy_time)

        # print filelist before sorting
        if verbose:
            print(
                f"[{self.SortFiles.__name__}] filelist and creation_times before sorting:"
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
            print(f"[{self.SortFiles.__name__}] filelist after sorting:")
            print((self.fileList, self.creation_times))

    # def LoadStream(
    #     self,
    #     skip_pulsedata=False,
    #     skip_timestamp=False,
    #     skiprows=3,
    #     skip_dataXY=False,
    #     verbose=False,
    # ):
    #     """
    #     Load lock-in amplifier data from single / multiple file(s).

    #     Parameters
    #     ----------
    #     self : (class) LIAsignal
    #         Signal from lock-in amplifier and also the processed data

    #     Keadevice : (class) Kea
    #         in KeaControl

    #     SQDsensor : (class) SQUID
    #         in DataAnalysis

    #     Expinfo : (class) Experiment
    #         in LIAControl

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
    #     Null
    #     """

    #     self.dataX = []
    #     self.dataY = []
    #     self.pulsedata = []

    #     def loadstream_DAQ(dataFile):
    #         # with h5py.File(singlefile, 'r', driver='core') as dataFile:
    #         self.demodfreq = dataFile[
    #             f"{self.device_id:s}/demods/{self.demod_index:d}/dmodfreq"
    #         ][0]
    #         self.samprate = dataFile[
    #             f"{self.device_id:s}/demods/{self.demod_index:d}/samprate"
    #         ][0]
    #         self.filter_TC = dataFile[
    #             f"{self.device_id:s}/demods/{self.demod_index:d}/filter_TC"
    #         ][0]
    #         self.filter_order = dataFile[
    #             f"{self.device_id:s}/demods/{self.demod_index:d}/filter_order"
    #         ][0]
    #         if not skip_dataXY:
    #             self.dataX += list(
    #                 dataFile[f"{self.device_id:s}/demods/{self.demod_index:d}/samplex"]
    #             )
    #             self.dataY += list(
    #                 dataFile[f"{self.device_id:s}/demods/{self.demod_index:d}/sampley"]
    #             )
    #             # check(sys.getsizeof(self.dataX))
    #             # check(sys.getsizeof(self.dataY))

    #         if not skip_pulsedata:
    #             self.pulsedata += list(
    #                 dataFile[f"{self.device_id:s}/demods/{self.demod_index:d}/auxin0"]
    #             )

    #         if not skip_timestamp:
    #             self.timestamp = np.linspace(
    #                 start=0,
    #                 stop=len(self.dataX) / self.samprate,
    #                 num=len(self.dataX),
    #                 endpoint=False,
    #                 dtype=float,
    #             )

    #         if "Expinfo" in dataFile.keys() and Expinfo is not None:
    #             Expinfo.name = dataFile[f"Expinfo/name"][0]
    #             Expinfo.dateandtime = dataFile[f"Expinfo/dateandtime"][0]
    #             Expinfo.exptype = dataFile[f"Expinfo/exptype"][0]
    #             self.exptype = dataFile[f"Expinfo/exptype"][0]

    #         if "Keadevice" in dataFile.keys() and Keadevice is not None:
    #             Keadevice.name = dataFile[f"Keadevice/name"][0]
    #             Keadevice.B1freq = dataFile[f"Keadevice/B1freq"][0]
    #             Keadevice.pulseamp = dataFile[f"Keadevice/pulseamp"][0]
    #             Keadevice.pulsedur = dataFile[f"Keadevice/pulsedur"][0]

    #         if "SQDsensor" in dataFile.keys():
    #             if SQDsensor is not None:
    #                 SQDsensor.name = dataFile[f"SQDsensor/name"][0]
    #                 SQDsensor.Mf = dataFile[f"SQDsensor/Mf"][0]
    #                 SQDsensor.Rf = dataFile[f"SQDsensor/Rf"][0]
    #                 SQDsensor.attenuation = dataFile[f"SQDsensor/attenuation"][0]
    #             self.y = dataFile[f"SQDsensor/Mf"][0]
    #             self.SQD_Rf = dataFile[f"SQDsensor/Rf"][0]
    #             self.attenuation = dataFile[f"SQDsensor/attenuation"][0]

    #     def loadstream_UI(dataFile):
    #         self.dataX += list(
    #             dataFile[
    #                 f"000/{self.device_id:s}/demods/{self.demod_index:d}/sample/x"
    #             ][:]
    #         )
    #         self.dataY += list(
    #             dataFile[
    #                 f"000/{self.device_id:s}/demods/{self.demod_index:d}/sample/y"
    #             ][:]
    #         )
    #         # self.pulsedata += list(dataFile[f'000/{self.device_id:s}/demods/{self.demod_index:d}/sample/auxin0'][:])

    #         self.demodfreq = dataFile[
    #             f"000/{self.device_id:s}/demods/{self.demod_index:d}/sample/frequency"
    #         ][0]
    #         self.samprate = dataFile[
    #             f"000/{self.device_id:s}/demods/{self.demod_index:d}/rate/value"
    #         ][0]
    #         self.filter_TC = dataFile[
    #             f"000/{self.device_id:s}/demods/{self.demod_index:d}/timeconstant/value"
    #         ][0]
    #         self.filter_order = dataFile[
    #             f"000/{self.device_id:s}/demods/{self.demod_index:d}/order/value"
    #         ][0]
    #         if not skip_timestamp:
    #             self.timestamp = np.linspace(
    #                 start=0,
    #                 stop=len(self.dataX) / self.samprate,
    #                 num=len(self.dataX),
    #                 endpoint=False,
    #                 dtype=float,
    #             )

    #     def loadstream_DAQ_continuous(dataFile):
    #         self.dataX += list(
    #             dataFile[
    #                 f"000/{self.device_id:s}/demods/{self.demod_index:d}/sample.x/value"
    #             ][:]
    #         )
    #         self.dataY += list(
    #             dataFile[
    #                 f"000/{self.device_id:s}/demods/{self.demod_index:d}/sample.y/value"
    #             ][:]
    #         )
    #         self.pulsedata += list(
    #             dataFile[
    #                 f"000/{self.device_id:s}/demods/{self.demod_index:d}/sample.auxin0/value"
    #             ][:]
    #         )

    #     def loadstream_NMRKineticSimu(dataFile):
    #         self.testtimestamp = dataFile["NMRKineticSimu/demods/0/timestamp"]
    #         # check(self.testtimestamp)
    #         # check(self.testtimestamp[0])
    #         self.demodfreq = dataFile["NMRKineticSimu/demods/0/dmodfreq"][0]
    #         self.samprate = dataFile["NMRKineticSimu/demods/0/samprate"][0]
    #         self.filter_TC = dataFile["NMRKineticSimu/demods/0/filter_TC"][0]
    #         self.filter_order = dataFile["NMRKineticSimu/demods/0/filter_order"][0]
    #         # check(self.demodfreq)
    #         # check(self.samprate)
    #         self.dataX += list(dataFile["NMRKineticSimu/demods/0/samplex"])
    #         self.dataY += list(dataFile["NMRKineticSimu/demods/0/sampley"])

    #     def loadstream_csv():
    #         self.exptype = "Pulsed-NMR"
    #         data = np.loadtxt(singlefile, delimiter=",")
    #         self.timestamp = 1e-6 * np.array(data[:, 0], dtype=np.float64)
    #         self.dataX += list(1e-6 * data[:, 1])
    #         self.dataY += list(1e-6 * data[:, 2])
    #         del data

    #         self.demodfreq = 0.0
    #         self.samprate = 1.0 / abs(self.timestamp[1] - self.timestamp[0])
    #         self.filterstatus = "off"
    #         self.filter_TC = 0.0
    #         self.filter_order = 0.0
    #         self.attenuation = 0.0
    #         self.acq_arr = np.array(
    #             [[0], [len(self.dataY)]], dtype=np.int64
    #         ).transpose()

    #     def loadstream_txt():
    #         self.exptype = "Pulsed-NMR"
    #         # TODO add the function to count text lines in the data file
    #         data = np.loadtxt(singlefile, skiprows=skiprows)
    #         self.dataX = data[:, 0]
    #         self.dataY = data[:, 1]
    #         del data
    #         self.pulsedata = np.zeros(len(self.dataY))

    #         self.demodfreq = 0.0
    #         # self.samprate = 1. / abs(self.timestamp[1]-self.timestamp[0])
    #         self.filterstatus = "off"
    #         self.filter_TC = 0.0
    #         self.filter_order = 0.0
    #         self.attenuation = 0.0
    #         self.acq_arr = np.array(
    #             [[0], [len(self.dataY)]], dtype=np.int64
    #         ).transpose()

    #     loadstream_methods = {
    #         "DAQ_record": loadstream_DAQ,
    #         "UI": loadstream_UI,
    #         "DAQ_continuous": loadstream_DAQ_continuous,
    #         "NMRKineticSimu": loadstream_NMRKineticSimu,
    #         "Kea_csv": loadstream_csv,
    #         "TecMag_txt": loadstream_txt,
    #     }  #

    #     for singlefile in self.filelist:
    #         if verbose:
    #             print("Loading data from " + singlefile)
    #         if singlefile[-3:] == ".h5":
    #             with h5py.File(
    #                 singlefile, "r", driver="core"
    #             ) as dataFile:  # h5py loading method
    #                 # check recording method
    #                 if verbose:
    #                     check(dataFile.keys())
    #                 # check(dataFile)
    #                 # check(sys.getsizeof(dataFile))
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
    #                 if verbose:
    #                     print("recordmethod: ", recordmethod)
    #                 # load stream
    #                 loadstream_methods[recordmethod](dataFile)
    #                 if verbose:
    #                     print("Loading Finished")
    #         elif singlefile[-4:] == ".csv":
    #             recordmethod = "Kea_csv"
    #             if verbose:
    #                 print("recordmethod: ", recordmethod)
    #             loadstream_methods[recordmethod]()
    #         elif singlefile[-4:] == ".tnt":
    #             recordmethod = "TecMag_tnt"
    #             # tnt file is not supported for now
    #         elif singlefile[-4:] == ".txt":
    #             recordmethod = "TecMag_txt"
    #             if verbose:
    #                 print("Reading txt file")
    #             # loadstream_txt()
    #             loadstream_methods[recordmethod]()
    #         else:
    #             raise ValueError("file type not in .h5, .csv, .tnt nor .txt")

    #     self.dataX = np.array(self.dataX, dtype=np.float64).flatten()
    #     self.dataY = np.array(self.dataY, dtype=np.float64).flatten()
    #     self.pulsedata = np.array(self.pulsedata, dtype=np.float64).flatten()
    #     self.chanstd_flag = False

    #     if self.attenuation is None:
    #         print(
    #             "Please remember to specify attenuation of the signal before it went to LIA. "
    #         )

    #     # self.GetMeas
    #     # if verbose:
    #     #     print('after flattening')
    #     #     print('self.dataX.shape', self.dataX.shape)
    #     #     print('self.dataY.shape', self.dataY.shape)
    #     #     print('self.pulsedata.shape', self.pulsedata.shape)

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
        assert self.samprate is not None
        assert plotrate is None or plotrate > 0
        lenofdata = len(self.dataX)
        if plotrate is None:
            plotrate = self.samprate
            plotintv = 1
            if lenofdata > maxlen:
                clear_lines()
                print(
                    f"Warning! length of data = {len(self.dataX):e} > max length {maxlen:e}. "
                    "This may result in long runtime. "
                )
        elif plotrate > self.samprate:
            clear_lines()
            print(
                "WARNING: plotrate > self.samprate. plotrate will be reassigned the value of self.samprate"
            )
            plotrate = self.samprate
            plotintv = 1
        elif plotrate <= self.samprate and plotrate > 0:
            plotintv = max(int(1.0 * self.samprate / plotrate), 1)
        else:
            raise ValueError("plotrate < 0. ")

        # plt.rc('font', size=6)  # font size for all figures
        # plt.rcParams["font.family"] = "serif"
        # plt.rcParams["font.serif"] = ["Times New Roman"]
        # plt.rcParams["font.family"] = "Times New Roman"
        # plt.rcParams["mathtext.fontset"] = 'cm'  # 'dejavuserif'
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
                self.timeStamp_s[0:-1:plotintv],
                self.pulsedata[0:-1:plotintv],
                label="pulse sequence",
                c="tab:purple",
            )
            pulse_ax.set_ylabel("Voltage [V]")
            pulse_ax.legend(loc="upper right")  # adjust the location of the legend

        def plotTS(ax, data, label, color, scatter):
            ax.plot(
                self.timeStamp_s[0:-1:plotintv],
                data[0:-1:plotintv],
                label=label,
                c=color,
            )
            if scatter is not None and len(scatter) > 0:
                for idx in scatter[0:]:
                    ax.scatter(
                        self.timeStamp_s[0:-1:plotintv][idx // plotintv],
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

        pulse_ax.set_xlim(self.timeStamp_s[0], self.timeStamp_s[-1])
        dataX_ax.set_xlim(self.timeStamp_s[0], self.timeStamp_s[-1])
        dataY_ax.set_xlim(self.timeStamp_s[0], self.timeStamp_s[-1])
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
        for singlefile in self.fileList:
            titletext += "\n" + singlefile  # include the data file name(s) in the title
        fig.suptitle(titletext, wrap=True)
        plt.tight_layout()
        # plt.grid()
        plt.show()

        return 0

    def displayHist(
        self,
        showchanXhist: bool = False,
        showchanYhist: bool = False,
        showFFTrealhist: bool = False,
        showFFTimghist: bool = False,
        showPSDhist: bool = False,
        showExcess: bool = False,
        excessThres: float = 3.4,
        freqRangeforHist=None,
        scale: str = "log",  # 'log' or 'linear'
        # normalizebysigma=True,
        showplt: bool = True,
        verbose=False,
    ):
        """
        Display the histograms

        Parameters
        ----------
        showchanXhist, showchanYhist, showPSDhist : bool
            choose to show the histograms of these channels / spectrum


        """
        # print(f'{self.displayHist.__name__} needs documentations. -- Yuzhe 2024-07-10')
        fontsize = 8
        plt.rc("font", size=fontsize)  # font size for all figures
        # plt.rcParams["font.family"] = "serif"
        # plt.rcParams["font.serif"] = ["Times New Roman"]
        # plt.rcParams["font.family"] = "Times New Roman"
        # plt.rcParams["mathtext.fontset"] = 'cm'  # 'dejavuserif'
        plt.style.use("seaborn-v0_8-deep")  # to specify different styles
        fig = plt.figure(figsize=(11 * 0.8, 7 * 0.8), dpi=150)
        width_ratios = [1, 1, 1]
        height_ratios = [1, 1]
        gs = gridspec.GridSpec(
            nrows=2, ncols=3, width_ratios=width_ratios, height_ratios=height_ratios
        )
        # fig.subplots_adjust(
        #     top=,
        #     bottom=,
        #     left=,
        #     right=,
        #     hspace=,
        #     wspace=)
        dataX_hist_ax = fig.add_subplot(gs[0, 0])  #
        dataY_hist_ax = fig.add_subplot(gs[0, 1])  #
        FFTreal_hist_ax = fig.add_subplot(gs[1, 0])  #
        FFTimag_hist_ax = fig.add_subplot(gs[1, 1])  #
        PSD_ax = fig.add_subplot(gs[0, 2])  #
        PSD_hist_ax = fig.add_subplot(gs[1, 2])  #

        #
        def GaussianHistPlot(data, title, ax):
            mean = np.mean(data)
            std = np.std(data)
            hist, bin_edges = np.histogram(
                data / std,
                bins=max(10, 7 * int(np.floor(np.amax(abs(data) / std)))),
                density=False,
            )
            binwidth = abs(bin_edges[1] - bin_edges[0])
            sumofcounts = len(data)
            hist_info = (
                f"sum of counts={sumofcounts:d}\nmean={mean:.1e} [V] std={std:.1e} [V]"
            )
            ax.set_xlabel("signal / $\\sigma$")

            hist_x = []
            hist_y = []
            for i, count in enumerate(hist):
                if count > 0:
                    hist_x.append((bin_edges[i] + bin_edges[i + 1]) / 2.0)
                    hist_y.append(count)
            ax.scatter(
                hist_x,
                hist_y,
                color="goldenrod",
                edgecolors="k",
                linewidths=1,
                marker="o",
                s=6,
                zorder=6,
                label="histogram",
            )
            # plot pdf of gaussian distribution (mean = 0, std = 1)
            xstamp = np.linspace(
                start=np.amin(bin_edges),
                stop=np.amax(bin_edges),
                num=max(100, len(bin_edges)),
                endpoint=True,
            )
            # ax.plot(xstamp, norm.pdf(xstamp, mean, 1), label='Gaussian pdf', linestyle='--')
            ax.plot(
                xstamp,
                sumofcounts * binwidth * norm.pdf(xstamp, 0, 1),
                label="Gaussian pdf",
                linestyle="--",
            )
            # sumofcounts * binwidth *
            ax.set_ylabel("count")
            ax.set_title(title + "\n" + hist_info, size=fontsize)
            ax.set_yscale(scale)
            ax.set_ylim(bottom=0.1)
            ax.legend(loc="best")
            ax.grid(True)
            del (
                mean,
                std,
                hist,
                bin_edges,
                binwidth,
                sumofcounts,
                hist_info,
                hist_x,
                hist_y,
                xstamp,
            )

        if showchanXhist:
            GaussianHistPlot(self.dataX, title="LIA channel X", ax=dataX_hist_ax)
        if showchanYhist:
            GaussianHistPlot(self.dataY, title="LIA channel Y", ax=dataY_hist_ax)

        if showFFTrealhist or showFFTimghist:
            if not hasattr(self, "avgFFT"):
                print(
                    "Warning: LIASignal object does not have the attribute 'avgFFT'. "
                    "Now self.GetFFT() is executed for generating FFT "
                    "for the histogram. "
                )
                self.getNoPulseFFT()

        if showFFTrealhist:
            GaussianHistPlot(self.FFT.real, title="FFT real", ax=FFTreal_hist_ax)
        if showFFTimghist:
            GaussianHistPlot(self.FFT.imag, title="FFT imag", ax=FFTimag_hist_ax)

        def plotPSD(x, y, title, ax: plt.Axes, freqRangeforHist):
            if freqRangeforHist is None:
                r0 = 0
                r1 = len(self.PSD)
            else:
                r0 = np.argmin(abs(self.frequencies - freqRangeforHist[0]))
                r1 = np.argmin(abs(self.frequencies - freqRangeforHist[1]))
            if abs(r1 - r0) < 100:
                print(
                    "Warning: abs(r1-r0) < 100. PSD bins in freqRangeforHist may be too less for histogram"
                )
            ax.plot(x, y, color="tab:green", label="normalized PSD", linewidth=1)
            ax.plot(
                x[r0:r1],
                y[r0:r1],
                color="tab:red",
                label="data for histogram",
                alpha=0.2,
                linewidth=2,
            )

            check((r0, r1))
            # ax.fill_between(x[r0:r1], np.amax(y), np.amin(y),
            #  color = 'r', alpha=0.1, zorder=6)
            # ax.plot(xstamp, chi2.pdf(xstamp, df=2), label='Chi-sq pdf\ndof=2', linestyle='--')
            # ax.plot(xstamp, sumofcounts * binwidth * chi2.pdf(xstamp, df=1), label='Chi-sq pdf dof=1', linestyle='--')
            ax.set_ylabel("normalized PSD")
            ax.set_xlabel("frequency [Hz]")
            ax.set_title(title + "\n", size=fontsize)
            ax.legend(loc="best")
            ax.grid(True)

        def ChisqHistPlot(ax, data, title, thres: float = None):
            """
            data is normalize by being dividing std
            """
            mean = np.mean(data)
            std = np.std(data)
            check(np.amax(data))
            hist, bin_edges = np.histogram(
                data, bins=max(10 * int(np.amax(data)), 100), density=False
            )
            binwidth = abs(bin_edges[1] - bin_edges[0])
            sumofcounts = len(data)
            hist_info = f"sum of counts={sumofcounts:d}\nmean={mean:.1e} [V^2/Hz] std={std:.1e} [V^2/Hz]"
            # if normalizebysigma:
            # bin_edges /= std
            ax.set_xlabel("PSD / $\\sigma_{\\mathrm{PSD}}$")
            # else:
            #     ax.set_xlabel('signal [V]')
            hist_x = []
            hist_y = []
            for i, count in enumerate(hist):
                if count > 0:
                    hist_x.append((bin_edges[i] + bin_edges[i + 1]) / 2.0)
                    hist_y.append(count)
            # TODO add this to every hist_x and hist_y
            hist_x = np.array(hist_x)
            hist_y = np.array(hist_y)
            if thres is not None:
                thresIndex = np.argmin(abs(hist_x - thres))
                check(np.sum(hist_y[thresIndex:]))
                check(np.sum(hist_y[thresIndex:]) / sumofcounts)
            ax.scatter(
                hist_x,
                hist_y,
                color="tab:red",
                edgecolors="k",
                linewidths=1,
                marker="o",
                s=6,
                zorder=6,
                label="histogram",
            )

            # plot pdf of gaussian distribution (mean = 0, std = 1)
            xstamp = np.linspace(
                start=0,
                stop=np.amax(bin_edges),
                num=max(100, len(bin_edges)),
                endpoint=True,
            )
            # ax.plot(xstamp, norm.pdf(xstamp, mean, 1), label='Gaussian pdf', linestyle='--')
            ax.plot(
                xstamp,
                2 * sumofcounts * binwidth * chi2.pdf(2.0 * xstamp, df=2),
                label="Chi-sq pdf\ndof=2",
                linestyle="--",
            )
            # sumofcounts * binwidth *
            # ax.plot(xstamp, sumofcounts * binwidth * chi2.pdf(xstamp, df=1), label='Chi-sq pdf dof=1', linestyle='--')
            ax.set_ylabel("count")
            ax.set_title(title + "\n" + hist_info, size=fontsize)
            ax.set_xscale("linear")
            ax.set_yscale(scale)
            ax.set_ylim(bottom=0.1)
            ax.legend(loc="best")
            ax.grid(True)
            del (
                mean,
                std,
                hist,
                bin_edges,
                sumofcounts,
                hist_info,
                hist_x,
                hist_y,
                xstamp,
            )

        if showPSDhist:
            if not hasattr(self, "PSD"):
                print(
                    "Warning: LIASignal object does not have the attribute 'avgPSD'. "
                    "Now self.GetNoPulsePSD() is executed for generating a power spectrum "
                    "for the histogram. "
                )
                self.getNoPulsePSD()

            plotPSD(
                x=self.frequencies,
                y=self.PSD,
                title="PSD",
                ax=PSD_ax,
                freqRangeforHist=freqRangeforHist,
            )

            if freqRangeforHist is None:
                r0 = 0
                r1 = len(self.PSD)
            else:
                r0 = np.argmin(abs(self.frequencies - freqRangeforHist[0]))
                r1 = np.argmin(abs(self.frequencies - freqRangeforHist[1]))
            ChisqHistPlot(
                ax=PSD_hist_ax,
                data=self.PSD[r0:r1] / np.std(self.PSD[r0:r1]),
                title="Histogram of the normalized PSD",
                thres=excessThres,
            )
            PSD_hist_ax.grid(True)
            del r0, r1

        titletext = "Histograms of data"
        for singlefile in self.fileList:
            titletext += "\n" + singlefile  # include the data file name(s) in the title
        fig.suptitle(titletext, wrap=True)
        plt.tight_layout()
        # plt.grid()
        if showplt:
            plt.show()
        return (
            dataX_hist_ax,
            dataY_hist_ax,
            FFTreal_hist_ax,
            FFTimag_hist_ax,
            PSD_ax,
            PSD_hist_ax,
        )

    # @record_runtime_YorN(RECORD_RUNTIME)
    def psdFindBaseline(
        self,
        HorCombn_opt: bool = False,
        HorCombn_step: int = None,
        poly_degree: int = 4,
        removeBaseline: bool = False,
        showStats: bool = False,
        freqRangeForStats: list = None,
        makeplot: bool = False,
        return_Baseline: bool = False,
        verbose: bool = False,
    ):
        """
        find baseline in the power spectrum
        """
        self.psdBaselineRemoval_flag = False
        if not hasattr(self, "PSD"):
            # clear_lines()
            print(
                f"[{self.psdFindBaseline.__name__}] Warning: object "
                "does not have the attribute PSD. "
                "Now self.GetNoPulsePSD() is executed for generating a power spectrum "
                "for the histogram. "
            )
            self.getNoPulsePSD()

        if len(self.PSD) < 100:
            # clear_lines()
            print(
                f"[{self.psdFindBaseline.__name__}] len(self.PSD) < 100. "
                "array may be too short for baseline diagnostics. "
            )

        if HorCombn_opt:
            assert HorCombn_step is not None
            if HorCombn_step == 1:
                # clear_lines()
                print(
                    f"[{self.psdFindBaseline.__name__}] Warning: HorCombn_step == 1. "
                    "No horizontal combination will be executed. "
                )
                freq = self.frequencies
                PSD = self.PSD
            else:
                num = len(self.PSD) // HorCombn_step
                HorLen = int(num * HorCombn_step)
                freq = self.frequencies[
                    HorCombn_step // 2 : HorLen + HorCombn_step // 2 : HorCombn_step
                ]
                PSD = self.PSD[0:HorLen].reshape((num, HorCombn_step))
                PSD = np.mean(PSD, axis=1)
                assert freq.shape == PSD.shape
        else:
            freq = self.frequencies
            PSD = self.PSD

        if len(PSD) > 1000000:
            # clear_lines()
            print(
                f"[{self.psdFindBaseline.__name__}] Warning: array too long. "
                + f"len(PSD)={len(PSD):.2e} > 1 000 000. "
                + "The computation time for finding the baseline could be long. "
            )

        # find baseline
        baseObj = BaselineRemoval(input_array=PSD)
        tic = time.time()
        # IModpoly method for removing baseline
        # default param: degree=2,repitition=100,gradient=0.001
        output = baseObj.IModPoly(degree=poly_degree)
        toc = time.time()
        # baselineFit_time = abs(toc - tic)

        if verbose:
            print(
                f"[{self.psdFindBaseline.__name__}] "
                + "BaselineRemoval time consumption: "
                + f"{abs(toc - tic):.2g} [s]"
            )
        del baseObj

        base_f = interpolate.interp1d(freq, (PSD - output), fill_value="extrapolate")
        baseline = base_f(self.frequencies)

        iter = 0
        while np.any(baseline == 0) and iter < 100:
            zero_indices = np.where(baseline == 0)[0]
            for idx in zero_indices:
                if idx == 0:
                    # If the zero is at the start, use the next element
                    baseline[idx] = baseline[idx + 1]
                elif idx == len(baseline) - 1:
                    # If the zero is at the end, use the previous element
                    baseline[idx] = baseline[idx - 1]
                else:
                    # Use the average of the neighbors
                    baseline[idx] = (baseline[idx - 1] + baseline[idx + 1]) / 2.0
            iter += 1

        if np.any(baseline == 0):
            raise ValueError(
                f"[{self.psdFindBaseline.__name__}] np.any(baseline == 0) is True. "
            )

        PSDlabel = "PSD"

        normPSD = self.PSD / baseline - 1.0
        normPSDlabel = "normalized PSD (baseline removed)"

        if np.any(normPSD > 1e3):
            print(
                f"[{self.psdFindBaseline.__name__}] Warning: np.any(normPSD > 1e3) is True. "
            )

        if showStats:
            if freqRangeForStats is None:
                r0 = 0
                r1 = len(self.PSD)
            else:
                r0 = np.argmin(abs(self.frequencies - freqRangeForStats[0]))
                r1 = np.argmin(abs(self.frequencies - freqRangeForStats[1]))

            PSDlabel += f"\nmean={np.mean(self.PSD[r0:r1]):.2g}"
            PSDlabel += f"\nstd={np.std(self.PSD[r0:r1]):.2g}"
            normPSDlabel += f"\nmean={np.mean(normPSD[r0:r1]):.2g}"
            normPSDlabel += f"\nstd={np.std(normPSD[r0:r1]):.2g}"
            print(f"[{self.psdFindBaseline.__name__}] stats:")
            print(PSDlabel)
            print(normPSDlabel)
            print("")

        if makeplot:
            plt.rc("font", size=8)  # font size for all figures
            fig = plt.figure(
                figsize=(11 * 0.8, 7 * 0.8), dpi=150
            )  # initialize a figure
            gs = gridspec.GridSpec(nrows=2, ncols=1)  # create grid for multiple figures
            ax00 = fig.add_subplot(gs[0, 0])
            ax10 = fig.add_subplot(gs[1, 0])

            if showStats:
                ax00.fill_between(
                    self.frequencies[r0:r1],
                    np.amax(self.PSD),
                    np.amin(self.PSD),
                    color="r",
                    alpha=0.1,
                    zorder=6,
                    label="Stats Range",
                )
            ax00.plot(self.frequencies, self.PSD, label=PSDlabel)
            if HorCombn_opt:
                ax00.plot(freq, PSD, label="PSD_HC", linestyle="-", c="g")
            ax00.plot(
                self.frequencies, baseline, label="baseline", linestyle="--", c="r"
            )
            ax00.set_ylabel("PSD [V^2/Hz]")

            if showStats:
                ax10.fill_between(
                    self.frequencies[r0:r1],
                    np.amax(normPSD),
                    np.amin(normPSD),
                    color="r",
                    alpha=0.1,
                    zorder=6,
                    label="Stats Range",
                )

            ax10.plot(self.frequencies, normPSD, label=normPSDlabel)

            for ax in [ax00, ax10]:
                ax.set_xlabel("frequency [Hz]")
                ax.legend(loc="best")
                ax.grid()

            plt.show()

        if removeBaseline:
            if verbose:
                print(
                    f"{self.psdFindBaseline.__name__} is removing the baseline in the PSD. "
                )
            self.PSD = normPSD
            del output
            self.psdBaselineRemoval_flag = True

        if return_Baseline:
            return baseline

    def FindPulse(
        self,
        trigger_mod=None,
        trigger_value=None,
        search_mod=None,
        pulseduration=None,
        verbose=False,
    ):
        """
        Find the start and end of excitation pulse in the Aux channel recording.

        (If you are reading in the small window, please go to the source for correct schematics. )

        Parameters
        ----------
        self : (class) LIAsignal
            Signal from lock-in amplifier and also the processed data

        trigger_mod : str
            Trigger mod. It must be either 'auto' or 'manual' (case-insensitive). Defaults to None.
            In auto mod, you do not need to specify a trigger_value, and trigger_value would be np.amax(self.pulsedata) / 2.
            In manual mod, you need to specify a trigger_value.

        trigger_value : float
            Value for detecting rising and falling edge of a pulse. Defaults to None.
            If trigger_mod == 'auto', trigger_value would be automatically chosen as np.amax(self.pulsedata) / 2.
            Otherwise, you need to specify a trigger_value.

        search_mod : str
            the mod of searching for pulses. It must be either 'auto' or 'manual' (case-insensitive). Defaults to None.
            If search_mod == 'auto', the function will automatically find the rising (start) and falling (end) edges of pulses.
            Otherwise, the function would automatically find the rising edges, and determine the falling edges by pulseduation.

        pulseduration : float
            the duration of pulse in [s]. Defaults to None.
            Only when search_mod == 'manual', it is necessary to specify pulseduration,
            so that the function can determine the falling edges of pulses.

        verbose : bool
            Choose True to display processing information. Defaults to False.

        schematics on the algorithm of FindPulse

        Concerning the end of acquisition, please refer to scenario (1) and (2)
        Concerning the start of acquisition, please refer to scenario (1) to (4)

        scenario (1) in either auto or manual mod
        when start[2] - end[2] >= self.acqTime + self.acqDelay
                    ┌───────┐            ┌───────┐            ┌───────┐            ┌───────┐
                    |       |            |       |            |       |            |       |
        ────────────┘       └────────────┘       └────────────┘       └────────────┘       └───────────
                    ↑       ↑            ↑       ↑            ↑       ↑            ↑       ↑          ↑
                discard   end[0]      start[0] end[1]      start[1] end[2]      start[2] end[2]     start[2]
        acquied data:
                            └────────────┘       └────────────┘       └────────────┘       └───────────

        scenario (2) in either auto or manual mod
        when start[2] - end[2] < self.acqTime + self.acqDelay
                    ┌───────┐            ┌───────┐            ┌───────┐            ┌───────┐
                    |       |            |       |            |       |            |       |
        ────────────┘       └────────────┘       └────────────┘       └────────────┘       └──
                    ↑       ↑            ↑       ↑            ↑       ↑            ↑       ↑
                discard   end[0]      start[0] end[1]      start[1] end[2]      start[2] discard
        acquied data:
                            └────────────┘       └────────────┘       └────────────┘

        scenario (3) in auto mod
        ────┐            ┌───────┐            ┌───────┐            ┌───
            |            |       |            |       |            |
            └────────────┘       └────────────┘       └────────────┘
            ↑            ↑       ↑            ↑       ↑            ↑
          end[0]      start[0] end[1]      start[1] end[2]      start[2]
        acquied data:
            └────────────┘       └────────────┘       └────────────┘

        scenario (4) in manual mod
        ────┐            ┌───────┐            ┌───────┐            ┌───
            |            |       |            |       |            |
            └────────────┘       └────────────┘       └────────────┘
            ↑            ↑       ↑            ↑       ↑            ↑
         discard      discard end[0]      start[0] end[1]      start[1]
        acquied data:
                                 └────────────┘       └────────────┘


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
        self.exptype = "Pulsed-NMR"  # FindPulse is only for pulsed-NMR experiment

        # determine the trigger mode
        assert trigger_mod is not None
        if trigger_mod.upper() in AUTO_LIST:
            trigger_value = np.amax(self.pulsedata) / 2.0
        elif trigger_mod.upper() in MANUAL_LIST:
            assert trigger_value is not None

        # determine the length of acquisition delay and acquisition time
        assert self.acqDelay is not None
        assert self.acqTime is not None
        acqdelaylen = int(np.ceil(self.acqDelay * self.samprate))
        if acqdelaylen <= 3:
            print(
                "!!WARNING!! acqDelay * samprate = %d. This could be too short for PSD. "
                % acqdelaylen
            )
        acqtimelen = int(np.ceil(self.acqTime * self.samprate))

        # start to search for pulses
        # it is recommended to use manual mode instead of Auto mode
        assert search_mod is not None
        assert (search_mod.upper() in AUTO_LIST) or (search_mod.upper() in MANUAL_LIST)

        # search for pulses in auto mode
        if search_mod.upper() in AUTO_LIST:
            startofpulse = np.flatnonzero(
                (self.pulsedata[1:] > trigger_value)
                & (self.pulsedata[:-1] < trigger_value)
            )
            endofpulse = np.flatnonzero(
                (self.pulsedata[1:] < trigger_value)
                & (self.pulsedata[:-1] > trigger_value)
            )

            # check the length of startofpulse and endofpulse
            if len(startofpulse) < 1 or len(endofpulse) < 1:
                raise ValueError(
                    "AUTO mode. Number of pulses smaller than 1. \nlen(startofpulse)<1 or len(endofpulse)<1"
                )

            if verbose:
                print("startofpulse.shape before adjustment ", startofpulse.shape)
                print("endofpulse.shape before adjustment ", endofpulse.shape)

            # discard the first element in the startofpulse
            # if it is ahead of the first element in the endofpulse
            if startofpulse[0] < endofpulse[0]:
                startofpulse = startofpulse[1:]

            # check again the length of startofpulse and endofpulse
            if len(startofpulse) < 1 or len(endofpulse) < 1:
                raise ValueError("len(startofpulse)<=1 or len(endofpulse)<=1")

            # discard the last element in the startofpulse
            # if it is ahead of the first element in the endofpulse
            if startofpulse[-1] < endofpulse[-1]:
                #        ┌────────────────┐
                #        |                |
                # ───────┘                └───────────
                #        ↑                ↑
                #  startofpulse[-1] endofpulse[-1]

                # # When there is usable signal after the last pulse start, like in scenario (1):
                # if (len(self.pulsedata) - 1) >= startofpulse[-1] + pulselen + (acqdelaylen + acqtimelen):
                #     startofpulse = np.append(startofpulse, [len(self.pulsedata)-1]) # add the final data point to startofpulse
                # else:
                #     endofpulse = endofpulse[0:-1]
                pass
            if verbose:
                print("startofpulse.shape after adjustment ", startofpulse.shape)
                print("endofpulse.shape after adjustment ", endofpulse.shape)

        # search for pulses in manual mode
        if search_mod.upper() in MANUAL_LIST:
            assert pulseduration is not None
            pulselen = int(pulseduration * self.samprate) + 1
            if verbose:
                print("pulseduration ", pulseduration)
                print("self.samprate ", self.samprate)
                print("pulselen ", pulselen)

            # find the start of pulses
            startofpulse = np.flatnonzero(
                (self.pulsedata[1:] > trigger_value)
                & (self.pulsedata[:-1] < trigger_value)
            )

            assert len(startofpulse) >= 1

            # When there is usable signal after the last pulse start, like in scenario (1)
            if (len(self.pulsedata) - 1) >= startofpulse[-1] + pulselen + (
                acqdelaylen + acqtimelen
            ):
                endofpulse = startofpulse + pulselen  # keep the last endofpulse
                startofpulse = np.append(
                    startofpulse, [len(self.pulsedata) - 1]
                )  # add the final data point to startofpulse
            # When there is no usable signal after the last pulse start, like in scenario (2)
            else:
                if len(startofpulse) == 1:
                    raise ValueError("no useable data. ")
                endofpulse = (
                    startofpulse[:-1] + pulselen
                )  # create endofpulse without the last pulse start

            startofpulse = startofpulse[1:]  # discard the first pulse start

        # save startofpulse and endofpulse
        self.startofpulse = startofpulse
        self.endofpulse = endofpulse
        del startofpulse, endofpulse

        # check if the lengths of data are too short
        if (acqdelaylen + acqtimelen) > np.amin(self.startofpulse - self.endofpulse):
            # print(f'acqdelaylen + acqtimelen = {acqdelaylen + acqtimelen}')
            # print(f'np.amin(self.startofpulse - self.endofpulse) = {np.amin(self.startofpulse - self.endofpulse)}')
            check(self.startofpulse)
            check(self.endofpulse)
            # raise ValueError('(points of acqDelay + points of acqTime) > min(endofpulse-startofpulse)')

        # save parameters
        self.acqdelaylen = acqdelaylen
        self.acqtimelen = acqtimelen
        startofacq = self.endofpulse + acqdelaylen
        self.acq_arr = np.array([startofacq, startofacq + acqtimelen]).transpose()
        # an example of self.acq_arr
        # self.acq_arr : ndarray(array([
        #    [ 15964,  18643],
        #    [ 42748,  45427],
        #    [ 69538,  72217],
        #    [ 96333,  99012],
        #    [123112, 125791],
        #    [149893, 152572],
        #    [176686, 179365],
        #    [203463, 206142],
        #    [230256, 232935],
        #    [257035, 259714],
        #    [283826, 286505],
        #    [310601, 313280],
        #    [337386, 340065],
        #    [364175, 366854],
        #    [390960, 393639]
        #                     ], dtype=int64)) [shape=(15, 2)]
        # here we have 15 chunks of acquired data
        del acqdelaylen, acqtimelen, startofacq

        if verbose:
            check(self.startofpulse)
            check(self.endofpulse)
            check(self.acq_arr)
            print("")
        # return False

    def FindSpinEcho(
        self,
        search_mod: str,
        trigger_value,
        pulseduration: float,
        echonumber: int,
        echotime: float,
        plot: bool,
        savepath: str,
        verbose=False,
    ):
        """
        Find the start and end of pi pulses in a spin-echo measurement.
        The first pulse, which is a pi/2 pulse, is discarded.


                    90° pulse      180° pulse           180° pulse           180° pulse
                                   ┌───────┐            ┌───────┐            ┌───────┐
                                   |       |            |       |            |       |
                    ┌───────┐      |       |            |       |            |       |
                    |       |      |       |            |       |            |       |
        ────────────┘       └──────┘       └────────────┘       └────────────┘       └───────────
                    ↑       ↑      ↑       ↑            ↑       ↑            ↑       ↑
                discard   discard discard end[0]      start[0] end[1]     start[1] discard


        Parameters
        ----------
        self : (class) LIAsignal
            Signal from lock-in amplifier and also the processed data

        search_mod : str


        trigger_value : float
            Value for detecting rising and falling edge of a pulse. Defaults to None.
            If trigger_mod == 'auto', trigger_value would be automatically chosen as np.amax(self.pulsedata) / 2.
            Otherwise, you need to specify a trigger_value.

        search_mod : str
            same as for FindPulse()

        pulseduration : float
            Duration of a single pulse in

        echonumber : float
            Number of 180° pulses

        echotime : float
            Time between the starting points of 2 180° pulses

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
        self.exptype = "CPMG"

        # determine the trigger mode
        assert search_mod is not None
        if search_mod.upper() in AUTO_LIST:
            trigger_value = np.amax(self.pulsedata) / 2.0
            check(trigger_value)
        elif search_mod.upper() in MANUAL_LIST:
            assert trigger_value is not None

        # if trigger_value == 0:
        #    trigger_value = np.amax(self.pulsedata) / 2.
        #    if verbose:
        #        check(trigger_value)

        # determine the length of acquisition delay and acquisition time
        assert self.acqDelay is not None
        assert self.acqTime is not None
        acqdelaylen = int(np.ceil(self.acqDelay * self.samprate))

        if acqdelaylen <= 3:
            print(
                f"!!WARNING!! acqDelay * samprate = {acqdelaylen}. This could be too short. "
            )

        acqtimelen = int(np.ceil(self.acqTime * self.samprate))
        self.acqtimelen = acqtimelen

        # parameter sanity check
        assert search_mod is not None
        assert (search_mod.upper() in AUTO_LIST) or (search_mod.upper() in MANUAL_LIST)

        # start to search for pulses
        # search for pulses in auto mode
        if search_mod.upper() in AUTO_LIST:
            startofpulse = np.flatnonzero(
                (self.pulsedata[1:] > trigger_value)
                & (self.pulsedata[:-1] < trigger_value)
            )
            endofpulse = np.flatnonzero(
                (self.pulsedata[1:] < trigger_value)
                & (self.pulsedata[:-1] > trigger_value)
            )

            # check the length of startofpulse and endofpulse
            if len(startofpulse) < 1:
                raise ValueError(
                    "AUTO mode. Number of pulses smaller than 1. \nlen(startofpulse)<1"
                )
            if len(endofpulse) < 1:
                raise ValueError(
                    "AUTO mode. Number of pulses smaller than 1. \nlen(endofpulse)<1"
                )
            if verbose:
                print("startofpulse.shape before adjustment ", startofpulse.shape)
                print("endofpulse.shape before adjustment ", endofpulse.shape)

            # discard the first element because it is the pi/2 pulse
            if verbose:
                print(
                    "removing first entry (90° pulse) from pulse arrays for spin echo analysis"
                )
            startofpulse = startofpulse[1:]
            endofpulse = endofpulse[1:]

            # if the first element in the startofpulses is ahead of that in the endofpulse
            # discard the first element in the startofpulse
            if startofpulse[0] < endofpulse[0]:
                startofpulse = startofpulse[1:]

            # check again the length of startofpulse and endofpulse
            if len(startofpulse) < 1 or len(endofpulse) < 1:
                raise ValueError("len(startofpulse)<=1 or len(endofpulse)<=1")

            # discard the last element in the endofpulse
            # if it is behind the last element in the startofpulse
            if startofpulse[-1] < endofpulse[-1]:
                #        ┌────────────────┐
                #        |                |
                # ───────┘                └───────────
                #        ↑                ↑
                #  startofpulse[-1] endofpulse[-1]
                endofpulse = endofpulse[0:-1]

            if verbose:
                print("startofpulse.shape after adjustment ", startofpulse.shape)
                print("endofpulse.shape after adjustment ", endofpulse.shape)

            #########################
            # # check mismatch
            # diff_default = startofpulse[0] - endofpulse[0]
            # if verbose: check(diff_default)
            # mismatches = np.ones(len(startofpulse))
            # mismatches[0] = 0
            # while 1 in mismatches:
            #     for i in range(1, len(startofpulse)):
            #         diff = startofpulse[i] - endofpulse[i]
            #         if diff != diff_default:
            #             endofpulse[i] += 1
            #             mismatches[i] = 1
            #             print("chunk mismatch ALERT")
            #             check(diff)
            #         else:
            #             mismatches[i] = 0
            # #check(mismatches)
            #########################

            # save variables
            self.startofpulse = startofpulse
            self.endofpulse = endofpulse

            # determine the acquisition array and the acquisition length
            startofacq = self.endofpulse + acqdelaylen
            acqlength = self.startofpulse[0] - self.endofpulse[0] - acqdelaylen
            endofacq = self.endofpulse + acqlength

            self.acq_arr = np.array([startofacq, endofacq]).transpose()
            # if verbose: check(self.acq_arr)
            if len(self.acq_arr) <= 3:
                print("WARNING! len(self.acq_arr) <= 3. Too less pulses?")
            # self.acq_arr = self.acq_arr[1:]

            del (
                startofpulse,
                endofpulse,
                startofacq,
                endofacq,
                acqdelaylen,
            )  # , mismatches

        elif search_mod.upper() in MANUAL_LIST:
            if pulseduration is None:
                pulseduration = self.pulseduration
            if pulseduration is None:
                raise ValueError("pulseduration and self.pulseduration is None")
            pulselen = int(pulseduration * self.samprate) + 1
            if verbose:
                print("pulseduration ", pulseduration)
                print("self.samprate ", self.samprate)
                print("pulselen ", pulselen)

            # find the start of pulses
            startofpulse = np.flatnonzero(
                (self.pulsedata[1:] > trigger_value)
                & (self.pulsedata[:-1] < trigger_value)
            )
            if verbose:
                check(self.pulsedata[1:])
                check(self.pulsedata[:-1])
                check(trigger_value)
            assert len(startofpulse) >= 1

            # When there is usable signal after the last pulse start,
            # like in FindPulse() documentation scenario (1)
            if (len(self.pulsedata) - 1) >= startofpulse[-1] + pulselen + (
                acqdelaylen + acqtimelen
            ):
                endofpulse = startofpulse + pulselen  # keep the last endofpulse
                startofpulse = np.append(
                    startofpulse, [len(self.pulsedata) - 1]
                )  # add the final data point to startofpulse
            # When there is no usable signal after the last pulse start, like in FindPulse() scenario (2)
            else:
                if len(startofpulse) == 1:
                    raise ValueError("no useable data. ")
                endofpulse = (
                    startofpulse[:-1] + pulselen
                )  # create endofpulse without the last pulse start

            if verbose:
                print(
                    "remove first entry (90° pulse) from pulse arrays for spin echo analysis"
                )
            startofpulse = startofpulse[1:]  # discard the first pulse start

            #################################################################
            # do a number of corrections which apparently worked a long time ago

            endofpulse = endofpulse[1:]  # why?  # Yuzhe bookmark

            if len(startofpulse) < 1 or len(endofpulse) < 1:
                return True
                # raise ValueError('number of pulses less than 1.')

            if verbose:
                print("startofpulse.shape before adjustment ", startofpulse.shape)
                print("endofpulse.shape before adjustment ", endofpulse.shape)

            if startofpulse[0] < endofpulse[0]:
                print("fixing first pulse start")
                startofpulse = startofpulse[1:]

            if startofpulse[-1] < endofpulse[-1]:
                # if IndexError: # unsafe edit just to get it working, we have to resolve it later
                #    pass
                print("fixing last pulse start")
                endofpulse = endofpulse[0:-1]

            if endofpulse[-1] > (len(self.pulsedata) - 1):
                print("fixing last pulse end")
                endofpulse = endofpulse[:-1]

            if verbose:
                print("startofpulse.shape after adjustment ", startofpulse.shape)
                print("endofpulse.shape after adjustment ", endofpulse.shape)

            #################################################################
            # end of corrections

            if verbose:
                check(startofpulse)
                check(endofpulse)

            # save startofpulse and endofpulse
            self.startofpulse = startofpulse
            self.endofpulse = endofpulse
            del startofpulse, endofpulse

            echotimelen = int(echotime * self.samprate)

            startofacq = self.endofpulse + acqdelaylen
            endofacq = self.endofpulse + echotimelen - pulselen
            self.acq_arr = np.array([startofacq, endofacq]).transpose()
            if verbose:
                check(self.acq_arr)
            if len(self.acq_arr) <= 3:
                print("WARNING! len(self.acq_arr) <= 3. Too less pulses?")

            del pulselen, startofacq, endofacq

        else:
            raise ValueError("search_mod not found")

        # do some final checks

        acqdelaylen = int(self.acqDelay * self.samprate)
        if acqdelaylen <= 3:
            print(
                f"!!WARNING!! acqDelay * samprate = {acqdelaylen} could be too small. "
            )

        echotimelen = int(echotime * self.samprate)
        pulse_diff = np.amin(self.startofpulse - self.endofpulse)
        if verbose:
            check(echotimelen)
            check(pulse_diff)
        # these 2 above should be the same I guess? when testing: difference 10

        pulse_interval = np.amax(self.endofpulse) - np.amin(self.endofpulse)
        echotime_total = echotimelen * echonumber
        if verbose:
            check(pulse_interval)
            check(echotime_total)
        # same for these 2, difference 1595

        if acqdelaylen > pulse_diff:
            print("acqdelaylen ", acqdelaylen)
            print("np.amin(self.startofpulse - self.endofpulse) ", pulse_diff)
            raise ValueError("points of acqDelay > min(endofpulse-startofpulse)")

        # check if the lengths of data are too short
        if (acqdelaylen + acqtimelen) > np.amin(self.startofpulse - self.endofpulse):
            print(f"acqdelaylen + acqtimelen = {acqdelaylen + acqtimelen}")
            print(
                f"np.amin(self.startofpulse - self.endofpulse) = {np.amin(self.startofpulse - self.endofpulse)}"
            )
            check(self.startofpulse)
            check(self.endofpulse)
            raise ValueError(
                "(points of acqDelay + points of acqTime) > min(endofpulse-startofpulse)"
            )

        del acqdelaylen, echotimelen, pulse_diff, pulse_interval, echotime_total

        # plot only data within acquisition windows
        if plot:
            fig, ax1 = plt.subplots(figsize=(16, 9), dpi=100)
            ax2 = ax1.twinx()
            # ax1.plot(self.timestamp, self.pulsedata, color="g")
            # ax2.plot(self.timestamp, (self.dataX), color="b")

            FIDall = []
            alltimes = []
            for i in range(len(self.acq_arr)):

                timedomain = self.timeStamp_s[self.acq_arr[i, 0] : self.acq_arr[i, 1]]
                alltimes.append(timedomain)

                pulsedata = self.pulsedata[self.acq_arr[i, 0] : self.acq_arr[i, 1]]

                FIDhere = (
                    self.dataX[self.acq_arr[i, 0] : self.acq_arr[i, 1]]
                    + 1j * self.dataY[self.acq_arr[i, 0] : self.acq_arr[i, 1]]
                )
                FIDall.append(FIDhere)

                # ax1.plot(timedomain, pulsedata, color="g")
                ax2.plot(timedomain, FIDhere, color="b")
            fig.savefig(savepath[:-4] + "_timeseries.png")

        # plt.figure()
        # plt.plot(self.startofpulse)
        # plt.plot(self.endofpulse)
        # plt.grid()
        # plt.title('p')
        # plt.show()
        # plt.close()

        # plt.figure()
        # plt.plot(self.startofpulse-self.endofpulse)
        # plt.grid()
        # plt.title('self.startofpulse-self.endofpulse')
        # plt.show()
        # plt.close()

    def GetAvgPSD(
        self,
        AVARPopt: bool = False,
        windowfunction="rectangle",
        decayfactor=-10,
        selectshots=[],
        verbose=False,
    ):
        """
        Compute average power spectrum.

        Parameters
        ----------
        (not implemented yet)AVARPopt : bool
            Option for computing the Allan variance of power spectra. Defaults to False.

        windowfunction : str
            window function for FFT.
            Available choices:
                'rectangle'
                'expdecay'
                'Hanning' or 'hanning' or 'Han' or 'han'
                'Hamming' or 'hamming'
                'Blackman'
            Defaults to 'rectangle'.

        decayfactor : float
            parameter for window function 'expdecay'.
            The 'expdecay' window function generates an array of [exp(-df*0), exp(-df*1), ... , exp(-df*(num-1))],
            where df stands for decayfactor.
            Defaults to -10.

        selectshots : list
            List of selected shots / chunks of acquired time-series for power spectra.
            e.g. [0] means that you only use the first shot.
            Defaults to [], which means all shots / chunks are included.

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
        https://holometer.fnal.gov/GH_FFT.pdf
        For details, see utils - stdPSD and stdLIAPSD
        """
        # initialize an all-zero PSD array
        PSD = np.zeros(abs(self.acq_arr[0, 1] - self.acq_arr[0, 0]))

        # select the shots / chunks for PSD
        if len(selectshots) == 0 or selectshots is None:
            # when all shots / chunks are included
            selectPulse_list = range(len(self.acq_arr[:, 0]))
        else:
            # when the user selects the shots / chunks
            selectPulse_list = selectshots

        # computed accumulated PSD of selected shots / chunks
        for i in selectPulse_list:
            frequencies, singlePSD = stdLIAPSD(
                data_x=self.dataX[self.acq_arr[i, 0] : self.acq_arr[i, 1]],
                data_y=self.dataY[self.acq_arr[i, 0] : self.acq_arr[i, 1]],
                samprate=self.samprate,
                demodfreq=self.demodfreq,
                attenuation=self.attenuation,
                windowfunction=windowfunction,
                decayfactor=decayfactor,
                showwindow=False,
                DTRCfilter=self.filterstatus,
                DTRCfilter_TC=self.DTRC_TC,
                DTRCfilter_order=self.DTRC_order,
                verbose=verbose,
            )
            PSD += singlePSD  # accumulate PSD
            del singlePSD  # delete singlePSD before going to the next iteration

        self.frequencies = frequencies  # store frequency axis
        self.PSD = PSD / len(selectPulse_list)  # store the average PSD
        self.selectPulse_list = (
            selectPulse_list  # This is for plotting in self.GetSpectrum()
        )
        del frequencies, PSD  # delete useless variables

    def GetAvgPSD_NoiseReduction(
        self,
        windowfunction="rectangle",
        decayfactor=-10,
        selectshots=[],
        verbose=False,
    ):
        """
        Compute average power spectrum, and substract the nosie background from the signal spectrum.
        This function is specially designed for pulsed-NMR data.
        The noise background is generated by the latter part of the acquired time-series,
        where the decay signal is weak, while the noise is the same.
        A schematic is included here:
            ┌───────┐                                                   ┌───────┐
            |       |                                                   |       |
        ────┘       └───────────────────────────────────────────────────┘
                    |←   →|←             →|       |←             →|←   →|
                      (a)       (b)                      (c)        (d)
        caption: (a) acqDelay. (b) acqTime for signal PSD. (c) acqTime for noise PSD.
        (d) safe space before the next pulse so that the noise PSD is not
        influenced by the next pulse. The length is chosen to be the same as acqDelay.

        This method can reduce or elimate noise peaks that are independent of excitation pulses,
        while keeping the decay signal.

        Parameters
        ----------
        windowfunction : str
            window function for FFT.
            Available choices:
                'rectangle'
                'expdecay'
                'Hanning' or 'hanning' or 'Han' or 'han'
                'Hamming' or 'hamming'
                'Blackman'
            Defaults to 'rectangle'.

        decayfactor : float
            parameter for window function 'expdecay'. Defaults to -10.
            The 'expdecay' window function generates an array of [exp(-df*0), exp(-df*1), ... , exp(-df*(num-1))],
            where df stands for decayfactor.

        selectshots : list
            List of selected shots / chunks of acquired time-series for power spectra.
            e.g. [0] means that you only use the first shot.
            Defaults to [], which means all shots / chunks are included.

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
        https://holometer.fnal.gov/GH_FFT.pdf
        For details, see utils - stdPSD and stdLIAPSD
        """

        # select the shots / chunks for PSD
        if len(selectshots) == 0 or selectshots is None:
            # when all shots / chunks are included
            selectPulse_list = range(len(self.acq_arr[:, 0]))
        else:
            # when the user selects the shots / chunks
            selectPulse_list = selectshots

        # get the length of reptime (intervals between pulses)
        reptimelen_list = []
        for i in selectPulse_list:
            reptimelen_list.append(self.startofpulse[i] - self.endofpulse[i])

        # check if the reptime's / intervals between pulses are large enough
        if np.amin(reptimelen_list) < 2 * (self.acqdelaylen + self.acqtimelen):
            print(
                "!!WARNING!! minimum reptimelen < 2 * (acqdelaylen + acqtimelen). You may need to increase repetiton time. Otherwise you could weaken the signal. "
            )

        del selectPulse_list, reptimelen_list

        # extract noise spectrum (noise background)
        startofacq = self.startofpulse - (self.acqdelaylen + self.acqtimelen)
        self.acq_arr = np.array([startofacq, startofacq + self.acqtimelen]).transpose()
        self.GetAvgPSD(
            windowfunction=windowfunction,
            decayfactor=decayfactor,
            selectshots=selectshots,
            verbose=verbose,
        )
        noisespectrum = self.PSD
        del self.PSD

        # extract signal spectrum
        startofacq = self.endofpulse + self.acqdelaylen
        self.acq_arr = np.array([startofacq, startofacq + self.acqtimelen]).transpose()
        self.GetAvgPSD(
            windowfunction=windowfunction,
            decayfactor=decayfactor,
            selectshots=selectshots,
            verbose=verbose,
        )

        # substract the nosie background from the signal spectrum
        self.PSD -= noisespectrum

        del startofacq, noisespectrum  # delete useless variables

    def GetAvgTSsq(self, verbose=False):
        """
        Compute the time-series mean square. The result is stored in self.AvgTSsq

        Parameters
        ----------
        verbose : bool, optional
            Defaults to False.

        Returns
        -------
        Null

        """
        sumofTSsq = 0.0
        self.avgTSsq = 0.0
        numofpulses = len(self.acq_arr[:, 0])
        for i in range(numofpulses):
            TSHere = (
                self.dataX[self.acq_arr[i, 0] : self.acq_arr[i, 1]]
                + 1j * self.dataY[self.acq_arr[i, 0] : self.acq_arr[i, 1]]
            )
            sumofTSsq += np.sum(abs(TSHere) ** 2)
        if verbose:
            plt.figure()
            plt.plot(np.real(sumofTSsq) / numofpulses, label="np.real(sumofTS)")
            plt.plot(np.imag(sumofTSsq) / numofpulses, label="np.imag(sumofTS)")
            # plt.scatter(np.real(sumofTS))
            # plt.scatter(np.imag(sumofTS))
            plt.grid()
            plt.title("Averaged TS square")
            plt.show()
        self.avgTSsq = np.real_if_close(sumofTSsq) / numofpulses
        self.avgFIDsq = self.avgTSsq
        del sumofTSsq, numofpulses
        return self.avgTSsq

    def GetTSpower(self, selectshots=[], verbose: bool = False):
        """
        Compute the power of time-series. The result is stored in self.TSPower

        Parameters
        ----------
        selectshots : list, optional
            List of selected shots / chunks of acquired time-series for power spectra.
            e.g. [0] means that you only use the first shot.
            Defaults to [], which means all shots / chunks are included.
        verbose : bool, optional
            Defaults to False.

        Returns
        -------
        self.TSPower

        """
        # select the shots / chunks for PSD
        if len(selectshots) == 0 or selectshots is None:
            # when all shots / chunks are included
            selectPulse_list = range(len(self.acq_arr[:, 0]))
        else:
            # when the user selects the shots / chunks
            selectPulse_list = selectshots

        # initialize sum of time-series square
        meanofTSsq = 0.0
        # loop over selectPulse_list to sum up the time-series square
        for i in selectPulse_list:
            TSHere = (
                self.dataX[self.acq_arr[i, 0] : self.acq_arr[i, 1]]
                + 1j * self.dataY[self.acq_arr[i, 0] : self.acq_arr[i, 1]]
            )
            meanofTSsq += np.mean(abs(TSHere) ** 2)

        # Compute the power of time-series
        # Power = 1 / T * Integrate_{0}^{T} s^2(t) dt
        #       = 1 / (N * 1 / SR) * Sum_{i=0}^{i=N-1} s^2_i * (1 / SR)
        #       = 1 / N * Sum_{i=0}^{i=N-1} s^2_i , where SR stands for sampling rate
        self.TSpower = np.real_if_close(meanofTSsq) / len(selectPulse_list)
        if verbose:
            check(self.TSpower)
        del meanofTSsq
        return self.TSpower

    def GetFFTpower(self, verbose: bool = False):
        """
        Compute the power of FT spectrum. The result is stored in self.FFTpower

        Parameters
        ----------
        verbose : bool, optional
            Defaults to False.

        Returns
        -------
        self.FFTpower

        """
        assert hasattr(self, "avgFFT")
        assert hasattr(self, "freq_resol")
        self.FFTpower = np.sum(np.abs(self.FFT) ** 2.0) * self.freq_resol_Hz
        return self.FFTpower

    def GetPSDpower(self, verbose: bool = False):
        """
        Compute the power of PSD. The result is stored in self.PSDpower

        Parameters
        ----------
        verbose : bool, optional
            Defaults to False.

        Returns
        -------
        self.FFTpower

        """
        assert hasattr(self, "PSD")
        assert hasattr(self, "freq_resol")
        self.PSDpower = np.sum(self.PSD) * self.freq_resol_Hz
        self.PSDpower

    def FitPSD(
        self,
        fitfunction="Lorentzian",  # 'Lorentzian' 'dualLorentzian' 'tribLorentzian' 'Gaussian 'dualGaussian'
        inputfitparas=["auto", "auto", "auto", "auto"],
        smooth=False,
        smoothlevel=1,
        fitrange=[0, -1],
        alpha=0.05,
        getresidual=False,
        getchisq=False,
        verbose=False,
    ):
        """
        Fit on power spectral line.

        Parameters
        ----------
        fitfunction : str
            Name of the fit function.
            All options: 'Lorentzian' 'dualLorentzian' 'tribLorentzian' 'Gaussian 'dualGaussian'
            'Polyeven' and 'auto' (all the options in FunctionParas.keys()).
            Defaults to 'Lorentzian'

        inputfitparas : list
            The initial guess for parameters.
            Defaults to ['auto','auto','auto','auto'],

        smooth : bool, optional
            Option to smooth the spectral line by moving average before fitting.
            Defaults to False.

        smoothlevel : int, optional
            The extend to which the spectral line is smoothed.
            Higher the value, smoother the line.
            Defaults to 1.
            To know more about the effects of it, check figures in
            CASPEr Code\\Supplementary\\Smoothforautofitting

        fitrange : list, optional
            The index range for the fitting.
            Defaults to [0, -1], which means the whole spectrum would be fed for fitting.

        alpha : float, optional
            The parameter for confidence interval (CI).
            Refer to https://en.wikipedia.org/wiki/Confidence_interval
            Defaults to 0.05.

        getresidual : bool, optional
            The option to get residual values after fitting.
            Defaults to False.

        getchisq : bool, optional
            The option to conduct chi-square test.
            Refer to https://en.wikipedia.org/wiki/Chi-squared_test
            Defaults to False.

        verbose : bool, optional
            Defaults to False.

        Returns
        --------
        if the fitting is unsuccessful, returns nothing
        else, returns
        self.popt : array
            result of fir parameters

        tval*self.perr : array
            The error in fit parameters under the specified confidence level
        """
        fitfunction = fitfunction.upper()
        # If the fitfunction is 'auto', all possible fitting would be tried.
        if fitfunction in AUTO_LIST:
            FitResult_arr = []
            fitfunction_list = list(FunctionParas.keys())
            for i in range(len(fitfunction_list)):
                self.FitPSD(
                    fitfunction=fitfunction_list[i],
                    smooth=smooth,
                    smoothlevel=smoothlevel,
                    fitrange=fitrange,
                    alpha=alpha,
                    getresidual=True,
                    getchisq=getchisq,
                    verbose=False,
                )
                FitResult_arr.append(
                    [self.residualval, self.chisq]
                )  # be careful with None here
            FitResult_arr = np.array(FitResult_arr)

            # The fitting with least residualval wins
            fitfunction = Function_dict[np.argmin(FitResult_arr[:, 0])][0]
            if verbose:
                print("Best fit function found: ", fitfunction)

        # initialize variables
        self.fitflag = False
        self.popt = None
        self.pcov = None
        self.residual = None
        self.residualval = None
        self.chisq = None
        self.fitcurves = []

        # analysis range
        ar = [0, len(self.PSD)]
        if fitrange is not None:
            for i, freqindex in enumerate(fitrange):
                if type(freqindex) == int:
                    ar[i] = freqindex
                if type(freqindex) == float:
                    print(
                        "WARNING!! fitrange should be index range (int) instead of float. The input fitrange may not be accepted. "
                    )

        # get the curve function from the dictionary
        curvefunction = Function_dict[fitfunction][1]

        # estimate fit parameters
        fitparas = Function_dict[fitfunction][2](
            datax=self.frequencies[ar[0] : ar[1]],
            datay=self.PSD[ar[0] : ar[1]],
            smooth=smooth,
            smoothlevel=smoothlevel,
            verbose=verbose,
        )

        # update fit parameters if the user specifies
        for i, input in enumerate(inputfitparas):
            if type(input) == int or type(input) == float:
                fitparas[i] = 1.0 * input

        # try to fit
        # check(fitparas)
        try:
            if verbose:
                print("start " + Function_dict[fitfunction][0] + " fitting")
            self.popt, self.pcov = scipy.optimize.curve_fit(
                curvefunction,
                self.frequencies[ar[0] : ar[1]],
                self.PSD[ar[0] : ar[1]],
                p0=fitparas,
                absolute_sigma=False,
            )
            self.fitflag = True
            if verbose:
                print(
                    "Obtained " + Function_dict[fitfunction][0] + " fit parameters ",
                    [x for x in self.popt],
                )
        except RuntimeError:
            print("fit RuntimeError")
        else:
            pass
        # del fitparas

        # If the fitting is unsuccessful, terminate the program.
        if self.fitflag is False:
            return

        # If the fitting is successful, generate a fit results, analysis and report.
        self.perr = np.sqrt(np.diag(self.pcov))
        dof = max(
            0, len(self.frequencies[ar[0] : ar[1]]) - len(self.popt)
        )  # degree of freedom
        # student-t value for the dof and confidence level
        tval = scipy.stats.distributions.t.ppf(1.0 - alpha / 2.0, dof)

        # Generate the array of fit curves
        if Function_dict[fitfunction][0] == "Lorentzian":
            self.popt[1] = abs(self.popt[1])
            self.fitcurves.append(
                Lorentzian(
                    self.frequencies[ar[0] : ar[1]],
                    self.popt[0],
                    self.popt[1],
                    self.popt[2],
                    self.popt[3],
                )
            )
        elif Function_dict[fitfunction][0] == "dualLorentzian":
            self.popt[1] = abs(self.popt[1])
            self.popt[4] = abs(self.popt[4])
            self.fitcurves.append(
                Lorentzian(
                    self.frequencies[ar[0] : ar[1]],
                    self.popt[0],
                    self.popt[1],
                    self.popt[2],
                    self.popt[6] / 2,
                )
            )
            self.fitcurves.append(
                Lorentzian(
                    self.frequencies[ar[0] : ar[1]],
                    self.popt[3],
                    self.popt[4],
                    self.popt[5],
                    self.popt[6] / 2,
                )
            )
            self.fitcurves.append(
                dualLorentzian(
                    self.frequencies[ar[0] : ar[1]],
                    self.popt[0],
                    self.popt[1],
                    self.popt[2],
                    self.popt[3],
                    self.popt[4],
                    self.popt[5],
                    self.popt[6],
                )
            )
        elif Function_dict[fitfunction][0] == "tribLorentzian":
            self.popt[1] = abs(self.popt[1])
            self.popt[4] = abs(self.popt[4])
            self.popt[7] = abs(self.popt[7])
            self.fitcurves.append(
                Lorentzian(
                    self.frequencies[ar[0] : ar[1]],
                    self.popt[0],
                    self.popt[1],
                    self.popt[2],
                    self.popt[9],
                )
            )
            self.fitcurves.append(
                Lorentzian(
                    self.frequencies[ar[0] : ar[1]],
                    self.popt[3],
                    self.popt[4],
                    self.popt[5],
                    self.popt[9],
                )
            )
            self.fitcurves.append(
                Lorentzian(
                    self.frequencies[ar[0] : ar[1]],
                    self.popt[6],
                    self.popt[7],
                    self.popt[8],
                    self.popt[9],
                )
            )
            self.fitcurves.append(
                tribLorentzian(
                    self.frequencies[ar[0] : ar[1]],
                    self.popt[0],
                    self.popt[1],
                    self.popt[2],
                    self.popt[3],
                    self.popt[4],
                    self.popt[5],
                    self.popt[6],
                    self.popt[7],
                    self.popt[8],
                    self.popt[9],
                )
            )
        elif Function_dict[fitfunction][0] == "Gaussian":
            self.fitcurves.append(
                Gaussian(
                    self.frequencies[ar[0] : ar[1]],
                    self.popt[0],
                    self.popt[1],
                    self.popt[2],
                    self.popt[3],
                )
            )
        elif Function_dict[fitfunction][0] == "dualGaussian":
            self.fitcurves.append(
                Gaussian(
                    self.frequencies[ar[0] : ar[1]],
                    self.popt[0],
                    self.popt[1],
                    self.popt[2],
                    self.popt[6] / 2,
                )
            )
            self.fitcurves.append(
                Gaussian(
                    self.frequencies[ar[0] : ar[1]],
                    self.popt[3],
                    self.popt[4],
                    self.popt[5],
                    self.popt[6] / 2,
                )
            )
            self.fitcurves.append(
                dualGaussian(
                    self.frequencies[ar[0] : ar[1]],
                    self.popt[0],
                    self.popt[1],
                    self.popt[2],
                    self.popt[3],
                    self.popt[4],
                    self.popt[5],
                    self.popt[6],
                )
            )

        # Compute residual and residual value.
        if getresidual:
            self.residual = self.PSD[ar[0] : ar[1]] - self.fitcurves[-1]
            self.residualval = np.sum(abs(self.residual))

        # Compute chi square
        # Not sure if this is the correct way to do so, but self.chisq is certainly helpful.
        if getchisq:
            self.chisq = np.sum(
                (self.PSD[ar[0] : ar[1]] - self.fitcurves[-1]) ** 2
                / abs(self.fitcurves[-1])
            )

        ############################################################################################
        # Start of generating fitreport
        self.fitreport = (
            Function_dict[fitfunction][0]
            + f" Fit ({(100 - 100 * alpha):.0f} % confidence level)\n"
        )
        fitresultlist = []
        for i in range(len(self.popt)):
            fitresultlist.append(ufloat(self.popt[i], tval * self.perr[i]))

        # some example results:

        # G:/SQUID NMR/Shim Run/BayesAutoShim_20221123_111042/runnumber_0.h5
        # "Center frequency = 1.347654(4) MHz\n"
        # "Linewidth = 212(11) Hz\n"
        # "Area = 1.25(5)e-8 $\phi_0^2$\n"
        # "Offset = 2.1(1)e-12 $\phi_0^2$/Hz\n"

        # G:/SQUID NMR/Shim Run/BayesAutoShim_20221123_111042/runnumber_14.h5
        # "Center frequency = 1.3489973(3) MHz\n"
        # "Linewidth = 29.7(7) Hz\n"
        # "Area = 2.04(3)e-8 $\phi_0^2$\n"
        # "Offset = 2.7(2)e-12 $\phi_0^2$/Hz\n"

        # G:/SQUID NMR/Shim Run/BayesAutoShim_20221221_162415/runnumber_42.h5
        # "Center frequency = 1.34838796(7) MHz\n"
        # "Linewidth = 7.8(2) Hz\n"
        # "Area = 4.80(8)e-9 $\phi_0^2$\n"
        # "Offset = 4.6(1)e-12 $\phi_0^2$/Hz\n"

        if Function_dict[fitfunction][0] == "Lorentzian":
            # self.fitreport += "Center frequency = 1.34838796(7) MHz\n"#f'center = {fitresultlist[0]:.6ue} Hz\n' #:.2f #:.6ue
            # self.fitreport += "Linewidth = 7.8(2) Hz\n"#f'linewidth = {abs(fitresultlist[1]):.2f} Hz\n' # linewidth
            # self.fitreport += "Area = 4.80(8)e-9 $\phi_0^2$\n"#farea = {fitresultlist[2]:.2e}\n'
            # self.fitreport += "Offset = 4.6(1)e-12 $\phi_0^2$/Hz\n"#f'offset = {fitresultlist[3]:.2e}\n'
            self.fitreport += "center = {:.3f} Hz\n".format(fitresultlist[0])
            self.fitreport += "linewidth = {:.3f} Hz\n".format(fitresultlist[1])
            self.fitreport += "area = {:.3e}\n".format(fitresultlist[2])
            self.fitreport += "offset = {:.2e}\n".format(fitresultlist[3])
        elif Function_dict[fitfunction][0] == "dualLorentzian":
            self.fitreport += "center0 = {:.2f} Hz\n".format(fitresultlist[0])
            self.fitreport += "linewidth0 = {:.2f} Hz\n".format(fitresultlist[1])
            self.fitreport += "area0 = {:.2e}\n".format(fitresultlist[2])
            self.fitreport += "center1 = {:.2f} Hz\n".format(fitresultlist[3])
            self.fitreport += "linewidth1 = {:.2f} Hz\n".format(fitresultlist[4])
            self.fitreport += "area1 = {:.2e}\n".format(fitresultlist[5])
            self.fitreport += "offset = {:.2e}\n".format(fitresultlist[6])
        elif Function_dict[fitfunction][0] == "tribLorentzian":
            self.fitreport += "center0 = {:.2f} Hz\n".format(fitresultlist[0])
            self.fitreport += "linewidth0 = {:.2f} Hz\n".format(fitresultlist[1])
            self.fitreport += "area0 = {:.2e}\n".format(fitresultlist[2])
            self.fitreport += "center1 = {:.2f} Hz\n".format(fitresultlist[3])
            self.fitreport += "linewidth1 = {:.2f} Hz\n".format(fitresultlist[4])
            self.fitreport += "area1 = {:.2e}\n".format(fitresultlist[5])
            self.fitreport += "center2 = {:.2f} Hz\n".format(fitresultlist[6])
            self.fitreport += "linewidth2 = {:.2f} Hz\n".format(fitresultlist[7])
            self.fitreport += "area2 = {:.2e}\n".format(fitresultlist[8])
            self.fitreport += "offset = {:.2e}\n".format(fitresultlist[9])
        elif Function_dict[fitfunction][0] == "Gaussian":
            self.fitreport += "center = {:.2f} Hz\n".format(fitresultlist[0])
            # self.fitreport += 'sigma = {:.2f} Hz\n'.format(fitresultlist[1])
            self.fitreport += "FWHM = {:.2f} Hz\n".format(2.35482 * fitresultlist[1])
            self.fitreport += "area = {:.2e}\n".format(fitresultlist[2])
            self.fitreport += "offset = {:.2e}\n".format(fitresultlist[3])
        elif Function_dict[fitfunction][0] == "dualGaussian":
            self.fitreport += "center0 = {:.2f} Hz\n".format(fitresultlist[0])
            self.fitreport += "FWHM0 = {:.2f} Hz\n".format(2.35482 * fitresultlist[1])
            self.fitreport += "area0 = {:.2e}\n".format(fitresultlist[2])
            self.fitreport += "center1 = {:.2f} Hz\n".format(fitresultlist[3])
            self.fitreport += "FWHM1 = {:.2f} Hz\n".format(2.35482 * fitresultlist[4])
            self.fitreport += "area1 = {:.2e}\n".format(fitresultlist[5])
            self.fitreport += "offset = {:.2e}\n".format(fitresultlist[6])

        if getresidual:
            self.fitreport += "sum of absolute residual = {:.2e}\n".format(
                self.residualval
            )
        if getchisq:
            self.fitreport += r"$\chi^2$ = {:.2e}\n".format(self.chisq)

        # self.fitreport = self.fitreport[:-1]
        self.fitrange = [ar[0], ar[1]]
        del ar, fitresultlist
        # End of generating fitreport
        ########################################################################

        if verbose:
            print(self.fitreport)
        return self.popt, tval * self.perr

    def FitPSDnoDict(
        self,
        fitfunction=None,  # 'Lorentzian' 'dualLorentzian' 'tribLorentzian' 'Gaussian 'dualGaussian' 'auto' 'Polyeven'
        inputfitparas=[],
        smooth=False,
        smoothlevel=1,
        fitrange=["auto", "auto"],
        alpha=0.05,
        getresidual=False,
        getchisq=False,
        verbose=False,
    ):

        # fitfunction = fitfunction.upper()

        self.fitflag = False
        self.popt = None
        self.pcov = None
        self.residual = None
        self.residualval = None
        self.chisq = None
        self.fitcurves = []
        ar = [0, len(self.PSD)]
        for i in range(len(fitrange)):
            if type(fitrange[i]) == int or type(fitrange[i]) == float:
                ar[i] = fitrange[i]

        fitparas = inputfitparas
        # numofparas = len(fitparas)
        # for i in range(len(inputfitparas)):
        #    if type(inputfitparas[i]) == int or type(inputfitparas[i]) == float:
        #        fitparas[i] = inputfitparas[i]

        try:
            if verbose:
                print("start fit")
            self.popt, self.pcov = scipy.optimize.curve_fit(
                f=fitfunction,
                xdata=self.frequencies[ar[0] : ar[1]],
                ydata=self.PSD[ar[0] : ar[1]],
                p0=fitparas,
                absolute_sigma=False,
            )
            self.fitflag = True
            if verbose:
                print("Obtained fit parameters ", [x for x in self.popt])
        except RuntimeError:
            print("fit RuntimeError")
        else:
            pass

        if self.fitflag:
            self.perr = np.sqrt(np.diag(self.pcov))
            dof = max(
                0, len(self.frequencies[ar[0] : ar[1]]) - len(self.popt)
            )  # degree of freedom
            # student-t value for the dof and confidence level
            tval = scipy.stats.distributions.t.ppf(1.0 - alpha / 2.0, dof)

            self.fitcurves.append(
                fitfunction(self.frequencies[ar[0] : ar[1]], self.popt[0])
            )  # self.popt[1]

            if getresidual:
                self.residual = self.PSD[ar[0] : ar[1]] - self.fitcurves[-1]
                self.residualval = np.sum(abs(self.residual))
            if getchisq:
                self.chisq = np.sum(
                    (self.PSD[ar[0] : ar[1]] - self.fitcurves[-1]) ** 2
                    / abs(self.fitcurves[-1])
                )
            self.fitreport = f" Fit ({(100 - 100 * alpha):.0f} % confidence level)\n"
            fitresultlist = []
            for i in range(len(self.popt)):
                fitresultlist.append(ufloat(self.popt[i], tval * self.perr[i]))
                self.fitreport += f"param{i} = {abs(fitresultlist[i]):.2f}"

            if getresidual:
                self.fitreport += "sum of absolute residual = {:.2e}\n".format(
                    self.residualval
                )
            if getchisq:
                self.fitreport += "chi-sq = {:.2e}\n".format(self.chisq)

            self.fitreport = self.fitreport[:-1]
            self.fitrange = [ar[0], ar[1]]
            del ar, fitresultlist
            if verbose:
                print(self.fitreport)
            return self.popt, tval * self.perr

    def FitFID(
        self,
        functionname="ExpCos",  # 'ExpCos','dualExpCos','tribExpCos'. 'auto' is not available yet
        inputfitparas=["auto", "auto", "auto", "auto", "auto"],
        smooth=False,
        smoothlevel=1,
        fitlen="auto",
        alpha=0.05,
        selectshot=[0],
        getresidual=False,
        getchisq=False,
        showplt=False,
        verbose=False,
    ):
        functionname = functionname.upper()

        self.FIDfitflag = False
        self.FIDpopt = None
        self.FIDpcov = None
        self.FIDresidual = None
        self.FIDresidualval = None
        self.FIDchisq = None
        self.FIDfitcurve = []

        self.timeStamp_s = np.linspace(
            start=0,
            stop=len(self.dataX) / self.samprate,
            num=len(self.dataX),
            endpoint=False,
            dtype=float,
        )

        ar = self.acq_arr[selectshot[0]].copy()
        ar[1] = ar[0] + int(self.samprate * 50e-3)
        # for i in range(len(fitrange)):
        if type(fitlen) == int:
            ar[1] = ar[0] + fitlen
        # if verbose:
        #     check(ar)
        # check(ar)
        # check(Functionname_dict[functionname][1])
        curvefunction = Function_dict[functionname][1]
        fitparas = Function_dict[functionname][2](
            s=self.dataX[ar[0] : ar[1]] + 1j * self.dataY[ar[0] : ar[1]],
            Lorpopt=self.popt,
            demodfreq=self.demodfreq,
        )
        for i, inputfitpara in enumerate(inputfitparas):
            if type(inputfitpara) == int or type(inputfitpara) == float:
                fitparas[i] = inputfitparas[i]

        # check(fitparas)
        try:
            if verbose:
                print("start " + Function_dict[functionname][0] + " fit")
            if Function_dict[functionname][0] in ["ExpCos", "dualExpCos", "tribExpCos"]:
                self.FIDpopt, self.FIDpcov = scipy.optimize.curve_fit(
                    curvefunction,
                    self.timeStamp_s[ar[0] : ar[1] : 20],
                    self.dataX[ar[0] : ar[1] : 20],
                    p0=fitparas,
                    absolute_sigma=False,
                )

            elif Function_dict[functionname][0] in [
                "ExpCosiSin",
                "dualExpCosiSin",
                "tribExpCosiSin",
            ]:
                # func = partial(curvefunction, t=self.timestamp[ar[0]:ar[1]], s=self.dataX[ar[0]:ar[1]]+1j*self.dataY[ar[0]:ar[1]])
                # check(list(fitparas))
                # self.FIDpopt, self.FIDpcov = scipy.optimize.curve_fit(
                #     curvefunction, self.timestamp[ar[0]:ar[1]], self.dataX[ar[0]:ar[1]]+1j*self.dataY[ar[0]:ar[1]] , p0=fitparas,
                #     absolute_sigma = False, method='lm')
                step = 4
                result = scipy.optimize.leastsq(
                    curvefunction,
                    x0=fitparas,
                    args=(
                        self.timeStamp_s[ar[0] : ar[1] : step],
                        np.array(
                            self.dataX[ar[0] : ar[1] : step]
                            + 1j * self.dataY[ar[0] : ar[1] : step]
                        ),
                    ),
                    maxfev=10000,
                    full_output=True,
                )
                self.FIDpopt = result[0]
                # check(self.timestamp[ar[0]:ar[1]:step].shape)
                check(self.FIDpopt)

            self.FIDfitflag = True
            if verbose:
                print(
                    "Obtained " + Function_dict[functionname][0] + " fit parameters ",
                    [x for x in self.popt],
                )
        except RuntimeError:
            print("fit RuntimeError")
        else:
            pass
        # del fitparas

        if self.FIDfitflag:

            if Function_dict[functionname][0] == "ExpCos":
                self.FIDfitcurve.append(
                    curvefunction(
                        self.timeStamp_s[ar[0] : ar[1]],
                        self.FIDpopt[0],
                        self.FIDpopt[1],
                        self.FIDpopt[2],
                        self.FIDpopt[3],
                        self.FIDpopt[-1],
                    )
                )
            elif Function_dict[functionname][0] == "ExpCosiSin":
                self.FIDfitcurve.append(
                    ExpCosiSin(
                        self.timeStamp_s[ar[0] : ar[1]],
                        self.FIDpopt[0],
                        self.FIDpopt[1],
                        self.FIDpopt[2],
                        self.FIDpopt[3],
                        self.FIDpopt[4],
                        self.FIDpopt[-1],
                    )
                )
            elif Function_dict[functionname][0] == "dualExpCos":
                self.FIDfitcurve.append(
                    ExpCos(
                        self.timeStamp_s[ar[0] : ar[1]],
                        self.FIDpopt[0],
                        self.FIDpopt[1],
                        self.FIDpopt[2],
                        self.FIDpopt[3],
                        self.FIDpopt[-1] / 2,
                    )
                )
                self.FIDfitcurve.append(
                    ExpCos(
                        self.timeStamp_s[ar[0] : ar[1]],
                        self.FIDpopt[4],
                        self.FIDpopt[5],
                        self.FIDpopt[6],
                        self.FIDpopt[7],
                        self.FIDpopt[-1] / 2,
                    )
                )
                self.FIDfitcurve.append(
                    dualExpCos(
                        self.timeStamp_s[ar[0] : ar[1]],
                        self.FIDpopt[0],
                        self.FIDpopt[1],
                        self.FIDpopt[2],
                        self.FIDpopt[3],
                        self.FIDpopt[4],
                        self.FIDpopt[5],
                        self.FIDpopt[6],
                        self.FIDpopt[7],
                        self.FIDpopt[8],
                    )
                )

            if getresidual or showplt:
                if "i" in Function_dict[functionname][0]:
                    self.FIDresidual = (
                        self.dataX[ar[0] : ar[1]]
                        + 1j * self.dataY[ar[0] : ar[1]]
                        - self.FIDfitcurve[-1]
                    )
                    self.FIDresidualval = np.sum(abs(self.FIDresidual))
                    self.FIDpcov = result[1] * np.var(
                        ExpCosiSinResidual(
                            params=self.FIDpopt,
                            t=self.timeStamp_s[ar[0] : ar[1]],
                            s=self.dataX[ar[0] : ar[1]]
                            + 1j * self.dataY[ar[0] : ar[1]],
                        )
                    )
                else:
                    self.FIDresidual = self.dataX[ar[0] : ar[1]] - self.FIDfitcurve[-1]
                    self.FIDresidualval = np.sum(abs(self.FIDresidual))

            if getchisq:
                self.FIDchisq = np.sum(
                    abs(self.dataX[ar[0] : ar[1]] - self.FIDfitcurve[-1]) ** 2
                    / abs(self.FIDfitcurve[-1])
                )

            self.FIDperr = np.sqrt(np.diag(self.FIDpcov))
            dof = max(0, abs(ar[1] - ar[0]) - len(self.FIDpopt))  # degree of freedom
            # student-t value for the dof and confidence level
            tval = scipy.stats.distributions.t.ppf(1.0 - alpha / 2.0, dof)

            # write fit report
            self.FIDfitreport = (
                Function_dict[functionname][0]
                + f" Fit ({100 - 100 * alpha:.0f} % confidence level)\n"
            )
            FIDfitresultlist = []
            for i, FIDpopti in enumerate(self.FIDpopt):
                FIDfitresultlist.append(ufloat(self.FIDpopt[i], tval * self.FIDperr[i]))

            FIDfitresultlist.append(
                ufloat(
                    1 / (np.pi * self.FIDpopt[1]),
                    abs(
                        1 / (np.pi * (self.FIDpopt[1] - tval * self.FIDperr[1]))
                        - 1 / (np.pi * self.FIDpopt[1])
                    ),
                )
            )
            if Function_dict[functionname][0] in ["ExpCos", "ExpCosiSin"]:
                self.FIDfitreport += f"Amp = {FIDfitresultlist[0]:.2e} V\n"
                self.FIDfitreport += f"T2* = {FIDfitresultlist[1]:.6f} s\n"
                self.FIDfitreport += f"linewidth = {FIDfitresultlist[-1]:.2f} Hz\n"
                self.FIDfitreport += f"nu = {FIDfitresultlist[2]:.3f} Hz\n"
                self.FIDfitreport += f"phi0 = {FIDfitresultlist[3]:.2f} \n"
                self.FIDfitreport += f"offset = {FIDfitresultlist[-1]:.2e} V\n"
            elif Function_dict[functionname][0] == "dualExpCos":
                self.FIDfitreport += f"Amp_0 = {FIDfitresultlist[0]:.2e} V\n"
                self.FIDfitreport += f"T2*_0 = {FIDfitresultlist[1]:.6f} s\n"
                self.FIDfitreport += (
                    f"linewidth_0 = {1/(np.pi*self.FIDpopt[1]):.2f} Hz\n"
                )
                self.FIDfitreport += f"nu_0 = {FIDfitresultlist[2]:.3f} Hz\n"
                self.FIDfitreport += f"phi0_0 = {FIDfitresultlist[3]:.2f} \n\n"
                self.FIDfitreport += f"Amp_1 = {FIDfitresultlist[4]:.2e} V\n"
                self.FIDfitreport += f"T2*_1 = {FIDfitresultlist[5]:.6f} s\n"
                self.FIDfitreport += (
                    f"linewidth_1 = {1/(np.pi*self.FIDpopt[5]):.2f} Hz\n"
                )
                self.FIDfitreport += f"nu_1 = {FIDfitresultlist[6]:.3f} Hz\n"
                self.FIDfitreport += f"phi0_1 = {FIDfitresultlist[7]:.2f} \n"
                self.FIDfitreport += f"offset = {FIDfitresultlist[-1]:.2e} V\n"

            if getresidual:
                self.FIDfitreport += (
                    f"sum of absolute residual = {self.FIDresidualval:.2e}\n"
                )
            if getchisq:
                self.FIDfitreport += f"chi-sq = {self.FIDchisq:.2e}\n"

            self.FIDfitreport = self.FIDfitreport[:-1]  # get rid of the last \n
            self.FIDfitrange = ar
            del ar, FIDfitresultlist
            if verbose:
                print(self.FIDfitreport)
            if showplt:
                fig = plt.figure(figsize=(16 * 0.8, 8 * 0.8))  #
                gs = gridspec.GridSpec(nrows=4, ncols=1)  #
                # fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
                #                     bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
                # make plot for time domain signal
                timedm_ax = fig.add_subplot(gs[0:3, 0])

                if "i" in Function_dict[functionname][0]:
                    timedm_ax.plot(
                        self.timeStamp_s[self.FIDfitrange[0] : self.FIDfitrange[1]],
                        np.real(self.FIDfitcurve[-1]),
                        c="tab:red",
                        label="fitcurve.real",
                    )
                    timedm_ax.plot(
                        self.timeStamp_s[self.FIDfitrange[0] : self.FIDfitrange[1]],
                        np.imag(self.FIDfitcurve[-1]),
                        c="tab:green",
                        label="fitcurve.imag\n" + self.FIDfitreport,
                    )
                else:
                    timedm_ax.plot(
                        self.timeStamp_s[self.FIDfitrange[0] : self.FIDfitrange[1]],
                        self.FIDfitcurve[-1],
                        c="tab:red",
                        label=self.FIDfitreport,
                    )
                    if Function_dict[functionname][0] == "dualExpCos":
                        timedm_ax.plot(
                            self.timeStamp_s[self.FIDfitrange[0] : self.FIDfitrange[1]],
                            self.FIDfitcurve[0],
                            "--",
                            c="tab:orange",
                            alpha=0.9,
                            label="ExpCos0",
                        )
                        timedm_ax.plot(
                            self.timeStamp_s[self.FIDfitrange[0] : self.FIDfitrange[1]],
                            self.FIDfitcurve[1],
                            "--",
                            c="tab:green",
                            alpha=0.9,
                            label="ExpCos1",
                        )

                # timedm_ax.plot(self.timestamp[self.FIDfitrange[0]:self.FIDfitrange[1]],
                #                curvefunction(
                #                    t=self.timestamp[self.FIDfitrange[0]:self.FIDfitrange[1]],
                #                    Amp=fitparas[0],
                #                    T2=fitparas[1],
                #                    nu=fitparas[2],
                #                    phi0=fitparas[3],
                #                     offset=fitparas[4],
                #                    ),
                #                c='tab:orange', label='estimated curve')
                timedm_ax.scatter(
                    self.timeStamp_s[self.FIDfitrange[0] : self.FIDfitrange[1]],
                    self.dataX[self.FIDfitrange[0] : self.FIDfitrange[1]],
                    c="tab:blue",
                    s=3,
                    label="dataX",
                )
                if "i" in Function_dict[functionname][0]:
                    timedm_ax.scatter(
                        self.timeStamp_s[self.FIDfitrange[0] : self.FIDfitrange[1]],
                        self.dataY[self.FIDfitrange[0] : self.FIDfitrange[1]],
                        c="tab:pink",
                        s=3,
                        label="dataY",
                    )
                timedm_ax.set_xlim(right=self.timeStamp_s[self.FIDfitrange[1]] + 0.03)
                timedm_ax.set_xlabel("time / s")
                timedm_ax.set_ylabel("Signal / V")
                timedm_ax.legend(loc="best")
                timedm_ax.grid()

                resi_ax = fig.add_subplot(gs[-1, 0])
                if "i" in Function_dict[functionname][0]:
                    resi_ax.plot(
                        self.timeStamp_s[self.FIDfitrange[0] : self.FIDfitrange[1]],
                        np.real(self.FIDresidual),
                        c="tab:blue",
                        label="residual.real",
                    )
                    resi_ax.plot(
                        self.timeStamp_s[self.FIDfitrange[0] : self.FIDfitrange[1]],
                        np.imag(self.FIDresidual),
                        c="tab:pink",
                        label="residual.imag",
                    )
                else:
                    resi_ax.plot(
                        self.timeStamp_s[self.FIDfitrange[0] : self.FIDfitrange[1]],
                        self.FIDresidual,
                        c="tab:blue",
                        label="residual",
                    )
                resi_ax.set_xlim(right=self.timeStamp_s[self.FIDfitrange[1]] + 0.05)
                resi_ax.set_xlabel("time / s")
                resi_ax.set_ylabel("Residual / V")
                resi_ax.legend(loc="best")
                resi_ax.set_xlim(right=self.timeStamp_s[self.FIDfitrange[1]] + 0.03)
                plt.show()

            return self.FIDpopt, tval * self.FIDperr

    def Hz2Index(self, freq_Hz):
        """
        Parameters
        ----------
        freq_Hz: float
            frequency in [Hz]

        Returns
        -------
        freq_index : int
            the index of the input frequency [Hz]

        raises ValueError if the input frequency value or the output index
        is out of range

        Examples
        -------
        Example of 7 elements
            index            0       1       2       3       4       5        6
            frequency [Hz]  0.1     0.2     0.3     0.4     0.5     0.6      0.7

            lenofdata = 7
            center_freq = self.frequencies[lenofdata//2] = 0.4 [Hz]
            self.freq_resol = 0.1 [Hz]

            input frequency: 0.6
            output index = lenofdata // 2 + int(np.round((freq_Hz - center_freq) / self.freq_resol))
                         = 7 // 2 + int( np.round( (0.6 - 0.4)/0.1 ) )
                         = 3 + 2
                         = 5

            input frequency: 0.66
            output index = lenofdata // 2 + int(np.round((freq_Hz - center_freq) / self.freq_resol))
                         = 7 // 2 + int( np.round( (0.66 - 0.4)/0.1 ) )
                         = 3 + 3
                         = 6

        Example of 8 elements
            index           0   1       2       3       4       5        6      7
            frequency[Hz]  0.1  0.2     0.3     0.4     0.5     0.6     0.7    0.8


            lenofdata = 8
            center_freq = self.frequencies[lenofdata//2] = 0.5 [Hz]
            self.freq_resol = 0.1 [Hz]

            input frequency: 0.6
            output index = lenofdata // 2 + int(np.round((freq_Hz - center_freq) / self.freq_resol))
                         = 8 // 2 + int( np.round( (0.6 - 0.5)/0.1 ) )
                         = 4 + 1
                         = 5

            input frequency: 0.66
            output index = lenofdata // 2 + int(np.round((freq_Hz - center_freq) / self.freq_resol))
                         = 8 // 2 + int( np.round( (0.66 - 0.5)/0.1 ) )
                         = 4 + 2
                         = 6

        """
        if freq_Hz < np.amin(self.frequencies) or freq_Hz > np.amax(self.frequencies):
            raise ValueError(
                f"[{self.Hz2Index.__name__}] frequency {freq_Hz:g} "
                + "out of range "
                + f"[{np.amin(self.frequencies):g}, {np.amax(self.frequencies):g}]. "
            )
        lenofdata = len(self.frequencies)
        center_freq = self.frequencies[lenofdata // 2]
        self.freq_resol_Hz = abs(
            self.frequencies[0] - self.frequencies[1]
        )  # frequency resolution
        freq_index = lenofdata // 2 + int(
            np.round((freq_Hz - center_freq) / self.freq_resol_Hz)
        )
        if freq_index < 0 or freq_index > (lenofdata - 1):
            raise ValueError(
                f"[{self.Hz2Index.__name__}] index out of range "
                + f"[0, {len(self.frequencies)-1:d}].  "
            )
        return freq_index

    def Index2Hz(self, freq_index):
        """
        Parameters
        ----------
        freq_index : int
            the index of the input frequency [Hz]

        Returns
        -------
        freq_index: float
            frequency in [Hz]

        raises ValueError if the inoput index or the output frequency value
        is out of range

        Examples
        -------
        Example of 7 elements
            index            0       1       2       3       4       5        6
            frequency [Hz]  0.1     0.2     0.3     0.4     0.5     0.6      0.7

            lenofdata = 7
            center_freq = self.frequencies[lenofdata//2] = 0.4 [Hz]
            self.freq_resol = 0.1 [Hz]freq_resol = 0.1 [Hz]

            input index: 3
            output frequency = center_freq + self.freq_resol * (freq_index - lenofdata // 2)
                         = 0.4 + 0.1 * (3 - 3)
                         = 0.4 + 0.0
                         = 0.4


        Example of 8 elements
            index           0   1       2       3       4       5        6      7
            frequency[Hz]  0.1  0.2     0.3     0.4     0.5     0.6     0.7    0.8

            lenofdata = 8
            center_freq = self.frequencies[lenofdata//2] = 0.5 [Hz]
            self.freq_resol = 0.1 [Hz]freq_resol = 0.1 [Hz]

            input index: 3
            output frequency = center_freq + self.freq_resol * (freq_index - lenofdata // 2)
                         = 0.5 + 0.1 * (3 - 4)
                         = 0.5 - 1.0
                         = 0.4

        """
        if freq_index < 0 or freq_index > (len(self.frequencies) - 1):
            raise ValueError(
                f"[{self.Index2Hz.__name__}] frequency index {freq_index:d} "
                + "out of range "
                + f"[0, {len(self.frequencies)-1:d}]. "
            )
        lenofdata = len(self.frequencies)
        center_freq = self.frequencies[lenofdata // 2]
        self.freq_resol_Hz = abs(
            self.frequencies[0] - self.frequencies[1]
        )  # frequency resolution
        freq_Hz = center_freq + self.freq_resol_Hz * (freq_index - lenofdata // 2)
        if freq_Hz < np.amin(self.frequencies) or freq_Hz > np.amax(self.frequencies):
            raise ValueError(
                f"[{self.Hz2Index.__name__}] frequency {freq_Hz:g} "
                + "out of range "
                + f"[{np.amin(self.frequencies):g}, {np.amax(self.frequencies):g}]. "
            )
        return freq_Hz

    def sgFilterPSD(
        self,
        window_length: int = 100,
        polyorder: int = 2,
        makeplot: bool = False,
        verbose: bool = False,
    ):
        assert hasattr(self, "PSD")
        sg = savgol_filter(self.PSD, window_length=window_length, polyorder=polyorder)
        if makeplot:
            fig = plt.figure(figsize=(10.0, 8.0), dpi=150)  # initialize a figure
            width_ratios = [1, 1]
            height_ratios = [1, 1, 2]
            gs = gridspec.GridSpec(
                nrows=3, ncols=2, width_ratios=width_ratios, height_ratios=height_ratios
            )  #
            # gs = gridspec.GridSpec(nrows=3, ncols=2)  #
            PSD_ax = fig.add_subplot(gs[0, :])
            # SG_ax = fig.add_subplot(gs[1,0], sharex=PSD_ax, sharey=PSD_ax)
            PSDsubSG_ax = fig.add_subplot(gs[1, :], sharex=PSD_ax, sharey=PSD_ax)
            PSD_hist_ax = fig.add_subplot(gs[2, 0])
            PSDsubSG_hist_ax = fig.add_subplot(gs[2, 1])

            PSD_ax.plot(self.frequencies, self.PSD, label="raw PSD")
            PSD_ax.plot(self.frequencies, sg, label="SG output")
            # SG_ax.plot(self.frequencies, sg, label='SG output')
            PSDsubSG_ax.plot(
                self.frequencies, self.PSD - sg, c="g", label="raw PSD - SG output"
            )

            for ax in [PSD_ax, PSDsubSG_ax]:  # SG_ax,
                ax.legend()
                ax.set_ylabel("PSD [$\\mathrm{V}^2/\\mathrm{Hz}$]")

            allmin = np.amin([np.amin(self.PSD), np.amin(sg), np.amin(self.PSD - sg)])
            allmax = np.amax([np.amax(self.PSD), np.amax(sg), np.amax(self.PSD - sg)])

            PSD_ax.set_ylim(min(-allmax * 0.1, allmin * 1.5), max(0, allmax * 1.5))
            PSDsubSG_ax.set_xlabel("Frequency [Hz]")

            def ChisqHistPlot(ax, data, title):
                """ """
                mean = np.mean(data)
                std = np.std(data)
                # check(np.amax(data))
                check(std)
                hist, bin_edges = np.histogram(data / std, bins="auto", density=False)
                binwidth = abs(bin_edges[1] - bin_edges[0])
                sumofcounts = len(data)
                hist_info = f"sum of counts={sumofcounts:d}\nmean={mean:.1e} [V^2/Hz] std={std:.1e} [V^2/Hz]"
                # if normalizebysigma:
                # bin_edges /= std
                ax.set_xlabel("PSD / $\\sigma_{\\mathrm{PSD}}$")
                # else:
                #     ax.set_xlabel('signal [V]')
                hist_x = []
                hist_y = []
                for i, count in enumerate(hist):
                    if count > 0:
                        hist_x.append((bin_edges[i] + bin_edges[i + 1]) / 2.0)
                        hist_y.append(count)
                ax.scatter(
                    hist_x,
                    hist_y,
                    color="tab:red",
                    edgecolors="k",
                    linewidths=1,
                    marker="o",
                    s=6,
                    zorder=6,
                    label="histogram",
                )

                # plot pdf of gaussian distribution (mean = 0, std = 1)
                xstamp = np.linspace(
                    start=0,
                    stop=np.amax(bin_edges),
                    num=max(100, len(bin_edges)),
                    endpoint=True,
                )
                # ax.plot(xstamp, norm.pdf(xstamp, mean, 1), label='Gaussian pdf', linestyle='--')
                ax.plot(
                    xstamp,
                    2 * sumofcounts * binwidth * chi2.pdf(2 * xstamp, df=2),
                    label="$\\chi^2$ Chi-sq pdf\ndof=2",
                    linestyle="--",
                )
                # sumofcounts * binwidth *
                # ax.plot(xstamp, sumofcounts * binwidth * chi2.pdf(xstamp, df=1), label='Chi-sq pdf dof=1', linestyle='--')
                ax.set_ylabel("count")
                ax.set_title(title + "\n" + hist_info)
                ax.set_xscale("linear")
                ax.set_yscale("log")
                ax.set_ylim(bottom=0.1)
                ax.legend(loc="best")
                ax.grid(True)
                del (
                    mean,
                    std,
                    hist,
                    bin_edges,
                    sumofcounts,
                    hist_info,
                    hist_x,
                    hist_y,
                    xstamp,
                )

            ChisqHistPlot(PSD_hist_ax, data=self.PSD, title="raw PSD histogram")
            ChisqHistPlot(
                PSDsubSG_hist_ax,
                data=self.PSD - sg,
                title="(raw PSD - SG) histogram",
            )

            fig.tight_layout()
            plt.show()

        return sg

    def psdMovAvgByStep(self, weights=None, step_len: int = 1, verbose: bool = False):
        self.frequencies, self.PSD = MovAvgByStep(
            xstamp=self.frequencies,
            rawsignal=self.PSD,
            weights=weights,
            step_len=step_len,
            verbose=verbose,
        )
        self.freq_resol_Hz = step_len * self.freq_resol_Hz

    def getSpinNoisePSDsub(
        self,
        windowfunction="rectangle",  # Hanning, Hamming, Blackman
        chunksize=33e-3,  # in second
        analysisrange=[0, -1],
        showstd=True,
        stddev_range=[35e3, 37e3],
        ploycorrparas=[],
        interestingfreq_list=[],
        selectshot=[],
        verbose=False,
    ):
        print("This function is out of date. Please use GetNoPulsePSD instead. ")
        self.chunksize = chunksize
        chunklen = int(chunksize * self.samprate)
        data_start = analysisrange[0]
        data_end = len(self.dataX) + analysisrange[1]
        numofchunk = (data_end - data_start) // chunklen

        acq_arr0 = data_start + chunklen * np.linspace(
            start=0, stop=numofchunk, num=numofchunk, endpoint=False, dtype=int
        )
        acq_arr1 = acq_arr0 + chunklen
        acq_arr = np.array([acq_arr0, acq_arr1]).transpose()
        del data_start, data_end, acq_arr0, acq_arr1
        if verbose:
            print("acq_arr.shape", acq_arr.shape)
        PSD = np.zeros(chunklen)
        # stdofPSD = np.array([np.linspace(1, numofpulse, numofpulse, endpoint=True), np.zeros(numofpulse)])
        i = 0
        frequencies, singlePSD = stdLIAPSD(
            data_x=self.dataX[acq_arr[i, 0] : acq_arr[i, 1]],
            data_y=self.dataY[acq_arr[i, 0] : acq_arr[i, 1]],
            samprate=self.samprate,  # in Hz
            demodfreq=self.demodfreq,  # in Hz
            attenuation=self.attenuation,  # in dB. Power ratio (10^(attenuation/10))
            windowfunction=windowfunction,  # Hanning, Hamming, Blackman
            DTRCfilter=self.filterstatus,
            DTRCfilter_TC=self.DTRC_TC,
            DTRCfilter_order=self.DTRC_order,
        )
        correction = np.zeros_like(frequencies)
        # print('np.average(correction) ', np.average(correction))
        if len(ploycorrparas) != 0:
            correction = PolyEven(
                frequencies,
                ploycorrparas[0],
                ploycorrparas[1],
                ploycorrparas[2],
                ploycorrparas[3],
                ploycorrparas[4],
                ploycorrparas[5],
                self.demodfreq,
            )
            # print('np.average(correction) ', np.average(correction))
        self.interestingfreqvalue_list = []
        self.interestingfreqindex_list = []
        self.interestingPSDvalue_list = []

        if interestingfreq_list is not None and len(interestingfreq_list) != 0:
            numofintfreq = range(len(interestingfreq_list))
            if verbose:
                print("Analysis of single-frequency-channel is activated. ")
            freqresol = abs(frequencies[0] - frequencies[1])  # frequency resolution
            for targetfreq in interestingfreq_list:
                if verbose:
                    print("frequency resolution %g Hz" % freqresol)
                    print(
                        "np.amin(abs(frequencies-targetfreq)) ",
                        np.amin(abs(frequencies - targetfreq)),
                    )
                if np.amin(abs(frequencies - targetfreq)) > freqresol:
                    raise ValueError(
                        "interesting frequency %g Hz is not included in the spectrum"
                        % (targetfreq)
                    )
                index = np.argmin(abs(frequencies - targetfreq))
                self.interestingfreqindex_list.append(index)
                self.interestingfreqvalue_list.append(frequencies[index])
                self.interestingPSDvalue_list.append([])
            del singlePSD, freqresol, index

        if showstd:
            stdstart = np.argmin(abs(frequencies - stddev_range[0]))
            stdend = np.argmin(abs(frequencies - stddev_range[1]))
            if abs(stdstart - stdend) <= 10:
                print(
                    "abs(stdstart-stdend)<=10\ntoo less data for computing standard deviation"
                )
        self.stdofPSD_list = []
        # self.stdofASD = []
        if len(selectshot) == 0:
            chunk_list = range(numofchunk)
        else:
            chunk_list = selectshot

        numofchunk = len(chunk_list)
        for i in chunk_list:
            frequencies, singlePSD = stdLIAPSD(
                data_x=self.dataX[acq_arr[i, 0] : acq_arr[i, 1]],
                data_y=self.dataY[acq_arr[i, 0] : acq_arr[i, 1]],
                samprate=self.samprate,  # in Hz
                demodfreq=self.demodfreq,  # in Hz
                attenuation=self.attenuation,  # in dB. Power ratio (10^(attenuation/10))
                windowfunction=windowfunction,  # Hanning, Hamming, Blackman
                DTRCfilter=self.filterstatus,
                DTRCfilter_TC=self.DTRC_TC,
                DTRCfilter_order=self.DTRC_order,
            )

            PSD += singlePSD - correction

            if showstd:
                self.stdofPSD_list.append(np.std(PSD[stdstart:stdend]) / (i + 1))
                # self.stdofASD.append(np.std(np.sqrt(PSD[stdstart:stdend]/ (i + 1)) ) )
            if interestingfreq_list is not None and len(interestingfreq_list) != 0:
                for i in numofintfreq:
                    (self.interestingPSDvalue_list[i]).append(
                        singlePSD[self.interestingfreqindex_list[i]]
                    )
        del singlePSD
        self.exptype = "Spin Noise Measurement"
        self.frequencies = frequencies
        self.PSD = PSD / numofchunk  # averaged PSD in V^2/Hz
        del acq_arr, frequencies, PSD
        self.interestingfreqindex_list = np.array(self.interestingfreqindex_list)
        self.interestingfreqvalue_list = np.array(self.interestingfreqvalue_list)
        self.interestingPSDvalue_list = np.array(self.interestingPSDvalue_list)
        self.correction = correction
        if verbose:
            print("self.interestingfreqindex_arr ", self.interestingfreqindex_list)
            print("self.interestingfreqvalue_arr ", self.interestingfreqvalue_list)
            print(
                "shape of self.interestingfreqindex_arr",
                (self.interestingPSDvalue_list).shape,
            )

    @record_runtime_YorN(RECORD_RUNTIME)
    def getNoPulsePSD(
        self,
        windowfunction="rectangle",
        decayfactor=-10,
        chunksize=None,  # sec
        analysisrange=[0, -1],
        getstd=False,
        stddev_range=None,
        polycorrparas=[],
        interestingfreq_list=[],
        selectshots=[],
        verbose=False,
    ):
        """
        Compute PSD of a no-pulse measurement.

        Parameters
        ----------
        windowfunction : str, optional
            window function for FFT.
            Available choices:
                'rectangle'
                'expdecay'
                'Hanning' or 'hanning' or 'Han' or 'han'
                'Hamming' or 'hamming'
                'Blackman'
            Defaults to 'rectangle'.

        decayfactor : float, optional
            parameter for window function 'expdecay'.
            The 'expdecay' window function generates an array of [exp(-df*0), exp(-df*1), ... , exp(-df*(num-1))],
            where df stands for decayfactor.
            Defaults to -10.

        chunksize : float, optional
            The size of a single chunk for computing one power spectrum in [s].
            This size determines how the time-series is sliced.
            You can set it to None or a very large chunksize so that you use the whole time-series.
            Defaults to None.

        analysisrange
            The range of time-series index that are analyzed.
            Defaults to [0,-1], which means all time-series ([0:-1]) are included.

        getstd : bool, optional
            The option whether to get standard deviation in stddev_range.
            Defaults to False.

        stddev_range : list, optional
            The frequency range in [Hz] from which the standard deviation is computed.
            Notice that this range means the absolute frequency range. e.g. [1.346e6 - 100, 1.346e6 + 200]
            Defaults to None. When it is None, the standard deviation would not be computed.

        polycorrparas : list, optional
            Parameters for a even-order polynomial correcting (subtracting) the power spectrum.
            check utils - PolyEven() for more details.
            Defaults to [], which means no correction would be made.

        interestingfreq_list : list, optional
            List of interesting frequencies, in [Hz].
            Usually users choose the frequencies near the Larmore frequency to see its amplitude over time.
            Defaults to [].

        selectshots : list, optional
            List of selected shots / chunks of acquired time-series for power spectra.
            e.g. [0] means that you only use the first shot.
            Defaults to [], which means all shots / chunks are included.

        verbose : bool, optional
            Defaults to False.

        Returns
        -------
        Null

        """

        # determine the chunksize
        # if the input chunksize is None or chunksize is larger than the whole time-series
        if chunksize is None or chunksize >= len(self.dataX) / self.samprate:
            chunklen = len(self.dataX)
            self.chunksize = chunklen / self.samprate
        else:
            self.chunksize = chunksize
            chunklen = int(chunksize * self.samprate)
        # check(chunklen)
        # determine the range of time-series for analysis and number of chunks
        data_start = analysisrange[0]
        data_end = len(self.dataX) + analysisrange[1]
        numofchunk = (data_end - data_start + 1) // chunklen
        # Notice that '//' means there could be some data cannot be used after the last chunk.
        # check(numofchunk)

        # determine the acquisition (chunks)
        acq_arr0 = data_start + chunklen * np.linspace(
            start=0, stop=numofchunk, num=numofchunk, endpoint=False, dtype=int
        )
        acq_arr1 = acq_arr0 + chunklen
        acq_arr = np.array([acq_arr0, acq_arr1]).transpose()
        del data_start, data_end, acq_arr0, acq_arr1
        if verbose:
            print("acq_arr.shape", acq_arr.shape)

        # Initialize PSD array. the length of PSD is the length of a chunk.
        PSD = np.zeros(chunklen)

        # Generate the frequency axis of the power spectrum
        frequencies = np.sort(
            np.fft.fftfreq(chunklen, d=1.0 / self.samprate) + self.demodfreq
        )

        # determine the correction (substraction) for PSD
        correction = np.zeros_like(frequencies)
        if polycorrparas is not None and len(polycorrparas) == 6:
            correction = PolyEven(
                frequencies,
                polycorrparas[0],
                polycorrparas[1],
                polycorrparas[2],
                polycorrparas[3],
                polycorrparas[4],
                polycorrparas[5],
                self.demodfreq,
            )

        # initialize lists of
        self.interestingfreqvalue_list = []  # interesting frequency values in [Hz]
        self.interestingfreqindex_list = []  # interesting frequency indexes
        self.interestingPSDvalue_list = []  # PSD of interesting frequencies

        # Store information of interesting frequencies
        if interestingfreq_list is not None and len(interestingfreq_list) != 0:
            numofintfreq = range(len(interestingfreq_list))
            if verbose:
                print("Analysis of single-frequency-bin is activated. ")
            freqresol = abs(frequencies[0] - frequencies[1])  # frequency resolution
            if verbose:
                check(freqresol)
            for intfreq in interestingfreq_list:
                if verbose:
                    check(np.amin(abs(frequencies - intfreq)))
                if np.amin(abs(frequencies - intfreq)) > freqresol:
                    raise ValueError(
                        f"interesting frequency {intfreq:g} [Hz] is not included in the spectrum"
                    )
                index = np.argmin(abs(frequencies - intfreq))
                self.interestingfreqindex_list.append(index)
                self.interestingfreqvalue_list.append(frequencies[index])
                self.interestingPSDvalue_list.append([])
            del freqresol, index

        # determine the range for computing the standard deviation
        if getstd:
            if stddev_range == []:
                stdstart = np.argmin(abs(frequencies - frequencies[0]))
                stdend = np.argmin(abs(frequencies - frequencies[-1]))
            else:
                stdstart = np.argmin(abs(frequencies - stddev_range[0]))
                stdend = np.argmin(abs(frequencies - stddev_range[-1]))
            if abs(stdstart - stdend) <= 10:
                print(
                    "WARNING!! abs(stdstart-stdend)<=10\ntoo less data for computing standard deviation"
                )

        # initialize the list of standard deviations
        # This list includes standard deviations computed during the accumulation of power spectra.
        # The process is like:
        # 1. Get the first power spectrum, then get one standard deviation.
        # 2. Get the second power spectrum, accumulate it onto the first one (plus).
        # Then compute the (standard deviation of the accumulated power spectrum) / 2.
        # Append this value to the list.
        # 3. Repeat step 2, until the end.
        self.stdofPSD_list = []

        # find the selected shots that are going to be used.
        if selectshots is None or len(selectshots) == 0:
            chunk_list = range(numofchunk)
        else:
            if np.amax(selectshots) + 1 > numofchunk:
                raise ValueError("np.amax(selectshots) + 1 > numofchunk. ")
            chunk_list = selectshots
        # check(chunk_list)
        # Compute power spectra for selected chunks.
        for i in chunk_list:
            frequencies, singlePSD = stdLIAPSD(
                data_x=self.dataX[acq_arr[i, 0] : acq_arr[i, 1]],
                data_y=self.dataY[acq_arr[i, 0] : acq_arr[i, 1]],
                samprate=self.samprate,
                demodfreq=self.demodfreq,
                attenuation=self.attenuation,
                windowfunction=windowfunction,
                decayfactor=decayfactor,
                DTRCfilter=self.filterstatus,
                DTRCfilter_TC=self.DTRC_TC,
                DTRCfilter_order=self.DTRC_order,
            )
            PSD += singlePSD - correction

            # Compute standard deviation
            if getstd:
                # self.stdofPSD_list.append(np.std(PSD[stdstart:stdend]) / (i + 1))
                self.stdofPSD_list.append(np.std(singlePSD[stdstart:stdend]))
            # Store PSD values of interesting frequencies
            if interestingfreq_list is not None and len(interestingfreq_list) != 0:
                for freqindex in self.interestingfreqindex_list:
                    (self.interestingPSDvalue_list[i]).append(singlePSD[freqindex])
            # delete singlePSD before going into the next iteration
            del singlePSD

        # store information
        self.frequencies = frequencies
        self.freq_resol_Hz = abs(
            frequencies[0] - frequencies[1]
        )  # frequency resolution
        self.PSD = PSD / len(chunk_list)  # averaged PSD in V^2/Hz
        self.acq_arr = acq_arr
        del acq_arr, frequencies, PSD
        self.correction = correction
        self.selectPulse_list = chunk_list  # This is for plotting in self.GetSpectrum()
        # if verbose:
        #     self.interestingfreqindex_arr=np.array(self.interestingfreqindex_list)
        #     self.interestingfreqvalue_arr=np.array(self.interestingfreqvalue_list)
        #     self.interestingPSDvalue_arr=np.array(self.interestingPSDvalue_list)
        #     print('self.interestingfreqindex_arr ', self.interestingfreqindex_arr)
        #     print('self.interestingfreqvalue_arr ', self.interestingfreqvalue_arr)
        #     print('shape of self.interestingfreqindex_arr', (self.interestingPSDvalue_arr).shape)

    @record_runtime_YorN(RECORD_RUNTIME)
    def getNoPulseFFTandPSD(
        self,
        windowfunction="rectangle",
        decayfactor=-10,
        chunksize=None,
        analysisrange=[0, -1],
        getstd=False,
        stddev_range=None,
        polycorrparas=[],
        interestingfreq_list=[],
        selectshots=[],
        verbose=False,
    ):
        """
        Compute PSD of a no-pulse measurement.

        Parameters
        ----------
        windowfunction : str, optional
            window function for FFT.
            Available choices:
                'rectangle'
                'expdecay'
                'Hanning' or 'hanning' or 'Han' or 'han'
                'Hamming' or 'hamming'
                'Blackman'
            Defaults to 'rectangle'.

        decayfactor : float, optional
            parameter for window function 'expdecay'.
            The 'expdecay' window function generates an array of [exp(-df*0), exp(-df*1), ... , exp(-df*(num-1))],
            where df stands for decayfactor.
            Defaults to -10.

        chunksize : float, optional
            The size of a single chunk for computing one power spectrum in [s].
            This size determines how the time-series is sliced.
            You can set it to None or a very large chunksize so that you use the whole time-series.
            Defaults to None.

        analysisrange
            The range of time-series index that are analyzed.
            Defaults to [0,-1], which means all time-series ([0:-1]) are included.

        getstd : bool, optional
            The option whether to get standard deviation in stddev_range.
            Defaults to False.

        stddev_range : list, optional
            The frequency range in [Hz] from which the standard deviation is computed.
            Notice that this range means the absolute frequency range. e.g. [1.346e6 - 100, 1.346e6 + 200]
            Defaults to None. When it is None, the standard deviation would not be computed.

        polycorrparas : list, optional
            Parameters for a even-order polynomial correcting (subtracting) the power spectrum.
            check utils - PolyEven() for more details.
            Defaults to [], which means no correction would be made.

        interestingfreq_list : list, optional
            List of interesting frequencies, in [Hz].
            Usually users choose the frequencies near the Larmore frequency to see its amplitude over time.
            Defaults to [].

        selectshots : list, optional
            List of selected shots / chunks of acquired time-series for power spectra.
            e.g. [0] means that you only use the first shot.
            Defaults to [], which means all shots / chunks are included.

        verbose : bool, optional
            Defaults to False.

        Returns
        -------
        Null

        """
        if not hasattr(self, "avgFFT"):
            self.getNoPulseFFT(
                windowfunction=windowfunction,
                decayfactor=decayfactor,
                chunksize=chunksize,
                analysisrange=analysisrange,
                getstd=getstd,
                stddev_range=stddev_range,
                polycorrparas=polycorrparas,
                interestingfreq_list=interestingfreq_list,
                selectshots=selectshots,
                verbose=verbose,
            )
        self.PSD = np.abs(self.FFT) ** 2.0

    @record_runtime_YorN(RECORD_RUNTIME)
    def getNoPulseFFT(
        self,
        windowfunction: str = "rectangle",
        decayfactor: float = -10,
        chunksize: float = None,
        analysisrange: list = [0, -1],
        getstd: bool = False,
        stddev_range: list = None,
        polycorrparas: list = [],
        interestingfreq_list: list = [],
        selectshots: list = [],
        verbose: bool = False,
    ):
        """
        Compute FFT of a (usually no-pulse) measurement.

        Parameters
        ----------
        windowfunction : str, optional
            window function for FFT.
            Available choices:
                'rectangle'
                'expdecay'
                'Hanning' or 'hanning' or 'Han' or 'han'
                'Hamming' or 'hamming'
                'Blackman'
            Defaults to 'rectangle'.

        decayfactor : float, optional
            parameter for window function 'expdecay'.
            The 'expdecay' window function generates an array of [exp(-df*0), exp(-df*1), ... , exp(-df*(num-1))],
            where df stands for decayfactor.
            Defaults to -10.

        chunksize : float, optional
            The size of a single chunk for computing one power spectrum in [s].
            This size determines how the time-series is sliced.
            You can set it to None or a very large chunksize so that you use the whole time-series.
            Defaults to None.

        analysisrange
            The range of time-series index that are analyzed.
            Defaults to [0,-1], which means all time-series ([0:-1]) are included.

        getstd : bool, optional
            The option whether to get standard deviation in stddev_range.
            Defaults to False.

        stddev_range : list, optional
            The frequency range in [Hz] from which the standard deviation is computed.
            Notice that this range means the absolute frequency range. e.g. [1.346e6 - 100, 1.346e6 + 200]
            Defaults to None. When it is None, the standard deviation would not be computed.

        polycorrparas : list, optional
            Parameters for a even-order polynomial correcting (subtracting) the power spectrum.
            check utils - PolyEven() for more details.
            Defaults to [], which means no correction would be made.

        interestingfreq_list : list, optional
            List of interesting frequencies, in [Hz].
            Usually users choose the frequencies near the Larmore frequency to see its amplitude over time.
            Defaults to [].

        selectshots : list, optional
            List of selected shots / chunks of acquired time-series for power spectra.
            e.g. [0] means that you only use the first shot.
            Defaults to [], which means all shots / chunks are included.

        verbose : bool, optional
            Defaults to False.

        Returns
        -------
        Null

        """

        # determine the chunksize
        # if the input chunksize is None or chunksize is larger than the whole time-series
        if chunksize is None or chunksize >= len(self.dataX) / self.samprate:
            chunklen = len(self.dataX)
            self.chunksize = chunklen / self.samprate
        else:
            self.chunksize = chunksize
            chunklen = int(chunksize * self.samprate)
        # check(chunklen)
        # determine the range of time-series for analysis and number of chunks
        data_start = analysisrange[0]
        data_end = len(self.dataX) + analysisrange[1]
        numofchunk = (data_end - data_start + 1) // chunklen
        # Notice that '//' means there could be some data cannot be used after the last chunk.
        # check(numofchunk)

        # determine the acquisition (chunks)
        acq_arr0 = data_start + chunklen * np.linspace(
            start=0, stop=numofchunk, num=numofchunk, endpoint=False, dtype=int
        )
        acq_arr1 = acq_arr0 + chunklen
        acq_arr = np.array([acq_arr0, acq_arr1]).transpose()
        del data_start, data_end, acq_arr0, acq_arr1
        if verbose:
            print("acq_arr.shape", acq_arr.shape)

        # Initialize FFT array. The length of FFT is the length of a chunk.
        FFT = np.zeros(chunklen, dtype=complex)

        # Generate the frequency axis of the power spectrum
        frequencies = np.sort(
            np.fft.fftfreq(chunklen, d=1.0 / self.samprate) + self.demodfreq
        )
        self.freq_resol_Hz = abs(
            frequencies[0] - frequencies[1]
        )  # frequency resolution

        # determine the correction (substraction) for FFT
        correction = np.zeros_like(frequencies)
        if polycorrparas is not None and len(polycorrparas) == 6:
            correction = PolyEven(
                frequencies,
                polycorrparas[0],
                polycorrparas[1],
                polycorrparas[2],
                polycorrparas[3],
                polycorrparas[4],
                polycorrparas[5],
                self.demodfreq,
            )

        # initialize lists of
        self.interestingfreqvalue_list = []  # interesting frequency values in [Hz]
        self.interestingfreqindex_list = []  # interesting frequency indexes
        self.interestingFFTvalue_list = []  # FFT of interesting frequencies

        # Store information of interesting frequencies
        if interestingfreq_list is not None and len(interestingfreq_list) != 0:
            numofintfreq = range(len(interestingfreq_list))
            if verbose:
                print("Analysis of single-frequency-bin is activated. ")
            freqresol = abs(frequencies[0] - frequencies[1])  # frequency resolution
            if verbose:
                check(freqresol)
            for intfreq in interestingfreq_list:
                if verbose:
                    check(np.amin(abs(frequencies - intfreq)))
                if np.amin(abs(frequencies - intfreq)) > freqresol:
                    raise ValueError(
                        f"interesting frequency {intfreq:g} [Hz] is not included in the spectrum"
                    )
                index = np.argmin(abs(frequencies - intfreq))
                self.interestingfreqindex_list.append(index)
                self.interestingfreqvalue_list.append(frequencies[index])
                self.interestingFFTvalue_list.append([])
            del freqresol, index

        # determine the range for computing the standard deviation
        if getstd:
            stdstart = np.argmin(abs(frequencies - stddev_range[0]))
            stdend = np.argmin(abs(frequencies - stddev_range[1]))
            if abs(stdstart - stdend) <= 10:
                print(
                    "WARNING!! abs(stdstart-stdend)<=10\ntoo less data for computing standard deviation"
                )

        # initialize the list of standard deviations
        # This list includes standard deviations computed during the accumulation of power spectra.
        # The process is like:
        # 1. Get the first power spectrum, then get one standard deviation.
        # 2. Get the second power spectrum, accumulate it onto the first one (plus).
        # Then compute the (standard deviation of the accumulated power spectrum) / 2.
        # Append this value to the list.
        # 3. Repeat step 2, until the end.
        self.stdofFFT_list = []

        # find the selected shots that are going to be used.
        if selectshots is None or len(selectshots) == 0:
            chunk_list = range(numofchunk)
        else:
            if np.amax(selectshots) + 1 > numofchunk:
                raise ValueError("np.amax(selectshots) + 1 > numofchunk. ")
            chunk_list = selectshots

        # Compute power spectra for selected chunks.
        for i in chunk_list:
            frequencies, singleFFT = stdLIAFFT(
                data_x=self.dataX[acq_arr[i, 0] : acq_arr[i, 1]],
                data_y=self.dataY[acq_arr[i, 0] : acq_arr[i, 1]],
                samprate=self.samprate,
                demodfreq=self.demodfreq,
                attenuation=self.attenuation,
                windowfunction=windowfunction,
                decayfactor=decayfactor,
                DTRCfilter=self.filterstatus,
                DTRCfilter_TC=self.DTRC_TC,
                DTRCfilter_order=self.DTRC_order,
            )
            FFT += singleFFT  # -correction

            # Compute standard deviation
            if getstd:
                self.stdofFFT_list.append(np.std(FFT[stdstart:stdend]) / (i + 1))
            # Store FFT values of interesting frequencies
            if interestingfreq_list is not None and len(interestingfreq_list) != 0:
                for freqindex in self.interestingfreqindex_list:
                    (self.interestingFFTvalue_list[i]).append(singleFFT[freqindex])
            # delete singleFFT before going into the next iteration
            del singleFFT

        # store information
        self.frequencies = frequencies
        self.FFT = FFT / len(chunk_list)  # averaged FFT in V/sqrt(Hz)
        self.acq_arr = acq_arr
        del acq_arr, frequencies, FFT
        self.correction = correction
        self.selectPulse_list = chunk_list  # This is for plotting in self.GetSpectrum()

    def MonoFreqAnalysis(
        self,
        showtimedomain=True,
        showfreqdomain=True,
        specscale="log",
        left_spc=0.1,
        top_spc=1 - 0.1,
        right_spc=1 - 0.05,
        bottom_spc=0.1,
        xgrid_spc=0.3,
        ygrid_spc=0.2,
        # vlinex = None,
        showplt_opt=True,
        saveplt_opt=False,
        saveplt_path=None,
        pltmsg="",
        verbose=False,
    ):
        """
        Not in use

        Args:
            showtimedomain : bool, optional Defaults to True.
            showfreqdomain : bool, optional Defaults to True.
            specscale : str, optional Defaults to 'log'.
            left_spc (float, optional Defaults to 0.1.
            top_spc : type_, optional Defaults to 1-0.1.
            right_spc : type_, optional Defaults to 1-.05.
            bottom_spc (float, optional Defaults to .1.
            xgrid_spc (float, optional Defaults to .3.
            ygrid_spc (float, optional Defaults to .2.
            showplt_opt : bool, optional Defaults to True.
            saveplt_opt : bool, optional Defaults to False.
            saveplt_path : type_, optional Defaults to None.
            pltmsg : str, optional Defaults to ''.
            verbose : bool, optional Defaults to False.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        for i in range(len(self.interestingfreqvalue_list)):
            if verbose:
                # print('timestamp.shape ', timestamp.shape)
                print(
                    "self.interestingPSDvalue_arr[i].shape",
                    self.interestingPSDvalue_list[i].shape,
                )
            if showtimedomain and (not showfreqdomain):
                fig = plt.figure(figsize=(14, 9))  #
                gs = gridspec.GridSpec(nrows=1, ncols=1)  #
                fig.subplots_adjust(
                    left=left_spc,
                    top=top_spc,
                    right=right_spc,
                    bottom=bottom_spc,
                    wspace=xgrid_spc,
                    hspace=ygrid_spc,
                )
                # make plot for time domain signal
                timedm_ax = fig.add_subplot(gs[0, 0])
                timestamp = np.linspace(
                    start=0,
                    stop=len(self.interestingPSDvalue_list[i]) * self.chunksize,
                    num=len(self.interestingPSDvalue_list[i]),
                    endpoint=False,
                )
                timedm_ax.plot(
                    timestamp,
                    self.interestingPSDvalue_list[i],
                    label="%5.7g MHz" % (1e-6 * self.interestingfreqvalue_list[i]),
                    c="tab:green",
                )
                timedm_ax.set_xlabel("time / s")
                timedm_ax.set_ylabel("Power spectral density / $V^2/Hz$")
                # timedm_ax.set_xlim(timestamp[0], timestamp[-1])
                timedm_ax.legend(loc="upper right")
            elif (not showtimedomain) and showfreqdomain:
                fig = plt.figure(figsize=(14, 9))  #
                gs = gridspec.GridSpec(nrows=1, ncols=1)  #
                fig.subplots_adjust(
                    left=left_spc,
                    top=top_spc,
                    right=right_spc,
                    bottom=bottom_spc,
                    wspace=xgrid_spc,
                    hspace=ygrid_spc,
                )
                # make plot for frequency domain signal
                freqdm_ax = fig.add_subplot(gs[0, 0])
                frequencies, PSD = stdPSD(
                    data=self.interestingPSDvalue_list[i],
                    samprate=1 / self.chunksize,
                    windowfunction="rectangle",  # Hanning, Hamming, Blackman
                    verbose=False,
                )
                freqdm_ax.plot(
                    frequencies,
                    PSD,
                    label="%5.7g MHz" % (1e-6 * self.interestingfreqvalue_list[i]),
                    c="tab:blue",
                )
                freqdm_ax.set_xscale(specscale)
                freqdm_ax.set_yscale(specscale)
                freqdm_ax.set_xlabel("frequency / Hz")
                freqdm_ax.set_ylabel("PSD of PSD amplitude / $V^{4}/Hz^{3}$")
                freqdm_ax.legend(loc="upper right")
            elif showtimedomain and showfreqdomain:
                fig = plt.figure(figsize=(21, 9))  #
                gs = gridspec.GridSpec(nrows=1, ncols=2)  #
                fig.subplots_adjust(
                    left=left_spc,
                    top=top_spc,
                    right=right_spc,
                    bottom=bottom_spc,
                    wspace=xgrid_spc,
                    hspace=ygrid_spc,
                )
                # make plot for time domain signal
                timedm_ax = fig.add_subplot(gs[0, 0])
                timestamp = np.linspace(
                    start=0,
                    stop=len(self.interestingPSDvalue_list[i]) * self.chunksize,
                    num=len(self.interestingPSDvalue_list[i]),
                    endpoint=False,
                )
                timedm_ax.plot(
                    timestamp,
                    self.interestingPSDvalue_list[i],
                    label="%5.7g MHz" % (1e-6 * self.interestingfreqvalue_list[i]),
                    c="tab:green",
                )
                timedm_ax.set_xlabel("time / s")
                timedm_ax.set_ylabel("Power spectral density / $V^2/Hz$")
                # timedm_ax.set_xlim(timestamp[0], timestamp[-1])
                timedm_ax.legend(loc="upper right")
                # make plot for frequency domain signal
                freqdm_ax = fig.add_subplot(gs[0, 1])
                frequencies, PSD = stdPSD(
                    data=self.interestingPSDvalue_list[i],
                    samprate=1 / self.chunksize,
                    windowfunction="rectangle",  # Hanning, Hamming, Blackman
                    verbose=False,
                )
                freqdm_ax.plot(
                    frequencies,
                    PSD,
                    label="%5.7g MHz" % (1e-6 * self.interestingfreqvalue_list[i]),
                    c="tab:blue",
                )
                freqdm_ax.set_xscale(specscale)
                freqdm_ax.set_yscale(specscale)
                freqdm_ax.set_xlabel("frequency / Hz")
                freqdm_ax.set_ylabel("PSD of PSD amplitude / $V^{4}/Hz^{3}$")
                freqdm_ax.legend(loc="upper right")
            else:
                raise ValueError("showtimedomain and showfreqdomain both False")

            titletext = "Mono-Frequency Channle Analysis"
            fig.suptitle(titletext, wrap=True)
            plt.grid()
            if saveplt_opt:
                plt.savefig(
                    saveplt_path + self.exptype + " " + titletext + pltmsg + ".png"
                )
            if showplt_opt:
                plt.show()
            plt.close()
            del fig, gs
        return 0

    def getAvgFFT(
        self,
        # AVARPopt: bool = False,
        windowfunction="rectangle",
        selectshots=[],
        showplt=False,
        specxlim=None,
        verbose=False,
    ):
        """
        Has not been maintained for a while. If you want to use it, please let Yuzhe check it.

        Args:
            windowfunction : str, optional Defaults to 'rectangle'.
            selectshots : list, optional Defaults to [].
            showplt : bool, optional Defaults to False.
            specxlim : type_, optional Defaults to None.
            verbose : bool, optional Defaults to False.
        """
        if not hasattr(self, "acqTime"):
            raise AttributeError(
                "'LIASignal' object has no attribute 'acqTime'. Maybe the stream is a no-pulse stream? "
            )
        acqtimelen = int(np.abs(self.acq_arr[0, 0] - self.acq_arr[0, 1]))
        FFT = np.zeros(acqtimelen, dtype=complex)

        if len(selectshots) == 0:
            selectFFTindex = range(len(self.endofpulse))
        else:
            selectFFTindex = selectshots
        # numofpulses = len(self.endofpulse)
        for i in selectFFTindex:
            frequencies, singleFFT = stdLIAFFT(
                data_x=self.dataX[self.acq_arr[i, 0] : self.acq_arr[i, 1]],
                data_y=self.dataY[self.acq_arr[i, 0] : self.acq_arr[i, 1]],
                samprate=self.samprate,  # in Hz
                demodfreq=self.demodfreq,  # in Hz
                attenuation=self.attenuation,  # in dB. Power ratio (10^(attenuation/10))
                windowfunction=windowfunction,  # Hanning, Hamming, Blackman
                DTRCfilter=self.filterstatus,
                DTRCfilter_TC=self.DTRC_TC,
                DTRCfilter_order=self.DTRC_order,
                verbose=verbose,
            )
            print(singleFFT.shape)
            FFT += singleFFT
            del singleFFT

        self.frequencies = frequencies
        self.FFT = FFT / len(self.endofpulse)
        self.PSD = abs(self.FFT) ** 2
        del frequencies, FFT
        if showplt:
            fig = plt.figure(figsize=(20, 8))  #
            gs = gridspec.GridSpec(nrows=2, ncols=2)
            fig.subplots_adjust(
                top=0.91, bottom=0.11, left=0.08, right=0.96, hspace=0.0, wspace=0.25
            )
            real_ax = fig.add_subplot(gs[0, 0])
            imag_ax = fig.add_subplot(gs[1, 0], sharex=real_ax, sharey=real_ax)
            amp_ax = fig.add_subplot(gs[0, 1], sharex=real_ax)
            phase_ax = fig.add_subplot(gs[1, 1], sharex=real_ax)
            real_ax.plot(
                self.frequencies,
                self.FFT.real,
                label="Real part of FFT",
                color="tab:blue",
            )
            real_ax.set_ylabel("Amplitude / a.u.")
            real_ax.grid(True)
            real_ax.legend(loc="upper right")
            # real_ax.tick_params(axis='y', left=False, labelleft=False)
            real_ax.tick_params(axis="x", bottom=False, labelbottom=False)
            # bottom, top, left, right : bool : Whether to draw the respective ticks
            # labelbottom, labeltop, labelleft, labelright : bool : Whether to draw the respective tick labels.
            imag_ax.plot(
                self.frequencies,
                self.FFT.imag,
                label="Imaginary part of FFT",
                color="tab:orange",
            )
            imag_ax.set_ylabel("Amplitude / a.u.")
            imag_ax.set_xlabel("Frequency / Hz")
            imag_ax.grid(True)
            # imag_ax.tick_params(axis='y', left=False, labelleft=False)
            imag_ax.legend(loc="upper right")
            amp_ax.plot(
                self.frequencies,
                np.abs(self.FFT) ** 2,
                label="Amplitude of FFT^2",
                color="tab:purple",
            )
            amp_ax.set_ylabel("Amplitude / a.u.")
            amp_ax.grid(True)
            amp_ax.legend(loc="upper right")
            # amp_ax.tick_params(axis='y', left=False, labelleft=False)
            amp_ax.tick_params(axis="x", bottom=False, labelbottom=False)

            phase_ax.plot(
                self.frequencies,
                np.angle(self.FFT, deg=True),
                label="Phase of  FFT",
                color="tab:cyan",
            )
            phase_ax.set_ylabel("Phase / $\\degree$")
            phase_ax.set_xlabel("Frequency / Hz")
            phase_ax.grid(True)
            phase_ax.legend(loc="upper right")
            if specxlim is not None:
                real_ax.set_xlim(specxlim[0], specxlim[1])
                imag_ax.set_xlim(specxlim[0], specxlim[1])
                amp_ax.set_xlim(specxlim[0], specxlim[1])
                phase_ax.set_xlim(specxlim[0], specxlim[1])

            titletext = self.exptype  # 'All shots of '+
            for singlefile in self.fileList:
                titletext += "\n" + singlefile
            fig.suptitle(titletext, wrap=True)  # , fontsize=8

            plt.show()

    def getSpectrogram(
        self,
        acq_time: list = [],
        FreqRange: list = [],
        delta_t: float = 1.0,
        delta_f: float = 1.0,
        showplt: bool = True,
        verbose: bool = False,
    ):
        # if not hasattr(self, 'timestamp'):
        #     self.timestamp = np.linspace(start=0, stop=len(self.dataX) / self.samprate, \
        #         num=len(self.dataX), endpoint=False, dtype=float)
        hop = int(delta_t * self.samprate)
        mfft = int(self.samprate / delta_f)  # length of data for a fft
        if len(acq_time) == 0:
            r0, r1 = 0, -1
            N = len(self.dataX)
        SFT = ShortTimeFFT(
            np.hanning(mfft),
            hop=hop,
            fs=self.samprate,
            mfft=mfft,
            scale_to="psd",
            fft_mode="centered",
        )
        Sf = SFT.stft(self.dataX + 1j * self.dataY)  # perform the STFT
        check(Sf.shape)
        check(np.mean(Sf))
        check(np.std(Sf))
        # In the plot, the time extent of the signal x is marked by
        # vertical dashed lines. Note that the SFT produces values
        # outside the time range of x. The shaded areas on the left
        # and the right indicate border effects caused by the window
        # slices in that area not fully being inside time range of x:

        fig = plt.figure(figsize=(8.0, 6.0), dpi=150)  # initialize a figure
        # to specify heights and widths of subfigures
        width_ratios = [1, 0.5]
        height_ratios = [0.25, 0.25, 1]
        gs = gridspec.GridSpec(
            nrows=3, ncols=2, width_ratios=width_ratios, height_ratios=height_ratios
        )  #
        dataX_ax = fig.add_subplot(gs[0, 0])
        dataY_ax = fig.add_subplot(gs[1, 0], sharex=dataX_ax, sharey=dataX_ax)
        specgm_ax = fig.add_subplot(gs[2, 0], sharex=dataX_ax)

        def plotTS(ax, data, label, color, scatter):
            ax.plot(self.timeStamp_s[0:-1], data[0:-1], label=label, c=color)
            if scatter is not None and len(scatter) > 0:
                for idx in scatter[0:]:
                    ax.scatter(
                        self.timeStamp_s[0:-1][idx],
                        data[0:-1][idx],
                        marker="*",
                        c="tab:red",
                    )
            ax.set_ylabel("Voltage [V]")
            ax.legend(loc="upper right")  # adjust the location of the legend

        if not hasattr(self, "Jumps_X"):
            self.tsCheckJump()
        plotTS(dataX_ax, self.dataX, "LIA X", "tab:green", self.Jumps_X)
        plotTS(dataY_ax, self.dataY, "LIA Y", "tab:brown", self.Jumps_Y)

        # fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit

        t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
        specgm_ax.set_title(
            rf"ShortTime FFT ({SFT.m_num*SFT.T:.1g}$\,s$ hanning window)"
        )
        specgm_ax.set(
            xlabel=f"time [sec] ({SFT.p_num(N)} slices, "
            + rf"$\Delta t = {SFT.delta_t:.1g}\,$s)",
            ylabel=f"Frequency [Hz] ({SFT.f_pts} bins, "
            + rf"$\Delta f = {SFT.delta_f:.1g}\,$Hz)",
            xlim=(t_lo, t_hi),
        )

        im1 = specgm_ax.imshow(
            abs(Sf), origin="lower", aspect="auto", extent=SFT.extent(N), cmap="viridis"
        )
        # ax1.plot(t_x, f_i, 'r--', alpha=.5, label='$f_i(t)$')
        fig.colorbar(im1, label="Magnitude $|S_x(t, f)|$")

        # Shade areas where window slices stick out to the side:
        for t0_, t1_ in [
            (t_lo, SFT.lower_border_end[0] * SFT.T),
            (SFT.upper_border_begin(N)[0] * SFT.T, t_hi),
        ]:
            specgm_ax.axvspan(t0_, t1_, color="w", linewidth=0, alpha=0.2)
        for t_ in [0, N * SFT.T]:  # mark signal borders with vertical line:
            specgm_ax.axvline(t_, color="y", linestyle="--", alpha=0.5)
        # specgm_ax.legend()
        fig.tight_layout()
        if showplt:
            plt.show()
        plt.close()
        return fig, specgm_ax

    def insertSignalFFT(self, func, verbose: bool = False):
        if not hasattr(self, "avgFFT"):
            self.getNoPulseFFT()
        self.FFT += func(self.frequencies)

    def getSpectrum(
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
        referfreq=None,
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
        legendloc="best",  # 'upper right' 'upper left' 'lower left' 'lower right'
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
            Only effective when showtimedomain is True.
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
            'V^2 / Hz', 'V / sqrt(Hz)', 'Phi0^2 / Hz' and 'Phi0 / sqrt(Hz)' respectively.
            Defaults to 'FluxPSD'.

        Mf : float, optional
            Feedback sensitivity which can be found in the SQUID specifications.
            For the SQUID we usually use, M_f = 31 706 \\Phi_0 / A???? Is this true?
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

        # Set plot font and fontsize
        plt.rc("font", size=12)
        # plt.rcParams["font.family"] = "Times New Roman"
        # plt.rcParams["mathtext.fontset"] = 'cm'  # 'dejavuserif'

        # If show standard deviation in the plot
        if showstd:
            # The start and the end of a range for computing the standard deviation
            # stdstart and stdend are the indice of the array
            if stddev_range is None:
                stdstart = 0
                stdend = len(self.frequencies)
            else:
                stdstart = np.argmin(abs(self.frequencies - stddev_range[0]))
                stdend = np.argmin(abs(self.frequencies - stddev_range[1]))

            # If there are not many data points in this range, a warning will be raised.
            if abs(stdstart - stdend) <= 10:
                print(
                    "abs(stdstart-stdend)<=10\ntoo little data for computing standard deviation"
                )

            # Compute the standard deviation
            self.PSDstd = np.std(self.PSD[stdstart:stdend])

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
                / (np.amax(self.PSD) - np.amin(self.PSD))
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
            if self.timeStamp_s is None:
                self.timeStamp_s = np.linspace(
                    start=0,
                    stop=len(self.dataX) / self.samprate,
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
        else:
            raise ValueError("no plot")

        # plot time-series
        if showtimedomain:
            # TTL signal of pulses
            if not hasattr(self, "pulsedata"):
                self.pulsedata = np.zeros(len(self.timeStamp_s))
            pulse_ax.plot(
                self.timeStamp_s, self.pulsedata, label="pulse sequence", c="tab:purple"
            )
            pulse_ax.set_ylabel("Voltage [V]")
            pulse_ax.set_xlim(self.timeStamp_s[0], self.timeStamp_s[-1])

            # X channel of LIA
            dataX_ax.plot(self.timeStamp_s, self.dataX, label="LIA X", c="tab:green")
            dataX_ax.set_ylabel("Voltage [V]")
            dataX_ax.set_xlim(self.timeStamp_s[0], self.timeStamp_s[-1])

            # Y channel of LIA
            dataY_ax.plot(self.timeStamp_s, self.dataY, label="LIA Y", c="tab:brown")
            dataY_ax.set_ylabel("Voltage [V]")
            dataY_ax.set_xlim(self.timeStamp_s[0], self.timeStamp_s[-1])
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
                        self.timeStamp_s[self.acq_arr[i, 0] : self.acq_arr[i, 1] + 1],
                        self.dataX[self.acq_arr[i, 0] : self.acq_arr[i, 1] + 1],
                        c="tab:orange",
                        alpha=0.9,
                    )
                    dataY_ax.plot(
                        self.timeStamp_s[self.acq_arr[i, 0] : self.acq_arr[i, 1] + 1],
                        self.dataY[self.acq_arr[i, 0] : self.acq_arr[i, 1] + 1],
                        c="tab:orange",
                        alpha=0.9,
                    )
                i = self.selectPulse_list[-1]
                dataX_ax.plot(
                    self.timeStamp_s[self.acq_arr[i, 0] : self.acq_arr[i, 1] + 1],
                    self.dataX[self.acq_arr[i, 0] : self.acq_arr[i, 1] + 1],
                    label="LIA X data for PSD",
                    c="tab:orange",
                    alpha=0.9,
                )
                dataY_ax.plot(
                    self.timeStamp_s[self.acq_arr[i, 0] : self.acq_arr[i, 1] + 1],
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
            if referfreq is None:
                # if there has been a successful fitting
                if self.fitflag:
                    referfreq = self.popt[0]
                # otherwise, choose the frequency of the data point with the maximum amplitude
                else:
                    referfreq = self.frequencies[np.argmax(self.PSD)]
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
                specxaxis = specx_dict[specxunit.upper()][1] * self.frequencies
                specxunit = specx_dict[specxunit.upper()][0]
            elif specxunit == "ppm":
                specxaxis = 1e6 * (self.frequencies / referfreq - 1.0)
                specxlabel = "Separation"
            elif specxunit == "ppb":
                specxaxis = 1e9 * (self.frequencies / referfreq - 1.0)
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
                        f"[{self.getSpectrum.__name__}] Warning: Mf is not specified. "
                        "Force spectrum 'ampunit' to be 'V'. "
                    )
                    ampunit = "V"

            if Rf is None and (ampunit == "Phi" or ampunit == "expPhi"):
                if hasattr(self, "SQD_Rf") and self.SQD_Rf is not None:
                    Rf = self.SQD_Rf
                else:
                    print(
                        f"[{self.getSpectrum.__name__}] Warning: Rf is not specified. "
                        "Force spectrum 'ampunit' to be 'V'. "
                    )
                    ampunit = "V"

            spectype = spectype.upper()
            ampunit = ampunit.upper()
            # Set y-axis unit
            # if ampunit.upper() == 'V':
            #     ampfactor = 1.
            if spectype == "PSD".upper():
                self.spectrum = self.PSD
                ampfactor = 1.0
                specyunit = r"$V^2/\\mathrm{Hz}$"
            elif spectype == "ASD".upper():
                self.spectrum = np.sqrt(self.PSD)
                ampfactor = 1.0
                specyunit = r"$V/\sqrt{\\mathrm{Hz}}$"
            elif spectype == "FluxPSD".upper() and ampunit != "expPhi".upper():
                ampfactor = (Mf / Rf) ** 2
                self.spectrum = self.PSD * ampfactor
                specyunit = r"$\Phi_{0}^{2}/\\mathrm{Hz}$"
            elif spectype == "FluxASD".upper() and ampunit != "expPhi".upper():
                ampfactor = Mf / Rf
                self.spectrum = np.sqrt(self.PSD) * ampfactor
                specyunit = r"$\Phi_{0}/\sqrt{\\mathrm{Hz}}$"

            elif ampunit == "expPhi".upper():
                ampfactor = (Mf / Rf) ** 2
                if spectype == "FluxPSD".upper():
                    if amppow is None:
                        amppow = int(np.log10(np.amax(self.PSD * ampfactor)) - 1)
                    self.spectrum = 10 ** (-amppow) * ampfactor * self.PSD
                    # determine the unit of the spectrum
                    if amppow == 0:
                        specyunit = r"$\Phi_{0}^{2}/\\mathrm{Hz}$"
                    else:
                        specyunit = r"$10^{%d}\ \\Phi_{0}^{2}/\\mathrm{Hz}$" % (amppow)
                elif spectype == "FluxASD".upper():
                    if amppow is None:
                        self.spectrum = np.sqrt(self.PSD * ampfactor)
                        amppow = int(np.log10(np.amax(self.spectrum)) - 1)
                    self.spectrum = 10 ** (-amppow) * np.sqrt(self.PSD * ampfactor)
                    if amppow == 0:
                        specyunit = r"$\\Phi_{0}/\sqrt{\\mathrm{Hz}}$"
                    else:
                        specyunit = r"$10^{%d}\ \\Phi_{0}/\sqrt{\\mathrm{Hz}}$" % (
                            amppow
                        )
                else:
                    raise ValueError("spectype wrong 315")
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
                    label=f"Standard deviation = {(ampfactor) * self.PSDstd:.3e} "
                    + specyunit,
                    c="tab:green",
                )

            # if to show fit curve and there has been a successful fitting
            if showfit and self.fitflag:
                # in case there are several fit curves, like dualGaussian
                for i in range(len(self.fitcurves) - 1):
                    spec_ax.plot(
                        specxaxis[self.fitrange[0] : self.fitrange[1]],
                        10 ** (-amppow) * ampfactor * self.fitcurves[i],
                        "--",
                        c="tab:red",
                        alpha=0.6,
                    )
                # plot the fit curve
                spec_ax.plot(
                    specxaxis[self.fitrange[0] : self.fitrange[1]],
                    10 ** (-amppow) * ampfactor * self.fitcurves[-1],
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
                            referfreq=referfreq,
                        )
                        # formatter = mticker.FuncFormatter(lambda x, pos: '{:.3f}'.format(1e-6*self.frequencies))
                        formatter = mticker.FuncFormatter(plotaxisfmt_partial)
                if specxunit == "Hz":
                    if specxunit2 == "ppm":
                        plotaxisfmt_partial = partial(
                            plotaxisfmt_Hz2ppm,
                            format_string=specx2format,
                            referfreq=referfreq,
                        )
                        # formatter = mticker.FuncFormatter(lambda x, pos: '{:.3f}'.format(1e-6*self.frequencies))
                        formatter = mticker.FuncFormatter(plotaxisfmt_partial)
                if specxunit == "MHz":
                    if specxunit2 == "ppm":
                        plotaxisfmt_partial = partial(
                            plotaxisfmt_MHz2ppm,
                            format_string=specx2format,
                            referfreq=referfreq,
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
                    self.spectrum - 10 ** (-amppow) * ampfactor * self.fitcurves[-1],
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

    def makeplot(
        self, filename, transpose_opt=False, alpha=0.5, showplt_opt=True, verbose=False
    ):
        if transpose_opt:
            self.frequencies = np.loadtxt(
                filename, dtype=float, skiprows=5
            ).transpose()[0]
            self.PSD = np.loadtxt(filename, dtype=float, skiprows=5).transpose()[1]
        else:
            self.frequencies = np.loadtxt(filename, dtype=float, skiprows=5)[0]
            self.PSD = np.loadtxt(filename, dtype=float, skiprows=5)[1]
        PolyEven_centered = partial(PolyEven, center=self.demodfreq, verbose=False)
        ar = [len(self.frequencies) // 8, -len(self.frequencies) // 8]
        if verbose:
            print("ar ", ar)
        fitparas = [self.PSD[len(self.PSD) // 2], 0, 0, 0, 0, 0]
        self.popt, self.pcov = scipy.optimize.curve_fit(
            PolyEven_centered,
            self.frequencies[ar[0] : ar[1]],
            self.PSD[ar[0] : ar[1]],
            fitparas,
        )
        self.fitflag = True
        self.perr = np.sqrt(np.diag(self.pcov))
        dof = max(0, len(self.frequencies[ar[0] : ar[1]]) - len(self.popt))
        # student-t value for the dof and confidence level
        tval = scipy.stats.distributions.t.ppf(1.0 - alpha / 2.0, dof)
        # if verbose:
        #     print('self.popt ', self.popt)
        #     print('self.pcov ', self.pcov)
        #     #print('self.pcov 2 ', self.pcov*self.popt**2/(len(self.frequencies[ar[0]:ar[1]])-len(self.popt)))
        #     print('self.perr ', self.perr)
        #     print('tval ', tval)
        self.fitresultlist = []
        for i in range(len(self.popt)):
            self.fitresultlist.append(ufloat(self.popt[i], tval * self.perr[i]))
        self.fitcurves = PolyEven_centered(
            self.frequencies[ar[0] : ar[1]],
            self.popt[0],
            self.popt[1],
            self.popt[2],
            self.popt[3],
            self.popt[4],
            self.popt[5],
        )
        self.residual = np.sum(abs(self.PSD[ar[0] : ar[1]] - self.fitcurves)) / np.sum(
            abs(self.PSD[ar[0] : ar[1]])
        )
        self.fitreport = "PolyEven_centered fit (95 % confidence level)\n"
        self.fitreport += "C0 = {:.6u}\n".format(self.fitresultlist[0])
        self.fitreport += "C2 = {:.6u}\n".format(self.fitresultlist[1])
        self.fitreport += "C4 = {:.6u}\n".format(self.fitresultlist[2])
        self.fitreport += "C6 = {:.6u}\n".format(self.fitresultlist[3])
        self.fitreport += "C8 = {:.6u}\n".format(self.fitresultlist[4])
        self.fitreport += "C10 = {:.6u}\n".format(self.fitresultlist[5])
        self.fitreport += "relative residual = {:.2%}".format(self.residual)
        if verbose:
            print(self.fitreport)
            print(
                "[%.6e,%.6e,%.6e,%.6e,%.6e,%.6e]"
                % (
                    self.popt[0],
                    self.popt[1],
                    self.popt[2],
                    self.popt[3],
                    self.popt[4],
                    self.popt[5],
                )
            )

        fig = plt.figure(figsize=(14, 9))  #
        gs = gridspec.GridSpec(nrows=1, ncols=1)  #
        # fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
        #                     bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
        spec_ax = fig.add_subplot(gs[0, 0])
        spec_ax.scatter(self.frequencies, self.PSD)
        spec_ax.plot(
            self.frequencies[ar[0] : ar[1]],
            self.fitcurves,
            "--",
            c="tab:red",
            alpha=0.7,
            label="Polynomial fit\n %5.3g, %5.3g, %5.3g, %5.3g"
            % (self.popt[0], self.popt[1], self.popt[2], self.popt[3]),
        )
        spec_ax.legend("upper right")
        if showplt_opt:
            plt.show()
        return 0

    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     print("\nInside __exit__")

    # def __del__(self):
    #     print("__del__")
