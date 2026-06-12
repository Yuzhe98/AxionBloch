from functools import partial

from axionbloch.dependency import *
from axionbloch.utils import Lorentzian_0edge


class Magnet:
    """DC magnetic field apparatus.

    Models a static polarising magnet whose field may be spatially inhomogeneous.
    Inhomogeneity is characterised by a Lorentzian lineshape with a given FWHM,
    and the field distribution is discretised into ``numPt`` spin packets for
    simulation.

    Attributes
    ----------
    name : str
    B0 : astropy Quantity
        Nominal field strength.
    direction : list or None
        Unit vector along B0 (laboratory frame).
    FWHM : astropy Quantity
        Fractional field inhomogeneity (dimensionless, e.g. ppm).
    B_spread : ndarray
        Sampled B0 values for each spin packet.
    ratios : ndarray
        Fractional weight of each spin packet (sums to 1).
    """

    name: str
    B0: Quantity | None = None
    direction: list | None = None
    FWHM: Quantity | None = None
    numPt: float = 1
    nFWHM: float = 10.0

    def __init__(
        self,
        name="magnet",
        B0: Quantity[unit.T] | None = None,
        direction: list | None = None,
        FWHM: Quantity[ppm] | None = None,
        numPt: float = 1,
        nFWHM: float = 10.0,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        name : str
            Human-readable label for this magnet.
        B0 : Quantity
            Nominal DC field strength (e.g. ``0.1 * unit.T``).
        direction : list, optional
            Unit vector specifying the field orientation in the lab frame.
        FWHM : Quantity
            Fractional full-width at half-maximum of the field inhomogeneity
            (dimensionless, e.g. ``1e-6 * unit.one`` for 1 ppm).
        numPt : int
            Number of discrete spin packets used to model field inhomogeneity.
            ``numPt=1`` treats the field as perfectly homogeneous.
        nFWHM : float
            Half-range of the field spread expressed in units of FWHM.
        verbose : bool
            Print diagnostic information.
        """
        logPrefix = f"[{self.__class__.__name__}.{self.__init__.__name__}]"
        self.name = name
        assert nFWHM >= 0
        self.nFWHM = nFWHM
        if B0 is None or FWHM is None:
            raise ValueError(logPrefix + " B0 and FWHM must be provided")

        self.B0 = B0
        self.direction = direction
        self.FWHM = FWHM
        self.B0_nW = self.nFWHM * self.FWHM * self.B0
        self.numPt = numPt
        self.FWHM_B0 = (self.B0 * self.FWHM).to(unit.T)
        self.B_spread = np.ones(1) * self.B0
        self.ratios = np.ones(1)
        self.setHomogeneity(verbose=verbose)

    def setHomogeneity(
        self,
        # lineshape:str = "Lorentizan",
        numPt: int | float = None,
        showPlot: bool = False,
        verbose: bool = False,
    ):
        """Set the spin-packet sampling of the field inhomogeneity.

        Discretises a Lorentzian field distribution into ``numPt`` spin packets
        using a non-uniform sampling scheme (density proportional to
        ``|x|^(1/2)`` in normalized coordinates) to oversample the peak region.
        Each packet is assigned a weight ``ratio`` equal to its fraction of the
        total Lorentzian area so that weighted averages reproduce the continuous
        lineshape.

        For ``numPt == 1`` or zero FWHM the field is treated as homogeneous and
        a single packet at ``B0`` with weight 1 is used.

        Parameters
        ----------
        numPt : int or float, optional
            Override the stored ``self.numPt``.
        showPlot : bool
            Display diagnostic histograms of the sampled B values and weights.
        verbose : bool
            Print diagnostic information.
        """
        logPrefix = f"[{self.__class__.__name__}.{self.setHomogeneity.__name__}]"
        # update self.numPt if it is None
        if numPt is not None:
            self.numPt = max(1, int(numPt))
        elif self.numPt is None:
            self.numPt = 1
        if verbose:
            print(logPrefix, f"numPt = {numPt}")
            print(logPrefix, f"self.numPt = {self.numPt}")
        # homogeneous field
        if self.numPt == 1 or self.FWHM_B0 == 0.0 or self.nFWHM == 0:
            self.B_spread = np.ones(1) * self.B0
            self.ratios = np.ones(1)
        # inhomogeneous field
        else:
            pdf = partial(
                Lorentzian_0edge,
                center=self.B0,
                FWHM=self.FWHM_B0,
                area=1,
                offset=0,
            )
            if verbose:
                print(logPrefix, f"inhomogeneous field. self.numPt = {self.numPt}")
            # uniform sampling over [-1, 1]
            uni_samp = np.linspace(start=-1, stop=1, num=self.numPt, endpoint=True)
            # transform uniform sampling to the desired distribution
            self.B_spread = (
                self.nFWHM * np.sign(uni_samp) * np.abs(uni_samp) ** 2
            ) * self.FWHM_B0 + self.B0
            if verbose:
                print(logPrefix, f"B_spread.shape = {self.B_spread.shape}")
                print(
                    f"{logPrefix} B_spread: {len(self.B_spread)} points  "
                    f"range=[{self.B_spread.min():.6g}, {self.B_spread.max():.6g}]"
                )

            if showPlot:
                fig = plt.figure(
                    figsize=(8.5 / 2.54, 6.5 / 2.54), dpi=300
                )  # initialize a figure
                gs = gridspec.GridSpec(
                    nrows=1, ncols=1
                )  # create grid for multiple figures
                ax = fig.add_subplot(gs[0, 0])
                hist, bin_edges = np.histogram(self.B_spread, bins=30)
                ax.plot(
                    (bin_edges[1:] - self.B0) / self.FWHM_B0,
                    hist,
                    label="",
                    marker="o",
                    color="goldenrod",
                    markeredgecolor="darkgoldenrod",
                )
                ax.set_xlabel("Magnetic field - $B_0$ (FWHM)")
                ax.set_ylabel("Number of sampling points")
                ax.set_title(
                    "Histogram of number of sampling \nover magnetic field spread"
                )
                plt.tight_layout()
                # plt.savefig('example figure - one-column.png', transparent=False)
                plt.show()

            # Define interval edges
            edges = np.zeros(len(self.B_spread) + 1) * self.B_spread[0].unit
            edges[1:-1] = (self.B_spread[:-1] + self.B_spread[1:]) / 2
            edges[0] = self.B_spread[0] - abs(edges[2] - edges[1]) / 2
            edges[-1] = self.B_spread[-1] + abs(edges[-2] - edges[-3]) / 2

            self.ratios = np.zeros(self.B_spread.shape)
            for i in range(len(self.B_spread)):
                a, b = edges[i], edges[i + 1]
                x = np.linspace(a, b, 32)
                self.ratios[i] = np.trapezoid(pdf(x), x)
            # high-frequency spins can create wiggles in the magnetization signal
            # to suppress wiggles, add hamming window to decrease the ratio of high-frequency spins
            # this is not ideal or physical. However, in simulations, we cannot use extremely large nFWHM
            # to account for the real magnetic field distribution. We truncate at edges, instead.
            self.ratios *= np.hamming(len(self.ratios)) ** 2
            # normalize ratios to sum to 1
            self.ratios /= np.sum(self.ratios)

            if verbose:
                print(
                    f"{logPrefix} ratios computed  sum={self.ratios.sum():.6g}  "
                    f"min={self.ratios.min():.4g}  max={self.ratios.max():.4g}"
                )

            if showPlot:
                fig = plt.figure(figsize=(8.5 / 2.54, 6.5 / 2.54), dpi=300)
                gs = gridspec.GridSpec(
                    nrows=1, ncols=1
                )  # create grid for multiple figures
                ax = fig.add_subplot(gs[0, 0])
                ax.plot(
                    (self.B_spread - self.B0) / self.FWHM_B0,
                    self.ratios,
                    label="",
                    marker="o",
                )
                ax.set_xlabel("Magnetic field - $B_0$ (FWHM)")
                ax.set_ylabel("")
                ax.set_title(
                    "Histogram of spin packet ratio over magnetic field spread"
                )
                plt.tight_layout()
                plt.show()

            if verbose:
                print(logPrefix, f"self.ratios normalized sum={self.ratios.sum():g}")
