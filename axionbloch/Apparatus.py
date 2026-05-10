from axionbloch.dependency import *

from functools import partial

from axionbloch.utils import Lorentzian


class Magnet:
    name:str
    B0: Quantity | None = None
    direction: list | None = None
    FWHM: Quantity | None = None
    numPt: float = 1
    nFWHM: float = 10.0
    def __init__(
        self,
        name="magnet",
        B0: Quantity | None = None,
        direction: list | None = None,
        FWHM: Quantity | None = None,
        numPt: float = 1,
        nFWHM: float = 10.0,
        verbose: bool = False,
    ):
        """
        name : str
            name of the SQUID. default to 'PhiC6L1W'. 'PhiC73L1' is the other option
        """
        self.name = name
        assert nFWHM >= 0
        self.nFWHM = nFWHM
        if B0 is None or FWHM is None:
            raise ValueError("B0 and FWHM must be provided")

        self.B0 = B0
        self.direction = direction
        self.FWHM = FWHM
        self.B0_nW = self.nFWHM * self.FWHM * self.B0
        self.numPt = numPt
        # self.B0_T = self.B0.to_value(unit.T)
        self.FWHM_B0 = (self.B0 * self.FWHM).to(unit.T)
        # self.B0_nW_T = self.B0_nW.to_value(unit.T)
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
        """
        set the homogeneity sampling using ...some complicated methods
        """
        # update self.numPt if
        if numPt is not None:
            self.numPt = max(1, int(numPt))
        elif self.numPt is None:
            self.numPt = 1

        # homogeneous field
        if self.numPt == 1 or self.FWHM_B0 == 0.0 or self.nFWHM == 0:
            self.B_spread = np.ones(1) * self.B0
            self.ratios = np.ones(1)
        # inhomogeneous field
        else:
            pdf = partial(
                Lorentzian,
                center=self.B0,
                FWHM=self.FWHM_B0,
                area=1,
                offset=0,
            )

            # uniform sampling over [-1, 1]
            uni_samp = np.linspace(start=-1, stop=1, num=self.numPt, endpoint=True)
            # transform uniform sampling to the desired distribution
            self.B_spread = (
                self.nFWHM * np.sign(uni_samp) * np.abs(uni_samp) ** 2
            ) * self.FWHM_B0 + self.B0  # exponent < 1 increases central density
            # plot histogram of the sampled B values
            if showPlot:
                fig = plt.figure(
                    figsize=(8.5 / 2.56, 6.5 / 2.56), dpi=300
                )  # initialize a figure
                gs = gridspec.GridSpec(
                    nrows=1, ncols=1
                )  # create grid for multiple figures
                ax = fig.add_subplot(gs[0, 0])
                hist, bin_edges = np.histogram(self.B_spread, bins=30)
                ax.plot(
                    (bin_edges[1:] - self.B0) / self.FWHM_B0,
                    hist,
                    label="", marker="o", color='goldenrod', markeredgecolor='darkgoldenrod'
                )
                ax.set_xlabel("Magnetic field - B0 (FWHM)")
                ax.set_ylabel("Number of sampling points")
                ax.set_title("Histogram of number of sampling \nover magnetic field spread")
                plt.tight_layout()
                # plt.savefig('example figure - one-column.png', transparent=False)
                plt.show()

            # Define interval edges (midpoints between adjacent x's)
            edges = np.zeros(len(self.B_spread) + 1) * self.B_spread.unit
            edges[1:-1] = (self.B_spread[:-1] + self.B_spread[1:]) / 2
            edges[0] = self.B_spread[0]
            edges[-1] = self.B_spread[-1]

            self.ratios = np.zeros(self.B_spread.shape)
            for i in range(len(self.B_spread)):
                a, b = edges[i], edges[i + 1]
                x = np.linspace(a, b, 32)
                self.ratios[i] = np.trapezoid(pdf(x), x)

            if showPlot:
                fig = plt.figure(
                    figsize=(8.5 / 2.56, 6.5 / 2.56), dpi=300
                )
                gs = gridspec.GridSpec(
                    nrows=1, ncols=1
                )  # create grid for multiple figures
                ax = fig.add_subplot(gs[0, 0])
                ax.plot(self.B_spread, self.ratios, label="", marker="o")
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_title("Histogram of spin packet ratio over magnetic field spread")
                plt.tight_layout()
                plt.show()

            # normalize ratios to sum to 1
            self.ratios /= np.sum(self.ratios)
