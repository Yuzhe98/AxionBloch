# src/Apparatus.py
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from functools import partial

from typing import Optional

from scipy.integrate import quad

from axionbloch.Envelope import PhysicalQuantity
from axionbloch.utils import PhysicalObject, Lorentzian


class Magnet(PhysicalObject):

    def __init__(
        self,
        name=None,
        B0: Optional[PhysicalQuantity] = None,
        FWHM: Optional[PhysicalQuantity] = None,
        numPt: float = 1,
        nFWHM: float = 10.0,
        verbose: bool = False,
    ):
        """
        name : str
            name of the SQUID. default to 'PhiC6L1W'. 'PhiC73L1' is the other option
        """
        super().__init__()

        self.name = name
        assert nFWHM >= 0
        self.nFWHM = nFWHM
        self.B0 = B0
        self.FWHM = FWHM
        self.B0_nW = self.nFWHM * self.FWHM * self.B0
        self.numPt = numPt
        self.FWHM_T = (self.B0 * self.FWHM).value_in("T")
        # Specify all physical quantities with units
        self.physicalQuantities = {"B0": "T", "FWHM": "", "B0_nW": "T"}
        # Specify general quantities
        self.generalQuantities = {"numPt": "float", "nFWHM": "float", "FWHM_T": "float"}
        # make sure that we use common units for quantities
        self.useCommonUnits()
        self.setHomogeneity()

    def setHomogeneity(
        self,
        # lineshape:str = "Lorentizan",
        numPt: int | float = None,
        showplt: bool = False,
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

        # FWHM_T = (self.B0 * self.FWHM).value_in("T")
        if self.numPt == 1 or self.FWHM_T == 0.0 or self.nFWHM == 0:
            self.B_vals_T = np.array([self.B0.value_in("T")])
            self.ratios = np.array([1.0])
        else:
            pdf = partial(
                Lorentzian,
                center=self.B0.value_in("T"),
                FWHM=self.FWHM_T,
                area=1,
                offset=0,
            )

            u = np.linspace(start=-1, stop=1, num=self.numPt, endpoint=True)
            self.B_vals_T = (
                self.nFWHM * np.sign(u) * np.abs(u) ** 2
            ) * self.FWHM_T + self.B0.value_in(
                "T"
            )  # exponent < 1 increases central density
            if showplt:
                fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
                gs = gridspec.GridSpec(
                    nrows=1, ncols=1
                )  # create grid for multiple figures
                ax00 = fig.add_subplot(gs[0, 0])
                hist, bin_edges = np.histogram(self.B_vals_T)
                # for i, count in enumerate(hist):
                #     if count > 0:
                #         ax00.scatter(bin_edges[i+1], count, color='goldenrod', edgecolors='darkgoldenrod', linewidths=0.8, marker='o', s=2, zorder=6)
                ax00.plot(
                    (bin_edges[1:] - self.B0.value_in("T")) / self.FWHM_T,
                    hist,
                    label="",
                )
                ax00.set_xlabel("Magnetic field - B0 (FWHM)")
                ax00.set_ylabel("Number of data points")
                plt.tight_layout()
                # plt.savefig('example figure - one-column.png', transparent=False)
                plt.show()

            # Define interval edges (midpoints between adjacent x's)
            edges = np.zeros(len(self.B_vals_T) + 1)
            edges[1:-1] = (self.B_vals_T[:-1] + self.B_vals_T[1:]) / 2
            edges[0] = -np.inf
            edges[-1] = np.inf

            self.ratios = np.zeros_like(self.B_vals_T)
            for i in range(len(self.B_vals_T)):
                a, b = edges[i], edges[i + 1]
                self.ratios[i], _ = quad(pdf, a, b)

            if showplt:
                fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
                gs = gridspec.GridSpec(
                    nrows=1, ncols=1
                )  # create grid for multiple figures
                ax00 = fig.add_subplot(gs[0, 0])

                ax00.plot(self.B_vals_T, self.ratios, label="")
                ax00.set_xlabel("")
                ax00.set_ylabel("")
                # ax00.set_xscale('log')
                # ax00.set_yscale('log')
                ax00.legend()
                plt.tight_layout()
                # plt.savefig('example figure - one-column.png', transparent=False)
                plt.show()

            self.ratios /= np.sum(self.ratios)
