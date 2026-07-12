"""Physical sample descriptor for NMR/spin-precession experiments.

:class:`Sample` holds the material and geometric properties of a single-phase
spin sample and provides helpers for computing spin density, thermal
polarization, and equilibrium magnetization.
"""

import numpy as np
from astropy import units as unit
from astropy.constants import codata2018 as const
from astropy.units import Quantity


class Sample:
    """Single-phase spin sample for NMR/spin-precession experiments.

    Stores material properties and geometry, and derives spin number density
    and total spin count on construction.  Density inputs should be evaluated
    at STP (273.15 K, 10⁵ Pa per IUPAC 1982).
    """

    def __init__(
        self,
        name: str | None = None,  # name of the sample
        gamma: (
            Quantity[unit.rad * unit.Hz / unit.T] | None
        ) = None,  # gyromagnetic ratio. Remember to input it with 2 pi
        massDensity: Quantity | None = None,  # mass density at STP
        molarMass: Quantity | None = None,  # molar mass
        numOfSpinsPerMolecule: Quantity | None = None,  # number of spins per molecule
        T2: Quantity | None = None,  # transverse relaxation time
        T1: Quantity | None = None,  # longitudinal relaxation time
        vol: Quantity | None = None,  # volume
        mu: Quantity | None = None,  # magnetic dipole moment per spin
        temp: Quantity | None = None,  # sample temperature
        pol: Quantity | None = None,  # spin polarization (dimensionless, 0–1)
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        name : str, optional
            Human-readable label for the sample.
        gamma : Quantity [rad Hz / T], optional
            Gyromagnetic ratio *including* the 2π factor, i.e. ``2π × γ_bar``.
        massDensity : Quantity [g/cm³], optional
            Mass density at STP.
        molarMass : Quantity [g/mol]
            Molar mass of the molecule (required).
        numOfSpinsPerMolecule : Quantity [dimensionless]
            Number of active spins per molecule.
        T2 : Quantity [s], optional
            Transverse (spin–spin) relaxation time.
        T1 : Quantity [s], optional
            Longitudinal (spin–lattice) relaxation time.
        vol : Quantity [cm³], optional
            Sample volume.
        mu : Quantity [J/T], optional
            Magnetic dipole moment of a single spin.
        temp : Quantity [K], optional
            Sample temperature; used as default in :meth:`getThermalPol`.
        pol : Quantity [dimensionless], optional
            Spin polarization; used as default in :meth:`getM0`.
        verbose : bool
            Print derived quantities after construction.
        """
        msgPrefix = f"[{self.__class__.__name__}.{self.__init__.__name__}]"
        self.name = name
        self.gamma = gamma

        self.massDensity = massDensity
        self.molarMass = molarMass
        self.numOfSpinsPerMolecule = numOfSpinsPerMolecule

        assert self.molarMass is not None
        self.spinNumDensity = (
            self.numOfSpinsPerMolecule * self.massDensity / self.molarMass * const.N_A
        ).to(unit.cm ** (-3))

        self.T2 = T2
        self.T1 = T1
        self.vol = vol

        self.totalNumOfSpins = (self.spinNumDensity * self.vol).to(unit.one)

        self.mu = mu
        self.temp = temp
        self.pol = pol

    def getThermalPol(
        self,
        B_pol: Quantity,
        temp: Quantity | None = None,
        verbose: bool = False,
    ) -> Quantity:
        """Return the thermal spin polarization at a given field and temperature.

        Uses the exact result ``tanh(ℏγB / 2k_BT)``.

        Parameters
        ----------
        B_pol : Quantity [T]
            Polarizing magnetic field strength.
        temp : Quantity [K], optional
            Sample temperature; falls back to ``self.temp`` when omitted.

        Returns
        -------
        Quantity [dimensionless]
            Thermal polarization in [0, 1].
        """
        msgPrefix = f"[{self.__class__.__name__}.{self.getThermalPol.__name__}]"
        # pol = hbar * self.gamma * B_pol / (2 * k * temp)  # approximate
        if temp is None:
            temp = self.temp
        assert (
            temp is not None
        ), "Temperature is required to compute thermal polarization. Please provide temp or set it in the Sample object."
        if temp.to(unit.K) < 0 * unit.K:
            raise ValueError("Temperature must be >= 0 K.")

        pol = np.tanh(const.hbar * self.gamma * B_pol / (2 * const.k_B * temp))  # exact
        pol = pol.to(unit.one)
        # check(pol)
        if verbose:
            print(
                msgPrefix,
                f"Thermal polarization at B_pol={B_pol.to(unit.T)} and temp={temp.to(unit.K)} is {pol.to(unit.one)}. ",
            )
        return pol

    def getM0(
        self,
        pol: Quantity | None = None,
        verbose: bool = False,
    ):
        """Return the bulk magnetization M0 = μ × pol × n_s [A/m].

        Parameters
        ----------
        pol : Quantity [dimensionless], optional
            Spin polarization; falls back to ``self.pol`` when omitted.

        Returns
        -------
        Quantity [A/m]
            Magnetization density.
        """
        msgPrefix = f"[{self.__class__.__name__}.{self.getM0.__name__}]"

        if pol is None:
            pol = self.pol
        assert (
            pol is not None
        ), "Polarization is required to compute M0. Please provide pol or set it in the Sample object."
        M0 = (self.mu * pol * self.totalNumOfSpins / self.vol).to(unit.A / unit.m)
        # self.M0_SPN = (self.mu * ns_SPN).to(unit.A / unit.m)
        if verbose:
            print(msgPrefix, f"Magnetization M0 is {M0}")
        return M0

    def getM0_SPN(
        self,
        verbose: bool = False,
    ):
        """Return the spin-projection-noise magnetization M0_SPN = μ × √N / V [A/m].

        Spin projection noise arises from quantum fluctuations: for N uncorrelated
        spin-1/2 particles the rms fluctuation in spin number is √N, giving an
        effective magnetization density μ√N / V.

        Returns
        -------
        Quantity [A/m]
            Spin-projection-noise magnetization density.
        """
        msgPrefix = f"[{self.__class__.__name__}.{self.getM0_SPN.__name__}]"

        M0_SPN = (self.mu * np.sqrt(self.totalNumOfSpins) / self.vol).to(
            unit.A / unit.m
        )
        self.M0_SPN = M0_SPN
        if verbose:
            print(msgPrefix, f"Spin projection noise magnetization M0_SPN is {M0_SPN}")
        return M0_SPN

    def getM0eqb(
        self,
        B_pol: Quantity,
        temp: Quantity | None = None,
        verbose: bool = False,
    ):
        """Return the thermal-equilibrium magnetization at given B_pol and temperature.

        Convenience wrapper: calls :meth:`getThermalPol` then :meth:`getM0`.

        Parameters
        ----------
        B_pol : Quantity [T]
            Polarizing magnetic field strength.
        temp : Quantity [K], optional
            Sample temperature; falls back to ``self.temp`` when omitted.

        Returns
        -------
        Quantity [A/m]
            Equilibrium magnetization density.
        """
        pol = self.getThermalPol(B_pol, temp, verbose)
        M0eqb = self.getM0(pol, verbose)
        if verbose:
            print(
                f"[{self.__class__.__name__}.{self.getM0eqb.__name__}] Equilibrium magnetization M0eqb is {M0eqb}"
            )
        return M0eqb
