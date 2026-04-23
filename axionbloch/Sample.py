import numpy as np
from typing import Optional
from astropy import units as u
from astropy.constants import codata2018 as const

# from axionbloch.enphylope import PhysicalQuantity as u.Quantity
from axionbloch.constants import (
    gamma_Xe129,
    gamma_p,
    mu_p,
    mu_Xe129,
)
from axionbloch.utils import PhysicalObject


class Sample(PhysicalObject):
    """
    Describe the sample used in experiments.
    Only consider samples in one phase.
    """

    def __init__(
        self,
        name: Optional[str] = None,  # name of the sample
        gamma: Optional[
            u.Quantity
        ] = None,  # gyromagnetic ratio. Remember to input it with 2 pi
        massDensity: Optional[u.Quantity] = None,  # mass density at STP
        molarMass: Optional[u.Quantity] = None,  # molar mass
        numOfSpinsPerMolecule: Optional[u.Quantity] = None,  # number of spins per molecule
        T2: Optional[u.Quantity] = None,  #
        T1: Optional[u.Quantity] = None,  #
        vol: Optional[u.Quantity] = None,  # volume
        mu: Optional[u.Quantity] = None,  # magnetic dipole moment
        temp: Optional[u.Quantity] = None,
        pol: Optional[u.Quantity] = None,
        verbose: bool = False,
    ):
        """

        Wikipedia: Standard temperature and pressure
        https://en.wikipedia.org/wiki/Standard_temperature_and_pressure
        In chemistry, IUPAC changed its definition of standard temperature and pressure in 1982:[1][2]

        Until 1982, STP was defined as a temperature of 273.15 K (0 °C, 32 °F) and an absolute pressure
        of exactly 1 atm (101.325 kPa).
        Since 1982, STP has been defined as a temperature of 273.15 K (0 °C, 32 °F) and an absolute
        pressure of exactly 105 Pa (100 kPa, 1 bar).
        STP should not be confused with the standard state commonly used in thermodynamic evaluations
        of the Gibbs energy of a reaction.

        NIST uses a temperature of 20 °C (293.15 K, 68 °F) and an absolute pressure of 1 atm
        (14.696 psi, 101.325 kPa).[3] This standard is also called normal temperature and pressure
        (abbreviated as NTP). However, a common temperature and pressure in use by NIST for
        thermodynamic experiments is 298.15 K (25°C, 77°F) and 1 bar (14.5038 psi, 100 kPa).[4][5] NIST
        also uses "15 °C (60 °F)" for the temperature compensation of refined petroleum products,
        despite noting that these two values are not exactly consistent with each other.[6]
        """
        super().__init__()
        self.name = name
        self.gamma = gamma

        self.massDensity = massDensity
        self.molarMass = molarMass
        self.numOfSpinsPerMolecule = numOfSpinsPerMolecule

        assert self.molarMass is not None
        self.spinNumDensity = (
            self.numOfSpinsPerMolecule * self.massDensity / self.molarMass * const.N_A
        ).to("cm**(-3)")

        self.T2 = T2
        self.T1 = T1
        self.vol = vol

        self.totalNumOfSpins = (self.spinNumDensity * self.vol).to("")

        self.mu = mu
        self.temp = temp
        self.pol = pol
        # Specify all physical quantities with units
        self.physicalQuantities = {
            "gamma": "Hz/T",
            "massDensity": "g/cm**3",
            "molarMass": "g/mol",
            "numOfSpinsPerMolecule": "",
            "spinNumDensity": "1/cm**3",
            "T2": "s",
            "T1": "s",
            "vol": "cm**3",
            "mu": "J/T",
            "temp": "K",
            "totalNumOfSpins": "",
            "pol": "",
        }
        # make sure that we use common units for quantities
        self.useCommonUnits()

    def getThermalPol(
        self,
        B_pol: u.Quantity,
        temp: Optional[u.Quantity] = None,
        verbose: bool = False,
    ):
        """
        return thermal polarization
        """
        # pol = hbar * self.gamma * B_pol / (2 * k * temp)  # approximate
        if temp is None:
            temp = self.temp
        assert (
            temp is not None
        ), "Temperature is required to compute thermal polarization. Please provide temp or set it in the Sample object."
        pol = np.tanh(const.hbar * self.gamma * B_pol / (2 * const.k_B * temp))  # exact
        pol = pol.to("")
        # check(pol)
        if verbose:
            print(f"[{self.getThermalPol.__name__}] Thermal polarization at B_pol={B_pol} and temp={temp} is {pol}")
        return pol

    def getM0(
        self,
        pol: Optional[u.Quantity] = None,
        verbose: bool = False,
    ):
        """
        compute magnetization M0
        """
        if pol is None:
            pol = self.pol
        assert (
            pol is not None
        ), "Polarization is required to compute M0. Please provide pol or set it in the Sample object."
        M0 = (self.mu * pol * self.totalNumOfSpins / self.vol).to("A/m")
        # self.M0_SPN = (self.mu * ns_SPN).to("A/m")
        if verbose:
            print(
                f"[{self.__class__.__name__}.{self.getM0.__name__}] Magnetization M0 is {M0}"
            )
        return M0

    def getM0eqb(
        self,
        B_pol: u.Quantity,
        temp: Optional[u.Quantity] = None,
        verbose: bool = False,
    ):
        """
        compute equilibrium magnetization M0eqb at given B_pol and temp
        """
        pol = self.getThermalPol(B_pol, temp, verbose)
        M0eqb = self.getM0(pol, verbose)
        if verbose:
            print(f"[{self.__class__.__name__}.{self.getM0eqb.__name__}] Equilibrium magnetization M0eqb is {M0eqb}")
        return M0eqb


liquid_Xe129 = Sample(
    name="Liquid Xe-129",  # name of the sample
    gamma=gamma_Xe129,  # [Hz/T]. Remember input it with 2 * np.pi
    massDensity=u.Quantity(3.1, "g / cm**3 "),  # mass density at STP
    molarMass=u.Quantity(131.29, "g / mol"),  # molar mass [g/mol]
    numOfSpinsPerMolecule=u.Quantity(1, ""),  # number of spins per molecule
    T2=u.Quantity(10, "minute"),  #
    T1=u.Quantity(15, "minute"),  #
    vol=u.Quantity(1, "cm**3"),
    mu=mu_Xe129,  # magnetic dipole moment
    verbose=False,
)

# CH3OH
methanol = Sample(
    name="C-12 Methanol",  # name of the sample
    gamma=gamma_p,  # [Hz/T]. Remember input it with 2 * np.pi
    massDensity=u.Quantity(0.792, "g / cm**3 "),
    molarMass=u.Quantity(32.04, "g / mol"),  # molar mass
    numOfSpinsPerMolecule=u.Quantity(4, ""),  # number of spins per molecule
    T2=u.Quantity(1, "s"),  #
    T1=u.Quantity(5, "s"),  #
    vol=u.Quantity(1, "cm**3"),
    mu=mu_p,  # magnetic dipole moment
    # boilpt=337.8,  # [K]
    # meltpt=175.6,  # [K]
    verbose=False,
)

# CH3CH2OH
ethanol = Sample(
    name="Ethanol",  # name of the sample
    gamma=gamma_p,  # [Hz/T]. Remember input it with 2 * np.pi
    massDensity=u.Quantity(0.78945, "g / cm**3 "),
    molarMass=u.Quantity(46.069, "g / mol"),  # molar mass
    numOfSpinsPerMolecule=u.Quantity(6, ""),  # number of spins per molecule
    T2=u.Quantity(1, "s"),  #
    T1=u.Quantity(5, "s"),  #
    vol=u.Quantity(1, "cm**3"),
    mu=mu_p,  # magnetic dipole moment
    # boilpt=351.38,  # [K]
    # meltpt=159.01,  # [K]
    verbose=False,
)
