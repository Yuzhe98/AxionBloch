import numpy as np
from typing import Optional

from src.Envelope import (
    PhysicalQuantity,
    _safe_convert,
    gamma_Xe129,
    gamma_p,
    mu_p,
    mu_Xe129,
    hbar,
    kB,
    mol_to_num,
)

from src.utils import PhysicalObject


class Sample(PhysicalObject):
    """
    Describe the sample used in experiments.
    Only consider samples in one phase.
    """

    def __init__(
        self,
        name: Optional[str] = None,  # name of the sample
        gamma: Optional[
            PhysicalQuantity
        ] = None,  # gyromagnetic ratio. Remember to input it with 2 pi
        massDensity: Optional[PhysicalQuantity] = None,  # mass density at STP
        molarMass: Optional[PhysicalQuantity] = None,  # molar mass
        numOfSpinsPerMolecule: Optional[
            PhysicalQuantity
        ] = None,  # number of spins per molecule
        T2: Optional[PhysicalQuantity] = None,  #
        T1: Optional[PhysicalQuantity] = None,  #
        vol: Optional[PhysicalQuantity] = None,  # volume
        mu: Optional[PhysicalQuantity] = None,  # magnetic dipole moment
        temp: Optional[PhysicalQuantity] = None,
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
            self.numOfSpinsPerMolecule * self.massDensity / self.molarMass * mol_to_num
        ).convert_to("cm**(-3)")

        self.T2 = T2
        self.T1 = T1
        self.vol = vol

        self.totalNumOfSpins = (self.spinNumDensity * self.vol).convert_to("")

        self.mu = mu
        self.temp = temp
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
        }
        # make sure that we use common units for quantities
        self.useCommonUnits()

    def getThermalPol(
        self,
        B_pol: PhysicalQuantity,
        temp: PhysicalQuantity,
    ):
        """
        return thermal polarization
        """
        # pol = hbar * self.gamma * B_pol / (2 * k * temp)  # approximate
        pol = np.tanh(hbar * self.gamma * B_pol / (2 * kB * temp))  # exact
        pol = pol.convert_to("")
        # check(pol)
        return pol

    def getM0(
        self,
        pol,
    ):
        """
        compute magnetization M0
        """
        M0 = (self.mu * pol * self.totalNumOfSpins).convert_to("A/m")
        # self.M0_SPN = (self.mu * ns_SPN).convert_to("A/m")
        return M0


liquid_Xe129 = Sample(
    name="Liquid Xe-129",  # name of the sample
    gamma=gamma_Xe129,  # [Hz/T]. Remember input it with 2 * np.pi
    massDensity=PhysicalQuantity(3.1, "g / cm**3 "),  # mass density at STP
    molarMass=PhysicalQuantity(131.29, "g / mol"),  # molar mass [g/mol]
    numOfSpinsPerMolecule=PhysicalQuantity(1, ""),  # number of spins per molecule
    T2=PhysicalQuantity(1000, "s"),  #
    T1=PhysicalQuantity(5000, "s"),  #
    vol=PhysicalQuantity(1, "cm**3"),
    mu=mu_Xe129,  # magnetic dipole moment
    verbose=False,
)

# CH3OH
methanol = Sample(
    name="C-12 Methanol",  # name of the sample
    gamma=gamma_p,  # [Hz/T]. Remember input it with 2 * np.pi
    massDensity=PhysicalQuantity(0.792, "g / cm**3 "),
    molarMass=PhysicalQuantity(32.04, "g / mol"),  # molar mass
    numOfSpinsPerMolecule=PhysicalQuantity(4, ""),  # number of spins per molecule
    T2=PhysicalQuantity(1, "s"),  #
    T1=PhysicalQuantity(5, "s"),  #
    vol=PhysicalQuantity(1, "cm**3"),
    mu=mu_p,  # magnetic dipole moment
    # boilpt=337.8,  # [K]
    # meltpt=175.6,  # [K]
    verbose=False,
)

# CH3CH2OH
ethanol = Sample(
    name="Ethanol",  # name of the sample
    gamma=gamma_p,  # [Hz/T]. Remember input it with 2 * np.pi
    massDensity=PhysicalQuantity(0.78945, "g / cm**3 "),
    molarMass=PhysicalQuantity(46.069, "g / mol"),  # molar mass
    numOfSpinsPerMolecule=PhysicalQuantity(6, ""),  # number of spins per molecule
    T2=PhysicalQuantity(1, "s"),  #
    T1=PhysicalQuantity(5, "s"),  #
    vol=PhysicalQuantity(1, "cm**3"),
    mu=mu_p,  # magnetic dipole moment
    # boilpt=351.38,  # [K]
    # meltpt=159.01,  # [K]
    verbose=False,
)
