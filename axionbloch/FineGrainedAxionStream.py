"""Fine-grained axion-stream model.

A fine-grained axion stream is a single-velocity coherent component of the
dark-matter axion field.  Unlike the diffuse Milky Way halo modelled in
:mod:`axionbloch.MilkyWayAxionHalo`, a stream has a well-defined velocity and
a correspondingly narrow spectral width.

This module provides :class:`FineGrainedAxionStream` for computing the
properties (coherence time, effective frequency, quality factor) of such a
stream.
"""

from axionbloch.dependency import *
from axionbloch.utils import PhysicalObject


class FineGrainedAxionStream(PhysicalObject):
    """Single fine-grained axion stream (axion field component).

    Represents a coherent axion stream with a fixed laboratory velocity
    ``v_lab`` and dark-matter density ``rho_E_DM``.  The quality factor
    ``Q_a = (c/v_lab)²`` is derived automatically unless overridden by ``Q_a``.

    Attributes
    ----------
    nu_a : Quantity [Hz]
        Axion Compton frequency.
    nu_a_eff : Quantity [Hz]
        Effective (blue-shifted) oscillation frequency.
    tau_a_est : Quantity [s]
        Estimated coherence time.
    FWHM : Quantity
        Fractional spectral width (= 1/Q_a).
    """

    def __init__(
        self,
        name="axion stream",
        nu_a: Quantity = None,  # compton frequency
        g_aNN: Quantity = None,
        Q_a: Quantity = None,
        # v_0: Quantity = 220.0 * unit.km / unit.s,  # Local (@ solar radius) galaxy circular rotation speed
        v_lab: Quantity = 233.0
        * unit.km
        / unit.s,  # Laboratory speed relative to the galactic rest frame
        # dark matter axion density in [GeV/cm**3]
        # Standard halo model (SHM): 0.3
        # A commonly-used value: 0.4
        # Refined standard halo model (SHM++) / Particle Data Group 2024: 0.55
        rho_E_DM: Quantity = 0.3 * unit.GeV / unit.cm**3,
        numStreams: int = 1,
        verbose: bool = False,
    ):
        """Initialize axion stream object.

        Parameters
        ----------
        nu_a : Quantity
            Axion Compton frequency.
        g_aNN : Quantity
            Axion-nucleon coupling in 1/GeV.
        Q_a : Quantity, optional
            Axion quality factor (dimensionless). Derived from ``v_lab`` if not given.
        v_lab : Quantity
            Laboratory speed relative to the galactic rest frame (default 233 km/s).
        rho_E_DM : Quantity
            Dark-matter energy density (default 0.3 GeV/cm³).
        numStreams : int
            Number of axion streams to simulate (default 1).
        verbose : bool
            Print input parameters and computed properties.
        """
        logPrefix = f"[{self.__class__.__name__}.{self.__init__.__name__}] "
        # super().__init__()
        # self.name = name
        # self.v_lab = v_lab

        # self.rho_E_DM = rho_E_DM
        # self.nu_a = nu_a
        # self.g_aNN = g_aNN

        # if Q_a is None:
        #     self.Q_a = (const.c / self.v_lab).to(unit.one) ** 2

        # self.FWHM = 1.0 / self.Q_a

        # self.nu_a_eff = self.nu_a * (1.0 + (self.v_lab / const.c).to(unit.one) ** 2)
        # self.nu_a_eff = self.nu_a_eff.to(unit.Hz)

        # # coherence time (estimated)
        # self.tau_a_est = 1.0 / (np.pi * self.FWHM * self.nu_a_eff)
        # self.tau_a_est = self.tau_a_est.to(unit.s)
        print(f"{logPrefix} not implemented yet")


