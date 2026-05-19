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
    ``Q_a = (c/v_lab)²`` is derived automatically unless overridden by ``Qa``.

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
        gaNN: Quantity = None,
        Qa: Quantity = None,
        # v_0: Quantity = 220.0 * unit.km / unit.s,  # Local (@ solar radius) galaxy circular rotation speed
        v_lab: Quantity = 233.0 * unit.km / unit.s,  # Laboratory speed relative to the galactic rest frame
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
        gaNN : Quantity
            Axion-nucleon coupling in 1/GeV.
        Qa : Quantity, optional
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
        super().__init__()
        self.name = name
        self.v_lab = v_lab

        self.rho_E_DM = rho_E_DM
        self.nu_a = nu_a
        self.gaNN = gaNN

        if Qa is None:
            self.Q_a = (const.c / self.v_lab).to(unit.one) ** 2

        self.FWHM = 1.0 / self.Q_a

        self.nu_a_eff = self.nu_a * (1.0 + (self.v_lab / const.c).to(unit.one) ** 2)
        self.nu_a_eff = self.nu_a_eff.to(unit.Hz)

        # coherence time (estimated)
        self.tau_a_est = 1.0 / (np.pi * self.FWHM * self.nu_a_eff)
        self.tau_a_est = self.tau_a_est.to(unit.s)

        # Specify all physical quantities with units
        self.quantities = {
            "v_0": "km/s",
            "v_lab": "km/s",
            "rho_E_DM": "GeV/cm**3",
            "nu_a": "Hz",
            "gaNN": "GeV**-1",
            "Qa": "",
            "FWHM": "",
            "nu_a_eff": "Hz",
            "tau_a_est": "s",
        }
        self.useCommonUnits()

    # def GetAxionWind(
    #     self,
    #     year=None,
    #     month=None,
    #     day=None,
    #     time_hms=None,
    #  year=None,
    #     month=None,
    #     day=None,
    #     time_hms=None,  # Use UTC time!
    #     # example
    #     # year=2024, month=9, day=10, time='14:35:16.235812',
    #     timeastro=None,
    #     # station: Station = None,
    #     latitude=None,
    #     longitude=None,
    #     elevation=None,
    #     verbose=False,
    # ):
    #     """
    #     Parameters
    #     ----------
    #     time_hms: needs to be in the format "15:47:18"
    #         if none is specified, use current time

    #     lat: latitude of experiment location
    #         if none is specified, use Mainz: 49.9916 deg north

    #     lon: longitude of experiment location
    #         if none is specified, use Mainz: 8.2353 deg east

    #     elev: height of experiment location
    #         if none is specified, use Uni Campus Mainz: 130 m

    #     Returns
    #     ---------
    #         1. the velocity 'v_lab' and 'v_ALP_perp' between lab frame and
    #     DM halo (SHM), in the galactic rest frame, for the specified
    #     coordinates and time
    #         2. angle [rad] between the CASPEr sensitive axis (z-direction =
    #     earth surface normal)
    #         3. v_ALP, v_ALP_perp, alpha_ALP go into self.

    #     """
    #     if verbose:
    #         print("now calculating wind angle")

    #     year = year or self.year
    #     month = month or self.month
    #     day = day or self.day
    #     time_hms = time_hms or self.time_hms

    #     if self.timeastro is None:
    #         if (year or month or day or time_hms) is None:
    #             time_DMmeasure = Time.now()  # UTC time
    #             # example of the astropy.time.Time.now() return value
    #             # 2024-09-11 14:27:44.732284
    #             print(
    #                 f"no date and time input provided, using current date and time: {time_DMmeasure}"
    #             )
    #         else:
    #             time_DMmeasure = rf"{year}-{month}-{day}T{time_hms}"
    #         if verbose:
    #             print(f"time input: {time_DMmeasure}")
    #         self.timeastro = Time(time_DMmeasure, format="isot", scale="utc")
    #         # example of timeastro
    #         # 2024-09-11T14:35:16.236

    #     if self.station is None:
    #         self.station = Mainz
    #         if verbose:
    #             print("no station specified, defaulting to CASPEr Mainz")

    #     lat = latitude or self.station.latitude_deg
    #     lon = longitude or self.station.longitude_deg
    #     elev = elevation or self.station.elevation

    #     # DMtimefrac = wind.FracDay(Y=2022, M=12, D=23)
    #     # if verbose:
    #     #     print("time of DM measurement (fractional days): ", DMtimefrac)

    #     # LABvel = wind.ACalcV(DMtimefrac)
    #     # if verbose:
    #     #     print("velocity (lab frame) @DM time: ", LABvel)

    #     DMtime, unit_North, unit_East, unit_Up, Vhalo = wind.get_CASPEr_vect(
    #         time=self.timeastro,
    #         lat=lat,
    #         lon=lon,
    #         elev=elev,
    #     )

    #     # print(type(Vhalo))
    #     Vlab = Vhalo.get_d_xyz()  # convert into a vector
    #     Bz = (
    #         unit_Up.get_xyz()
    #     )  # our leading field is pointing up perpendicular to earth's surface

    #     alpha_ALP = angle_between(Vlab, Bz).value
    #     v_ALP = np.linalg.norm(Vlab.value) * 1e3
    #     v_ALP_perp = v_ALP * math.sin(alpha_ALP)

    #     if verbose:
    #         # print("time of DM measurement: ", DMtime)
    #         print("Bz vector @DM time (galaxy frame):", Bz)
    #         print("v_halo @DM time (galaxy frame):", Vhalo)
    #         print("v_lab @DM time:", Vlab)
    #         print("angle between sensitive axis & lab velocity @DM time: ", alpha_ALP)

    #     ###############################################################################################
    #     # do not delete
    #     self.windangle = alpha_ALP
    #     self.v_lab = v_ALP
    #     self.v_ALP_perp = v_ALP_perp
    #     self.alpha_ALP = alpha_ALP
    #     return v_ALP, v_ALP_perp, alpha_ALP

    # # check Gramolin paper for functions:
    # # axion_beta, axion_lambda, axion_C_para, axion_C_perp
    # def axion_beta(self, nu_a, nu):
    #     if nu <= nu_a:
    #         return 0.0
    #     else:
    #         return (
    #             (2 * self.c * self.v_lab)
    #             / self.v_0**2
    #             * np.sqrt(2 * (nu - nu_a) / nu_a)
    #         )

    # def axion_lambda(
    #     self,
    #     nu_a,
    #     nu,
    #     alpha,
    # ):
    #     p0 = (2 * self.c**2) / (np.sqrt(np.pi) * self.v_0 * self.v_lab * nu_a)
    #     p1 = np.exp(
    #         -self.axion_beta(nu_a=nu_a, nu=nu) ** 2 * self.v_0**2 / (4 * self.v_lab**2)
    #         - (self.v_lab / self.v_0) ** 2
    #     )
    #     p2 = np.sinh(self.axion_beta(nu_a=nu_a, nu=nu))
    #     return p0 * p1 * p2

    # def axion_C_para(
    #     self,
    #     alpha,
    # ):
    #     return self.v_0**2 / 2.0 + self.v_lab**2 * np.cos(alpha) ** 2

    # def axion_C_perp(
    #     self,
    #     alpha,
    # ):
    #     return self.v_0**2 + self.v_lab**2 * np.sin(alpha) ** 2

    # def lineshape_t(self, nu, nu_a=None, case="grad_perp") -> np.ndarray:
    #     """
    #     axion_lineshape at time t
    #     """
    #     # v_ALP, v_ALP_perp, alpha_ALP = get_ALP_wind(\
    #     if nu_a is None:
    #         nu_a = self.nu_a
    #     if self.station is None:
    #         self.station = Mainz
    #         # print('no station specified, defaulting to CASPEr Mainz')
    #     self.GetAxionWind(
    #         year=self.year,
    #         month=self.month,
    #         day=self.day,
    #         time_hms=self.time_hms,
    #         latitude=self.station.latitude_deg,
    #         longitude=self.station.longitude_deg,
    #         elevation=self.station.elevation,
    #         verbose=False,
    #     )

    #     return axion_lineshape(
    #         v_0=self.v_0,
    #         v_lab=self.v_lab,
    #         nu_a=nu_a,
    #         nu=nu,
    #         case=case,
    #         alpha=self.windangle,
    #     )  # type: ignore

    #     conv_step_len = 1.0 * conv_step / abs(xstamp[1]-xstamp[0])
    #     if conv_step_len < 1.0:
    #         check(conv_step_len)
    #         raise ValueError('conv_step_len too short. Increase conv_step.')
    #     conv_step_len = int(conv_step_len)

    #     conv_step_num = int(1.0 * abs(xstamp[-1]-xstamp[0]) / conv_step)
    #     for i in range(conv_step_num):
    #         if i * conv_step_len + conv_line_len > len(signal)-1:
    #             break
    #         conv_xstamp.append([i * conv_step + xstamp[0]])
    #         p = signal[i * conv_step_len:i * conv_step_len + conv_line_len] * conv_line
    #         conv_signal.append(np.sum(p)/np.sum(conv_line)**2)
    #     return conv_xstamp, conv_signal
