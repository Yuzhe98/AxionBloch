import warnings

from axionbloch.dependency import *
from axionbloch.utils import PhysicalObject, check, check_norm


class MilkyWayAxionHalo(PhysicalObject):
    """Axion dark-matter field (axion wind) from the Milky Way halo.

    Models the pseudomagnetic field experienced by nuclear spins due to the
    coherent oscillation of axion dark matter.  Uses the Standard Halo Model
    (SHM) velocity distribution.

    The class provides:

    - Axion field properties (Compton frequency, quality factor, coherence time).
    - Analytical lineshapes for gradient and non-gradient coupling cases
      (Gramolin et al.).
    - Stochastic amplitude spectra for time-domain simulations.
    """

    # ----------- parameters ----------- #
    # Compton frequency
    nu_a: Quantity[unit.Hz]
    # axion mass
    m_a: Quantity[unit.kg]
    # axion-nucleon coupling strength
    g_aNN: Quantity[unit.GeV ** (-1)]
    # axion quality factor
    Qa: Quantity[unit.one]
    # Local (@ solar radius) galaxy circular rotation speed
    v_0: Quantity[unit.km / unit.s] = 220.0 * unit.km / unit.s
    # Laboratory speed relative to the galactic rest frame
    v_lab: Quantity[unit.km / unit.s] = 233.0 * unit.km / unit.s
    # dark matter axion energy density
    # Standard halo model (SHM): 0.3
    # A commonly-used value: 0.4
    # Refined standard halo model (SHM++) / Particle Data Group 2024: 0.55
    rho_E_DM: Quantity[unit.GeV / unit.cm**3] = 0.3 * unit.GeV / unit.cm**3
    nu_a_eff: Quantity[unit.Hz]
    tau_a_est: Quantity[unit.s]

    # ----------- ----------- ----------- #
    def __init__(
        self,
        name="Milky Way Axion Halo",
        nu_a: Quantity = None,
        m_a: Quantity = None,
        g_aNN: Quantity = None,
        Qa: Quantity = None,
        v_0: Quantity = 220.0 * unit.km / unit.s,
        v_lab: Quantity = 233.0 * unit.km / unit.s,
        windAngle: Quantity = None,
        rho_E_DM: Quantity = 0.3 * unit.GeV / unit.cm**3,
        verbose: bool = False,
    ):
        """Initialise the Milky Way axion halo model.

        Provide either ``nu_a`` or ``m_a`` (or both, in which case they are
        checked for consistency).  If ``Qa`` is omitted it is estimated from the
        Standard Halo Model as ``(c / v_lab)^2``.

        Parameters
        ----------
        name : str
            Human-readable label.
        nu_a : Quantity, optional
            Axion Compton frequency (Hz).
        m_a : Quantity, optional
            Axion mass (kg or equivalent).
        g_aNN : Quantity, optional
            Axion-nucleon coupling constant (GeV⁻¹).
        Qa : Quantity, optional
            Axion quality factor (dimensionless).  Defaults to ``(c/v_lab)^2``.
        v_0 : Quantity
            Local circular-rotation speed of the galaxy (km/s).
        v_lab : Quantity
            Speed of the laboratory relative to the galactic rest frame (km/s).
        windAngle : Quantity, optional
            Angle between the sensitive axis and the axion wind direction (rad).
        rho_E_DM : Quantity
            Local dark-matter energy density (GeV/cm³).
            SHM: 0.3, commonly-used: 0.4, SHM++ / PDG 2024: 0.55.
        verbose : bool
            Print derived quantities after construction.
        """
        super().__init__()
        self.name = name
        self.v_0 = v_0
        self.v_lab = v_lab

        self.rho_E_DM = rho_E_DM

        assert (
            nu_a is not None or m_a is not None
        ), "Either nu_a (axion Compton frequency) or m_a (axion mass) needs to be specified"

        if nu_a is not None and m_a is not None:
            # check consistency
            nu_a_from_m_a = m_a * const.c**2 / const.hbar
            if not np.isclose(nu_a.value, nu_a_from_m_a.value, rtol=1e-6):
                raise ValueError(
                    f"Inconsistent nu_a and m_a: nu_a from m_a = {nu_a_from_m_a}, provided nu_a = {nu_a}"
                )
        elif nu_a is not None and m_a is None:
            self.nu_a = nu_a
            self.m_a = nu_a * const.hbar / const.c**2
        elif nu_a is None and m_a is not None:
            self.m_a = m_a
            self.nu_a = m_a * const.c**2 / const.hbar

        self.g_aNN = g_aNN

        if Qa is None:
            self.Qa = (const.c / self.v_lab) ** 2.0

        self.FWHM = 1.0 / self.Qa

        #
        self.nu_a_eff = self.nu_a * (1 + self.v_lab**2 / const.c**2)
        self.nu_a_eff = self.nu_a_eff

        # coherence time (estimated)
        self.tau_a_est = 1.0 / (np.pi * self.FWHM * self.nu_a_eff)
        self.tau_a_est = self.tau_a_est

        # Specify all physical quantities with units
        self.quantities = {
            "v_0": "km/s",
            "v_lab": "km/s",
            "rho_E_DM": "GeV/cm**3",
            "nu_a": "Hz",
            "g_aNN": "1/GeV",
            "Qa": "",
            "FWHM": "",
            "nu_a_eff": "Hz",
            "tau_a_est": "s",
        }
        # # self.generalQuantities = {"RCF_freq": "Hz", "rate": "Hz", "duration": "s"}
        # # self.Omega_a_rms = 0.5 * self.g_aNN * (2 * const.hbar * const.c * self.rho_E_DM)**(1/2) * self.v_lab * np.cos(windAngle) * Quantity(1e-15, "T")
        # self.useCommonUnits()

    # @classmethod
    def getRabiFreq(
        self,
        g_aNN: Quantity[unit.GeV ** (-1)] | None = None,
        case="grad_perp",
        verbose=False,
    ) -> Quantity:
        """
        get the Rabi frequency of the pseudomagnetic field amplitude in [Hz] for the specified case
        case: "non-grad", "grad_par" or "grad_perp", determines the lineshape function to use
        """
        logPrefix = f"[{self.__class__.__name__}.{self.getRabiFreq.__name__}]"
        if g_aNN is None:
            if self.g_aNN is None:
                raise ValueError("g_aNN cannot be None")
            else:
                g_aNN = self.g_aNN
        # if case == "non-grad":
        #     Omega_rms = 0.5 * self.g_aNN * (2 * const.c * self.rho_E_DM) ** (
        #         1 / 2
        #     ) * self.v_lab * np.cos(self.windAngle)
        # elif case == "grad_par":
        #     Omega_rms = 0.5 * self.g_aNN * (2 * const.c * self.rho_E_DM) ** (
        #         1 / 2
        #     ) * self.v_lab * np.cos(self.windAngle) * self.FWHM**(1 / 2)
        # el
        if case == "grad_perp":
            Omega_rms = (
                0.5
                * g_aNN
                * (2 * const.hbar * const.c * self.rho_E_DM) ** (1 / 2)
                * self.v_lab
            )
            Omega_rms = Omega_rms
        else:
            raise ValueError(
                f"case {case} not recognized, should be 'grad_perp'"
            )  #  'non-grad', 'grad_par' or
        if verbose:
            print(logPrefix, f"axion wind Rabi frequency (case={case}): {Omega_rms}")
        return Omega_rms

    @staticmethod
    def axion_lineshape(
        v_0: Quantity[unit.m / unit.s],
        v_lab: Quantity[unit.m / unit.s],
        nu_a:Quantity[unit.Hz],
        nu:Quantity,
        case:str="non-grad",
        alpha:Quantity=0.0*unit.rad,
        verbose:bool=False,
    ):
        """Calculate the analytical axion power-spectral-density lineshape.

        Implements Eqs. (12), (19), (20) of the Gramolin lineshape paper for
        the non-gradient, parallel-gradient, and perpendicular-gradient coupling
        cases respectively.

        .. warning::
            ``nu`` should not be too far above ``nu_a``.  The ratio
            ``nu / nu_a`` must remain below ~1.03 to avoid numerical overflow in
            the ``sinh(beta)`` term.

        Parameters
        ----------
        v_0 : Quantity [m/s]
            Local circular-rotation speed of the Milky Way.
        v_lab : Quantity [m/s]
            Speed of the laboratory in the galactic rest frame.
        nu_a : Quantity [Hz]
            Axion Compton frequency.
        nu : Quantity array [Hz]
            Frequency array at which to evaluate the lineshape.  Must be
            uniformly spaced.
        case : str
            Coupling geometry: ``'non-grad'``, ``'grad_par'``, or
            ``'grad_perp'``.
        alpha : Quantity [rad]
            Angle between the sensitive axis and the axion wind velocity.
        verbose : bool
            Print diagnostic information.

        Returns
        -------
        Quantity array, shape ``(len(nu),)`` [Hz⁻²]
            Power spectral density lineshape, normalised so that
            ``integral(lineshape * dnu) = 1``.

        References
        ----------
        A. Gramolin et al., https://github.com/gramolin/lineshape
        """
        logPrefix = f"[{MilkyWayAxionHalo.__name__}.{MilkyWayAxionHalo.axion_lineshape.__name__}]"
        # ----------- prepare to generate the axion lineshape ----------- #
        # return the lineshape under certain special circumstances
        v_0, v_lab = Quantity(np.abs(v_0)), Quantity(np.abs(v_lab))
        
        # Q_a = 1e6
        Q_a = (const.c / v_0) ** 2.0
        Q_a = Q_a.to(unit.one)
        FWHM = 1 / Q_a

        full_lineshape = np.zeros_like(nu) * nu.unit**-2
        RBW = np.abs(nu[1] - nu[0])

        ## Find the index of the first non-zero element
        ## the elements in the full_lineshape before nu_a are set to zeros
        # find the index corresponding to frequency > nu_a
        positive_indices = np.where(nu > nu_a)[0]
        if positive_indices.size > 0:
            nu_a_indx = positive_indices[0]
            if verbose:
                print(logPrefix,
                    f"nu_a = {nu_a}, first frequency element > nu_a is nu[{nu_a_indx}] = {nu[nu_a_indx]}")
        # if there is no element >= nu_a, return an array of zeros
        else:
            if verbose:
                print(
                    logPrefix,
                    f"all input frequencies are < nu_a = {nu_a}, returning an array of zeros",
                )
            return full_lineshape
        del positive_indices

        # cut off the array at ~10 axion linewidths
        # the elements in the full_lineshape after the cutoff are set to zeros
        cutoff_indices = np.where(nu > (1 + 10 * FWHM) * nu_a)[0]
        if cutoff_indices.size > 0:
            if verbose:
                print(logPrefix,
                    f"cutoff frequency is {(1 + 10 * FWHM) * nu_a}, first frequency element > cutoff frequency is nu[{cutoff_indices[0]}] = {nu[cutoff_indices[0]]}")
            cutoff_idx = cutoff_indices[0]
        # elsewise set the cutoff index to the last index of the array
        else:
            if verbose:
                print(
                    logPrefix,
                    f"all input frequencies are < cutoff frequency {(1 + 10 * FWHM) * nu_a}, setting cutoff index to the last index of the array",
                )
            cutoff_idx = len(nu) - 1

        # if cutoff_indx == nu_a_indx:
        #     full_lineshape[nu_a_indx] = 1.0 / RBW
        #     check_norm(nu, full_lineshape)
        #     return full_lineshape
        # ------------------- end of preparations ---------------------- #

        def _axion_lineshape(v_0, v_lab, nu_a, freq, case="non-grad", alpha=0.0):
            """
            Calculate analytical lineshapes.
            freq[0] > nu_a
            freq[-1] < 103% * nu_a

            Parameters
            ----------

            Return
            ------
            A float array of the axion lineshape

            Reference
            ---------
            A. Gramolin: https://github.com/gramolin/lineshape

            """
            assert case in [
                "non-grad",
                "grad_par",
                "grad_perp",
            ], "Case should be 'non-grad', 'grad_par', or 'grad_perp'!"

            beta = 2 * const.c * v_lab * np.sqrt(2 * (freq - nu_a) / nu_a) / v_0**2  # Eq. (13) in Gramolin lineshape paper
            beta = beta.to_value(unit.one)
            # WARNING:
            # Analytically, `beta` can take very large magnitudes.
            # However, for numerical calculations using `np.sinh(beta)`,
            # values with |beta| >> 700 will overflow in double precision.
            # To avoid overflow, ensure |beta| is smaller than ~700.
            if np.max(np.abs(beta)) > 700:
                warnings.warn(
                    "Magnitude of beta is too large for np.sinh. "
                    "Values with |beta| > 700 may overflow in double precision.",
                    RuntimeWarning,
                )

            if case == "non-grad":  # Non-gradient case, Eq. (12)
                ax_PSD_lineshape = (
                    2
                    * const.c**2
                    * np.exp(-((0.5 * beta * v_0 / v_lab) ** 2) - (v_lab / v_0) ** 2)
                    * np.sinh(beta)
                    / (np.sqrt(np.pi) * v_0 * v_lab * nu_a)
                )
            elif case == "grad_par":  # Parallel gradient case, Eq. (19)
                factor = (
                    np.cos(alpha) ** 2
                    - (1 / np.tanh(beta) - 1.0 / beta) * (2 - 3 * np.sin(alpha) ** 2) / beta
                )
                ax_PSD_lineshape = (
                    (4 * const.c**2 / (v_0**2 + 2 * (v_lab * np.cos(alpha)) ** 2))
                    * (freq / nu_a - 1)
                    * factor
                    * _axion_lineshape(v_0, v_lab, nu_a, freq)
                )
            elif case == "grad_perp":  # Perpendicular gradient case, Eq. (20)
                factor = (
                    np.sin(alpha) ** 2
                    + (1.0 / np.tanh(beta) - 1.0 / beta)
                    * (2.0 - 3.0 * np.sin(alpha) ** 2)
                    / beta
                )
                ax_PSD_lineshape = (
                    (2 * const.c**2 / (v_0**2 + (v_lab * np.sin(alpha)) ** 2))
                    * (freq / nu_a - 1)
                    * factor
                    * _axion_lineshape(v_0, v_lab, nu_a, freq)
                )
            else:
                return np.zeros_like(nu) * nu.unit**-2

            return ax_PSD_lineshape

        # ---------------- generate axion linshape ----------------- #
        # if RBW is smaller than 0.1 * axion_linewidth (usual case),
        # then the script uses input frequencies to get the lineshape
        if RBW <= 0.1 * FWHM * nu_a:
            if verbose:
                print(
                    logPrefix,
                    f"RBW = {RBW:.3e} is <= 0.1 * FWHM * nu_a = {0.1 * FWHM * nu_a:.3e}, using input frequencies to get the lineshape",
                )
            full_lineshape[nu_a_indx : cutoff_idx + 1] += _axion_lineshape(
                v_0, v_lab, nu_a, nu[nu_a_indx : cutoff_idx + 1], case, alpha
            )
        # elsewise, use finer frequencies to get the lineshape first
        else:
            # chose the indices corresponding to a range
            # within [idx(nu_a) - 1, idx(nu_a + 10 Delta nu_a)]
            if verbose:
                print(
                    logPrefix,
                    f"RBW = {RBW:.3e} is > 0.1 * FWHM * nu_a = {(0.1 * FWHM * nu_a).to(RBW.unit):.3e}, using finer frequencies to get the lineshape",
                )
            start_idx = max(0, nu_a_indx - 1)
            freq_start = nu[start_idx]
            freq_stop = nu[cutoff_idx]
            _factor = np.ceil(RBW / (0.01 * FWHM * nu_a))
            fine_RBW = RBW / _factor
            fine_freqs = (
                np.arange(
                    start=freq_start.to_value(nu.unit),
                    stop=(freq_stop + RBW).to_value(nu.unit),
                    step=fine_RBW.to_value(nu.unit),
                )
                * nu.unit
            )
            fine_lineshape = np.zeros_like(fine_freqs) * fine_freqs.unit**-2
            # find the index corresponding to frequency > nu_a
            positive_indices = np.where(fine_freqs > nu_a)[0]
            if positive_indices.size > 0:
                fine_nu_a_idx = positive_indices[0]
                # Compute finely-sampled lineshape
                fine_lineshape[fine_nu_a_idx:] += _axion_lineshape(
                    v_0, v_lab, nu_a, fine_freqs[fine_nu_a_idx:], case, alpha
                )
                # Bin fine lineshape onto coarse grid
                # Only bin into bins at or above nu_a_indx to avoid non-zero values below nu_a
                for idx in range(start_idx, cutoff_idx + 1):
                    # Find fine frequencies within this coarse bin
                    fine_indices = np.where(
                        (fine_freqs > nu[idx]) & (fine_freqs <= nu[idx] + RBW)
                    )[0]
                    # Integrate fine lineshape over the bin and add to full_lineshape
                    full_lineshape[idx] += (
                        np.sum(fine_lineshape[fine_indices]) * fine_RBW / RBW
                    )
            # if there is no element >= nu_a, return an array of zeros
            else:
                return full_lineshape
            del positive_indices
        # ---------------- end of generation ----------------- #
        check_norm(nu, full_lineshape)
        return full_lineshape

    def getAmpSpectra(
        self,
        frequencies: np.ndarray,
        case: str = "grad_perp",
        numSpectra: int = 1,
        rand_seed: int = None,
        use_stoch: bool = True,
        verbose: bool = False,
    ) -> np.ndarray:
        """Return complex amplitude spectra for the axion pseudo-magnetic field.

        Draws ``numSpectra`` realisations of the stochastic axion field in the
        frequency domain using the SHM lineshape as the power spectral density.
        Each realisation has random phases uniformly distributed in ``[0, 2π)``.
        When ``use_stoch=True``, amplitudes are additionally drawn from an
        exponential distribution, matching the expected statistics of a
        narrowband stochastic signal.

        Parameters
        ----------
        frequencies : Quantity array [Hz]
            Absolute frequencies at which to evaluate the spectrum.
        case : str
            Lineshape coupling case: ``'non-grad'``, ``'grad_par'``, or
            ``'grad_perp'``.
        numSpectra : int
            Number of independent field realisations to generate.
        rand_seed : int, optional
            Seed for the random-number generator (for reproducibility).
        use_stoch : bool
            If ``True``, draw stochastic amplitudes; if ``False``, use the
            deterministic ``sqrt(PSD)`` amplitude.
        verbose : bool
            Unused; reserved for future diagnostic output.

        Returns
        -------
        ndarray, shape ``(numSpectra, len(frequencies))``
            Complex amplitude spectra.
        """

        PSD_lineshape = MilkyWayAxionHalo.axion_lineshape(
            v_0=self.v_0,
            v_lab=self.v_lab,
            nu_a=self.nu_a,
            nu=frequencies,
            case=case,
            alpha=0.0*unit.rad,
        )

        shape = (numSpectra, len(frequencies))

        rng = (
            np.random.default_rng(seed=rand_seed)
            if rand_seed is not None
            else np.random.default_rng()
        )

        # phases over frequency
        phases = np.exp(1j * 2 * np.pi * rng.random(shape))

        # amplitude spectra (complex) over frequency, shape = (numFields, numSteps)
        if use_stoch:
            stochastic = rng.exponential(scale=1.0, size=shape)
            ampSpectra = (
                np.sqrt(stochastic * PSD_lineshape) * phases
            )  # shape = (numFields, numSteps)
        else:
            ampSpectra = np.sqrt(PSD_lineshape) * phases  # shape = (numFields, numSteps)

        return ampSpectra

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
