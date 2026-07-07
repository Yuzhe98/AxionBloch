import warnings

import astropy.coordinates as coord

from axionbloch.dependency import *
from axionbloch.utils import check_norm
from axionbloch.MilkyWay import MilkyWay


class MilkyWayAxionHalo:
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
    Q_a: Quantity[unit.one]
    # Local (@ solar radius) galaxy circular rotation speed
    v_0: Quantity[unit.km / unit.s] = 220.0 * unit.km / unit.s
    # Laboratory speed relative to the galactic rest frame
    v_lab: Quantity[unit.km / unit.s] = 233.0 * unit.km / unit.s
    # dark matter axion energy density
    # Standard halo model (SHM): 0.3
    # A commonly-used value: 0.4
    # Refined standard halo model (SHM++) / Particle Data Group 2024: 0.55
    rho_E_DM: Quantity[unit.GeV / unit.cm**3] = 0.3 * unit.GeV / unit.cm**3
    windAngle: Quantity[unit.rad] | None = None
    nu_a_eff: Quantity[unit.Hz]
    FWHM_a: Quantity[ppm]
    FWHM_frequency: Quantity[unit.Hz]
    tau_a_est: Quantity[unit.s]
    tau_a: Quantity[unit.s]

    # ----------- ----------- ----------- #
    def __init__(
        self,
        name="Milky Way Axion Halo",
        nu_a: Quantity = None,
        m_a: Quantity = None,
        g_aNN: Quantity = None,
        Q_a: Quantity = None,
        v_0: Quantity = 220.0 * unit.km / unit.s,
        v_lab: Quantity = 233.0 * unit.km / unit.s,
        windAngle: Quantity = None,
        rho_E_DM: Quantity = 0.3 * unit.GeV / unit.cm**3,
        verbose: bool = False,
    ):
        """Initialize the Milky Way axion halo model.

        Provide either ``nu_a`` or ``m_a`` (or both, in which case they are
        checked for consistency).  If ``Q_a`` is omitted it is estimated from the
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
        Q_a : Quantity, optional
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

        logPrefix = f"[{self.__class__.__name__}.{self.__init__.__name__}]"
        self.name = name
        self.v_0 = v_0
        self.v_lab = v_lab
        self.windAngle = windAngle

        self.rho_E_DM = rho_E_DM

        assert (
            nu_a is not None or m_a is not None
        ), "Either nu_a (axion Compton frequency) or m_a (axion mass) needs to be specified"

        if nu_a is not None and m_a is not None:
            # check consistency
            nu_a_from_m_a = m_a * const.c**2 / const.h
            if not np.isclose(nu_a.value, nu_a_from_m_a.to_value(nu_a.unit), rtol=1e-6):
                raise ValueError(
                    f"Inconsistent nu_a and m_a: nu_a from m_a = {nu_a_from_m_a}, provided nu_a = {nu_a}"
                )
        elif nu_a is not None and m_a is None:
            self.nu_a = nu_a
            self.m_a = nu_a * const.h / const.c**2
        elif nu_a is None and m_a is not None:
            self.m_a = m_a
            self.nu_a = m_a * const.c**2 / const.h

        self.g_aNN = g_aNN

        if Q_a is None:
            self.Q_a = (const.c / self.v_0) ** 2.0
        else:
            self.Q_a = Q_a

        self.FWHM = 1.0 / self.Q_a
        self.FWHM_a = self.FWHM.to(ppm)
        self.FWHM_frequency = (self.FWHM * self.nu_a).to(unit.Hz)

        # effective axion frequency considering second-order Doppler effect
        self.nu_a_eff = self.nu_a * (1 + 0.5 * self.v_lab**2 / const.c**2)
        self.nu_a_eff = self.nu_a_eff

        # coherence time (estimated)
        self.tau_a_est = 1.0 / (np.pi * self.FWHM * self.nu_a_eff)
        self.tau_a_est = self.tau_a_est
        self.tau_a = 1.0 / (np.pi * self.FWHM * self.nu_a)

    @staticmethod
    def _cartesian_xyz(cartesian, xyz_unit: unit.Unit) -> Quantity:
        """Return astropy Cartesian components as ``(..., 3)`` Quantity arrays."""
        xyz = (
            cartesian.d_xyz.to(xyz_unit)
            if hasattr(cartesian, "d_xyz")
            else cartesian.xyz.to(xyz_unit)
        )
        if xyz.ndim == 1:
            return xyz
        return np.moveaxis(xyz.value, 0, -1) * xyz_unit

    @staticmethod
    def _normalize_vectors(vectors: Quantity | np.ndarray) -> Quantity | np.ndarray:
        """Normalize a vector or vector stack with the vector axis last."""
        norm = np.sqrt(np.sum(vectors**2, axis=-1))
        return vectors / np.expand_dims(norm, axis=-1)

    @staticmethod
    def _station_basis_itrs(station) -> dict[str, np.ndarray]:
        """Local north/east/up unit vectors in the station's ITRS frame."""
        lat = station.location.lat.to_value(unit.rad)
        lon = station.location.lon.to_value(unit.rad)

        east = np.array([-np.sin(lon), np.cos(lon), 0.0])
        north = np.array(
            [
                -np.sin(lat) * np.cos(lon),
                -np.sin(lat) * np.sin(lon),
                np.cos(lat),
            ]
        )
        up = np.array(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        )
        return {"north": north, "east": east, "west": -east, "up": up, "zenith": up}

    @staticmethod
    def _galcen_to_icrs_rotation() -> np.ndarray:
        """Rotation matrix from Galactocentric Cartesian axes to ICRS/GCRS axes."""

        def _icrs_hat(l_deg, b_deg):
            sc = coord.SkyCoord(
                l=l_deg * unit.deg, b=b_deg * unit.deg, frame="galactic"
            ).icrs
            return np.array(
                [sc.cartesian.x.value, sc.cartesian.y.value, sc.cartesian.z.value]
            )

        return np.column_stack([_icrs_hat(180, 0), _icrs_hat(90, 0), _icrs_hat(0, 90)])

    @staticmethod
    def getHaloVelocity(
        time: Time | None = None,
        station=None,
        galcen_frame: coord.Galactocentric | None = None,
        as_wind: bool = True,
    ) -> Quantity:
        """Compute the lab/halo relative velocity using astropy coordinates.

        This is the astropy-based integration of the TASSLE ``get_halo_vel``
        idea.  When ``station`` is provided, the station's GCRS state is
        transformed into the Galactocentric frame, so Earth's surface rotation
        is included.  Without a station, the Earth-centre velocity from
        :class:`~axionbloch.MilkyWay.MilkyWay` is used.

        Parameters
        ----------
        time : astropy.time.Time, optional
            Observation epoch; defaults to ``Time.now()``.
        station : axionbloch.Station.Station, optional
            Lab location.  Provide this to include daily rotation.
        galcen_frame : astropy.coordinates.Galactocentric, optional
            Custom galactocentric frame.
        as_wind : bool
            If ``True`` return the axion-wind velocity (halo relative to lab),
            i.e. the negative of the lab velocity in the halo frame.

        Returns
        -------
        Quantity
            Velocity vector in km/s.  Shape is ``(3,)`` for scalar time and
            ``(N, 3)`` for vector time.
        """
        if time is None:
            time = Time.now()
        if galcen_frame is None:
            galcen_frame = coord.Galactocentric()

        if station is None:
            lab_velocity = MilkyWay(time=time, galcen_frame=galcen_frame).get_v_lab()
        else:
            gcrs = station.location.get_gcrs(time)
            galcen = coord.SkyCoord(gcrs).transform_to(galcen_frame)
            diff = galcen.cartesian.differentials["s"]
            lab_velocity = MilkyWayAxionHalo._cartesian_xyz(diff, unit.km / unit.s)

        return -lab_velocity if as_wind else lab_velocity

    @staticmethod
    def getLabBasis(
        time: Time | None = None,
        station=None,
        galcen_frame: coord.Galactocentric | None = None,
    ) -> dict[str, np.ndarray]:
        """Return local north/east/up unit vectors in Galactocentric axes.

        The basis vectors are constructed as small ITRS displacements at the
        station and transformed with astropy to the Galactocentric frame.  This
        follows the spirit of TASSLE's ``get_CASPEr_vect`` while using the
        package's :class:`~axionbloch.Station.Station` objects.
        """
        if station is None:
            raise ValueError("station must be set to compute the lab basis")
        if time is None:
            time = Time.now()
        if galcen_frame is None:
            galcen_frame = coord.Galactocentric()

        from astropy.coordinates import GCRS

        loc_itrs = station.location.get_itrs(obstime=time)
        origin_gcrs = loc_itrs.transform_to(GCRS(obstime=time)).cartesian.xyz.to(unit.m)
        gcrs_to_galcen = MilkyWayAxionHalo._galcen_to_icrs_rotation().T

        basis_itrs = MilkyWayAxionHalo._station_basis_itrs(station)
        basis_galcen = {}
        for name, direction in basis_itrs.items():
            displacement = 1.0e3 * unit.m
            displaced = coord.ITRS(
                x=loc_itrs.cartesian.x + direction[0] * displacement,
                y=loc_itrs.cartesian.y + direction[1] * displacement,
                z=loc_itrs.cartesian.z + direction[2] * displacement,
                obstime=time,
            )
            displaced_gcrs = displaced.transform_to(GCRS(obstime=time))
            vec = (displaced_gcrs.cartesian.xyz.to(unit.m) - origin_gcrs).value
            if vec.ndim == 1:
                vec = vec / np.linalg.norm(vec)
                basis_galcen[name] = gcrs_to_galcen @ vec
            else:
                vec = np.moveaxis(vec, 0, -1)
                vec = MilkyWayAxionHalo._normalize_vectors(vec)
                basis_galcen[name] = vec @ gcrs_to_galcen.T
        return basis_galcen

    @staticmethod
    def projectHaloVelocity(
        time: Time | None = None,
        station=None,
        axis: str | np.ndarray | Quantity = "up",
        galcen_frame: coord.Galactocentric | None = None,
    ) -> Quantity:
        """Project the astropy halo wind onto a local sensitive axis.

        ``axis`` may be ``'up'``/``'zenith'``, ``'north'``, ``'east'``,
        ``'west'``, ``'parallel'``/
        ``'z'`` (alias for ``'up'``), ``'perp'`` (magnitude perpendicular to
        local up), ``'magnitude'``, or an explicit unit vector in Galactocentric
        Cartesian coordinates.
        """
        wind = MilkyWayAxionHalo.getHaloVelocity(
            time=time,
            station=station,
            galcen_frame=galcen_frame,
            as_wind=True,
        )

        if isinstance(axis, str):
            axis_key = axis.lower()
            if axis_key in {"magnitude", "speed"}:
                return np.sqrt(np.sum(wind**2, axis=-1)).to(unit.km / unit.s)

            basis = MilkyWayAxionHalo.getLabBasis(
                time=time, station=station, galcen_frame=galcen_frame
            )
            if axis_key in {"parallel", "z"}:
                axis_key = "up"
            if axis_key in {"perp", "perpendicular"}:
                up = basis["up"]
                parallel = np.sum(wind * up, axis=-1)
                return np.sqrt(np.sum(wind**2, axis=-1) - parallel**2).to(
                    unit.km / unit.s
                )
            if axis_key not in basis:
                raise ValueError(
                    "axis must be 'up'/'zenith', 'north', 'east', 'west', "
                    "'perp', 'magnitude', or an explicit vector"
                )
            axis_vec = basis[axis_key]
        else:
            axis_vec = axis.to_value(unit.one) if isinstance(axis, Quantity) else axis
            axis_vec = MilkyWayAxionHalo._normalize_vectors(np.asarray(axis_vec))

        return np.sum(wind * axis_vec, axis=-1).to(unit.km / unit.s)

    @staticmethod
    def gradientPowerCoefficient(
        v_0: Quantity,
        v_lab: Quantity,
        alpha: Quantity,
        case: str = "grad_perp",
    ) -> Quantity:
        """Return the Gramolin gradient-power coefficient ``C``.

        The paper's total gradient signal powers are proportional to
        ``C_parallel = v_0**2 / 2 + v_lab**2 cos(alpha)**2`` and
        ``C_perp = v_0**2 + v_lab**2 sin(alpha)**2``.  Coupling constants and
        the common ``rho_DM / c**2`` factor are intentionally omitted here so
        this can be used as a relative modulation coefficient.
        """
        v_0 = v_0.to(unit.km / unit.s)
        v_lab = v_lab.to(unit.km / unit.s)
        if case == "grad_par":
            return (v_0**2 / 2 + v_lab**2 * np.cos(alpha) ** 2).to(
                (unit.km / unit.s) ** 2
            )
        if case == "grad_perp":
            return (v_0**2 + v_lab**2 * np.sin(alpha) ** 2).to((unit.km / unit.s) ** 2)
        if case == "non-grad":
            return np.ones(np.shape(v_lab.value)) * unit.one
        raise ValueError("case must be 'non-grad', 'grad_par', or 'grad_perp'")

    def setKinematicsWithMilkyWay(self, mw, verbose: bool = False) -> None:
        """Update ``v_lab``, ``windAngle``, ``nu_a_eff`` and ``tau_a_est`` from MilkyWay.

        This mirrors the convenience style used by
        :meth:`axionbloch.EarthBoundAxionHalo.EarthBoundAxionHalo.findGradientsWithMilkyWay`:
        a prepared :class:`~axionbloch.MilkyWay.MilkyWay` object supplies the
        astropy kinematic context, while this class keeps the axion-field
        quantities.
        """
        self.v_lab = mw.get_v_lab_magnitude()
        if mw.station is not None:
            self.windAngle = mw.get_wind_angle()
        self.Q_a = (const.c / self.v_0) ** 2.0
        self.FWHM = 1.0 / self.Q_a
        self.FWHM_a = self.FWHM.to(ppm)
        self.FWHM_frequency = (self.FWHM * self.nu_a).to(unit.Hz)
        self.nu_a_eff = self.nu_a * (1 + 0.5 * self.v_lab**2 / const.c**2)
        self.tau_a_est = 1.0 / (np.pi * self.FWHM * self.nu_a_eff)
        self.tau_a = 1.0 / (np.pi * self.FWHM * self.nu_a)

        if verbose:
            print(f"[{self.__class__.__name__}.setKinematicsWithMilkyWay]")
            print(f"v_lab     = {self.v_lab.to(unit.km / unit.s):.3f}")
            if self.windAngle is not None:
                print(f"windAngle = {self.windAngle.to(unit.deg):.2f}")

    def setKinematicsFromAstropy(
        self,
        time: Time | None = None,
        station=None,
        galcen_frame: coord.Galactocentric | None = None,
        verbose: bool = False,
    ) -> None:
        """Update halo kinematics from astropy time/station inputs."""

        mw = MilkyWay(
            time=time if time is not None else Time.now(),
            station=station,
            galcen_frame=galcen_frame,
        )
        self.setKinematicsWithMilkyWay(mw, verbose=verbose)

    def findKinematicsOverTime(
        self,
        station,
        meas_times: list[Time] | Time,
        sensitive_axis: str | np.ndarray | Quantity = "up",
        include_rotation: bool = True,
        galcen_frame: coord.Galactocentric | None = None,
        verbose: bool = False,
    ) -> dict:
        """Milky Way halo kinematics over a list of epochs.

        This is the time-domain companion to the static SHM parameters.  It
        exposes the daily/annual modulation ingredients entering the gradient
        lineshape: lab speed, angle to the sensitive axis, and parallel /
        perpendicular wind projections.

        Parameters
        ----------
        station : axionbloch.Station.Station
            Lab location and default sensitive-axis orientation.
        meas_times : list of astropy.time.Time or astropy.time.Time
            Epochs at which to evaluate the kinematics.
        sensitive_axis : str or vector
            ``'up'`` by default.  String axes are interpreted by
            :meth:`projectHaloVelocity`.
        include_rotation : bool
            If ``True`` use the station GCRS state and include Earth's surface
            rotation.  If ``False`` use the Earth-centre velocity supplied by
            :class:`~axionbloch.MilkyWay.MilkyWay`.
        galcen_frame : astropy.coordinates.Galactocentric, optional
            Custom galactocentric frame.
        verbose : bool
            Print per-step progress.

        Returns
        -------
        dict
            Keys include ``times``, ``v_lab_magnitude``, ``wind_angle``,
            ``wind_parallel``, ``wind_perp``, and ``nu_a_eff``.
        """

        if isinstance(meas_times, Time):
            if meas_times.isscalar:
                iter_times = [meas_times]
                times = Time(iter_times)
            else:
                times = meas_times.reshape(-1)
                iter_times = list(times)
        else:
            iter_times = list(meas_times)
            times = Time(iter_times)

        speeds, angles, parallel_vals, perp_vals = [], [], [], []
        cos_alpha_vals = []

        for i, meas_time in enumerate(iter_times):
            if verbose:
                print(
                    f"[{self.__class__.__name__}.{self.findKinematicsOverTime.__name__}] "
                    f"step {i + 1}/{len(iter_times)}  t={meas_time.iso}"
                )

            if include_rotation:
                speed = self.projectHaloVelocity(
                    time=meas_time,
                    station=station,
                    axis="magnitude",
                    galcen_frame=galcen_frame,
                )
                parallel = self.projectHaloVelocity(
                    time=meas_time,
                    station=station,
                    axis=sensitive_axis,
                    galcen_frame=galcen_frame,
                )
                perp = np.sqrt(speed**2 - parallel**2).to(unit.km / unit.s)
                cos_alpha = np.clip((parallel / speed).to_value(unit.one), -1.0, 1.0)
                angle = np.arccos(np.abs(cos_alpha)) * unit.rad
            else:
                mw = MilkyWay(
                    time=meas_time, station=station, galcen_frame=galcen_frame
                )
                speed = mw.get_v_lab_magnitude()
                angle = mw.get_wind_angle()
                parallel = speed * np.cos(angle)
                perp = speed * np.sin(angle)
                cos_alpha = np.cos(angle).to_value(unit.one)

            speeds.append(speed.to(unit.km / unit.s))
            angles.append(angle.to(unit.rad))
            parallel_vals.append(parallel.to(unit.km / unit.s))
            perp_vals.append(perp.to(unit.km / unit.s))
            cos_alpha_vals.append(cos_alpha)

        v_lab = np.array([v.value for v in speeds]) * speeds[0].unit
        wind_angle = np.array([a.value for a in angles]) * angles[0].unit
        wind_parallel = (
            np.array([v.value for v in parallel_vals]) * parallel_vals[0].unit
        )
        wind_perp = np.array([v.value for v in perp_vals]) * perp_vals[0].unit
        nu_a_eff = self.nu_a * (1 + 0.5 * v_lab**2 / const.c**2)

        return {
            "times": times,
            "v_lab_magnitude": v_lab,
            "wind_angle": wind_angle,
            "cos_alpha": np.array(cos_alpha_vals),
            "wind_parallel": wind_parallel,
            "wind_perp": wind_perp,
            "nu_a_eff": nu_a_eff.to(self.nu_a.unit),
        }

    def findLineshapeOverTime(
        self,
        frequencies: Quantity,
        station,
        meas_times: list[Time] | Time,
        case: str = "grad_perp",
        sensitive_axis: str | np.ndarray | Quantity = "up",
        include_rotation: bool = True,
        galcen_frame: coord.Galactocentric | None = None,
        verbose: bool = False,
    ) -> dict:
        """Evaluate the SHM axion lineshape over a sequence of epochs.

        The modulation comes from the astropy-derived lab speed and the
        time-dependent wind angle.  For gradient cases this captures the daily
        orientation modulation; over longer spans it also captures annual speed
        modulation.
        """
        kinematics = self.findKinematicsOverTime(
            station=station,
            meas_times=meas_times,
            sensitive_axis=sensitive_axis,
            include_rotation=include_rotation,
            galcen_frame=galcen_frame,
            verbose=verbose,
        )

        lineshapes = []
        power_coefficients = []
        for speed, alpha in zip(
            kinematics["v_lab_magnitude"], kinematics["wind_angle"]
        ):
            lineshapes.append(
                self.axion_lineshape(
                    v_0=self.v_0,
                    v_lab=speed,
                    nu_a=self.nu_a,
                    nu=frequencies,
                    case=case,
                    alpha=alpha,
                    verbose=False,
                )
            )
            power_coefficients.append(
                self.gradientPowerCoefficient(
                    v_0=self.v_0, v_lab=speed, alpha=alpha, case=case
                )
            )

        PSD = np.array([line.to_value(lineshapes[0].unit) for line in lineshapes])
        PSD = PSD * lineshapes[0].unit
        power_coefficient = (
            np.array(
                [c.to_value(power_coefficients[0].unit) for c in power_coefficients]
            )
            * power_coefficients[0].unit
        )
        power_spectrum = power_coefficient[:, np.newaxis] * PSD
        relative_power = (
            power_coefficient / np.max(power_coefficient)
            if power_coefficient.unit != unit.one
            else power_coefficient
        )

        return {
            **kinematics,
            "frequencies": frequencies,
            "case": case,
            "lineshape": PSD,
            "power_coefficient": power_coefficient,
            "relative_power": relative_power.to(unit.one),
            "power_spectrum_shape": power_spectrum,
        }

    def makeLineshapeFrequencyGrid(
        self,
        *,
        frequency_span: Quantity | None = None,
        frequency_span_ppm: float | None = None,
        num_frequency_points: int = 20001,
    ) -> Quantity:
        """Return a frequency grid suitable for SHM axion PSD evaluation.

        The grid starts at ``nu_a`` and extends over the positive kinetic-energy
        side of the line.  If no span is supplied, it covers ten times the
        nominal SHM width ``nu_a / Q_a``.

        Parameters
        ----------
        frequency_span : Quantity, optional
            Absolute span above ``nu_a``.
        frequency_span_ppm : float, optional
            Fractional span above ``nu_a`` in ppm.  Ignored when
            ``frequency_span`` is supplied.
        num_frequency_points : int
            Number of grid points.

        Returns
        -------
        Quantity
            Frequency grid in the same unit as ``nu_a``.
        """

        if num_frequency_points < 3:
            raise ValueError("num_frequency_points must be at least 3")

        if frequency_span is None:
            if frequency_span_ppm is None:
                frequency_span = 10.0 * self.FWHM * self.nu_a
            else:
                frequency_span = frequency_span_ppm * ppm * self.nu_a
        else:
            frequency_span = frequency_span.to(self.nu_a.unit)

        return (
            self.nu_a
            + np.linspace(
                0.0,
                frequency_span.to_value(self.nu_a.unit),
                num_frequency_points,
            )
            * self.nu_a.unit
        )

    @staticmethod
    def measureLineshapeFWHM(
        frequencies: Quantity,
        spectrum: Quantity | np.ndarray,
        *,
        nu_a: Quantity | None = None,
    ) -> dict:
        """Measure the FWHM of a sampled PSD or power spectrum.

        Linear interpolation is used for the two half-maximum crossings.  The
        method is agnostic to whether ``spectrum`` is a normalized PSD or a
        power spectrum; multiplying a spectrum by a constant does not change
        the returned width.

        Parameters
        ----------
        frequencies : Quantity
            Frequency grid.
        spectrum : Quantity or ndarray
            Sampled PSD or power spectrum on ``frequencies``.
        nu_a : Quantity, optional
            Reference frequency used to report the fractional width in ppm.
            Defaults to the first frequency grid point.

        Returns
        -------
        dict
            Contains ``FWHM_frequency``, ``FWHM_a``, ``tau_a``, peak and
            half-maximum diagnostics.
        """

        freq = frequencies
        values = (
            spectrum.to_value(spectrum.unit)
            if isinstance(spectrum, Quantity)
            else np.asarray(spectrum)
        )
        values = np.asarray(values, dtype=float)

        if freq.ndim != 1 or values.ndim != 1:
            raise ValueError("frequencies and spectrum must be one-dimensional")
        if len(freq) != len(values):
            raise ValueError("frequencies and spectrum must have the same length")
        if len(freq) < 3:
            raise ValueError("at least three frequency points are required")
        if not np.all(np.isfinite(values)):
            raise ValueError("spectrum contains non-finite values")

        peak_idx = int(np.argmax(values))
        peak_value = values[peak_idx]
        if peak_value <= 0:
            raise ValueError("spectrum maximum must be positive")
        half_max = 0.5 * peak_value
        x = freq.to_value(unit.Hz)

        def _crossing(i_low, i_high):
            x0, x1 = x[i_low], x[i_high]
            y0, y1 = values[i_low], values[i_high]
            if y1 == y0:
                return 0.5 * (x0 + x1)
            return x0 + (half_max - y0) * (x1 - x0) / (y1 - y0)

        lower_idx = None
        for idx in range(peak_idx - 1, -1, -1):
            if values[idx] <= half_max <= values[idx + 1]:
                lower_idx = idx
                break
        if lower_idx is None:
            lower_frequency = x[0]
        else:
            lower_frequency = _crossing(lower_idx, lower_idx + 1)

        upper_idx = None
        for idx in range(peak_idx, len(values) - 1):
            if values[idx] >= half_max >= values[idx + 1]:
                upper_idx = idx
                break
        if upper_idx is None:
            raise ValueError(
                "upper half-maximum crossing was not found; use a wider "
                "frequency grid"
            )
        upper_frequency = _crossing(upper_idx, upper_idx + 1)

        FWHM_frequency = (upper_frequency - lower_frequency) * unit.Hz
        if FWHM_frequency <= 0 * unit.Hz:
            raise ValueError("measured FWHM is not positive")

        if nu_a is None:
            nu_a = Quantity(freq[0])
        else:
            nu_a = nu_a.to(unit.Hz)
        FWHM_fraction = (FWHM_frequency / nu_a).to(unit.one)
        FWHM_a = FWHM_fraction.to(ppm)
        tau_a = (1.0 / (np.pi * FWHM_fraction * nu_a)).to(unit.s)

        return {
            "FWHM_frequency": FWHM_frequency.to(frequencies.unit),
            "FWHM_fraction": FWHM_fraction,
            "FWHM_a": FWHM_a,
            "tau_a": tau_a,
            "lower_half_max_frequency": (lower_frequency * unit.Hz).to(
                frequencies.unit
            ),
            "upper_half_max_frequency": (upper_frequency * unit.Hz).to(
                frequencies.unit
            ),
            "peak_frequency": freq[peak_idx].to(frequencies.unit),
            "peak_value": peak_value
            * (spectrum.unit if isinstance(spectrum, Quantity) else unit.one),
            "half_max": half_max
            * (spectrum.unit if isinstance(spectrum, Quantity) else unit.one),
        }

    def findLineshapeAtStationAndTime(
        self,
        station,
        meas_time: Time | None = None,
        frequencies: Quantity | None = None,
        case: str = "grad_perp",
        sensitive_axis: str | np.ndarray | Quantity = "up",
        include_rotation: bool = True,
        galcen_frame: coord.Galactocentric | None = None,
        update: bool = True,
        verbose: bool = False,
        **frequency_grid_kwargs,
    ) -> dict:
        """Evaluate the axion PSD seen by a station at one time.

        The method obtains ``v_lab`` and the wind angle ``alpha`` from
        :meth:`findKinematicsOverTime`, then passes them to the static
        :meth:`axion_lineshape` implementation.

        Parameters
        ----------
        station : axionbloch.Station.Station
            Laboratory location.
        meas_time : astropy.time.Time, optional
            Observation time.  Defaults to ``Time.now()``.
        frequencies : Quantity, optional
            Frequency grid.  If omitted, :meth:`makeLineshapeFrequencyGrid` is
            used.  Keyword arguments such as ``frequency_span_ppm`` and
            ``num_frequency_points`` are forwarded to that grid maker.
        case : str
            ``'non-grad'``, ``'grad_par'``, or ``'grad_perp'``.
        sensitive_axis : str or vector
            Local axis used to define the wind angle.
        include_rotation : bool
            Include the station's surface rotation in the lab velocity.
        galcen_frame : astropy.coordinates.Galactocentric, optional
            Custom Galactocentric frame.
        update : bool
            If ``True``, update ``v_lab``, ``windAngle`` and ``nu_a_eff`` on
            this object.

        Returns
        -------
        dict
            Kinematics plus ``frequencies``, normalized ``PSD``,
            ``power_coefficient`` and ``power_spectrum``.
        """

        if meas_time is None:
            meas_time = Time.now()
        if frequencies is None:
            frequencies = self.makeLineshapeFrequencyGrid(**frequency_grid_kwargs)

        kinematics = self.findKinematicsOverTime(
            station=station,
            meas_times=meas_time,
            sensitive_axis=sensitive_axis,
            include_rotation=include_rotation,
            galcen_frame=galcen_frame,
            verbose=verbose,
        )

        speed = kinematics["v_lab_magnitude"][0]
        alpha = kinematics["wind_angle"][0]
        lineshape = self.axion_lineshape(
            v_0=self.v_0,
            v_lab=speed,
            nu_a=self.nu_a,
            nu=frequencies,
            case=case,
            alpha=alpha,
            verbose=verbose,
        )
        power_coefficient = self.gradientPowerCoefficient(
            v_0=self.v_0,
            v_lab=speed,
            alpha=alpha,
            case=case,
        )
        power_spectrum = power_coefficient * lineshape

        if update:
            self.v_lab = speed
            self.windAngle = alpha
            self.nu_a_eff = kinematics["nu_a_eff"][0]

        return {
            **kinematics,
            "time": kinematics["times"][0],
            "frequency": frequencies,
            "frequencies": frequencies,
            "case": case,
            "alpha": alpha,
            "lineshape": lineshape,
            "PSD": lineshape,
            "power_coefficient": power_coefficient,
            "power_spectrum": power_spectrum,
        }

    def findLineshapeFWHMAtStation(
        self,
        station,
        meas_time: Time | None = None,
        frequencies: Quantity | None = None,
        case: str = "grad_perp",
        sensitive_axis: str | np.ndarray | Quantity = "up",
        spectrum: str = "PSD",
        include_rotation: bool = True,
        galcen_frame: coord.Galactocentric | None = None,
        update: bool = True,
        verbose: bool = False,
        **frequency_grid_kwargs,
    ) -> dict:
        """Find the FWHM of the station/time-dependent axion PSD.

        Parameters are the same as :meth:`findLineshapeAtStationAndTime`.  ``spectrum``
        chooses whether the width is measured from the normalized ``'PSD'`` /
        ``'lineshape'`` or from the scaled ``'power_spectrum'``.  The two are
        normally identical because the power coefficient is frequency
        independent at fixed time.

        When ``update=True``, this method updates:

        - ``FWHM_frequency``: width in frequency units,
        - ``FWHM_a``: fractional width expressed in ppm,
        - ``tau_a = 1 / (pi * FWHM_a * nu_a)`` using ``FWHM_a`` as a
          dimensionless fraction.
        """

        result = self.findLineshapeAtStationAndTime(
            station=station,
            meas_time=meas_time,
            frequencies=frequencies,
            case=case,
            sensitive_axis=sensitive_axis,
            include_rotation=include_rotation,
            galcen_frame=galcen_frame,
            update=update,
            verbose=verbose,
            **frequency_grid_kwargs,
        )

        spectrum_key = spectrum.lower()
        if spectrum_key in {"psd", "lineshape"}:
            sampled_spectrum = result["lineshape"]
            measured_spectrum = "PSD"
        elif spectrum_key in {"power", "power_spectrum", "powerspectrum"}:
            sampled_spectrum = result["power_spectrum"]
            measured_spectrum = "power_spectrum"
        else:
            raise ValueError("spectrum must be 'PSD'/'lineshape' or 'power_spectrum'")

        fwhm = self.measureLineshapeFWHM(
            result["frequencies"],
            sampled_spectrum,
            nu_a=self.nu_a,
        )

        if update:
            self.FWHM_frequency = fwhm["FWHM_frequency"]
            self.FWHM_a = fwhm["FWHM_a"]
            self.tau_a = fwhm["tau_a"]

        return {
            **result,
            **fwhm,
            "measured_spectrum": measured_spectrum,
        }

    def plotPeriodicModulation(
        self,
        station,
        meas_times: list[Time] | Time,
        frequencies: Quantity | None = None,
        case: str = "grad_perp",
        sensitive_axis: str | np.ndarray | Quantity = "up",
        include_rotation: bool = True,
        frequency_indices: list[int] | None = None,
        showPlot: bool = True,
        verbose: bool = False,
    ) -> tuple:
        """Plot Milky Way axion wind periodic modulation.

        Returns the matplotlib figure and the same result dictionary produced
        by :meth:`findLineshapeOverTime` when ``frequencies`` is supplied, or
        :meth:`findKinematicsOverTime` otherwise.
        """

        if frequencies is None:
            result = self.findKinematicsOverTime(
                station=station,
                meas_times=meas_times,
                sensitive_axis=sensitive_axis,
                include_rotation=include_rotation,
                verbose=verbose,
            )
            has_lineshape = False
        else:
            result = self.findLineshapeOverTime(
                frequencies=frequencies,
                station=station,
                meas_times=meas_times,
                case=case,
                sensitive_axis=sensitive_axis,
                include_rotation=include_rotation,
                verbose=verbose,
            )
            has_lineshape = True

        times = result["times"]
        t0 = times[0]
        t_hours = (times - t0).to_value(unit.hour)
        nrows = 4 if has_lineshape else 2

        fig = plt.figure(figsize=(12 / 2.54, (3.0 * nrows) / 2.54), dpi=300)
        grid = gridspec.GridSpec(nrows=nrows, ncols=1)
        axes = [fig.add_subplot(grid[i, 0]) for i in range(nrows)]

        axes[0].plot(t_hours, result["v_lab_magnitude"].to_value(unit.km / unit.s))
        axes[0].set_ylabel("$|v_\\mathrm{lab}|$ (km/s)")

        axes[1].plot(t_hours, result["wind_angle"].to_value(unit.deg))
        axes[1].set_ylabel("$\\alpha$ (deg)")

        if has_lineshape:
            axes[2].plot(t_hours, result["relative_power"].to_value(unit.one))
            axes[2].set_ylabel("relative power")

            PSD = result["lineshape"]
            power_spectrum = result["power_spectrum_shape"]
            freqs = result["frequencies"]
            if frequency_indices is None:
                frequency_indices = [
                    int(
                        np.argmax(
                            np.mean(
                                power_spectrum.to_value(power_spectrum.unit), axis=0
                            )
                        )
                    )
                ]
            for idx in frequency_indices:
                axes[3].plot(
                    t_hours,
                    power_spectrum[:, idx].to_value(power_spectrum.unit),
                    label=f"{freqs[idx]:.6g}",
                )
            axes[3].set_ylabel(f"PSD shape ({power_spectrum.unit})")
            axes[3].legend(loc="best", fontsize=7)

        for ax in axes:
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel(f"Hours since {t0.iso}")

        title = f"{self.name} modulation"
        if station is not None:
            title += f" at {station.name}"
        fig.suptitle(title)
        plt.tight_layout()

        if showPlot:
            plt.show()
        return fig, result

    def getRabiFreq(
        self,
        g_aNN: Quantity[unit.GeV ** (-1)] | None = None,
        case="grad_perp",
        alpha: Quantity | None = None,
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
        if alpha is None:
            alpha = self.windAngle if self.windAngle is not None else 0.0 * unit.rad

        if case in {"grad_par", "grad_perp"}:
            velocity_rms = np.sqrt(
                self.gradientPowerCoefficient(
                    v_0=self.v_0, v_lab=self.v_lab, alpha=alpha, case=case
                )
            )
            Omega_rms = (
                0.5
                * g_aNN
                * (2 * const.hbar * const.c * self.rho_E_DM) ** (1 / 2)
                * velocity_rms
            ).to(unit.Hz)
        else:
            raise ValueError(
                f"case {case} not recognized, should be 'grad_par' or 'grad_perp'"
            )
        if verbose:
            print(
                logPrefix,
                f"axion wind Rabi frequency (case={case}): {Omega_rms.to(unit.Hz)}",
            )
        return Omega_rms

    @staticmethod
    def axion_lineshape(
        v_0: Quantity[unit.m / unit.s],
        v_lab: Quantity[unit.m / unit.s],
        nu_a: Quantity[unit.Hz],
        nu: Quantity,
        case: str = "non-grad",
        alpha: Quantity = 0.0 * unit.rad,
        verbose: bool = False,
    ):
        """Calculate the analytical axion PSD lineshape.

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
            Power spectral density lineshape, normalized so that
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
                print(
                    logPrefix,
                    f"nu_a = {nu_a}, first frequency element > nu_a is nu[{nu_a_indx}] = {nu[nu_a_indx]}",
                )
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
                print(
                    logPrefix,
                    f"cutoff frequency is {(1 + 10 * FWHM) * nu_a}, first frequency element > cutoff frequency is nu[{cutoff_indices[0]}] = {nu[cutoff_indices[0]]}",
                )
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

            References
            ----------
            A. Gramolin: https://github.com/gramolin/lineshape

            """
            assert case in [
                "non-grad",
                "grad_par",
                "grad_perp",
            ], "Case should be 'non-grad', 'grad_par', or 'grad_perp'!"

            beta = (
                2 * const.c * v_lab * np.sqrt(2 * (freq - nu_a) / nu_a) / v_0**2
            )  # Eq. (13) in Gramolin lineshape paper
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
                    - (1 / np.tanh(beta) - 1.0 / beta)
                    * (2 - 3 * np.sin(alpha) ** 2)
                    / beta
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
            fine_lineshape = np.zeros_like(fine_freqs) * fine_freqs[0].unit ** -2
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
        alpha: Quantity | None = None,
        numSpectra: int = 1,
        rand_seed: int = None,
        use_stoch: bool = True,
        verbose: bool = False,
    ) -> np.ndarray:
        """Return complex amplitude spectra for the axion pseudo-magnetic field.

        Draws ``numSpectra`` realizations of the stochastic axion field in the
        frequency domain using the SHM lineshape as the power spectral density.
        Each realization has random phases uniformly distributed in ``[0, 2π)``.
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
            Number of independent field realizations to generate.
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

        logPrefix = f"[{self.__class__.__name__}.{self.getAmpSpectra.__name__}]"
        if alpha is None:
            alpha = self.windAngle if self.windAngle is not None else 0.0 * unit.rad

        PSD_lineshape = MilkyWayAxionHalo.axion_lineshape(
            v_0=self.v_0,
            v_lab=self.v_lab,
            nu_a=self.nu_a,
            nu=frequencies,
            case=case,
            alpha=alpha,
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
            ampSpectra = (
                np.sqrt(PSD_lineshape) * phases
            )  # shape = (numFields, numSteps)

        return ampSpectra
