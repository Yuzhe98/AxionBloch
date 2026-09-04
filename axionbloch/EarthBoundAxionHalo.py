from scipy.interpolate import interp1d
from scipy.signal import correlate as correlate

from axionbloch.dependency import *
from axionbloch.GravBoundAxionHalo import GravBoundAxionHalo
from axionbloch.Station import Station
from axionbloch.utils import check


def PREM_density(radius_km):
    """Return the analytic PREM density in g/cm³ at radius ``radius_km``.

    Both scalar values and NumPy-compatible arrays are accepted.
    """
    msgPrefix = f"[{PREM_density.__name__}]"
    # Use the coefficients of the polynomials describing the Preliminary Reference Earth Model (PREM) to find out density.
    radius_km = np.abs(np.asarray(radius_km, dtype=float))

    # Earth's radius in km (converted to meters in DataFrame)
    EARTH_RADIUS_KM = 6371

    # PREM density functions for each region (x = r/6371)
    def density_inner_core(x):
        return 13.0885 - 8.8381 * x**2

    def density_outer_core(x):
        return 12.5815 - 1.2638 * x - 3.6426 * x**2 - 5.5281 * x**3

    def density_lower_mantle(x):
        return 7.9565 - 6.4761 * x + 5.5283 * x**2 - 3.0807 * x**3

    def density_transition_zone(x):
        return np.where(
            x <= 5771 / EARTH_RADIUS_KM,
            5.3197 - 1.4836 * x,
            np.where(
                x <= 5971 / EARTH_RADIUS_KM,
                11.2494 - 8.0298 * x,
                7.1089 - 3.8045 * x,
            ),
        )

    def density_lvz_lid(x):
        return 2.6910 + 0.6924 * x

    def density_crust(x):
        return np.where(x <= 6356 / EARTH_RADIUS_KM, 2.900, 2.600)

    def density_ocean(x):
        return 1.020

    def density_space(x):
        return 0.0

    # # Generate radius values (in km) at 10 km intervals
    # radius_km = np.arange(0, EARTH_RADIUS_KM., 10)

    # Calculate density for each radius
    # density = []
    x = radius_km / EARTH_RADIUS_KM
    density = np.select(
        [
            radius_km <= 1221.5,
            radius_km <= 3480.0,
            radius_km <= 5701.0,
            radius_km <= 6151.0,
            radius_km <= 6346.6,
            radius_km <= 6368.0,
            radius_km <= 6371.0,
        ],
        [
            density_inner_core(x),
            density_outer_core(x),
            density_lower_mantle(x),
            density_transition_zone(x),
            density_lvz_lid(x),
            density_crust(x),
            density_ocean(x),
        ],
        default=density_space(x),
    )
    return density.item() if density.ndim == 0 else density


def PREM_density_profile(num_samples=6372):
    """Sample the analytic PREM model from Earth's center to its surface."""
    radius_km = np.linspace(0.0, 6371.0, num_samples)
    radius_m = radius_km * 1000.0
    density_kg_m3 = PREM_density(radius_km) * 1000.0
    return radius_m, density_kg_m3


def getCumulativeMass():
    """
    Returns the cumulative mass as a function of radius.
    Uses PREM-like model for interior, point-mass approximation for exterior.
    """
    msgPrefix = f"[{getCumulativeMass.__name__}]"
    radius_m, density_kg_m3 = PREM_density_profile()
    r = radius_m * unit.meter
    rho = density_kg_m3 * (unit.kg / unit.meter**3)

    # Compute shell thickness
    dr = np.gradient(r)

    # Shell volume and mass
    dV = 4 * np.pi * r**2 * dr
    dm = rho * dV

    # Cumulative mass
    M_r = np.cumsum(dm)
    return r, M_r


def _earth_grav_potential_profile():
    """Return the PREM potential profile with ``Phi(infinity) = 0``.

    For a spherical Earth, ``dPhi/dr = G M(<r) / r**2``.  Integrating this
    relation inward from the surface boundary condition includes the constant
    potential contribution from all shells exterior to each radius.
    """
    r, M_r = getCumulativeMass()
    phi_surface = -const.G * M_r[-1] / r[-1]
    dphi_dr = np.zeros(r.shape) * (unit.joule / unit.kg / unit.meter)
    dphi_dr[1:] = const.G * M_r[1:] / r[1:] ** 2

    shell_integrals = 0.5 * (dphi_dr[:-1] + dphi_dr[1:]) * np.diff(r)
    phi_inside = np.zeros(r.shape) * (unit.joule / unit.kg)
    phi_inside[-1] = phi_surface
    phi_inside[:-1] = phi_surface - np.cumsum(shell_integrals[::-1])[::-1]

    return r, phi_inside


def earth_grav_potential_infty():
    """
    Returns a function Phi(r[m]) [J/kg], valid both inside and outside Earth.
    Uses PREM-like model for interior, point-mass approximation for exterior.
    """
    msgPrefix = f"[{earth_grav_potential_infty.__name__}]"
    r, Phi_inside = _earth_grav_potential_profile()
    M_total = getCumulativeMass()[1][-1]

    # Extend to radii beyond Earth's surface
    r_max = r[-1]
    # Resolve the near-Earth profile while still extending to 1000 R_earth.
    r_outside = np.geomspace(1, 1000, 800) * r_max
    Phi_outside = -const.G * M_total / r_outside

    # Combine inside and outside
    r_full = np.concatenate([r, r_outside])
    Phi_full = np.concatenate(
        [
            Phi_inside,
            Phi_outside,
        ]
    )
    # Phi_full -= np.amin(Phi_full)

    # Enforce symmetry: add negative r values
    r_sym = np.concatenate([-r_full[::-1], r_full])  # mirror and append
    Phi_sym = np.concatenate([Phi_full[::-1], Phi_full])  # symmetric values

    # Optional: sort to ensure increasing r (for interp1d)
    sorted_indices = np.argsort(r_sym)
    r_sym_sorted = r_sym[sorted_indices]
    Phi_sym_sorted = Phi_sym[sorted_indices]

    # Interpolation function: now Phi_func(-r) = Phi_func(r)
    r_unit = unit.R_earth
    Phi_unit = unit.megajoule / unit.kilogram

    # Interpolation function: now Phi_func(-r) = Phi_func(r)
    Phi_func = interp1d(
        r_sym_sorted.to_value(r_unit),
        Phi_sym_sorted.to_value(Phi_unit),
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )

    return Phi_func, r_unit, Phi_unit


def earth_grav_potential_earth_center():
    """
    Returns a function Phi(r), valid both inside and outside Earth.
    Uses PREM-like model for interior, point-mass approximation for exterior.
    """
    msgPrefix = f"[{earth_grav_potential_earth_center.__name__}]"
    r, Phi_inside = _earth_grav_potential_profile()
    M_total = getCumulativeMass()[1][-1]

    # Extend to radii beyond Earth's surface
    r_max = r[-1]
    r_outside = np.linspace(
        r_max, 1000 * r_max, 100000
    )  # from surface to 1000 Earth radii
    Phi_outside: Quantity = -const.G * M_total / r_outside

    # Combine inside and outside
    r_full = np.concatenate([r, r_outside])
    Phi_full = np.concatenate([Phi_inside, Phi_outside])
    # Phi_full -= np.amin(Phi_full)

    # Enforce symmetry: add negative r values
    r_sym = np.concatenate([-r_full[::-1], r_full])  # mirror and append
    Phi_sym = np.concatenate([Phi_full[::-1], Phi_full])  # symmetric values

    # Optional: sort to ensure increasing r (for interp1d)
    sorted_indices = np.argsort(r_sym)
    r_sym_sorted = r_sym[sorted_indices]
    Phi_sym_sorted = Phi_sym[sorted_indices]
    # Keep the historical convention of this helper: Phi(center) = 0.
    Phi_sym_sorted -= np.amin(Phi_sym_sorted)

    # Interpolation function: now Phi_func(-r) = Phi_func(r)
    r_unit = unit.R_earth
    Phi_unit = unit.megajoule / unit.kilogram

    Phi_func = interp1d(
        r_sym_sorted.to_value(r_unit),
        Phi_sym_sorted.to_value(Phi_unit),
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    return Phi_func, r_unit, Phi_unit


def plot_earth_grav_potential(showplot=True):
    msgPrefix = f"[{plot_earth_grav_potential.__name__}]"
    radius_m, density_kg_m3 = PREM_density_profile()
    density_r = radius_m * unit.meter
    density_rho = density_kg_m3 * (unit.kg / unit.meter**3)
    density_unit = unit.g / unit.cm**3

    # cumulative mass
    mass_r, mass_M_r = getCumulativeMass()

    # Compare both potential conventions in the bottom panel.
    Phi_func, r_unit, Phi_unit = earth_grav_potential_infty()
    Phi_center_func, center_r_unit, center_Phi_unit = (
        earth_grav_potential_earth_center()
    )
    # extend to radii beyond Earth's surface
    r_extended = np.linspace(0, 3, 1000) * unit.R_earth
    Phi_extended = Phi_func(r_extended.to_value(r_unit)) * (Phi_unit)
    Phi_center_extended = (
        Phi_center_func(r_extended.to_value(center_r_unit)) * center_Phi_unit
    )

    # use units for plotting:
    r_unit = unit.R_earth
    Phi_unit = unit.megajoule / unit.kilogram

    # ------------- Plot ---------------------

    # plot style
    plt.rc("font", size=6)  # font size for all figures
    # plt.rcParams['font.family'] = 'serif'
    plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams['mathtext.fontset'] = 'dejavuserif'

    # Make math text match Times New Roman
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["mathtext.rm"] = "Times New Roman"

    cm = 1 / 2.54  # convert cm to inch

    fig = plt.figure(figsize=(8.5 * cm, 8.5 * cm), dpi=300)  # initialize a figure

    gs = gridspec.GridSpec(nrows=3, ncols=1)

    # fix the margins
    left = 0.15
    bottom = 0.11
    right = 0.865
    top = 0.88
    wspace = 0.2
    hspace = 0.1
    fig.subplots_adjust(
        left=left, top=top, right=right, bottom=bottom, wspace=wspace, hspace=hspace
    )

    density_ax = fig.add_subplot(gs[0, 0])
    mass_ax = fig.add_subplot(gs[1, 0])
    pot_ax = fig.add_subplot(gs[2, 0])

    # density profile
    density_ax.plot(
        density_r.to_value(r_unit),
        density_rho.to_value(density_unit),
        label="Density Profile",
        color="darkblue",
    )
    density_ax.set_ylabel("Density $(\\mathrm{g}\\,\\mathrm{cm}^{-3})$")

    # cumulative mass
    mass_ax.plot(
        mass_r.to_value(r_unit),
        mass_M_r.to_value(unit.kg) / 1e24,
        label="Mass Profile",
        color="darkgreen",
    )
    mass_ax.set_ylabel("Enclosed mass $(10^{24}\\,\\mathrm{kg})$")
    mass_ax.ticklabel_format(useOffset=False)

    # gravitational potential
    (infinity_line,) = pot_ax.plot(
        r_extended.to_value(r_unit),
        Phi_extended.to_value(Phi_unit),
        label="$\\Phi(\\infty)=0$",
        color="darkorange",
        linestyle="-",
    )
    center_ax = pot_ax.twinx()
    (center_line,) = center_ax.plot(
        r_extended.to_value(r_unit),
        Phi_center_extended.to_value(Phi_unit),
        label="$\\Phi(0)=0$",
        color="tab:purple",
        linestyle="--",
    )
    # pot_ax.axvline(
    #     x=(1 * unit.R_earth).to_value(r_unit),
    #     color="k",
    #     linestyle="dotted",
    #     linewidth=1,
    #     alpha=0.8,
    #     # label="Earth radius",
    # )

    pot_ax.set_xlabel("Radius (earth radius)")
    # pot_ax.set_ylabel("Grav. Pot. (MJ/kg) ref. to $\\infty$")
    pot_ax.set_ylabel(
        "$\\Phi_\\oplus$ ($\\mathrm{MJ}\\,\\mathrm{kg}^{-1}$), $\\Phi(\\infty)=0$"
    )
    center_ax.set_ylabel(
        "$\\Phi_\\oplus$ ($\\mathrm{MJ}\\,\\mathrm{kg}^{-1}$), $\\Phi(0)=0$"
    )
    pot_ax.legend(
        [infinity_line, center_line],
        [infinity_line.get_label(), center_line.get_label()],
        loc="lower right",
        frameon=False,
    )

    density_ax.set_ylim(-0.5, 15.5)
    pot_ax.set_ylim(-130, 5)
    phi_infinity_center = Phi_extended[0].to_value(Phi_unit)
    center_ax.set_ylim(-130 - phi_infinity_center, -phi_infinity_center + 5)
    center_ax.set_yticks([0, 25, 50, 75, 100])

    pot_ax.set_yticks([-125, -100, -75, -50, -25, 0])

    xlimits = (0, 3)
    pot_ax.set_xlim(xlimits)
    density_ax.set_xlim(xlimits)
    mass_ax.set_xlim(xlimits)
    density_ax.set_xticklabels([])
    mass_ax.set_xticklabels([])

    # Show radius in km at the top of the density panel.
    density_km_ax = density_ax.twiny()
    density_km_ax.set_xlim(
        xlimits[0] * (1 * unit.R_earth).to_value(unit.km),
        xlimits[1] * (1 * unit.R_earth).to_value(unit.km),
    )
    density_km_ax.set_xlabel("Radius (km)")
    density_km_ax.tick_params(direction="in", pad=2)

    fig.suptitle("Earth Profiles (from PREM Data)")
    fig.align_ylabels([density_ax, mass_ax, pot_ax])
    # fig.tight_layout()
    plt.savefig("Earth-profiles-(PREM-data).png", transparent=False)
    if showplot:
        plt.show()
    else:
        plt.close(fig)


# Backward-compatible alias for callers using the original name.
plot_earth_grav_potential = plot_earth_grav_potential


class EarthBoundAxionHalo(GravBoundAxionHalo):
    # Create the "axion stream" (axion field) object
    # you can get properties of the axion field, computed based on the input information
    nu_a: Quantity[unit.Hz] | None = None  # axion Compton frequency
    m_a: Quantity[unit.g] | None = None  # axion mass
    N: int = int(2**12)
    extent: Quantity[unit.m] = 128.0 * unit.R_earth
    a_0: Quantity[unit.eV] | None = None
    totalMassEnclosed: Quantity[unit.kg] | None = 4e-9 * unit.M_earth
    g_aNN: Quantity[unit.GeV**-1] = 1e-9 * unit.GeV**-1

    def __init__(
        self,
        name="Earth-Bound Axion Halo",
        nu_a: Quantity[unit.Hz] | None = None,
        m_a: Quantity[unit.g] | None = None,
        N: int = int(2**12),
        extent: Quantity[unit.m] = 128.0 * unit.R_earth,
        getPot=earth_grav_potential_earth_center,
        a_0: Quantity[unit.eV] | None = None,
        totalMassEnclosed: Quantity[unit.kg] | None = 4e-9 * unit.M_earth,
        g_aNN: Quantity[unit.GeV**-1] = 1e-9 * unit.GeV**-1,
        verbose: bool = False,
    ):
        msgPrefix = f"[{self.__class__.__name__}.{self.__init__.__name__}]"
        super().__init__(
            name=name,
            nu_a=nu_a,
            m_a=m_a,
            N=N,
            extent=extent,
            getPot=getPot,
            a_0=a_0,
            g_aNN=g_aNN,
            verbose=verbose,
        )
        # convert potential and kinetic energy magnitude to atto-eV for easier computation
        self.E_unit = unit.attoelectronvolt
        self.pot = self.pot.to(self.E_unit)
        self.T_magnitude = self.T_magnitude.to(self.E_unit)
        self.N_a = totalMassEnclosed / self.m_a
        if a_0 is not None:
            self.a_0 = a_0
        else:
            self.a_0_reduced = np.sqrt(
                2 * self.N_a * const.hbar**3 * const.c / self.m_a
            ).si
        self.totalMassEnclosed = totalMassEnclosed

    # ------------------------------------------------------------------
    # Gradient at arbitrary direction / time
    # ------------------------------------------------------------------

    def findGradientsAtEarthSurface(
        self,
        station: Station,
        stateCoefficients: dict[str, complex],
        meas_time: Time | None = None,
        truncRadius: Quantity | None = 3 * unit.earthRad,
        include_lorentz_boost: bool = True,
        relative_velocity: Quantity | None = None,
        showPlot: bool = False,
        verbose: bool = False,
    ) -> tuple:
        """Compute the wavefunction gradient toward a station at the Earth's surface.

        A convenience wrapper around :meth:`findGradientsAtDirection` with a
        default truncation radius suited to ground-based experiments.

        Parameters
        ----------
        station : Station
            Geographic station whose latitude, longitude, and elevation define
            the direction.
        stateCoefficients : dict of str to complex
            Mapping from eigenstate names to coefficients.
            Coefficients are normalized automatically.
        meas_time : Time, optional
            Measurement epoch. Uses :meth:`astropy.time.Time.now` if omitted.
        truncRadius : Quantity, optional
            Radial cutoff for the interpolation grid.
        showPlot : bool
            Plot the wavefunction and gradient profiles.
        verbose : bool
            Print timing and diagnostic information.

        Returns
        -------
        r, R_r, r_line, grad_r, grad_theta, grad_phi
            Same six arrays as :meth:`findGradients`.
        """
        if meas_time is None:
            meas_time = Time.now()
        return self.findGradientsAtDirection(
            stateCoefficients=stateCoefficients,
            station=station,
            meas_time=meas_time,
            truncRadius=truncRadius,
            include_lorentz_boost=include_lorentz_boost,
            relative_velocity=relative_velocity,
            showPlot=showPlot,
            verbose=verbose,
        )

    def findGradients(
        self,
        stateCoefficients: dict[str, complex],
        station: Station | None = None,
        meas_time: Time | None = None,
        truncRadius: Quantity | None = 3 * unit.earthRad,
        include_lorentz_boost: bool = True,
        relative_velocity: Quantity | None = None,
        showPlot: bool = False,
        verbose: bool = False,
    ) -> tuple:
        """Compute the wavefunction gradient toward a geographic direction.

        The direction is specified by a :class:`~axionbloch.Station.Station`;
        raises :exc:`ValueError` when none is provided.

        Parameters
        ----------
        stateCoefficients : dict of str to complex
            Mapping from eigenstate names to coefficients.
            Coefficients are normalized automatically.
        station : Station, optional
        meas_time : Time, optional
            Measurement epoch. Uses :meth:`astropy.time.Time.now` if omitted.
        truncRadius : Quantity, optional
        showPlot : bool
        verbose : bool

        Returns
        -------
        r, R_r, r_line, grad_r, grad_theta, grad_phi
        """
        if station is None:
            raise ValueError("[EarthBoundAxionHalo.findGradients] Provide station=.")
        if meas_time is None:
            meas_time = Time.now()
        return self.findGradientsAtDirection(
            stateCoefficients=stateCoefficients,
            station=station,
            meas_time=meas_time,
            truncRadius=truncRadius,
            include_lorentz_boost=include_lorentz_boost,
            relative_velocity=relative_velocity,
            showPlot=showPlot,
            verbose=verbose,
        )

    def findGradientsWithMilkyWay(
        self,
        mw,
        stateCoefficients: dict[str, complex],
        truncRadius: Quantity | None = None,
        showPlot: bool = False,
        verbose: bool = False,
    ) -> dict:
        """Compute the gradient at a station with time-dependent galactic context.

        Uses the station embedded in the :class:`~axionbloch.MilkyWay.MilkyWay`
        instance for the geographic direction, then enriches the result with
        galactic kinematics derived from that same object:

        - Lab velocity :math:`\\mathbf{v}_\\mathrm{lab}` (magnitude and direction).
        - Wind angle between :math:`\\mathbf{v}_\\mathrm{lab}` and the
          station's projection axis.
        - Gradient in Cartesian ITRS coordinates (useful for projecting onto
          a non-vertical :math:`\\mathbf{B}_0`).
        - Projection of the gradient onto the projection axis (local
          zenith/radial direction).

        Parameters
        ----------
        mw : :class:`~axionbloch.MilkyWay.MilkyWay`
            Must have :attr:`~axionbloch.MilkyWay.MilkyWay.station` set.
        stateCoefficients : dict of str to complex
            Mapping from eigenstate names to coefficients.
            Coefficients are normalized automatically.
        truncRadius : Quantity, optional
            Radial cutoff.
        showPlot : bool
            Plot the wavefunction and gradient profiles.
        verbose : bool
            Print diagnostics.

        Returns
        -------
        dict with keys
            ``r``, ``R_r``, ``r_line`` — radial grid and wavefunction.

            ``grad_r``, ``grad_theta``, ``grad_phi`` — spherical gradient
            components along the full r_line.

            ``grad_r_surface``, ``grad_theta_surface``, ``grad_phi_surface`` —
            values at Earth's surface (:math:`r = R_\\oplus`).

            ``grad_cartesian_itrs`` — Cartesian gradient vector in ITRS
            (x = prime meridian, z = north pole) at Earth's surface.

            ``nvec_gcrs`` — station's unit normal in GCRS (equatorial inertial),
            changes with time as Earth rotates.

            ``v_lab``, ``v_lab_magnitude`` — lab velocity in galactic frame.

            ``wind_angle`` — angle between v_lab and the projection axis [rad].

        Examples
        --------
        >>> from astropy.time import Time
        >>> from axionbloch.MilkyWay import MilkyWay
        >>> from axionbloch.Station import Baltimore
        >>> mw = MilkyWay(time=Time('2024-06-21T14:00:00'), station=Baltimore)
        >>> result = halo.findGradientsWithMilkyWay(
        ...     mw,
        ...     stateCoefficients={"2p": 1},
        ...     truncRadius=2 * unit.R_earth,
        ... )
        >>> print(result['wind_angle'].to('deg'))
        >>> print(result['grad_r_surface'])
        """
        if mw.station is None:
            raise ValueError(
                "MilkyWay.station must be set before calling findGradientsWithMilkyWay."
            )

        station = mw.station

        # ---- Gradient in geographic spherical coordinates ----
        r, R_r, r_line, grad_r, grad_theta, grad_phi = self.findGradientsAtDirection(
            stateCoefficients=stateCoefficients,
            station=station,
            meas_time=mw.time,
            truncRadius=truncRadius,
            showPlot=showPlot,
            verbose=verbose,
        )

        # ---- Gradient values at Earth's surface ----
        earth_idx = int(np.argmin(np.abs(r_line - 1.0 * unit.R_earth)))
        gr_s = grad_r[earth_idx]
        gt_s = grad_theta[earth_idx]
        gp_s = grad_phi[earth_idx]

        # ---- Convert to Cartesian ITRS ----
        # Spherical unit vectors at (theta_s, phi_s) in ITRS
        theta_s = float(station.theta.to_value(unit.rad))
        phi_s = float(station.phi.to_value(unit.rad))
        sin_t, cos_t = np.sin(theta_s), np.cos(theta_s)
        sin_p, cos_p = np.sin(phi_s), np.cos(phi_s)

        r_hat = np.array([sin_t * cos_p, sin_t * sin_p, cos_t])
        theta_hat = np.array([cos_t * cos_p, cos_t * sin_p, -sin_t])
        phi_hat = np.array([-sin_p, cos_p, 0.0])

        # grad_theta / grad_phi carry an extra 'rad' denominator because θ,φ
        # are in radians.  Since rad is dimensionless, convert to the same unit
        # as grad_r via dimensionless_angles() before combining.
        base_grad_unit = gr_s.unit
        gt_s_compat = gt_s.to(base_grad_unit, equivalencies=unit.dimensionless_angles())
        gp_s_compat = gp_s.to(base_grad_unit, equivalencies=unit.dimensionless_angles())

        grad_cartesian_itrs = (
            gr_s * r_hat + gt_s_compat * theta_hat + gp_s_compat * phi_hat
        )

        # ---- Galactic context from MilkyWay ----
        nvec_gcrs = mw.get_nvec_gcrs()
        v_lab = mw.get_v_lab()
        v_lab_mag = mw.get_v_lab_magnitude()
        wind_angle = mw.get_wind_angle()

        if verbose:
            print(f"[findGradientsWithMilkyWay] station       = {station.name}")
            print(f"[findGradientsWithMilkyWay] time          = {mw.time.iso}")
            print(f"[findGradientsWithMilkyWay] |v_lab|       = {v_lab_mag:.3f}")
            print(
                f"[findGradientsWithMilkyWay] wind_angle    = {wind_angle.to(unit.deg):.2f}"
            )
            print(f"[findGradientsWithMilkyWay] grad_r surf.  = {gr_s}")
            print(f"[findGradientsWithMilkyWay] grad_th surf. = {gt_s}")
            print(f"[findGradientsWithMilkyWay] grad_ph surf. = {gp_s}")
            print(
                f"[findGradientsWithMilkyWay] grad Cartesian (ITRS) = {grad_cartesian_itrs}"
            )

        return {
            # radial grid and wavefunction
            "r": r,
            "R_r": R_r,
            "r_line": r_line,
            # spherical gradient profiles
            "grad_r": grad_r,
            "grad_theta": grad_theta,
            "grad_phi": grad_phi,
            # gradient at Earth's surface
            "grad_r_surface": gr_s,
            "grad_theta_surface": gt_s,
            "grad_phi_surface": gp_s,
            "grad_cartesian_itrs": grad_cartesian_itrs,
            # galactic context
            "nvec_gcrs": nvec_gcrs,
            "v_lab": v_lab,
            "v_lab_magnitude": v_lab_mag,
            "wind_angle": wind_angle,
            # metadata
            "station": station,
            "time": mw.time,
        }

    def getBfield(
        self,
        rate_Hz: float,
        timeLen: int,
        rand_seed: int,
        numFields: int = 1,
        verbose: bool = False,
    ):
        msgPrefix = f"[{self.__class__.__name__}.{self.getBfield.__name__}]"
        pass

    def coh_time_g1(self):
        """
        x : complex-valued time series
        dt: sampling interval
        method: "1e" or "integral"
        """
        msgPrefix = f"[{self.__class__.__name__}.{self.coh_time_g1.__name__}]"
        x = self.Ba[:, 0] - np.mean(self.Ba[:, 0])
        dt = 1 / self.rate_Hz
        E = np.array(x)  # complex field

        N = len(E)

        # tic = time.time()
        corr = correlate(E, E.conj(), mode="full")
        # toc = time.time()
        # print(f"Time taken for correlation: {toc - tic:.3f} seconds")
        fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
        gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures
        ax00 = fig.add_subplot(gs[0, 0])
        ax00.plot(np.abs(corr), label="")
        ax00.set_xlabel("")
        ax00.set_ylabel("corr (arb. units)")
        ax00.legend()
        fig.suptitle("", wrap=True)
        plt.tight_layout()
        plt.show()

        check(len(corr) // 2)
        corr = corr[len(corr) // 2 :]
        g1 = corr / corr[0]

        fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
        gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures
        ax00 = fig.add_subplot(gs[0, 0])
        ax00.plot(g1.real, label="real part")
        ax00.plot(g1.imag, label="imaginary part")
        ax00.set_xlabel("time (s)")
        ax00.set_ylabel("g1 (arb. units)")
        ax00.legend()
        fig.suptitle("", wrap=True)
        plt.tight_layout()
        plt.show()

        tau = 2 * np.sum(np.abs(g1)) * dt
        return tau

    def plotEigenStates(
        self,
        numStates: int = 8,
        startState: int = 0,
        truncRadius: Quantity | None = 3 * unit.earthRad,
        xlim=(-0.3, 5.3),
        ylim=None,
        showPlot=True,
        savefig=False,
    ):
        """Plot a vertical stack of reduced radial wavefunctions.

        Eigenstates are plotted from lowest energy (bottom) to highest (top),
        sharing a common y-scale so amplitudes can be compared visually.

        Parameters
        ----------
        numStates : int
            How many consecutive states (starting from ``startState``) to show.
        startState : int
            Index into the energy-sorted state list at which to begin.
        xlim : tuple or None
            x-axis limits in units of Earth radii.
        ylim : tuple or None
            Shared y-axis limits.  Auto-computed from peak amplitude if None.
        """
        msgPrefix = f"[{self.__class__.__name__}.{self._plotEigenStates.__name__}]"
        self.sortByEigenE()

        startIdx = self.N // 2 + 1  # avoid r=0 singularity
        if truncRadius is None or type(truncRadius) != Quantity:
            stopIdx = -1
        elif truncRadius.unit.is_equivalent(self.r.unit):
            stopIdx = startIdx + np.argmin(np.abs(self.r[startIdx:] - truncRadius))
        else:
            raise TypeError(
                msgPrefix + " truncRadius unit is not equivalent to length. "
            )

        plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams["font.family"] = "Times New Roman"
        fig = plt.figure(figsize=(8.5 / 2.54, 8.5 * 9 / 16 / 2.54), dpi=300)
        ax = fig.add_subplot(111)
        # fig.subplots_adjust(left=0.22, bottom=0.14, right=0.67, top=0.95)
        ax.axvline(
            x=(1 * unit.earthRad).to_value(self.r.unit),
            color="red",
            linestyle="dotted",
            alpha=0.8,
            label="Earth radius",
        )
        self._plotEigenStates(
            ax=ax,
            startIdx=startIdx,
            stopIdx=stopIdx,
            numStates=numStates,
            startState=startState,
        )
        # ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))

        plt.tight_layout()

        if savefig:
            plt.savefig("Earth eigenstates.pdf")

        if showPlot:
            plt.show()


if __name__ == "__main__":
    plot_earth_grav_potential()
