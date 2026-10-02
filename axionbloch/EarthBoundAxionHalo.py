import warnings

from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.signal import correlate as correlate

from axionbloch.dependency import *
from axionbloch.GravBoundAxionHalo import GravBoundAxionHalo
from axionbloch.Station import Station
from axionbloch.utils import check, linestyles, markers, _getLocationPrefix


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


def earth_grav_potential_infty():
    """
    Returns a function Phi(r[m]) [J/kg], valid both inside and outside Earth.
    Uses PREM-like model for interior, point-mass approximation for exterior.
    """
    msgPrefix = f"[{earth_grav_potential_infty.__name__}]"

    # Build the interior profile by integrating dPhi/dr inward from the
    # surface, where the exterior point-mass boundary condition is known.
    r, M_r = getCumulativeMass()
    phi_surface = -const.G * M_r[-1] / r[-1]
    dphi_dr = np.zeros(r.shape) * (unit.joule / unit.kg / unit.meter)
    dphi_dr[1:] = const.G * M_r[1:] / r[1:] ** 2
    shell_integrals = 0.5 * (dphi_dr[:-1] + dphi_dr[1:]) * np.diff(r)
    Phi_inside = np.zeros(r.shape) * (unit.joule / unit.kg)
    Phi_inside[-1] = phi_surface
    Phi_inside[:-1] = phi_surface - np.cumsum(shell_integrals[::-1])[::-1]
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


def earth_grav_potential_infty2():
    """Return the PREM gravitational potential using the explicit shell integral.

    For radii inside Earth this evaluates

    ``Phi(r) = -G * (M(<r) / r + integral_r^R dM(r') / r')``.

    The potential is normalized so that ``Phi(infinity) = 0`` and is extended
    outside Earth with the point-mass expression ``-G M_earth / r``.
    """
    msgPrefix = f"[{earth_grav_potential_infty2.__name__}]"  # Diagnostic prefix.
    r, M_r = getCumulativeMass()  # Radius grid and enclosed mass profile.
    density_rho = PREM_density(  # Evaluate the PREM density on that grid.
        r.to_value(unit.km)
    ) * (
        unit.g / unit.cm**3
    )  # Attach the density's original PREM units.
    density_rho = density_rho.to(  # Convert density to SI for the integral.
        unit.kg / unit.meter**3
    )

    # A spherical shell has dM = 4*pi*r'^2*rho(r')*dr'.
    # Therefore dM/r' = 4*pi*r'*rho(r')*dr', which is the integrand below.
    shell_integrand = 4.0 * np.pi * density_rho * r  # Units: kg / m^2.
    shell_integral_values = np.zeros(r.size)
    # Reverse the grid so SciPy integrates from the surface inward.
    reversed_integrand = shell_integrand.to_value(unit.kg / unit.meter**2)[::-1]
    reversed_radius = r.to_value(unit.meter)[::-1]  # Radius values for SciPy.
    # The reversed radius decreases, so negate the result to obtain
    # the positive integral from each radius r to the surface R_earth.
    shell_integral_values[:-1] = -cumulative_trapezoid(
        reversed_integrand, reversed_radius
    )[::-1]
    # Restore the physical units removed for the SciPy calculation.
    shell_integral = shell_integral_values * (unit.kg / unit.meter)
    # Compute the interior-mass term M(r)/r, excluding r=0 to avoid 0/0.
    mass_over_radius = np.zeros(r.size) * (unit.kg / unit.meter)
    mass_over_radius[1:] = M_r[1:] / r[1:]  # Units: kg / m.
    # Combine both terms in Phi(r) = -G*[M(r)/r + integral dM(r')/r'].
    phi_inside = -const.G * (mass_over_radius + shell_integral)

    r_max = r[-1]  # Earth's modeled surface radius.
    r_outside = (
        np.geomspace(1.0, 1000.0, 800)  # Log-spaced exterior grid to 1000 Earth radii.
        * r_max
    )
    phi_outside = -const.G * M_r[-1] / r_outside  # Exterior point-mass potential.

    r_full = np.concatenate([r, r_outside])  # Join interior and exterior radii.
    phi_full = np.concatenate([phi_inside, phi_outside])  # Join potential values.
    r_sym = np.concatenate([-r_full[::-1], r_full])  # Mirror for |r| symmetry.
    phi_sym = np.concatenate([phi_full[::-1], phi_full])  # Mirror potential.
    sorted_indices = np.argsort(r_sym)  # Ensure increasing radius for interp1d.

    r_unit = unit.R_earth  # Public radius unit expected by callers.
    Phi_unit = unit.megajoule / unit.kilogram  # Public potential unit.
    return (
        interp1d(
            r_sym[sorted_indices].to_value(r_unit),
            phi_sym[sorted_indices].to_value(Phi_unit),
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
        ),
        r_unit,
        Phi_unit,
    )


def earth_grav_potential_earth_center():
    """
    Returns a function Phi(r), valid both inside and outside Earth.
    Uses PREM-like model for interior, point-mass approximation for exterior.
    """
    msgPrefix = f"[{earth_grav_potential_earth_center.__name__}]"
    Phi_infinity, r_unit, Phi_unit = earth_grav_potential_infty()
    center_value = Phi_infinity(0.0)

    def Phi_func(radius):
        """Shift the infinity-normalized potential so Phi(0) = 0."""
        return Phi_infinity(radius) - center_value

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
    Phi_integral_func, integral_r_unit, integral_Phi_unit = (
        earth_grav_potential_infty2()
    )
    Phi_center_func, center_r_unit, center_Phi_unit = (
        earth_grav_potential_earth_center()
    )
    # extend to radii beyond Earth's surface
    r_extended = np.linspace(0, 3, 1000) * unit.R_earth
    Phi_extended = Phi_func(r_extended.to_value(r_unit)) * (Phi_unit)
    Phi_center_extended = (
        Phi_center_func(r_extended.to_value(center_r_unit)) * center_Phi_unit
    )
    Phi_integral_extended = (
        Phi_integral_func(r_extended.to_value(integral_r_unit)) * integral_Phi_unit
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
        label="$\\Phi(r)=-G[M(r)/r+\\int_r^{R_\\oplus}dM(r')/r']$",
        color="darkorange",
        linestyle="-",
        # zorder=8,
    )
    (infinity_integral_line,) = pot_ax.plot(
        r_extended.to_value(r_unit),
        Phi_integral_extended.to_value(Phi_unit),
        label="$\\Phi(r)=-G[M(r)/r+\\int_r^{R_\\oplus}dM(r')/r']$",
        color="tab:purple",
        linestyle=":",
        # zorder=10,
    )
    center_ax = pot_ax.twinx()
    # (center_line,) = center_ax.plot(
    #     r_extended.to_value(r_unit),
    #     Phi_center_extended.to_value(Phi_unit),
    #     label="$\\Phi(0)=0$",
    #     color="tab:red",
    #     linestyle="--",
    #     zorder=1,
    # )
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
    # pot_ax.legend(
    #     [infinity_line, infinity_integral_line, center_line],
    #     [
    #         infinity_line.get_label(),
    #         infinity_integral_line.get_label(),
    #         center_line.get_label(),
    #     ],
    #     loc="lower right",
    #     frameon=False,
    # )
    pot_ax.legend(
        [infinity_line, infinity_integral_line],
        [
            infinity_line.get_label(),
            infinity_integral_line.get_label(),
        ],
        loc="lower right",
        frameon=False,
    )
    # density
    density_ax.set_ylim(-0.5, 15.5)
    # potential
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
    rho_M_DM_SHM: Quantity[unit.g / unit.cm**3] = 0.3 * unit.GeV / (const.c**2 * unit.cm**3)

    def __init__(
        self,
        name="Earth-Bound Axion Halo",
        nu_a: Quantity[unit.Hz] | None = None,
        m_a: Quantity[unit.g] | None = None,
        N: int = int(2**12),
        extent: Quantity[unit.m] = 128.0 * unit.R_earth,
        getPot=earth_grav_potential_infty,
        a_0: Quantity[unit.eV] | None = None,
        totalMassEnclosed: Quantity[unit.kg] | None = 4e-9 * unit.M_earth,
        g_aNN: Quantity[unit.GeV**-1] = 1e-9 * unit.GeV**-1,
        rho_M_DM_SHM: Quantity[unit.g / unit.cm**3] = 0.3 * unit.GeV / (const.c**2 * unit.cm**3),
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
        self.rho_M_DM_SHM = rho_M_DM_SHM

    @staticmethod
    def _integrateRadialProbability(r, u, lower, upper):
        """Integrate radial probability with interpolated interval endpoints.

        Parameters
        ----------
        r : astropy.units.Quantity, shape (N,)
            Strictly increasing radial coordinates with length units, including
            domain endpoints when integrating the full domain.
        u : astropy.units.Quantity, shape (N,)
            Real or complex reduced radial wavefunction u=rR, in length**(-1/2),
            sampled on r. Supply zero boundary values where appropriate.
        lower, upper : astropy.units.Quantity
            Scalar length bounds satisfying r[0] <= lower <= upper <= r[-1].
            Units may differ from r provided they are compatible.

        Returns
        -------
        probability : astropy.units.Quantity
            Dimensionless scalar integral of abs(u)**2. Endpoint wavefunction
            values are obtained with np.interp; np.trapezoid integrates the
            squared magnitudes using the interior samples and both endpoints.

        Notes
        -----
        Interpolation does not recover unresolved wavefunction structure.
        Equal bounds return zero. Accuracy requires grid convergence.
        """
        # Reject extrapolation beyond the wavefunction grid.
        if not r[0] <= lower <= upper <= r[-1]:
            raise ValueError("Integration limits must be ordered and inside the grid.")
        # Keep native interior samples and insert the exact integration limits.
        interior = (r > lower) & (r < upper)
        interval_r = np.concatenate((Quantity([lower]), r[interior], Quantity([upper])))
        # Interpolate u at the limits, then integrate probability density |u|^2.
        interval_u = np.interp(interval_r, r, u)
        return np.trapezoid(np.abs(interval_u)**2, x=interval_r)

    def inferHaloMass(self, shell_mass, mass_uncertainty, radius_range,
                           state_names=None):
        """Infer halo mass from a measured shell mass for each eigenstate.

        The solved wavefunctions retain their unit-probability normalization;
        this method does not rescale or modify the stored eigenstates.

        Parameters
        ----------
        shell_mass : astropy.units.Quantity
            Finite, nonnegative scalar mass measured between the shell radii.
            Any mass unit is accepted, for example 0.3e-9 * unit.M_earth.
        mass_uncertainty : astropy.units.Quantity
            Finite, nonnegative scalar absolute uncertainty on shell_mass, in
            any mass unit. Propagated linearly, even when shell_mass is zero.
            No confidence level or positive upper limit is inferred.
        radius_range : astropy.units.Quantity or sequence of Quantity, shape (2,)
            Geocentric inner and outer radii (not altitudes), with length units.
            Must satisfy 0 <= inner < outer <= self.extent.
        state_names : sequence of str or None, optional
            Already-solved eigenstate labels, e.g. ["1s", "2p"]. None selects
            all solved states. Each state is an independent halo hypothesis,
            not a component of a mixture. TISE is not solved automatically.

        Returns
        -------
        results : dict
            Mapping from state label to shell_fraction (dimensionless),
            total_mass, total_mass_uncertainty, enclosed_mass, and
            enclosed_mass_uncertainty. All masses are Quantity in Earth masses.
            Total mass covers [0, extent]; enclosed mass covers [0, outer].

        Warns
        -----
        UserWarning
            Before integration if fewer than 10 stored grid samples lie in the
            shell, counting samples exactly on either boundary.

        Notes
        -----
        The finite-domain total approximates the full halo only after checking
        outer-radius convergence. The Earth-only potential ignores self-gravity.
        Each call replaces the stored inferred models with the selected states.
        Wavefunction and grid copies preserve the inference if TISE is rerun;
        call this method again to use new solutions. Constructor attributes
        totalMassEnclosed, N_a and field amplitudes are unchanged.
        A UserWarning flags shell fractions at or below float64 machine epsilon.
        This is a sensitivity diagnostic, not an absolute probability accuracy
        limit; check grid-resolution and outer-radius convergence independently.
        """
        # Validate the measured mass and its absolute uncertainty, retaining units.
        for name, value in (("shell_mass", shell_mass), ("mass_uncertainty", mass_uncertainty)):
            if not isinstance(value, Quantity) or not value.unit.is_equivalent(unit.kg):
                raise TypeError(f"{name} must be a mass Quantity.")
            if not value.isscalar or not np.isfinite(value) or value < 0 * unit.kg:
                raise ValueError(f"{name} must be finite, scalar and nonnegative.")
        # Interpret the shell as geocentric radii inside the simulated domain.
        bounds = Quantity(radius_range)
        if not bounds.unit.is_equivalent(unit.m):
            raise TypeError("radius_range must contain length quantities.")
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
            raise ValueError("radius_range must contain two finite radii.")
        lower, upper = bounds
        if not 0 * unit.m <= lower < upper <= self.extent:
            raise ValueError("Require 0 <= inner < outer <= extent.")
        # Warn before integration when the measured shell is sparsely sampled.
        points_in_shell = np.count_nonzero((self.r >= lower) & (self.r <= upper))
        if points_in_shell < 10:
            warnings.warn(
                f"The specified mass interval [{lower}, {upper}] contains only "
                f"{points_in_shell} radial grid points (fewer than 10). "
                "The inferred mass and density may be inaccurate; increase "
                "the grid resolution and check convergence. Interpolation "
                "does not resolve missing wavefunction structure.",
                UserWarning,
                stacklevel=2,
            )
        # Select independent single-state hypotheses from the solved eigenstates.
        names = list(self.states) if state_names is None else list(state_names)
        if not names:
            raise ValueError("Solve and select at least one eigenstate first.")
        # Restore the fixed zero-boundary coordinates omitted by the TISE solver.
        r = np.concatenate((0 * self.r[:1], self.r, self.extent.reshape(1)))
        models, results = {}, {}
        for name in names:
            if name not in self.states:
                raise ValueError(f"Eigenstate {name!r} has not been solved.")
            # Copy each state and append its zero endpoint amplitudes.
            u = np.pad(self.states[name]["u_r"], (1, 1)).copy()
            # Integrate the domain, measured shell, and interior of its outer radius.
            # Integrating the shell directly avoids subtracting nearly equal totals.
            total = self._integrateRadialProbability(r, u, r[0], r[-1])
            shell = self._integrateRadialProbability(r, u, lower, upper)
            inside = self._integrateRadialProbability(r, u, r[0], upper)
            if not np.isfinite(shell) or shell <= 0 or not np.isfinite(total) or total <= 0:
                raise ValueError(f"{name}: unresolved shell probability; cannot infer halo mass.")
            # Flag very small fractions without treating epsilon as an accuracy cutoff.
            shell_fraction = (shell / total).to(unit.one)
            if shell_fraction <= np.finfo(np.float64).eps:
                warnings.warn(
                    f"{name}: shell_fraction={shell_fraction:.6g} is at or below "
                    f"float64 machine epsilon ({np.finfo(np.float64).eps:.3g}). "
                    "This does not necessarily imply an inaccurate probability, "
                    "but mass and density inference can be sensitive to errors "
                    "in small wavefunction amplitudes. Check grid-resolution "
                    "and outer-radius convergence.",
                    UserWarning,
                    stacklevel=2,
                )
            # Convert probability into mass using the measured shell constraint.
            # Propagate absolute uncertainty separately so a zero central mass works.
            scale, error_scale = shell_mass / shell, mass_uncertainty / shell
            # Store a snapshot for later density queries and report masses in M_earth.
            models[name] = {"r": r.copy(), "u": u, "scale": scale,
                            "error_scale": error_scale}
            results[name] = {
                "shell_fraction": shell_fraction,
                "total_mass": (scale * total).to(unit.M_earth),
                "total_mass_uncertainty": (error_scale * total).to(unit.M_earth),
                "enclosed_mass": (scale * inside).to(unit.M_earth),
                "enclosed_mass_uncertainty": (error_scale * inside).to(unit.M_earth),
            }
        # Replace the previous inference only after every selected state succeeds.
        self._shell_mass_models = models
        return results

    def _shellModels(self, state_names):
        """Select stored single-state mass inferences.

        Parameters
        ----------
        state_names : sequence of str or None
            Labels previously evaluated by inferHaloMass. None selects all
            models from the most recent inference.

        Returns
        -------
        models : dict
            Mapping from label to stored grid, wavefunction, mass scale and
            uncertainty scale. Values reference the internal model dictionaries.

        Raises
        ------
        ValueError
            If inference has not run, the selection is empty, or a label is absent.
        """
        # Require an existing inference before evaluating mass or density.
        models = getattr(self, "_shell_mass_models", {})
        if not models:
            raise ValueError("Call inferHaloMass first.")
        # Resolve an optional subset without modifying the stored models.
        names = list(models) if state_names is None else list(state_names)
        if not names or any(name not in models for name in names):
            raise ValueError("Select states previously evaluated with inferHaloMass.")
        return {name: models[name] for name in names}

    def getEnclosedMass(self, radius=None, state_names=None):
        """Return enclosed mass and quoted uncertainty for each inferred state.

        Parameters
        ----------
        radius : astropy.units.Quantity or None, optional
            Finite scalar geocentric length in [0, model outer radius]. None
            uses the outer boundary, returning the total simulated-domain mass.
        state_names : sequence of str or None, optional
            Labels evaluated by inferHaloMass. None selects all stored models.

        Returns
        -------
        results : dict
            Mapping from state label to mass and uncertainty, both scalar
            Quantity in Earth masses. The integral extends from zero to radius.

        Notes
        -----
        Requires a prior call to inferHaloMass. Uses its stored wavefunctions
        and linearly propagated uncertainty, without interpreting confidence levels.
        """
        # Evaluate each independent hypothesis using its saved wavefunction.
        result = {}
        for name, model in self._shellModels(state_names).items():
            r = model["r"]
            # Default to the saved domain boundary; otherwise validate the radius.
            cutoff = r[-1] if radius is None else radius
            if not isinstance(cutoff, Quantity) or not cutoff.unit.is_equivalent(unit.m):
                raise TypeError("radius must be a length Quantity.")
            if not cutoff.isscalar or not np.isfinite(cutoff):
                raise ValueError("radius must be finite and scalar.")
            # Scale the probability inside the cutoff into mass and uncertainty.
            fraction = self._integrateRadialProbability(r, model["u"], r[0], cutoff)
            result[name] = {"mass": (model["scale"] * fraction).to(unit.M_earth),
                            "uncertainty": (model["error_scale"] * fraction).to(unit.M_earth)}
        return result

    def getDensity(self, radii, state_names=None):
        """Return angularly averaged mass density and its quoted uncertainty.

        Parameters
        ----------
        radii : astropy.units.Quantity
            Finite positive scalar or nonempty 1-D array of geocentric lengths,
            bounded above by each inferred model's outer radius. Zero is
            excluded to avoid evaluating u/r at the origin. No extrapolation.
        state_names : sequence of str or None, optional
            Labels evaluated by inferHaloMass. None selects all stored models.

        Returns
        -------
        results : dict
            Mapping from state label to density and uncertainty, both Quantity
            in g/cm**3 with the same shape as radii. These are physical mass
            densities, not ratios to rho_M_DM_SHM.

        Notes
        -----
        Requires inferHaloMass first. Linearly interpolates u with units intact
        and evaluates scale*abs(u)**2/(4*pi*radii**2). This is the angular
        average, not the directional density of a non-spherical eigenstate.
        """
        # Require physical radii and exclude the undefined u/r evaluation at zero.
        if not isinstance(radii, Quantity) or not radii.unit.is_equivalent(unit.m):
            raise TypeError("radii must be a length Quantity.")
        if radii.ndim > 1 or radii.size == 0 or not np.all(np.isfinite(radii)) or np.any(radii <= 0 * unit.m):
            raise ValueError("radii must be finite, positive, scalar or 1-D.")
        # Use the snapshots from the latest mass inference, not newly solved states.
        result = {}
        for name, model in self._shellModels(state_names).items():
            if np.any(radii > model["r"][-1]):
                raise ValueError("radii exceed the inferred model's domain.")
            # Evaluate u on the requested radii without stripping Astropy units.
            u = np.interp(radii, model["r"], model["u"])
            # Divide radial probability density by spherical area for its angular mean.
            profile = abs(u)**2 / (4 * np.pi * radii**2)
            # Apply the inferred mass and uncertainty scales to the same spatial shape.
            result[name] = {
                "density": (model["scale"] * profile).to(unit.g / unit.cm**3),
                "uncertainty": (model["error_scale"] * profile).to(unit.g / unit.cm**3),
            }
        return result

    def getDensityAtEarthSurface(self, state_names=None):
        """Return mean density at one Earth radius for the inferred states.

        Parameters
        ----------
        state_names : sequence of str or None, optional
            Labels evaluated by the latest inferHaloMass call, e.g.
            ["1s", "2p"]. None selects all states from that inference.

        Returns
        -------
        results : dict
            Mapping from state label to density and uncertainty (scalar
            Quantity in g/cm**3), plus density_ratio and uncertainty_ratio
            (dimensionless Quantity in units of self.rho_M_DM_SHM).

        Notes
        -----
        Call inferHaloMass first. Uses its saved wavefunctions and mass
        scales without solving TISE again. The model must extend to at
        least one Earth radius. Densities are angular averages, and the
        quoted uncertainty is propagated without a confidence-level assumption.
        """
        # Reuse the radius-based calculation and its inference/domain checks.
        results = self.getDensity(1 * unit.R_earth, state_names=state_names)

        # Accept the SHM reference in mass-density or energy-density units.
        reference = self.rho_M_DM_SHM.to(
            unit.g / unit.cm**3
        )
        if not reference.isscalar or not np.isfinite(reference) or reference <= 0 * reference.unit:
            raise ValueError("rho_M_DM_SHM must be a positive finite scalar density.")

        # Provide SHM ratios alongside physical densities, keeping all units intact.
        for result in results.values():
            result["density_ratio"] = (result["density"] / reference).to(unit.one)
            result["uncertainty_ratio"] = (result["uncertainty"] / reference).to(unit.one)
        return results

    def plotDMdensity(self, radii, state_names=None, showPlot=True,
                    radius_unit=unit.R_earth, density_unit=None, scales=("log", "log")):
        """Plot central densities and quoted uncertainty magnitudes separately.

        Parameters
        ----------
        radii : astropy.units.Quantity, shape (N,)
            At least two strictly increasing positive geocentric lengths,
            within the inferred model domain. Passed to getDensity.
        state_names : sequence of str or None, optional
            Labels evaluated by inferHaloMass. None selects all stored models.
        showPlot : bool, optional
            Display the figure with plt.show() if True (default). False still
            constructs and returns the figure for saving or customization.
        radius_unit : astropy.units.Unit, optional
            Length unit for the horizontal axis; defaults to unit.R_earth.
        density_unit : astropy.units.Unit or None, optional
            Mass-density unit for the vertical axes. None (default) plots
            dimensionless ratios to self.rho_M_DM_SHM. The SHM reference may
            be a mass or energy density and is converted using mass_energy.
        scales : tuple of str, optional
            Horizontal and vertical axis scales, each "linear" or "log".
            Passed to axes.set_xscale and axes.set_yscale.
            Defaults to ("log", "log") for logarithmic axes when possible.
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure of width 13 cm at 300 dpi, with tight_layout applied.
        axes : numpy.ndarray of matplotlib.axes.Axes, shape (2,)
            Central density panel followed by the quoted uncertainty panel.
        profiles : dict
            Unscaled physical density results from getDensity, in g/cm**3.

        Notes
        -----
        Requires inferHaloMass first. The separate uncertainty panel avoids
        implying a positive confidence interval on logarithmic axes. The SHM
        reference must be a positive finite scalar. Each panel uses a logarithmic
        y-axis if any selected value is positive, otherwise a linear axis.
        """
        # Compute physical profiles first; unit choices below affect only the display.
        profiles = self.getDensity(radii, state_names)
        if radii.ndim != 1 or radii.size < 2 or np.any(np.diff(radii) <= 0 * unit.m):
            raise ValueError("Plot radii must be a strictly increasing 1-D array.")
        # Accept SHM density supplied either as mass density or energy density.
        reference = self.rho_M_DM_SHM.to(unit.g / unit.cm**3, equivalencies=unit.mass_energy())
        if not reference.isscalar or not np.isfinite(reference) or reference <= 0 * reference.unit:
            raise ValueError("rho_M_DM_SHM must be a positive finite scalar density.")
        # Separate central estimates and uncertainties instead of clipping an error band.
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(13 / 2.54, 8 / 2.54), dpi=300)
        for name, profile in profiles.items():
            for ax, key in zip(axes, ("density", "uncertainty")):
                # Choose SHM ratios or physical units, converting to numbers only for plotting.
                plotted = ((profile[key] / reference).to(unit.one)
                           if density_unit is None else profile[key].to(density_unit))
                ax.plot(radii.to_value(radius_unit), plotted.value, label=name)
        # Use logarithmic axes when possible, allowing identically zero panels.
        for ax, key in zip(axes, ("density", "uncertainty")):
            ax.set_xscale(scales[0])
            if any(np.any(profile[key] > 0 * profile[key].unit) for profile in profiles.values()):
                ax.set_yscale(scales[1])
            ax.grid(alpha=.25)
            if key == "density":
                ax.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
        # Label the displayed units, then arrange and optionally show the figure.
        if density_unit is None:
            axes[0].set_ylabel("$\\rho^{\\oplus} / \\rho^{\\mathrm{SHM}}$")
            axes[1].set_ylabel("$\\Delta\\rho^{\\oplus} / \\rho^{\\mathrm{SHM}}$")
        else:
            axes[0].set_ylabel(f"Mean density ({density_unit})")
            axes[1].set_ylabel(f"Quoted uncertainty ({density_unit})")
        axes[1].set_xlabel(f"Radius ({radius_unit})")
        fig.tight_layout()
        if showPlot:
            plt.show()
        return fig, axes, profiles

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

    def plot_u_r(self, state_names=None, showPlot=True, verbose=False,
                 figsize=(8.5 / 2.54, 5.5 / 2.54)):
        """Overlay real u(r)=rR(r), following DM_overdensity/plot-rR_r.py.

        Parameters
        ----------
        state_names : sequence of str or None, optional
            Nonempty sequence of spectroscopic labels in plotting order.
            None uses ["1s", "2s", "3s", "2p", "4s", "3d", "5s", "3p", "6s", "4f"].
        showPlot : bool, optional
            Display the figure if True (default). False returns it without
            calling plt.show().
        verbose : bool, optional
            Pass timing/energy diagnostics to any automatic TISE solves.
            Defaults to False; unused when all requested states are available.
        figsize : tuple of float, optional
            Figure width and height in inches. Default is (8.5/2.54, 5.5/2.54),
            corresponding to 8.5 by 5.5 cm. Resolution is fixed at 300 dpi.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure with tight_layout applied.
        ax : matplotlib.axes.Axes
            Lines for real u(r), with radii in Earth radii and amplitudes in
            Earth-radius**(-1/2). Uses the example's color/line-style cycles
            and 1.4-point lines, with no legend or explicit axis limits.

        Notes
        -----
        If no states exist, solve the requested angular channels first,
        retaining at least 20 radial states per channel (limited by N).
        Otherwise reuse solved states and solve only channels with missing
        requested states. The instance's mass and grid settings are retained.
        """
        # Preserve the example's default state selection and plotting order.
        if state_names is None:
            state_names = ["1s", "2s", "3s", "2p", "4s", "3d", "5s", "3p", "6s", "4f"]
        names = list(state_names)
        if not names:
            raise ValueError("Select at least one eigenstate to plot.")
        # Parse n and orbital label l; channel state count is n-l.
        # Collect only states that have not already been solved.
        required = {}
        for name in names:
            if (not isinstance(name, str) or len(name) < 2
                    or not name[:-1].isdigit() or name[-1] not in self.orbitalLabels):
                raise ValueError(f"Invalid eigenstate label: {name!r}")
            l = self.orbitalLabels.index(name[-1])
            count = int(name[:-1]) - l
            if not 1 <= count <= self.N:
                raise ValueError(f"Eigenstate {name!r} is invalid or exceeds the grid size.")
            if name not in self.states:
                required[l] = max(required.get(l, 0), count)
        # Solve a fresh instance with the example's channel coverage, or fill gaps.
        if not self.states:
            self.solve_TISE_3D(
                l_vals=sorted(required),
                max_n_r=max(min(20, self.N), max(required.values())),
                verbose=verbose,
            )
        else:
            for l, count in sorted(required.items()):
                self.solve_TISE_3D(l_vals=[l], max_n_r=count, verbose=verbose)

        # Reproduce the example's color/style cycles and plot u in Earth-radius units.
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        for idx, name in enumerate(names):
            ax.plot(
                self.r.to_value(unit.R_earth),
                self.states[name]["u_r"].real.to_value(unit.R_earth**-0.5),
                label=name,
                color=colors[idx % len(colors)],
                linestyle=linestyles[idx % len(linestyles)],
                marker=markers[idx % len(markers)],
                markevery=0.1,
                markersize=2,
                # linewidth=1.4,
            )
        # Add physical axis labels and finish layout without forcing limits or a legend.
        ax.set_xlabel(r"$r\,(R_\oplus)$")
        ax.set_ylabel(r"$r\,R(r)\,(R_\oplus^{-1/2})$")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()
        if showPlot:
            plt.show()
        return fig, ax

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

        startIdx = 0  # all stored radii are positive
        if truncRadius is None or type(truncRadius) != Quantity:
            stopIdx = self.N
        elif truncRadius.unit.is_equivalent(self.r.unit):
            stopIdx = startIdx + np.argmin(np.abs(self.r[startIdx:] - truncRadius))
        else:
            raise TypeError(
                f"{_getLocationPrefix()} {msgPrefix} truncRadius unit is not equivalent to length. "
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
