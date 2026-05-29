import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import correlate as correlate

from scipy.interpolate import interp1d

from astropy import units as unit
from astropy.constants import codata2018 as const
from astropy.coordinates import EarthLocation
from astropy.units import Quantity

from axionbloch.utils import check
from axionbloch.GravBoundAxionHalo import GravBoundAxionHalo


def loadPEMdata(
    filepath="axionbloch/data/Earth_Models/PEM_Parametric_Earth_Models_data.txt",
):
    logPrefix = f"[{loadPEMdata.__name__}]"
    data = {
        "radius_m": [],
        "density_kg_m3": [],
        "vpv": [],
        "vsv": [],
        "Q_kappa": [],
        "Q_miu": [],
        "vph": [],
        "vsh": [],
        "eta": [],
    }

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            values = [float(x) for x in line.split()]

            for key, val in zip(data.keys(), values):
                data[key].append(val)

    # sort by radius
    order = sorted(range(len(data["radius_m"])), key=lambda i: data["radius_m"][i])

    for key in data:
        data[key] = [data[key][i] for i in order]
    for key in data:
        data[key] = np.asanyarray(data[key])
    return data


def PREM_density(radius_km):
    logPrefix = f"[{PREM_density.__name__}]"
    # Use the coefficients of the polynomials describing the Preliminary Reference Earth Model (PREM) to find out density.
    radius_km = np.abs(radius_km)

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
        if x <= 5771 / EARTH_RADIUS_KM:
            return 5.3197 - 1.4836 * x
        elif x <= 5971 / EARTH_RADIUS_KM:
            return 11.2494 - 8.0298 * x
        else:
            return 7.1089 - 3.8045 * x

    def density_lvz_lid(x):
        return 2.6910 + 0.6924 * x

    def density_crust(x):
        if x <= 6356 / EARTH_RADIUS_KM:
            return 2.900
        else:
            return 2.600

    def density_ocean(x):
        return 1.020

    def density_space(x):
        return 0.0

    # # Generate radius values (in km) at 10 km intervals
    # radius_km = np.arange(0, EARTH_RADIUS_KM., 10)

    # Calculate density for each radius
    # density = []
    x = radius_km / EARTH_RADIUS_KM
    # radius_km = radius_km
    if radius_km <= 1221.5:
        return density_inner_core(x)
    elif radius_km <= 3480.0:
        return density_outer_core(x)
    elif radius_km <= 5701.0:
        return density_lower_mantle(x)
    elif radius_km <= 6151.0:
        return density_transition_zone(x)
    elif radius_km <= 6346.6:
        return density_lvz_lid(x)
    elif radius_km <= 6368.0:
        return density_crust(x)
    elif radius_km <= 6371.0:
        return density_ocean(x)
    else:
        return density_space(x)


def earth_grav_potential_infty():
    """
    Returns a function Phi(r[m]) [J/kg], valid both inside and outside Earth.
    Uses PREM-like model for interior, point-mass approximation for exterior.
    """
    logPrefix = f"[{earth_grav_potential_infty.__name__}]"
    # Load the data (assumed to return a DataFrame with 'radius_m' and 'density_kg_m3')
    data = loadPEMdata()

    # Extract radius and density
    r = data["radius_m"]
    rho = data["density_kg_m3"]

    # Compute shell thickness
    dr = np.gradient(r)

    # Shell volume and mass
    dV = 4 * np.pi * r**2 * dr
    dm = rho * dV

    # Cumulative mass
    M_r = np.cumsum(dm)
    M_total = M_r[-1]

    # Gravitational constant
    G = const.G.value  # m³/kg/s²

    # Extend to radii beyond Earth's surface
    r_max = r[-1]
    r_outside = np.linspace(
        r_max, 1000 * r_max, 800
    )  # from surface to 1000 Earth radii
    Phi_outside = -G * M_total / r_outside

    # Compute gravitational potential inside Earth
    Phi_inside = np.zeros_like(r)
    Phi_inside[1:] = 2 * Phi_outside[0] + G * M_r[1:] / r[1:]
    Phi_inside[0] = 2 * Phi_outside[0] + G * M_r[1] / r[1]  # avoid zero-division

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

    # Interpolation function: now Phi_func(-r) = Phi_func(r)
    Phi_func = interp1d(
        r_sym_sorted,
        Phi_sym_sorted,
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )

    return Phi_func


def get_CumulativeMass():
    """
    Returns the cumulative mass as a function of radius.
    Uses PREM-like model for interior, point-mass approximation for exterior.
    """
    logPrefix = f"[{get_CumulativeMass.__name__}]"
    # Load the data (assumed to return a DataFrame with 'radius_m' and 'density_kg_m3')
    data = loadPEMdata()

    # Extract radius and density
    r = data["radius_m"] * unit.meter
    rho = data["density_kg_m3"] * (unit.kg / unit.meter**3)

    # Compute shell thickness
    dr = np.gradient(r)

    # Shell volume and mass
    dV = 4 * np.pi * r**2 * dr
    dm = rho * dV

    # Cumulative mass
    M_r = np.cumsum(dm)
    return r, M_r


def earth_grav_potential_earth_center():
    """
    Returns a function Phi(r), valid both inside and outside Earth.
    Uses PREM-like model for interior, point-mass approximation for exterior.
    """
    logPrefix = f"[{earth_grav_potential_earth_center.__name__}]"
    r, M_r = get_CumulativeMass()
    M_total = M_r[-1]

    # Extend to radii beyond Earth's surface
    r_max = r[-1]
    r_outside = np.linspace(
        r_max, 1000 * r_max, 100000
    )  # from surface to 1000 Earth radii
    Phi_outside: Quantity = -const.G * M_total / r_outside  # G = 6.67430e-11 m³/kg/s²

    # Compute gravitational potential inside Earth
    Phi_inside = np.zeros_like(r) / r.unit * Phi_outside[0].unit
    Phi_inside[1:] = 2 * Phi_outside[0] + const.G * M_r[1:] / r[1:]
    Phi_inside[0] = 2 * Phi_outside[0] + const.G * M_r[1] / r[1]  # avoid zero-division

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
    logPrefix = f"[{plot_earth_grav_potential.__name__}]"
    # load the data to obtain density profile
    data = loadPEMdata()
    density_r = data["radius_m"] * unit.meter
    density_rho = data["density_kg_m3"] * (unit.kg / unit.meter**3)

    # cumulative mass
    mass_r, mass_M_r = get_CumulativeMass()

    # potential referring to earth center
    Phi_func, r_unit, Phi_unit = earth_grav_potential_earth_center()
    # extend to radii beyond Earth's surface
    r_extended = np.linspace(0, 3 * mass_r[-1], 1000)
    Phi_extended = Phi_func(r_extended.to_value(r_unit)) * Phi_unit

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

    cm = 1 / 2.56  # convert cm to inch

    fig = plt.figure(figsize=(8.5 * cm, 8.5 * cm), dpi=300)  # initialize a figure

    gs = gridspec.GridSpec(nrows=3, ncols=1)  # create grid for multiple figures

    # # fix the margins
    # left=
    # bottom=
    # right=
    # top=
    # wspace=
    # hspace=
    # fig.subplots_adjust(left=left, top=top, right=right,
    #                     bottom=bottom, wspace=wspace, hspace=hspace)

    density_ax = fig.add_subplot(gs[0, 0])
    mass_ax = fig.add_subplot(gs[1, 0], sharex=density_ax)
    pot_ax = fig.add_subplot(gs[2, 0], sharex=density_ax)

    # density profile
    density_ax.plot(
        density_r.to_value(r_unit),
        density_rho.to_value(density_rho.unit),
        label="Density Profile",
        color="darkblue",
    )
    density_ax.set_xlabel(f"Radius (earth radius)")
    density_ax.set_ylabel(f"Density ({density_rho.unit})")

    # cumulative mass
    mass_ax.plot(
        mass_r.to_value(r_unit),
        mass_M_r.to_value(unit.kg) / 1e24,
        label="Mass Profile",
        color="darkgreen",
    )
    mass_ax.set_xlabel(f"Radius (earth radius)")
    mass_ax.set_ylabel(f"Enclosed Mass ($10^{{24}}\\,$kg)")
    mass_ax.ticklabel_format(useOffset=False)

    # gravitational potential
    pot_ax.plot(
        r_extended.to_value(r_unit),
        Phi_extended.to_value(Phi_unit),
        label="Earth grav. pot.",
        color="darkorange",
    )
    pot_ax.axvline(
        x=(1 * unit.R_earth).to_value(r_unit),
        color="k",
        linestyle="dotted",
        linewidth=1,
        alpha=0.8,
        label="Earth radius",
    )

    pot_ax.set_xlabel(f"Radius (earth radius)")
    pot_ax.set_ylabel(f"Grav. Pot. (MJ/kg) ref. to $\\infty$")
    pot_ax.legend()

    fig.suptitle("Earth Profiles (from PEM Data)")
    fig.tight_layout()
    # plt.savefig("figures/Earth-Profiles-(PEM-Data).png", transparent=False)
    if showplot:
        plt.show()
    else:
        plt.close(fig)


class EarthBoundAxionHalo(GravBoundAxionHalo):
    # Create the "axion stream" (axion field) object
    # you can get properties of the axion field, computed based on the input information
    def __init__(
        self,
        name="Earth-Bound Axion Halo",
        nu_a: Quantity = None,  # axion Compton frequency
        N: int = int(2**12),
        extent: Quantity = 128.0 * unit.R_earth,
        getPot=earth_grav_potential_earth_center,
        a_0: Quantity | None = None,
        rhoE_DM: Quantity = 1e16 * 0.3 * unit.GeV / unit.cm**3,
        totalMassEnclosed: Quantity | None = 4e-9 * unit.M_earth,
        g_aNN: Quantity = 1e-9 * unit.GeV**-1,
        verbose: bool = False,
    ):
        logPrefix = f"[{self.__class__.__name__}.__init__]"
        super().__init__(
            name=name,
            nu_a=nu_a,
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
        if a_0 is not None:
            self.a_0 = a_0
        elif rhoE_DM is not None:
            self.a_0 = (2 * const.hbar**3 * rhoE_DM / (self.m_a**2 * const.c)) ** 0.5
        else:
            raise ValueError("Either a_0 or rhoE_DM must be provided.")
        self.totalMassEnclosed = totalMassEnclosed

    # ------------------------------------------------------------------
    # Gradient at arbitrary direction / time
    # ------------------------------------------------------------------

    def findGradientsAtDirection(
        self,
        location: EarthLocation,
        stateNames: list[str] | None = None,
        truncRadius: Quantity | None = None,
        showPlot: bool = False,
        verbose: bool = False,
    ) -> tuple:
        """Compute the wavefunction gradient along an arbitrary direction.

        A convenience wrapper around :meth:`findGradients` that accepts an
        :class:`~astropy.coordinates.EarthLocation` directly, without requiring
        a pre-defined :class:`~axionbloch.Station.Station`.

        Parameters
        ----------
        location : EarthLocation
            Geodetic latitude, longitude, and elevation of the direction.
            For example: ``EarthLocation(lat=0*u.deg, lon=0*u.deg, height=0*u.m)``
            points toward the equator on the prime meridian.
        stateNames : list of str, optional
            Eigenstates to superimpose (e.g. ``['2p']``).  Defaults to the
            lowest-energy solved state.
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

        Examples
        --------
        >>> from astropy.coordinates import EarthLocation
        >>> from astropy import units as u
        >>> # Direction toward the equator on the prime meridian
        >>> r, R_r, r_line, gr, gt, gp = halo.findGradientsAtDirection(
        ...     location=EarthLocation(lat=0*u.deg, lon=0*u.deg, height=0*u.m),
        ...     stateNames=['2p'],
        ...     truncRadius=2 * u.R_earth,
        ... )
        """
        return self.findGradients(
            stateNames=stateNames,
            location=location,
            truncRadius=truncRadius,
            showPlot=showPlot,
            verbose=verbose,
        )

    def findGradientsWithMilkyWay(
        self,
        mw,
        stateNames: list[str] | None = None,
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
          station's sensitive axis.
        - Gradient in Cartesian ITRS coordinates (useful for projecting onto
          a non-vertical :math:`\\mathbf{B}_0`).
        - Projection of the gradient onto the sensitive axis (radial direction).

        Parameters
        ----------
        mw : :class:`~axionbloch.MilkyWay.MilkyWay`
            Must have :attr:`~axionbloch.MilkyWay.MilkyWay.station` set.
        stateNames : list of str, optional
            Eigenstates to include (e.g. ``['2p']``).
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

            ``wind_angle`` — angle between v_lab and the sensitive axis [rad].

        Examples
        --------
        >>> from astropy.time import Time
        >>> from axionbloch.MilkyWay import MilkyWay
        >>> from axionbloch.Station import Baltimore
        >>> mw = MilkyWay(time=Time('2024-06-21T14:00:00'), station=Baltimore)
        >>> result = halo.findGradientsWithMilkyWay(
        ...     mw, stateNames=['2p'], truncRadius=2 * unit.R_earth
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
        r, R_r, r_line, grad_r, grad_theta, grad_phi = self.findGradients(
            stateNames=stateNames,
            station=station,
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
        phi_s   = float(station.phi.to_value(unit.rad))
        sin_t, cos_t = np.sin(theta_s), np.cos(theta_s)
        sin_p, cos_p = np.sin(phi_s),   np.cos(phi_s)

        r_hat     = np.array([ sin_t * cos_p,  sin_t * sin_p,  cos_t ])
        theta_hat = np.array([ cos_t * cos_p,  cos_t * sin_p, -sin_t ])
        phi_hat   = np.array([-sin_p,           cos_p,          0.0   ])

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
        nvec_gcrs  = mw.get_nvec_gcrs()
        v_lab      = mw.get_v_lab()
        v_lab_mag  = mw.get_v_lab_magnitude()
        wind_angle = mw.get_wind_angle()

        if verbose:
            print(f"[findGradientsWithMilkyWay] station       = {station.name}")
            print(f"[findGradientsWithMilkyWay] time          = {mw.time.iso}")
            print(f"[findGradientsWithMilkyWay] |v_lab|       = {v_lab_mag:.3f}")
            print(f"[findGradientsWithMilkyWay] wind_angle    = {wind_angle.to(unit.deg):.2f}")
            print(f"[findGradientsWithMilkyWay] grad_r surf.  = {gr_s}")
            print(f"[findGradientsWithMilkyWay] grad_th surf. = {gt_s}")
            print(f"[findGradientsWithMilkyWay] grad_ph surf. = {gp_s}")
            print(f"[findGradientsWithMilkyWay] grad Cartesian (ITRS) = {grad_cartesian_itrs}")

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
        logPrefix = f"[{self.__class__.__name__}.{self.getBfield.__name__}]"
        # 1 MHz axion
        eigenEnergies_eV = [
            8.185620266405553e-19,
            2.230884526115819e-18,
            3.17699654698912e-18,
            3.3025178098192375e-18,
            3.302688446107381e-18,
            3.8484585622919455e-18,
            4.1004249795459864e-18,
            4.1004251503701215e-18,
            4.341652193125044e-18,
            4.407871816010306e-18,
            4.408009935951216e-18,
            4.649824763113033e-18,
            4.793034098628449e-18,
            4.7930342283222715e-18,
            4.8831463914209464e-18,
            4.919259056702106e-18,
            4.919342580168352e-18,
            5.042907628381992e-18,
            5.124371776790224e-18,
            5.124371863603223e-18,
            5.1684784792208225e-18,
            5.1897014793057166e-18,
            5.189752636536869e-18,
            5.260382049853425e-18,
            5.310111086559351e-18,
            5.310111144839848e-18,
            5.335147509172691e-18,
            5.348511363425495e-18,
            5.348544297168566e-18,
            5.3924738060527425e-18,
            5.424802508063358e-18,
            5.4248025483118064e-18,
            5.440435917665003e-18,
            5.449341478496205e-18,
            5.449363729093159e-18,
            5.478480245818686e-18,
            5.500595123173712e-18,
            5.5005951518592556e-18,
            5.511033438087251e-18,
            5.517245349489185e-18,
            5.517261015401705e-18,
            5.537528824223334e-18,
            5.553289963907005e-18,
            5.553289984958548e-18,
            5.560615718245721e-18,
            5.565112331351988e-18,
            5.565123747711691e-18,
            5.579789494642934e-18,
            5.591403811323711e-18,
            5.591403827177387e-18,
        ]
        self.rate_Hz: float = rate_Hz
        self.timeLen: int = timeLen
        self.duration_s: float = self.timeLen / self.rate_Hz
        eigenEnergies_eV = np.asarray(eigenEnergies_eV)
        eigenFreqs_Hz: np.ndarray = eigenEnergies_eV
        if verbose:
            print(eigenFreqs_Hz.mean())
        B_rms_T = 1e-15
        self.Ba = np.zeros(timeLen)
        timeStamp = np.arange(timeLen) / rate_Hz
        rng = (
            np.random.default_rng(seed=rand_seed)
            if rand_seed is not None
            else np.random.default_rng()
        )

        # phases
        phases = (
            2 * np.pi * rng.random((len(eigenFreqs_Hz), numFields))
        )  # shape: (numFreqs, numFields)

        phase_time: np.ndarray = (
            2 * np.pi * eigenFreqs_Hz[:, None] * timeStamp[None, :]
        )  # shape: (numFreqs, timeLen)

        total_phase = phase_time[:, :, None] + phases[:, None, :]
        # shape: (numFreqs, timeLen, numFields)

        self.Ba = np.exp(1j * total_phase).sum(axis=0)  # shape: (timeLen, numFields)
        self.Ba *= B_rms_T
        if verbose:
            print("phases.shape", phases.shape)
            print("phase_time.shape", phase_time.shape)
            print("total_phase.shape", total_phase.shape)
            print("self.Ba.shape", self.Ba.shape)
        return self.Ba

    def coh_time_g1(self):
        """
        x : complex-valued time series
        dt: sampling interval
        method: "1e" or "integral"
        """
        logPrefix = f"[{self.__class__.__name__}.{self.coh_time_g1.__name__}]"
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
        print("tau =", tau)
        print("duration =", self.duration_s)
        if tau > self.duration_s:
            print("WARNING: tau > self.duration_s")
        return tau


# rate_Hz = 3e-2
# duration_s = 1e6
# timeLen = int(rate_Hz * duration_s)
# axion = EarthBoundAxionHalo()
# axion.getBfield(rate_Hz=rate_Hz, timeLen=timeLen, rand_seed=1, numFields=1, verbose=False)
# axion.coh_time_g1()
