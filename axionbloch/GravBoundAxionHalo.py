"""Solver for gravitationally-bound axion halos around bodies.

:class:`GravBoundAxionHalo` solves the time-independent Schrödinger equation
(TISE) on a 1-D radial grid using a finite-difference Hamiltonian, returning
bound-state wavefunctions and energies for each orbital quantum number *l*.

:class:`EarthBoundAxionHalo` (in :mod:`axionbloch.EarthBoundAxionHalo`) is a
subclass pre-configured with the Earth's gravitational potential.
"""

import time

from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import eigh
from scipy.sparse import diags

# for wavefunction construction
from scipy.special import sph_harm_y

from axionbloch.dependency import *
from axionbloch.Station import Station
from axionbloch.utils import high_contrast_extended as colors


class GravBoundAxionHalo:
    """Solve the TISE for axions gravitationally bound to compact bodies.

    Constructs a 1-D radial finite-difference Hamiltonian ``H = T + V`` on a
    uniform grid spanning ``[-extent/2, extent/2]``, diagonalizes it for each
    requested angular-momentum channel *l*, and stores the resulting
    wavefunctions and energy expectation values in :attr:`states`.

    Note: input / output units are in SI, while internal calculations are in
    atomic units.
    """

    # Map l to spectroscopic labels (s, p, d, f, ...)
    orbitalLabels = ["s", "p", "d", "f", "g", "h", "i", "k", "l", "m"]
    pot: Quantity
    mass_enclosed: Quantity | None
    g_aNN: Quantity | None
    a_0: Quantity | None

    def __init__(
        self,
        name="Gravitationally Bound Axion Halo",
        nu_a: Quantity | None = None,  # axion Compton frequency
        N: int = int(2**12),
        extent: Quantity = 128.0 * unit.R_earth,
        getPot=None,
        mass_enclosed=None,
        g_aNN=None,
        a_0=None,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        name : str
            Human-readable label.
        nu_a : Quantity [Hz]
            Axion Compton frequency, used to derive the axion mass.
        N : int
            Number of grid points (default 4096).
        extent : Quantity
            Full radial range of the grid (centred on r = 0).
        getPot : callable
            Zero-argument function returning ``(Phi_func, r_unit, Phi_unit)``
            where ``Phi_func(r_in_r_unit)`` is the gravitational potential
            profile.
        mass_enclosed : Quantity, optional
            Enclosed mass profile (used for gradient calculations).
        g_aNN : Quantity, optional
            Axion-nucleon coupling constant (GeV⁻¹).
        verbose : bool
            Print axion mass and frequency after construction.
        """
        msgPrefix = f"[{self.__class__.__name__}.{self.__init__.__name__}]"
        self.name = name
        self.nu_a = nu_a
        # axion mass derived from Compton frequency: m = h * nu / c²
        self.m_a = self.nu_a * const.h / const.c**2
        if verbose:
            print(msgPrefix, "axion Compton frequency =", self.nu_a)
            print(
                msgPrefix,
                f"axion mass = {self.m_a.to(unit.kg)}",
                f"= {self.m_a.to_value(unit.eV / const.c**2):g} eV/c^2",
            )

        self.N = N
        self.extent: Quantity = extent
        self.dr: Quantity = self.extent / self.N
        # symmetric grid centred on r=0
        self.r: Quantity = np.linspace(-self.extent / 2, self.extent / 2, self.N)
        Phi_func, r_unit, Phi_unit = getPot()
        # TODO: Try also to use the infinity as the reference point
        # gravitational potential V = m_a * Phi(r), evaluated on the radial grid
        self.pot: Quantity = (
            self.m_a
            * np.asarray(Phi_func((self.r / r_unit).to_value(unit.one)))
            * Phi_unit
        )  # convert r to meter for potential function

        self.mass_enclosed = mass_enclosed
        self.g_aNN = g_aNN
        self.a_0 = a_0

        # prefactor for the kinetic energy finite-difference stencil: ℏ²/(2m dr²)
        self.T_magnitude: Quantity = (const.hbar**2) / (2 * self.m_a) / self.dr**2

        self.states: dict = {}
        self.l_vals = []

        self.E_unit = unit.attoelectronvolt
        # usually we do not need all N eigenstates, so we store only the first max_n_r states for each l.

    def showValueAndUnits(self):
        """Print values and units of the key physical quantities.

        Ideally the numerical values should be close to 1 and units should be
        identical for quantities compared or added together, making unit
        mismatches easy to spot during development.
        """
        msgPrefix = f"[{self.__class__.__name__}.{self.showValueAndUnits.__name__}]"
        print(msgPrefix, "self.r.mean():", self.r.mean())  # mean of radial grid
        print(msgPrefix, "self.r.std():", self.r.std())  # spread of radial grid
        print(msgPrefix, "self.dr:", self.dr)  # grid spacing
        print(
            msgPrefix, "self.pot.mean():", self.pot.mean()
        )  # mean gravitational potential energy
        print(
            msgPrefix, "self.pot.std():", self.pot.std()
        )  # spread of gravitational potential energy
        print(
            msgPrefix, "self.T_magnitude:", self.T_magnitude
        )  # kinetic energy prefactor ℏ²/(2m dr²)
        print(msgPrefix, "self.m_a:", self.m_a.si)  # axion mass

    def solve_TISE_3D_l(
        self,
        l: int,  # angular momentum quantum number
        showPlot: bool = False,
        max_n_r: int = 10,  # maximum radial quantum number to plot
        verbose: bool = False,
    ):
        """Solve the TISE for a single angular-momentum channel *l*.

        Builds the finite-difference Hamiltonian ``H = T + V_eff`` (with the
        centrifugal barrier included in ``V_eff``), diagonalizes it with
        ``scipy.linalg.eigh``, normalizes the lowest ``max_n_r`` eigenstates,
        computes expectation values of T, V, and V_eff, and stores the results
        in :attr:`states`.

        Parameters
        ----------
        l : int
            Orbital angular-momentum quantum number.
        showPlot : bool
            If True, plot the reduced radial wavefunctions after solving.
        max_n_r : int
            Number of radial eigenstates to retain (counted from the ground
            state of channel *l*).
        verbose : bool
            Print timing and eigen-energy tables.
        """
        msgPrefix = f"[{self.__class__.__name__}.{self.solve_TISE_3D_l.__name__}]"
        # Effective potential including centrifugal term ℏ²l(l+1)/(2m r²)
        Veff = Quantity(
            self.pot + const.hbar**2 * l * (l + 1) / (2 * self.m_a * self.r**2)
        )

        main = -2.0 * np.ones(self.N)
        off = 1.0 * np.ones(self.N - 1)

        # ----------------- start of dimensionless computation ---------------- #
        # Kinetic energy operator: T = -(ℏ²/2m) ∇², discretised as a tridiagonal matrix
        lap = diags([off, main, off], [-1, 0, 1])
        T = -1 * self.T_magnitude.to_value(self.pot.unit) * lap

        # Hamiltonian H = T + diag(V_eff), all values in self.pot.unit
        H = T + diags(Veff.to_value(self.pot.unit), 0)

        H_dense = H.toarray()

        # Solve eigenvalue problem; energies in self.pot.unit, states are dimensionless
        tic = time.time()
        energies_pot_unit, states = eigh(H_dense)
        toc = time.time()
        if verbose:
            print(
                msgPrefix, f"N={self.N} l={l} Eigensolver took {toc - tic:.3f} seconds"
            )
        # ----------------- end of dimensionless computation ---------------- #
        energies = np.zeros_like(energies_pot_unit) * self.pot.unit
        for i, e in enumerate(energies_pot_unit):
            energies[i] = e * self.pot.unit

        if verbose:
            print(msgPrefix, "Eigen-energies:")
            print("[")
            for i, e in enumerate(energies_pot_unit[0:max_n_r]):
                print(f"{e:.6e},")
            print("]")
            print(f"* {self.pot.unit}")

        # Skip the first half of the grid (r < 0) plus a few extra points to avoid r=0 singularity
        start_index = self.N // 2 + 5

        # For l>0 the even-indexed eigenstates carry the correct parity; skip odd ones
        if l == 0:
            iter_range = np.arange(max_n_r)
        else:
            iter_range = np.arange(2 * max_n_r)[::2]

        for i, _n_r in enumerate(iter_range):

            u_r = states[:, _n_r]
            # radial wavefunction R(r) = u(r)/r
            R_r = u_r / self.r

            # Normalize so that 4π ∫ |u(r)|² dr = 1
            integral = np.sqrt(
                1.0
                * np.trapezoid(
                    np.abs(u_r[start_index:]) ** 2,
                    self.r[start_index:],
                )
            )
            R_r = R_r / integral
            u_r = u_r / integral

            # Potential energy expectation value ⟨V⟩ = ∫ |u|² V dr
            V_expect = np.trapezoid(
                np.abs(u_r[start_index:]) ** 2 * self.pot[start_index:],
                self.r[start_index:],
            )

            # Effective potential expectation value ⟨V_eff⟩ = ∫ |u|² V_eff dr
            Veff_expect = np.trapezoid(
                np.abs(u_r[start_index:]) ** 2 * Veff[start_index:],
                self.r[start_index:],
            )

            # Kinetic energy via second derivative of u_r: ⟨T⟩ = -(ℏ²/2m) ∫ u* u'' dr
            # Initialize with correct units: [u''] = [u] / [r]²
            du2_dr2 = np.zeros(u_r.shape) * u_r.unit / self.r.unit**2
            du2_dr2[1:-1] = (u_r[2:] - 2 * u_r[1:-1] + u_r[:-2]) / (
                self.r[1] - self.r[0]
            ) ** 2
            T_expect = -(const.hbar**2 / (2 * self.m_a)) * np.trapezoid(
                np.conj(u_r[start_index:]) * du2_dr2[start_index:], self.r[start_index:]
            )
            # Reduced wavefunction normalized by discrete L2 norm (used for plotting only)
            R_reduced = (
                1.0
                * (states[:, _n_r]) ** 1
                / np.sqrt(np.trapezoid(np.abs(states[:, _n_r]) ** 2, self.r))
            )
            # print(
            #     f"n_r={n_r}, l={l_val}: T={T_expect:.3e}, V={V_expect:.3e}, Veff={Veff_expect:.3e}, \
            #     E_total={(T_expect+Veff_expect):.3e}, eigen_E={E[n_r]:.3e}"
            # )
            # print(f"{T_expect:.6e},")
            n = i + l + 1
            self.states[f"{n}{self.orbitalLabels[l]}"] = {
                "key_info": "",
                "name": f"{n}{self.orbitalLabels[l]}",
                "n_r_l": (i, l),
                "n_r": (i),
                "l": (l),
                "eigenE": energies[_n_r],
                "T_expect": T_expect,
                "V_expect": V_expect,
                "Veff_expect": Veff_expect,
                "eigenE_expect": (
                    T_expect + Veff_expect
                ),  # TODO check if this is consistent with eigenE
                "u_r": u_r,
                "R_r": R_r,
                "R_reduced": R_reduced,
            }

        if showPlot:
            R_reduced = (
                1.0
                * (states[:, :max_n_r]) ** 1
                / np.sqrt(
                    np.trapezoid(np.abs(states[:, :max_n_r]) ** 2, self.r, axis=0)
                )
            )
            # R_reduced.shape = (N, max_n_r)
            # TODO: complete this
            # slider_plot_earth(
            #     dataX=self.r[start_index:],
            #     dataY=(R_reduced[start_index:, :]),
            #     title=f"Reduced radial wavefunction (l={l})",
            #     # xlabel="r (earth_radius)",
            #     xlim=None,
            #     show_real_imag=True,
            # )

    def solve_TISE_3D(
        self,
        l_vals=[3],  # angular momentum quantum number
        max_n_r: int = 64,  # maximum principal quantum number to plot
        showPlot: bool = False,
        verbose: bool = False,
    ):
        """Solve the TISE for all requested angular-momentum channels.

        Iterates over ``l_vals``, calling :meth:`solve_TISE_3D_l` for each,
        then sorts all accumulated states by eigen-energy.

        Parameters
        ----------
        l_vals : list of int
            Orbital angular-momentum quantum numbers to solve.
        max_n_r : int
            Number of radial eigenstates to retain per channel.
        showPlot : bool
            Forwarded to :meth:`solve_TISE_3D_l`.
        verbose : bool
            Forwarded to :meth:`solve_TISE_3D_l`.
        """
        msgPrefix = f"[{self.__class__.__name__}.{self.solve_TISE_3D.__name__}]"
        for l in l_vals:
            if verbose:
                print(msgPrefix, f"Solving for l={l}")
            self.solve_TISE_3D_l(
                l=l,
                showPlot=showPlot,
                max_n_r=max_n_r,
                verbose=verbose,
            )
            self.l_vals.append(l)
        self.sortByEigenE()

    def getStateNames(self):
        """Return a list of state label strings (e.g. ``['1s', '2s', '2p']``)."""
        msgPrefix = f"[{self.__class__.__name__}.{self.getStateNames.__name__}]"
        return [state["name"] for state in self.states.values()]

    def getStateEnergies(self):
        """Return a list of eigen-energies in the units stored in ``self.states``."""
        msgPrefix = f"[{self.__class__.__name__}.{self.getStateEnergies.__name__}]"
        return [state["eigenE"] for state in self.states.values()]

    def getStateAmplitudeSpectrum(
        self,
        stateCoefficients: dict[str, complex],
    ) -> dict:
        """Return normalized state amplitudes and their energy shifts.

        The eigen-energy ``E_n`` is interpreted as the shift relative to an
        axion at rest:

        ``m_a,eff - m_a(v=0) = E_n / c**2``

        and

        ``nu - nu_a = E_n / h``.

        Parameters
        ----------
        stateCoefficients : dict of str to complex
            Mapping from eigenstate names to complex coefficients, for example
            ``{"2p": 1, "3p": 1 + 1j}``. The coefficients are normalized in
            the same way as in :meth:`findGradientsAtDirection`.

        Returns
        -------
        dict
            Energy-sorted state names, normalized complex coefficients,
            amplitudes, eigen-energies, effective-mass shifts, and Compton
            frequency shifts.
        """
        coefficients = self._resolveStateCoefficients(stateCoefficients)
        state_names = sorted(
            coefficients,
            key=lambda name: self.states[name]["eigenE"],
        )
        eigenE = unit.Quantity([self.states[name]["eigenE"] for name in state_names])
        normalized_coefficients = np.asarray(
            [coefficients[name] for name in state_names],
            dtype=complex,
        )

        return {
            "state_names": state_names,
            "coefficients": normalized_coefficients,
            "amplitudes": np.abs(normalized_coefficients),
            "eigenE": eigenE,
            "mass_shift": (eigenE / const.c**2).to(unit.kg),
            "frequency_shift": (eigenE / const.h).to(unit.Hz),
        }

    def plotStateAmplitudeVsEigenEnergy(
        self,
        stateCoefficients: dict[str, complex],
        energy_unit=unit.attoelectronvolt,
        frequency_unit=unit.mHz,
        ax: Axes | None = None,
        showPlot: bool = True,
    ):
        """Plot state amplitude against eigen-energy with two x-axis scales.

        The lower x axis shows the effective axion mass shift
        ``m_a,eff - m_a(v=0) = E_n / c**2``. Its numerical values are
        expressed in ``energy_unit / c**2``. The upper x axis shows the
        equivalent Compton frequency shift ``nu - nu_a = E_n / h``.

        Parameters
        ----------
        stateCoefficients : dict of str to complex
            Mapping from eigenstate names to complex coefficients. Plotted
            amplitudes are the absolute values of the normalized coefficients.
        energy_unit : astropy Unit
            Energy unit used for the lower mass-shift scale. The default gives
            an axis in ``aeV/c^2``.
        frequency_unit : astropy Unit
            Frequency unit used on the upper x axis. The default is mHz.
        ax : matplotlib.axes.Axes, optional
            Existing lower-axis object. A new figure and axis are created when
            omitted.
        showPlot : bool
            Call :func:`matplotlib.pyplot.show` when true.

        Returns
        -------
        tuple
            ``(figure, lower_axis, upper_frequency_axis, spectrum)``.
        """
        spectrum = self.getStateAmplitudeSpectrum(stateCoefficients)
        eigenE = spectrum["eigenE"].to(energy_unit)
        energy_values = eigenE.to_value(energy_unit)
        amplitudes = spectrum["amplitudes"]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8.5 / 2.54, 8.5 / 2.54 * 10 / 16), dpi=300)
        else:
            fig = ax.figure

        ax.vlines(
            energy_values,
            ymin=0,
            ymax=amplitudes,
            color="tab:blue",
            linewidth=1.5,
        )
        ax.scatter(
            energy_values,
            amplitudes,
            color="tab:blue",
            zorder=3,
        )
        for index, (energy, amplitude, state_name) in enumerate(
            zip(
                energy_values,
                amplitudes,
                spectrum["state_names"],
            )
        ):
            ax.annotate(
                state_name,
                xy=(energy, amplitude),
                xytext=(0, 5 + 12 * (index % 2)),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        energy_unit_label = energy_unit.to_string("latex_inline")[1:-1]
        ax.set_xlim(-0.05 * np.amax(energy_values), 1.05 * np.max(energy_values))
        ax.set_xlabel("$m-m_a\\,$" f"$\\left({energy_unit_label}/c^2\\right)$")
        # Leave vertical room for the staggered state labels.
        ax.set_ylim(0, 1.8 * np.max(amplitudes))
        # ax.set_ylabel("$|c_{nlm}|$")
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.set_yticks([])

        # The upper x axis represents the same E_n values as frequency shifts.
        # twiny() is used because this is a second horizontal scale, not a
        # second dependent variable.
        frequency_ax = ax.twiny()
        lower_limits = np.asarray(ax.get_xlim()) * energy_unit
        upper_limits = (lower_limits / const.h).to_value(frequency_unit)
        frequency_ax.set_xlim(upper_limits)
        frequency_unit_label = frequency_unit.to_string("latex_inline")[1:-1]
        frequency_ax.set_xlabel(
            "$\\nu-\\nu_a\\,$" f"$\\left({frequency_unit_label}\\right)$"
        )

        fig.tight_layout()
        if showPlot:
            plt.show()

        return fig, ax, frequency_ax, spectrum

    def _resolveStateCoefficients(
        self,
        stateCoefficients: dict[str, complex] | None,
    ) -> dict[str, complex]:
        """Validate and normalize an ordered eigenstate coefficient mapping."""
        if not self.states:
            raise RuntimeError(
                "No eigenstates are available. Call solve_TISE_3D first."
            )
        if stateCoefficients is None:
            raise ValueError("stateCoefficients must be provided.")
        if not isinstance(stateCoefficients, dict):
            raise TypeError("stateCoefficients must be a dictionary.")
        if not stateCoefficients:
            raise ValueError("stateCoefficients must not be empty.")

        available_states = set(self.states)
        missing_states = set(stateCoefficients) - available_states
        if missing_states:
            raise KeyError(
                f"Unknown states {sorted(missing_states)}. "
                f"Available states: {sorted(available_states)}"
            )

        resolved = {}
        for name, coefficient in stateCoefficients.items():
            if not np.isscalar(coefficient):
                raise TypeError(f"Coefficient for {name!r} must be a scalar.")
            try:
                coefficient = complex(coefficient)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Coefficient for {name!r} must be a numeric scalar."
                ) from exc
            if not np.isfinite(coefficient):
                raise ValueError(f"Coefficient for {name!r} must be finite.")
            resolved[name] = coefficient

        coefficient_norm = np.linalg.norm(list(resolved.values()))
        if coefficient_norm == 0:
            raise ValueError("At least one state coefficient must be nonzero.")
        return {
            name: coefficient / coefficient_norm
            for name, coefficient in resolved.items()
        }

    def findGradientsAtDirection(
        self,
        stateCoefficients: dict[str, complex],
        station: Station | None = None,
        meas_time: Time | None = None,
        truncRadius: Quantity | None = None,
        include_lorentz_boost: bool = True,
        relative_velocity: Quantity | None = None,
        showPlot: bool = True,
        verbose: bool = False,
    ):
        """Compute and plot the 3-D gradient of the total wavefunction at a direction.

        Superimposes the requested eigenstates with user-specified complex
        coefficients, computes the spherical-coordinate gradient
        (∂_r, ∂_θ/r, ∂_φ/(r sinθ)), interpolates each component onto a radial
        line pointing toward the requested direction, and plots the total
        wavefunction and its three gradient components.

        The direction should be specified by a :class:`~axionbloch.Station.Station`
        object.

        Parameters
        ----------
        stateCoefficients : dict of str to complex
            Mapping from eigenstate labels to coefficients ``c_nlm``, for
            example ``{"2p": 1, "3p": 1 + 1j}``. Coefficients are
            automatically normalized so that
            ``sum(abs(c_nlm)**2) = 1``.
        station : Station, optional
            Geographic station whose latitude, longitude, and elevation define
            the direction.
        truncRadius : Quantity
            Truncation radius for reducing computation and plotting time.
        include_lorentz_boost : bool
            Add the first-order laboratory-frame gradient induced by motion
            through the halo. For each mode, this contribution is
            ``-1j * (omega_n / c**2) * v_rel * psi_n``, where
            ``omega_n = (m_a * c**2 + E_n) / hbar``.
        relative_velocity : Quantity, shape (3,), optional
            Laboratory velocity relative to the halo, expressed in the
            solar-Z Cartesian frame. If omitted, use the station's velocity
            from Earth's rotation, corresponding to a nonrotating
            geocentric halo. Pass an explicit zero vector for a corotating
            halo.
        Examples
        --------
        A coherent superposition is specified by one dictionary:

        >>> from astropy.time import Time
        >>> from axionbloch.Station import Mainz
        >>> result = halo.findGradientsAtDirection(
        ...     stateCoefficients={"2p": 1, "3p": 1 + 1j},
        ...     station=Mainz,
        ...     meas_time=Time("2022-12-14T12:00:00"),
        ...     showPlot=False,
        ... )
        """

        msgPrefix = (
            f"[{self.__class__.__name__}.{self.findGradientsAtDirection.__name__}]"
        )

        # --- Resolve direction from station ---
        assert station is not None, msgPrefix + " Please provide a Station."

        if meas_time is None:
            print(
                msgPrefix, "Warning: time not provided. Using Time.now() as the input. "
            )
            meas_time = Time.now()
        # geodetic lat/lon → spherical colatitude (theta) and azimuth (phi)
        r_station, station_theta_solarZ, station_phi_solarZ = station.in_solarZ_frame(
            meas_time=meas_time
        )

        # avoid r=0 singularity
        start_index = self.N // 2 + 5
        # stop at a desired radius to make computation more efficient
        if truncRadius is None or type(truncRadius) != Quantity:
            stop_index = -1
        elif truncRadius.unit.is_equivalent(self.r.unit):
            stop_index = start_index + np.argmin(
                np.abs(self.r[start_index:] - truncRadius)
            )
        else:
            raise TypeError(
                msgPrefix + " truncRadius unit is not equivalent to length. "
            )

        if verbose:
            print(msgPrefix, "(start_index, stop_index) =", (start_index, stop_index))

        # update r and Nr
        r = self.r[start_index:stop_index]
        self.sortByEigenE()
        stateCoefficients = self._resolveStateCoefficients(
            stateCoefficients=stateCoefficients,
        )
        stateNames = list(stateCoefficients)

        Nr, Ntheta, Nphi = len(r), 100, 100
        theta_1Dgrid = np.linspace(0, PI, Ntheta)
        phi_1Dgrid = np.linspace(0, 2 * PI, Nphi)
        dr = r[1] - r[0]
        dtheta = theta_1Dgrid[1] - theta_1Dgrid[0]
        dphi = phi_1Dgrid[1] - phi_1Dgrid[0]

        # 3-D spherical mesh of shape (Nr, Ntheta, Nphi)
        R_grid, Theta_grid, Phi_grid = np.meshgrid(
            r, theta_1Dgrid, phi_1Dgrid, indexing="ij"
        )
        # grid.shape=(Nr, Ntheta, Nphi)
        # R_grid unit: [length]
        # Theta_grid / Phi_grid unit: radian

        # Accumulate the spatial, time-independent superposition on the 3-D
        # mesh. WF_total contains the full spatial dependence of all selected
        # eigenstates.
        WF_total = (
            np.zeros(R_grid.shape, dtype=complex)
            * self.states[stateNames[0]]["R_r"].unit
        )

        # WF_direction is WF_total restricted to the radial line pointing
        # toward the station:
        #
        #   WF_direction(r) =
        #       sum_j c_j R_j(r) Y_lj^mj(theta_station, phi_station).
        #
        # Thus, it is a one-dimensional complex array over r. It is neither a
        # single state's radial wavefunction R_r nor the full 3-D WF_total.
        # Time-dependent factors, including the Compton phase and
        # exp(-i E_j t / hbar), are not included. Currently m = 0 is used for
        # every selected state.
        WF_direction = np.zeros(len(r), dtype=complex) * WF_total.unit
        angular_frequency_WF_total = (
            np.zeros(R_grid.shape, dtype=complex) * WF_total.unit / unit.s
        )

        for name in stateNames:
            state = self.states[name]
            n_r, l, m = state["n_r"], state["l"], 0
            c = stateCoefficients[name]
            eigenE_expect = state["eigenE_expect"]
            # radial part broadcast over angular axes
            R_nl = state["R_r"][start_index:stop_index, None, None]
            # angular part: Y_lm(theta, phi) — note argument order for sph_harm_y
            # Y_lm = sph_harm_y(m, l, Phi_grid, Theta_grid) wrong!
            Y_lm = sph_harm_y(l, m, Theta_grid, Phi_grid)
            # wavefunction
            mode_WF = c * R_nl * Y_lm  # * np.exp(-1j * E * t)
            WF_total += mode_WF
            Y_direction = sph_harm_y(
                l,
                m,
                station_theta_solarZ.to_value(unit.rad),
                station_phi_solarZ.to_value(unit.rad),
            )
            WF_direction += c * state["R_r"][start_index:stop_index] * Y_direction
            mode_angular_frequency = (
                (self.m_a * const.c**2 + eigenE_expect) / const.hbar
            ).to(
                1 / unit.s,
                equivalencies=unit.dimensionless_angles(),
            )
            angular_frequency_WF_total += mode_angular_frequency * mode_WF

        # radial derivative ∂Ψ/∂r
        dphi_dr = np.gradient(WF_total, dr, axis=0)
        # angular derivatives
        dWF_dtheta = np.gradient(WF_total, dtheta, axis=1)
        dWF_dphi = np.gradient(WF_total, dphi, axis=2)

        # spherical-coordinate gradient components
        grad_r = dphi_dr
        grad_theta = dWF_dtheta / R_grid
        # small regularization prevents division by zero at theta=0 and π
        grad_phi = dWF_dphi / (R_grid * np.sin(Theta_grid) + 1e-12 * R_grid.unit)

        if include_lorentz_boost:
            if relative_velocity is None:
                # vx, vy, vz in solar-Z frame. z axis points along the Earth-Sun line
                # x follows Earth heliocentric velocity, orthogonalized against ẑ
                # ŷ = ẑ × x̂
                relative_velocity = station.rotation_velocity_in_solarZ_frame(meas_time)
            if not isinstance(relative_velocity, Quantity):
                raise TypeError(
                    msgPrefix
                    + " relative_velocity must be an astropy Quantity with velocity units."
                )
            if relative_velocity.shape != (
                3,
            ) or not relative_velocity.unit.is_equivalent(unit.m / unit.s):
                raise ValueError(
                    msgPrefix
                    + " relative_velocity must have shape (3,) and velocity units."
                )

            velocity = relative_velocity.to(unit.m / unit.s)
            speed = np.linalg.norm(velocity)
            beta = (speed / const.c).to(unit.one)
            if beta >= 1 * unit.one:
                raise ValueError(msgPrefix + " relative_velocity must be below c.")

            # here Theta_grid and Phi_grid are grid of the space
            theta_values = Theta_grid.to_value(unit.rad)
            phi_values = Phi_grid.to_value(unit.rad)
            sin_theta, cos_theta = np.sin(theta_values), np.cos(theta_values)
            sin_phi, cos_phi = np.sin(phi_values), np.cos(phi_values)

            velocity_r = (
                velocity[0] * sin_theta * cos_phi
                + velocity[1] * sin_theta * sin_phi
                + velocity[2] * cos_theta
            )
            velocity_theta = (
                velocity[0] * cos_theta * cos_phi
                + velocity[1] * cos_theta * sin_phi
                - velocity[2] * sin_theta
            )
            velocity_phi = -velocity[0] * sin_phi + velocity[1] * cos_phi

            # First order in v/c: the (gamma - 1) spatial correction is
            # O(v^2/c^2) and is omitted. The time-derivative term remains
            # because omega_n contains the rapid ALP Compton oscillation.
            boost_scale = -1j * angular_frequency_WF_total / const.c**2

            with unit.add_enabled_equivalencies(unit.dimensionless_angles()):
                grad_r = grad_r + boost_scale * velocity_r
                grad_theta = grad_theta + boost_scale * velocity_theta
                grad_phi = grad_phi + boost_scale * velocity_phi

            if verbose:
                print(msgPrefix, "relative velocity =", velocity)
                print(msgPrefix, "v/c = beta =", beta)

        if verbose:
            print(
                msgPrefix, "grad_r.shape =", grad_r.shape, "grad_r.unit =", grad_r.unit
            )
            print(
                msgPrefix,
                "grad_theta.shape =",
                grad_theta.shape,
                "grad_theta.unit =",
                grad_theta.unit,
            )
            print(
                msgPrefix,
                "grad_phi.shape =",
                grad_phi.shape,
                "grad_phi.unit =",
                grad_phi.unit,
            )
            # example of earth bound halo
            # grad_r.shape = (Nr, Ntheta, Nphi) grad_r.unit = 1 / (rad(1/2) [length](5/2))
            # grad_theta.shape = (Nr, Ntheta, Nphi) grad_theta.unit = 1 / (rad(3/2) earthRad(5/2))
            # grad_phi.shape = (Nr, Ntheta, Nphi) grad_phi.unit = 1 / (rad(3/2) earthRad(5/2))

        # R_grid.shape = (Nr, Ntheta, Nphi)
        # grad_r.shape = (Nr, Ntheta, Nphi)

        tic = time.time()
        interp_r = RegularGridInterpolator(
            (r.value, theta_1Dgrid.value, phi_1Dgrid.value),
            grad_r.value,
            bounds_error=False,
            fill_value=None,
        )
        interp_theta = RegularGridInterpolator(
            (r.value, theta_1Dgrid.value, phi_1Dgrid.value),
            grad_theta.value,
            bounds_error=False,
            fill_value=None,
        )
        interp_phi = RegularGridInterpolator(
            (r.value, theta_1Dgrid.value, phi_1Dgrid.value),
            grad_phi.value,
            bounds_error=False,
            fill_value=None,
        )
        toc = time.time()
        if verbose:
            print(msgPrefix, f"interpolation time: {toc-tic:.2e} s")

        # sample gradient along the radial line toward the station
        Nr_plot = 2**10
        r_line = np.linspace(r[0], truncRadius, Nr_plot)

        # points = [[r, theta_direction, phi_direction], ...]
        points = np.array(
            [
                [
                    r_pt,
                    station_theta_solarZ.to_value(unit.rad),
                    station_phi_solarZ.to_value(unit.rad),
                ]
                for r_pt in r_line.value
            ]
        )

        tic = time.time()
        grad_r_line = np.asarray(interp_r(points)) * grad_r.unit
        grad_theta_line = np.asarray(interp_theta(points)) * grad_theta.unit
        grad_phi_line = np.asarray(interp_phi(points)) * grad_phi.unit
        toc = time.time()
        if verbose:
            print(msgPrefix, f"gradient along station direction time: {toc-tic:.2e} s")
        if showPlot:
            self.plotGradients(
                station=station,
                label=station.name,
                r=r,
                R_r=WF_direction,
                r_line=r_line,
                grad_r_line=grad_r_line,
                grad_theta_line=grad_theta_line,
                grad_phi_line=grad_phi_line,
            )
        if verbose:
            earthRad_idx = np.argmin(np.abs(r_line - 1 * unit.earthRad))
            print(msgPrefix, "r_line index @ station =", earthRad_idx)
            print(msgPrefix, "grad_r @ station =", grad_r_line[earthRad_idx])
            print(msgPrefix, "grad_theta @ station =", grad_theta_line[earthRad_idx])
            print(msgPrefix, "grad_phi @ station =", grad_phi_line[earthRad_idx])
        # The second return value is the combined time-independent spatial
        # wavefunction along the station direction, not a single state's R_r.
        return (
            r,
            WF_direction,
            r_line,
            grad_r_line,
            grad_theta_line,
            grad_phi_line,
        )

    def plotGradients(
        self,
        r,
        R_r,
        r_line,
        grad_r_line,
        grad_theta_line,
        grad_phi_line,
        station: Station | None = None,
        label: str | None = None,
    ):
        """Plotting helper for :meth:`findGradients`.

        Parameters
        ----------
        station : Station, optional
            If provided, ``station.name`` is used as the plot title.
        label : str, optional
            Fallback title when ``station`` is ``None``.
        """

        fig = plt.figure(figsize=(8.5 / 2.54, 8.5 / 2.54), dpi=300)
        grid = gridspec.GridSpec(
            nrows=4,
            ncols=1,
            # width_ratios=[5, 0.1],
            # height_ratios=[4, 1],
            # hspace=0.1,
            # wspace=0.2,
        )
        left = 0.22
        bottom = 0.07
        right = 0.67
        top = 0.95
        wspace = 0.2
        hspace = 0.38
        fig.subplots_adjust(
            left=left, top=top, right=right, bottom=bottom, wspace=wspace, hspace=hspace
        )

        axion_ax = fig.add_subplot(grid[0, 0])
        grad_r_ax = fig.add_subplot(grid[1, 0], sharex=axion_ax)
        grad_theta_ax = fig.add_subplot(grid[2, 0], sharex=axion_ax)
        grad_phi_ax = fig.add_subplot(grid[3, 0], sharex=axion_ax)

        axes = [axion_ax, grad_r_ax, grad_theta_ax, grad_phi_ax]

        axion_ax.plot(
            r,
            np.real(R_r),
            label="$\\mathrm{Re}[\\Psi(r)]$",
            linestyle="--",
            zorder=4,
            linewidth=2,
        )
        axion_ax.plot(
            r,
            np.imag(R_r),
            label="$\\mathrm{Im}[\\Psi(r)]$",
            linestyle="--",
            color="tab:red",
            zorder=4,
            linewidth=2,
        )

        grad_r_ax.plot(
            r_line,
            grad_r_line.real,
            label="r gradient real",
            color=colors[1],
        )
        grad_r_ax.plot(
            r_line,
            grad_r_line.imag,
            label="r gradient imag",
        )

        grad_theta_ax.plot(
            r_line,
            grad_theta_line.real,
            label="theta gradient real",
            color=colors[2],
        )
        grad_theta_ax.plot(
            r_line,
            grad_theta_line.imag,
            label="theta gradient imag",
        )

        grad_phi_ax.plot(
            r_line,
            grad_phi_line.real,
            label="phi gradient real",
            color=colors[3],
        )
        grad_phi_ax.plot(
            r_line,
            grad_phi_line.imag,
            label="phi gradient imag",
        )

        grad_phi_ax.set_xlabel(f"$r\\,({r.unit.to_string('latex_inline')[1:-1]})$")
        # .unit.to_string("latex_inline")[1:-1] is used to remove two $ signs in the string
        ylabels = [
            # total wavefunction along the selected direction
            "$\\Psi$\n"
            + "$\\left("
            + R_r.unit.to_string("latex_inline")[1:-1]
            + "\\right)$",
            # r gradient
            "$\\partial_r\\Psi$\n"
            + "$\\left("
            + grad_r_line.unit.to_string("latex_inline")[1:-1]
            + "\\right)$",
            # theta gradient
            "$\\frac{1}{r}\\partial_\\theta \\Psi$\n"
            + "$\\left("
            + grad_theta_line.unit.to_string("latex_inline")[1:-1]
            + "\\right)$",
            # phi gradient
            "$\\frac{1}{r\\sin\\theta}\\partial_\\varphi\\Psi$\n"
            + "$\\left("
            + grad_phi_line.unit.to_string("latex_inline")[1:-1]
            + "\\right)$",
        ]
        # axion_ax.set_xlim(right=truncRadius.value)
        for i, ax in enumerate(axes):
            ax.axvline(
                x=(1 * unit.earthRad).to_value(r_line.unit),
                color="red",
                linestyle="dotted",
                # linewidth=2,
                alpha=1,
                label="Earth radius",
            )
            ax.legend(loc="upper left")
            ax.set_ylabel(ylabels[i], rotation=0, loc="center", labelpad=22)

        for ax in axes:
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        _title = station.name if station is not None else (label or "")
        fig.suptitle(f"Gradient at {_title}")
        plt.tight_layout()
        plt.show()

    def findGradientsOverTime(
        self,
        stateCoefficients: dict[str, complex],
        station: Station,
        meas_times: list[Time],
        truncRadius: Quantity | None = None,
        include_lorentz_boost: bool = True,
        relative_velocity: Quantity | None = None,
        verbose: bool = False,
    ) -> dict:
        """Gradient components at a station evaluated over a list of epochs.

        Calls :meth:`findGradientsAtDirection` for each time and collects the
        three gradient values at Earth's surface (r = 1 R_earth).

        Parameters
        ----------
        stateCoefficients : dict of str to complex
            Mapping from eigenstate names to coefficients.
        station : :class:`~axionbloch.Station.Station`
            Geographic location.
        meas_times : iterable of :class:`astropy.time.Time`
            Epochs at which to evaluate the gradients.
        truncRadius : Quantity, optional
            Radial truncation passed through to :meth:`findGradientsAtDirection`.
        include_lorentz_boost : bool
            Include the velocity-induced laboratory-frame gradient.
        relative_velocity : Quantity, shape (3,), optional
            Fixed laboratory velocity relative to the halo in the solar-Z
            frame. If omitted, determine Earth's rotation velocity separately
            at each epoch.
        verbose : bool
            Print per-step progress.

        Returns
        -------
        dict with keys:

        * ``'times'``      — input time list
        * ``'grad_r'``     — Quantity array, shape ``(N_times,)``
        * ``'grad_theta'`` — Quantity array, shape ``(N_times,)``
        * ``'grad_phi'``   — Quantity array, shape ``(N_times,)``
        """
        msgPrefix = f"[{self.__class__.__name__}.{self.findGradientsOverTime.__name__}]"

        grad_r_vals, grad_theta_vals, grad_phi_vals = [], [], []

        for i, meas_time in enumerate(meas_times):
            if verbose:
                print(msgPrefix, f"step {i}/{len(meas_times)}  t={meas_time.iso}")
            _, _, r_line, grad_r_line, grad_theta_line, grad_phi_line = (
                self.findGradientsAtDirection(
                    stateCoefficients=stateCoefficients,
                    station=station,
                    meas_time=meas_time,
                    truncRadius=truncRadius,
                    include_lorentz_boost=include_lorentz_boost,
                    relative_velocity=relative_velocity,
                    showPlot=False,
                    verbose=False,
                )
            )
            idx = np.argmin(np.abs(r_line - station.R))
            grad_r_vals.append(grad_r_line[idx])
            grad_theta_vals.append(grad_theta_line[idx])
            grad_phi_vals.append(grad_phi_line[idx])
        # if verbose:
        #     print(msgPrefix, f"grid + interpolators built in {_time.time()-tic:.2f} s")

        return {
            "times": meas_times,
            "grad_r": np.array([v.value for v in grad_r_vals]) * grad_r_vals[0].unit,
            "grad_theta": np.array([v.value for v in grad_theta_vals])
            * grad_theta_vals[0].unit,
            "grad_phi": np.array([v.value for v in grad_phi_vals])
            * grad_phi_vals[0].unit,
        }

    def findOmega_aOverTime(
        self,
        stateCoefficients: dict[str, complex],
        station: Station,
        meas_times,
        truncRadius: Quantity | None = None,
        g_aNN: Quantity[unit.GeV ** (-1)] = 1e-9 * unit.GeV ** (-1),
        include_lorentz_boost: bool = True,
        relative_velocity: Quantity | None = None,
        verbose: bool = False,
    ) -> dict:
        """Axion-induced Rabi-frequency amplitude over a list of epochs.

        Calls :meth:`findGradientsOverTime` and converts each complex
        wavefunction-gradient phasor to its real oscillation amplitude,
        ``Omega_a = factor * abs(gradient)``. Radian units are treated as
        dimensionless during conversion.

        Parameters
        ----------
        stateCoefficients : dict of str to complex
            Mapping from eigenstate names to coefficients.
        station : :class:`~axionbloch.Station.Station`
            Geographic location.
        meas_times : iterable of :class:`astropy.time.Time`
            Epochs at which to evaluate Omega_a.
        truncRadius : Quantity, optional
            Radial truncation passed through to :meth:`findGradientsOverTime`.
        verbose : bool
            Print per-step progress.

        Examples
        --------
        >>> from axionbloch.Station import Mainz
        >>> result = halo.findOmega_aOverTime(
        ...     stateCoefficients={"2p": 1, "3p": 1 + 1j},
        ...     station=Mainz,
        ...     meas_times=times,
        ... )

        Returns
        -------
        dict with keys:

        * ``'times'``      — input time list
        * ``'Omega_a_r'``     — Quantity array, shape ``(N_times,)``, Omega_a from radial gradient
        * ``'Omega_a_theta'`` — Quantity array, shape ``(N_times,)``, Omega_a from theta gradient
        * ``'Omega_a_phi'``   — Quantity array, shape ``(N_times,)``, Omega_a from phi gradient
        """
        msgPrefix = f"[{self.__class__.__name__}.{self.findOmega_aOverTime.__name__}]"

        if g_aNN is None:
            raise ValueError(
                msgPrefix
                + " g_aNN is not set. Please provide g_aNN in the method input."
            )

        gradients = self.findGradientsOverTime(
            stateCoefficients=stateCoefficients,
            station=station,
            meas_times=meas_times,
            truncRadius=truncRadius,
            include_lorentz_boost=include_lorentz_boost,
            relative_velocity=relative_velocity,
            verbose=verbose,
        )

        # Convert gradients to m^-1 (dropping radian dimension if present)
        grad_r = gradients["grad_r"]
        grad_theta = gradients["grad_theta"]
        grad_phi = gradients["grad_phi"]

        # The physical axion field gradient is the real oscillation generated
        # by this complex phasor. Its peak amplitude is the phasor magnitude,
        # retaining both cosine and sine quadratures.
        factor = (
            const.c
            * g_aNN
            * np.sqrt(self.N_a * const.hbar**3 * const.c / (2 * self.m_a))
        )
        Omega_a_r = factor * np.abs(grad_r)
        Omega_a_theta = factor * np.abs(grad_theta)
        Omega_a_phi = factor * np.abs(grad_phi)

        return {
            "times": gradients["times"],
            "Omega_a_r": Omega_a_r.to(
                unit.Hz, equivalencies=unit.dimensionless_angles()
            ),
            "Omega_a_theta": Omega_a_theta.to(
                unit.Hz, equivalencies=unit.dimensionless_angles()
            ),
            "Omega_a_phi": Omega_a_phi.to(
                unit.Hz, equivalencies=unit.dimensionless_angles()
            ),
        }

    def findrmsOmega_aOverTime(
        self,
        stateCoefficients: dict[str, complex],
        station: Station,
        meas_times,
        truncRadius: Quantity | None = None,
        g_aNN: Quantity[unit.GeV ** (-1)] = 1e-9 * unit.GeV ** (-1),
        include_lorentz_boost: bool = True,
        relative_velocity: Quantity | None = None,
        verbose: bool = False,
    ) -> dict:
        """RMS Omega_a over a list of epochs.

        Calls :meth:`findOmega_aOverTime` and computes the root-mean-square
        (RMS) value for each gradient component.

        Parameters
        ----------
        stateCoefficients : dict of str to complex
            Mapping from eigenstate names to coefficients.
        station : :class:`~axionbloch.Station.Station`
            Geographic location.
        meas_times : iterable of :class:`astropy.time.Time`
            Epochs at which to evaluate Omega_a.
        truncRadius : Quantity, optional
            Radial truncation passed through to :meth:`findOmega_aOverTime`.
        g_aNN : Quantity, optional
            Axion-nucleon coupling constant (default: 1e-9 GeV⁻¹).
        verbose : bool
            Print per-step progress.

        Returns
        -------
        dict with keys:

        * ``'rms_Omega_a_r'``     — RMS of Omega_a from radial gradient
        * ``'rms_Omega_a_theta'`` — RMS of Omega_a from theta gradient
        * ``'rms_Omega_a_phi'``   — RMS of Omega_a from phi gradient
        """
        msgPrefix = (
            f"[{self.__class__.__name__}.{self.findrmsOmega_aOverTime.__name__}]"
        )

        Omega_results = self.findOmega_aOverTime(
            stateCoefficients=stateCoefficients,
            station=station,
            meas_times=meas_times,
            truncRadius=truncRadius,
            g_aNN=g_aNN,
            include_lorentz_boost=include_lorentz_boost,
            relative_velocity=relative_velocity,
            verbose=verbose,
        )

        # Compute RMS for each component: RMS = sqrt(mean(x^2))
        Omega_a_r = Omega_results["Omega_a_r"]
        Omega_a_theta = Omega_results["Omega_a_theta"]
        Omega_a_phi = Omega_results["Omega_a_phi"]

        rms_r = np.sqrt(np.mean(Omega_a_r**2))
        rms_theta = np.sqrt(np.mean(Omega_a_theta**2))
        rms_phi = np.sqrt(np.mean(Omega_a_phi**2))

        if verbose:
            print(msgPrefix, f"RMS Omega_a_r = {rms_r}")
            print(msgPrefix, f"RMS Omega_a_theta = {rms_theta}")
            print(msgPrefix, f"RMS Omega_a_phi = {rms_phi}")

        return {
            "rms_Omega_a_r": rms_r,
            "rms_Omega_a_theta": rms_theta,
            "rms_Omega_a_phi": rms_phi,
        }

    def findGradientsOverStates(
        self,
        station: Station,
        meas_time: Time,
        stateNamesDict: dict,
        truncRadius: Quantity | None = None,
        include_lorentz_boost: bool = True,
        relative_velocity: Quantity | None = None,
        showPlot: bool = True,
        verbose: bool = False,
    ) -> dict:
        """Compute and plot gradient components for multiple state combinations.

        Calls :meth:`findGradientsAtDirection` for each state combination in
        ``stateNamesDict``, computes the three gradient components at a given
        direction, and optionally plots them in a combined figure.

        Parameters
        ----------
        station : :class:`~axionbloch.Station.Station`
            Geographic location defining the direction.
        meas_time : :class:`astropy.time.Time`
            Measurement epoch.
        stateNamesDict : dict
            Dictionary mapping comparison labels to state/coefficient
            dictionaries. For example,
            ``{"2p": {"2p": 1.0}, "mix": {"2p": 0.8, "3p": 0.6j}}``.
        truncRadius : Quantity, optional
            Radial truncation passed through to :meth:`findGradientsAtDirection`.
        showPlot : bool
            If True, display the plot of gradient components (default: True).
        verbose : bool
            Print per-step progress.

        Returns
        -------
        dict with keys:
        * ``'state_labels'`` — list of state combination labels
        * ``'grad_r'``     — dict of Quantity arrays for each state combination
        * ``'grad_theta'`` — dict of Quantity arrays for each state combination
        * ``'grad_phi'``   — dict of Quantity arrays for each state combination
        * ``'r_line'``     — common radial grid for all combinations
        """
        msgPrefix = (
            f"[{self.__class__.__name__}.{self.findGradientsOverStates.__name__}]"
        )

        results = {
            "state_labels": list(stateNamesDict.keys()),
            "grad_r": {},
            "grad_theta": {},
            "grad_phi": {},
        }

        for label, stateSelection in stateNamesDict.items():
            if verbose:
                print(msgPrefix, f"Computing gradients for: {label}")

            if not isinstance(stateSelection, dict) or not stateSelection:
                raise TypeError(
                    "Each stateNamesDict value must be a nonempty "
                    "state/coefficient dictionary."
                )
            r, R_r, r_line, grad_r_line, grad_theta_line, grad_phi_line = (
                self.findGradientsAtDirection(
                    stateCoefficients=stateSelection,
                    station=station,
                    meas_time=meas_time,
                    truncRadius=truncRadius,
                    include_lorentz_boost=include_lorentz_boost,
                    relative_velocity=relative_velocity,
                    showPlot=False,
                    verbose=verbose,
                )
            )

            results["grad_r"][label] = grad_r_line
            results["grad_theta"][label] = grad_theta_line
            results["grad_phi"][label] = grad_phi_line

        results["r_line"] = r_line

        # Plot all gradients if requested
        if showPlot:
            self._plotGradientsOverStates(
                station=station,
                stateNamesDict=stateNamesDict,
                results=results,
            )

        return results

    def compareGradientsOverStates(
        self,
        station: Station,
        meas_time: Time,
        stateNamesDict: dict,
        truncRadius: Quantity | None = None,
        showPlot: bool = True,
        verbose: bool = False,
    ) -> dict:
        """Compare gradient components across different eigenstate combinations.

        Calls :meth:`findGradientsOverStates` and extracts gradient values at
        the station's location. Plots the three gradient components as scatter
        points for each state combination.

        Parameters
        ----------
        station : :class:`~axionbloch.Station.Station`
            Geographic location defining the direction and evaluation radius.
        meas_time : :class:`astropy.time.Time`
            Measurement epoch.
        stateNamesDict : dict
            Dictionary mapping labels to state name lists. Example:
            ``{"2p": ["2p"], "2p and 3p": ["2p", "3p"]}``.
        truncRadius : Quantity, optional
            Radial truncation passed through to :meth:`findGradientsOverStates`.
        showPlot : bool
            If True, display the comparison plot (default: True).
        verbose : bool
            Print per-step progress.

        Returns
        -------
        dict with keys:
        * ``'state_labels'`` — list of state combination labels
        * ``'grad_r'``       — Quantity array of grad_r values at station radius
        * ``'grad_theta'``   — Quantity array of grad_theta values at station radius
        * ``'grad_phi'``     — Quantity array of grad_phi values at station radius
        * ``'r_eval'``       — station radius used for evaluation
        """
        msgPrefix = (
            f"[{self.__class__.__name__}.{self.compareGradientsOverStates.__name__}]"
        )

        # Get gradients over full radial range for all state combinations
        results_full = self.findGradientsOverStates(
            station=station,
            meas_time=meas_time,
            stateNamesDict=stateNamesDict,
            truncRadius=truncRadius,
            showPlot=False,
            verbose=verbose,
        )

        # Extract values at specified radius
        r_line = results_full["r_line"]

        # Evaluate at station's radius
        r_eval = (station.R).to(r_line.unit)
        eval_idx = np.argmin(np.abs(r_line - r_eval))

        state_labels = results_full["state_labels"]
        grad_r_vals = []
        grad_theta_vals = []
        grad_phi_vals = []

        for label in state_labels:
            grad_r_vals.append(results_full["grad_r"][label][eval_idx])
            grad_theta_vals.append(results_full["grad_theta"][label][eval_idx])
            grad_phi_vals.append(results_full["grad_phi"][label][eval_idx])

        comparison_results = {
            "state_labels": state_labels,
            "grad_r": np.array([v.value for v in grad_r_vals]) * grad_r_vals[0].unit,
            "grad_theta": np.array([v.value for v in grad_theta_vals])
            * grad_theta_vals[0].unit,
            "grad_phi": np.array([v.value for v in grad_phi_vals])
            * grad_phi_vals[0].unit,
            "r_eval": r_eval,
        }

        if showPlot:
            self._plotGradientsComparison(
                station=station,
                meas_time=meas_time,
                r_eval=r_eval,
                comparison_results=comparison_results,
            )

        return comparison_results

    def _plotGradientsComparison(
        self,
        station: Station,
        meas_time: Time,
        r_eval: Quantity,
        comparison_results: dict,
    ):
        """Plotting helper for :meth:`compareGradientsOverStates`.

        Creates a figure with three subplots (one for each gradient component)
        and plots scatter points for each state combination.

        Parameters
        ----------
        station : Station
            Used for the plot title.
        meas_time : :class:`astropy.time.Time`
            Measurement epoch, included in the plot title.
        r_eval : Quantity
            Evaluation radius, included in the plot title.
        comparison_results : dict
            Output from :meth:`compareGradientsOverStates`.
        """

        fig = plt.figure(figsize=(10 / 2.54, 8 / 2.54), dpi=300)
        grid = gridspec.GridSpec(
            nrows=3,
            ncols=1,
        )
        left = 0.25
        bottom = 0.1
        right = 0.85
        top = 0.93
        wspace = 0.2
        hspace = 0.45
        fig.subplots_adjust(
            left=left, top=top, right=right, bottom=bottom, wspace=wspace, hspace=hspace
        )

        grad_r_ax = fig.add_subplot(grid[0, 0])
        grad_theta_ax = fig.add_subplot(grid[1, 0])
        grad_phi_ax = fig.add_subplot(grid[2, 0])

        axes = [grad_r_ax, grad_theta_ax, grad_phi_ax]
        grad_arrays = [
            comparison_results["grad_r"],
            comparison_results["grad_theta"],
            comparison_results["grad_phi"],
        ]
        state_labels = comparison_results["state_labels"]
        x_positions = np.arange(len(state_labels))

        # First pass: plot and find ylim range
        bottom_min, top_max = 0, 0
        for i, (ax, grad_vals) in enumerate(zip(axes, grad_arrays)):
            ax.scatter(
                x_positions,
                grad_vals.real,
                s=20,
                color=colors[i % len(colors)],
                alpha=1,
                zorder=3,
            )
            ax.set_xticks(x_positions)
            if i == len(axes) - 1:
                ax.set_xticklabels(state_labels, rotation=15, ha="right")
            else:
                ax.set_xticklabels([])
            ax.grid(True, alpha=0.3, axis="y")
            bottom, top = ax.get_ylim()
            bottom_min = min(bottom_min, bottom)
            top_max = max(top_max, top)

        # Set consistent ylim for all axes (symmetric around zero)
        max_abs = max(abs(bottom_min), abs(top_max))
        ylim = (-max_abs, max_abs)
        for i, ax in enumerate(axes):
            ax.set_ylim(ylim)

            if i == len(axes) - 1:
                ax.set_xlabel("Population distribution")

            # Y-axis labels for gradients
            ylabels = [
                "$\\partial_r\\phi$",
                "$\\frac{1}{r}\\partial_\\theta \\phi$",
                "$\\frac{1}{r\\sin\\theta}\\partial_\\varphi\\phi$",
            ]
            ax.set_ylabel(
                ylabels[i] + f"\n({grad_vals.unit.to_string('latex_inline')})",
                rotation=0,
                loc="center",
                labelpad=30,
            )

        _station = station.name if station is not None else ""
        _time = meas_time.iso if meas_time is not None else ""
        r_unit = (
            r_eval.unit.to_string("latex_inline")[1:-1] if r_eval is not None else ""
        )
        r_value = r_eval.value if r_eval is not None else ""
        r_str = f"${r_value:g}\\,{r_unit}$" if r_eval is not None else "unknown"
        fig.suptitle(f"Gradient Comparison at {_station} (r = {r_str})\n{_time}")
        plt.tight_layout()
        plt.show()

    def _plotGradientsOverStates(
        self,
        station: Station,
        stateNamesDict: dict,
        results: dict,
    ):
        """Plotting helper for :meth:`findGradientsOverStates`.

        Creates a figure with three subplots (one for each gradient component)
        and overlays the gradient curves for each state combination.

        Parameters
        ----------
        station : Station
            Used for the plot title.
        stateNamesDict : dict
            Maps labels to state name lists.
        results : dict
            Output from :meth:`findGradientsOverStates`.
        """

        fig = plt.figure(figsize=(12 / 2.54, 8 / 2.54), dpi=300)
        grid = gridspec.GridSpec(
            nrows=3,
            ncols=1,
        )
        left = 0.22
        bottom = 0.1
        right = 0.67
        top = 0.93
        wspace = 0.2
        hspace = 0.4
        fig.subplots_adjust(
            left=left, top=top, right=right, bottom=bottom, wspace=wspace, hspace=hspace
        )

        grad_r_ax = fig.add_subplot(grid[0, 0])
        grad_theta_ax = fig.add_subplot(grid[1, 0], sharex=grad_r_ax)
        grad_phi_ax = fig.add_subplot(grid[2, 0], sharex=grad_r_ax)

        axes = [grad_r_ax, grad_theta_ax, grad_phi_ax]

        # Plot gradients for each state combination
        for i, (label, stateNames) in enumerate(stateNamesDict.items()):
            color = colors[i % len(colors)]

            grad_r_line = results["grad_r"][label]
            grad_theta_line = results["grad_theta"][label]
            grad_phi_line = results["grad_phi"][label]
            r_line = results["r_line"]

            grad_r_ax.plot(
                r_line,
                grad_r_line.real,
                label=f"{label}",
                color=color,
                linestyle="-",
            )

            grad_theta_ax.plot(
                r_line,
                grad_theta_line.real,
                label=f"{label}",
                color=color,
                linestyle="-",
            )

            grad_phi_ax.plot(
                r_line,
                grad_phi_line.real,
                label=f"{label}",
                color=color,
                linestyle="-",
            )

        grad_phi_ax.set_xlabel(f"$r\\,({r_line.unit.to_string('latex_inline')[1:-1]})$")

        ylabels = [
            "$\\partial_r\\phi$\n"
            + "$\\left("
            + results["grad_r"][list(stateNamesDict.keys())[0]].unit.to_string(
                "latex_inline"
            )[1:-1]
            + "\\right)$",
            "$\\frac{1}{r}\\partial_\\theta \\phi$\n"
            + "$\\left("
            + results["grad_theta"][list(stateNamesDict.keys())[0]].unit.to_string(
                "latex_inline"
            )[1:-1]
            + "\\right)$",
            "$\\frac{1}{r\\sin\\theta}\\partial_\\varphi\\phi$\n"
            + "$\\left("
            + results["grad_phi"][list(stateNamesDict.keys())[0]].unit.to_string(
                "latex_inline"
            )[1:-1]
            + "\\right)$",
        ]

        for i, ax in enumerate(axes):
            ax.axvline(
                x=(1 * unit.earthRad).to_value(r_line.unit),
                color="red",
                linestyle="dotted",
                alpha=1,
                label="Earth radius" if i == len(axes) - 1 else "",
            )
            ax.set_ylabel(ylabels[i], rotation=0, loc="center", labelpad=22)

        # Add single legend to the bottom plot only
        grad_phi_ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)

        _title = station.name if station is not None else ""
        fig.suptitle(f"Gradients at {_title}")
        plt.tight_layout()
        plt.show()

    def plotEigenstate(
        self,
        n_r: int,
        l: int,
    ):
        """Plot the reduced radial wavefunction for a single eigenstate.

        If the requested state has not been solved yet, calls
        :meth:`solve_TISE_3D_l` automatically.

        Parameters
        ----------
        n_r : int
            Radial quantum number (number of radial nodes).
        l : int
            Orbital angular-momentum quantum number.
        """
        msgPrefix = f"[{self.__class__.__name__}.{self.plotEigenstate.__name__}]"
        n = n_r + l + 1
        name = f"{n}{self.orbitalLabels[l]}"
        # key = f"{n_r}_{l}"
        if name not in self.states:
            self.solve_TISE_3D_l(
                l=l,
                showPlot=False,
                max_n_r=n_r + 1,
                verbose=False,
            )

        eigenstate = self.states[name]
        u_r = eigenstate["u_r"]
        R_reduced = eigenstate["R_reduced"]

        start_index = self.N // 2 + 5  # avoid r=0 singularity

        plt.rc("font", size=14)  # Default text
        plt.rc("figure", titlesize=14)  # Figure title
        fig = plt.figure(figsize=(6.0, 4.0), dpi=150)

        gs = gridspec.GridSpec(nrows=1, ncols=1)

        ax00 = fig.add_subplot(gs[0, 0])

        ax00.plot(
            self.r[start_index:],
            R_reduced[start_index:].real,
            label="real",
            # color="tab:blue",
            alpha=1,
            linestyle="-",
        )
        ax00.plot(
            self.r[start_index:],
            R_reduced[start_index:].imag,
            label="imaginary",
            # color="tab:blue",
            alpha=1,
            linestyle="-",
        )
        ax00.axvline(
            x=1,
            color="blue",
            linestyle="dotted",
            # linewidth=1,
            alpha=0.8,
            label="Earth radius",
        )
        ax00.legend()
        ax00.set_xlabel("r (earth radius)")

        ax00.set_xlim(-0.3, 10.3)
        fig.suptitle(
            f"Axion Compton frequency {self.nu_a.to_value(unit.Hz):.3e} Hz\nReduced radial wavefunction (n_r={n_r}, l={l})"
        )
        fig.tight_layout()
        plt.show()

    def stackEigenStates(
        self,
        numStates: int = 8,
        startState: int = 0,
        xlim=(-0.3, 5.3),
        ylim=None,
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
        msgPrefix = f"[{self.__class__.__name__}.{self.stackEigenStates.__name__}]"
        self.sortByEigenE()

        mass_eV_c2 = 4.135667696e-09  # axion mass in eV/c^2
        c_m_s = 2.99792458e8  # speed of light in m/s

        plt.rc("font", size=14)  # Default text
        plt.rc("figure", titlesize=14)  # Figure title
        fig = plt.figure(figsize=(5.3, 9.0), dpi=150)
        gs = gridspec.GridSpec(
            nrows=numStates, ncols=1
        )  # create grid for multiple figures
        left = 0.04
        bottom = 0.11
        right = 0.571
        top = 0.95
        wspace = 0.2
        hspace = 0.967
        fig.subplots_adjust(
            left=left, top=top, right=right, bottom=bottom, wspace=wspace, hspace=hspace
        )

        # # Print header
        # print("")
        # print(
        #     f"{'n_r':<6} {'l':<4} {'Principal n':<14} {'Name':<6} {'Eigen E (eV)':<15}{'Kinetic T (eV)':<15} {'Mean v (m/s)':<15}"
        # )
        # print("-" * 65)
        start_index = self.N // 2 + 5  # avoid r=0 singularity
        axes = []
        i = 0
        for key, eigenstate in list(self.states.items())[
            startState : startState + numStates
        ]:
            n_r = eigenstate["n_r_l"][0]
            l_val = eigenstate["n_r_l"][1]
            principal_n = n_r + l_val + 1
            name = eigenstate["name"]
            T_expect = eigenstate["T_expect"]
            v_m_s_mean = c_m_s * np.sqrt(2 * T_expect / mass_eV_c2)
            # print(
            #     f"{n_r:<6} {l_val:<4} {principal_n:<14} {name:<6} {eigenstate['eigenE']:1.3e} {T_expect:15.3e} {v_m_s_mean:15.3e}"
            # )
            ax = fig.add_subplot(gs[numStates - i - 1, 0])
            ax.plot(
                self.r[start_index:],
                eigenstate["R_reduced"][start_index:].real,
                label="real",
                alpha=1,
                linestyle="-",
            )
            ax.plot(
                self.r[start_index:],
                eigenstate["R_reduced"][start_index:].imag,
                label="imaginary",
                alpha=1,
                linestyle="-",
            )
            ax.axvline(
                x=1,
                color="blue",
                linestyle="dotted",
                alpha=0.8,
                label="Earth radius",
            )
            if xlim is not None:
                ax.set_xlim(xlim)
            # ax.set_ylabel("Reduced R(r)")
            # ax.legend()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{name} (n_r={n_r}, l={l_val})")

            axes.append(ax)
            i += 1

        if ylim is None:
            all_values = np.concatenate(
                [eigenstate["R_reduced"] for eigenstate in self.states.values()]
            )
            ylim = (
                -1.2 * np.amax(np.abs(all_values)),
                1.2 * np.amax(np.abs(all_values)),
            )
        for ax in axes:
            ax.set_ylim(ylim)

        axes[0].set_xlabel(
            "r (earth radius)" + f"\n$\\nu_a$={self.nu_a.to_value(unit.Hz):.0e} Hz"
        )
        axes[0].set_xticks([0, 1, 2, 3, 4, 5])
        axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        # plt.tight_layout()
        plt.show()

    def plotEigenStates(
        self,
        numStates: int = 8,
        startState: int = 0,
        truncRadius: Quantity | None = None,
        xlim=(-0.3, 5.3),
        ylim=None,
        showPlot=True,
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
        fig = plt.figure(figsize=(8.5 / 2.54, 5.5 / 2.54), dpi=300)
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.22, bottom=0.14, right=0.67, top=0.95)
        self._plotEigenStates(
            ax=ax,
            startIdx=startIdx,
            stopIdx=stopIdx,
            numStates=numStates,
            startState=startState,
        )
        plt.tight_layout()

        if showPlot:
            plt.show()

    def _plotEigenStates(self, ax: Axes, startIdx, stopIdx, numStates, startState):
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

        for key, eigenstate in list(self.states.items())[
            startState : startState + numStates
        ]:
            n_r, l_val = eigenstate["n_r_l"]
            name = eigenstate["name"]
            r_plot = self.r[startIdx:stopIdx]
            u_r = eigenstate["u_r"][startIdx:stopIdx]
            ax.plot(r_plot, u_r.real, linestyle="-", label=f"{name}")

        # if ylim is None:
        #     sel = list(self.states.items())[startState : startState + numStates]
        #     all_values = np.concatenate(
        #         [es["u_r"][start_index:stop_index] for _, es in sel]
        #     )
        #     peak = 1.2 * np.amax(np.abs(all_values))
        #     ylim = (-peak, peak)

        # if xlim is not None:
        #     ax.set_xlim(xlim)

        xUnit = self.r.unit.to_string("latex_inline")[1:-1]
        yUnit = eigenstate["u_r"].unit.to_string("latex_inline")[1:-1]
        ax.set_xlabel(f"$r\\,({xUnit})$")
        ax.set_ylabel(
            f"$r R_{{nl}}$\n$\\left({yUnit}\\right)$",
            rotation=0,
            loc="center",
            labelpad=15,
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))

    def plotEigenEnergiesInPot(
        self,
    ):
        """Plot eigen-energies superimposed on the gravitational potential.

        Draws a horizontal line for each bound state at its eigen-energy,
        spanning the classical turning points where the potential crosses that
        energy level.
        """
        msgPrefix = (
            f"[{self.__class__.__name__}.{self.plotEigenEnergiesInPot.__name__}]"
        )
        self.sortByEigenE()

        start_index = 0  # start from r>0

        fig = plt.figure(figsize=(6.0, 4.0), dpi=150)
        gs = gridspec.GridSpec(nrows=1, ncols=1)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(
            self.r[start_index:],
            self.pot[start_index:],
            label="Grav. Potential",
            alpha=1,
            linestyle="-",
        )

        for key, eigenstate in self.states.items():
            # find the classical turning point (where V(r) ≈ E)
            cross_x_indx = np.argmin(
                np.abs(self.pot[start_index:] - eigenstate["eigenE"])
            )
            xmax = self.extent / 2 - np.abs(self.r[cross_x_indx])
            xmax = np.abs(self.r[cross_x_indx])
            ax.hlines(
                y=eigenstate["eigenE"],
                xmin=-xmax,
                xmax=xmax,
                colors="k",
                alpha=0.5,
            )

        ax.set_xlabel("r (earth radius)")
        ax.set_ylabel("Energy (eV)")
        ax.set_xlim(-5.2, 5.2)
        ax.legend()
        fig.suptitle(f"Eigen-energies ($\\nu_a$={self.nu_a.to_value(unit.Hz):.0e} Hz)")
        # fig.suptitle(f"Eigen-energies")
        plt.show()

    def sortByEigenE(self):
        """Sort ``self.states`` in-place by ascending eigen-energy."""
        msgPrefix = f"[{self.__class__.__name__}.{self.sortByEigenE.__name__}]"
        self.states = dict(
            sorted(self.states.items(), key=lambda item: item[1]["eigenE"])
        )

    def findHighProbStates(
        self,
        radius_range=[
            0.9 * unit.R_earth,
            1.1 * unit.R_earth,
        ],
        threshold=1e-2,
    ):
        """Print and plot eigenstates with significant probability in a radial shell.

        Iterates over all solved states, computes the integrated probability
        inside ``radius_range``, and reports the norm and partial integral for
        each state.

        Parameters
        ----------
        radius_range : list of Quantity
            ``[r_min, r_max]`` defining the radial shell of interest.
        threshold : float
            Minimum probability fraction to flag a state (currently unused,
            reserved for future filtering).
        """
        msgPrefix = f"[{self.__class__.__name__}.{self.findHighProbStates.__name__}]"
        # find eigen-states which has high probability around earth radius
        self.sortByEigenE()
        states = []
        radius_start = radius_range[0].to_value(unit.a0)
        radius_stop = radius_range[1].to_value(unit.a0)

        if max(radius_start, radius_stop) > np.amax(self.r) or min(
            radius_start, radius_stop
        ) < np.amin(self.r):
            raise ValueError("Radius range too large. ")

        start_indx = np.argmin(np.abs(self.r - radius_start))
        stop_indx = np.argmin(np.abs(self.r - radius_stop))

        print("")
        for key, eigenstate in self.states.items():

            n_r = eigenstate["n_r_l"][0]
            l_val = eigenstate["n_r_l"][1]

            R_reduced = eigenstate["R_reduced"]
            norm = np.trapezoid(np.abs(R_reduced) ** 2, self.r)
            integral = np.trapezoid(
                np.abs(R_reduced[start_indx:stop_indx]) ** 2,
                self.r[start_indx:stop_indx],
            )
            # print("key =", key, "; n_r and l are:", n_r, l_val)
            print(msgPrefix, f"(n_r, l) = ({n_r}, {l_val})")
            print(f"eigen-energy = {eigenstate['eigenE']:.3e} eV")
            print("norm =", norm)
            print("integral =", integral)
            print("")
            self.plotEigenstate(n_r=n_r, l_val=l_val)

    def listEigenStates(
        self,
    ):
        """Print a formatted table of all solved eigenstates.

        Columns: radial quantum number, *l*, principal quantum number, label,
        eigen-energy (eV), kinetic energy (eV), and mean axion speed (m/s).
        States are listed in ascending eigen-energy order.
        """
        msgPrefix = f"[{self.__class__.__name__}.{self.listEigenStates.__name__}]"
        # find eigen-states which has high probability around earth radius
        self.sortByEigenE()

        mass_eV_c2 = 4.135667696e-09  # axion mass in eV/c^2
        c_m_s = 2.99792458e8  # speed of light in m/s

        # Print header
        print("")
        print(
            f"{'n_r':<6} {'l':<4} {'Principal n':<14} {'Name':<6} {'Eigen E (eV)':<15}{'Kinetic T (eV)':<15} {'Mean v (m/s)':<15}"
        )
        print("-" * 65)

        for key, eigenstate in self.states.items():

            n_r = eigenstate["n_r_l"][0]
            l_val = eigenstate["n_r_l"][1]
            principal_n = n_r + l_val + 1
            name = eigenstate["name"]
            T_expect = eigenstate["T_expect"]
            v_m_s_mean = c_m_s * np.sqrt(2 * T_expect / mass_eV_c2)
            print(
                f"{n_r:<6} {l_val:<4} {principal_n:<14} {name:<6} {eigenstate['eigenE']:1.3e} {T_expect:15.3e} {v_m_s_mean:15.3e}"
            )
