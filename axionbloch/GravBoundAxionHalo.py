"""Solver for gravitationally-bound axion halos around bodies.

:class:`GravBoundAxionHalo` solves the time-independent Schrödinger equation
(TISE) on a 1-D radial grid using a finite-difference Hamiltonian, returning
bound-state wavefunctions and energies for each orbital quantum number *l*.

:class:`EarthBoundAxionHalo` (in :mod:`axionbloch.EarthBoundAxionHalo`) is a
subclass pre-configured with the Earth's gravitational potential.
"""

from calendar import c
import time

from axionbloch.dependency import *

from scipy.sparse import diags
from scipy.linalg import eigh

# for wavefunction construction
from scipy.special import sph_harm_y
from scipy.interpolate import RegularGridInterpolator

from astropy.coordinates import EarthLocation

from axionbloch.Station import Station
from axionbloch.utils import high_contrast_extended as colors, check


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
        logPrefix = f"[{self.__class__.__name__}.{self.__init__.__name__}]"
        self.name = name
        self.nu_a = nu_a
        # axion mass derived from Compton frequency: m = h * nu / c²
        self.m_a = self.nu_a * const.h / const.c**2
        if verbose:
            print(logPrefix, "axion Compton frequency =", self.nu_a)
            print(
                logPrefix,
                "axion mass =",
                self.m_a.to(unit.kg),
                " =",
                self.m_a.to(unit.eV / const.c**2),
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
        logPrefix = f"[{self.__class__.__name__}.{self.showValueAndUnits.__name__}]"
        print(logPrefix, "self.r.mean():", self.r.mean())   # mean of radial grid
        print(logPrefix, "self.r.std():", self.r.std())     # spread of radial grid
        print(logPrefix, "self.dr:", self.dr)               # grid spacing
        print(logPrefix, "self.pot.mean():", self.pot.mean())  # mean gravitational potential energy
        print(logPrefix, "self.pot.std():", self.pot.std())    # spread of gravitational potential energy
        print(logPrefix, "self.T_magnitude:", self.T_magnitude)  # kinetic energy prefactor ℏ²/(2m dr²)
        print(logPrefix, "self.m_a:", self.m_a.si)              # axion mass

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
        logPrefix = f"[{self.__class__.__name__}.{self.solve_TISE_3D_l.__name__}]"
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
                logPrefix, f"N={self.N} l={l} Eigensolver took {toc - tic:.3f} seconds"
            )
        # ----------------- end of dimensionless computation ---------------- #
        energies = np.zeros_like(energies_pot_unit) * self.pot.unit
        for i, e in enumerate(energies_pot_unit):
            energies[i] = e * self.pot.unit

        if verbose:
            print(logPrefix, "Eigen-energies:")
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
                "T_eV": T_expect,
                "V_eV": V_expect,
                "Veff_eV": Veff_expect,
                "E_eV": (T_expect + Veff_expect),
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
        logPrefix = f"[{self.__class__.__name__}.{self.solve_TISE_3D.__name__}]"
        for l in l_vals:
            if verbose:
                print(logPrefix, f"Solving for l={l}")
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
        logPrefix = f"[{self.__class__.__name__}.{self.getStateNames.__name__}]"
        return [state["name"] for state in self.states.values()]

    def getStateEnergies(self):
        """Return a list of eigen-energies in the units stored in ``self.states``."""
        logPrefix = f"[{self.__class__.__name__}.{self.getStateEnergies.__name__}]"
        return [state["eigenE"] for state in self.states.values()]

    def findGradientsAtDirection(
        self,
        stateNames: list[str],
        station: Station | None = None,
        meas_time: Time | None = None,
        truncRadius: Quantity | None = None,
        showPlot: bool = True,
        verbose: bool = False,
    ):
        """Compute and plot the 3-D gradient of the total wavefunction at a direction.

        Superimposes the requested eigenstates (with equal weight), computes the
        spherical-coordinate gradient (∂_r, ∂_θ/r, ∂_φ/(r sinθ)), interpolates
        each component onto a radial line pointing toward the requested direction,
        and plots all four quantities (wavefunction + three gradient components).

        The direction should be specified by a :class:`~axionbloch.Station.Station`
        object.

        Parameters
        ----------
        stateNames : list of str
            Labels of eigenstates to include (e.g. ``['1s', '2p']``).
            Defaults to the lowest-energy state.
        station : Station, optional
            Geographic location; its ``location`` (lat / lon / elevation) is
            used as the direction.  Mutually exclusive with ``location``.
        truncRadius : Quantity
            Truncation radius for reducing computation and plotting time.
        """

        logPrefix = (
            f"[{self.__class__.__name__}.{self.findGradientsAtDirection.__name__}]"
        )

        # --- Resolve direction from station or EarthLocation ---
        assert station is not None, logPrefix + " Please provide a Station."

        if meas_time is None:
            print(
                logPrefix, "Warning: time not provided. Using Time.now() as the input. "
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
                logPrefix + " truncRadius unit is not equivalent to length. "
            )

        if verbose:
            print(logPrefix, "(start_index, stop_index) =", (start_index, stop_index))

        # update r and Nr
        r = self.r[start_index:stop_index]
        self.sortByEigenE()

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

        # if stateNames is not specified, default to the lowest-energy state (first in the sorted list) for gradient calculation
        if stateNames is None or len(stateNames) == 0:
            stateNames = [state["name"] for state in self.states.values()][:1]

        # Accumulate superposition of eigenstates on the 3-D mesh
        WF_total = (
            np.zeros(R_grid.shape, dtype=complex)
            * self.states[stateNames[0]]["R_r"].unit
        )

        for name in stateNames:
            state = self.states[name]
            n_r, l, m = state["n_r"], state["l"], 0
            c = 1.0  # equal weight. This can be modified to account for different distributions
            E_eV = state["E_eV"]
            # radial part broadcast over angular axes
            R_nl = state["R_r"][start_index:stop_index, None, None]
            # angular part: Y_lm(theta, phi) — note argument order for sph_harm_y
            # Y_lm = sph_harm_y(m, l, Phi_grid, Theta_grid) wrong!
            Y_lm = sph_harm_y(l, m, Theta_grid, Phi_grid)
            # wavefunction
            WF_total += c * R_nl * Y_lm  # * np.exp(-1j * E * t)

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

        if verbose:
            print(
                logPrefix, "grad_r.shape =", grad_r.shape, "grad_r.unit =", grad_r.unit
            )
            print(
                logPrefix,
                "grad_theta.shape =",
                grad_theta.shape,
                "grad_theta.unit =",
                grad_theta.unit,
            )
            print(
                logPrefix,
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
            print(logPrefix, f"interpolation time: {toc-tic:.2e} s")

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
            print(logPrefix, f"gradient along station direction time: {toc-tic:.2e} s")
        if showPlot:
            self.plotGradients(
                stateNames=stateNames,
                station=station,
                label=station.name,
                r=r,
                R_r=state["R_r"][start_index:stop_index],
                r_line=r_line,
                grad_r_line=grad_r_line,
                grad_theta_line=grad_theta_line,
                grad_phi_line=grad_phi_line,
            )
        if verbose:
            earthRad_idx = np.argmin(np.abs(r_line - 1 * unit.earthRad))
            print(logPrefix, "r_line index @ station =", earthRad_idx)
            print(logPrefix, "grad_r @ station =", grad_r_line[earthRad_idx])
            print(logPrefix, "grad_theta @ station =", grad_theta_line[earthRad_idx])
            print(logPrefix, "grad_phi @ station =", grad_phi_line[earthRad_idx])
        return (
            r,
            state["R_r"][start_index:stop_index],
            r_line,
            grad_r_line,
            grad_theta_line,
            grad_phi_line,
        )

    def plotGradients(
        self,
        stateNames: str,
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

        # for name, state in list(self.states.items())[:1]:
        for name in stateNames:
            # print(name, state["E_eV"], "eV")
            state = self.states[name]
            axion_ax.plot(
                r,
                np.real(R_r),
                label=name + " $\\mathrm{Re}[R(r)]$",
                # np.abs(state["R_r"]) ** 2,
                # label=name + " $R^{2}(r)$",
                linestyle="--",
                # color="tab:blue",
                zorder=4,
                linewidth=2,
            )
            axion_ax.plot(
                r,
                np.imag(R_r),
                label="$\\mathrm{Im}[R(r)]$",
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
            # radial wavefunction
            "$R_{nl}$\n"
            + "$\\left("
            + state["R_r"].unit.to_string("latex_inline")[1:-1]
            + "\\right)$",
            # r gradient
            "$\\partial_r\\phi$\n"
            + "$\\left("
            + grad_r_line.unit.to_string("latex_inline")[1:-1]
            + "\\right)$",
            # theta gradient
            "$\\frac{1}{r}\\partial_\\theta \\phi$\n"
            + "$\\left("
            + grad_theta_line.unit.to_string("latex_inline")[1:-1]
            + "\\right)$",
            # phi gradient
            "$\\frac{1}{r\\sin\\theta}\\partial_\\varphi\\phi$\n"
            + "$\\left("
            + grad_phi_line.unit.to_string("latex_inline")[1:-1]
            + "\\right)$",
        ]
        print(ylabels)
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
        stateNames: list[str],
        station: Station,
        meas_times,
        truncRadius: Quantity | None = None,
        verbose: bool = False,
    ) -> dict:
        """Gradient components at a station evaluated over a list of epochs.

        Calls :meth:`findGradientsAtDirection` for each time and collects the
        three gradient values at Earth's surface (r = 1 R_earth).

        Parameters
        ----------
        stateNames : list of str
            Eigenstate labels to superimpose.
        station : :class:`~axionbloch.Station.Station`
            Geographic location.
        times : iterable of :class:`astropy.time.Time`
            Epochs at which to evaluate the gradients.
        truncRadius : Quantity, optional
            Radial truncation passed through to :meth:`findGradientsAtDirection`.
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
        logPrefix = f"[{self.__class__.__name__}.{self.findGradientsOverTime.__name__}]"

        grad_r_vals, grad_theta_vals, grad_phi_vals = [], [], []

        for i, meas_time in enumerate(meas_times):
            if verbose:
                print(logPrefix, f"step {i}/{len(meas_times)}  t={meas_time.iso}")
            _, _, r_line, grad_r_line, grad_theta_line, grad_phi_line = (
                self.findGradientsAtDirection(
                    stateNames=stateNames,
                    station=station,
                    meas_time=meas_time,
                    truncRadius=truncRadius,
                    showPlot=False,
                    verbose=False,
                )
            )
            idx = np.argmin(np.abs(r_line - station.R))
            grad_r_vals.append(grad_r_line[idx])
            grad_theta_vals.append(grad_theta_line[idx])
            grad_phi_vals.append(grad_phi_line[idx])
        # if verbose:
        #     print(logPrefix, f"grid + interpolators built in {_time.time()-tic:.2f} s")

        return {
            "times": meas_times,
            "grad_r": np.array([v.value for v in grad_r_vals]) * grad_r_vals[0].unit,
            "grad_theta": np.array([v.value for v in grad_theta_vals])
            * grad_theta_vals[0].unit,
            "grad_phi": np.array([v.value for v in grad_phi_vals])
            * grad_phi_vals[0].unit,
        }

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
        logPrefix = f"[{self.__class__.__name__}.{self.plotEigenstate.__name__}]"
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
        logPrefix = f"[{self.__class__.__name__}.{self.stackEigenStates.__name__}]"
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
            T_eV = eigenstate["T_eV"]
            v_m_s_mean = c_m_s * np.sqrt(2 * T_eV / mass_eV_c2)
            # print(
            #     f"{n_r:<6} {l_val:<4} {principal_n:<14} {name:<6} {eigenstate['eigenE']:1.3e} {T_eV:15.3e} {v_m_s_mean:15.3e}"
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
        logPrefix = f"[{self.__class__.__name__}.{self._plotEigenStates.__name__}]"
        self.sortByEigenE()

        startIdx = self.N // 2 + 1  # avoid r=0 singularity
        if truncRadius is None or type(truncRadius) != Quantity:
            stopIdx = -1
        elif truncRadius.unit.is_equivalent(self.r.unit):
            stopIdx = startIdx + np.argmin(np.abs(self.r[startIdx:] - truncRadius))
        else:
            raise TypeError(
                logPrefix + " truncRadius unit is not equivalent to length. "
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
        logPrefix = (
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
        logPrefix = f"[{self.__class__.__name__}.{self.sortByEigenE.__name__}]"
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
        logPrefix = f"[{self.__class__.__name__}.{self.findHighProbStates.__name__}]"
        # find eigen-states which has high probability around earth radius
        self.sortByEigenE()
        states = []
        radius_start = radius_range[0].to_value(unit.a0)
        radius_stop = radius_range[1].to_value(unit.a0)

        if max(radius_start, radius_stop) > np.amax(self.r) or min(
            radius_start, radius_stop
        ) < np.amin(self.r):
            raise ValueError(f"Radius range too large. ")

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
            print(logPrefix, f"(n_r, l) = ({n_r}, {l_val})")
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
        logPrefix = f"[{self.__class__.__name__}.{self.listEigenStates.__name__}]"
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
            T_eV = eigenstate["T_eV"]
            v_m_s_mean = c_m_s * np.sqrt(2 * T_eV / mass_eV_c2)
            print(
                f"{n_r:<6} {l_val:<4} {principal_n:<14} {name:<6} {eigenstate['eigenE']:1.3e} {T_eV:15.3e} {v_m_s_mean:15.3e}"
            )
