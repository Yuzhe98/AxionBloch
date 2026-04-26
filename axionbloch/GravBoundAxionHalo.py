import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from astropy import units as unit
from astropy.constants import codata2018 as const
from astropy.units import Quantity

from scipy.sparse import diags

# from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

# for wavefunction construction
from scipy.special import sph_harm_y
from scipy.interpolate import RegularGridInterpolator

from axionbloch.Station import Station
from axionbloch.constants import AtomicUnits as AU
from axionbloch.utils import high_contrast_extended as colors, check


class GravBoundAxionHalo:
    """
    A class to solve the time-independent Schrodinger equation for axions gravitationally bound to to objects like Earth or Sun.
    """

    # Note: input / output units are in SI, while internal calculations are in atomic units.

    # Map l to labels (s, p, d, f, ...)
    orbitalLabels = ["s", "p", "d", "f", "g", "h", "i", "k", "l", "m"]

    def __init__(
        self,
        name="Gravitationally Bound Axion Halo",
        nu_a: Quantity | None = None,  # axion Compton frequency
        N: int = int(2**12),
        extent: Quantity = 128.0 * unit.R_earth,
        getPot=None,
        mass_enclosed: Quantity | None = None,
        g_aNN: Quantity | None = None,
        verbose: bool = False,
    ):
        self.name = name
        self.nu_a = nu_a
        # axion mass
        ma = self.nu_a * const.h / const.c**2
        if verbose:
            print("axion Compton frequency =", self.nu_a)
            print(
                "axion mass =",
                ma.to(unit.kg),
                " =",
                ma.to(unit.eV / const.c**2),
            )

        # axion mass in atomic units
        self.ma: Quantity = ma  # 1 MHz axion mass: 7e-45 kilogram

        self.N = N
        self.extent: Quantity = extent
        self.dr: Quantity = self.extent / self.N
        # r = np.linspace(dr * 1e-12, extent, N)  # starts at dr (not zero)
        self.r: Quantity = np.linspace(
            -self.extent / 2, self.extent / 2, self.N
        )  # starts at dr (not zero)
        Phi_func, r_unit, Phi_unit = getPot()
        # TODO: Try also to use the infinity as the reference point
        # gravitational potential
        self.pot: Quantity = (
            self.ma * Phi_func((self.r / r_unit).to_value(unit.one)) * Phi_unit
        )  # convert r to meter for potential function

        self.mass_enclosed: Quantity = mass_enclosed
        self.g_aNN: Quantity = g_aNN

        self.T_magnitude: Quantity = (const.hbar**2) / (2 * self.ma) / self.dr**2

        self.states: dict = {}
        self.l_vals = []

        self.E_unit = unit.attoelectronvolt
        # usually we do not need all N eigenstates, so we store only the first max_n_r states for each l.

    def showValueAndUnits(self):
        """
        Print values and units of physical quantities. Ideally, the values should be close to 1 and units should be identical for quantities that are compared or added together in the code.
        """
        check(self.r.mean())
        check(self.r.std())
        check(self.dr)
        check(self.pot.mean())
        check(self.pot.std())
        check(self.T_magnitude)

    def solve_TISE_3D_l(
        self,
        l: int,  # angular momentum quantum number
        showPlot: bool = False,
        max_n_r: int = 10,  # maximum radial quantum number to plot
        verbose: bool = False,
    ):
        # TODO: check the units and the constants. I do not think we are safe with the equations here.
        # I believe if we set V and T units correctly, everthing should be fine then.
        # Parameters
        Veff: Quantity = self.pot + l * (l + 1) * const.hbar**2 / (2 * self.ma * self.r**2)
        Veff = Veff.to(self.pot.unit)

        # ----------------- start of dimensionless computation ---------------- #
        # Kinetic energy operator
        main = -2.0 * np.ones(self.N)
        off = 1.0 * np.ones(self.N - 1)

        lap = diags([off, main, off], [-1, 0, 1])
        T = -1 * self.T_magnitude.value * lap

        # Hamiltonian
        H = T + diags(Veff.value, 0)

        H_dense = H.toarray()

        # Solve eigenvalue problem
        tic = time.time()
        energies, states = eigh(H_dense)
        toc = time.time()
        if verbose:
            print(f"N={self.N} l={l} Eigensolver took {toc - tic:.3f} seconds")
        # ----------------- end of dimensionless computation ---------------- #
        if verbose:
            print("Eigen-energies in eV:")
            print("[")
            for i, e in enumerate(energies[0:max_n_r]):
                print(f"{e:.6e},")
            print("]")
            print(f"* {self.E_unit}")

        start_index = self.N // 2 + 5  # avoid r=0 singularity
        # print("Kinetic-energies in eV:")
        # print("[")
        
        if l == 0:
            iter_range = np.arange(max_n_r)
        else:
            iter_range = np.arange(2 * max_n_r)[::2]
        
        for i, _n_r in enumerate(iter_range):

            u_r = states[:, _n_r]
            R_r = u_r / self.r

            # Normalize
            integral = np.sqrt(
                np.trapezoid(
                    np.abs(R_r[start_index:]) ** 2 * self.r[start_index:] ** 2,
                    self.r[start_index:],
                )
            )
            R_r /= integral
            u_r /= integral

            # Potential energy (V only)
            V_expect = np.trapezoid(
                np.abs(R_r[start_index:]) ** 2
                * self.r[start_index:] ** 2
                * self.pot[start_index:],
                self.r[start_index:],
            )

            # Potential energy (V effective)
            Veff_expect = np.trapezoid(
                np.abs(R_r[start_index:]) ** 2
                * self.r[start_index:] ** 2
                * Veff[start_index:],
                self.r[start_index:],
            )

            # Kinetic energy via second derivative of u_r
            du2_dr2 = np.zeros_like(u_r)
            du2_dr2[1:-1] = (u_r[2:] - 2 * u_r[1:-1] + u_r[:-2]) / (
                self.r[1] - self.r[0]
            ) ** 2
            T_expect = -(const.hbar**2 / (2 * self.ma)) * np.trapezoid(
                np.conj(u_r[start_index:]) * du2_dr2[start_index:], self.r[start_index:]
            )
            R_reduced = (
                1.0
                * (states[:, _n_r]) ** 1
                / np.sqrt(np.trapezoid(np.abs(states[:, _n_r]) ** 2, self.r))
            )
            # print(
            #     f"n_r={n_r}, l={l_val}: T={T_expect/AU.eV:.3e}, V={V_expect/AU.eV:.3e}, Veff={Veff_expect/AU.eV:.3e}, \
            #     E_total={(T_expect+Veff_expect)/AU.eV:.3e}, eigen_E={E[n_r]/AU.eV:.3e}"
            # )
            # print(f"{T_expect/AU.eV:.6e},")
            n = i + l + 1
            self.states[f"{n}{self.orbitalLabels[l]}"] = {
                "key_info": "",
                "name": f"{n}{self.orbitalLabels[l]}",
                "n_r_l": (i, l),
                "n_r": (i),
                "l": (l),
                "eigenE_eV": energies[_n_r],
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
            # TODO: complete this or delete this
            # slider_plot_earth(
            #     dataX=self.r[start_index:] / AU.earth_radius,
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
        for l in l_vals:
            self.solve_TISE_3D_l(
                l=l,
                showPlot=showPlot,
                max_n_r=max_n_r,
                verbose=verbose,
            )
            self.l_vals.append(l)
        self.sortByEigenE()

    def getStateNames(self):
        return [state["name"] for state in self.states.values()]

    def getStateEnergies(self):
        return [state["eigenE_eV"] for state in self.states.values()]

    def findGradients(self, stateNames=[], station: Station = None):
        # avoid r=0 singularity
        start_index = self.N // 2 + 5
        stop_index = (
            start_index + 2**7
        )  # TODO: this should not be a number, but a radius range
        # update r and Nr
        r = self.r[start_index:stop_index]
        Nr = len(r)
        self.sortByEigenE()

        Nr, Ntheta, Nphi = len(r), 100, 100
        theta = np.linspace(0, np.pi, Ntheta)
        phi = np.linspace(0, 2 * np.pi, Nphi)
        dr = r[1] - r[0]
        dtheta = theta[1] - theta[0]
        dphi = phi[1] - phi[0]

        # mesh
        # tic = time.time()
        max_n_r = len(self.states.keys())
        R_grid, Theta_grid, Phi_grid = np.meshgrid(r, theta, phi, indexing="ij")
        # R_grid.shape=(Nr, Ntheta, Nphi)

        # toc = time.time()
        # print(f"mesh time: {toc-tic:.2e} s")
        WF_total = np.zeros_like(R_grid, dtype=complex)
        # chosenOnes = 3
        if stateNames is None or len(stateNames) == 0:
            stateNames = [state["name"] for state in self.states.values()][:1]
        # for name, state in list(self.eigenStates.items())[:1]:
        for name in stateNames:
            # print(name, state["E_eV"], "eV")
            state = self.states[name]
            n_r, l, m = state["n_r"], state["l"], 0
            c = 1.0
            E_eV = state["E_eV"]
            # radial part (interpolated onto grid)
            R_nl = state["R_r"][start_index:stop_index, None, None]

            # angular part
            # Y_lm = sph_harm_y(m, l, Phi_grid, Theta_grid) wrong!
            Y_lm = sph_harm_y(l, m, Theta_grid, Phi_grid)

            WF_total += c * R_nl * Y_lm  # * np.exp(-1j * E * t)

        # radial derivative
        dphi_dr = np.gradient(WF_total, dr, axis=0)

        # angular derivatives
        dWF_dtheta = np.gradient(WF_total, dtheta, axis=1)
        dWF_dphi = np.gradient(WF_total, dphi, axis=2)

        # components of gradient
        grad_r = dphi_dr
        grad_theta = dWF_dtheta / R_grid
        grad_phi = dWF_dphi / (R_grid * np.sin(Theta_grid) + 1e-12)

        # project the spherical gradient onto a specific direction (from the center of the sphere to the station)
        theta_station_rad = station.theta.to_value(unit.rad)  # polar angle
        phi_station_rad = station.phi.to_value(unit.rad)  # azimuthal angle

        # R_grid.shape = (Nr, Ntheta, Nphi)
        # grad_r.shape = same

        tic = time.time()
        interp_r = RegularGridInterpolator(
            (r, theta, phi), grad_r, bounds_error=False, fill_value=None
        )
        interp_theta = RegularGridInterpolator(
            (r, theta, phi), grad_theta, bounds_error=False, fill_value=None
        )
        interp_phi = RegularGridInterpolator(
            (r, theta, phi), grad_phi, bounds_error=False, fill_value=None
        )
        toc = time.time()
        print(f"interp time: {toc-tic:.2e} s")

        # sample points along radial line
        Nr_plot = 2**10
        r_line = np.linspace(r[0], 3 * AU.earth_radius, Nr_plot)

        # points = [[r, theta_station, phi_station], ...]
        points = np.array([[ri, theta_station_rad, phi_station_rad] for ri in r_line])

        tic = time.time()
        grad_r_line = np.asarray(interp_r(points))
        grad_theta_line = np.asarray(interp_theta(points))
        grad_phi_line = np.asarray(interp_phi(points))
        toc = time.time()
        print(f"get gradient along a certain direction time: {toc-tic:.2e} s")

        plt.rc("font", size=16)  # Default text

        fig = plt.figure(figsize=(8, 10))
        grid = gridspec.GridSpec(
            nrows=4,
            ncols=1,
            # width_ratios=[5, 0.1],
            # height_ratios=[4, 1],
            # hspace=0.1,
            # wspace=0.2,
        )
        left = 0.125
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

        grad_phi_ax.set_xlabel("r (earth radius)")

        # for name, state in list(self.states.items())[:1]:
        for name in stateNames:
            # print(name, state["E_eV"], "eV")
            state = self.states[name]
            axion_ax.plot(
                r / AU.earth_radius,
                np.real(state["R_r"][start_index:stop_index]) * (AU.earth_radius**1.5),
                label=name + " $\\mathrm{Re}[R(r)]$",
                # np.abs(state["R_r"]) ** 2,
                # label=name + " $R^{2}(r)$",
                linestyle="--",
                # color="tab:blue",
                zorder=4,
                linewidth=2,
            )
            # ax.plot(
            #     r / AU.earth_radius,
            #     np.imag(state["u_r"]),
            #     label="$\\mathrm{Im}[R(r) r]$",
            #     linestyle="--",
            #     color="tab:red",
            #     zorder=4,
            #     linewidth=2,
            # )
            # ax.plot(
            #     r,
            #     np.abs(state["u_r"]),
            #     label="$|R(r) r|$",
            #     color="k",
            #     linewidth=4,
            # )

        grad_r_ax.plot(
            r_line / AU.earth_radius,
            grad_r_line.real * (AU.earth_radius**2.5),
            label="r gradient real",
            color=colors[1],
        )
        # grad_r_ax.plot(
        #     r_line / AU.earth_radius,
        #     grad_r_line.imag,
        #     label="r gradient imag",
        # )

        grad_theta_ax.plot(
            r_line / AU.earth_radius,
            grad_theta_line.real * (AU.earth_radius**2.5),
            label="theta gradient real",
            color=colors[2],
        )
        # grad_theta_ax.plot(
        #     r_line / AU.earth_radius,
        #     grad_theta_line.imag,
        #     label="theta gradient imag",
        # )

        grad_phi_ax.plot(
            r_line / AU.earth_radius,
            grad_phi_line.real * (AU.earth_radius**2.5),
            label="phi gradient real",
            color=colors[3],
        )
        # grad_phi_ax.plot(
        #     r_line / AU.earth_radius,
        #     grad_phi_line.imag,
        #     label="phi gradient imag",
        # )

        grad_phi_ax.set_xlabel("r (earth radius)")
        ylables = [
            "",
            "$\\partial_r\\phi$",
            "$\\frac{1}{r}\\partial_\\theta \\phi$",
            "$\\frac{1}{r\\sin\\theta}\\partial_\\varphi\\phi$",
        ]  # "$a(\\mathbf{r}, t)$"

        axion_ax.set_xlim(right=3)
        for i, ax in enumerate(axes):
            ax.axvline(
                x=1,
                color="red",
                linestyle="dotted",
                linewidth=2,
                alpha=1,
                label="Earth radius",
            )
            ax.legend(loc="upper left")
            ax.set_ylabel(ylables[i])

        for ax in axes:
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        plt.tight_layout()
        plt.show()

    def plotEigenstate(
        self,
        n_r: int,
        l: int,
    ):
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

        # font size
        plt.rc("font", size=14)  # Default text
        plt.rc("figure", titlesize=14)  # Figure title
        fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure

        gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures

        ax00 = fig.add_subplot(gs[0, 0])

        ax00.plot(
            self.r[start_index:] / AU.earth_radius,
            R_reduced[start_index:].real,
            label="real",
            # color="tab:blue",
            alpha=1,
            linestyle="-",
        )
        ax00.plot(
            self.r[start_index:] / AU.earth_radius,
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
        ax00.set_xlabel("r (earth_radius)")

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
        """
        Plot eigenstate wavefunction in column. From low to high, this functions plots the eigenstates with increasing eigen-energy.
        """
        self.sortByEigenE()

        mass_eV_c2 = 4.135667696e-09  # axion mass in eV/c^2
        c_m_s = 2.99792458e8  # speed of light in m/s

        plt.rc("font", size=14)  # Default text
        plt.rc("figure", titlesize=14)  # Figure title
        fig = plt.figure(figsize=(5.3, 9.0), dpi=150)  # initialize a figure
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
            #     f"{n_r:<6} {l_val:<4} {principal_n:<14} {name:<6} {eigenstate['eigenE_eV']:1.3e} {T_eV:15.3e} {v_m_s_mean:15.3e}"
            # )
            ax = fig.add_subplot(gs[numStates - i - 1, 0])
            ax.plot(
                self.r[start_index:] / AU.earth_radius,
                eigenstate["R_reduced"][start_index:].real,
                label="real",
                alpha=1,
                linestyle="-",
            )
            ax.plot(
                self.r[start_index:] / AU.earth_radius,
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

        # fig.suptitle(f"Axion $\\nu_a$ {self.nu_a.value_in('Hz'):.0e} Hz")
        # plt.tight_layout()
        plt.show()

    def plotEigenEnergiesInPot(
        self,
    ):
        """
        Plot eigen-energies in the gravitational potential
        """
        self.sortByEigenE()

        mass_eV_c2 = 4.135667696e-09  # axion mass in eV/c^2
        c_m_s = 2.99792458e8  # speed of light in m/s

        start_index = 0  # start from r>0

        fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
        gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(
            self.r[start_index:] / AU.earth_radius,
            self.pot[start_index:],
            label="Grav. Potential",
            alpha=1,
            linestyle="-",
        )

        for key, eigenstate in self.states.items():
            cross_x_indx = np.argmin(
                np.abs(self.pot[start_index:] - eigenstate["eigenE_eV"])
            )
            xmax = (
                self.extent / 2 / AU.earth_radius
                - np.abs(self.r[cross_x_indx]) / AU.earth_radius
            )
            xmax = np.abs(self.r[cross_x_indx]) / AU.earth_radius
            ax.hlines(
                y=eigenstate["eigenE_eV"],
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
        self.states = dict(
            sorted(self.states.items(), key=lambda item: item[1]["eigenE_eV"])
        )

    def findHighProbStates(
        self,
        radius_range=[
            0.9 * unit.R_earth,
            1.1 * unit.R_earth,
        ],
        threshold=1e-2,
    ):
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
            print(f"(n_r, l) = ({n_r}, {l_val})")
            print(f"eigen-energy = {eigenstate['eigenE_eV']:.3e} eV")
            print("norm =", norm)
            print("integral =", integral)
            print("")
            self.plotEigenstate(n_r=n_r, l_val=l_val)

    def listEigenStates(
        self,
    ):
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
                f"{n_r:<6} {l_val:<4} {principal_n:<14} {name:<6} {eigenstate['eigenE_eV']:1.3e} {T_eV:15.3e} {v_m_s_mean:15.3e}"
            )
