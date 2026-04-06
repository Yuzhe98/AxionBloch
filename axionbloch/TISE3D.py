import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from axionbloch.enphylope import PhysicalQuantity
from axionbloch.constants import hbar, kB, c, h_Planck
from axionbloch.utils import earth_radius_m


class AxionTISE3D:
    # Note: input / output units are in SI, while internal calculations are in atomic units.
    def __init__(
        self,
        nu_a: PhysicalQuantity = None,  # axion Compton frequency
        N: int = int(2**12),
        extent: PhysicalQuantity = PhysicalQuantity(128.0 * earth_radius_m, "m"),
        verbose: bool = False,
    ):
        self.nu_a = nu_a
        # axion mass
        ma = self.nu_a * h_Planck / c**2
        if verbose:
            print("axion Compton frequency =", nu_a)
            print(
                "axion mass =",
                ma.to("kg"),
                " =",
                ma.to("eV/c**2"),
            )

        # axion mass in atomic units
        self.ma = ma.value_in("kg") * kg  # 1 MHz axion mass: 7e-45 kilogram

        self.N = N
        self.extent = extent.value_in("m") * meter
        self.dr = self.extent / self.N
        # r = np.linspace(dr * 1e-12, extent, N)  # starts at dr (not zero)
        self.r = np.linspace(
            -self.extent / 2, self.extent / 2, self.N
        )  # starts at dr (not zero)
        Phi_func = earth_grav_potential_earth_center_au()
        # TODO: Try to use the infinity as the reference point
        factor = 1e0
        self.V = self.ma * Phi_func(self.r) * factor  # gravitational potential

        self.eigenStates: dict = {}
        # usually we do not need all N eigenstates, so we store only the first max_n_r states for each l.

        # Map l to labels (s, p, d, f, ...)
        self.orbitalLabels = ["s", "p", "d", "f", "g", "h", "i", "k", "l", "m"]

    def solve_axion_TISE_3D_l(
        self,
        l_val: int = 3,  # angular momentum quantum number
        showPlot: bool = False,
        max_n_r: int = 10,  # maximum radial quantum number to plot
        verbose: bool = False,
    ):
        # Parameters
        Veff = self.V + l_val * (l_val + 1) * hbar**2 / (2 * self.ma * self.r**2)
        # Kinetic energy operator
        main = -2.0 * np.ones(self.N)
        off = 1.0 * np.ones(self.N - 1)

        lap = diags([off, main, off], [-1, 0, 1]) / self.dr**2
        T = -(hbar**2) / (2 * self.ma) * lap

        # Hamiltonian
        H = T + diags(Veff, 0)

        H_dense = H.toarray()

        # Solve eigenvalue problem
        tic = time.time()
        E, U = eigh(H_dense)
        toc = time.time()
        if verbose:
            print(f"N={self.N} l={l_val} Eigensolver took {toc - tic:.3f} seconds")

        # print("Eigen-energies in eV:")
        # # print("[")
        # for i, e in enumerate(E[0:max_n_r]):
        #     print(f"{e/eV:.6e},")
        # # print("]")

        start_index = self.N // 2 + 5  # avoid r=0 singularity
        # print("Kinetic-energies in eV:")
        # print("[")
        if l_val == 0:
            iter_range = np.arange(max_n_r)
        else:
            iter_range = np.arange(2 * max_n_r)[::2]
        for i, _n_r in enumerate(iter_range):

            u_r = U[:, _n_r]
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
                * self.V[start_index:],
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
            T_expect = -(hbar**2 / (2 * self.ma)) * np.trapezoid(
                np.conj(u_r[start_index:]) * du2_dr2[start_index:], self.r[start_index:]
            )
            R_reduced = (
                1.0
                * (U[:, _n_r]) ** 1
                / np.sqrt(np.trapezoid(np.abs(U[:, _n_r]) ** 2, self.r))
            )
            # print(
            #     f"n_r={n_r}, l={l_val}: T={T_expect/eV:.3e}, V={V_expect/eV:.3e}, Veff={Veff_expect/eV:.3e}, \
            #     E_total={(T_expect+Veff_expect)/eV:.3e}, eigen_E={E[n_r]/eV:.3e}"
            # )
            # print(f"{T_expect/eV:.6e},")
            principal_n = i + l_val + 1
            self.eigenStates[f"{i}_{l_val}"] = {
                "key_info": "",
                "name": f"{principal_n}{self.orbitalLabels[l_val]}",
                "n_r_l": (i, l_val),
                "eigenE_eV": E[_n_r] / eV,
                "T_eV": T_expect / eV,
                "V_eV": V_expect / eV,
                "Veff_eV": Veff_expect / eV,
                "E_eV": (T_expect + Veff_expect) / eV,
                "u_r": u_r,
                "R_reduced": R_reduced,
            }

        if showPlot:
            R_reduced = (
                1.0
                * (U[:, :max_n_r]) ** 1
                / np.sqrt(np.trapezoid(np.abs(U[:, :max_n_r]) ** 2, self.r, axis=0))
            )
            # R_reduced.shape = (N, max_n_r)
            slider_plot_earth(
                dataX=self.r[start_index:] / (earth_radius_m * meter),
                dataY=(R_reduced[start_index:, :]),
                title=f"Reduced radial wavefunction (l={l_val})",
                # xlabel="r (earth_radius)",
                xlim=None,
                show_real_imag=True,
            )

    def solve_axion_TISE_3D(
        self,
        l_vals=[3],  # angular momentum quantum number
        max_n_r: int = 10,  # maximum principal quantum number to plot
        showPlot: bool = False,
        verbose: bool = False,
    ):
        for l_val in l_vals:
            self.solve_axion_TISE_3D_l(
                l_val=l_val,
                showPlot=showPlot,
                max_n_r=max_n_r,
                verbose=verbose,
            )
        self.sortByEigenE()

    def plotEigenstate(
        self,
        n_r: int,
        l_val: int,
    ):
        key = f"{n_r}_{l_val}"
        if key not in self.eigenStates:
            self.solve_axion_TISE_3D_l(
                l_val=l_val,
                showPlot=False,
                max_n_r=n_r + 1,
                verbose=False,
            )

        eigenstate = self.eigenStates[key]
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
            self.r[start_index:] / (earth_radius_m * meter),
            R_reduced[start_index:].real,
            label="real",
            # color="tab:blue",
            alpha=1,
            linestyle="-",
        )
        ax00.plot(
            self.r[start_index:] / (earth_radius_m * meter),
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
            f"Axion Compton frequency {self.nu_a.value_in('Hz'):.3e} Hz\nReduced radial wavefunction (n_r={n_r}, l={l_val})"
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
        for key, eigenstate in list(self.eigenStates.items())[
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
                self.r[start_index:] / (earth_radius_m * meter),
                eigenstate["R_reduced"][start_index:].real,
                label="real",
                alpha=1,
                linestyle="-",
            )
            ax.plot(
                self.r[start_index:] / (earth_radius_m * meter),
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
                [eigenstate["R_reduced"] for eigenstate in self.eigenStates.values()]
            )
            ylim = (
                -1.2 * np.amax(np.abs(all_values)),
                1.2 * np.amax(np.abs(all_values)),
            )
        for ax in axes:
            ax.set_ylim(ylim)

        axes[0].set_xlabel(
            "r (earth radius)" + f"\n$\\nu_a$={self.nu_a.value_in('Hz'):.0e} Hz"
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
            self.r[start_index:] / (earth_radius_m * meter),
            self.V[start_index:] / eV,
            label="Grav. Potential",
            alpha=1,
            linestyle="-",
        )

        for key, eigenstate in self.eigenStates.items():
            cross_x_indx = np.argmin(
                np.abs(self.V[start_index:] / eV - eigenstate["eigenE_eV"])
            )
            xmax = self.extent / 2 / (earth_radius_m * meter) - np.abs(
                self.r[cross_x_indx]
            ) / (earth_radius_m * meter)
            xmax = np.abs(self.r[cross_x_indx]) / (earth_radius_m * meter)
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
        fig.suptitle(f"Eigen-energies ($\\nu_a$={self.nu_a.value_in('Hz'):.0e} Hz)")
        # fig.suptitle(f"Eigen-energies")
        plt.show()

    def sortByEigenE(self):
        self.eigenStates = dict(
            sorted(self.eigenStates.items(), key=lambda item: item[1]["eigenE_eV"])
        )

    def findHighProbStates(
        self,
        radius_range=[
            PhysicalQuantity(0.9 * earth_radius_m, "m"),
            PhysicalQuantity(1.1 * earth_radius_m, "m"),
        ],
        threshold=1e-2,
    ):
        # find eigen-states which has high probability around earth radius
        self.sortByEigenE()
        states = []
        radius_start = radius_range[0].value_in("m") * meter
        radius_stop = radius_range[1].value_in("m") * meter

        if max(radius_start, radius_stop) > np.amax(self.r) or min(
            radius_start, radius_stop
        ) < np.amin(self.r):
            raise ValueError(f"Radius range too large. ")

        start_indx = np.argmin(np.abs(self.r - radius_start))
        stop_indx = np.argmin(np.abs(self.r - radius_stop))

        print("")
        for key, eigenstate in self.eigenStates.items():

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
            print(f"eigen-energy = {eigenstate["eigenE_eV"]:.3e} eV")
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

        for key, eigenstate in self.eigenStates.items():

            n_r = eigenstate["n_r_l"][0]
            l_val = eigenstate["n_r_l"][1]
            principal_n = n_r + l_val + 1
            name = eigenstate["name"]
            T_eV = eigenstate["T_eV"]
            v_m_s_mean = c_m_s * np.sqrt(2 * T_eV / mass_eV_c2)
            print(
                f"{n_r:<6} {l_val:<4} {principal_n:<14} {name:<6} {eigenstate["eigenE_eV"]:1.3e} {T_eV:15.3e} {v_m_s_mean:15.3e}"
            )
