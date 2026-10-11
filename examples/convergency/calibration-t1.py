"""Plot longitudinal T1 relaxation and simulation-rate convergence.

The normalized longitudinal magnetization is initialized to either zero or
ten times its equilibrium value and relaxes toward ``M_eqb``. The left column
shows RK4 trajectories and residuals; the right panel shows relative L2 error
versus simulation rate for both initial conditions.

Run from the repository root with::

    python examples/convergency/calibration-t1.py
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.dependency import PI, ppm, unit
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation
from axionbloch.utils import high_contrast_extended


RCF_FREQ = 1.0 * unit.MHz
T1 = 2 * unit.s
T2 = 0.5 * T1
FWHM = 0.01 * ppm
N_FWHM = 0.0
DURATION = 5 * T1
CONVERGENCE_RATES_HZ = np.geomspace(1.0e2, 1.0e7, 7)
T1_PLOT_RATES_HZ = (100.0,)
INITIAL_MZ_VALUES = (0.0, 10.0)

# Figure labels and limits are kept together for easy editing.
T1_Y_LABEL = r"$M$ ($M_{\mathrm{eqb}}$)"
TIME_X_LABEL = "Time (s)"
RESIDUAL_Y_LABEL = "Residual"
CONVERGENCE_X_LABEL = "Simulation rate (Hz)"
CONVERGENCE_Y_LABEL = r"$\epsilon_{L^2}$"


def simulate_t1_relaxation(
    rate_hz: float, initial_mz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, simulated Mz, and analytic Mz for one case."""
    sample = Sample(
        name="t1_relaxation",
        gamma=gamma_p,
        massDensity=0.789 * unit.g / unit.cm**3,
        molarMass=46.069 * unit.g / unit.mol,
        numOfSpinsPerMolecule=6 * unit.one,
        T1=T1,
        T2=T2,
        vol=1.0 * unit.cm**3,
        mu=mu_p,
        temp=300.0 * unit.K,
        verbose=False,
    )
    magnet = Magnet(
        name="t1_homogeneous_magnet",
        B0=RCF_FREQ / (sample.gamma / (2 * PI)),
        FWHM=FWHM,
        numPt=1,
        nFWHM=N_FWHM,
    )
    simulation = Simulation(
        name=f"t1_rate_{rate_hz:g}Hz_Mz0_{initial_mz:g}",
        sample=sample,
        magnet=magnet,
        excField=MagField(name="zero_field"),
        RCF_freq=RCF_FREQ,
        rate=rate_hz * unit.Hz,
        duration=DURATION,
        init_M=initial_mz * unit.one,
        init_M_theta=0.0 * unit.deg,
        init_M_phi=0.0 * unit.deg,
        verbose=False,
    )
    simulation.magnet.setHomogeneity(numPt=1)
    simulation.excField.B_vec = np.zeros((simulation.timeLen - 1, 3)) * unit.T
    simulation.generateTrajectories(integrator="RK4")

    time_s = simulation.getTimeStamp().to_value(unit.s)
    mz_simulated = simulation.trjry[0, :, 2]
    t1_s = T1.to_value(unit.s)
    mz_theory = 1.0 + (initial_mz - 1.0) * np.exp(-time_s / t1_s)
    return time_s, mz_simulated, mz_theory


def relative_l2_error(simulated: np.ndarray, theory: np.ndarray) -> float:
    """Return relative L2 error of simulated Mz against the T1 law."""
    return float(np.sqrt(np.sum((simulated - theory) ** 2) / np.sum(theory**2)))


def configure_plot_axes(ax_relaxation, ax_residual, ax_convergence) -> None:
    """Apply labels, residual scale, and grids to the figure axes."""
    ax_relaxation.set_ylabel(T1_Y_LABEL)
    ax_relaxation.grid(True, linewidth=0.45, alpha=0.35)

    ax_residual.set_xlabel(TIME_X_LABEL)
    ax_residual.set_ylabel(RESIDUAL_Y_LABEL)
    ax_residual.grid(True, linewidth=0.45, alpha=0.35)
    residual_ymin, residual_ymax = ax_residual.get_ylim()
    residual_max_abs = max(abs(residual_ymin), abs(residual_ymax))
    power = int(np.ceil(np.log10(residual_max_abs)))
    residual_tick = 10.0**power
    residual_limit = 1.2 * residual_tick
    ax_residual.set_ylim(-residual_limit, residual_limit)
    ax_residual.set_yticks((-residual_tick, 0.0, residual_tick))
    ax_residual.set_yticklabels(
        (rf"$-10^{{{power}}}$", "$0$", rf"$10^{{{power}}}$")
    )

    ax_convergence.set_xlabel(CONVERGENCE_X_LABEL)
    ax_convergence.set_ylabel(CONVERGENCE_Y_LABEL)
    ax_convergence.grid(True, which="both", linewidth=0.45, alpha=0.35)
    plt.setp(ax_relaxation.get_xticklabels(), visible=False)


def main() -> None:
    # Reuse each plotted 100 Hz trajectory in the convergence calculation,
    # avoiding separate duplicate simulations for the left-hand panels.
    trajectory_data = {
        initial_mz: simulate_t1_relaxation(T1_PLOT_RATES_HZ[0], initial_mz)
        for initial_mz in INITIAL_MZ_VALUES
    }
    convergence_errors = {initial_mz: [] for initial_mz in INITIAL_MZ_VALUES}
    for rate_hz in CONVERGENCE_RATES_HZ:
        for initial_mz in INITIAL_MZ_VALUES:
            if rate_hz == T1_PLOT_RATES_HZ[0]:
                _, mz_simulated, mz_theory = trajectory_data[initial_mz]
            else:
                _, mz_simulated, mz_theory = simulate_t1_relaxation(
                    rate_hz, initial_mz
                )
            convergence_errors[initial_mz].append(
                relative_l2_error(mz_simulated, mz_theory)
            )
    convergence_errors = {
        initial_mz: np.asarray(errors)
        for initial_mz, errors in convergence_errors.items()
    }

    print(
        f"T1={T1.to_value(unit.s):g} s, observation duration="
        f"{DURATION.to_value(unit.s):g} s, initial Mz/M_eqb={INITIAL_MZ_VALUES}"
    )
    print("rate (Hz) | epsilon_L2 (Mz0=0) | epsilon_L2 (Mz0=10)")
    for index, rate_hz in enumerate(CONVERGENCE_RATES_HZ):
        print(
            f"{rate_hz:9.1f} | {convergence_errors[0.0][index]:17.6e} "
            f"| {convergence_errors[10.0][index]:17.6e}"
        )

    # Match the paper-figure styling used by calibration-excitation.py.
    linewidth = 1.0
    plt.rc("font", size=10)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    cm = 1.0 / 2.54
    fig = plt.figure(figsize=(17.0 * cm, 7.0 * cm), dpi=300)
    gs = gridspec.GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        width_ratios=(1.0, 1.0),
        height_ratios=(1.0, 0.33),
    )
    fig.subplots_adjust(
        left=0.115, right=0.98, bottom=0.188, top=0.91, wspace=0.38, hspace=0.10
    )
    ax_relaxation = fig.add_subplot(gs[0, 0])
    ax_residual = fig.add_subplot(gs[1, 0], sharex=ax_relaxation)
    ax_convergence = fig.add_subplot(gs[:, 1])

    # Color identifies the initial condition; marker identifies simulation rate.
    initial_colors = {
        0.0: high_contrast_extended[2],
        10.0: high_contrast_extended[3],
    }
    initial_labels = {
        0.0: r"$0$",
        10.0: r"$10\,M_{\mathrm{eqb}}$",
    }
    first_time_s = trajectory_data[INITIAL_MZ_VALUES[0]][0]
    for initial_mz in INITIAL_MZ_VALUES:
        color = initial_colors[initial_mz]
        theory = 1.0 + (initial_mz - 1.0) * np.exp(
            -first_time_s / T1.to_value(unit.s)
        )
        ax_relaxation.plot(
            first_time_s,
            theory,
            color=color,
            linestyle="-",
            linewidth=linewidth,
            label=rf"$M_z^{{\mathrm{{theo}}}}$, {initial_labels[initial_mz]}",
        )

    ax_relaxation.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label=r"$M_z=M_{\mathrm{eqb}}$",
    )

    for initial_mz in INITIAL_MZ_VALUES:
        color = initial_colors[initial_mz]
        for rate_index, rate_hz in enumerate(T1_PLOT_RATES_HZ):
            time_s, mz_simulated, mz_theory = trajectory_data[initial_mz]
            marker = "o" if rate_index == 0 else "s"
            stride = max(1, len(time_s) // 50)
            ax_relaxation.scatter(
                time_s[::stride],
                mz_simulated[::stride],
                s=11,
                marker=marker,
                facecolors="none",
                edgecolors=color,
                linewidths=0.7,
                label=rf"$M_z^{{\mathrm{{simu}}}}$, {initial_labels[initial_mz]}",
                zorder=3,
            )
            ax_residual.scatter(
                time_s[::stride],
                (mz_simulated - mz_theory)[::stride],
                s=11,
                marker=marker,
                facecolors="none",
                edgecolors=color,
                linewidths=0.7,
                zorder=3,
            )

    ax_relaxation.legend(frameon=False, fontsize=6.5, loc="upper right")
    for initial_mz, color in initial_colors.items():
        ax_convergence.loglog(
            CONVERGENCE_RATES_HZ,
            convergence_errors[initial_mz],
            "s-",
            color=color,
            linewidth=linewidth,
            markersize=3.5,
            label=initial_labels[initial_mz],
        )
    ax_convergence.legend(frameon=False, fontsize=8, loc="upper right")

    ax_relaxation.text(-0.18, 1.04, "(a)", transform=ax_relaxation.transAxes, fontsize=10)
    ax_convergence.text(
        -0.18, 1.04, "(b)", transform=ax_convergence.transAxes, fontsize=10
    )
    configure_plot_axes(ax_relaxation, ax_residual, ax_convergence)
    fig.align_ylabels((ax_relaxation, ax_residual))

    output = Path(__file__).with_name("calibration-t1.png")
    fig.savefig(output, dpi=300)
    print(f"Saved plot to {output}")
    pdf_output = Path(__file__).with_name("calibration-t1.pdf")
    fig.savefig(pdf_output)
    print(f"Saved plot to {pdf_output}")
    plt.show(block=True)


if __name__ == "__main__":
    main()
