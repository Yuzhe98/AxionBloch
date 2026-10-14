"""Plot transverse T2 relaxation and simulation-rate convergence.

The normalized magnetization starts along +y with magnitude
``M_eqb * 1`` and relaxes toward zero. The left panel compares RK4
trajectories at two rates with the analytic exponential, and the right panel
shows relative L2 error versus simulation rate.

Run from the repository root with::

    python examples/convergency/calibration-t2.py
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
T1 = 1.0e6 * unit.s
T2 = 1.0 * unit.s
FWHM = 0.1 * ppm
N_FWHM = 0.0
DURATION = 5.0 * T2
CONVERGENCE_RATES_HZ = np.geomspace(100.0, 1.0e5, 7)
T2_PLOT_RATES_HZ = (100.0, 1000.0)

# Figure labels and limits are kept together for easy editing.
T2_Y_LABEL = r"$M$ ($M_{\mathrm{eqb}}$)"
TIME_X_LABEL = "Time (s)"
RESIDUAL_Y_LABEL = "Residual\n($M_{\\mathrm{eqb}}$)"
CONVERGENCE_X_LABEL = "Simulation rate (Hz)"
CONVERGENCE_Y_LABEL = r"$\epsilon_{L^2}$"


def simulate_t2_decay(rate_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, simulated My, and analytic My for one simulation rate."""
    sample = Sample(
        name="t2_relaxation",
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
        name="t2_homogeneous_magnet",
        B0=RCF_FREQ / (sample.gamma / (2 * PI)),
        FWHM=FWHM,
        numPt=1,
        nFWHM=N_FWHM,
    )
    simulation = Simulation(
        name=f"t2_rate_{rate_hz:g}Hz",
        sample=sample,
        magnet=magnet,
        excField=MagField(name="zero_field"),
        RCF_freq=RCF_FREQ,
        rate=rate_hz * unit.Hz,
        duration=DURATION,
        init_M=1.0 * unit.one,
        init_M_theta=90.0 * unit.deg,
        init_M_phi=90.0 * unit.deg,
        verbose=False,
    )
    simulation.magnet.setHomogeneity(numPt=1)
    simulation.excField.B_vec = np.zeros((simulation.timeLen - 1, 3)) * unit.T
    simulation.generateTrajectories(integrator="RK4")

    time_s = simulation.getTimeStamp().to_value(unit.s)
    my_simulated = simulation.trjry[0, :, 1]
    my_theory = np.exp(-time_s / T2.to_value(unit.s))
    return time_s, my_simulated, my_theory


def relative_l2_error(simulated: np.ndarray, theory: np.ndarray) -> float:
    """Return the relative L2 error of simulated My against the T2 law."""
    return float(np.sqrt(np.sum((simulated - theory) ** 2) / np.sum(theory**2)))


def configure_plot_axes(ax_decay, ax_residual, ax_convergence) -> None:
    """Apply axis labels, grids, and shared figure styling."""
    ax_decay.set_ylabel(T2_Y_LABEL)
    ax_decay.grid(True, linewidth=0.45, alpha=0.35)

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
    plt.setp(ax_decay.get_xticklabels(), visible=False)


def main() -> None:
    decay_data = {
        rate_hz: simulate_t2_decay(rate_hz) for rate_hz in T2_PLOT_RATES_HZ
    }
    convergence_errors = []
    for rate_hz in CONVERGENCE_RATES_HZ:
        time_s, my_simulated, my_theory = simulate_t2_decay(rate_hz)
        convergence_errors.append(relative_l2_error(my_simulated, my_theory))
    convergence_errors = np.asarray(convergence_errors)

    print(f"T2={T2.to_value(unit.ms):g} ms, observation duration={DURATION.to_value(unit.s):g} s")
    print("rate (Hz) | relative epsilon_L2")
    for rate_hz, error in zip(CONVERGENCE_RATES_HZ, convergence_errors):
        print(f"{rate_hz:9.1f} | {error:.6e}")

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
        left=0.142, right=0.98, bottom=0.188, top=0.91, wspace=0.38, hspace=0.10
    )
    ax_decay = fig.add_subplot(gs[0, 0])
    ax_residual = fig.add_subplot(gs[1, 0], sharex=ax_decay)
    ax_convergence = fig.add_subplot(gs[:, 1])

    first_time_s, _, first_theory = decay_data[T2_PLOT_RATES_HZ[0]]
    ax_decay.plot(
        first_time_s,
        first_theory,
        color="black",
        linestyle="-",
        linewidth=linewidth,
        label=r"$M_y^\mathrm{theo}$",
    )
    for index, rate_hz in enumerate(T2_PLOT_RATES_HZ):
        time_s, my_simulated, my_theory = decay_data[rate_hz]
        color = high_contrast_extended[index + 2]
        marker = "o" if index == 0 else "s"
        # Keep both rates visually identifiable, including the dense 1000 Hz
        # trajectory, while retaining uniformly spaced simulation samples.
        stride = max(1, len(time_s) // 50)
        ax_decay.scatter(
            time_s[::stride],
            my_simulated[::stride],
            s=11,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=0.7,
            label=rf"$M_y^{{\mathrm{{simu}}}}$, {rate_hz:g} Hz",
            zorder=3,
        )
        ax_residual.scatter(
            time_s[::stride],
            (my_simulated - my_theory)[::stride],
            s=11,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=0.7,
            zorder=3,
        )
    ax_decay.legend(frameon=False, fontsize=8, loc="upper right")
    ax_convergence.loglog(
        CONVERGENCE_RATES_HZ,
        convergence_errors,
        "s-",
        color=high_contrast_extended[4],
        linewidth=linewidth,
        markersize=3.5,
    )
    ax_decay.text(-0.18, 1.04, "(a)", transform=ax_decay.transAxes, fontsize=10)
    ax_convergence.text(
        -0.18, 1.04, "(b)", transform=ax_convergence.transAxes, fontsize=10
    )
    configure_plot_axes(ax_decay, ax_residual, ax_convergence)
    fig.align_ylabels((ax_decay, ax_residual))

    output = Path(__file__).with_name("calibration-t2.png")
    fig.savefig(output, dpi=300)
    print(f"Saved plot to {output}")
    pdf_output = Path(__file__).with_name("calibration-t2.pdf")
    fig.savefig(pdf_output)
    print(f"Saved plot to {pdf_output}")
    plt.show(block=True)


if __name__ == "__main__":
    main()
