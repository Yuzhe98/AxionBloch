"""Plot free-induction decay from field inhomogeneity and packet convergence.

The initial magnetization is along +y and intrinsic relaxation is negligible.
An ideal Lorentzian distribution predicts exponential free-induction decay,
but the magnet packet weights include an edge window. The figure compares
simulations with the prediction from the actual high-resolution weighted
field distribution.

Run from the repository root with::

    python examples/convergency/calibration-inhomogeneity.py
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
FWHM = 1 * ppm
N_FWHM = 200.0
T1 = 1.0e6 * unit.s
T2 = 1.0e6 * unit.s
RATE = 10000.0 * unit.Hz
NUM_PTS = (200, 1000, 10000, 100_000)
PLOT_NUM_PTS = (200, 1000)
REFERENCE_NUM_PTS = NUM_PTS[-1]  # Use the highest packet count as the reference.

FWHM_FREQ = (RCF_FREQ * FWHM).to(unit.Hz)
TDELTA = (1.0 / (np.pi * FWHM_FREQ)).to(unit.s)
DURATION = 5.0 * TDELTA

# Keep figure labels and output settings together for easy editing.
MAGNETIZATION_Y_LABEL = r"$M \, (M_{\mathrm{eqb}})$"
TIME_X_LABEL = "Time (s)"
RESIDUAL_Y_LABEL = "Residual"
NUM_PT_X_LABEL = r"$\mathtt{Magnet.numPt}$"
CONVERGENCE_Y_LABEL = r"$\epsilon_{L^2}$"


def make_sample() -> Sample:
    """Return the common sample definition used for all packet counts."""
    return Sample(
        name="inhomogeneity_relaxation",
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


def make_magnet(sample: Sample, num_pts: int) -> Magnet:
    """Return the discretized, windowed Lorentzian field distribution."""
    return Magnet(
        name=f"inhomogeneity_numPt_{num_pts}",
        B0=RCF_FREQ / (sample.gamma / (2 * PI)),
        FWHM=FWHM,
        numPt=num_pts,
        nFWHM=N_FWHM,
    )


def weighted_distribution_prediction(magnet: Magnet, time_s: np.ndarray) -> np.ndarray:
    """Predict My using the packet frequencies and normalized weights."""
    gamma_over_2pi = (gamma_p / (2 * PI)).to_value(unit.Hz / unit.T)
    detuning_hz = gamma_over_2pi * magnet.B_spread.to_value(unit.T) - RCF_FREQ.to_value(
        unit.Hz
    )
    weights = np.asarray(magnet.ratios, dtype=float)
    phase = 2.0 * np.pi * detuning_hz[:, None] * time_s[None, :]
    # Initial My is real; precession about z makes My the cosine transform.
    signal = np.sum(weights[:, None] * np.cos(phase), axis=0)
    return signal * np.exp(-time_s / T2.to_value(unit.s))


def simulate_free_induction_decay(
    num_pts: int,
) -> tuple[np.ndarray, np.ndarray, Magnet]:
    """Return time, simulated normalized My, and the actual packet magnet."""
    sample = make_sample()
    magnet = make_magnet(sample, num_pts)
    simulation = Simulation(
        name=f"inhomogeneity_numPt_{num_pts}",
        sample=sample,
        magnet=magnet,
        excField=MagField(name="zero_field"),
        RCF_freq=RCF_FREQ,
        rate=RATE,
        duration=DURATION,
        init_M=1.0 * unit.one,
        init_M_theta=90.0 * unit.deg,
        init_M_phi=90.0 * unit.deg,
        verbose=False,
    )
    # Restore the requested quadrature after Simulation initialization so
    # numPt is the only changed inhomogeneity-convergence parameter.
    simulation.magnet.setHomogeneity(numPt=num_pts)
    simulation.excField.B_vec = np.zeros((simulation.timeLen - 1, 3)) * unit.T
    simulation.generateTrajectories(integrator="RK4")

    time_s = simulation.getTimeStamp().to_value(unit.s)
    my_simulated = simulation.trjry[0, :, 1]
    return time_s, my_simulated, simulation.magnet


def relative_l2_error(simulated: np.ndarray, theory: np.ndarray) -> float:
    """Return relative L2 error of simulated My against its reference."""
    return float(np.sqrt(np.sum((simulated - theory) ** 2) / np.sum(theory**2)))


def configure_axes(ax_signal, ax_residual, ax_convergence) -> None:
    """Apply labels, symmetric residual limits, and grids."""
    ax_signal.set_ylabel(MAGNETIZATION_Y_LABEL)
    ax_signal.grid(True, linewidth=0.45, alpha=0.35)

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
    ax_residual.set_yticklabels((rf"$-10^{{{power}}}$", "$0$", rf"$10^{{{power}}}$"))

    ax_convergence.set_xlabel(NUM_PT_X_LABEL)
    ax_convergence.set_ylabel(CONVERGENCE_Y_LABEL)
    ax_convergence.grid(True, which="both", linewidth=0.45, alpha=0.35)
    plt.setp(ax_signal.get_xticklabels(), visible=False)


def main() -> None:
    data = {num_pts: simulate_free_induction_decay(num_pts) for num_pts in NUM_PTS}
    time_s = data[NUM_PTS[0]][0]
    reference_magnet = make_magnet(make_sample(), REFERENCE_NUM_PTS)
    reference_signal = weighted_distribution_prediction(reference_magnet, time_s)
    convergence_errors = {
        num_pts: relative_l2_error(simulated, reference_signal)
        for num_pts, (_, simulated, _) in data.items()
    }

    print(
        f"FWHM={FWHM.to_value(ppm):g} ppm ({FWHM_FREQ.to_value(unit.Hz):g} Hz), "
        f"Tdelta={TDELTA.to_value(unit.s):g} s, rate={RATE.to_value(unit.Hz):g} Hz, "
        f"duration={DURATION.to_value(unit.s):g} s, nFWHM={N_FWHM:g}, "
        f"reference numPt={REFERENCE_NUM_PTS}"
    )
    print(
        "numPt | relative epsilon_L2 vs high-resolution weighted-distribution reference"
    )
    for num_pts in NUM_PTS:
        print(f"{num_pts:5d} | {convergence_errors[num_pts]:.6e}")

    # Match the paper-figure styling used by the other calibration figures.
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
    ax_signal = fig.add_subplot(gs[0, 0])
    ax_residual = fig.add_subplot(gs[1, 0], sharex=ax_signal)
    ax_convergence = fig.add_subplot(gs[:, 1])

    ax_signal.plot(
        time_s,
        reference_signal,
        color="black",
        linestyle="-",
        linewidth=linewidth,
        label=rf"$M_y^{{\mathrm{{theo}}}}$",
    )
    # ax_signal.plot(
    #     time_s,
    #     ideal_exponential,
    #     color="0.4",
    #     linestyle="--",
    #     linewidth=linewidth,
    #     label="Ideal Lorentzian exponential",
    # )
    for index, num_pts in enumerate(PLOT_NUM_PTS):
        _, simulated, _ = data[num_pts]
        color = high_contrast_extended[index + 2]
        marker = ("o", "s", "^")[index]
        stride = max(1, len(time_s) // 50)
        ax_signal.scatter(
            time_s[::stride],
            simulated[::stride],
            s=11,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=0.7,
            label=rf"$M_y^{{\mathrm{{simu}}}}$, {num_pts}",
            zorder=3,
        )
        ax_residual.scatter(
            time_s[::stride],
            (simulated - reference_signal)[::stride],
            s=11,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=0.7,
            zorder=3,
        )
    ax_signal.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax_convergence.loglog(
        NUM_PTS,
        [convergence_errors[num_pts] for num_pts in NUM_PTS],
        "s-",
        color=high_contrast_extended[4],
        linewidth=linewidth,
        markersize=3.5,
    )

    ax_signal.text(-0.18, 1.04, "(a)", transform=ax_signal.transAxes, fontsize=10)
    ax_convergence.text(
        -0.18, 1.04, "(b)", transform=ax_convergence.transAxes, fontsize=10
    )
    configure_axes(ax_signal, ax_residual, ax_convergence)
    fig.align_ylabels((ax_signal, ax_residual))

    output = Path(__file__).with_name("calibration-inhomogeneity.png")
    fig.savefig(output, dpi=300)
    print(f"Saved plot to {output}")
    pdf_output = Path(__file__).with_name("calibration-inhomogeneity.pdf")
    fig.savefig(pdf_output)
    print(f"Saved plot to {pdf_output}")
    plt.show(block=True)


if __name__ == "__main__":
    main()
