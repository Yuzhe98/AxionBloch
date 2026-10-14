"""Calibrate hard-pulse amplitude from free decay and plot Rabi oscillation.

The free-decay signal calibrates the rectangular x-pulse amplitude for a
90-degree flip. The figure then shows post-pulse My versus input tip angle,
component residuals, and RK4 rate convergence against an exact pulse-aware
Bloch prediction.

Run from the repository root with::

    python examples/convergency/calibration-excitation.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.dependency import PI, ppm, unit
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation
from axionbloch.utils import high_contrast_extended


RCF_FREQ = 1.0 * unit.MHz
RATE = 1.0 * unit.kHz
PULSE_STEPS = 100
T90_DURATION = PULSE_STEPS / RATE
T1 = 1.0e6 * unit.s
T2 = 1.0e5 * unit.s
FREE_DECAY_DURATION = 0.1 * unit.s
CONVERGENCE_RATES_HZ = np.geomspace(100.0, 1.0e6, 6)
RABI_PLOT_RATES_HZ = (100.0, 1000.0)
RABI_INPUT_ANGLES_DEG = np.linspace(0.0, 360.0, 50)

# Figure axis labels, ticks, and limits are kept together for easy editing.
RABI_Y_LIMITS = (-1.08, 1.08)
TIP_ANGLE_TICKS_DEG = (0.0, 90.0, 180.0, 270.0, 360.0)
RABI_Y_LABEL = r"$M_y$ ($M_{\mathrm{eqb}}$)"
RESIDUAL_X_LABEL = r"User-specified tip angle $\theta$ (°)"
RESIDUAL_Y_LABEL = "Residual\n($M_{\\mathrm{eqb}}$)"
CONVERGENCE_X_LABEL = "Simulation rate (Hz)"
CONVERGENCE_Y_LABEL = r"$\epsilon_{L^2}$"


def simulate_free_decay(
    peak_b1_t: float, pulse_steps: int = PULSE_STEPS
) -> dict[str, np.ndarray | float]:
    """Simulate a hard x pulse and fit the following free-decay signal.

    peak_b1_t uses the library peak-amplitude convention; the x component
    during the pulse is peak_b1_t / 2. The free-decay fit extrapolates Mx+iMy
    and Mz to pulse end, then reconstructs the signed rotation angle.
    """
    sample = Sample(
        name="rabi_free_decay_calibration",
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
        name="rabi_homogeneous_magnet",
        B0=RCF_FREQ / (sample.gamma / (2 * PI)),
        # A nonzero nominal width avoids undefined linewidth conversions;
        # nFWHM=0 keeps this calibration physically homogeneous.
        FWHM=0.1 * ppm,
        numPt=1,
        nFWHM=0.0,
    )
    pulse_duration = pulse_steps / RATE
    simulation = Simulation(
        name=f"rabi_free_decay_B1_{peak_b1_t:.6e}T",
        sample=sample,
        magnet=magnet,
        excField=MagField(name="rectangular_x_pulse"),
        RCF_freq=RCF_FREQ,
        rate=RATE,
        duration=pulse_duration + FREE_DECAY_DURATION,
        verbose=False,
    )
    simulation.magnet.setHomogeneity(numPt=1)

    field = np.zeros((simulation.timeLen - 1, 3)) * unit.T
    field[:pulse_steps, 0] = (0.5 * peak_b1_t) * unit.T
    simulation.excField.B_vec = field
    simulation.generateTrajectories(integrator="RK4")

    time_s = simulation.getTimeStamp().to_value(unit.s)
    free_time_s = time_s[pulse_steps:] - time_s[pulse_steps]
    mx, my, mz = simulation.trjry[0, pulse_steps:, :3].T

    # Free decay at resonance is exponential. Least squares on the full record
    # removes single-sample noise from the inferred pulse-end state.
    decay_t2 = np.exp(-free_time_s / T2.to_value(unit.s))
    z_free = mx + 1j * my
    z0 = np.sum(decay_t2 * z_free) / np.sum(decay_t2**2)
    decay_t1 = np.exp(-free_time_s / T1.to_value(unit.s))
    mz0 = 1.0 + np.sum(decay_t1 * (mz - 1.0)) / np.sum(decay_t1**2)
    angle_rad = np.arctan2(z0.imag, mz0) % (2.0 * np.pi)

    return {
        "time_s": free_time_s,
        "mx": mx,
        "my": my,
        "mz": mz,
        "z0": z0,
        "mz0": float(mz0),
        "angle_rad": float(angle_rad),
    }


def helper_amplitude(tip_angle) -> float:
    """Get calibrated peak B1 for the configured pulse grid."""
    return float(
        MagField._calibratedHardPulseAmplitude(
            timeStep=1.0 / RATE,
            gamma=gamma_p,
            pulseSteps=PULSE_STEPS,
            tip_angle=tip_angle,
        ).to_value(unit.T)
    )


def free_decay_calibrated_amplitude(tip_angle) -> float:
    """Find peak B1 from the angle measured by post-pulse free decay."""
    target_rad = tip_angle.to_value(unit.rad)
    if not 0.0 < target_rad < np.pi:
        raise ValueError("Target angle must be between 0 and 180 degrees")

    pulse_s = (PULSE_STEPS / RATE).to_value(unit.s)
    continuous_estimate = 2.0 * target_rad / (
        gamma_p.to_value(unit.rad * unit.Hz / unit.T) * pulse_s
    )

    def residual(b1_t: float) -> float:
        measured = simulate_free_decay(b1_t)["angle_rad"]
        return measured - target_rad

    return float(brentq(residual, 0.0, 2.0 * continuous_estimate, xtol=1e-14))


def simulate_pulse_at_tip_angle(
    rate_hz: float, tip_angle_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return simulated and pulse-aware theory magnetization for one angle.

    All input angles use the same 100 ms duration. ``setXYPulse_new`` computes
    the discrete-RK4-calibrated amplitude independently for each requested
    angle at this simulation rate and pulse step count.
    """
    pulse_steps = int(round(T90_DURATION.to_value(unit.s) * rate_hz))
    dt_s = 1.0 / rate_hz
    rate = rate_hz * unit.Hz
    pulse_duration_s = pulse_steps * dt_s
    # Simulation requires at least five integration steps. For ordinary pulse
    # grids, retain only the n pulse intervals: state index n is the pulse-end
    # magnetization, so no free-decay record is needed here.
    total_steps = max(pulse_steps, 5)
    sample = Sample(
        name="rabi_rate_convergence",
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
        name="rabi_rate_convergence_magnet",
        B0=RCF_FREQ / (sample.gamma / (2 * PI)),
        FWHM=0.1 * ppm,
        numPt=1,
        nFWHM=0.0,
    )
    simulation = Simulation(
        name=f"rabi_rate_{rate_hz:g}Hz",
        sample=sample,
        magnet=magnet,
        excField=MagField(name="constant_x_pulse"),
        RCF_freq=RCF_FREQ,
        rate=rate,
        # Simulation uses ceil(duration * rate) - 1 integration intervals.
        # One extra boundary point gives exactly total_steps intervals.
        duration=(total_steps + 1) * dt_s * unit.s,
        verbose=False,
    )
    simulation.magnet.setHomogeneity(numPt=1)
    simulation.excField.setXYPulse_new(
        timeStep=dt_s * unit.s,
        timeLen=simulation.timeLen,
        gamma=gamma_p,
        pulseSteps=pulse_steps,
        nu_rot=0.0 * unit.Hz,
        tip_angle=tip_angle_deg * unit.deg,
    )
    simulation.generateTrajectories(integrator="RK4")

    state = simulation.trjry[0, pulse_steps, :3]

    # Reference the requested angle in continuous time, while retaining the
    # finite-pulse T1/T2 relaxation. Using the RK4-corrected B1 here would
    # incorrectly treat a discrete-integration correction as physical physics
    # and shift the theoretical angle, especially at low rates and high tips.
    r1 = 1.0 / T1.to_value(unit.s)
    r2 = 1.0 / T2.to_value(unit.s)
    target_angle_rad = np.deg2rad(tip_angle_deg)
    omega_x = target_angle_rad / pulse_duration_s
    generator = np.array(
        [
            [-r2, 0.0, 0.0, 0.0],
            [0.0, -r2, omega_x, 0.0],
            [0.0, -omega_x, -r1, r1],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    initial = np.array([0.0, 0.0, 1.0, 1.0])
    theory = (expm(generator * pulse_duration_s) @ initial)[:3]
    return state, theory


def rabi_rate_convergence() -> np.ndarray:
    """Return normalized component L2 errors over the common angle sweep."""
    errors = []
    for rate_hz in CONVERGENCE_RATES_HZ:
        states_and_theory = [
            simulate_pulse_at_tip_angle(rate_hz, angle_deg)
            for angle_deg in RABI_INPUT_ANGLES_DEG
        ]
        states = np.asarray([item[0] for item in states_and_theory])
        theory = np.asarray([item[1] for item in states_and_theory])
        delta = states[:, :2] - theory[:, :2]
        # Mx's ideal reference is zero, so normalize both component errors by
        # the nonzero My trajectory norm rather than dividing by sum(Mx_theory²).
        reference_norm = np.sum(theory[:, 1] ** 2)
        epsilon_mx = np.sqrt(np.sum(delta[:, 0] ** 2) / reference_norm)
        epsilon_my = np.sqrt(np.sum(delta[:, 1] ** 2) / reference_norm)
        errors.append((epsilon_mx, epsilon_my, np.hypot(epsilon_mx, epsilon_my)))
    return np.asarray(errors)


def configure_plot_axes(ax_rabi, ax_residual, ax_convergence) -> None:
    """Apply axis labels, ticks, limits, and grids in one place."""
    ax_rabi.set_ylabel(RABI_Y_LABEL)
    ax_rabi.set_ylim(*RABI_Y_LIMITS)
    ax_rabi.grid(True, linewidth=0.45, alpha=0.35)

    ax_residual.set_xlabel(RESIDUAL_X_LABEL)
    ax_residual.set_ylabel(RESIDUAL_Y_LABEL)
    ax_residual.set_xticks(TIP_ANGLE_TICKS_DEG)
    ax_residual.grid(True, linewidth=0.45, alpha=0.35)

    # Read the autoscaled residual range, then round outward to a symmetric
    # power-of-ten bound so all residual points remain visible.
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
    plt.setp(ax_rabi.get_xticklabels(), visible=False)


def main() -> None:
    b90_helper = helper_amplitude(90.0 * unit.deg)
    b90_free_decay = free_decay_calibrated_amplitude(90.0 * unit.deg)
    b90_trace = simulate_free_decay(b90_free_decay)
    # Both rate traces use the same 50 requested angles and 100 ms pulse.
    rabi_data = {}
    for rate_hz in RABI_PLOT_RATES_HZ:
        results = [
            simulate_pulse_at_tip_angle(rate_hz, angle_deg)
            for angle_deg in RABI_INPUT_ANGLES_DEG
        ]
        rabi_data[rate_hz] = (
            np.asarray([item[0] for item in results]),
            np.asarray([item[1] for item in results]),
        )
    theory_states = rabi_data[max(RABI_PLOT_RATES_HZ)][1]
    rate_errors = rabi_rate_convergence()
    print(f"RK4-helper B1 for 90 degrees: {b90_helper:.9e} T")
    print(f"Free-decay-calibrated B1 for 90 degrees: {b90_free_decay:.9e} T")
    print(
        f"Rabi sweep: {T90_DURATION.to_value(unit.ms):g} ms pulse, "
        f"{len(RABI_INPUT_ANGLES_DEG)} angles, rates "
        f"{RABI_PLOT_RATES_HZ[0]:g} Hz ({int(T90_DURATION.to_value(unit.s) * RABI_PLOT_RATES_HZ[0])} steps) and "
        f"{RABI_PLOT_RATES_HZ[1]:g} Hz ({int(T90_DURATION.to_value(unit.s) * RABI_PLOT_RATES_HZ[1])} steps)"
    )
    print(
        "Free-decay fit at calibrated B1: "
        f"tip={np.rad2deg(b90_trace['angle_rad']):.9f} deg, "
        f"Mxy0={abs(b90_trace['z0']):.9f}, Mz0={b90_trace['mz0']:.9f}"
    )
    print("rate (Hz) | max tip-angle error (deg) | min |M| | max |My residual|")
    for rate_hz, (states, theory) in rabi_data.items():
        requested = RABI_INPUT_ANGLES_DEG
        actual = np.rad2deg(np.arctan2(states[:, 1], states[:, 2])) % 360.0
        angle_error = (actual - requested + 180.0) % 360.0 - 180.0
        magnitude = np.linalg.norm(states, axis=1)
        max_my_residual = np.max(np.abs(states[:, 1] - theory[:, 1]))
        print(
            f"{rate_hz:9.1f} | {np.max(np.abs(angle_error)):24.6e} | "
            f"{np.min(magnitude):.9f} | {max_my_residual:.6e}"
        )
    print("rate (Hz) | epsilon_Mx | epsilon_My | epsilon_combined")
    for rate_hz, (epsilon_mx, epsilon_my, epsilon_total) in zip(
        CONVERGENCE_RATES_HZ, rate_errors
    ):
        print(
            f"{rate_hz:9.1f} | {epsilon_mx:10.3e} | {epsilon_my:10.3e} "
            f"| {epsilon_total:15.3e}"
        )

    # Match the paper-figure styling used by plot-SpinEcho_and_CW.py.
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
        width_ratios=(1., 1.0),
        height_ratios=(1.0, 0.33),
    )
    fig.subplots_adjust(
        left=0.142, right=0.98, bottom=0.188, top=0.91, wspace=0.38, hspace=0.10
    )
    ax_rabi = fig.add_subplot(gs[0, 0])
    ax_residual = fig.add_subplot(gs[1, 0], sharex=ax_rabi)
    ax_convergence = fig.add_subplot(gs[:, 1])

    ax_rabi.plot(
        RABI_INPUT_ANGLES_DEG,
        theory_states[:, 1],
        # Requested tip angle is the common, amplitude-scaled input grid.
        # Relaxation during the pulse makes this exact prediction nearly sin(theta).
        color="black",
        linestyle="-",
        linewidth=linewidth,
        label="$M_y^\\mathrm{theo}$",
    )
    for index, rate_hz in enumerate(RABI_PLOT_RATES_HZ):
        simulated, theory = rabi_data[rate_hz]
        color = high_contrast_extended[index + 2]
        rate_label = f"{rate_hz:g} Hz"
        ax_rabi.scatter(
            RABI_INPUT_ANGLES_DEG,
            simulated[:, 1],
            s=11,
            marker="o" if index == 0 else "s",
            facecolors="none",
            edgecolors=color,
            linewidths=0.7,
            label="$M_y^\\mathrm{simu}$" + f", {rate_label}",
            zorder=3,
        )
        ax_residual.scatter(
            RABI_INPUT_ANGLES_DEG,
            simulated[:, 1] - theory[:, 1],
            s=11,
            marker="o" if index == 0 else "s",
            facecolors="none",
            edgecolors=color,
            linewidths=0.7,
            zorder=3,
        )

    # ax_rabi.set_title(
    #     f"{T90_DURATION.to_value(unit.ms):g} ms pulse; "
    #     f"{len(RABI_INPUT_ANGLES_DEG)} input angles",
    #     pad=5,
    # )
    # # ax_convergence.set_title("Rate convergence", pad=6)

    ax_rabi.legend(frameon=False, fontsize=8, loc="upper right")
    # ax_convergence.legend(frameon=False, fontsize=8, loc="upper right")

    # For an on-resonance x pulse Mx remains zero by symmetry; the residual
    # panel therefore shows the nontrivial My residual and its zero baseline.

    ax_convergence.loglog(
        CONVERGENCE_RATES_HZ,
        rate_errors[:, 1],
        "s-",
        color=high_contrast_extended[4],
        linewidth=linewidth,
        markersize=3.5,
        label=r"$\epsilon_{M_y}$",
    )

    # ax_convergence.text(
    #     0.04,
    #     0.04,
    #     r"$\epsilon_{M_x}=0$ by symmetry",
    #     transform=ax_convergence.transAxes,
    #     fontsize=7,
    # )

    ax_rabi.text(-0.18, 1.04, "(a)", transform=ax_rabi.transAxes, fontsize=10)
    ax_convergence.text(
        -0.18, 1.04, "(b)", transform=ax_convergence.transAxes, fontsize=10
    )
    configure_plot_axes(ax_rabi, ax_residual, ax_convergence)
    fig.align_ylabels((ax_rabi, ax_residual))
    output = Path(__file__).with_name("calibration-excitation.png")
    fig.savefig(output, dpi=300)
    print(f"Saved plot to {output}")
    pdf_output = Path(__file__).with_name("calibration-excitation.pdf")
    fig.savefig(pdf_output)
    print(f"Saved plot to {pdf_output}")
    # Keep the GUI window open for interactive inspection after saving files.
    plt.show(block=True)


if __name__ == "__main__":
    main()
