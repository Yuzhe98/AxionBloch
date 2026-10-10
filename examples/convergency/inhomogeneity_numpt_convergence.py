"""Check packet-count convergence against a pulse-aware Bloch prediction.

The theoretical trajectory propagates each packet through the sampled RF
pulse with an exact matrix exponential (including detuning and T1/T2
relaxation), then evaluates its free decay analytically. Comparing this
prediction across packet grids measures quadrature convergence; comparing the
RK4 result with its same-grid prediction measures time-integration error.

Run from the repository root with::

    python examples/convergency/inhomogeneity_numpt_convergence.py
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.dependency import PI, ppm, unit
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation


RCF_FREQ = 1.0 * unit.MHz
DELTA_NU_L = 1.0 * unit.Hz
FWHM = 0.1 * ppm
N_FWHM = 10.0
T1 = 1.0e6 * unit.s
T2 = 10.0 * unit.s
RATE = 4000.0 * unit.Hz
DURATION = 3.0 * unit.s
PULSE_STEPS = 5
NUM_PTS = [25, 50, 100, 200, 400, 800, 1600]
REFERENCE_NUM_PTS = 3201


def make_magnet(num_pts: int) -> Magnet:
    """Construct the same truncated Lorentzian field grid at a chosen size."""
    sample_gamma_over_2pi = gamma_p / (2 * PI)
    return Magnet(
        name=f"numPt_{num_pts}",
        B0=(RCF_FREQ - DELTA_NU_L) / sample_gamma_over_2pi,
        FWHM=FWHM,
        numPt=num_pts,
        nFWHM=N_FWHM,
    )


def pulse_aware_prediction(
    magnet: Magnet,
    pulse_b_t: np.ndarray,
    t_free_s: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Predict weighted complex Mx+iMy after the actual sampled pulse.

    Each packet starts at equilibrium (0, 0, 1). During the pulse, a 4x4
    affine Bloch generator is exponentiated for every piecewise-constant
    field sample. Its z component is the packet's rotating-frame detuning;
    the x/y components come directly from the pulse builder. After the pulse,
    transverse relaxation and precession are analytic.
    """
    gamma = gamma_p.to_value(unit.rad * unit.Hz / unit.T)
    gamma_over_2pi = gamma / (2 * np.pi)
    r1 = 1.0 / T1.to_value(unit.s)
    r2 = 1.0 / T2.to_value(unit.s)
    detuning_hz = (
        gamma_over_2pi * magnet.B_spread.to_value(unit.T)
        - RCF_FREQ.to_value(unit.Hz)
    )
    pulse_components = pulse_b_t[:, :2]
    packet_weights = np.asarray(magnet.ratios, dtype=float)
    pulse_end_transverse = np.empty(len(packet_weights), dtype=complex)

    for packet_i, detuning in enumerate(detuning_hz):
        # Column-vector convention: dM/dt = gamma * (M x B) plus Bloch
        # relaxation. The last coordinate makes the Mz recovery affine.
        omega_z = 2 * np.pi * detuning
        state = np.array([0.0, 0.0, 1.0, 1.0])
        for bx, by in pulse_components:
            omega_x = gamma * bx
            omega_y = gamma * by
            generator = np.array(
                [
                    [-r2, omega_z, -omega_y, 0.0],
                    [-omega_z, -r2, omega_x, 0.0],
                    [omega_y, -omega_x, -r1, r1],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
            state = expm(generator * dt_s) @ state
        pulse_end_transverse[packet_i] = state[0] + 1j * state[1]

    phase = np.exp(-2j * np.pi * detuning_hz[:, None] * t_free_s[None, :])
    decay = np.exp(-t_free_s / T2.to_value(unit.s))
    packet_signals = pulse_end_transverse[:, None] * phase * decay[None, :]
    return np.sum(packet_weights[:, None] * packet_signals, axis=0)


def simulate_case(num_pts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return free-decay times, RK4 signal, and same-grid exact prediction."""
    sample = Sample(
        name="numPt_convergence",
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
    simu = Simulation(
        name=f"numPt_convergence_{num_pts}",
        sample=sample,
        magnet=make_magnet(num_pts),
        excField=MagField(name="90deg_pulse"),
        RCF_freq=RCF_FREQ,
        rate=RATE,
        duration=DURATION,
        verbose=False,
    )
    # Simulation initialization may select its own quadrature size. Restore
    # the requested grid so only numPt changes in this sweep.
    simu.magnet.setHomogeneity(numPt=num_pts)
    simu.excField.set90DegPulse(
        timeStep=simu.timeStep,
        timeLen=simu.timeLen,
        gamma=sample.gamma,
        t90=PULSE_STEPS * simu.timeStep,
        nu_rot=DELTA_NU_L,
    )
    pulse_b_t = simu.excField.B_vec.to_value(unit.T)[:PULSE_STEPS].copy()
    time_s = simu.getTimeStamp().to_value(unit.s)
    t_free_s = time_s[PULSE_STEPS:] - time_s[PULSE_STEPS]
    theory = pulse_aware_prediction(
        simu.magnet,
        pulse_b_t,
        t_free_s,
        simu.timeStep.to_value(unit.s),
    )
    simu.generateTrajectories(integrator="RK4")
    m = simu.trjry[0, PULSE_STEPS:, :2]
    simulated = m[:, 0] + 1j * m[:, 1]
    return t_free_s, simulated, theory


def component_errors(signal: np.ndarray, reference: np.ndarray) -> tuple[float, float, float]:
    """Return normalized L2 errors for Mx, My, and their combined vector."""
    difference = signal - reference
    denominator = np.sum(np.abs(reference) ** 2)
    epsilon_mx = np.sqrt(np.sum(difference.real**2) / denominator)
    epsilon_my = np.sqrt(np.sum(difference.imag**2) / denominator)
    return float(epsilon_mx), float(epsilon_my), float(np.hypot(epsilon_mx, epsilon_my))


if __name__ == "__main__":
    t_reference, _, theory_reference = simulate_case(REFERENCE_NUM_PTS)
    results = [simulate_case(num_pts) for num_pts in NUM_PTS]

    print(
        f"Rate={RATE.to_value(unit.Hz):g} Hz, FWHM={FWHM.to_value(ppm):g} ppm, "
        f"reference numPt={REFERENCE_NUM_PTS}; pulse-aware theory includes "
        "detuning and relaxation during the sampled pulse."
    )
    print(
        "numPt | theory vs ref (Mx, My, total) | "
        "RK4 vs same-grid theory (Mx, My, total)"
    )
    quadrature_errors = []
    integrator_errors = []
    for num_pts, (t_free_s, simulated, theory) in zip(NUM_PTS, results):
        quadrature_error = component_errors(theory, theory_reference)
        integrator_error = component_errors(simulated, theory)
        quadrature_errors.append(quadrature_error)
        integrator_errors.append(integrator_error)
        print(
            f"{num_pts:5d} | "
            f"{quadrature_error[0]:.3e}, {quadrature_error[1]:.3e}, "
            f"{quadrature_error[2]:.3e} | "
            f"{integrator_error[0]:.3e}, {integrator_error[1]:.3e}, "
            f"{integrator_error[2]:.3e}"
        )

    csv_path = Path(__file__).with_name("inhomogeneity_numpt_convergence.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["numPt", "quadrature_Mx", "quadrature_My", "quadrature_total",
             "integrator_Mx", "integrator_My", "integrator_total"]
        )
        for num_pts, quadrature_error, integrator_error in zip(
            NUM_PTS, quadrature_errors, integrator_errors
        ):
            writer.writerow([num_pts, *quadrature_error, *integrator_error])
    print(f"Saved component errors to {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for values, label, marker in (
        (quadrature_errors, "Quadrature: theory vs reference", "o"),
        (integrator_errors, "RK4: simulation vs same-grid theory", "s"),
    ):
        axes[0].loglog(NUM_PTS, [v[0] for v in values], marker + "-", label=f"{label}: Mx")
        axes[0].loglog(NUM_PTS, [v[1] for v in values], marker + "--", label=f"{label}: My")
    axes[0].set(
        xlabel="Inhomogeneity packets (numPt)",
        ylabel="Normalized component L2 error",
        title="Error sources separated",
    )
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    for num_pts, (t_free_s, simulated, _) in zip(NUM_PTS, results):
        if num_pts in (200, 800, 1600):
            axes[1].plot(t_free_s, simulated.real, label=f"Mx, numPt={num_pts}")
    axes[1].plot(t_reference, theory_reference.real, "k--", label="pulse-aware reference Mx")
    axes[1].set(xlabel="Time after pulse (s)", ylabel="$M_x$", title="Signal convergence")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("Pulse-aware inhomogeneity convergence at 0.1 ppm")
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("inhomogeneity_numpt_convergence.png"), dpi=180)
    plt.show()
