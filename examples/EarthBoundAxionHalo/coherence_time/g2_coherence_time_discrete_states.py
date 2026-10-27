"""Simulate g2 coherence for a finite set of Earth-bound axion states.

The coefficients are drawn once (random complex amplitudes) and then held
fixed. The resulting field is a deterministic superposition of eigenstates.
The plotted axion field is real, a(t) = Re[A(t) exp(-i 2 pi f_c t)]/sqrt(2).
For g2, the script uses the carrier-cycle-averaged real-field intensity
<a(t)^2>_carrier = |A(t)|^2/4, then computes g2 from a simulated envelope
time series and from the exact time-averaged finite-mode expression. It
reports the first 1/e crossing of g2(tau) - 1.

Run from the repository root with:
    python examples/EarthBoundAxionHalo/coherence_time/g2_coherence_time_discrete_states.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from pathlib import Path


# The input offsets below are milli-Hz (mHz), as specified. Convert them to Hz
# before adding them to the 1.3 MHz Compton carrier. The small offsets set the
# slow beat pattern and hence the g2 coherence curve.
COMPTON_FREQUENCY_HZ = 1.3e6
OFFSETS_MILLIHZ = np.concatenate(
    (
        np.array([-0.25, -0.10]),
        np.linspace(-0.09, 0.0, 10),
    )
)
OFFSETS_HZ = OFFSETS_MILLIHZ * 1e-3
EIGENFREQUENCIES_HZ = COMPTON_FREQUENCY_HZ + OFFSETS_HZ

# A fixed seed makes the random coefficients reproducible. Each coefficient
# has an independent random amplitude and phase (circular complex Gaussian).
RANDOM_SEED = 20261026
SIMULATION_DURATION_S = 2.0e6
SAMPLES_PER_FASTEST_BEAT = 100
MAX_LAG_FRACTION = 0.5


def draw_random_coefficients(n_states: int, seed: int) -> np.ndarray:
    """Return reproducible random complex field amplitudes."""
    rng = np.random.default_rng(seed)
    coeff = rng.normal(size=n_states) + 1j * rng.normal(size=n_states)
    return coeff / np.sqrt(2.0 * n_states)


def simulate_envelope(
    offsets_hz: np.ndarray, coefficients: np.ndarray, duration_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the complex envelope, omitting the common Compton carrier."""
    # The offsets determine the slow envelope. Sampling the full carrier here
    # would need sub-microsecond steps even though the envelope evolves over hours.
    relative_hz = offsets_hz - np.mean(offsets_hz)
    max_beat_hz = np.max(np.abs(relative_hz[:, None] - relative_hz[None, :]))
    dt = 1.0 / (SAMPLES_PER_FASTEST_BEAT * max_beat_hz)
    time_s = np.arange(0.0, duration_s, dt)

    field = np.zeros(time_s.size, dtype=complex)
    for frequency_hz, coefficient in zip(relative_hz, coefficients):
        field += coefficient * np.exp(-2j * np.pi * frequency_hz * time_s)
    return time_s, field


def g1_from_envelope(envelope: np.ndarray) -> np.ndarray:
    """Estimate the real-field g1 envelope from the complex envelope A(t)."""
    n = envelope.size
    max_lag = max(2, int(n * MAX_LAG_FRACTION))
    corr = fftconvolve(envelope, np.conjugate(envelope[::-1]), mode="full")[n - 1 :]
    corr = corr[:max_lag] / np.arange(n, n - max_lag, -1)
    return np.abs(corr / np.mean(np.abs(envelope) ** 2))


def g2_from_time_series(intensity: np.ndarray) -> np.ndarray:
    """Estimate cycle-averaged real-field g2 with overlap normalization.

    The physical real field is a(t)=Re[A(t) exp(-i 2 pi f_c t)]/sqrt(2).
    Averaging a(t)^2 over carrier cycles gives |A(t)|^2/4. The constant 1/4
    cancels in normalized g2, so the simulated envelope intensity |A|^2 is
    used here.
    """
    n = intensity.size
    max_lag = max(2, int(n * MAX_LAG_FRACTION))
    # Linear autocorrelation avoids wrapping the end of the record onto its
    # beginning. Divide by the number of contributing pairs at each lag.
    corr = fftconvolve(intensity, intensity[::-1], mode="full")[n - 1 :]
    corr = corr[:max_lag] / np.arange(n, n - max_lag, -1)
    return corr / np.mean(intensity) ** 2


def exact_time_averaged_g2(
    offsets_hz: np.ndarray, coefficients: np.ndarray, tau_s: np.ndarray
) -> np.ndarray:
    """Evaluate the finite-mode expression after averaging over start time.

    For each beat frequency nu, B_nu = sum_(j,k: f_j-f_k=nu) c_j c_k*.
    Then g2(tau) = sum_nu |B_nu|^2 exp(-i 2 pi nu tau) / <I>^2.
    Equal beat frequencies are grouped before squaring their amplitudes.
    """
    beat_hz = []
    beat_amplitudes = []
    for j, fj in enumerate(offsets_hz):
        for k, fk in enumerate(offsets_hz):
            beat_hz.append(fj - fk)
            beat_amplitudes.append(coefficients[j] * coefficients[k].conjugate())

    # Frequencies are specified on a decimal mHz grid. Rounding groups the
    # mathematically identical differences despite floating-point arithmetic.
    beat_hz = np.round(np.asarray(beat_hz), decimals=12)
    beat_amplitudes = np.asarray(beat_amplitudes)
    unique_beats, inverse = np.unique(beat_hz, return_inverse=True)
    grouped = np.zeros(unique_beats.size, dtype=complex)
    np.add.at(grouped, inverse, beat_amplitudes)

    mean_intensity = np.sum(np.abs(coefficients) ** 2)
    weights = np.abs(grouped) ** 2
    phases = np.exp(-2j * np.pi * unique_beats[:, None] * tau_s[None, :])
    return np.real(weights @ phases) / mean_intensity**2


def first_excess_efold_time(tau_s: np.ndarray, g2: np.ndarray) -> float | None:
    """Scan from zero delay and interpolate the first 1/e crossing of g2-1."""
    excess = g2 - 1.0
    if excess[0] <= 0:
        return None
    target = excess[0] / np.e
    # Explicit left-to-right scan matters because discrete-mode curves can
    # oscillate and cross the threshold more than once.
    for i in range(1, len(excess)):
        if excess[i] <= target:
            x0, x1 = tau_s[i - 1], tau_s[i]
            y0, y1 = excess[i - 1], excess[i]
            if y1 == y0:
                return float(x1)
            return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
    return None


def first_threshold_time(
    tau_s: np.ndarray, values: np.ndarray, threshold: float
) -> float | None:
    """Linearly interpolate the first downward crossing of a threshold."""
    crossings = np.flatnonzero(values[1:] <= threshold)
    if crossings.size == 0:
        return None
    i = int(crossings[0] + 1)
    x0, x1 = tau_s[i - 1], tau_s[i]
    y0, y1 = values[i - 1], values[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))


def main() -> None:
    coefficients = draw_random_coefficients(len(OFFSETS_HZ), RANDOM_SEED)
    time_s, envelope = simulate_envelope(
        OFFSETS_HZ, coefficients, SIMULATION_DURATION_S
    )
    intensity = np.abs(envelope) ** 2
    g2_simulated = g2_from_time_series(intensity)
    tau_s = np.arange(g2_simulated.size) * (time_s[1] - time_s[0])
    g2_analytic = exact_time_averaged_g2(OFFSETS_HZ, coefficients, tau_s)
    g1_simulated = g1_from_envelope(envelope)[: tau_s.size]
    coefficient_weights = np.abs(coefficients) ** 2
    g1_analytic = np.abs(
        np.sum(
            coefficient_weights[:, None]
            * np.exp(-2j * np.pi * OFFSETS_HZ[:, None] * tau_s[None, :]),
            axis=0,
        )
        / np.sum(coefficient_weights)
    )

    tau_coh = first_excess_efold_time(tau_s, g2_simulated)
    tau_coh_analytic = first_excess_efold_time(tau_s, g2_analytic)
    tau_g1 = first_threshold_time(tau_s, g1_simulated, 1.0 / np.e)
    tau_g1_analytic = first_threshold_time(tau_s, g1_analytic, 1.0 / np.e)

    print(f"Compton frequency: {COMPTON_FREQUENCY_HZ / 1e6:.6f} MHz")
    print(f"Number of states: {len(OFFSETS_HZ)}")
    print("Offsets (mHz):", np.array2string(OFFSETS_MILLIHZ, precision=4))
    print("Random coefficients (fixed for this realization):")
    for i, (offset, coefficient) in enumerate(zip(OFFSETS_MILLIHZ, coefficients)):
        print(
            f"  state {i:2d}: offset={offset: .5f} mHz, "
            f"c={coefficient.real:+.5f}{coefficient.imag:+.5f}j"
        )
    print(f"Time samples: {len(time_s)}; dt={time_s[1] - time_s[0]:.3f} s")
    print("Coherence definition: first 1/e crossing of g2(tau) - 1")
    print("g1 definition: first 1/e crossing of the real-field correlation envelope")
    print(f"Numerical g1 coherence time: {tau_g1:.3f} s")
    print(f"Analytic g1 coherence time: {tau_g1_analytic:.3f} s")
    if tau_coh is None:
        print("Numerical g2: no 1/e crossing in the simulated lag range")
    else:
        print(f"Numerical g2 coherence time: {tau_coh:.3f} s")
    if tau_coh_analytic is None:
        print("Analytic time-average g2: no 1/e crossing in the simulated lag range")
    else:
        print(f"Analytic time-average g2 coherence time: {tau_coh_analytic:.3f} s")

    # Resolve a short segment of the real, carrier-frequency axion field for
    # display. The long g2 calculation uses the complex envelope, avoiding
    # billions of samples needed to resolve a 1.3 MHz carrier for hours.
    field_window_s = 10e-6
    field_dt_s = 1.0 / (20.0 * COMPTON_FREQUENCY_HZ)
    field_time_s = np.arange(0.0, field_window_s, field_dt_s)
    field_complex = np.zeros(field_time_s.size, dtype=complex)
    for frequency_hz, coefficient in zip(EIGENFREQUENCIES_HZ, coefficients):
        field_complex += coefficient * np.exp(
            -2j * np.pi * frequency_hz * field_time_s
        )
    axion_field_a = np.real(field_complex) / np.sqrt(2.0)

    fig, (field_ax, envelope_ax, g2_ax) = plt.subplots(
        3,
        1,
        figsize=(10, 10),
        gridspec_kw={"height_ratios": [1, 0.9, 1.35]},
    )
    field_ax.plot(field_time_s * 1e6, axion_field_a, lw=1.0, color="tab:blue")
    field_ax.set_xlabel("Time (microseconds)")
    field_ax.set_ylabel(r"$a(t)$ (arb. units)")
    field_ax.set_title(
        r"Axion field $a(t)=\mathrm{Re}[A(t)]/\sqrt{2}$ "
        "(1.3 MHz carrier resolved)"
    )
    field_ax.grid(True, alpha=0.25)

    envelope_window_s = min(20_000.0, time_s[-1])
    envelope_mask = time_s <= envelope_window_s
    envelope_ax.plot(
        time_s[envelope_mask], np.sqrt(intensity[envelope_mask]), color="tab:green"
    )
    envelope_ax.set_xlabel("Time (s)")
    envelope_ax.set_ylabel(r"Envelope $|A(t)|$ (arb. units)")
    envelope_ax.set_title("Slow field envelope from the mHz frequency offsets")
    envelope_ax.grid(True, alpha=0.25)

    g2_ax.plot(tau_s, g2_simulated, label="Simulated time-series estimate", lw=1.5)
    g2_ax.plot(
        tau_s,
        g2_analytic,
        "--",
        label="Analytic time-averaged finite-mode result",
        lw=1.4,
    )
    g2_ax.axhline(1.0, color="black", lw=0.8, alpha=0.6)
    if tau_coh is not None:
        g2_threshold = 1.0 + (g2_simulated[0] - 1.0) / np.e
        g2_ax.axhline(
            g2_threshold,
            color="tab:red",
            linestyle="--",
            label=rf"$1/e$ excess threshold: {g2_threshold:.3f}",
        )
        g2_ax.axvline(
            tau_coh,
            color="tab:red",
            linestyle=":",
            label=rf"Numerical $\tau_{{coh}}={tau_coh:.1f}$ s",
        )
        g2_ax.scatter([tau_coh], [g2_threshold], color="tab:red", zorder=4)
    g2_ax.set_xlabel(r"Delay $\tau$ (s)")
    g2_ax.set_ylabel(r"$g^{(2)}(\tau)$")
    g2_ax.set_title(
        r"Cycle-averaged real-field coherence: first $1/e$ crossing of $g^{(2)}-1$"
    )
    # Focus this panel on the first-coherence region so the threshold marker is
    # visible; the full lag-range data are still computed above.
    g2_ax.set_xlim(0.0, min(20_000.0, tau_s[-1]))
    g2_ax.grid(True, alpha=0.25)
    g2_ax.legend()
    fig.tight_layout()
    output_path = Path(__file__).with_name("g2_coherence_time_discrete_states.png")
    plt.savefig(output_path, dpi=180)
    print(f"Saved plot: {output_path}")

    # A separate comparison makes each coherence convention and threshold
    # visible on its own curve.
    compare_fig, (g1_ax, compare_g2_ax) = plt.subplots(2, 1, figsize=(9, 8))
    g1_ax.plot(tau_s, g1_simulated, label="Numerical real-field correlation", lw=1.5)
    g1_ax.plot(tau_s, g1_analytic, "--", label="Analytic mode sum", lw=1.3)
    g1_ax.axhline(1.0 / np.e, color="tab:red", linestyle="--", label=r"$1/e$ threshold")
    if tau_g1 is not None:
        g1_ax.axvline(tau_g1, color="tab:red", linestyle=":")
        g1_ax.scatter([tau_g1], [1.0 / np.e], color="tab:red", zorder=4)
    g1_ax.set_xlim(0.0, min(20_000.0, tau_s[-1]))
    g1_ax.set_xlabel(r"Delay $\tau$ (s)")
    g1_ax.set_ylabel(r"$|g^{(1)}(\tau)|$")
    g1_ax.set_title(r"Real-field autocorrelation envelope: $1/e$ crossing")
    g1_ax.grid(True, alpha=0.25)
    g1_ax.legend()

    compare_g2_ax.plot(tau_s, g2_simulated, label="Numerical cycle-averaged g2", lw=1.5)
    compare_g2_ax.plot(tau_s, g2_analytic, "--", label="Analytic finite-mode g2", lw=1.3)
    compare_g2_ax.axhline(1.0, color="black", lw=0.8, alpha=0.6)
    compare_threshold = 1.0 + (g2_simulated[0] - 1.0) / np.e
    compare_g2_ax.axhline(
        compare_threshold,
        color="tab:red",
        linestyle="--",
        label=rf"$1/e$ excess threshold: {compare_threshold:.3f}",
    )
    if tau_coh is not None:
        compare_g2_ax.axvline(tau_coh, color="tab:red", linestyle=":")
        compare_g2_ax.scatter(
            [tau_coh], [compare_threshold], color="tab:red", zorder=4
        )
    compare_g2_ax.set_xlim(0.0, min(20_000.0, tau_s[-1]))
    compare_g2_ax.set_xlabel(r"Delay $\tau$ (s)")
    compare_g2_ax.set_ylabel(r"$g^{(2)}(\tau)$")
    compare_g2_ax.set_title(r"Cycle-averaged intensity correlation: $1/e$ excess crossing")
    compare_g2_ax.grid(True, alpha=0.25)
    compare_g2_ax.legend()
    compare_fig.tight_layout()
    compare_path = Path(__file__).with_name("g1_g2_coherence_comparison.png")
    compare_fig.savefig(compare_path, dpi=180)
    print(f"Saved comparison plot: {compare_path}")
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
