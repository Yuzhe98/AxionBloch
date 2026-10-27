"""Monte Carlo estimate of real-axion-field g1 coherence.

The field is represented by a carrier at the Compton frequency multiplied by
the complex envelope A(t). Uniform random state phases are sampled while the
state magnitudes and frequencies are held fixed. The phase-ensemble average
of A*(0) A(tau) estimates the envelope g1; its magnitude is compared with the
analytic weighted-frequency sum.

Plots are saved under outputs/g1_monte_carlo/ so existing figures are kept.

Run from the repository root with:
    python examples/EarthBoundAxionHalo/coherence_time/mc_g1_coherence.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from g2_coherence_time_discrete_states import (
    OFFSETS_HZ,
    RANDOM_SEED,
    SAMPLES_PER_FASTEST_BEAT,
    draw_random_coefficients,
    first_threshold_time,
)


DEFAULT_N_REALIZATIONS = 100_000
N_BATCHES = 100
MAX_DELAY_S = 20_000.0
PHASE_SEED = 271828


def analytic_g1_envelope(
    offsets_hz: np.ndarray, weights: np.ndarray, tau_s: np.ndarray
) -> np.ndarray:
    """Analytic envelope |g1| for independently randomized mode phases."""
    phases = np.exp(-2j * np.pi * offsets_hz[:, None] * tau_s[None, :])
    g1 = (weights[:, None] * phases).sum(axis=0) / weights.sum()
    return np.abs(g1)


def main(n_realizations: int = DEFAULT_N_REALIZATIONS) -> None:
    if n_realizations < N_BATCHES:
        raise ValueError(f"Use at least {N_BATCHES} realizations for the batch estimate.")

    base_coefficients = draw_random_coefficients(
        len(OFFSETS_HZ), RANDOM_SEED
    )
    amplitudes = np.abs(base_coefficients)
    weights = amplitudes**2

    dt_s = 1.0 / (
        SAMPLES_PER_FASTEST_BEAT * np.ptp(OFFSETS_HZ)
    )
    tau_s = np.arange(int(MAX_DELAY_S / dt_s) + 1) * dt_s
    analytic = analytic_g1_envelope(OFFSETS_HZ, weights, tau_s)
    analytic_tau = first_threshold_time(tau_s, analytic, 1.0 / np.e)

    phase_matrix = np.exp(-2j * np.pi * OFFSETS_HZ[:, None] * tau_s[None, :])
    batch_size = n_realizations // N_BATCHES
    used_realizations = batch_size * N_BATCHES
    rng = np.random.default_rng(PHASE_SEED)
    batch_g1 = np.empty((N_BATCHES, tau_s.size), dtype=complex)
    batch_denominators = np.empty(N_BATCHES)

    start = time.perf_counter()
    for batch_index in range(N_BATCHES):
        phases = rng.uniform(
            0.0,
            2.0 * np.pi,
            size=(batch_size, len(OFFSETS_HZ)),
        )
        coefficients = amplitudes[None, :] * np.exp(1j * phases)
        field_at_zero = coefficients.sum(axis=1)
        field_at_delay = coefficients @ phase_matrix
        numerator = np.mean(
            np.conjugate(field_at_zero)[:, None] * field_at_delay,
            axis=0,
        )
        denominator = np.mean(np.abs(field_at_zero) ** 2)
        batch_denominators[batch_index] = denominator
        batch_g1[batch_index] = numerator / denominator
    elapsed_s = time.perf_counter() - start

    # Aggregate numerator and denominator across all phase draws, then take
    # the magnitude to obtain the Monte Carlo ensemble g1 envelope.
    mc_g1_complex = np.average(batch_g1, axis=0, weights=batch_denominators)
    mc_envelope = np.abs(mc_g1_complex)
    mc_tau = first_threshold_time(tau_s, mc_envelope, 1.0 / np.e)

    batch_envelopes = np.abs(batch_g1)
    batch_tau = np.array(
        [first_threshold_time(tau_s, curve, 1.0 / np.e) for curve in batch_envelopes],
        dtype=float,
    )
    batch_tau = batch_tau[np.isfinite(batch_tau)]
    mean_envelope = batch_envelopes.mean(axis=0)
    sem_envelope = batch_envelopes.std(axis=0, ddof=1) / np.sqrt(N_BATCHES)
    ci_low = np.maximum(0.0, mean_envelope - 1.96 * sem_envelope)
    ci_high = mean_envelope + 1.96 * sem_envelope

    output_dir = Path(__file__).parent / "outputs" / "g1_monte_carlo"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase realizations used: {used_realizations:,}")
    print(f"Independent batches: {N_BATCHES} x {batch_size:,}")
    print(f"MC correlation calculation time: {elapsed_s:.3f} s")
    print(f"Average per phase realization: {1e3 * elapsed_s / used_realizations:.4f} ms")
    print("Varied: independent uniform phases; held fixed: frequencies and amplitudes")
    print(f"Analytic |g1| first 1/e crossing: {analytic_tau:.3f} s")
    print(f"MC ensemble |g1| first 1/e crossing: {mc_tau:.3f} s")
    if batch_tau.size:
        q16, median, q84 = np.percentile(batch_tau, [16, 50, 84])
        print(f"Batch-estimate crossing median: {median:.3f} s")
        print(f"Batch-estimate 16th-84th percentile: {q16:.3f} to {q84:.3f} s")
    print(f"Saved plots in: {output_dir}")

    # Plot analytic curve and MC estimate with batch-based uncertainty.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(
        tau_s,
        ci_low,
        ci_high,
        color="tab:blue",
        alpha=0.2,
        label="95% batch-mean interval",
    )
    ax.plot(tau_s, mean_envelope, color="tab:blue", lw=1.4, label="MC mean of batch |g1|")
    ax.plot(tau_s, analytic, "--", color="black", lw=1.6, label="Analytic |g1|")
    ax.axhline(1.0 / np.e, color="tab:red", linestyle="--", label=r"$1/e$ threshold")
    if analytic_tau is not None:
        ax.axvline(
            analytic_tau,
            color="black",
            linestyle=":",
            label=f"Analytic crossing: {analytic_tau:.0f} s",
        )
    if mc_tau is not None:
        ax.axvline(
            mc_tau,
            color="tab:blue",
            linestyle=":",
            label=f"MC crossing: {mc_tau:.0f} s",
        )
    ax.set_xlabel(r"Delay $\tau$ (s)")
    ax.set_ylabel(r"Real-field correlation envelope $|g^{(1)}(\tau)|$")
    ax.set_title(
        f"Axion g1 coherence: {used_realizations:,} random-phase realizations"
    )
    ax.set_xlim(0.0, MAX_DELAY_S)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    curve_path = output_dir / "g1_mc_vs_analytic.png"
    fig.savefig(curve_path, dpi=180)

    # Batch-level crossing times visualize finite-MC uncertainty in the root.
    hist_fig, hist_ax = plt.subplots(figsize=(8, 5))
    hist_ax.hist(batch_tau, bins=20, color="tab:blue", alpha=0.75, edgecolor="white")
    hist_ax.axvline(
        analytic_tau,
        color="black",
        linestyle="--",
        lw=2,
        label=f"Analytic: {analytic_tau:.1f} s",
    )
    if mc_tau is not None:
        hist_ax.axvline(
            mc_tau,
            color="tab:orange",
            linestyle=":",
            lw=2,
            label=f"Full MC: {mc_tau:.1f} s",
        )
    hist_ax.set_xlabel(r"First $1/e$ crossing of $|g^{(1)}|$ (s)")
    hist_ax.set_ylabel("Number of batches")
    hist_ax.set_title(f"Batch coherence-time estimates ({batch_size:,} draws each)")
    hist_ax.grid(True, axis="y", alpha=0.25)
    hist_ax.legend()
    hist_fig.tight_layout()
    hist_path = output_dir / "g1_mc_coherence_time_distribution.png"
    hist_fig.savefig(hist_path, dpi=180)

    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=DEFAULT_N_REALIZATIONS,
        help=f"number of random-phase draws (default: {DEFAULT_N_REALIZATIONS})",
    )
    main(parser.parse_args().n_realizations)
