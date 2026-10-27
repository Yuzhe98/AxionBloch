"""Monte Carlo g1 coherence estimated explicitly from the real axion field.

Random phases are drawn for the 12 modes while amplitudes and frequencies are
fixed. For each phase realization, the two real-field quadratures are sampled
at t=0 and t=tau, combined into the analytic signal, and demodulated by the
1.3 MHz carrier. The resulting ensemble correlation is the slow envelope of
the real-field autocorrelation. Plots go to a new output subfolder.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from g2_coherence_time_discrete_states import (
    COMPTON_FREQUENCY_HZ,
    OFFSETS_HZ,
    RANDOM_SEED,
    SAMPLES_PER_FASTEST_BEAT,
    draw_random_coefficients,
    first_threshold_time,
)


N_BATCHES = 100
N_REALIZATIONS = 100_000
MAX_DELAY_S = 20_000.0
PHASE_SEED = 271828


def analytic_g1(offsets_hz: np.ndarray, weights: np.ndarray, tau_s: np.ndarray):
    """Exact mode-sum envelope of the real-field autocorrelation."""
    mode_phase = np.exp(-2j * np.pi * offsets_hz[:, None] * tau_s[None, :])
    return np.abs((weights[:, None] * mode_phase).sum(axis=0) / weights.sum())


def mc_g1_from_real_field(
    amplitudes: np.ndarray,
    offsets_hz: np.ndarray,
    tau_s: np.ndarray,
    n_realizations: int,
    n_batches: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate g1 using real-field I/Q samples for random phase draws.

    a_I(t)=Re[z(t)]/sqrt(2) and a_Q(t)=Im[z(t)]/sqrt(2), where
    z(t)=A(t) exp(-i 2 pi f_c t). Reconstructing z from these real-field
    quadratures and removing the known carrier yields A(t). The correlation
    is then accumulated as <A*(0) A(tau)>/<|A(0)|^2>.
    """
    batch_size = n_realizations // n_batches
    used = batch_size * n_batches
    rng = np.random.default_rng(seed)
    offsets_phase = np.exp(-2j * np.pi * offsets_hz[:, None] * tau_s[None, :])
    carrier = np.exp(-2j * np.pi * COMPTON_FREQUENCY_HZ * tau_s)

    batch_g1 = np.empty((n_batches, tau_s.size), dtype=complex)
    batch_den = np.empty(n_batches)
    start = time.perf_counter()
    for b in range(n_batches):
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(batch_size, amplitudes.size))
        coeff = amplitudes[None, :] * np.exp(1j * phase)

        # Real-field quadratures at the reference time t=0.
        envelope_0 = coeff.sum(axis=1)
        real_a0 = np.real(envelope_0) / np.sqrt(2.0)
        imag_a0 = np.imag(envelope_0) / np.sqrt(2.0)
        analytic_field_0 = np.sqrt(2.0) * (real_a0 + 1j * imag_a0)

        # Real-field quadratures at delayed times, including the carrier.
        envelope_delay = coeff @ offsets_phase
        carrier_field = envelope_delay * carrier[None, :]
        real_a_delay = np.real(carrier_field) / np.sqrt(2.0)
        imag_a_delay = np.imag(carrier_field) / np.sqrt(2.0)
        analytic_field_delay = np.sqrt(2.0) * (
            real_a_delay + 1j * imag_a_delay
        )

        # Demodulate the real-field analytic signal to recover its slow envelope.
        envelope_delay_recovered = analytic_field_delay * np.conjugate(carrier)[None, :]
        numerator = np.mean(
            np.conjugate(analytic_field_0)[:, None] * envelope_delay_recovered,
            axis=0,
        )
        denominator = np.mean(np.abs(analytic_field_0) ** 2)
        batch_g1[b] = numerator / denominator
        batch_den[b] = denominator
    elapsed = time.perf_counter() - start

    # Weighted aggregation recovers the ratio of the total sampled numerator
    # and denominator across the independent batches.
    mc_complex = np.average(batch_g1, axis=0, weights=batch_den)
    return mc_complex, batch_g1, elapsed


def main(n_realizations: int = N_REALIZATIONS) -> None:
    if n_realizations < N_BATCHES:
        raise ValueError(f"Use at least {N_BATCHES} realizations.")

    original_coeff = draw_random_coefficients(len(OFFSETS_HZ), RANDOM_SEED)
    amplitudes = np.abs(original_coeff)
    weights = amplitudes**2
    dt_s = 1.0 / (SAMPLES_PER_FASTEST_BEAT * np.ptp(OFFSETS_HZ))
    tau_s = np.arange(int(MAX_DELAY_S / dt_s) + 1) * dt_s

    exact = analytic_g1(OFFSETS_HZ, weights, tau_s)
    exact_tau = first_threshold_time(tau_s, exact, 1.0 / np.e)
    mc_complex, batch_complex, elapsed_s = mc_g1_from_real_field(
        amplitudes,
        OFFSETS_HZ,
        tau_s,
        n_realizations,
        N_BATCHES,
        PHASE_SEED,
    )
    mc_envelope = np.abs(mc_complex)
    mc_tau = first_threshold_time(tau_s, mc_envelope, 1.0 / np.e)
    batch_envelope = np.abs(batch_complex)
    batch_tau = np.array(
        [first_threshold_time(tau_s, curve, 1.0 / np.e) for curve in batch_envelope],
        dtype=float,
    )
    batch_tau = batch_tau[np.isfinite(batch_tau)]
    mean_batch = batch_envelope.mean(axis=0)
    sem = batch_envelope.std(axis=0, ddof=1) / np.sqrt(N_BATCHES)

    batch_size = n_realizations // N_BATCHES
    used = batch_size * N_BATCHES
    output_dir = Path(__file__).parent / "outputs" / "real_field_g1_monte_carlo"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Real-field phase realizations used: {used:,}")
    print(f"Independent batches: {N_BATCHES} x {batch_size:,}")
    print("Varied: random mode phases; fixed: mode frequencies and amplitudes")
    print(f"MC g1 calculation time: {elapsed_s:.3f} s")
    print(f"MC time per realization: {elapsed_s / used * 1e3:.5f} ms")
    print(f"Analytic real-field correlation-envelope tau(1/e): {exact_tau:.3f} s")
    print(f"MC real-field correlation-envelope tau(1/e): {mc_tau:.3f} s")
    if batch_tau.size:
        q16, median, q84 = np.percentile(batch_tau, [16, 50, 84])
        print(f"Batch crossings, median: {median:.3f} s")
        print(f"Batch crossings, 16th-84th percentile: {q16:.3f}-{q84:.3f} s")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(
        tau_s,
        np.maximum(0.0, mean_batch - 1.96 * sem),
        mean_batch + 1.96 * sem,
        color="tab:blue",
        alpha=0.18,
        label="95% batch-mean interval",
    )
    ax.plot(tau_s, mean_batch, color="tab:blue", lw=1.4, label="MC mean from real field")
    ax.plot(tau_s, exact, "--", color="black", lw=1.6, label="Analytic mode sum")
    ax.axhline(1.0 / np.e, color="tab:red", linestyle="--", label=r"$1/e$ threshold")
    if exact_tau is not None:
        ax.axvline(exact_tau, color="black", linestyle=":", label=f"Analytic: {exact_tau:.0f} s")
    if mc_tau is not None:
        ax.axvline(mc_tau, color="tab:blue", linestyle=":", label=f"MC: {mc_tau:.0f} s")
    ax.set_xlabel(r"Delay $\tau$ (s)")
    ax.set_ylabel(r"Envelope of real-field autocorrelation $|g^{(1)}(\tau)|$")
    ax.set_title(f"Carrier-demodulated real axion field ({used:,} MC phase draws)")
    ax.set_xlim(0.0, MAX_DELAY_S)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "real_axion_g1_mc_vs_analytic.png"
    fig.savefig(path, dpi=180)
    print(f"Saved plot: {path}")

    hist_fig, hist_ax = plt.subplots(figsize=(8, 5))
    hist_ax.hist(
        batch_tau,
        bins=20,
        color="tab:blue",
        alpha=0.75,
        edgecolor="white",
        label="1000 phase draws per batch",
    )
    hist_ax.axvline(
        exact_tau,
        color="black",
        linestyle="--",
        lw=2,
        label=f"Analytic: {exact_tau:.1f} s",
    )
    hist_ax.axvline(
        mc_tau,
        color="tab:orange",
        linestyle=":",
        lw=2,
        label=f"Full MC: {mc_tau:.1f} s",
    )
    hist_ax.set_xlabel(r"First $1/e$ crossing of real-field $|g^{(1)}|$ (s)")
    hist_ax.set_ylabel("Number of batches")
    hist_ax.set_title("Monte Carlo batch coherence-time estimates")
    hist_ax.grid(True, axis="y", alpha=0.25)
    hist_ax.legend()
    hist_fig.tight_layout()
    hist_path = output_dir / "real_axion_g1_coherence_time_histogram.png"
    hist_fig.savefig(hist_path, dpi=180)
    print(f"Saved histogram: {hist_path}")

    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-realizations", type=int, default=N_REALIZATIONS
    )
    main(parser.parse_args().n_realizations)
