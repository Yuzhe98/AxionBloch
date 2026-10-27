"""Monte Carlo phase study for discrete Earth-bound axion eigenstates.

This companion script leaves the single-realization example unchanged. It
keeps that example's state amplitudes fixed, redraws only the state phases,
and computes a finite-mode, time-averaged g2 curve and first 1/e-excess
coherence time for each phase configuration.

Run from the repository root with:
    python examples/EarthBoundAxionHalo/coherence_time/mc_phase_coherence.py
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
    exact_time_averaged_g2,
    first_excess_efold_time,
)


N_MONTE_CARLO = 500
PHASE_RANDOM_SEED = 314159
MAX_PLOT_DELAY_S = 20_000.0
N_CURVES_TO_SHOW = 30


def main(n_monte_carlo: int = N_MONTE_CARLO) -> None:
    # Preserve the amplitudes used by the original reproducible example.
    reference_coefficients = draw_random_coefficients(
        len(OFFSETS_HZ), RANDOM_SEED
    )
    amplitudes = np.abs(reference_coefficients)
    original_phases = np.angle(reference_coefficients)

    max_beat_hz = np.ptp(OFFSETS_HZ)
    dt_s = 1.0 / (SAMPLES_PER_FASTEST_BEAT * max_beat_hz)
    # Keep the original 40 s delay resolution, but compute only the first
    # 20,000 s, which comfortably covers the first crossing for this spectrum.
    n_lags = int(MAX_PLOT_DELAY_S / dt_s) + 1
    tau_s = np.arange(n_lags) * dt_s
    tau_coh_values = np.empty(n_monte_carlo)
    all_curves = np.empty((n_monte_carlo, tau_s.size))

    rng = np.random.default_rng(PHASE_RANDOM_SEED)
    configurations = rng.uniform(
        0.0, 2.0 * np.pi, size=(n_monte_carlo, len(OFFSETS_HZ))
    )

    simulation_start = time.perf_counter()
    for i, phases in enumerate(configurations):
        coefficients = amplitudes * np.exp(1j * phases)
        g2 = exact_time_averaged_g2(OFFSETS_HZ, coefficients, tau_s)
        tau_coh = first_excess_efold_time(tau_s, g2)
        tau_coh_values[i] = np.nan if tau_coh is None else tau_coh

        excess = g2 - 1.0
        all_curves[i] = excess / excess[0] if excess[0] > 0 else np.nan
    simulation_elapsed_s = time.perf_counter() - simulation_start

    valid = tau_coh_values[np.isfinite(tau_coh_values)]
    if valid.size == 0:
        raise RuntimeError(
            "No phase realization crossed the 1/e excess threshold. "
            "Increase SIMULATION_DURATION_S in the parent example."
        )

    # Reference result for the original fixed phase realization.
    reference_g2 = exact_time_averaged_g2(
        OFFSETS_HZ, reference_coefficients, tau_s
    )
    original_tau_coh = first_excess_efold_time(tau_s, reference_g2)

    q16, median, q84 = np.percentile(valid, [16, 50, 84])
    longest_index = int(np.nanargmax(tau_coh_values))
    longest_phases = configurations[longest_index]
    longest_coefficients = amplitudes * np.exp(1j * longest_phases)
    longest_g2 = exact_time_averaged_g2(
        OFFSETS_HZ, longest_coefficients, tau_s
    )
    longest_threshold = 1.0 + (longest_g2[0] - 1.0) / np.e
    # Find and retain the first bracket explicitly by scanning from tau=0.
    longest_cross_index = next(
        i for i in range(1, len(tau_s)) if longest_g2[i] <= longest_threshold
    )
    longest_crossing_is_first = bool(
        np.all(longest_g2[:longest_cross_index] > longest_threshold)
    )
    print(f"Monte Carlo realizations: {n_monte_carlo}")
    print(f"Time for g2 curves + threshold scan: {simulation_elapsed_s:.3f} s")
    print(
        f"Average per realization: "
        f"{1000 * simulation_elapsed_s / n_monte_carlo:.3f} ms"
    )
    print("State amplitudes: fixed to the original seeded example")
    print("State phases: independent uniform draws on [0, 2 pi)")
    print("Coherence definition: first 1/e crossing of g2(tau)-1")
    print(f"Valid crossings: {valid.size}/{n_monte_carlo}")
    print(f"Median coherence time: {median:.3f} s")
    print(f"16th-84th percentile: {q16:.3f} s to {q84:.3f} s")
    print(f"Minimum-maximum: {valid.min():.3f} s to {valid.max():.3f} s")
    print(f"Original fixed-phase example: {original_tau_coh:.3f} s")
    print("Longest-time sample first-crossing check:")
    print(f"  g2(0) = {longest_g2[0]:.9f}")
    print(f"  threshold = 1 + (g2(0)-1)/e = {longest_threshold:.9f}")
    print(
        f"  bracket: ({tau_s[longest_cross_index - 1]:.0f} s, "
        f"{longest_g2[longest_cross_index - 1]:.9f}) to "
        f"({tau_s[longest_cross_index]:.0f} s, "
        f"{longest_g2[longest_cross_index]:.9f})"
    )
    print(f"  all earlier samples above threshold: {longest_crossing_is_first}")

    fig, (curve_ax, hist_ax) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.45, 1]}
    )
    plot_mask = tau_s <= MAX_PLOT_DELAY_S
    for curve in all_curves[:N_CURVES_TO_SHOW]:
        curve_ax.plot(
            tau_s[plot_mask], curve[plot_mask], color="tab:blue", alpha=0.15, lw=0.8
        )

    # Show the median across all configurations with a quantile band. Curves
    # are already normalized so every realization starts at one.
    p16, p50, p84 = np.nanpercentile(all_curves[:, plot_mask], [16, 50, 84], axis=0)
    curve_ax.fill_between(
        tau_s[plot_mask], p16, p84, color="tab:orange", alpha=0.25, label="16–84%"
    )
    curve_ax.plot(
        tau_s[plot_mask], p50, color="tab:orange", lw=2, label="Median normalized excess"
    )
    longest_normalized_excess = all_curves[longest_index]
    curve_ax.plot(
        tau_s[plot_mask],
        longest_normalized_excess[plot_mask],
        color="tab:green",
        lw=2,
        label=f"Longest sample: {tau_coh_values[longest_index]:.0f} s",
    )
    curve_ax.axhline(1.0 / np.e, color="black", linestyle="--", label=r"$1/e$ threshold")
    if original_tau_coh is not None:
        curve_ax.axvline(
            original_tau_coh,
            color="tab:red",
            linestyle=":",
            label=f"Original phases: {original_tau_coh:.0f} s",
        )
    curve_ax.set_xlabel(r"Delay $\tau$ (s)")
    curve_ax.set_ylabel(r"$(g^{(2)}(\tau)-1)/(g^{(2)}(0)-1)$")
    curve_ax.set_title("Phase-dependent g2 decay")
    curve_ax.set_ylim(-0.25, 1.1)
    curve_ax.grid(True, alpha=0.25)
    curve_ax.legend(fontsize=8)

    hist_ax.hist(valid, bins=30, color="tab:blue", alpha=0.75, edgecolor="white")
    hist_ax.axvline(median, color="tab:orange", lw=2, label=f"Median: {median:.0f} s")
    hist_ax.axvline(
        tau_coh_values[longest_index],
        color="tab:green",
        linestyle="--",
        lw=2,
        label=f"Longest: {tau_coh_values[longest_index]:.0f} s",
    )
    if original_tau_coh is not None:
        hist_ax.axvline(
            original_tau_coh,
            color="tab:red",
            linestyle=":",
            lw=2,
            label=f"Original phases: {original_tau_coh:.0f} s",
        )
    hist_ax.set_xlabel(r"First $1/e$ coherence time (s)")
    hist_ax.set_ylabel("Number of phase configurations")
    hist_ax.set_title("Monte Carlo distribution")
    hist_ax.grid(True, axis="y", alpha=0.25)
    hist_ax.legend(fontsize=8)

    fig.suptitle(
        f"Random phases, fixed state amplitudes (N={n_monte_carlo})", y=1.02
    )
    fig.tight_layout()
    suffix = "" if n_monte_carlo == N_MONTE_CARLO else f"_{n_monte_carlo}"
    output_path = Path(__file__).with_name(f"mc_phase_coherence_results{suffix}.png")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved plot: {output_path}")

    # Inspect the actual g2 threshold bracket for the longest realization.
    check_fig, check_ax = plt.subplots(figsize=(8.5, 5))
    check_ax.plot(tau_s, longest_g2, color="tab:blue", lw=1.6, label="Longest-time sample")
    check_ax.axhline(
        longest_threshold,
        color="tab:red",
        linestyle="--",
        label=rf"$1/e$ excess threshold = {longest_threshold:.4f}",
    )
    check_ax.axhline(1.0, color="black", lw=0.8, alpha=0.6, label=r"$g^{(2)}=1$")
    check_ax.axvline(
        tau_coh_values[longest_index],
        color="tab:green",
        linestyle=":",
        label=f"First crossing = {tau_coh_values[longest_index]:.1f} s",
    )
    check_ax.scatter(
        tau_s[longest_cross_index - 1 : longest_cross_index + 1],
        longest_g2[longest_cross_index - 1 : longest_cross_index + 1],
        color="tab:green",
        zorder=4,
    )
    check_ax.set_xlim(0.0, min(10_000.0, tau_s[-1]))
    check_ax.set_xlabel(r"Delay $\tau$ (s)")
    check_ax.set_ylabel(r"$g^{(2)}(\tau)$")
    check_ax.set_title("Left-to-right first-crossing check for longest MC result")
    check_ax.grid(True, alpha=0.25)
    check_ax.legend()
    check_fig.tight_layout()
    check_path = Path(__file__).with_name(
        f"mc_phase_longest_first_crossing_check{suffix}.png"
    )
    check_fig.savefig(check_path, dpi=180)
    print(f"Saved crossing-check plot: {check_path}")
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-mc",
        type=int,
        default=N_MONTE_CARLO,
        help=f"number of random phase realizations (default: {N_MONTE_CARLO})",
    )
    args = parser.parse_args()
    main(args.n_mc)
