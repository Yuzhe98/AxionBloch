"""Make one combined plot of a(t), its envelope, g1, and g2.

The example uses Monte Carlo phase realization 105, with the original seeded
state magnitudes. The plotted g1 is the envelope of the real-field
autocorrelation, with the fast Compton carrier removed; g2 is the
cycle-averaged intensity correlation of a(t).
Output is placed in outputs/g1_monte_carlo/ without replacing earlier plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from g2_coherence_time_discrete_states import (
    COMPTON_FREQUENCY_HZ,
    EIGENFREQUENCIES_HZ,
    OFFSETS_HZ,
    RANDOM_SEED,
    SAMPLES_PER_FASTEST_BEAT,
    draw_random_coefficients,
    exact_time_averaged_g2,
    first_excess_efold_time,
    first_threshold_time,
)
from mc_phase_coherence import PHASE_RANDOM_SEED


MAX_DELAY_S = 20_000.0
FIELD_PLOT_WINDOW_S = 10e-6
PHASE_REALIZATION_INDEX = 105


def main() -> None:
    reference_coefficients = draw_random_coefficients(
        len(OFFSETS_HZ), RANDOM_SEED
    )
    rng = np.random.default_rng(PHASE_RANDOM_SEED)
    phase_draws = rng.uniform(
        0.0,
        2.0 * np.pi,
        size=(PHASE_REALIZATION_INDEX + 1, len(OFFSETS_HZ)),
    )
    coefficients = np.abs(reference_coefficients) * np.exp(
        1j * phase_draws[PHASE_REALIZATION_INDEX]
    )
    weights = np.abs(coefficients) ** 2

    # Resolve the real axion oscillation at the 1.3 MHz carrier.
    field_dt_s = 1.0 / (20.0 * COMPTON_FREQUENCY_HZ)
    field_time_s = np.arange(0.0, FIELD_PLOT_WINDOW_S, field_dt_s)
    field_complex = np.zeros(field_time_s.size, dtype=complex)
    for frequency_hz, coefficient in zip(EIGENFREQUENCIES_HZ, coefficients):
        field_complex += coefficient * np.exp(
            -2j * np.pi * frequency_hz * field_time_s
        )
    axion_field = np.real(field_complex) / np.sqrt(2.0)

    # Plot the slow positive/mirrored envelope of the real field. A common
    # carrier phase does not affect the envelope, so the mHz offsets suffice.
    dt_s = 1.0 / (SAMPLES_PER_FASTEST_BEAT * np.ptp(OFFSETS_HZ))
    envelope_time_s = np.arange(0.0, MAX_DELAY_S + dt_s, dt_s)
    envelope_complex = np.zeros(envelope_time_s.size, dtype=complex)
    for offset_hz, coefficient in zip(OFFSETS_HZ, coefficients):
        envelope_complex += coefficient * np.exp(
            -2j * np.pi * offset_hz * envelope_time_s
        )
    real_field_envelope = np.abs(envelope_complex) / np.sqrt(2.0)

    # Correlation curves on the same slow delay grid.
    tau_s = envelope_time_s
    g1_complex = (
        weights[:, None]
        * np.exp(-2j * np.pi * OFFSETS_HZ[:, None] * tau_s[None, :])
    ).sum(axis=0) / weights.sum()
    g1 = np.abs(g1_complex)
    g2 = exact_time_averaged_g2(OFFSETS_HZ, coefficients, tau_s)
    tau_g1 = first_threshold_time(tau_s, g1, 1.0 / np.e)
    tau_g2 = first_excess_efold_time(tau_s, g2)
    g2_threshold = 1.0 + (g2[0] - 1.0) / np.e

    fig, (field_ax, envelope_ax, corr_ax) = plt.subplots(
        3,
        1,
        figsize=(11, 10),
        gridspec_kw={"height_ratios": [0.85, 1.15, 1.25]},
    )

    field_ax.plot(field_time_s * 1e6, axion_field, color="tab:blue", lw=1.1)
    field_ax.set_xlabel("Time (microseconds)")
    field_ax.set_ylabel(r"$a(t)$ (arb. units)")
    field_ax.set_title(r"Real axion oscillation $a(t)=\mathrm{Re}[A(t)]/\sqrt{2}$")
    field_ax.grid(True, alpha=0.25)

    envelope_ax.fill_between(
        envelope_time_s,
        -real_field_envelope,
        real_field_envelope,
        color="tab:green",
        alpha=0.22,
        label="Field between mirrored envelopes",
    )
    envelope_ax.plot(
        envelope_time_s,
        real_field_envelope,
        color="tab:green",
        lw=1.5,
        label="Positive envelope",
    )
    envelope_ax.plot(
        envelope_time_s,
        -real_field_envelope,
        color="tab:green",
        lw=1.5,
        linestyle="--",
        label="Mirrored envelope",
    )
    envelope_ax.set_xlabel("Time (s)")
    envelope_ax.set_ylabel(r"Real-field envelope (arb. units)")
    envelope_ax.set_title("Slow envelope from the mHz-spaced eigenfrequencies")
    envelope_ax.grid(True, alpha=0.25)
    envelope_ax.legend(loc="upper right")

    g1_line, = corr_ax.plot(tau_s, g1, color="tab:blue", lw=1.5, label=r"$|g^{(1)}(\tau)|$")
    corr_ax.axhline(
        1.0 / np.e,
        color="tab:blue",
        linestyle="--",
        alpha=0.75,
        label=r"$g^{(1)}$ threshold $1/e$",
    )
    if tau_g1 is not None:
        corr_ax.axvline(tau_g1, color="tab:blue", linestyle=":", alpha=0.9)

    g2_ax = corr_ax.twinx()
    g2_line, = g2_ax.plot(
        tau_s, g2, color="tab:orange", lw=1.4, label=r"$g^{(2)}(\tau)$"
    )
    g2_ax.axhline(
        g2_threshold,
        color="tab:orange",
        linestyle="--",
        alpha=0.75,
        label=r"$g^{(2)}$ 1/e excess threshold",
    )
    if tau_g2 is not None:
        g2_ax.axvline(tau_g2, color="tab:orange", linestyle=":", alpha=0.9)

    corr_ax.set_xlabel(r"Delay $\tau$ (s)")
    corr_ax.set_ylabel(r"$|g^{(1)}(\tau)|$", color="tab:blue")
    g2_ax.set_ylabel(r"$g^{(2)}(\tau)$", color="tab:orange")
    corr_ax.tick_params(axis="y", labelcolor="tab:blue")
    g2_ax.tick_params(axis="y", labelcolor="tab:orange")
    corr_ax.set_title(
        "Correlation curves: "
        f"first-crossing times g1={tau_g1:.0f} s, g2={tau_g2:.0f} s"
    )
    corr_ax.grid(True, alpha=0.25)
    handles = [g1_line, g2_line]
    labels = [line.get_label() for line in handles]
    for axis in (corr_ax, g2_ax):
        extra_handles, extra_labels = axis.get_legend_handles_labels()
        for handle, label in zip(extra_handles, extra_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    corr_ax.legend(handles, labels, loc="upper right", ncol=2, fontsize=8)

    fig.suptitle(
        f"One Monte Carlo phase example (realization {PHASE_REALIZATION_INDEX}; "
        f"g2 coherence {tau_g2:.0f} s)",
        y=0.995,
    )
    fig.tight_layout()
    output_dir = Path(__file__).parent / "outputs" / "g1_monte_carlo"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "single_example_field_envelope_g1_g2.png"
    fig.savefig(output_path, dpi=180)
    print(f"Monte Carlo phase realization: {PHASE_REALIZATION_INDEX}")
    print(f"g1 first 1/e crossing: {tau_g1:.3f} s")
    print(f"g2 first 1/e-excess crossing: {tau_g2:.3f} s")
    print(f"Saved plot: {output_path}")
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
