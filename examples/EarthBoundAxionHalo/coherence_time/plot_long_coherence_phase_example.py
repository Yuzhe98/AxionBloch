"""Plot one Monte Carlo phase draw with a long first g2 crossing.

This reproduces realization 105 from mc_phase_coherence.py (zero-based index),
whose first 1/e-excess crossing is about 5.73 ks. It writes a separate plot and
does not modify the original single-realization or Monte Carlo figures.
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
)
from mc_phase_coherence import PHASE_RANDOM_SEED, N_MONTE_CARLO


PHASE_REALIZATION_INDEX = 105
DELAY_MAX_S = 20_000.0


def main() -> None:
    amplitudes = np.abs(
        draw_random_coefficients(len(OFFSETS_HZ), RANDOM_SEED)
    )
    rng = np.random.default_rng(PHASE_RANDOM_SEED)
    phase_draws = rng.uniform(
        0.0, 2.0 * np.pi, size=(N_MONTE_CARLO, len(OFFSETS_HZ))
    )
    phases = phase_draws[PHASE_REALIZATION_INDEX]
    coefficients = amplitudes * np.exp(1j * phases)

    dt_s = 1.0 / (SAMPLES_PER_FASTEST_BEAT * np.ptp(OFFSETS_HZ))
    tau_s = np.arange(int(DELAY_MAX_S / dt_s) + 1) * dt_s
    g2 = exact_time_averaged_g2(OFFSETS_HZ, coefficients, tau_s)
    tau_coh_s = first_excess_efold_time(tau_s, g2)
    if tau_coh_s is None:
        raise RuntimeError("This phase realization did not cross the threshold.")

    threshold = 1.0 + (g2[0] - 1.0) / np.e
    i_cross = next(i for i in range(1, len(g2)) if g2[i] <= threshold)

    # Slow complex envelope of the mode sum. A common frequency shift changes
    # only its phase, not |A(t)|, so offsets are sufficient for this panel.
    time_s = np.arange(0.0, DELAY_MAX_S + dt_s, dt_s)
    envelope = np.zeros(time_s.size, dtype=complex)
    for offset_hz, coefficient in zip(OFFSETS_HZ, coefficients):
        envelope += coefficient * np.exp(-2j * np.pi * offset_hz * time_s)

    # Resolve a short physical real-field segment at the 1.3 MHz carrier.
    field_dt_s = 1.0 / (20.0 * COMPTON_FREQUENCY_HZ)
    field_time_s = np.arange(0.0, 10e-6, field_dt_s)
    complex_field = np.zeros(field_time_s.size, dtype=complex)
    for frequency_hz, coefficient in zip(EIGENFREQUENCIES_HZ, coefficients):
        complex_field += coefficient * np.exp(
            -2j * np.pi * frequency_hz * field_time_s
        )
    axion_field = np.real(complex_field) / np.sqrt(2.0)

    fig, (envelope_ax, field_ax, g2_ax) = plt.subplots(
        3,
        1,
        figsize=(10, 10),
        gridspec_kw={"height_ratios": [1, 0.8, 1.2]},
    )
    envelope_ax.plot(time_s, np.abs(envelope), color="tab:green", lw=1.4)
    envelope_ax.set_xlabel("Simulation time (s)")
    envelope_ax.set_ylabel(r"Field envelope $|A(t)|$ (arb. units)")
    envelope_ax.set_title("Slow envelope for Monte Carlo phase realization 105")
    envelope_ax.grid(True, alpha=0.25)

    field_ax.plot(field_time_s * 1e6, axion_field, color="tab:blue", lw=1.0)
    field_ax.set_xlabel("Time (microseconds)")
    field_ax.set_ylabel(r"$a(t)$ (arb. units)")
    field_ax.set_title(r"Real field $a(t)=\mathrm{Re}[A(t)]/\sqrt{2}$")
    field_ax.grid(True, alpha=0.25)

    g2_ax.plot(tau_s, g2, color="tab:purple", lw=1.5)
    g2_ax.axhline(
        threshold,
        color="tab:red",
        linestyle="--",
        label=rf"$1/e$ excess threshold = {threshold:.4f}",
    )
    g2_ax.axhline(1.0, color="black", lw=0.8, alpha=0.6)
    g2_ax.axvline(
        tau_coh_s,
        color="tab:green",
        linestyle=":",
        label=rf"First crossing $= {tau_coh_s:.1f}$ s",
    )
    g2_ax.scatter(
        tau_s[i_cross - 1 : i_cross + 1],
        g2[i_cross - 1 : i_cross + 1],
        color="tab:green",
        zorder=4,
    )
    g2_ax.set_xlabel(r"Delay $\tau$ (s)")
    g2_ax.set_ylabel(r"$g^{(2)}(\tau)$")
    g2_ax.set_title("First left-to-right 1/e crossing of the excess correlation")
    g2_ax.grid(True, alpha=0.25)
    g2_ax.legend()
    fig.tight_layout()

    output_path = Path(__file__).with_name("phase_example_5731s_envelope_and_g2.png")
    fig.savefig(output_path, dpi=180)
    print(f"Phase realization index: {PHASE_REALIZATION_INDEX}")
    print(f"g2(0): {g2[0]:.9f}")
    print(f"1/e excess threshold: {threshold:.9f}")
    print(f"First-crossing coherence time: {tau_coh_s:.3f} s")
    print(f"Saved plot: {output_path}")
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
