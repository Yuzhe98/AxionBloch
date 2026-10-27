"""Five-hour Rabi-envelope coherence example for a uniform nlm mixture.

Run from the repository root with:
    python examples/EarthBoundAxionHalo/coherence_time/rabi_g1_uniform_states_5h.py

At one selected point in the 72-hour Mainz window, each mode's Lorentz-boosted
Rabi-gradient phasor is frozen. Random initial phases and the bound-state
frequency offsets then evolve the complex Rabi envelope for five hours.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as unit
from astropy.time import Time

from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo
from axionbloch.Station import Mainz


STATE_NAMES = ["1s", "2p", "2s", "3d", "3p", "4f"]
RABI_SAMPLE_HOUR = 12.0  # Must be between 0 and 72 h, measured from t0 below.
SIMULATION_DURATION_S = 5.0 * 3600.0
RANDOM_PHASE_SEED = 20261027
SAMPLES_PER_FASTEST_BEAT = 100
FIELD_PLOT_WINDOW_S = 10e-6

NU_A = 1.348575 * unit.MHz
T0 = Time(datetime(2022, 12, 13, 7, 0, tzinfo=timezone(timedelta(hours=1))))
SHELL_MASS_LIMIT = 4e-9 * unit.M_earth
SHELL_RADII = [12300 * unit.km, 384000 * unit.km]
MAX_EARTH_ENCLOSED_MASS = 0.01 * unit.M_earth  # inferHaloMass default


def first_threshold_time(tau_s: np.ndarray, values: np.ndarray, threshold: float):
    """Return the interpolated first downward crossing, or None."""
    crossings = np.flatnonzero(values[1:] <= threshold)
    if not crossings.size:
        return None
    i = int(crossings[0] + 1)
    x0, x1 = tau_s[i - 1], tau_s[i]
    y0, y1 = values[i - 1], values[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))


def main() -> None:
    if not 0.0 <= RABI_SAMPLE_HOUR <= 72.0:
        raise ValueError("RABI_SAMPLE_HOUR must be between 0 and 72.")

    # Match the current rabi_m_states.py frequency and grid settings.
    halo = EarthBoundAxionHalo(
        nu_a=NU_A,
        N=int(7.5e6),
        extent=1e4 * unit.R_earth,
    )
    halo.solve_TISE_3D(l_vals=[0, 1, 2, 3], max_n_r=2)

    # The six nl states have equal population; each state's population is split
    # uniformly among its 2l+1 degenerate m modes.
    l_by_state = {name: halo.states[name]["l"] for name in STATE_NAMES}
    state_population = 1.0 / len(STATE_NAMES)
    mode_population = {
        (name, m): state_population / (2 * l_by_state[name] + 1)
        for name in STATE_NAMES
        for m in range(-l_by_state[name], l_by_state[name] + 1)
    }

    # Infer the maximum total mass for this mixture from the shell and
    # Earth-enclosed limits. Single-state inferHaloMass results provide the
    # state-specific shell and enclosed probability fractions.
    mass_limits_by_state = halo.inferHaloMass(
        SHELL_MASS_LIMIT,
        SHELL_RADII,
        STATE_NAMES,
    )
    mixture_shell_fraction = sum(
        state_population * mass_limits_by_state[name]["shell_fraction"].to_value(unit.one)
        for name in STATE_NAMES
    )
    mixture_enclosed_fraction = sum(
        state_population
        * (
            mass_limits_by_state[name]["enclosed_mass"]
            / mass_limits_by_state[name]["total_mass"]
        ).to_value(unit.one)
        for name in STATE_NAMES
    )
    mass_from_shell = SHELL_MASS_LIMIT / mixture_shell_fraction
    mass_from_earth = MAX_EARTH_ENCLOSED_MASS / mixture_enclosed_fraction
    halo_mass = min(mass_from_shell, mass_from_earth, halo.maxTotalDMmass)
    halo.N_a = halo_mass / halo.m_a

    print("Standalone state limits versus equal-population mixture shares:")
    for name in STATE_NAMES:
        l_value = l_by_state[name]
        standalone_mass = mass_limits_by_state[name]["total_mass"]
        nl_share = state_population * halo_mass
        per_m_share = nl_share / (2 * l_value + 1)
        print(
            f"  {name}: standalone max={standalone_mass.to(unit.M_earth):.6g}; "
            f"mixture share={nl_share.to(unit.M_earth):.6g}; "
            f"per m={per_m_share.to(unit.M_earth):.6g}"
        )

    sample_time = T0 + RABI_SAMPLE_HOUR * unit.hour
    sample_time_local = sample_time.to_datetime(
        timezone=timezone(timedelta(hours=1))
    ).strftime("%Y-%m-%d %H:%M CET")
    omega_factor = (
        const.c
        * halo.g_aNN
        * np.sqrt(halo.N_a * const.hbar**3 * const.c / (2 * halo.m_a))
    )

    # Calculate one complex boosted tangential Rabi-gradient vector per nlm
    # mode at the frozen Mainz epoch. Radial Omega is intentionally excluded.
    # Rabi output uses cyclic frequency Omega/(2*pi), mHz.
    modes = list(mode_population)
    mode_rabi_vectors = np.empty((len(modes), 2), dtype=complex)
    frequency_offsets_hz = np.empty(len(modes), dtype=float)
    for i, (name, m) in enumerate(modes):
        _, _, r_line, grad_r, grad_theta, grad_phi = halo.findGradientsAlongDirection(
            stateCoefficients={(name, m): 1.0},
            station=Mainz,
            meas_time=sample_time,
            truncRadius=Mainz.R,
            include_lorentz_boost=True,
            showPlot=False,
        )
        station_index = int(np.argmin(np.abs(r_line - Mainz.R)))
        gradient_components = (
            grad_theta[station_index],
            grad_phi[station_index],
        )
        mode_rabi_vectors[i] = [
            (omega_factor * gradient).to_value(
                unit.mHz,
                equivalencies=unit.dimensionless_angles(),
            )
            / (2.0 * np.pi)
            for gradient in gradient_components
        ]
        frequency_offsets_hz[i] = (
            halo.states[name]["eigenE"] / const.h
        ).to_value(unit.Hz)

    rng = np.random.default_rng(RANDOM_PHASE_SEED)
    random_phases = rng.uniform(0.0, 2.0 * np.pi, size=len(modes))
    modal_coefficients = np.array(
        [np.sqrt(mode_population[mode]) for mode in modes]
    ) * np.exp(1j * random_phases)

    # Remove the common offset; it changes only the overall envelope phase.
    frequency_offsets_hz -= np.mean(frequency_offsets_hz)
    max_beat_hz = float(np.ptp(frequency_offsets_hz))
    if max_beat_hz <= 0.0:
        raise ValueError("The selected states have no frequency spread.")
    dt_s = 1.0 / (SAMPLES_PER_FASTEST_BEAT * max_beat_hz)
    time_s = np.arange(0.0, SIMULATION_DURATION_S + 0.5 * dt_s, dt_s)
    beat_phases = np.exp(
        -2j * np.pi * frequency_offsets_hz[:, None] * time_s[None, :]
    )
    envelope_components = np.einsum(
        "j,jk,jt->tk", modal_coefficients, mode_rabi_vectors, beat_phases
    )
    envelope_amplitude = np.linalg.norm(envelope_components, axis=1) / np.sqrt(2.0)

    # For g1, combine equal-frequency m modes coherently before forming their
    # spectral powers. The m substates are degenerate in this central potential.
    grouped_vectors = {}
    grouped_frequencies = {}
    for i, (name, _m) in enumerate(modes):
        grouped_vectors[name] = grouped_vectors.get(
            name, np.zeros(2, dtype=complex)
        ) + modal_coefficients[i] * mode_rabi_vectors[i]
        grouped_frequencies[name] = frequency_offsets_hz[i]
    g1_frequencies = np.array([grouped_frequencies[name] for name in STATE_NAMES])
    g1_weights = np.array(
        [np.vdot(grouped_vectors[name], grouped_vectors[name]).real for name in STATE_NAMES]
    )
    tau_s = time_s.copy()
    g1_complex = (
        g1_weights[:, None]
        * np.exp(-2j * np.pi * g1_frequencies[:, None] * tau_s[None, :])
    ).sum(axis=0) / g1_weights.sum()
    g1 = np.abs(g1_complex)
    tau_g1 = first_threshold_time(tau_s, g1, 1.0 / np.e)

    # Resolve a short carrier segment for the top panel; the five-hour
    # simulation uses only the slow envelope and does not sample the MHz carrier.
    carrier_dt_s = 1.0 / (20.0 * NU_A.to_value(unit.Hz))
    carrier_time_s = np.arange(0.0, FIELD_PLOT_WINDOW_S, carrier_dt_s)
    carrier_beats = np.exp(
        -2j * np.pi * frequency_offsets_hz[:, None] * carrier_time_s[None, :]
    )
    carrier_envelope = np.einsum(
        "j,jk,jt->tk", modal_coefficients, mode_rabi_vectors, carrier_beats
    )
    carrier = np.exp(-2j * np.pi * NU_A.to_value(unit.Hz) * carrier_time_s)
    carrier_rabi = np.real(carrier_envelope[:, 0] * carrier) / np.sqrt(2.0)

    # Exact time-averaged vector g1 from the finite mode spectrum.
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 8,
        }
    )
    fig, (field_ax, envelope_ax, corr_ax) = plt.subplots(
        3,
        1,
        figsize=(11 / 2.54, 10 / 2.54),
        dpi=300,
        gridspec_kw={"height_ratios": [0.85, 1.15, 1.25]},
    )
    field_ax.plot(carrier_time_s * 1e6, carrier_rabi, color="tab:blue", lw=1.1)
    field_ax.set_xlabel("Time (microseconds)")
    field_ax.set_ylabel(r"$\theta$ Rabi drive (mHz)")
    field_ax.set_title(r"Carrier-resolved real $\theta$ Rabi drive")
    field_ax.grid(True, alpha=0.25)

    envelope_ax.fill_between(
        time_s / 3600.0,
        -envelope_amplitude,
        envelope_amplitude,
        color="tab:green",
        alpha=0.22,
        label="Envelope bounds",
    )
    envelope_ax.plot(
        time_s / 3600.0,
        envelope_amplitude,
        color="tab:green",
        lw=1.5,
        label="Positive envelope",
    )
    envelope_ax.plot(
        time_s / 3600.0,
        -envelope_amplitude,
        color="tab:green",
        lw=1.5,
        linestyle="--",
        label="Mirrored envelope",
    )
    envelope_ax.set_xlabel("Time from frozen Rabi sample (hours)")
    envelope_ax.set_ylabel(r"$\theta$-$\phi$ Rabi envelope (mHz)")
    envelope_ax.set_title("Five-hour random-phase tangential Rabi envelope")
    envelope_ax.grid(True, alpha=0.25)
    envelope_ax.legend(loc="upper right", framealpha=0.85)

    corr_ax.plot(tau_s / 3600.0, g1, color="tab:blue", lw=1.5, label=r"$|g^{(1)}(\tau)|$")
    corr_ax.axhline(1.0 / np.e, color="tab:blue", linestyle="--", label=r"$1/e$ threshold")
    if tau_g1 is not None:
        corr_ax.axvline(tau_g1 / 3600.0, color="tab:blue", linestyle=":")
    corr_ax.set_xlabel(r"Delay $\tau$ (hours)")
    corr_ax.set_ylabel(r"$|g^{(1)}(\tau)|$")
    corr_ax.set_title(
        r"$\theta$-$\phi$ Rabi-envelope coherence: "
        + (f"first 1/e crossing = {tau_g1 / 3600.0:.3f} h" if tau_g1 is not None
           else "no 1/e crossing within five hours")
    )
    corr_ax.set_xlim(0.0, SIMULATION_DURATION_S / 3600.0)
    corr_ax.grid(True, alpha=0.25)
    corr_ax.legend(loc="upper right", framealpha=0.85)

    fig.suptitle(
        f"Tangential Rabi coherence at Mainz, frozen {sample_time_local}",
        fontsize=8,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_dir = Path(__file__).parent / "outputs" / "rabi_g1_coherence"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "uniform_nlm_rabi_g1_5h.png"
    fig.savefig(output_path, dpi=300)

    print(f"Mixture total mass used: {halo_mass.to(unit.M_earth):.6g}")
    print(f"Frozen Lorentz-boost sample epoch: {sample_time_local} ({sample_time.iso} UTC)")
    print(f"Random phase seed: {RANDOM_PHASE_SEED}; modes: {len(modes)}")
    print(f"Simulation duration: {SIMULATION_DURATION_S / 3600:.1f} h")
    if tau_g1 is None:
        print("g1 coherence time: no first 1/e crossing within the five-hour window")
    else:
        print(f"g1 coherence time (first 1/e crossing): {tau_g1:.3f} s ({tau_g1 / 3600:.4f} h)")
    print(f"Saved plot: {output_path}")
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
