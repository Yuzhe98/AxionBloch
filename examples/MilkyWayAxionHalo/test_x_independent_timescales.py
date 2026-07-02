"""Fit x while varying tau_a, T2star, and T2 independently.

This scan complements ``test_signal_formula.py``.  Frequency controls the
Milky-Way axion coherence time, while T2star and intrinsic T2 are specified
directly in seconds instead of being derived from fixed dimensionless ratios.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as unit

from axionbloch.MilkyWayAxionHalo import MilkyWayAxionHalo
from test_signal_formula import (
    add_global_model,
    analyze_case,
    fit_global_x_model,
    frequency_tag,
    make_case,
    plot_frequency_curves,
    run_case,
    save_csv,
    x_ratio_model,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frequencies",
        nargs="+",
        type=float,
        default=[1e3, 1e5, 1e7],
        metavar="HZ",
    )
    parser.add_argument(
        "--T2star-seconds",
        nargs="+",
        type=float,
        default=[0.1, 1.0, 10.0],
        metavar="S",
    )
    parser.add_argument(
        "--T2-seconds",
        nargs="+",
        type=float,
        default=[0.3, 3.0, 30.0],
        metavar="S",
    )
    parser.add_argument("--num-fields", type=int, default=1000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Fields simulated simultaneously (statistics are combined exactly).",
    )
    parser.add_argument(
        "--duration-factor",
        type=float,
        default=None,
        help="Optional duration/T2star override.",
    )
    parser.add_argument(
        "--points-per-short-time",
        type=float,
        default=30.0,
        help="Samples per min(tau_a, T2star).",
    )
    parser.add_argument("--g-ann", type=float, default=1e-11)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=SCRIPT_DIR / "x-independent-timescales",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if any(value <= 0 for value in args.frequencies):
        parser.error("Frequencies must be positive.")
    if any(value <= 0 for value in args.T2star_seconds):
        parser.error("T2star values must be positive.")
    if any(value <= 0 for value in args.T2_seconds):
        parser.error("T2 values must be positive.")
    if args.num_fields < 1000:
        parser.error("--num-fields must be at least 1000.")
    if args.batch_size < 1 or args.batch_size > args.num_fields:
        parser.error("--batch-size must be between 1 and --num-fields.")
    if args.duration_factor is not None and args.duration_factor <= 0:
        parser.error("--duration-factor must be positive.")
    if args.points_per_short_time <= 0:
        parser.error("--points-per-short-time must be positive.")
    return args


def run_batched_case(
    params,
    field_mode: str,
    total_fields: int,
    batch_size: int,
    seed: int,
    verbose: bool,
):
    """Run field batches and exactly combine moments of M_perp squared."""
    sum_mean_q = None
    sum_second_q = None
    completed = 0
    aggregate_entry = None

    while completed < total_fields:
        current_size = min(batch_size, total_fields - completed)
        batch_params = dict(params)
        batch_params["numFields"] = current_size
        batch_params["rand_seed"] = seed + completed
        simulations = run_case(batch_params, field_mode, verbose)
        entry = simulations.pool[0]
        simu = entry.simu
        mean_q = np.asarray(simu.Mxy_rms) ** 2
        std_q = np.asarray(simu.Mxy_rss) ** 2
        second_q = std_q**2 + mean_q**2

        if sum_mean_q is None:
            sum_mean_q = current_size * mean_q
            sum_second_q = current_size * second_q
            aggregate_entry = entry
        else:
            sum_mean_q += current_size * mean_q
            sum_second_q += current_size * second_q

        completed += current_size
        simu.cleanup()
        del simulations
        gc.collect()

    mean_q = sum_mean_q / total_fields
    variance_q = np.maximum(
        sum_second_q / total_fields - mean_q**2,
        0,
    )
    aggregate_entry.simu.Mxy_rms = np.sqrt(mean_q)
    # analyze_case expects Mxy_rss**2 = std(Mxy**2).
    aggregate_entry.simu.Mxy_rss = variance_q**0.25
    aggregate_entry.params["numFields"] = total_fields
    return aggregate_entry


def plot_summary(
    results: list[dict[str, object]],
    global_parameters: np.ndarray,
    path: Path,
) -> plt.Figure:
    fig, (x_axis, chi2_axis) = plt.subplots(
        1, 2, figsize=(11, 4.3), constrained_layout=True
    )
    T2star_values = sorted(
        {round(float(result["T2star_s"]), 9) for result in results}
    )
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(T2star_values)))

    for T2star, color in zip(T2star_values, colors):
        subset = [
            result
            for result in results
            if np.isclose(result["T2star_s"], T2star)
        ]
        x_axis.scatter(
            [result["ratio"] for result in subset],
            [result["x_fit"] for result in subset],
            color=color,
            s=35,
            label=rf"$T_2^*={T2star:g}$ s",
        )
        chi2_axis.scatter(
            [result["ratio"] for result in subset],
            [result["reduced_chi2_fit"] for result in subset],
            color=color,
            s=35,
            label=rf"$T_2^*={T2star:g}$ s",
        )

    ratio_grid = np.logspace(
        np.log10(min(result["ratio"] for result in results)),
        np.log10(max(result["ratio"] for result in results)),
        500,
    )
    x_axis.plot(
        ratio_grid,
        x_ratio_model(ratio_grid, global_parameters),
        color="black",
        linewidth=1.8,
        label="bounded global fit",
    )
    x_axis.plot(
        ratio_grid,
        0.5 + 0.5 * ratio_grid / (1 + ratio_grid),
        color="tab:orange",
        linestyle="--",
        label="original interpolation",
    )
    x_axis.set_xscale("log")
    x_axis.set_ylim(0.48, 1.02)
    x_axis.set_xlabel(r"$\tau_a/T_2^*$")
    x_axis.set_ylabel(r"$x$")
    x_axis.grid(alpha=0.25)
    x_axis.legend(fontsize=8)

    chi2_axis.set_xscale("log")
    chi2_axis.set_yscale("log")
    chi2_axis.set_xlabel(r"$\tau_a/T_2^*$")
    chi2_axis.set_ylabel(r"diagonal reduced $\chi^2$ (fitted $x$)")
    chi2_axis.grid(alpha=0.25)
    chi2_axis.legend(fontsize=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, facecolor="white")
    return fig


def main() -> None:
    args = parse_args()
    valid_pairs = [
        (T2star_s, T2_s)
        for T2star_s in args.T2star_seconds
        for T2_s in args.T2_seconds
        if T2_s > T2star_s
    ]
    total_cases = len(args.frequencies) * len(valid_pairs)
    results = []
    case_number = 0

    for frequency_Hz in args.frequencies:
        axion_probe = MilkyWayAxionHalo(
            nu_a=frequency_Hz * unit.Hz,
            g_aNN=args.g_ann / unit.GeV,
        )
        tau_a_s = axion_probe.tau_a_est.to_value(unit.s)
        for T2star_s, T2_s in valid_pairs:
            case_number += 1
            ratio = tau_a_s / T2star_s
            T2_over_T2star = T2_s / T2star_s
            field_mode = "coherent" if ratio >= 3 else "stochastic"
            duration_factor = (
                args.duration_factor
                if args.duration_factor is not None
                else (5.0 if field_mode == "coherent" else 3.0)
            )
            print(
                f"[{case_number}/{total_cases}] nu_a={frequency_Hz:.4g} Hz, "
                f"tau_a={tau_a_s:.4g} s, T2star={T2star_s:g} s, "
                f"T2={T2_s:g} s, mode={field_mode}",
                flush=True,
            )
            params = make_case(
                frequency_Hz * unit.Hz,
                ratio,
                T2_over_T2star,
                min(args.batch_size, args.num_fields),
                duration_factor,
                args.points_per_short_time,
                args.g_ann,
                1.0,
                field_mode,
                args.seed + case_number - 1,
            )
            entry = run_batched_case(
                params,
                field_mode,
                args.num_fields,
                args.batch_size,
                args.seed + 10_000 * case_number,
                args.verbose,
            )
            result = analyze_case(entry)
            results.append(result)
            entry.simu.cleanup()
            del entry
            gc.collect()

    global_parameters = fit_global_x_model(results)
    add_global_model(results, global_parameters)
    log10_r0, power = global_parameters
    print(
        "\nBounded global fit:\n"
        "x(r) = 0.5 + 0.5 / (1 + (r0/r)^p)\n"
        f"r0={10**log10_r0:.6g}, p={power:.6g}",
        flush=True,
    )

    prefix = args.output_prefix
    csv_path = prefix.parent / f"{prefix.name}.csv"
    summary_path = prefix.parent / f"{prefix.name}-summary.png"
    save_csv(results, csv_path)
    figures = [plot_summary(results, global_parameters, summary_path)]
    curve_paths = []
    for frequency_Hz in args.frequencies:
        subset = [
            result
            for result in results
            if np.isclose(result["frequency_Hz"], frequency_Hz)
        ]
        path = prefix.parent / (
            f"{prefix.name}-curves-{frequency_tag(frequency_Hz)}.png"
        )
        figures.append(plot_frequency_curves(subset, path))
        curve_paths.append(path)

    print(f"Saved results: {csv_path}")
    print(f"Saved summary: {summary_path}")
    for path in curve_paths:
        print(f"Saved curves:  {path}")
    for figure in figures:
        plt.close(figure)


if __name__ == "__main__":
    main()
