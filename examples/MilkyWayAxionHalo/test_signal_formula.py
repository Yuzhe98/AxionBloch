"""Test the stochastic axion-signal buildup formula from the 2025-12-22 note.

The tested expression is

    s(t) = [1 - exp(-t / (x T2star))]**x
           * gamma * B * T2star * sqrt(tau_a / (tau_a + T2star)).

For every axion frequency and requested ``tau_a / T2star`` ratio, this script
chooses the magnet linewidth that gives the desired T2star, runs an ensemble
of stochastic Bloch simulations, and compares the RMS transverse
magnetization, ``sqrt(<Mx**2 + My**2>)``, with the formula.

The note did not specify how ``x`` changes between its limiting values:
``x = 1/2`` for ``tau_a << T2star`` and ``x = 1`` for ``tau_a >> T2star``.
The script therefore compares three choices:

1. an independent best-fit x for every simulation;
2. the original limit-preserving interpolation
   ``x = 0.5 * (1 + tau_a / (tau_a + T2star))``;
3. a bounded logistic ``x(tau_a / T2star)`` fitted across all cases, with
   the physical limits ``x(0) = 1/2`` and ``x(infinity) = 1``.

A diagonal reduced chi-square is calculated using the ensemble uncertainty of
the RMS.  It is useful for comparing models here, but adjacent time samples
are correlated, so it is not a formal goodness-of-fit probability.

Example
-------
Run the default coherent-limit scan with 1000 field realizations:

    python examples/MilkyWayAxionHalo/test_signal_formula.py

Run a smaller exploratory scan and display the plots:

    python examples/MilkyWayAxionHalo/test_signal_formula.py \
        --frequencies 1e3 1e6 --ratios 10 30 100 \
        --T2-over-T2star 1.05 3 100 --num-fields 1000 --show
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

import matplotlib
import numpy as np
from astropy import units as unit
from scipy.optimize import least_squares, minimize_scalar

# Saving a scan should not require a working GUI/Tk installation.  Supplying
# --show leaves the user's interactive backend unchanged.
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_Xe129N, mu_Xe129N
from axionbloch.dependency import PI
from axionbloch.MilkyWayAxionHalo import MilkyWayAxionHalo
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulations
from axionbloch.SimuTypes import SimuParams


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frequencies",
        nargs="+",
        type=float,
        default=[1e3, 1e5, 1e7],
        metavar="HZ",
        help="Axion frequencies in Hz (default: 1e3 1e5 1e7).",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.03, 0.1, 0.3],
        metavar="RATIO",
        help="tau_a/T2star values (default: 0.03 0.1 0.3).",
    )
    parser.add_argument(
        "--T2-over-T2star",
        nargs="+",
        type=float,
        default=[1.05, 3.0, 100.0],
        metavar="RATIO",
        help="Intrinsic T2/T2star values (default: 1.05 3 100).",
    )
    parser.add_argument(
        "--g-ann",
        type=float,
        default=1e-11,
        help="Axion-nucleon coupling in GeV^-1 (default: 1e-11).",
    )
    parser.add_argument(
        "--field-mode",
        choices=("coherent", "stochastic"),
        default="stochastic",
        help="Field generator; coherent is appropriate for T2star << tau_a.",
    )
    parser.add_argument(
        "--numerical-magnetization-scale",
        type=float,
        default=1e12,
        help="Linear numerical scale removed before reporting M/Meqb.",
    )
    parser.add_argument(
        "--num-fields",
        type=int,
        default=1000,
        help="Stochastic field realizations per case (default: 1000).",
    )
    parser.add_argument(
        "--duration-factor",
        type=float,
        default=3.0,
        help="Simulation duration in units of T2star (default: 3).",
    )
    parser.add_argument(
        "--points-per-short-time",
        type=float,
        default=30.0,
        help="Samples per min(tau_a, T2star) (default: 30).",
    )
    parser.add_argument(
        "--seed", type=int, default=10, help="Base random seed (default: 10)."
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=SCRIPT_DIR / "signal-formula-scan",
        help="Path prefix for the PNG and CSV outputs.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display the plot after saving it."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print simulation diagnostics."
    )
    args = parser.parse_args()

    if any(frequency <= 0 for frequency in args.frequencies):
        parser.error("All frequencies must be positive.")
    if any(ratio <= 0 for ratio in args.ratios):
        parser.error("All tau_a/T2star ratios must be positive.")
    if any(ratio <= 1 for ratio in args.T2_over_T2star):
        parser.error("All T2/T2star ratios must be greater than 1.")
    if args.g_ann <= 0:
        parser.error("--g-ann must be positive.")
    if args.numerical_magnetization_scale < 1:
        parser.error("--numerical-magnetization-scale must be at least 1.")
    if args.num_fields < 1000:
        parser.error("--num-fields must be at least 1000 for this RMS test.")
    if args.duration_factor <= 0 or args.points_per_short_time <= 0:
        parser.error("Duration and sampling factors must be positive.")
    return args


def signal_formula(
    time: unit.Quantity,
    gamma: unit.Quantity,
    B_rms: unit.Quantity,
    tau_a: unit.Quantity,
    T2star: unit.Quantity,
    x: float,
) -> np.ndarray:
    """Return the dimensionless transverse-signal prediction."""
    buildup = (
        1.0 - np.exp(-(time / (x * T2star)).to_value(unit.one))
    ) ** x
    saturation = (
        np.abs(gamma)
        * B_rms
        * T2star
        * np.sqrt((tau_a / (tau_a + T2star)).to_value(unit.one))
    ).to_value(unit.rad)
    return buildup * saturation


def make_case(
    frequency: unit.Quantity,
    ratio: float,
    T2_over_T2star: float,
    num_fields: int,
    duration_factor: float,
    points_per_short_time: float,
    g_ann: float,
    numerical_magnetization_scale: float,
    field_mode: str,
    seed: int,
) -> SimuParams:
    """Construct one weak-drive simulation with a prescribed tau_a/T2star."""
    axion = MilkyWayAxionHalo(
        nu_a=frequency,
        g_aNN=g_ann / unit.GeV,
        verbose=False,
    )
    tau_a = axion.tau_a_est.to(unit.s)
    target_T2star = (tau_a / ratio).to(unit.s)

    # Choose intrinsic T2 independently while retaining the requested combined
    # T2star = (1/Tdelta + 1/T2)^(-1).
    intrinsic_T2 = T2_over_T2star * target_T2star
    Tdelta = 1 / (1 / target_T2star - 1 / intrinsic_T2)
    # Start at the physical equilibrium magnetization.  A finite T1=T2
    # replenishes longitudinal polarization in both field regimes.
    init_M_scale = 1.0
    T1 = intrinsic_T2
    sample = Sample(
        name="Liquid Xe-129",
        gamma=gamma_Xe129N,
        massDensity=3.1 * unit.g / unit.cm**3,
        molarMass=131.29 * unit.g / unit.mol,
        numOfSpinsPerMolecule=1 * unit.one,
        T2=intrinsic_T2,
        T1=T1,
        vol=1 * unit.cm**3,
        mu=mu_Xe129N,
        temp=163 * unit.K,
        pol=None,
        verbose=False,
    )

    B0 = (axion.nu_a_eff / (sample.gamma / (2 * PI))).to(unit.T)
    fractional_FWHM = (
        1 / (np.pi * axion.nu_a_eff * Tdelta)
    ).to(unit.one)
    nFWHM = 20
    magnet = Magnet(
        B0=B0,
        FWHM=fractional_FWHM,
        nFWHM=nFWHM,
        verbose=False,
    )

    B_rms = (
        np.abs(axion.getRabiFreq() / (sample.gamma / (2 * PI)))
    ).to(unit.T)
    duration = duration_factor * target_T2star
    coherence_rate = points_per_short_time / min(tau_a, target_T2star)
    max_detuning = nFWHM / (np.pi * Tdelta)
    rate = max(coherence_rate, 25 * max_detuning)

    return {
        "key_info": {
            "nu_a": frequency,
            "tau_a_over_T2star": ratio * unit.one,
            "T2_over_T2star": T2_over_T2star * unit.one,
        },
        "axion": axion,
        "sample": sample,
        "magnet": magnet,
        "excField": MagField(),
        "B_a_rms": B_rms,
        "numFields": num_fields,
        "rand_seed": seed,
        "field_mode": field_mode,
        "init_M": init_M_scale * unit.one,
        "init_M_theta": np.pi * unit.rad,
        "init_M_phi": 0 * unit.rad,
        "rate": rate.to(unit.Hz),
        "duration": duration.to(unit.s),
    }


def run_case(params: SimuParams, field_mode: str, verbose: bool):
    """Run one stochastic-FFT or coherent-limit field ensemble."""
    simulations = Simulations(
        name="Signal-formula scan",
        all_params=[params],
    )
    if field_mode == "stochastic":
        simulations.run(autoStart=True, verbose=verbose)
        return simulations

    simulations.setup(verbose=verbose)
    simu = simulations.pool[0].simu
    # Use uniform detuning spacing with a recurrence time beyond the plotted
    # duration.  The production nonuniform grid has coarse outer spacing and
    # its discrete FID can turn negative at late times.
    max_detuning = np.abs(
        simu.sample.gamma
        / (2 * PI)
        * simu.magnet.B0_nW
    ).to(unit.Hz)
    num_packets = max(
        401,
        int(
            np.ceil(
                (
                    8 * max_detuning * simu.duration
                ).to_value(unit.one)
            )
        )
        + 1,
    )
    lower_B = (-simu.magnet.B0_nW).to_value(unit.T)
    upper_B = simu.magnet.B0_nW.to_value(unit.T)
    simu.magnet.B_spread = (
        np.linspace(lower_B, upper_B, num_packets) * unit.T
    )
    simu.magnet.numPt = num_packets
    simu.RCF_freq = 0 * unit.Hz

    # Magnet.setHomogeneity also applies a squared Hamming window.  Replace it
    # with exact integrated Lorentzian-bin probabilities for this validation.
    normalized_B = (
        simu.magnet.B_spread / simu.magnet.FWHM_B0
    ).to_value(unit.one)
    edges = np.empty(len(normalized_B) + 1)
    edges[1:-1] = 0.5 * (normalized_B[:-1] + normalized_B[1:])
    edges[0] = normalized_B[0] - 0.5 * (
        normalized_B[1] - normalized_B[0]
    )
    edges[-1] = normalized_B[-1] + 0.5 * (
        normalized_B[-1] - normalized_B[-2]
    )
    cdf_at_edges = 0.5 + np.arctan(2 * edges) / np.pi
    simu.magnet.ratios = np.diff(cdf_at_edges)
    simu.magnet.ratios /= simu.magnet.ratios.sum()
    num_fields = params["numFields"]
    rng = np.random.default_rng(params["rand_seed"])

    # In the T2star << tau_a limit, each realization has constant random
    # quadratures over the measurement.  Their scale ensures
    # sqrt(<Bx**2 + By**2>) = B_a_rms.
    quadratures = rng.normal(
        scale=params["B_a_rms"].to_value(unit.T) / np.sqrt(2),
        size=(num_fields, 2),
    )
    simu.excField.numFields = num_fields
    simu.excField.B_vec = np.zeros(
        (num_fields, simu.numSteps, 3), dtype=float
    ) * unit.T
    simu.excField.B_vec[:, :, 0] = quadratures[:, 0, np.newaxis] * unit.T
    simu.excField.B_vec[:, :, 1] = quadratures[:, 1, np.newaxis] * unit.T
    simu.generateTrajectories(cleanup=True, verbose=verbose)
    simu.keepMeanStd()
    return simulations


def reduced_chi_square(
    observed: np.ndarray,
    predicted: np.ndarray,
    uncertainty: np.ndarray,
    indices: np.ndarray,
    num_parameters: int,
) -> tuple[float, int, float]:
    """Return diagonal chi-square, degrees of freedom, and reduced chi-square."""
    valid = indices[
        np.isfinite(observed[indices])
        & np.isfinite(predicted[indices])
        & np.isfinite(uncertainty[indices])
        & (uncertainty[indices] > 0)
    ]
    residual = (observed[valid] - predicted[valid]) / uncertainty[valid]
    chi2 = float(np.sum(residual**2))
    dof = max(len(valid) - num_parameters, 1)
    return chi2, dof, chi2 / dof


def analyze_case(entry) -> dict[str, object]:
    """Fit one case using the RMS transverse magnetization."""
    simu = entry.simu
    tau_a = simu.axion.tau_a_est.to(unit.s)
    T2star = simu.T2star.to(unit.s)
    ratio = (tau_a / T2star).to_value(unit.one)
    T2_over_T2star = (simu.sample.T2 / T2star).to_value(unit.one)
    B_rms = entry.params["B_a_rms"]
    field_mode = entry.params["field_mode"]
    num_fields = entry.params["numFields"]
    magnetization_scale = entry.params["init_M"].to_value(unit.one)
    observed = np.asarray(simu.Mxy_rms) / magnetization_scale
    time = np.arange(len(observed)) * simu.timeStep

    # keepMeanStd stores sqrt(std(Mxy**2)) as Mxy_rss.  Apply the delta
    # method to obtain the standard error of sqrt(mean(Mxy**2)).
    std_of_squared_signal = (
        np.asarray(simu.Mxy_rss) / magnetization_scale
    ) ** 2
    denominator = 2 * observed * np.sqrt(num_fields)
    uncertainty = np.divide(
        std_of_squared_signal,
        denominator,
        out=np.zeros_like(observed),
        where=denominator > 0,
    )
    fit_indices = np.unique(
        np.linspace(1, len(observed) - 1, min(250, len(observed) - 1)).astype(int)
    )
    fit_indices = fit_indices[
        np.isfinite(uncertainty[fit_indices]) & (uncertainty[fit_indices] > 0)
    ]
    coherence_factor = np.sqrt(
        (tau_a / (tau_a + T2star)).to_value(unit.one)
    )
    # B_a_rms describes the total transverse stochastic field.  Each
    # independent driving quadrature has RMS B_a_rms/sqrt(2).
    amplitude_factor = 1 / np.sqrt(2) if field_mode == "stochastic" else 1.0
    saturation = (
        amplitude_factor
        * np.abs(simu.sample.gamma)
        * B_rms
        * T2star
        * coherence_factor
    ).to_value(unit.rad)
    observed_normalized = observed / saturation
    uncertainty_normalized = uncertainty / saturation
    time_over_T2star = (time / T2star).to_value(unit.one)

    def normalized_prediction(x: float) -> np.ndarray:
        return (1 - np.exp(-time_over_T2star / x)) ** x

    def objective(x: float) -> float:
        chi2, _, _ = reduced_chi_square(
            observed_normalized,
            normalized_prediction(x),
            uncertainty_normalized,
            fit_indices,
            num_parameters=1,
        )
        return chi2

    fit = minimize_scalar(objective, bounds=(0.5, 1.0), method="bounded")
    x_fit = float(fit.x)
    x_original = 0.5 * (1 + ratio / (1 + ratio))
    x_coherent = 1.0
    fit_normalized = normalized_prediction(x_fit)
    original_normalized = normalized_prediction(x_original)
    # In the T2star << tau_a limit the exact coherent-field expression has
    # saturation gamma * B * T2star, without the general interpolation's
    # finite-coherence prefactor.
    coherent_normalized = normalized_prediction(x_coherent) / coherence_factor
    chi2_fit, dof_fit, reduced_chi2_fit = reduced_chi_square(
        observed_normalized,
        fit_normalized,
        uncertainty_normalized,
        fit_indices,
        num_parameters=1,
    )
    chi2_original, dof_original, reduced_chi2_original = reduced_chi_square(
        observed_normalized,
        original_normalized,
        uncertainty_normalized,
        fit_indices,
        num_parameters=0,
    )
    chi2_coherent, dof_coherent, reduced_chi2_coherent = reduced_chi_square(
        observed_normalized,
        coherent_normalized,
        uncertainty_normalized,
        fit_indices,
        num_parameters=0,
    )

    return {
        "frequency_Hz": simu.axion.nu_a.to_value(unit.Hz),
        "ratio": ratio,
        "T2_over_T2star": T2_over_T2star,
        "tau_a_s": tau_a.to_value(unit.s),
        "T2star_s": T2star.to_value(unit.s),
        "magnet_FWHM": simu.magnet.FWHM.to_value(unit.one),
        "B_rms_T": B_rms.to_value(unit.T),
        "num_fields": num_fields,
        "magnetization_scale": magnetization_scale,
        "field_mode": field_mode,
        "amplitude_factor": amplitude_factor,
        "x_fit": x_fit,
        "x_original": x_original,
        "x_coherent": x_coherent,
        "chi2_fit": chi2_fit,
        "dof_fit": dof_fit,
        "reduced_chi2_fit": reduced_chi2_fit,
        "chi2_original": chi2_original,
        "dof_original": dof_original,
        "reduced_chi2_original": reduced_chi2_original,
        "chi2_coherent": chi2_coherent,
        "dof_coherent": dof_coherent,
        "reduced_chi2_coherent": reduced_chi2_coherent,
        "endpoint_over_original": float(
            observed_normalized[-1] / original_normalized[-1]
        ),
        "saturation": saturation,
        "time_over_T2star": time_over_T2star,
        "observed_normalized": observed_normalized,
        "uncertainty_normalized": uncertainty_normalized,
        "fit_indices": fit_indices,
        "fit_normalized": fit_normalized,
        "original_normalized": original_normalized,
        "coherent_normalized": coherent_normalized,
    }


def x_ratio_model(ratio: np.ndarray | float, parameters: np.ndarray) -> np.ndarray:
    """Bounded logistic model with x(0)=1/2 and x(infinity)=1."""
    log10_r0, power = parameters
    ratio = np.asarray(ratio)
    r0 = 10**log10_r0
    return 0.5 + 0.5 / (1 + (r0 / ratio) ** power)


def fit_global_x_model(results: list[dict[str, object]]) -> np.ndarray:
    """Fit one ratio-dependent x model directly to all simulated curves."""

    def residuals(parameters: np.ndarray) -> np.ndarray:
        all_residuals = []
        for result in results:
            indices = result["fit_indices"]
            x_value = float(x_ratio_model(result["ratio"], parameters))
            prediction = (
                1 - np.exp(-result["time_over_T2star"] / x_value)
            ) ** x_value
            all_residuals.append(
                (
                    result["observed_normalized"][indices] - prediction[indices]
                )
                / result["uncertainty_normalized"][indices]
            )
        return np.concatenate(all_residuals)

    fit = least_squares(
        residuals,
        x0=np.array([0.0, 1.0]),
        bounds=(
            np.array([-3.0, 0.05]),
            np.array([3.0, 5.0]),
        ),
    )
    return fit.x


def add_global_model(
    results: list[dict[str, object]], parameters: np.ndarray
) -> None:
    for result in results:
        x_global = float(x_ratio_model(result["ratio"], parameters))
        global_normalized = (
            1 - np.exp(-result["time_over_T2star"] / x_global)
        ) ** x_global
        chi2, dof, reduced_chi2 = reduced_chi_square(
            result["observed_normalized"],
            global_normalized,
            result["uncertainty_normalized"],
            result["fit_indices"],
            num_parameters=0,
        )
        result["x_global"] = x_global
        result["global_normalized"] = global_normalized
        result["chi2_global"] = chi2
        result["dof_global"] = dof
        result["reduced_chi2_global"] = reduced_chi2


def save_csv(results: list[dict[str, object]], path: Path) -> None:
    scalar_keys = [
        "frequency_Hz",
        "ratio",
        "T2_over_T2star",
        "tau_a_s",
        "T2star_s",
        "magnet_FWHM",
        "B_rms_T",
        "num_fields",
        "magnetization_scale",
        "field_mode",
        "amplitude_factor",
        "x_fit",
        "x_original",
        "x_coherent",
        "x_global",
        "chi2_fit",
        "dof_fit",
        "reduced_chi2_fit",
        "chi2_original",
        "dof_original",
        "reduced_chi2_original",
        "chi2_coherent",
        "dof_coherent",
        "reduced_chi2_coherent",
        "chi2_global",
        "dof_global",
        "reduced_chi2_global",
        "endpoint_over_original",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=scalar_keys)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in scalar_keys})


def frequency_tag(frequency_Hz: float) -> str:
    exponent = int(np.round(np.log10(frequency_Hz)))
    return f"1e{exponent}Hz"


def plot_frequency_curves(
    frequency_results: list[dict[str, object]],
    path: Path,
) -> matplotlib.figure.Figure:
    num_columns = min(
        4, len({result["ratio"] for result in frequency_results})
    )
    num_rows = int(np.ceil(len(frequency_results) / num_columns))
    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(4.1 * num_columns, 3.2 * num_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )

    for result, ax in zip(frequency_results, axes.flat):
        time = result["time_over_T2star"]
        saturation = result["saturation"]
        if result["field_mode"] == "coherent":
            reference_normalized = result["coherent_normalized"]
            reference_label = "coherent theory $x=1$"
            reference_chi2 = result["reduced_chi2_coherent"]
        else:
            reference_normalized = result["original_normalized"]
            reference_label = (
                rf"interpolation $x={result['x_original']:.3f}$"
            )
            reference_chi2 = result["reduced_chi2_original"]
        ax.plot(
            time,
            result["observed_normalized"] * saturation,
            color="black",
            linewidth=1.1,
            label="simulation RMS",
        )
        ax.plot(
            time,
            result["fit_normalized"] * saturation,
            color="tab:blue",
            linewidth=1.4,
            label=f"individual $x={result['x_fit']:.3f}$",
        )
        ax.plot(
            time,
            result["global_normalized"] * saturation,
            color="tab:green",
            linestyle="-.",
            linewidth=1.3,
            label=f"global $x={result['x_global']:.3f}$",
        )
        ax.plot(
            time,
            reference_normalized * saturation,
            color="tab:orange",
            linestyle="--",
            linewidth=1.2,
            label=reference_label,
        )
        ax.set_title(
            f"$\\tau_a/T\\_2^*={result['ratio']:.2g}$, "
            f"$T_2/T\\_2^*={result['T2_over_T2star']:.3g}$"
            "\n"
            f"$\\chi_\\nu^2$: fit={result['reduced_chi2_fit']:.2g}, "
            f"reference={reference_chi2:.2g}"
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    for ax in axes.flat[len(frequency_results) :]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("$t/T_2^*$")
    for ax in axes[:, 0]:
        ax.set_ylabel("$M_{\\perp,\\mathrm{rms}}/M_{\\mathrm{eqb}}$")
    fig.suptitle(
        f"$\\nu_a={frequency_results[0]['frequency_Hz']:.0e}$ Hz, "
        f"$N_\\mathrm{{fields}}={frequency_results[0]['num_fields']}$"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, facecolor="white")
    return fig


def plot_fit_summary(
    results: list[dict[str, object]],
    parameters: np.ndarray,
    path: Path,
) -> matplotlib.figure.Figure:
    fig, (x_axis, chi2_axis) = plt.subplots(
        1, 2, figsize=(11, 4.2), constrained_layout=True
    )
    T2_ratios = sorted(
        {round(float(result["T2_over_T2star"]), 6) for result in results}
    )
    for T2_ratio in T2_ratios:
        subset = [
            result
            for result in results
            if np.isclose(result["T2_over_T2star"], T2_ratio)
        ]
        x_axis.scatter(
            [result["ratio"] for result in subset],
            [result["x_fit"] for result in subset],
            s=28,
            label=rf"$T_2/T_2^*={T2_ratio:.3g}$",
        )

    ratio_grid = np.logspace(
        np.log10(min(result["ratio"] for result in results)),
        np.log10(max(result["ratio"] for result in results)),
        300,
    )
    coherent_mode = results[0]["field_mode"] == "coherent"
    reference_x = (
        np.ones_like(ratio_grid)
        if coherent_mode
        else 0.5 * (1 + ratio_grid / (1 + ratio_grid))
    )
    reference_label = (
        "coherent-limit theory" if coherent_mode else "original interpolation"
    )
    x_axis.plot(
        ratio_grid,
        x_ratio_model(ratio_grid, parameters),
        color="black",
        linewidth=1.8,
        label="global logistic",
    )
    x_axis.plot(
        ratio_grid,
        reference_x,
        color="tab:orange",
        linestyle="--",
        linewidth=1.5,
        label=reference_label,
    )
    x_axis.set_xscale("log")
    x_axis.set_xlabel("$\\tau_a/T_2^*$")
    x_axis.set_ylabel("$x$")
    x_axis.grid(alpha=0.25)
    x_axis.legend(fontsize=8)

    for T2_ratio in T2_ratios:
        subset = [
            result
            for result in results
            if np.isclose(result["T2_over_T2star"], T2_ratio)
        ]
        ratios = sorted({result["ratio"] for result in subset})
        chi2_axis.plot(
            ratios,
            [
                np.median(
                    [
                        result[
                            "reduced_chi2_coherent"
                            if coherent_mode
                            else "reduced_chi2_original"
                        ]
                        for result in subset
                        if np.isclose(result["ratio"], ratio)
                    ]
                )
                for ratio in ratios
            ],
            marker="o",
            linewidth=1.5,
            label=f"reference, $T_2/T_2^*={T2_ratio:.3g}$",
        )
    chi2_axis.set_xscale("log")
    chi2_axis.set_yscale("log")
    chi2_axis.set_xlabel("$\\tau_a/T_2^*$")
    chi2_axis.set_ylabel("diagonal reduced $\\chi^2$")
    chi2_axis.grid(alpha=0.25)
    chi2_axis.legend(fontsize=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, facecolor="white")
    return fig


def main() -> None:
    args = parse_args()
    results = []
    total_cases = (
        len(args.frequencies)
        * len(args.T2_over_T2star)
        * len(args.ratios)
    )
    case_number = 0
    for frequency_Hz in args.frequencies:
        for T2_over_T2star in args.T2_over_T2star:
            for ratio in args.ratios:
                case_number += 1
                print(
                    f"[{case_number}/{total_cases}] "
                    f"nu_a={frequency_Hz:.4g} Hz, "
                    f"tau_a/T2star={ratio:.4g}, "
                    f"T2/T2star={T2_over_T2star:.4g}",
                    flush=True,
                )
                params = make_case(
                    frequency_Hz * unit.Hz,
                    ratio,
                    T2_over_T2star,
                    args.num_fields,
                    args.duration_factor,
                    args.points_per_short_time,
                    args.g_ann,
                    args.numerical_magnetization_scale,
                    args.field_mode,
                    args.seed + case_number - 1,
                )
                # Run one case at a time so large trajectory and derivative
                # arrays from the ensemble can be released immediately.
                simulations = run_case(
                    params,
                    field_mode=args.field_mode,
                    verbose=args.verbose,
                )
                results.append(analyze_case(simulations.pool[0]))
                simulations.pool[0].simu.cleanup()
                del simulations
                gc.collect()

    global_parameters = fit_global_x_model(results)
    add_global_model(results, global_parameters)

    output_prefix = args.output_prefix
    csv_path = output_prefix.parent / f"{output_prefix.name}.csv"
    summary_path = output_prefix.parent / f"{output_prefix.name}-fit-summary.png"
    save_csv(results, csv_path)
    figures = [plot_fit_summary(results, global_parameters, summary_path)]
    curve_paths = []
    for frequency_Hz in args.frequencies:
        frequency_results = [
            result
            for result in results
            if np.isclose(result["frequency_Hz"], frequency_Hz)
        ]
        curve_path = output_prefix.parent / (
            f"{output_prefix.name}-curves-{frequency_tag(frequency_Hz)}.png"
        )
        figures.append(plot_frequency_curves(frequency_results, curve_path))
        curve_paths.append(curve_path)

    log10_r0, power = global_parameters
    coherent_mode = args.field_mode == "coherent"
    reference_key = (
        "reduced_chi2_coherent"
        if coherent_mode
        else "reduced_chi2_original"
    )
    print(
        "\nGlobal x(r) fit:\n"
        "x(r) = 0.5 + 0.5 / (1 + (r0/r)^p)\n"
        f"r0={10**log10_r0:.6g}, p={power:.6g}\n"
    )
    print(
        "frequency_Hz  tau_a/T2star  T2/T2star  x_fit  "
        "red_chi2_fit  red_chi2_reference"
    )
    for result in results:
        print(
            f"{result['frequency_Hz']:12.4g}  "
            f"{result['ratio']:12.4g}  "
            f"{result['T2_over_T2star']:9.4g}  "
            f"{result['x_fit']:5.3f}  "
            f"{result['reduced_chi2_fit']:12.4g}  "
            f"{result[reference_key]:17.4g}"
        )
    print(f"\nSaved results: {csv_path}")
    print(f"Saved summary: {summary_path}")
    for curve_path in curve_paths:
        print(f"Saved curves:  {curve_path}")

    if args.show:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    main()
