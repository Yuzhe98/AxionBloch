"""Plot MW axion PSD or power spectrum for static cases or station settings.
Reproducing FIG. 2(a) at Gramolin spectral signature paper. 
Examples
--------
Default Boston/time velocity, static alpha = 0 and pi/2 limits for all cases::

    python examples/MilkyWayAxionHalo/plot-axion-lineshape-at-station.py

Station-derived bias-field angle instead::

    python examples/MilkyWayAxionHalo/plot-axion-lineshape-at-station.py --mode station --axis north

Custom location::

    python examples/MilkyWayAxionHalo/plot-axion-lineshape-at-station.py --latitude-deg 42.3484 --longitude-deg -71.1002
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as unit
from astropy.time import Time
from astropy.utils import iers

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from axionbloch.dependency import ppm
from axionbloch.MilkyWayAxionHalo import MilkyWayAxionHalo
from axionbloch.Station import Station
import axionbloch.Station as station_module

iers.conf.auto_download = False

FIGURE_DIR = Path(__file__).resolve().parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

CASE_LABELS = {
    "grad_perp": "$\\lambda_\\perp$",
    "grad_par": "$\\lambda_\\parallel$",
    "non-grad": "$\\lambda$",
}
CASE_COLORS = {
    "non-grad": "red",
    "grad_par": "blue",
    "grad_perp": "green",
}
ALPHA_LINESTYLES = {
    "station": "-.",
    "0": "-",
    "pi/2": ":",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot MW axion PSD or power spectrum seen by a station at one time."
        )
    )
    parser.add_argument(
        "--station",
        default="Boston",
        help="Preset station name from axionbloch.Station, used unless latitude/longitude are set.",
    )
    parser.add_argument(
        "--latitude-deg",
        type=float,
        default=None,
        help="Custom signed latitude in degrees; north is positive.",
    )
    parser.add_argument(
        "--longitude-deg",
        type=float,
        default=None,
        help="Custom signed longitude in degrees; east is positive, west is negative.",
    )
    parser.add_argument(
        "--elevation-m",
        type=float,
        default=0.0,
        help="Custom station elevation in meters.",
    )
    parser.add_argument(
        "--time",
        default="2022-01-01T00:00:00",
        help="Measurement time, interpreted by astropy Time. Default is UTC-like ISO.",
    )
    parser.add_argument(
        "--nu-a",
        type=float,
        default=1.0,
        help="Axion Compton frequency value.",
    )
    parser.add_argument(
        "--nu-a-unit",
        default="MHz",
        help="Astropy unit string for --nu-a, e.g. Hz, kHz, MHz.",
    )
    parser.add_argument(
        "--axis",
        default="zenith",
        help="Bias field direction: north, west, east, zenith/up, or another axis accepted by MilkyWayAxionHalo.",
    )
    parser.add_argument(
        "--mode",
        default="static-alpha",
        choices=["static-alpha", "station"],
        help=(
            "static-alpha plots the explicit axion_lineshape cases at alpha=0 "
            "and pi/2. station plots the selected cases at the station-derived "
            "bias-field angle."
        ),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["non-grad", "grad_par", "grad_perp"],
        choices=["grad_perp", "grad_par", "non-grad"],
        help="Lineshape cases to plot.",
    )
    parser.add_argument(
        "--spectrum",
        default="PSD",
        choices=["PSD", "power_spectrum"],
        help="Plot PSD or power_spectrum without normalization.",
    )
    parser.add_argument(
        "--frequency-span-ppm",
        type=float,
        default=None,
        help="Frequency span above nu_a in ppm. Default uses 10 times nominal SHM FWHM.",
    )
    parser.add_argument(
        "--num-frequency-points",
        type=int,
        default=20001,
        help="Number of frequency points.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PDF path. Default is examples/MilkyWayAxionHalo/figures/MW-axion-lineshape-at-station.pdf.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively after saving.",
    )
    return parser.parse_args()


def _line_settings(args: argparse.Namespace) -> list[tuple[str, unit.Quantity | None, str]]:
    if args.mode == "station":
        return [(case, None, CASE_LABELS[case]) for case in args.cases]

    settings = []
    for case in args.cases:
        if case == "non-grad":
            settings.append((case, 0.0 * unit.rad, "non-grad"))
        else:
            settings.append((case, 0.0 * unit.rad, f"{CASE_LABELS[case]}, $\\alpha=0$"))
            settings.append(
                (case, 0.5 * np.pi * unit.rad, f"{CASE_LABELS[case]}, $\\alpha=\\pi/2$")
            )
    return settings


def _make_station(args: argparse.Namespace):
    if args.latitude_deg is None and args.longitude_deg is None:
        if not hasattr(station_module, args.station):
            raise ValueError(
                f"Unknown station preset {args.station!r}. "
                "Use --latitude-deg and --longitude-deg for a custom station."
            )
        return getattr(station_module, args.station)

    if args.latitude_deg is None or args.longitude_deg is None:
        raise ValueError("Set both --latitude-deg and --longitude-deg for a custom station.")

    latitude = abs(args.latitude_deg) * unit.deg
    longitude = abs(args.longitude_deg) * unit.deg
    return Station(
        name="Custom station",
        NSsemisphere="N" if args.latitude_deg >= 0 else "S",
        EWsemisphere="E" if args.longitude_deg >= 0 else "W",
        latitude=latitude,
        longitude=longitude,
        elevation=args.elevation_m * unit.m,
        verbose=False,
    )


def _add_watermark(fig, script_path: Path) -> None:
    figure_width_points = fig.get_size_inches()[0] * 72
    approximate_character_width_points = 0.6 * 4
    maximum_characters = int(figure_width_points / approximate_character_width_points)
    watermark = textwrap.fill(
        f"Generated by: {script_path}",
        width=maximum_characters,
    )
    fig.text(
        0.02,
        0.008,
        watermark,
        fontsize=6,
        color="gray",
        wrap=True,
    )


def main() -> None:
    args = _parse_args()
    station = _make_station(args)
    meas_time = Time(args.time, scale="utc")
    nu_a = args.nu_a * unit.Unit(args.nu_a_unit)
    axion = MilkyWayAxionHalo(nu_a=nu_a)
    frequencies = axion.makeLineshapeFrequencyGrid(
        frequency_span_ppm=args.frequency_span_ppm,
        num_frequency_points=args.num_frequency_points,
    )
    kinematics = axion.findKinematicsOverTime(
        station=station,
        meas_times=meas_time,
        sensitive_axis=args.axis,
        include_rotation=True,
    )
    v_lab = kinematics["v_lab_magnitude"][0]
    station_alpha = kinematics["wind_angle"][0]

    fig, ax = plt.subplots(figsize=(13 / 2.54, 8 / 2.54), dpi=300)
    fig.patch.set_facecolor("white")

    plotted_results = {}
    y_unit = None
    for case, alpha, label_prefix in _line_settings(args):
        alpha_for_case = station_alpha if alpha is None else alpha
        PSD = axion.axion_lineshape(
            v_0=axion.v_0,
            v_lab=v_lab,
            nu_a=axion.nu_a,
            nu=frequencies,
            case=case,
            alpha=alpha_for_case,
        )
        power_coefficient = axion.gradientPowerCoefficient(
            v_0=axion.v_0,
            v_lab=v_lab,
            alpha=alpha_for_case,
            case=case,
        )
        power_spectrum = power_coefficient * PSD
        y_quantity = (
            (axion.nu_a * PSD / 1e6).to(unit.one) if args.spectrum == "PSD" else power_spectrum
        )
        FWHM = axion.measureLineshapeFWHM(
            frequencies,
            y_quantity,
            nu_a=axion.nu_a,
        )

        alpha_key = "station" if alpha is None else f"{alpha.to_value(unit.rad):.6g}"
        result_key = f"{case}:{alpha_key}"
        plotted_results[result_key] = {
            "case": case,
            "alpha": alpha_for_case,
            "PSD": PSD,
            "power_spectrum": power_spectrum,
            **FWHM,
        }

        y_unit = y_quantity.unit
        y = y_quantity.to_value(y_unit)
        x_ppm = ((frequencies - axion.nu_a) / axion.nu_a).to_value(ppm)
        fwhm_ppm = FWHM["FWHM_a"].to_value(ppm)
        label = (
            f"{label_prefix}, FWHM={fwhm_ppm:.3g} ppm, "
            f"$\\tau_a$={FWHM['tau_a'].to_value(unit.s):.3g} s"
        )
        if alpha is None:
            linestyle = ALPHA_LINESTYLES["station"]
            line_zorder = 4
        elif np.isclose(alpha.to_value(unit.rad), 0.0):
            linestyle = ALPHA_LINESTYLES["0"]
            line_zorder = 2
        else:
            linestyle = ALPHA_LINESTYLES["pi/2"]
            line_zorder = 3
        ax.plot(
            x_ppm,
            y,
            label=label,
            color=CASE_COLORS[case],
            linestyle=linestyle,
            zorder=line_zorder,
        )

        # half_max = FWHM["half_max"].to_value(y_unit)
        # lower_ppm = (
        #     (FWHM["lower_half_max_frequency"] - axion.nu_a) / axion.nu_a
        # ).to_value(ppm)
        # upper_ppm = (
        #     (FWHM["upper_half_max_frequency"] - axion.nu_a) / axion.nu_a
        # ).to_value(ppm)
        # ax.hlines(
        #     half_max,
        #     lower_ppm,
        #     upper_ppm,
        #     colors=ax.lines[-1].get_color(),
        #     linestyles=":",
        #     linewidth=0.8,
        #     zorder=line_zorder + 0.1,
        # )

    station_label = station.name
    station_label_short = (
        args.station
        if args.latitude_deg is None and args.longitude_deg is None
        else station.name
    )
    station_alpha_deg = station_alpha.to_value(unit.deg)
    v_lab_value = v_lab.to_value(unit.km / unit.s)

    ax.set_xlabel("$\\nu/\\nu_a - 1$ (ppm)")
    if args.spectrum == "PSD":
        ax.set_ylabel("$\\nu_a \\lambda / 10^6$")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False, fontsize=7)
    ax.set_title(
        (
            f"MW axion {args.spectrum} at {station_label_short}\n"
            f"mode={args.mode}, bias axis={args.axis}, time={meas_time.iso}\n"
            f"$v_\\mathrm{{lab}}$={v_lab_value:.2f} km/s, "
            f"station $\\alpha$={station_alpha_deg:.2f} deg"
        ),
        fontsize=8,
    )

    _add_watermark(fig, Path(__file__).resolve())
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    output_path = (
        Path(args.output)
        if args.output is not None
        else FIGURE_DIR / "MW-axion-lineshape-at-station.pdf"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white", transparent=False)

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved {output_path}")
    print(f"station = {station_label}")
    print(f"time = {meas_time.iso}")
    print(f"axis = {args.axis}")
    print(f"mode = {args.mode}")
    print(f"v_lab = {v_lab_value:.3f} km/s")
    print(f"station alpha = {station_alpha_deg:.3f} deg")
    for key, result in plotted_results.items():
        print(
            f"{key}: alpha = {result['alpha'].to_value(unit.deg):.6g} deg, "
            f"FWHM = {result['FWHM_a'].to_value(ppm):.6g} ppm, "
            f"tau_a = {result['tau_a'].to_value(unit.s):.6g} s"
        )


if __name__ == "__main__":
    main()
