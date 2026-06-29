"""Plot daily and annual modulation of the Milky-Way axion lineshape."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as unit
from astropy.time import Time
from astropy.utils import iers

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from axionbloch.MilkyWayAxionHalo import MilkyWayAxionHalo
from axionbloch.Station import Mainz


iers.conf.auto_download = False

FIGURE_DIR = Path(__file__).resolve().parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

station = Mainz
axion = MilkyWayAxionHalo(nu_a=1.0 * unit.MHz)

# Frequencies around the Compton frequency.  The window is intentionally wider
# than the SHM linewidth so the plotted PSD traces are well normalized.
frequencies = axion.nu_a + np.linspace(-0.5, 8.0, 320) * unit.Hz


def save_daily_modulation():
    """Daily modulation from Earth's rotation at a fixed date."""
    times = Time("2024-06-21T00:00:00", scale="utc") + np.linspace(
        0, 24, 49
    ) * unit.hour

    fig, result = axion.plotPeriodicModulation(
        station=station,
        meas_times=times,
        frequencies=frequencies,
        case="grad_perp",
        showPlot=False,
    )
    fig.set_size_inches(7.0, 6.0)
    out = FIGURE_DIR / "MWAxion_daily_modulation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out}")
    print(
        "Daily wind-angle range:",
        f"{np.min(result['wind_angle'].to_value(unit.deg)):.2f} deg to",
        f"{np.max(result['wind_angle'].to_value(unit.deg)):.2f} deg",
    )


def save_daily_directional_power_modulation():
    """Paper-style daily modulation of cos(alpha), P_parallel, and P_perp."""
    times = Time("2024-06-21T00:00:00", scale="utc") + np.linspace(
        0, 24, 145
    ) * unit.hour
    t_hours = (times - times[0]).to_value(unit.hour)
    axes_to_plot = {
        "North": "north",
        "West": "west",
        "Zenith": "zenith",
    }
    styles = {
        "North": {"color": "tab:red", "linestyle": "-"},
        "West": {"color": "tab:blue", "linestyle": "--"},
        "Zenith": {"color": "tab:green", "linestyle": "-."},
    }

    cos_alpha = {}
    p_par = {}
    p_perp = {}
    for label, axis in axes_to_plot.items():
        kin = axion.findKinematicsOverTime(
            station=station,
            meas_times=times,
            sensitive_axis=axis,
            include_rotation=True,
        )
        alpha = kin["wind_angle"]
        cos_alpha[label] = kin["cos_alpha"]
        p_par[label] = axion.gradientPowerCoefficient(
            axion.v_0,
            kin["v_lab_magnitude"],
            alpha,
            case="grad_par",
        )
        p_perp[label] = axion.gradientPowerCoefficient(
            axion.v_0,
            kin["v_lab_magnitude"],
            alpha,
            case="grad_perp",
        )

    p_par_norm = {
        label: vals / max(np.max(v) for v in p_par.values()) for label, vals in p_par.items()
    }
    p_perp_norm = {
        label: vals / max(np.max(v) for v in p_perp.values())
        for label, vals in p_perp.items()
    }

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.5), dpi=200, sharex=True)
    for label in axes_to_plot:
        axes[0].plot(t_hours, cos_alpha[label], label=label, **styles[label])
        axes[1].plot(
            t_hours,
            p_par_norm[label].to_value(unit.one),
            label=label,
            **styles[label],
        )
        axes[2].plot(
            t_hours,
            p_perp_norm[label].to_value(unit.one),
            label=label,
            **styles[label],
        )

    axes[0].axhline(0, color="0.6", linewidth=0.8)
    axes[0].set_ylabel(r"$\cos\alpha$")
    axes[1].set_ylabel(r"$P_\parallel$ (norm.)")
    axes[2].set_ylabel(r"$P_\perp$ (norm.)")
    axes[2].set_xlabel("Hours since 2024-06-21 00:00 UTC")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle(f"Daily directional modulation at {station.name}")
    fig.tight_layout()

    out = FIGURE_DIR / "MWAxion_daily_directional_power.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out}")
    print(
        "Daily P_parallel normalized span:",
        f"{min(np.min(v.to_value(unit.one)) for v in p_par_norm.values()):.3f} to",
        f"{max(np.max(v.to_value(unit.one)) for v in p_par_norm.values()):.3f}",
    )
    print(
        "Daily P_perp normalized span:",
        f"{min(np.min(v.to_value(unit.one)) for v in p_perp_norm.values()):.3f} to",
        f"{max(np.max(v.to_value(unit.one)) for v in p_perp_norm.values()):.3f}",
    )


def save_annual_modulation():
    """Annual modulation from Earth's orbit, sampled at fixed UTC time."""
    times = Time("2024-01-01T12:00:00", scale="utc") + np.linspace(
        0, 365, 73
    ) * unit.day

    result = axion.findLineshapeOverTime(
        frequencies=frequencies,
        station=station,
        meas_times=times,
        case="grad_perp",
        include_rotation=False,
    )

    t_days = (result["times"] - result["times"][0]).to_value(unit.day)
    psd = result["power_spectrum_shape"]
    peak_idx = int(np.argmax(np.mean(psd.to_value(psd.unit), axis=0)))

    fig, axes = plt.subplots(4, 1, figsize=(7.0, 7.2), dpi=200, sharex=True)
    axes[0].plot(t_days, result["v_lab_magnitude"].to_value(unit.km / unit.s))
    axes[0].set_ylabel(r"$|v_\mathrm{lab}|$ (km/s)")

    axes[1].plot(t_days, result["wind_angle"].to_value(unit.deg))
    axes[1].set_ylabel(r"$\alpha$ (deg)")

    axes[2].plot(t_days, result["relative_power"].to_value(unit.one))
    axes[2].set_ylabel("relative power")

    axes[3].plot(t_days, psd[:, peak_idx].to_value(psd.unit))
    axes[3].set_ylabel(f"PSD shape ({psd.unit})")
    axes[3].set_xlabel("Days since 2024-01-01 12:00 UTC")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{axion.name} annual modulation at {station.name}")
    fig.tight_layout()

    out = FIGURE_DIR / "MWAxion_annual_modulation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out}")
    print(
        "Annual speed range:",
        f"{np.min(result['v_lab_magnitude'].to_value(unit.km / unit.s)):.2f} km/s to",
        f"{np.max(result['v_lab_magnitude'].to_value(unit.km / unit.s)):.2f} km/s",
    )


if __name__ == "__main__":
    save_daily_modulation()
    save_daily_directional_power_modulation()
    save_annual_modulation()
