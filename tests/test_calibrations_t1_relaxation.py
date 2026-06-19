"""Calibration tests – T1 longitudinal relaxation recovery.

Usage
-----
Run all 9 cases::

    pytest tests/test_calibrations_t1_relaxation.py

Run a single case with a diagnostic plot::

    pytest "tests/test_calibrations_t1_relaxation.py::test_t1_recovery[1.0s-1e+06Hz]" --show-plots

Run all calibration suites::

    pytest tests/test_calibrations_free_decay.py tests/test_calibrations_t1_relaxation.py tests/test_calibrations_cw_nmr.py

T1 longitudinal relaxation calibration
--------------------------------------
A 90° pulse is applied, tipping Mz to zero.  The longitudinal magnetization
then recovers toward equilibrium (M0_eqb = 1) as

    Mz(t) = 1 + (Mz0 − 1) · exp(−t / T1)

where Mz0 = Mz(t90) is the simulated post-pulse longitudinal magnetization
(which includes finite-pulse T1 relaxation during the pulse) and t is measured
from the end of the pulse.

Physics
-------
After the 90° pulse ends, the RF field is zero, so the Bloch equation for
the longitudinal component decouples completely:

    dMz/dt = −(Mz − M0_eq) / T1

with exact solution

    Mz(t) = M0_eq + (Mz0 − M0_eq) · exp(−t / T1).

Normalized to M0_eqb = 1:

    Mz(t) = 1 + (Mz0 − 1) · exp(−t / T1).

The test uses the actual simulated Mz0 as the initial condition, which absorbs
any finite-pulse error, and focuses the χ² on the recovery time constant T1.

The agreement is quantified via

    χ² = ‖Mz − Mz_expected‖² / ‖Mz_expected − 1‖² ≤ _CHI2_TOLERANCE

where the denominator uses the departure from equilibrium (Mz_expected − 1)
to focus the metric on relaxation dynamics rather than the plateau.

Simulation parameters
---------------------
    duration = N_T1 × T1
    rate     = N_PER_T1 / T1

This keeps the total RK4 step count at exactly N_T1 × N_PER_T1 = 2500 for
all cases.  A single on-resonance spin packet (nFWHM=0) has zero precession
frequency, so no stability floor is needed.

Parametrisation
---------------
  • three T1 values:       1 ms, 1 s, 1 ks
  • three RCF frequencies: 1 kHz, 1 MHz, 1 GHz
  Total: 9 test cases, all expected to pass.
"""

import pytest

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_H1, mu_p
from axionbloch.dependency import *
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation

# ---------------------------------------------------------------------------
# Shared simulation parameters
# ---------------------------------------------------------------------------
# nFWHM=0 forces a single on-resonance spin packet (uniform field).  This
# avoids two problems that arise with inhomogeneous fields for large T1:
#   1. Simulation.__init__ computes numPt ∝ duration × nFWHM × FWHM × RCF_freq,
#      which reaches 400 000 spin packets for T1=1000 s / 1 GHz and makes
#      construction take >60 s.
#   2. Fast-precessing outer packets (δ ≈ nFWHM × FWHM × RCF_freq) require a
#      high integration rate for RK4 stability, inflating the step count.
# Field inhomogeneity has no effect on Mz recovery (T1 is uniform), so a
# single on-resonance packet is sufficient and exact.
_FWHM = 1.0 * ppm  # placeholder — ignored because nFWHM=0
_T2_T1_RATIO = 0.1  # T2 = T1 × _T2_T1_RATIO  (NMR condition T2 ≤ T1)
_NFWHM = 0.0  # uniform field: one spin packet at B0, no precession
_T90_STEPS = 5  # 90° pulse length in time steps

# χ² tolerance: ‖Mz − Mz_expected‖² / ‖Mz_expected − 1‖²
_CHI2_TOLERANCE = 1e-7

# Timing
_N_T1 = 5.0  # observe for 5 × T1
_N_PER_T1 = 500  # RK4 samples per T1 (sufficient for single-packet case)

# ---------------------------------------------------------------------------
# Parametrisation
# ---------------------------------------------------------------------------
_T1_CASES = [1e-3 * unit.s, 1.0 * unit.s, 1e3 * unit.s]
_RCF_FREQS = [1.0 * unit.kHz, 1.0 * unit.MHz, 1.0 * unit.GHz]

_GAMMA = gamma_H1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_t1_simulation(
    RCF_freq: Quantity,
    T1: Quantity,
    rate: Quantity,
    duration: Quantity,
) -> Simulation:
    """Build and run a 90°-pulse T1-recovery simulation.

    Parameters
    ----------
    RCF_freq:
        Rotating-frame carrier frequency.
    T1:
        Longitudinal relaxation time.
    rate:
        Simulation sampling rate.
    duration:
        Observation window (total time, including the 90° pulse).
    """
    T2 = T1 * _T2_T1_RATIO

    sample = Sample(
        name="calibration_proton",
        gamma=gamma_H1,
        massDensity=0.789 * unit.g / unit.cm**3,
        molarMass=46.069 * unit.g / unit.mol,
        numOfSpinsPerMolecule=6 * unit.one,
        T1=T1,
        T2=T2,
        vol=1.0 * unit.cm**3,
        mu=mu_p,
        temp=300.0 * unit.K,
        verbose=False,
    )

    # On-resonance: ν_L = RCF_freq → B0 = RCF_freq / (γ/2π)
    B0 = RCF_freq / (sample.gamma / (2 * PI))

    magnet = Magnet(
        name="calibration_magnet",
        B0=B0,
        FWHM=_FWHM,
        nFWHM=_NFWHM,
    )

    simu = Simulation(
        name=(f"t1_{RCF_freq.to_value(unit.Hz):.3g}Hz" f"_{T1.to_value(unit.s):.3g}s"),
        sample=sample,
        magnet=magnet,
        excField=MagField(name="90deg_pulse"),
        RCF_freq=RCF_freq,
        rate=rate,
        duration=duration,
        verbose=False,
    )

    # On-resonance 90° pulse: nu_rot=0 gives a DC pulse along x in the rotating frame
    # TODO calibrate 90 deg pulse
    simu.excField.set90DegPulse(
        timeStep=simu.timeStep,
        timeLen=simu.timeLen,
        gamma=simu.sample.gamma,
        t90=_T90_STEPS * simu.timeStep,
        nu_rot=0.0 * unit.Hz,
    )

    simu.generateTrajectories(integrator="RK4")
    return simu


def _t1_expected_curve(simu: Simulation, t90_steps: int) -> np.ndarray:
    """Expected Mz(t) recovery curve after the 90° pulse.

    Uses the actual simulated Mz0 at the end of the pulse as the initial
    condition, absorbing finite-pulse T1 relaxation into the starting point.

    Returns an array of length ``simu.timeLen − t90_steps``.
    """
    t_s = simu.getTimeStamp().to_value(unit.s)
    t_sig = t_s[t90_steps:] - t_s[t90_steps]  # time since end of pulse
    T1_s = simu.sample.T1.to_value(unit.s)
    Mz0 = float(simu.trjry[0, t90_steps, 2])  # post-pulse Mz / M0_eqb
    return 1.0 + (Mz0 - 1.0) * np.exp(-t_sig / T1_s)


# ---------------------------------------------------------------------------
# Optional diagnostic plots
# ---------------------------------------------------------------------------


def _show_figure(fig, name: str) -> None:
    """Display *fig* interactively (TkAgg/QtAgg) or as a PNG file (Agg fallback)."""
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    else:
        import os
        import tempfile

        tmp = tempfile.NamedTemporaryFile(
            prefix=f"{name}_", suffix=".png", delete=False
        )
        path = tmp.name
        tmp.close()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        os.startfile(path)


def _plot_t1_result(
    simu: Simulation,
    t90_steps: int,
    T1: Quantity,
    chi2: float,
) -> None:
    """Two-panel diagnostic figure for one T1 test case.

    Panel 1 – Mx(t), My(t) after the 90° pulse (T2 transverse decay).
    Panel 2 – Mz(t) vs expected T1 recovery curve.
    """
    t_s = simu.getTimeStamp().to_value(unit.s)
    t_sig = t_s[t90_steps:]
    Mx = simu.trjry[0, t90_steps:, 0]
    My = simu.trjry[0, t90_steps:, 1]
    Mz = simu.trjry[0, t90_steps:, 2]
    expected_curve = _t1_expected_curve(simu, t90_steps)

    marksize = 1
    cm = 1 / 2.54
    fig = plt.figure(figsize=(2 * 8.5 * cm, 2 * 0.4 * 8.5 * cm), dpi=300)
    gs = gridspec.GridSpec(nrows=1, ncols=2)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    axes: list[Axes] = [ax00, ax01]

    ax: Axes = axes[0]
    ax.plot(t_sig, Mx, label="$M_x / M_\\mathrm{eqb}$", marker="o", markersize=marksize)
    ax.plot(t_sig, My, label="$M_y / M_\\mathrm{eqb}$", marker="*", markersize=marksize)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$M / M_\\mathrm{eqb}$")
    ax.set_title("$M_x$, $M_y$ vs time")
    ax.legend()

    ax = axes[1]
    ax.scatter(t_sig, Mz, label="$M_z$ simulation", s=marksize, color="tab:red")
    ax.plot(
        t_sig,
        expected_curve,
        label="$1+(M_{z0}-1)e^{-t/T_1}$ expected",
        color="tab:purple",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$M / M_\\mathrm{eqb}$")
    ax.set_title("$M_z$ T1 recovery")
    ax.legend()

    fig.suptitle(
        f"T1 recovery  RCF={simu.RCF_freq:.3g}  "
        f"$T_1$={T1:.3g}  $T_2$={simu.sample.T2:.3g}\n"
        f"$\\chi^2$={chi2:.2e}"
    )
    plt.tight_layout()
    _show_figure(fig, simu.name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "RCF_freq", _RCF_FREQS, ids=lambda f: f"{f.to_value(unit.Hz):.3g}Hz"
)
@pytest.mark.parametrize("T1", _T1_CASES, ids=lambda t: f"{t.to_value(unit.s):.3g}s")
def test_t1_recovery(
    RCF_freq: Quantity,
    T1: Quantity,
    show_plots: bool,
):
    """Longitudinal Mz(t) recovery matches T1 exponential formula.

    After a 90° pulse tips Mz to zero, free recovery obeys

        Mz(t) = 1 + (Mz0 − 1) · exp(−t / T1)

    where Mz0 is the actual simulated post-pulse value and Mz is normalized
    to M0_eqb = 1.

    Tolerance: χ² = ‖Mz − Mz_expected‖² / ‖Mz_expected − 1‖² ≤ _CHI2_TOLERANCE.
    """
    # Rate floor: must also resolve the fastest spin-packet precession (RK4 stability)
    duration = _N_T1 * T1
    rate = _N_PER_T1 / T1

    simu = _build_t1_simulation(RCF_freq, T1, rate, duration)

    Mz = simu.trjry[0, _T90_STEPS:, 2]
    expected_curve = _t1_expected_curve(simu, _T90_STEPS)
    departure = expected_curve - 1.0  # (Mz0 − 1)·exp(−t/T1), always ≤ 0

    chi2 = float(np.sum((Mz - expected_curve) ** 2) / np.sum(departure**2))

    if show_plots:
        _plot_t1_result(simu, _T90_STEPS, T1, chi2)

    assert chi2 <= _CHI2_TOLERANCE, (
        f"RCF_freq={RCF_freq}, T1={T1:.3g}, T2={simu.sample.T2:.3g}: "
        f"chi2={chi2:.2e} (tol={_CHI2_TOLERANCE:.0e})"
    )
