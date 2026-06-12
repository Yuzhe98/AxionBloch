"""Calibration tests – T1 relaxation of a hyperpolarized sample.

Usage
-----
Run all 27 cases::

    pytest tests/test_calibrations_t1_hyperpolarized.py

Run a single case with a diagnostic plot::

    pytest "tests/test_calibrations_t1_hyperpolarized.py::test_t1_hyperpolarized[10.0x-1s-1e+06Hz]" --show-plots

Run together with the thermal T1 suite::

    pytest tests/test_calibrations_t1_relaxation.py tests/test_calibrations_t1_hyperpolarized.py

Hyperpolarized T1 calibration
------------------------------
The sample starts in a longitudinally overpolarized state Mz(0) = k > 1
(k times the thermal equilibrium magnetization M0_eqb) with no transverse
magnetization.  No excitation pulse is applied.  Mz then decays toward
equilibrium as

    Mz(t) = 1 + (k − 1) · exp(−t / T1).

This tests the T1 relaxation term of the Bloch solver in the recovery
direction Mz > 1 → 1, complementing the thermal T1 test which exercises
Mz < 1 → 1 (post-90° pulse).

Hyperpolarization setup
-----------------------
The overpolarization factor k is set via ``sample.pol``:

    pol = k × pol_thermal(B0)

where ``pol_thermal = tanh(ℏγB0 / 2k_BT)`` is the thermal equilibrium
polarization.  The Simulation constructor then computes

    init_M = getM0(pol) / M0_eqb = pol / pol_thermal = k.

With ``init_M_theta = 0`` (default), the initial state is
Mx = My = 0, Mz = k — a purely longitudinal overpolarization.

Physics
-------
With no excitation field (B1 = 0) and a single on-resonance spin packet
(nFWHM = 0), the Bloch equations decouple completely:

    dMx/dt = −Mx / T2    → Mx = 0 (from initial Mx = 0)
    dMy/dt = −My / T2    → My = 0 (from initial My = 0)
    dMz/dt = −(Mz − 1) / T1  → Mz(t) = 1 + (k − 1) · exp(−t / T1)

The test verifies the T1 decay term via

    χ² = ‖Mz − Mz_expected‖² / ‖Mz_expected − 1‖² ≤ _CHI2_TOLERANCE

where the denominator uses the departure from equilibrium to normalize.

Simulation parameters
---------------------
    duration = N_T1 × T1
    rate     = N_PER_T1 / T1

This keeps the total RK4 step count at N_T1 × N_PER_T1 = 2500 for all cases.
nFWHM = 0 (single on-resonance packet) avoids the numPt explosion that arises
at large T1 × RCF_freq and keeps the simulation fast.

Parametrisation
---------------
  • three overpolarization factors:  k = 2, 10, 100
  • three T1 values:                 1 ms, 1 s, 1 ks
  • three RCF frequencies:           1 kHz, 1 MHz, 1 GHz
  Total: 27 test cases, all expected to pass.
"""

import pytest

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p, mu_p
from axionbloch.dependency import *
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation

# ---------------------------------------------------------------------------
# Shared simulation parameters
# ---------------------------------------------------------------------------
_FWHM = 1.0 * ppm  # placeholder — ignored because nFWHM=0
_T2_T1_RATIO = 0.1  # T2 = T1 × _T2_T1_RATIO  (NMR condition T2 ≤ T1)
_NFWHM = 0.0  # uniform field: one spin packet at B0, no precession
_TEMP = 300.0 * unit.K  # sample temperature for thermal polarization reference

# χ² tolerance: ‖Mz − Mz_expected‖² / ‖Mz_expected − 1‖²
_CHI2_TOLERANCE = 1e-7

# Timing
_N_T1 = 5.0  # observe for 5 × T1
_N_PER_T1 = 500  # RK4 samples per T1 (sufficient for single-packet case)

# ---------------------------------------------------------------------------
# Parametrisation
# ---------------------------------------------------------------------------
_OVERPOL_CASES = [2.0, 10.0, 100.0, 1e4]  # overpolarization factor k
_T1_CASES = [1e-3 * unit.s, 1.0 * unit.s, 1e3 * unit.s]
_RCF_FREQS = [1.0 * unit.kHz, 1.0 * unit.MHz, 1.0 * unit.GHz]

_GAMMA = gamma_p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_hyperpol_simulation(
    RCF_freq: Quantity,
    T1: Quantity,
    k: float,
    rate: Quantity,
    duration: Quantity,
) -> Simulation:
    """Build and run a hyperpolarized T1-relaxation simulation (no pulse).

    Parameters
    ----------
    RCF_freq:
        Rotating-frame carrier frequency.
    T1:
        Longitudinal relaxation time.
    k:
        Overpolarization factor: initial Mz / M0_eqb = k.
    rate:
        Simulation sampling rate.
    duration:
        Observation window.
    """
    T2 = T1 * _T2_T1_RATIO

    # Create sample without polarization first, then set hyperpolarization
    sample = Sample(
        name="calibration_proton",
        gamma=gamma_p,
        massDensity=0.789 * unit.g / unit.cm**3,
        molarMass=46.069 * unit.g / unit.mol,
        numOfSpinsPerMolecule=6 * unit.one,
        T1=T1,
        T2=T2,
        vol=1.0 * unit.cm**3,
        mu=mu_p,
        temp=_TEMP,
        pol=None,
        verbose=False,
    )

    # On-resonance: ν_L = RCF_freq → B0 = RCF_freq / (γ/2π)
    B0 = RCF_freq / (sample.gamma / (2 * PI))

    # Set hyperpolarization: pol = k × pol_thermal so that init_M = k
    pol_thermal = sample.getThermalPol(B_pol=B0)
    sample.pol = k * pol_thermal

    magnet = Magnet(
        name="calibration_magnet",
        B0=B0,
        FWHM=_FWHM,
        nFWHM=_NFWHM,
    )

    simu = Simulation(
        name=(
            f"hyperpol_{RCF_freq.to_value(unit.Hz):.3g}Hz"
            f"_{T1.to_value(unit.s):.3g}s"
            f"_k{k:.4g}"
        ),
        sample=sample,
        magnet=magnet,
        excField=MagField(name="zero"),
        RCF_freq=RCF_freq,
        rate=rate,
        duration=duration,
        verbose=False,
    )

    # No excitation field: B_vec = 0 throughout — pure T1 decay
    simu.excField.B_vec = np.zeros((1, simu.numSteps, 3)) * unit.T

    simu.generateTrajectories(integrator="RK4")
    return simu


def _hyperpol_expected_curve(simu: Simulation) -> np.ndarray:
    """Expected Mz(t) decay from the hyperpolarized initial state.

    Uses the actual Mz(0) from the trajectory as the initial condition.
    For a pure longitudinal overpolarization this equals k to machine
    precision, but reading it directly makes the test self-consistent.

    Returns an array of length ``simu.timeLen``.
    """
    t_s = simu.getTimeStamp().to_value(unit.s)
    T1_s = simu.sample.T1.to_value(unit.s)
    Mz0 = float(simu.trjry[0, 0, 2])  # initial Mz / M0_eqb (= k at t=0)
    return 1.0 + (Mz0 - 1.0) * np.exp(-t_s / T1_s)


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


def _plot_hyperpol_result(
    simu: Simulation,
    T1: Quantity,
    k: float,
    chi2: float,
) -> None:
    """Two-panel diagnostic figure for one hyperpolarized T1 test case.

    Panel 1 – Mz(t) simulation vs expected decay curve.
    Panel 2 – (Mz − 1) / (k − 1) on a log scale;
               should lie on exp(−t/T1).
    """
    t_s = simu.getTimeStamp().to_value(unit.s)
    Mz = simu.trjry[0, :, 2]
    expected_curve = _hyperpol_expected_curve(simu)
    Mz0 = float(simu.trjry[0, 0, 2])

    cm = 1 / 2.54
    marksize = 1
    fig = plt.figure(figsize=(2 * 8.5 * cm, 2 * 0.4 * 8.5 * cm), dpi=300)
    gs = gridspec.GridSpec(nrows=1, ncols=2)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    axes: list[Axes] = [ax00, ax01]

    ax: Axes = axes[0]
    ax.scatter(t_s, Mz, label="$M_z$ simulation", s=marksize, color="tab:red")
    ax.plot(
        t_s, expected_curve, label="$1+(k-1)e^{-t/T_1}$ expected", color="tab:purple"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$M_z / M_\\mathrm{eqb}$")
    ax.set_title("$M_z$ T1 decay")
    ax.legend()

    ax = axes[1]
    norm_sim = (Mz - 1.0) / (Mz0 - 1.0)
    norm_exp = np.exp(-t_s / simu.sample.T1.to_value(unit.s))
    # avoid log(0) at t=0 end
    valid = norm_sim > 0
    ax.semilogy(
        t_s[valid],
        norm_sim[valid],
        label="$(M_z-1)/(k-1)$ simulation",
        color="tab:red",
        marker="o",
        markersize=marksize,
        linestyle="none",
    )
    ax.semilogy(t_s, norm_exp, label="$e^{-t/T_1}$ expected", color="tab:purple")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("")
    ax.set_title("Exponential decay (log scale)")
    ax.legend()

    fig.suptitle(
        f"Hyperpolarized T1  RCF={simu.RCF_freq:.3g}  "
        f"$T_1$={T1:.3g}  $k$={k:.4g}\n"
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
@pytest.mark.parametrize("k", _OVERPOL_CASES, ids=lambda x: f"{x:.4g}x")
def test_t1_hyperpolarized(
    RCF_freq: Quantity,
    T1: Quantity,
    k: float,
    show_plots: bool,
):
    """Hyperpolarized Mz(t) decays to thermal equilibrium with time constant T1.

    The sample starts with Mz(0) = k > 1 (k times M0_eqb) and no transverse
    magnetization.  Without any RF excitation, the longitudinal component
    decays as

        Mz(t) = 1 + (k − 1) · exp(−t / T1).

    Tolerance: χ² = ‖Mz − Mz_expected‖² / ‖Mz_expected − 1‖² ≤ _CHI2_TOLERANCE.
    """
    duration = _N_T1 * T1
    rate = _N_PER_T1 / T1

    simu = _build_hyperpol_simulation(RCF_freq, T1, k, rate, duration)

    Mz = simu.trjry[0, :, 2]
    expected_curve = _hyperpol_expected_curve(simu)
    departure = expected_curve - 1.0  # (k − 1)·exp(−t/T1), always > 0

    chi2 = float(np.sum((Mz - expected_curve) ** 2) / np.sum(departure**2))

    if show_plots:
        _plot_hyperpol_result(simu, T1, k, chi2)

    assert chi2 <= _CHI2_TOLERANCE, (
        f"RCF_freq={RCF_freq}, T1={T1:.3g}, k={k:.4g}: "
        f"chi2={chi2:.2e} (tol={_CHI2_TOLERANCE:.0e})"
    )
