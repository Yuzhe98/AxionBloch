"""Calibration tests – free-decay signal envelope.

Usage
-----
Run all 36 cases::

    pytest tests/test_calibrations_free_decay.py

Run a single case with a diagnostic plot::

    pytest "tests/test_calibrations_free_decay.py::test_free_decay_envelope[1ppm-1-1e+06Hz]" --show-plots

Filter by FWHM and frequency::

    pytest tests/test_calibrations_free_decay.py --show-plots -k "[1ppm-"

Run both calibration suites::

    pytest tests/test_calibrations_free_decay.py tests/test_calibrations_cw_nmr.py

Free-decay calibration
----------------------
A 90° pulse is applied and the free-decay amplitude envelope |Mxy(t)| =
sqrt(Mx² + My²) is compared to the analytically expected free-decay kernel.

Physics
-------
The simulator runs in a rotating frame at RCF_freq.  B0 is chosen so that

    ν_L = RCF_freq − Delta_nu_L  (Larmor below carrier),

where Delta_nu_L = rel_detuning × RCF_freq.  After a 90° pulse the
transverse magnetisation precesses freely.  In the rotating frame:

    M+(t) = exp(−2πi · Delta_nu_L · t) · FD_sub(t)

where the free-decay kernel is

    FD_sub(t) = Σᵢ wᵢ · exp(2πi · δᵢ · t),

    δᵢ = γ/(2π) · B_spread_i

is the detuning of spin packet i from the centre Larmor frequency, and
wᵢ are the Hamming-squared weights from the magnet model.  The amplitude
envelope is independent of Delta_nu_L:

    |Mxy(t)| = |FD_sub(t)|.

The test quantifies agreement between simulation and theory via the full
transverse-vector relative L2 error. ``epsilon_l2_squared`` is retained as
the squared form used for the tolerance check.

Simulation parameters
---------------------
    duration = N_T2STAR × T₂*_analytic
    rate     = max(N_PER_T2STAR × Delta_nu_L, N_PER_T2STAR / T₂*_analytic)

This keeps the total RK4 step count ≈ N_T2STAR × N_PER_T2STAR = 2000
regardless of FWHM or RCF_freq.

Parametrisation
---------------
  • four FWHM values:        0.1, 1, 10, 20 ppm
  • three RCF frequencies:   1 kHz, 1 MHz, 1 GHz
  • three Larmor detunings:  0, 1, 10 ppm of RCF_freq
  Total: 36 test cases, all expected to pass.
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
_T1 = 1e6 * unit.s  # negligible longitudinal relaxation
_T2 = 1e3 * unit.s  # negligible intrinsic T2 → T2* ≈ Tdelta
_NFWHM = 10.0  # half-width range; larger nFWHM adds outer packets that
# precess ~1 rad during the 90° pulse, raising the vector error
_T90_STEPS = 5  # 90° pulse length in time steps

# Squared relative L2 tolerance over the full post-pulse trajectory.
_EPSILON_L2_SQUARED_TOLERANCE = 1e-4

# Adaptive timing
_N_T2STAR = 10.0  # observe for 10 × T₂*_analytic
_N_PER_T2STAR = 500  # RK4 samples per T₂*_analytic

# ---------------------------------------------------------------------------
# Parametrisation
# ---------------------------------------------------------------------------
_FWHM_CASES = [0.1 * ppm, 1.0 * ppm, 10.0 * ppm, 20.0 * ppm]
_RCF_FREQS = [1.0 * unit.kHz, 1.0 * unit.MHz, 1.0 * unit.GHz]
_REL_DETUNINGS = [0.0 * ppm, 1.0 * ppm, 10.0 * ppm]

_GAMMA = gamma_p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_free_decay_simulation(
    RCF_freq: Quantity,
    Delta_nu_L: Quantity,
    FWHM: Quantity,
    rate: Quantity,
    duration: Quantity,
) -> Simulation:
    """Build and run a 90°-pulse free-decay simulation.

    Parameters
    ----------
    RCF_freq:
        Rotating-frame carrier frequency.
    Delta_nu_L:
        Larmor offset below RCF_freq: ν_L = RCF_freq − Delta_nu_L.
    FWHM:
        Fractional field inhomogeneity (ppm or dimensionless).
    rate:
        Simulation sampling rate.
    duration:
        Observation window.
    """
    sample = Sample(
        name="calibration_proton",
        gamma=gamma_p,
        massDensity=0.789 * unit.g / unit.cm**3,
        molarMass=46.069 * unit.g / unit.mol,
        numOfSpinsPerMolecule=6 * unit.one,
        T1=_T1,
        T2=_T2,
        vol=1.0 * unit.cm**3,
        mu=mu_p,
        temp=300.0 * unit.K,
        verbose=False,
    )

    B0 = (RCF_freq - Delta_nu_L) / (sample.gamma / (2 * PI))

    magnet = Magnet(
        name="calibration_magnet",
        B0=B0,
        FWHM=FWHM,
        nFWHM=_NFWHM,
    )

    simu = Simulation(
        name=(
            f"fd_{RCF_freq.to_value(unit.Hz):.3g}Hz"
            f"_{Delta_nu_L.to_value(unit.Hz):.1f}Hz"
            f"_{FWHM.to_value(ppm):.4g}ppm"
        ),
        sample=sample,
        magnet=magnet,
        excField=MagField(name="90deg_pulse"),
        RCF_freq=RCF_freq,
        rate=rate,
        duration=duration,
        verbose=False,
    )

    simu.excField.set90DegPulse(
        timeStep=simu.timeStep,
        timeLen=simu.timeLen,
        gamma=simu.sample.gamma,
        t90=_T90_STEPS * simu.timeStep,
        nu_rot=Delta_nu_L,
    )

    simu.generateTrajectories(integrator="RK4")
    return simu


def _fd_expected_vector(
    simu: Simulation,
    t90_steps: int,
    Delta_nu_L: Quantity,
    initial_transverse: complex = 1j,
) -> np.ndarray:
    """Expected ``(Mx, My)`` trajectory after the 90-degree pulse.

    Returns an array with shape ``(simu.timeLen - t90_steps, 2)``.
    Each spin packet i contributes exp(-2πi · δᵢ · t), where
    δᵢ = γ/(2π) · B_i − RCF_freq is the rotating-frame detuning.
    """
    t_s = simu.getTimeStamp().to_value(unit.s)
    t_sig = t_s[t90_steps:] - t_s[t90_steps]  # time since end of pulse

    gamma_over_2pi = _GAMMA.to_value(unit.rad * unit.Hz / unit.T) / (2 * np.pi)
    B_spread = simu.magnet.B_spread.to_value(unit.T)
    detuning_Hz = gamma_over_2pi * B_spread - simu.RCF_freq.to_value(unit.Hz)

    T2_s = simu.sample.T2.to_value(unit.s)
    fd = np.zeros(len(t_sig), dtype=complex)
    for detuning, w in zip(detuning_Hz, simu.magnet.ratios):
        fd += w * np.exp(-2j * np.pi * detuning * t_sig)
    transverse = (
        initial_transverse * fd * np.exp(-t_sig / T2_s)
    )
    return np.column_stack((transverse.real, transverse.imag))


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


def _plot_free_decay_result(
    simu: Simulation,
    t90_steps: int,
    Delta_nu_L: Quantity,
    T2star: Quantity,
    epsilon_l2: float,
) -> None:
    """Show a two-panel diagnostic figure for one free-decay test case.

    Panel 1 – Mx(t), My(t) after the 90° pulse (oscillation at Delta_nu_L).
    Panel 2 – |Mxy(t)| vs expected envelope |FD_sub(t)|.
    """
    t_s = simu.getTimeStamp().to_value(unit.s)
    t_sig = t_s[t90_steps:]
    Mx = simu.trjry[0, t90_steps:, 0]
    My = simu.trjry[0, t90_steps:, 1]
    Mxy = np.sqrt(Mx**2 + My**2)
    expected_vector = _fd_expected_vector(
        simu, t90_steps, Delta_nu_L, Mx[0] + 1j * My[0]
    )
    expected_curve = np.linalg.norm(expected_vector, axis=1)

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
    ax.scatter(t_sig, Mxy, label="$|M_{xy}|$ simulation", s=marksize, color="tab:red")
    ax.plot(
        t_sig,
        expected_curve,
        label="$|\\mathrm{FD}_\\mathrm{sub}|$ expected",
        color="tab:purple",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$M / M_\\mathrm{eqb}$")
    ax.set_title("$|M_{xy}|$ envelope")
    ax.legend()

    fig.suptitle(
        f"Free decay  RCF={simu.RCF_freq:.3g}  "
        f"FWHM={simu.magnet.FWHM.to(ppm):.4g}  "
        f"signal={Delta_nu_L.to(unit.Hz):.4g}\n"
        "$T_2^*$" + f"_analytic={T2star:.3g}  "
        f"$\\epsilon_{{L^2}}$={epsilon_l2:.2e}"
    )
    plt.tight_layout()
    _show_figure(fig, simu.name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "RCF_freq", _RCF_FREQS, ids=lambda f: f"{f.to_value(unit.Hz):.3g}Hz"
)
@pytest.mark.parametrize(
    "rel_detuning", _REL_DETUNINGS, ids=lambda f: f"{f.to_value(ppm):.4g}"
)
@pytest.mark.parametrize("FWHM", _FWHM_CASES, ids=lambda f: f"{f.to_value(ppm):.4g}ppm")
def test_free_decay_envelope(
    RCF_freq: Quantity,
    rel_detuning: Quantity,
    FWHM: Quantity,
    show_plots: bool,
):
    """Free-decay |Mxy(t)| envelope matches expected free-decay kernel.

    ν_L = RCF_freq − Delta_nu_L  (spins below carrier).
    After 90° pulse, |Mxy(t)| = |FD_sub(t)| = |Σᵢ wᵢ exp(2πi · δᵢ · t)|
    where δᵢ = γ/(2π) · B_spread_i.

    Tolerance: the full-vector relative L2 error satisfies
    ``epsilon_l2_squared <= _EPSILON_L2_SQUARED_TOLERANCE``.
    """
    Tdelta = (1.0 / (np.pi * FWHM.to(unit.one) * RCF_freq)).to(unit.s)
    T2star = (_T2 * Tdelta / (_T2 + Tdelta)).to(unit.s)
    Delta_nu_L = rel_detuning * RCF_freq

    duration = _N_T2STAR * T2star
    rate = max(_N_PER_T2STAR * Delta_nu_L, (_N_PER_T2STAR / T2star))

    simu = _build_free_decay_simulation(RCF_freq, Delta_nu_L, FWHM, rate, duration)

    Mperp_simu = simu.trjry[0, _T90_STEPS:, :2]
    initial_transverse = complex(Mperp_simu[0, 0], Mperp_simu[0, 1])
    Mperp_theo = _fd_expected_vector(
        simu, _T90_STEPS, Delta_nu_L, initial_transverse
    )
    epsilon_l2_squared = float(
        np.sum((Mperp_simu - Mperp_theo) ** 2) / np.sum(Mperp_theo**2)
    )
    epsilon_l2 = np.sqrt(epsilon_l2_squared)

    t_end = simu.getTimeStamp()[-1]

    if show_plots:
        _plot_free_decay_result(simu, _T90_STEPS, Delta_nu_L, T2star, epsilon_l2)

    assert epsilon_l2_squared <= _EPSILON_L2_SQUARED_TOLERANCE, (
        f"RCF_freq={RCF_freq}, FWHM={FWHM.to(ppm):.4g}, "
        f"T2*_analytic={T2star:.3g}, rel_detuning={rel_detuning}: "
        f"t_end={t_end:.4g} ({(t_end / T2star).to_value(unit.one):.2f}·T2*_analytic), "
        f"epsilon_l2={epsilon_l2:.2e}, "
        f"epsilon_l2_squared={epsilon_l2_squared:.2e} "
        f"(tol={_EPSILON_L2_SQUARED_TOLERANCE:.0e})"
    )
