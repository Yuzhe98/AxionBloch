"""Calibration tests – CW (continuous-wave) NMR signal buildup.

Usage
-----
Run all 24 cases::

    pytest tests/test_calibrations_cw_nmr.py

Run a single case with a diagnostic plot::

    pytest "tests/test_calibrations_cw_nmr.py::test_cw_signal_buildup[1ppm-1.0-1e+06Hz]" --show-plots

Filter by FWHM and frequency::

    pytest tests/test_calibrations_cw_nmr.py --show-plots -k "1ppm-1.0"

Run both calibration suites::

    pytest tests/test_calibrations_free_decay.py tests/test_calibrations_cw_nmr.py

CW NMR calibration
------------------
A weak continuous-wave (CW) RF drive is applied on resonance while the
transverse magnetization |Mxy| = sqrt(Mx² + My²) is monitored over time.

Physics
-------
The simulator runs in a rotating frame at RCF_freq.  B0 is chosen so that

    ν_L = RCF_freq − signal_offset  (Larmor below carrier),

and the drive frequency is nu_rot = signal_offset  (exactly on resonance in
the rotating frame).

In the sub-rotating frame (co-rotating with the spins at signal_offset) the
drive is a static field along x, and the on-resonance Bloch equations give, in
the linear-response (weak-drive) limit:

    M+_sub(t) = γ · B₁_eff · M₀ · ∫₀ᵗ FD_sub(t′) dt′

where

    FD_sub(t) = Σᵢ wᵢ · exp(2πi · δᵢ · t)

is the free-decay kernel for the Hamming-windowed spin-packet distribution, and

    δᵢ = γ/(2π) · B_spread_i − (RCF_freq − signal_offset)

is the detuning of spin packet i in the sub-rotating frame.

The transverse magnitude is the same in the rotating and sub-rotating frames:

    |Mxy(t)| = |M+_sub(t)| = γ · B₁_eff · |∫₀ᵗ FD_sub(t′) dt′|

This is the formula used as the reference in each assertion.

Note on the analytic approximation
-----------------------------------
For a pure Lorentzian P(δ) with FWHM Γ = 1/(π·T₂*):

    ∫₀ᵗ FD_sub(t′) dt′  →  T₂* · (1 − e^{−t/T₂*})   as t → ∞

so the ensemble amplitude simplifies to

    |Mxy(t)| / M₀ = γ · B₁_eff · T₂* · (1 − e^{−t/T₂*}).

The Magnet uses a Hamming-squared window on the spin-packet weights which
modifies the effective lineshape and makes T₂*_actual ≈ 1.8 × T₂*_analytic.
Rather than correcting T₂*_analytic, the test computes the expected amplitude
directly from the FD integral so no approximation is needed.

Weak-drive condition
--------------------
B₁_eff is chosen so that the normalized steady-state amplitude:

    γ · B₁_eff · T₂*_analytic = TIP_ANGLE  (= 0.01 rad)

which sets

    B₁_input = 2 · TIP_ANGLE / (γ · T₂*_analytic)

The weak-drive approximation requires γ · B₁_eff · t_end << 1 rad.  With
t_end = 5 · T₂*_analytic and B₁_eff = TIP_ANGLE/(γ · T₂*_analytic):

    γ · B₁_eff · t_end = 5 · TIP_ANGLE = 0.05 rad  ✓

Adaptive simulation window
--------------------------
    duration = min(N_T2STAR × T₂*_analytic, MAX_DURATION)
    rate     = max(MIN_RATE, N_PER_T2STAR / T₂*_analytic)

This keeps the total RK4 step count ≈ N_T2STAR × N_PER_T2STAR = 500 and
the number of spin packets ≈ 43 regardless of the FWHM or RCF_freq.

Parametrisation
---------------
  • four FWHM values: 0.1, 1, 10, 20 ppm
  • three RCF frequencies: 1 kHz, 1 MHz, 1 GHz
  • two signal offsets: 1.0, 5.0 Hz
  Total: 24 test cases, all expected to pass.
"""

import pytest
from axionbloch.dependency import *
from axionbloch.Apparatus import Magnet
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation
from axionbloch.constants import gamma_p, mu_p

# ---------------------------------------------------------------------------
# Shared simulation parameters
# ---------------------------------------------------------------------------
_T1 = 1e6 * unit.s  # negligible longitudinal relaxation
_T2 = 1e6 * unit.s  # negligible intrinsic T2 → T2* ≈ Tdelta
_NFWHM = 10.0  # half-width range for spin-packet sampling

# Weak-drive condition: γ·B₁_eff·t_end = 5·TIP_ANGLE << 1 rad
_TIP_ANGLE = 0.001 * unit.rad  # γ·B₁_eff·T₂*_analytic (rad)
_TOLERANCE = 0.05  # 5 % relative tolerance on the CW amplitude

# Adaptive timing (same logic as free-decay calibration)
_MAX_DURATION = 5.0 * unit.s  # observation window cap
_MIN_RATE = 200.0 * unit.Hz
_N_T2STAR = 10.0  # observe for 10 × T₂*_analytic
_N_PER_T2STAR = 100  # RK4 samples per T₂*_analytic

# ---------------------------------------------------------------------------
# Parametrisation
# ---------------------------------------------------------------------------
_FWHM_CASES = [0.1 * ppm, 1.0 * ppm, 10.0 * ppm, 20.0 * ppm]
_RCF_FREQS = [1.0 * unit.kHz, 1.0 * unit.MHz, 1.0 * unit.GHz]
_SIGNAL_OFFSETS = [1.0 * unit.Hz, 5.0 * unit.Hz]

# gyromagnetic ratio (rad Hz / T) — used at numpy boundaries only
_GAMMA = gamma_p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_cw_simulation(
    RCF_freq: Quantity,
    signal_offset: Quantity,
    FWHM: Quantity,
    B1: Quantity,
    rate: Quantity,
    duration: Quantity,
) -> Simulation:
    """Build and run a CW NMR simulation.

    Parameters
    ----------
    RCF_freq:
        Rotating-frame carrier frequency.
    signal_offset:
        Larmor offset below ``RCF_freq``; drive is placed on resonance.
    FWHM:
        Fractional field inhomogeneity (ppm or dimensionless).
    B1:
        Peak B1 amplitude.  The effective RWA amplitude is B1/2.
    rate:
        Simulation sampling rate.
    duration:
        Observation window.
    """
    FWHM_dimless = FWHM.to(unit.one)

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

    B0 = (RCF_freq - signal_offset) / (sample.gamma / (2 * PI))

    magnet = Magnet(
        name="calibration_magnet",
        B0=B0,
        FWHM=FWHM_dimless,
        nFWHM=_NFWHM,
    )

    simu = Simulation(
        name=(
            f"cw_{RCF_freq.to_value(unit.Hz):.3g}Hz"
            f"_{signal_offset.to_value(unit.Hz):.1f}Hz"
            f"_{FWHM.to_value(ppm):.4g}ppm"
        ),
        sample=sample,
        magnet=magnet,
        excField=MagField(name="cw_drive"),
        RCF_freq=RCF_freq,
        rate=rate,
        duration=duration,
        verbose=False,
    )

    # CW drive on resonance: nu_rot = signal_offset → drive co-rotates with spins
    simu.excField.setXYPulse(
        timeStep=simu.timeStep,
        timeLen=simu.timeLen,
        B1=B1,
        nu_rot=signal_offset,
    )

    simu.generateTrajectories(integrator="RK4")
    return simu


def _cw_expected_amplitude(
    simu: Simulation,
    B1_eff: Quantity,
    signal_offset: Quantity,
) -> float:
    """Expected |Mxy(t_end)| from the free-decay integral formula.

    Computes γ · B₁_eff · |∫₀^t_end FD_sub(t) dt| where FD_sub is
    evaluated using the spin-packet distribution of ``simu.magnet`` in the
    sub-rotating frame co-rotating with the drive at ``signal_offset``.

    Each spin packet i contributes exp(2πi·δᵢ·t) with detuning
    δᵢ = γ/(2π)·B_spread_i − (RCF_freq − signal_offset).
    """
    rcf_Hz = simu.RCF_freq.to_value(unit.Hz)
    sig_Hz = signal_offset.to_value(unit.Hz)
    B1_eff_T = B1_eff.to_value(unit.T)

    delta_nu_i = _GAMMA.to_value(unit.rad * unit.Hz / unit.T) / (
        2 * np.pi
    ) * simu.magnet.B_spread.to_value(unit.T) - (rcf_Hz - sig_Hz)
    t_s = simu.getTimeStamp().to_value(unit.s)
    dt = t_s[1] - t_s[0]
    fd = np.zeros(len(t_s), dtype=complex)
    for dnu, w in zip(delta_nu_i, simu.magnet.ratios):
        fd += w * np.exp(2j * np.pi * dnu * t_s)
    cw_integral = np.cumsum(fd) * dt
    return float(
        _GAMMA.to_value(unit.rad * unit.Hz / unit.T)
        * B1_eff_T
        * np.abs(cw_integral[-1])
    )


# ---------------------------------------------------------------------------
# Optional diagnostic plots
# ---------------------------------------------------------------------------


def _plot_cw_result(
    simu: Simulation,
    B1_eff: Quantity,
    signal_offset: Quantity,
    T2star: Quantity,
    expected: float,
) -> None:
    """Show a two-panel diagnostic figure for one CW NMR test case.

    Panel 1 – time-domain Mx, My (shows oscillation at signal_offset while
               the envelope builds up).
    Panel 2 – |Mxy(t)| buildup envelope with the FD-integral expected curve
               and the analytic Lorentzian approximation for comparison.
    """
    # Extract floats at the numpy/matplotlib boundary
    sig_Hz = signal_offset.to_value(unit.Hz)
    B1_eff_T = B1_eff.to_value(unit.T)
    T2star_s = T2star.to_value(unit.s)
    rcf_Hz = simu.RCF_freq.to_value(unit.Hz)

    t_s = simu.getTimeStamp().to_value(unit.s)
    Mx = simu.trjry[0, :, 0]
    My = simu.trjry[0, :, 1]
    Mxy = np.sqrt(Mx**2 + My**2)

    # FD-integral running envelope (re-use helper computation)
    delta_nu_i = _GAMMA.to_value(unit.rad * unit.Hz / unit.T) / (
        2 * np.pi
    ) * simu.magnet.B_spread.to_value(unit.T) - (rcf_Hz - sig_Hz)
    dt = t_s[1] - t_s[0]
    fd = np.zeros(len(t_s), dtype=complex)
    for dnu, w in zip(delta_nu_i, simu.magnet.ratios):
        fd += w * np.exp(2j * np.pi * dnu * t_s)
    expected_curve = (
        _GAMMA.to_value(unit.rad * unit.Hz / unit.T)
        * B1_eff_T
        * np.abs(np.cumsum(fd) * dt)
    )

    # Analytic Lorentzian approximation (for reference)
    analytic_curve = (
        np.sqrt(2) * _TIP_ANGLE.to_value(unit.rad) * (1.0 - np.exp(-t_s / T2star_s))
    )

    cm = 1 / 2.54  # convert cm to inch
    fig = plt.figure(
        figsize=(2 * 8.5 * cm, 2 * 0.4 * 8.5 * cm), dpi=300
    )  # initialize a figure following APS journal requirements
    gs = gridspec.GridSpec(nrows=1, ncols=2)  # create grid for multiple figures
    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    axes: list[Axes] = [ax00, ax01]

    ax: Axes = axes[0]
    ax.plot(t_s, Mx, label="$M_x / M_\\mathrm{eqb}$")
    ax.plot(t_s, My, label="$M_y / M_\\mathrm{eqb}$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$M / M_\\mathrm{eqb}$")
    ax.set_title("$M_x$, $M_y$ vs time")
    ax.legend()

    ax = axes[1]
    ax.plot(t_s, Mxy, lw=1.5, label="$|M_{xy} / M_\\mathrm{eqb}|$ simulation")
    ax.plot(t_s, expected_curve, "--", lw=1.5, label="FD amplitude (expected)")
    ax.plot(
        t_s,
        analytic_curve,
        ":",
        lw=1.2,
        color="gray",
        label=f"Analytic T₂*={T2star:.3g}",
    )
    # ax.axhline(
    #     expected, color="C1", ls="--", alpha=0.4, label="Final amplitude (expected)"
    # )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$M / M_\\mathrm{eqb}$")
    ax.set_title("$|M_{xy}|$ buildup")
    ax.legend()

    fig.suptitle(
        f"CW NMR  RCF={simu.RCF_freq:.3g}  "
        f"FWHM={simu.magnet.FWHM.to(ppm):.4g}  "
        f"signal={signal_offset:.4g}\n"
        "$T_2^*$" + f"analytic={T2star:.3g}  "
        f"error={abs(float(Mxy[-1]) - expected) / expected * 100:.2f} %"
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "RCF_freq", _RCF_FREQS, ids=lambda f: f"{f.to_value(unit.Hz):.3g}Hz"
)
@pytest.mark.parametrize(
    "signal_offset", _SIGNAL_OFFSETS, ids=lambda f: f"{f.to_value(unit.Hz):.4g}"
)
@pytest.mark.parametrize("FWHM", _FWHM_CASES, ids=lambda f: f"{f.to_value(ppm):.4g}ppm")
def test_cw_signal_buildup(
    RCF_freq: Quantity,
    signal_offset: Quantity,
    FWHM: Quantity,
    show_plots: bool,
):
    """CW signal amplitude matches γ·B₁_eff·|∫FD dt| at t = t_end.

    ``ν_L = RCF_freq − signal_offset``  (spins below carrier)
    Drive: ``nu_rot = signal_offset`` → on resonance in rotating frame.

    Drive amplitude ``B₁_input = 2·TIP_ANGLE / (γ·T₂*_analytic)`` ensures
    ``γ·B₁_eff·t_end = 5·TIP_ANGLE = 0.05 rad`` (weak-drive condition).

    The expected amplitude is computed from the free-decay integral formula,
    which accounts for the Hamming-windowed spin-packet distribution exactly
    and avoids assumptions about the functional form of T₂* decay.
    """
    Tdelta = (1.0 / (np.pi * FWHM.to(unit.one) * RCF_freq)).to(unit.s)
    T2star = (_T2 * Tdelta / (_T2 + Tdelta)).to(unit.s)  # harmonic mean ≈ Tdelta

    # Adaptive observation window
    duration = min(_N_T2STAR * T2star, _MAX_DURATION)

    # Adaptive rate: enough RK4 samples per T₂* for accurate integration
    rate = max(_MIN_RATE, (_N_PER_T2STAR / T2star).to(unit.Hz))

    # Weak drive: γ·B₁_eff·T₂*_analytic = TIP_ANGLE → B₁_input = 2·TIP_ANGLE/(γ·T₂*)
    B1_input = (2.0 * _TIP_ANGLE / (np.abs(_GAMMA) * T2star)).to(unit.T)
    B1_eff = 0.5 * B1_input

    simu = _build_cw_simulation(RCF_freq, signal_offset, FWHM, B1_input, rate, duration)

    # |Mxy(t)| = sqrt(Mx² + My²) is the smooth buildup envelope (not oscillating)
    Mxy_end = float(np.sqrt(simu.trjry[0, -1, 0] ** 2 + simu.trjry[0, -1, 1] ** 2))

    # Expected amplitude from exact linear-response formula for the actual distribution
    expected = _cw_expected_amplitude(simu, B1_eff, signal_offset)

    t_end = simu.getTimeStamp()[-1]
    nutation = (np.abs(_GAMMA) * B1_eff * t_end).to(unit.rad)

    if show_plots:
        _plot_cw_result(simu, B1_eff, signal_offset, T2star, expected)

    assert abs(Mxy_end - expected) <= _TOLERANCE * expected, (
        f"RCF_freq={RCF_freq}, FWHM={FWHM.to(ppm):.4g}, "
        f"T2*_analytic={T2star:.3g}, signal_offset={signal_offset}: "
        f"B1_input={B1_input:.3g} (γ·B1_eff·t_end={nutation:.3f}), "
        f"t_end={t_end:.4g} ({(t_end / T2star).to_value(unit.one):.2f}·T2*_analytic), "
        f"expected |Mxy|={expected:.5g}, got {Mxy_end:.5g} "
        f"(err={abs(Mxy_end - expected) / expected * 100:.1f}%, "
        f"tol={_TOLERANCE * 100:.0f}%)"
    )
