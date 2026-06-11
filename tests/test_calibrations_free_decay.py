"""Calibration tests – free-decay signal.

Usage
-----
Run all 108 cases::

    pytest tests/test_calibrations_free_decay.py

Run a single case with a diagnostic plot::

    pytest "tests/test_calibrations_free_decay.py::test_free_decay_peak_frequency[0.0-1ppm-1.0-1e+06Hz]" --show-plots

Filter by FWHM and frequency::

    pytest tests/test_calibrations_free_decay.py --show-plots -k "1ppm and 1e+06Hz"

Run both calibration suites::

    pytest tests/test_calibrations_free_decay.py tests/test_calibrations_cw_nmr.py

Free-decay calibration
----------------------
A 90° pulse is applied and the free-decay signal (Mx + i·My) is observed
in the rotating frame.  The test sweeps over four magnet homogeneity
values (in ppm), three RCF (carrier) frequencies, three Larmor-frequency
offsets, and three demodulation detunings.

Physics
-------
The simulator runs in a rotating frame at ``RCF_freq``.  We set

    B0 = (RCF_freq - signal_offset) / (γ / 2π)

so that the Larmor frequency is

    ν_L = γ B0 / 2π = RCF_freq - signal_offset.

The z-axis offset in the rotating frame is

    Ω_z = 2π (ν_L − RCF_freq) = −2π · signal_offset.

The Bloch equation for M+ = Mx + iMy gives dM+/dt = −i Ω_z M+, so
after a 90° pulse M+ oscillates at

    f_signal = +signal_offset.   (positive)

The free-decay envelope decays at the effective rate

    1/T2star = 1/T2 + 1/Tdelta,

where T2 is the intrinsic relaxation time and
Tdelta = 1/(π · FWHM_freq) is the dephasing time from magnet
inhomogeneity.  FWHM is specified in ppm (parts per million) of B0:

    FWHM_freq  = FWHM_ppm × 1e-6 × RCF_freq      (Hz)
    Tdelta     = 1 / (π × FWHM_ppm × 1e-6 × RCF_freq)

Because Tdelta ∝ 1/RCF_freq, the same ppm value gives a much shorter
dephasing time at higher field.

Adaptive simulation window
--------------------------
Rather than skipping short-T2star cases the simulation duration is set to
``10 × T2star`` (capped at ``_MAX_DURATION``) and the sampling rate is
set to ``max(_MIN_RATE, _N_PER_T2STAR / T2star)`` so that each run
always produces ~1 000 RK4 steps regardless of T2star.  The FFT
frequency resolution is ``1 / duration``; the assertion tolerance is
``2 × resolution``.  For very short T2star the tolerance is coarse (poor
frequency resolution), so the test verifies correct simulation execution;
for long T2star it is tight and gives a precise frequency calibration.

Applying digital demodulation at ``f_demod = f_signal + δ`` shifts the
oscillation to

    f_peak = f_signal − f_demod = −δ.

We parametrise over:
  • four FWHM values: 0.1, 1, 10, 20 ppm
  • three RCF frequencies: 1 kHz, 1 MHz, 1 GHz
  • three signal offsets: 1.0, 2.5, 5.0 Hz
  • three demodulation detunings δ: 0.0, 0.3, −0.7 Hz
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as unit

from axionbloch.Apparatus import Magnet
from axionbloch.Sample import Sample
from axionbloch.SimuTools import MagField, Simulation
from axionbloch.constants import gamma_p, mu_p
from axionbloch.dependency import ppm  # 1 ppm = 1e-6 (dimensionless)

PI = np.pi * unit.rad

# ---------------------------------------------------------------------------
# Shared simulation parameters
# ---------------------------------------------------------------------------
_T1 = 1e6 * unit.s  # negligible longitudinal relaxation
_T2 = 1e6 * unit.s  # negligible intrinsic T2 → T2star ≈ Tdelta
_T90_STEPS = 5  # 90° pulse length in time steps
_NFWHM = 5.0  # half-width range for spin-packet sampling

# Adaptive timing: duration = min(10 × T2star, _MAX_DURATION),
#                  rate     = max(_MIN_RATE, _N_PER_T2STAR / T2star)
_MAX_DURATION = 10.0 * unit.s
_MIN_RATE = 200.0 * unit.Hz
_N_PER_T2STAR = 100  # target samples per T2star for RK4 accuracy

# ---------------------------------------------------------------------------
# Parametrisation
# ---------------------------------------------------------------------------
# Four magnet homogeneity values in ppm
_FWHM_CASES = [0.1 * ppm, 1.0 * ppm, 10.0 * ppm, 20.0 * ppm]

# Three carrier (RCF) frequencies spanning audio, RF, and microwave bands
_RCF_FREQS = [1.0 * unit.kHz, 1.0 * unit.MHz, 1.0 * unit.GHz]

# Three Larmor-frequency offsets below RCF_freq: ν_L = RCF_freq − signal_offset
_SIGNAL_OFFSETS = [1.0 * unit.Hz, 2.5 * unit.Hz, 5.0 * unit.Hz]

# Three demodulation detunings δ: f_demod = f_signal + δ → peak at −δ
_DEMOD_DETUNES = [0.0 * unit.Hz, 0.3 * unit.Hz, -0.7 * unit.Hz]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_free_decay_simulation(
    RCF_freq: unit.Quantity,
    signal_offset: unit.Quantity,
    FWHM: unit.Quantity,
    rate: unit.Quantity,
    duration: unit.Quantity,
) -> Simulation:
    """Build and run a 90°-pulse free-decay simulation.

    Parameters
    ----------
    RCF_freq:
        Rotating-frame carrier frequency.
    signal_offset:
        Larmor offset below ``RCF_freq``.
    FWHM:
        Fractional field homogeneity (in ppm or dimensionless).
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
            f"free_decay_{RCF_freq.to_value(unit.Hz):.3g}Hz"
            f"_{signal_offset.to_value(unit.Hz):.1f}Hz"
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
        nu_rot=signal_offset,
    )

    simu.generateTrajectories(integrator="RK4")
    return simu


def _free_decay_peak_frequency(
    simu: Simulation, t90_steps: int, demodfreq: unit.Quantity
) -> unit.Quantity:
    """Return the FFT peak frequency of the demodulated free-decay signal.

    Parameters
    ----------
    simu:
        Completed simulation (``trjry`` populated).
    t90_steps:
        Trajectory steps occupied by the excitation pulse to skip.
    demodfreq:
        Digital demodulation frequency in the rotating frame.
    """
    t_s = simu.getTimeStamp().to_value(unit.s)
    t_sig = t_s[t90_steps:]
    demodfreq_Hz = demodfreq.to_value(unit.Hz)

    sig = simu.trjry[0, t90_steps:, 0] + 1j * simu.trjry[0, t90_steps:, 1]
    demod_sig = sig * np.exp(-2j * np.pi * demodfreq_Hz * t_sig)

    N = len(demod_sig)
    dt = t_sig[1] - t_sig[0]
    spectrum = np.fft.fftshift(np.fft.fft(demod_sig))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=dt))
    return freqs[np.argmax(np.abs(spectrum))] * unit.Hz


# ---------------------------------------------------------------------------
# Optional diagnostic plots
# ---------------------------------------------------------------------------


def _plot_free_decay_result(
    simu: Simulation,
    t90_steps: int,
    demodfreq: unit.Quantity,
    expected_freq: unit.Quantity,
    peak: unit.Quantity,
    freq_resolution: unit.Quantity,
) -> None:
    """Show a two-panel diagnostic figure for one free-decay test case.

    Panel 1 – raw Mx(t) and My(t) after the 90° pulse.
    Panel 2 – FFT magnitude of the demodulated signal with the expected peak
               and tolerance band marked.
    """
    # Extract floats at the matplotlib boundary
    demodfreq_Hz = demodfreq.to_value(unit.Hz)
    expected_Hz = expected_freq.to_value(unit.Hz)
    peak_Hz = peak.to_value(unit.Hz)
    freq_res_Hz = freq_resolution.to_value(unit.Hz)

    t_s = simu.getTimeStamp().to_value(unit.s)
    t_sig = t_s[t90_steps:]
    Mx = simu.trjry[0, t90_steps:, 0]
    My = simu.trjry[0, t90_steps:, 1]
    sig = Mx + 1j * My
    demod = sig * np.exp(-2j * np.pi * demodfreq_Hz * t_sig)
    N = len(demod)
    dt = t_sig[1] - t_sig[0]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(demod)))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=dt))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(t_sig, Mx, lw=0.8, label="Mx")
    ax.plot(t_sig, My, lw=0.8, label="My")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Magnetisation (normalised)")
    ax.set_title("Free-decay signal (rotating frame)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(freqs, spectrum, lw=1.0)
    ax.axvline(
        expected_Hz, color="C1", lw=1.5, ls="--", label=f"expected {expected_freq:.3g}"
    )
    ax.axvline(peak_Hz, color="C2", lw=1.0, ls=":", label=f"detected {peak:.3g}")
    ax.axvspan(
        expected_Hz - 2 * freq_res_Hz,
        expected_Hz + 2 * freq_res_Hz,
        alpha=0.15,
        color="C1",
        label=f"±2·Δf tol",
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|FFT|")
    ax.set_title("Demodulated spectrum")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Free decay  RCF={simu.RCF_freq:.3g}  "
        f"FWHM={simu.magnet.FWHM.to(ppm):.4g}\n"
        f"demod={demodfreq:.3g}  "
        f"peak err={abs(peak - expected_freq):.4g}  "
        f"tol=±{2 * freq_resolution:.4g}"
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
@pytest.mark.parametrize(
    "demod_detune", _DEMOD_DETUNES, ids=lambda d: f"{d.to_value(unit.Hz):.4g}"
)
def test_free_decay_peak_frequency(
    RCF_freq: unit.Quantity,
    signal_offset: unit.Quantity,
    FWHM: unit.Quantity,
    demod_detune: unit.Quantity,
    show_plots: bool,
):
    """Free-decay peak matches expected detuning after digital demodulation.

    ``ν_L = RCF_freq − signal_offset``
    → M+ oscillates at ``f_signal = +signal_offset``.
    Envelope decays with ``T2star = T2 · Tdelta / (T2 + Tdelta) ≈ Tdelta``.
    Simulation window = ``min(10 · T2star, _MAX_DURATION)``;
    sampling rate    = ``max(_MIN_RATE, _N_PER_T2STAR / T2star)``.
    Demodulating at ``f_demod = f_signal + demod_detune``
    → free-decay peak at ``f_peak = −demod_detune``.
    Tolerance = 2 × FFT frequency resolution = 2 / (N_sig × dt).
    """
    Tdelta = (1.0 / (np.pi * FWHM.to(unit.one) * RCF_freq)).to(unit.s)
    T2star = (_T2 * Tdelta / (_T2 + Tdelta)).to(unit.s)  # harmonic mean ≈ Tdelta

    # Adaptive observation window: 10 × T2star, capped at _MAX_DURATION
    duration = min(10.0 * T2star, _MAX_DURATION)

    # Adaptive rate: enough samples per T2star for accurate RK4 integration
    rate = max(_MIN_RATE, (_N_PER_T2STAR / T2star).to(unit.Hz))

    simu = _build_free_decay_simulation(RCF_freq, signal_offset, FWHM, rate, duration)

    f_demod = signal_offset + demod_detune
    peak = _free_decay_peak_frequency(simu, t90_steps=_T90_STEPS, demodfreq=f_demod)
    expected_freq = -demod_detune

    t = simu.getTimeStamp()
    dt = t[1] - t[0]
    N_sig = simu.timeLen - _T90_STEPS
    freq_resolution = (1.0 / (N_sig * dt)).to(unit.Hz)

    if show_plots:
        _plot_free_decay_result(
            simu, _T90_STEPS, f_demod, expected_freq, peak, freq_resolution
        )

    assert abs(peak - expected_freq) <= 2.0 * freq_resolution, (
        f"RCF_freq={RCF_freq}, FWHM={FWHM.to(ppm):.4g}, "
        f"T2*={T2star:.3g}, signal_offset={signal_offset}, "
        f"demod_detune={demod_detune}: "
        f"expected peak at {expected_freq:.4f}, got {peak:.4f} "
        f"(resolution={freq_resolution:.4f}, "
        f"tolerance={2 * freq_resolution:.4f})"
    )
