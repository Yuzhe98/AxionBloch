# Example: liquid Xe-129 at 14 T excited by N pulses with arbitrary delays.
#
# Pulse sequence (3 pulses, deliberately unequal interpulse delays):
#
#   90°(x)  ──── τ₁ ────  180°(y)  ──── τ₁ ────  [echo₁]  ──── τ₁+τ₂ ────  180°(y)  ──── τ₂ ────  [echo₂]
#
# with τ₁ = 2·Tdelta, τ₂ = 4·Tdelta (≠ τ₁, demonstrating "arbitrary" delays).
# delays[i] is the free-precession interval BEFORE pulse i.
import time

from axionbloch.dependency import *
from axionbloch.SimuTools import MagField, Simulation
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_Xe129, mu_Xe129, gamma_p, mu_p
from axionbloch.utils import check
from axionbloch.kits.Simu_Signal import Simu_Signal

# ──────────────────────────────────────────────────────────────────────────── #
# Physical parameters (all as Quantity)
# ──────────────────────────────────────────────────────────────────────────── #


T1 = 15 * unit.min  # very long longitudinal relaxation (not the focus here)
T2 = 10.0 * unit.min  # transverse relaxation time
Tdelta = 1 * unit.ms  # dephasing time set by field inhomogeneity

# ──────────────────────────────────────────────────────────────────────────── #
# Sample
# ──────────────────────────────────────────────────────────────────────────── #

sample = Sample(
    name="Liquid Xe-129",
    gamma=gamma_Xe129,  # negative for Xe-129
    massDensity=3.1 * unit.g / unit.cm**3,
    molarMass=131.29 * unit.g / unit.mol,
    numOfSpinsPerMolecule=1 * unit.one,
    T2=T2,
    T1=T1,
    vol=1 * unit.cm**3,
    mu=mu_Xe129,
    temp=163 * unit.K,
    verbose=False,
)
# CH3CH2OH
# sample = Sample(
#     name="Ethanol",
#     gamma=gamma_p,
#     massDensity=0.78945 * unit.g / unit.cm**3,
#     molarMass=46.069 * unit.g / unit.mol,
#     numOfSpinsPerMolecule=6 * unit.one,
#     T2=T2,
#     T1=T1,
#     vol=1 * unit.cm**3,
#     mu=mu_p,
#     temp=300 * unit.K,
#     verbose=False,
# )

B0 = 14.0 * unit.T  # magnet field magnitude

# Larmor frequency of Xe-129 at 14 T  (take |gamma| since gamma_Xe129 < 0)
nu_L = (np.abs(sample.gamma) / (2 * PI) * B0).to(unit.MHz)  # ≈ 165.8 MHz
check(nu_L)

signalFreqRot = 0.0 * unit.Hz  # signal offset in the rotating frame
RCF_freq = nu_L - signalFreqRot  # rotating-frame carrier frequency

# ──────────────────────────────────────────────────────────────────────────── #
# Pulse sequence definition (all as Quantity)
# ──────────────────────────────────────────────────────────────────────────── #
Npulses = 100

tip_angles = (18 / 180) * PI * np.ones(Npulses)

# delays = np.arange(55, 55 * (Npulses + 1), 5) * unit.ms
delays = 10 * np.ones(Npulses) * unit.ms

phases = 0 * PI * np.ones(Npulses)
# phases[1:-1:2] += PI
# phases = np.random.uniform(0, 2 * np.pi, Npulses) * unit.rad

tip_angles = tip_angles[0:Npulses]
tip_angles[0] = (18 / 180) * PI
delays = delays[0:Npulses]
phases = phases[0:Npulses]

simuRate = 25 * unit.kHz
pulseDur = 5 / simuRate  # 5 time-steps; defined here so duration can include it
duration = (
    np.sum(delays) + Npulses * pulseDur + np.max(delays)
)  # all pulses + one full trailing window
check(duration.si)
# ──────────────────────────────────────────────────────────────────────────── #
# Magnet
# For Xe-129 (gamma < 0) the simulation requires B0 < 0 so that
# gamma * B0 > 0 and the Larmor frequency in the rotating frame is positive.
# Passing B0 directly as a Tesla Quantity avoids a T/rad unit artefact that
# arises when computing B0 = freq / (gamma / 2π).
# ──────────────────────────────────────────────────────────────────────────── #

FWHM = (1.0 / (np.pi * Tdelta) / nu_L).to(ppm)

magnet = Magnet(
    name="14 T detection magnet",
    B0=B0,
    FWHM=FWHM,
    nFWHM=10.0,
)
magnet.setHomogeneity(numPt=500, showPlot=False)
print(f"B0           = {magnet.B0:.4g}")
print(f"numPt        = {magnet.numPt}")

# ──────────────────────────────────────────────────────────────────────────── #
# Excitation field
# ──────────────────────────────────────────────────────────────────────────── #

excField = MagField(name="N-pulse arbitrary-delay sequence")

simu = Simulation(
    name="Xe-129 14T N-pulse simulation",
    sample=sample,
    magnet=magnet,
    excField=excField,
    RCF_freq=RCF_freq,
    rate=simuRate,
    duration=duration,
    verbose=False,
)

print(f"Larmor freq  = {nu_L:g}")
print(f"T2*          = {simu.T2star:g}")
print(f"Tdelta       = {simu.Tdelta:g}")
print(f"numSteps     = {simu.numSteps}")
print(f"timeStep     = {simu.timeStep:g}")

simu.excField.setNPulsesArbDelay(
    timeStep=simu.timeStep,
    timeLen=simu.timeLen,
    gamma=simu.sample.gamma,
    pulseDur=pulseDur,
    tip_angles=tip_angles,
    delays=delays,
    nu_rot=signalFreqRot,
    phases=phases,
    verbose=False,
)

simu.estimateRuntime(verbose=True)
# check(simu.excField.B_vec.shape)

# ──────────────────────────────────────────────────────────────────────────── #
# Run simulation
# ──────────────────────────────────────────────────────────────────────────── #

tic = time.perf_counter()
simu.generateTrajectories()
toc = time.perf_counter()
print(f"generateTrajectories time = {toc - tic:.3f} s")

simu.keepMeanStd()
simu.displayTrjry(verbose=True)

# ──────────────────────────────────────────────────────────────────────────── #
# Optionally save results
# ──────────────────────────────────────────────────────────────────────────── #

save_data = False

if save_data:
    simu.saveToPkl(
        fileDir="examples/", fileName="example_NpulsesArbDelay.pkl", verbose=True
    )

# ──────────────────────────────────────────────────────────────────────────── #
# Extract per-pulse signal windows
# ──────────────────────────────────────────────────────────────────────────── #

# Reconstruct pulse start/end sample indices (mirrors setNPulsesArbDelay logic)
_pulseLen = int(np.round(pulseDur / simu.timeStep))  # = 5 steps
_pulse_starts = []
_pulse_ends = []
_idx = 0
_acq_delay = 1 * unit.ms
for i in range(Npulses):
    _idx += int(np.round(delays[i] / simu.timeStep))
    _pulse_starts.append(_idx)
    _idx += _pulseLen
    _pulse_ends.append(_idx)

# Complex transverse magnetization time series
TS_complex = simu.M_mean[:, 0] + 1j * simu.M_mean[:, 1]
total_steps = len(TS_complex)

# Slice the free-precession window after each pulse
_windows = []
for i in range(Npulses):
    t0 = _pulse_ends[i] + int((_acq_delay * simu.rate).to_value(unit.one))
    t1 = _pulse_starts[i + 1] if i + 1 < Npulses else total_steps
    _windows.append(TS_complex[t0:t1])

check(np.std(np.asarray(_windows)))
check((np.asarray(_windows).shape))
# Truncate to the shortest window so the result is rectangular
window_len = min(len(w) for w in _windows)
signals_2d = np.array([w[:window_len] for w in _windows])
# signals_2d.shape == (Npulses, window_len)

timestamp_window = simu.getTimeStamp()[:window_len]  # time axis for one window

print(
    f"signals_2d shape: {signals_2d.shape}  "
    f"({Npulses} pulses × {window_len} samples/window)"
)

signal = Simu_Signal()
signal.TS_raw = simu.M_mean[:, 0] + 1j * simu.M_mean[:, 1]
signal.TS_raw *= unit.one
signal.TS = signals_2d * unit.one
check(signal.TS.shape)
signal.sampRate = simu.rate
signal.demodFreq = simu.RCF_freq
signal.getFS(
    winInfo={"name": "expDecay", "param": 20},
    verbose=True
)
# signal.plotTS()
signal.getStackedSpectra(freqs=signal.freqs, specStack=signal.PSD)
# signal.getStackedSpectra(
#     freqs=np.arange(signal.TS.shape[1]) * unit.one, specStack=signal.TS.imag
# )
halfwidth = np.abs(FWHM * simu.magnet.B0 * simu.sample.gamma / (2 * PI))
print(halfwidth.to(unit.kHz))
print(np.mean(signal.freqs))
integrals = signal.getFSIntegrals(
    freq_center=signal.demodFreq,
    halfwidth=halfwidth,
)
indices = np.arange(integrals.shape[0])


fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures

ax00 = fig.add_subplot(gs[0, 0])
ax00.plot(indices, integrals.real, label='real')
ax00.plot(indices, integrals.imag, label='imag')
ax00.set_xlabel("")
ax00.set_ylabel("")
# ax00.set_xscale('log')
# ax00.set_yscale('log')
# ax00.legend()
# #############################################################################
fig.suptitle("", wrap=True)

plt.tight_layout()
# plt.savefig('example figure - one-column.png', transparent=False)
plt.show()
