from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from axionbloch.Apparatus import Magnet
from axionbloch.constants import gamma_p
from axionbloch.dependency import PI, ppm, unit

# An ideal instantaneous 90-degree pulse leaves M+ = 1. The subsequent
# ensemble free induction is the characteristic function of the field spread.
RCF_FREQ = 1.0 * unit.MHz
FWHM = 1.0 * ppm
N_FWHM = 100.0
NUM_PTS = (100, 500, 2_000, 10_000)

fwhm_hz = (RCF_FREQ * FWHM).to_value(unit.Hz)
t_delta = 1.0 / (np.pi * fwhm_hz)
time = np.linspace(0.0, 3.0 * t_delta, 1000)
ideal = np.exp(-time / t_delta)

def sample_decay(num_pts):
    b0 = RCF_FREQ / (gamma_p / (2 * PI))
    magnet = Magnet(B0=b0, FWHM=FWHM, nFWHM=N_FWHM, numPt=num_pts)
    detuning = (gamma_p / (2 * PI)).to_value(unit.Hz / unit.T) * (
        magnet.B_spread.to_value(unit.T) - b0.to_value(unit.T)
    )
    # Exact free evolution per packet; this isolates field sampling from RK4
    # timestep and finite-pulse errors.
    signal = np.exp(2j * np.pi * detuning[:, None] * time[None, :])
    return np.dot(magnet.ratios, signal), magnet

results = {n: sample_decay(n) for n in NUM_PTS}
errors = {n: np.sqrt(np.mean(np.abs(results[n][0] - ideal) ** 2)) for n in NUM_PTS}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
ax = axes[0]
ax.plot(time / t_delta, ideal, "k-", lw=2.2, label=r"Ideal $e^{-t/T_\delta}$")
for n in NUM_PTS:
    signal, _ = results[n]
    ax.plot(time / t_delta, signal.real, lw=1.1, label=f"{n:,} packets")
ax.set(xlabel=r"Time / $T_\delta$", ylabel=r"$M_x/M_0$", title="Lorentzian free-induction decay")
ax.grid(alpha=.3)
ax.legend(fontsize=8)

ax = axes[1]
for n in NUM_PTS:
    signal, _ = results[n]
    ax.plot(time / t_delta, signal.real - ideal, lw=1.1, label=f"{n:,} packets")
ax.axhline(0, color="black", lw=.8)
ax.set(xlabel=r"Time / $T_\delta$", ylabel="Residual", title="Residual from ideal Lorentzian decay")
ax.grid(alpha=.3)
ax.legend(fontsize=8)
fig.suptitle(f"Midpoint Lorentzian sampling; FWHM={fwhm_hz:g} Hz, truncation=±{N_FWHM:g} FWHM")
fig.tight_layout()

out = Path(__file__).with_name("calibration-lorentzian-t2star.png")
fig.savefig(out, dpi=200)
fig.savefig(out.with_suffix(".pdf"))
print(f"T_delta = {t_delta:g} s; ideal T2* = T_delta for negligible intrinsic relaxation")
print("packets | RMS envelope error vs ideal")
for n in NUM_PTS:
    print(f"{n:7d} | {errors[n]:.6e}")
print(f"Saved {out}")
print(f"Saved {out.with_suffix('.pdf')}")
