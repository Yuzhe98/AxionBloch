# plot the daily modulation of the axion gradient at various stations on Earth, for an Earth-bound axion halo with a given Compton frequency and axion-nucleon coupling. The modulation is due to the rotation of the Earth, which changes the direction of the station relative to the axion halo. The code solves the time-independent Schrödinger equation for the axion halo and then computes the gradient at each station as a function of time over one day. Finally, it plots the results.
import os

from axionbloch.dependency import *
from axionbloch.EarthBoundAxionHalo import (
    EarthBoundAxionHalo,
)
from axionbloch.utils import check
from axionbloch.Station import (
    Mainz,
    Baltimore,
    Sanya,
    Tokyo,
    Geneva,
    Sydney,
    BuenosAires,
    CapeTown,
    Mumbai,
)

stations = [
    Mainz,
    Geneva,
    Baltimore,
    Tokyo,
    Mumbai,
    Sanya,
    Sydney,
    CapeTown,
    BuenosAires,
]
# stations = [
#     CapeTown,
#     Mumbai,
# ]
# station = Baltimore
rhoE_DM = 0.3 * unit.GeV / unit.cm**3

# states_to_check = ["1s", "2s", "2p", "3s", "3p", "3d", "4s", "4p", "4d"]
states_to_check = ["2p"]
measurement_times = []
halo = EarthBoundAxionHalo(
    nu_a=1.348 * unit.MHz,  # axion Compton frequency in Hz
    N=int(2**12),  # number of grid points
    extent=128.0 * unit.R_earth,  # spatial extent of the grid in units of earth radius
    g_aNN=1e-9 * unit.GeV**-1,  # axion-nucleon coupling
    verbose=True,
)

halo.showValueAndUnits()


halo.solve_TISE_3D(
    l_vals=[1],  # angular momentum quantum number
    max_n_r=64,  # maximum principal quantum number to plot
    verbose=False,
)

# ── gradient at Mainz over 2022-12-14 ────────────────────────────────────────
# TODO make this a function in GravHalo class.
# from astropy.time import Time
# import matplotlib.pyplot as plt
# from matplotlib import gridspec

from zoneinfo import ZoneInfo
from datetime import datetime
t0 = Time(datetime(2022, 12, 13, 7, 0, tzinfo=ZoneInfo("Europe/Berlin")))
t_hours = np.linspace(0, 72, 72) * unit.hour
meas_times = t0 + t_hours  # every 1 hour

result = halo.findGradientsOverTime(
    stateNames=states_to_check,
    station=Mainz,
    meas_times=meas_times,
    truncRadius=5 * unit.R_earth,
    verbose=True,
)

fig = plt.figure(figsize=(8.5 / 2.54, 8.5 / 2.54), dpi=300)
gs = gridspec.GridSpec(3, 1)
fig.subplots_adjust(left=0.22, bottom=0.12, right=0.67, top=0.93, hspace=0.12)

ax_r = fig.add_subplot(gs[0])
ax_theta = fig.add_subplot(gs[1], sharex=ax_r, sharey=ax_r)
ax_phi = fig.add_subplot(gs[2], sharex=ax_r, sharey=ax_r)
# check(result["grad_r"][0:10])
ax_r.plot(t_hours, result["grad_r"], label="Re", color="tab:blue")
# ax_r.plot(t_hours, result["grad_r"].imag, label="Im", color="tab:orange", ls="--")
ax_theta.plot(t_hours, result["grad_theta"], color="tab:green")
# ax_theta.plot(t_hours, result["grad_theta"].imag, color="tab:orange", ls="--")
ax_phi.plot(t_hours, result["grad_phi"], color="tab:brown")
# ax_phi.plot(t_hours, result["grad_phi"].imag, color="tab:orange", ls="--")

ax_phi.set_xlabel("time (hour)")


def _ylabel(label, q):
    u_str = q.unit.to_string("latex_inline")[1:-1]
    return f"${label}$\n$\\left({u_str}\\right)$"


ax_r.set_ylabel(
    _ylabel("\\partial_r \\phi", result["grad_r"]),
    rotation=0,
    loc="center",
    labelpad=22,
)
ax_theta.set_ylabel(
    _ylabel("\\frac{1}{r}\\partial_\\theta \\phi", result["grad_theta"]),
    rotation=0,
    loc="center",
    labelpad=22,
)
ax_phi.set_ylabel(
    _ylabel("\\frac{1}{r\\!\\sin\\theta}\\partial_\\varphi\\phi", result["grad_phi"]),
    rotation=0,
    loc="center",
    labelpad=22,
)

for ax in [ax_r, ax_theta]:
    plt.setp(ax.get_xticklabels(), visible=False)

# ax_r.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7)
fig.suptitle(
    f"Mainz  ·  {states_to_check[0]}  ·  $\\nu_a={halo.nu_a.to_value(unit.MHz):.3f}$ MHz\nstart from {t0}"
)
ylim_abs = 0
for ax in [ax_r, ax_phi, ax_theta]:
    _ylim_upper, _ylim_bottom = ax.get_ylim()
    ylim_abs = np.amax(np.abs([ylim_abs, (_ylim_upper), (_ylim_bottom)]))

ax_r.set_ylim(-1 * ylim_abs, ylim_abs)
plt.tight_layout()
plt.show()
