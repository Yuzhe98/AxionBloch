"""Examples for MilkyWay and MilkyWayAxionHalo."""

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy import units as unit

from axionbloch.MilkyWay import MilkyWay
from axionbloch.MilkyWayAxionHalo import MilkyWayAxionHalo
from axionbloch.Station import Mainz, Sanya

station = Mainz
# station = Sanya
# ── Example 1: basic galactic kinematics ─────────────────────────────────────

t = Time("2024-06-01T12:00:00", scale="utc")
mw = MilkyWay(time=t, station=station)

print("=== MilkyWay: basic quantities ===")
print("v_lab magnitude :", mw.get_v_lab_magnitude().to(unit.km / unit.s))
print("v_lab vector    :", mw.get_v_lab().to(unit.km / unit.s))
print("v_sun           :", mw.get_sun_velocity().to(unit.km / unit.s))
print("v_earth_helio   :", mw.get_earth_heliocentric_velocity().to(unit.km / unit.s))
print("Earth position  :", mw.get_earth_position().to(unit.kpc))
print(
    "Earth's distance from the galactic centre     :",
    mw.get_galactocentric_radius().to(unit.kpc),
)
print()

# ── Example 2: station-specific quantities ───────────────────────────────────

print(f"=== MilkyWay: station quantities ({station.name}) ===")
print("nvec galactic   :", mw.get_nvec_galactic())
print("nvec GCRS       :", mw.get_nvec_gcrs())
print("Station (l, b)  :", mw.get_station_galactic_lb().to(unit.deg))
print("Wind angle      :", mw.get_wind_angle().to(unit.deg))
print("v_rotation      :", mw.get_earth_rotation_velocity())
print()
mw.summary()

# ── Example 3: daily modulation of the wind angle ────────────────────────────

print("\n=== MilkyWay: daily modulation over 24 h ===")
t_day = Time("2024-06-01", scale="utc") + np.linspace(0, 24, 144) * unit.hour
mw_day = MilkyWay(time=t_day, station=station)
angles_deg = np.array([
    MilkyWay(time=ti, station=station).get_wind_angle().to(unit.deg).value
    for ti in t_day
])
print(f"Wind angle range: {angles_deg.min():.1f}° – {angles_deg.max():.1f}°")

# ── Example 4: 3-D visualization ─────────────────────────────────────────────

fig = mw.plot(show=True)
# fig.savefig("milkyway_kinematics.png", dpi=120, bbox_inches="tight")
# plt.close(fig)
# print("\nSaved milkyway_kinematics.png")


# ── Example 9: combine MilkyWay + MilkyWayAxionHalo ──────────────────────────

print("\n=== Combined: time-varying wind angle ===")
times = Time("2024-06-01", scale="utc") + np.arange(0, 24, 6) * unit.hour
for ti in times:
    mw_i = MilkyWay(time=ti, station=station)
    angle_i = mw_i.get_wind_angle()
    axion_i = MilkyWayAxionHalo(
        nu_a=1.0 * unit.kHz,
        g_aNN=1e-9 / unit.GeV,
        windAngle=angle_i,
    )
    rabi_i = axion_i.getRabiFreq(case="grad_perp")
    print(
        f"  {ti.iso[:16]}  angle={angle_i.to(unit.deg).value:.1f}°  "
        f"Rabi={rabi_i.to(unit.Hz):.4e}"
    )
