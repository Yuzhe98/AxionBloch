"""Galactic kinematics: Earth and lab orientation relative to the Milky Way.

All computations use :mod:`astropy.coordinates` frame transformations, which
account for Earth's rotation, precession, and nutation automatically.

Key frames used
---------------
Galactocentric
    Inertial frame centred on the galactic centre (GC).  The x-axis points
    from GC toward the Sun; z points toward the galactic north pole; y is
    determined by the right-hand rule.
    This is the rest frame of the dark-matter halo.

ICRS
    Solar-system barycentric frame (no rotation relative to distant quasars).

ITRS
    Earth-fixed frame (rotates with Earth).

The transformation chain for a station's position is:
``ITRS → GCRS → ICRS → Galactocentric``
and is handled entirely by :mod:`astropy.coordinates` at the requested epoch.
"""

import math

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import art3d
import numpy as np
from astropy import units as unit
from astropy.coordinates import (
    EarthLocation,
    Galactocentric,
    SkyCoord,
    get_body_barycentric_posvel,
)
import astropy.coordinates as coord
from astropy.time import Time
from astropy.units import Quantity

from axionbloch.Station import Station


def _draw_3d_arrow(ax, start, end, color="tomato", shaft_lw=2.0,
                   head_length=0.30, head_radius=0.10, n_faces=24, zorder=10):
    """Draw a rotation-correct 3D arrow as a line shaft + Poly3DCollection cone."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    d = end - start
    d_norm = np.linalg.norm(d)
    if d_norm == 0:
        return
    d_hat = d / d_norm

    shaft_tip = end - d_hat * head_length
    ax.plot(
        [start[0], shaft_tip[0]], [start[1], shaft_tip[1]], [start[2], shaft_tip[2]],
        color=color, lw=shaft_lw, zorder=zorder, solid_capstyle="round",
    )

    # two orthonormal vectors perpendicular to d_hat
    ref = np.array([1., 0., 0.]) if abs(d_hat[0]) < 0.9 else np.array([0., 1., 0.])
    e1 = np.cross(d_hat, ref); e1 /= np.linalg.norm(e1)
    e2 = np.cross(d_hat, e1)

    theta = np.linspace(0, 2 * np.pi, n_faces + 1)
    base = np.array([shaft_tip + head_radius * (np.cos(t) * e1 + np.sin(t) * e2)
                     for t in theta])
    tris = [[base[i], base[i + 1], end] for i in range(n_faces)]
    cone = art3d.Poly3DCollection(tris, zorder=zorder)
    cone.set_facecolor(color)
    cone.set_edgecolor(color)
    ax.add_collection3d(cone)


class MilkyWay:
    """Galactic kinematics and lab orientation relative to the Milky Way.

    Given an observation epoch (:class:`astropy.time.Time`) this class
    provides the quantities needed for axion dark-matter NMR experiments:

    - Earth's position and velocity in the
      :class:`~astropy.coordinates.Galactocentric` rest frame.
    - The lab velocity :math:`\\mathbf{v}_\\mathrm{lab}` (Sun's galactic
      rotation + peculiar velocity + Earth's instantaneous orbital velocity).
    - The station's sensitive-axis unit vector in galactic coordinates
      (changes daily as Earth rotates — the daily-modulation signal).
    - The angle between the axion wind and the sensitive axis.
    - Earth's surface rotation velocity at the station.

    Parameters
    ----------
    time : :class:`astropy.time.Time`
        Observation epoch.  Scalar or array.
    station : :class:`~axionbloch.Station.Station`, optional
        Experimental site.  Required for :meth:`get_nvec_galactic`,
        :meth:`get_wind_angle`, and :meth:`get_earth_rotation_velocity`.
    galcen_frame : :class:`astropy.coordinates.Galactocentric`, optional
        Custom galactocentric frame definition (e.g. different
        ``galcen_distance`` or ``galcen_v_sun``).  Defaults to the
        astropy built-in values (Gravity Collaboration 2018 / Drimmel &
        Poggio 2018):

        * ``galcen_distance`` = 8.122 kpc
        * ``galcen_v_sun`` = (12.9, 245.6, 7.78) km/s
          (peculiar + circular rotation in galactocentric Cartesian)

    Examples
    --------
    >>> from astropy.time import Time
    >>> from axionbloch.Station import Mainz
    >>> from axionbloch.MilkyWay import MilkyWay
    >>> mw = MilkyWay(time=Time('2024-06-01'), station=Mainz)
    >>> mw.get_v_lab_magnitude()
    ...  # ≈ 240–270 km/s depending on Earth's orbital phase
    >>> mw.get_wind_angle().to('deg')
    ...  # angle between v_lab and local vertical at Mainz
    >>> mw.summary()
    """

    def __init__(
        self,
        time: Time,
        station: Station | None = None,
        galcen_frame: coord.Galactocentric | None = None,
    ):
        self.time = time
        self.station = station
        self.galcen_frame = (
            galcen_frame if galcen_frame is not None else coord.Galactocentric()
        )
        self._earth_galcen: SkyCoord | None = None

    # ------------------------------------------------------------------
    # Earth in the galactocentric frame
    # ------------------------------------------------------------------

    @property
    def earth_galactocentric(self) -> SkyCoord:
        """Earth's position and velocity in the Galactocentric frame (cached).

        Obtained by fetching Earth's barycentric state in ICRS with
        :func:`~astropy.coordinates.get_body_barycentric_posvel` and
        transforming to :attr:`galcen_frame`.  The result is cached for the
        lifetime of the object.
        """
        if self._earth_galcen is None:
            pos, vel = get_body_barycentric_posvel("earth", self.time)
            coord = SkyCoord(
                x=pos.x,
                y=pos.y,
                z=pos.z,
                v_x=vel.x.to(unit.km / unit.s),
                v_y=vel.y.to(unit.km / unit.s),
                v_z=vel.z.to(unit.km / unit.s),
                frame="icrs",
                representation_type="cartesian",
                differential_type="cartesian",
            )
            self._earth_galcen = coord.transform_to(self.galcen_frame)
        return self._earth_galcen

    def get_earth_position(self) -> Quantity:
        """Earth's Cartesian position in the Galactocentric frame.

        Returns
        -------
        Quantity, shape (3,) [kpc]
            :math:`(x, y, z)` in galactocentric Cartesian coordinates.
            The Sun sits at approximately :math:`(-8.1, 0, 0)` kpc.
        """
        gc = self.earth_galactocentric
        return (
            np.array(
                [
                    gc.x.to(unit.kpc).value,
                    gc.y.to(unit.kpc).value,
                    gc.z.to(unit.kpc).value,
                ]
            )
            * unit.kpc
        )

    def get_galactocentric_radius(self) -> Quantity:
        """Earth's distance from the galactic centre (kpc)."""
        return np.sqrt(np.sum(self.get_earth_position() ** 2)).to(unit.kpc)

    # ------------------------------------------------------------------
    # Lab velocity in the galactic rest frame
    # ------------------------------------------------------------------

    def get_earth_velocity(self) -> Quantity:
        """Earth's velocity vector in the galactic rest frame (km/s).

        Combines three contributions:

        1. **Sun's galactic rotation** (~220 km/s in the :math:`+y` direction).
        2. **Sun's peculiar velocity** (~13 km/s, encoded in
           ``galcen_frame.galcen_v_sun``).
        3. **Earth's orbital velocity** around the Sun (~30 km/s, varies
           annually, obtained from the ephemeris).

        Returns
        -------
        Quantity, shape (3,) [km/s]
            :math:`(v_x, v_y, v_z)` in galactocentric Cartesian coordinates.
        """
        gc = self.earth_galactocentric
        return (
            np.array(
                [
                    gc.v_x.to(unit.km / unit.s).value,
                    gc.v_y.to(unit.km / unit.s).value,
                    gc.v_z.to(unit.km / unit.s).value,
                ]
            )
            * unit.km
            / unit.s
        )

    def get_v_lab(self) -> Quantity:
        """Total lab velocity vector in the galactic rest frame (km/s).

        Synonym for :meth:`get_earth_velocity`.  Surface rotation at the
        station (~0.3–0.5 km/s) is not included here; obtain it from
        :meth:`get_earth_rotation_velocity` if needed.

        Returns
        -------
        Quantity, shape (3,) [km/s]
        """
        return self.get_earth_velocity()

    def get_v_lab_magnitude(self) -> Quantity:
        """Magnitude :math:`|\\mathbf{v}_\\mathrm{lab}|` (km/s).

        Typically in the range 230–270 km/s depending on Earth's orbital
        phase.
        """
        v = self.get_v_lab()
        return np.sqrt(np.sum(v**2)).to(unit.km / unit.s)

    def get_sun_velocity(self) -> Quantity:
        """Sun's velocity in the galactic rest frame (km/s).

        Returns the ``galcen_v_sun`` vector from the galactocentric frame
        definition, which encodes both circular rotation and peculiar motion.
        This is the time-independent part of :math:`\\mathbf{v}_\\mathrm{lab}`.

        Returns
        -------
        Quantity, shape (3,) [km/s]
        """
        gsun = self.galcen_frame.galcen_v_sun
        return (
            np.array(
                [
                    gsun.d_x.to(unit.km / unit.s).value,
                    gsun.d_y.to(unit.km / unit.s).value,
                    gsun.d_z.to(unit.km / unit.s).value,
                ]
            )
            * unit.km
            / unit.s
        )

    def get_earth_heliocentric_velocity(self) -> Quantity:
        """Earth's velocity relative to the Sun in ICRS (km/s).

        This is the time-varying (annual) contribution to
        :math:`\\mathbf{v}_\\mathrm{lab}`, with magnitude ~30 km/s.

        Returns
        -------
        Quantity, shape (3,) [km/s]
        """
        _, vel = get_body_barycentric_posvel("earth", self.time)
        return (
            np.array(
                [
                    vel.x.to(unit.km / unit.s).value,
                    vel.y.to(unit.km / unit.s).value,
                    vel.z.to(unit.km / unit.s).value,
                ]
            )
            * unit.km
            / unit.s
        )

    # ------------------------------------------------------------------
    # Earth surface rotation at the station
    # ------------------------------------------------------------------

    def get_earth_rotation_velocity(self) -> Quantity:
        """Surface velocity at the station due to Earth's sidereal rotation.

        Computes :math:`\\boldsymbol{\\omega} \\times \\mathbf{r}` in the
        ITRS frame, where
        :math:`|\\boldsymbol{\\omega}| = 2\\pi / T_\\mathrm{sidereal}` and
        :math:`\\mathbf{r}` is the station's geocentric Cartesian position.

        The sidereal period used is :math:`T_\\mathrm{sid} = 86164.091` s.
        At a latitude of 50° this gives ~0.30 km/s; at the equator ~0.46 km/s.

        Returns
        -------
        Quantity, shape (3,) [km/s]
            Velocity vector in ITRS (Earth-fixed) Cartesian coordinates.

        Raises
        ------
        ValueError
            If :attr:`station` is ``None``.
        """
        if self.station is None:
            raise ValueError("station must be set to compute Earth rotation velocity")

        loc = self._make_earth_location()
        itrs = loc.get_itrs(obstime=self.time)
        r_m = np.array(
            [
                itrs.cartesian.x.to(unit.m).value,
                itrs.cartesian.y.to(unit.m).value,
                itrs.cartesian.z.to(unit.m).value,
            ]
        )

        T_sid = 86164.091  # sidereal day in seconds
        omega = 2 * math.pi / T_sid  # rad/s

        v_rot_m_s = np.cross([0.0, 0.0, omega], r_m)
        return (v_rot_m_s / 1e3) * unit.km / unit.s

    # ------------------------------------------------------------------
    # Station orientation in galactic frame
    # ------------------------------------------------------------------

    def get_nvec_galactic(self) -> np.ndarray:
        """Station's outward unit normal in galactocentric Cartesian coordinates.

        Transforms the station's ITRS geocentric position to the
        Galactocentric frame at :attr:`time`.  The transformation chain
        ``ITRS → GCRS → ICRS → Galactocentric`` handles Earth's rotation,
        precession, and nutation automatically, so the returned direction
        changes continuously as the Earth rotates (daily modulation).

        Returns
        -------
        ndarray, shape (3,)
            Dimensionless unit vector :math:`\\hat{n}` in galactocentric
            Cartesian coordinates.

        Raises
        ------
        ValueError
            If :attr:`station` is ``None``.
        """
        if self.station is None:
            raise ValueError("station must be set to compute nvec")

        loc = self._make_earth_location()
        itrs = loc.get_itrs(obstime=self.time)
        galcen = SkyCoord(itrs).transform_to(self.galcen_frame)

        r = np.array(
            [
                galcen.x.to(unit.m).value,
                galcen.y.to(unit.m).value,
                galcen.z.to(unit.m).value,
            ]
        )
        return r / np.linalg.norm(r)

    def get_station_galactic_lb(self) -> Quantity:
        """Station's direction in galactic longitude / latitude (l, b).

        Returns
        -------
        Quantity, shape (2,) [deg]
            ``[l, b]`` in degrees.

        Raises
        ------
        ValueError
            If :attr:`station` is ``None``.
        """
        if self.station is None:
            raise ValueError("station must be set")

        loc = self._make_earth_location()
        itrs = loc.get_itrs(obstime=self.time)
        gal = SkyCoord(itrs).galactic
        return np.array([gal.l.deg, gal.b.deg]) * unit.deg

    def get_nvec_gcrs(self) -> np.ndarray:
        """Station's outward unit normal in GCRS (geocentric inertial) coordinates.

        Unlike :meth:`get_nvec_galactic`, which returns a direction in the
        galactocentric frame (essentially Earth's direction from the galactic
        centre, the same for all stations since Earth is tiny at 8 kpc),
        this method returns the station's local vertical in the **geocentric
        equatorial** frame.  This vector:

        - differs between stations (latitude / longitude),
        - rotates with the Earth as a function of time (daily modulation),
        - is the correct axis for wind-angle calculations.

        Returns
        -------
        ndarray, shape (3,)
            Unit vector in GCRS Cartesian (x = vernal equinox, z = celestial
            north pole).

        Raises
        ------
        ValueError
            If :attr:`station` is ``None``.
        """
        if self.station is None:
            raise ValueError("station must be set to compute nvec_gcrs")
        from astropy.coordinates import GCRS

        loc = self._make_earth_location()
        gcrs = loc.get_itrs(obstime=self.time).transform_to(GCRS(obstime=self.time))
        r = np.array(
            [
                gcrs.cartesian.x.to(unit.m).value,
                gcrs.cartesian.y.to(unit.m).value,
                gcrs.cartesian.z.to(unit.m).value,
            ]
        )
        return r / np.linalg.norm(r)

    # ------------------------------------------------------------------
    # Axion wind angle
    # ------------------------------------------------------------------

    def get_wind_angle(self) -> Quantity:
        """Angle between the lab velocity and the station's sensitive axis.

        Computes the angle between:

        * **v_lab** expressed in ICRS orientation (galactocentric velocity
          rotated to equatorial Cartesian via the galactic-to-ICRS rotation),
        * the station's local vertical **nvec** in GCRS (equatorial inertial),
          which changes as the Earth rotates.

        This correctly captures both the station-dependent and the
        time-dependent (daily modulation) contributions.  In a typical NMR
        axion search the static bias field :math:`\\mathbf{B}_0` is aligned
        with the local vertical, so this is the angle :math:`\\alpha` entering
        the ``grad_perp`` / ``grad_par`` lineshape.

        Returns
        -------
        Quantity [rad]
            Angle :math:`\\alpha \\in [0, \\pi]`.

        Raises
        ------
        ValueError
            If :attr:`station` is ``None``.
        """
        if self.station is None:
            raise ValueError("station must be set to compute wind angle")
        v_hat_icrs = self._v_lab_direction_icrs()
        n = self.get_nvec_gcrs()
        cos_alpha = float(np.clip(np.dot(v_hat_icrs, n), -1.0, 1.0))
        return math.acos(cos_alpha) * unit.rad

    def _v_lab_direction_icrs(self) -> np.ndarray:
        """v_lab unit vector expressed in ICRS / GCRS orientation.

        Transforms the galactocentric velocity vector to the ICRS equatorial
        Cartesian frame by applying the galactic-to-ICRS rotation matrix,
        which is built from the known ICRS directions of the galactic
        coordinate axes:

        * galactocentric +x = GC → Sun = galactic (l=180°, b=0°),
        * galactocentric +y = galactic rotation direction = (l=90°, b=0°),
        * galactocentric +z = galactic north pole = (b=90°).
        """
        from astropy.coordinates import SkyCoord as _SC

        def _icrs_hat(l_deg, b_deg):
            sc = _SC(l=l_deg * unit.deg, b=b_deg * unit.deg, frame="galactic").icrs
            return np.array(
                [sc.cartesian.x.value, sc.cartesian.y.value, sc.cartesian.z.value]
            )

        # Rotation matrix: each column is a galactocentric basis vector in ICRS
        R = np.column_stack([_icrs_hat(180, 0), _icrs_hat(90, 0), _icrs_hat(0, 90)])

        v_galcen = self.get_v_lab().to(unit.km / unit.s).value
        v_icrs = R @ v_galcen
        return v_icrs / np.linalg.norm(v_icrs)

    # ------------------------------------------------------------------
    # Sun's galactocentric state
    # ------------------------------------------------------------------

    def get_sun_galactocentric(self) -> tuple[Quantity, Quantity]:
        """Sun's position and velocity in the Galactocentric frame.

        The Sun's position in the Galactocentric frame is at
        ``(−galcen_distance, 0, 0)`` by construction; the velocity is
        read from ``galcen_frame.galcen_v_sun`` (peculiar + circular rotation).

        Returns
        -------
        position : Quantity, shape (3,) [kpc]
        velocity : Quantity, shape (3,) [km/s]
        """
        # The Sun is at the ICRS origin — transform zero to galactocentric
        sun_icrs = SkyCoord(
            x=0 * unit.au,
            y=0 * unit.au,
            z=0 * unit.au,
            v_x=0 * unit.km / unit.s,
            v_y=0 * unit.km / unit.s,
            v_z=0 * unit.km / unit.s,
            frame="icrs",
            representation_type="cartesian",
            differential_type="cartesian",
        )
        sun_gc = sun_icrs.transform_to(self.galcen_frame)
        pos = (
            np.array(
                [
                    sun_gc.x.to(unit.kpc).value,
                    sun_gc.y.to(unit.kpc).value,
                    sun_gc.z.to(unit.kpc).value,
                ]
            )
            * unit.kpc
        )
        vel = self.get_sun_velocity()
        return pos, vel

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self):
        """Print a human-readable summary of galactic kinematic quantities."""
        print(f"Epoch          : {self.time.iso}")
        print(
            f"Galcen frame   : distance={self.galcen_frame.galcen_distance}, "
            f"v_sun={self.galcen_frame.galcen_v_sun}"
        )
        print()

        pos = self.get_earth_position()
        vel = self.get_earth_velocity()
        v_mag = self.get_v_lab_magnitude()
        r_gc = self.get_galactocentric_radius()
        print(f"Earth position (galactocentric) : {pos}")
        print(f"Earth velocity (galactocentric) : {vel}")
        print(f"|v_lab|                         : {v_mag:.3f}")
        print(f"Earth–GC distance               : {r_gc:.4f}")

        sun_pos, sun_vel = self.get_sun_galactocentric()
        print(f"Sun   position (galactocentric) : {sun_pos}")
        print(f"Sun   velocity (galactocentric) : {sun_vel}")
        print()

        if self.station is not None:
            n = self.get_nvec_galactic()
            alpha = self.get_wind_angle()
            lb = self.get_station_galactic_lb()
            v_rot = self.get_earth_rotation_velocity()
            print(f"Station        : {self.station.name}")
            print(f"nvec (galcen)  : {n}")
            print(f"Station (l, b) : {lb}")
            print(f"Wind angle     : {alpha.to(unit.deg):.2f}")
            print(f"|v_rotation|   : {np.linalg.norm(v_rot.value):.4f} {v_rot.unit}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        show: bool = True,
        figsize: tuple = (16, 7),
    ) -> Figure:
        """3-D illustration of the Sun and Earth in the Milky Way.

        Produces a dark-background figure with two 3-D panels:

        * **Left** — Bird's-eye view of the Milky Way with spiral arms, the
          galactic centre, the Sun's circular orbit, the Sun's current
          position, and the lab velocity vector.
        * **Right** — Solar system (not to scale) showing Earth's orbit,
          Earth's current position, the Earth's rotation axis (two arrows
          for N / S poles), and the Earth's equatorial plane.  If
          :attr:`station` is set, the station's sensitive axis is also drawn.

        Parameters
        ----------
        show : bool
            Call ``plt.show()`` at the end (default ``True``).
        figsize : tuple of float
            Figure size in inches.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=figsize, facecolor="#0d0d0d")
        ax_mw = fig.add_subplot(121, projection="3d")
        ax_ss = fig.add_subplot(122, projection="3d")

        self._plot_galaxy(ax_mw)
        self._plot_solar_system(ax_ss)

        title = f"Galactic Kinematics  ·  {self.time.iso[:10]}"
        # if self.station is not None:
        #     title += f"  ·  {self.station.name}"
        fig.suptitle(title, color="white", fontsize=11, fontweight="bold", y=0.98)

        plt.tight_layout(rect=[0.01, 0.01, 0.96, 0.96])
        fig.subplots_adjust(wspace=0.15)
        # plt.tight_layout(rect=[0, 0, 1, 0.96])

        if show:
            plt.show()
        return fig

    def _plot_galaxy(self, ax) -> None:
        """Populate the Milky Way overview panel (internal)."""
        rng = np.random.default_rng(42)

        # ---- background stars ----
        n_bg = 4000
        r_bg = rng.exponential(scale=4.0, size=n_bg)
        phi_bg = rng.uniform(0, 2 * np.pi, size=n_bg)
        z_bg = rng.normal(0, 0.12, size=n_bg)
        ax.scatter(
            r_bg * np.cos(phi_bg),
            r_bg * np.sin(phi_bg),
            z_bg,
            c="white",
            s=0.4,
            alpha=0.25,
            linewidths=0,
            zorder=1,
        )

        # ---- four logarithmic spiral arms: r(θ) = r0 · exp(b·θ) ----
        b, r0 = 0.22, 2.6  # pitch angle, starting radius (kpc)
        theta_arm = np.linspace(0, 3.8 * np.pi, 400)
        # colors = ["#5eaeff", "#ffda73", "#ff8060", "#80ffaa"]
        # names = ["Norma", "Sct–Cen", "Sagittarius", "Perseus"]
        # for i, (color, name) in enumerate(zip(colors, names)):
        for i in range(4):
            offset = i * np.pi / 2
            r_a = r0 * np.exp(b * theta_arm)
            mask = r_a < 16.0
            r_p, t_p = r_a[mask], theta_arm[mask] + offset
            # scatter stars along each arm with transverse and vertical spread
            spread = rng.normal(0, 0.18, size=r_p.size)
            x_s = r_p * np.cos(t_p) - spread * np.sin(t_p)
            y_s = r_p * np.sin(t_p) + spread * np.cos(t_p)
            z_s = rng.normal(0, 0.06, size=r_p.size)
            sizes = rng.exponential(scale=3.5, size=r_p.size)
            ax.scatter(
                x_s,
                y_s,
                z_s,
                s=sizes,
                alpha=0.65,
                linewidths=0,
                zorder=3,
                c="deepskyblue",
                # c=color, label=name,
            )

        # ---- galactic centre ----
        gc_ellipse = Ellipse(
            xy=(0, 0),
            width=4.8,
            height=1.6,
            facecolor="#fffacd",
            alpha=0.85,
            zorder=8,
        )
        ax.add_patch(gc_ellipse)
        art3d.pathpatch_2d_to_3d(gc_ellipse, z=0, zdir="z")
        ax.plot(
            [], [], color="#fffacd", marker="o", ms=8, ls="none", alpha=0.8, label="Galaxy center"
        )

        # ---- Sun's circular orbit ----
        r_sun = self.get_galactocentric_radius().to(unit.kpc).value
        phi_c = np.linspace(0, 2 * np.pi, 300)
        ax.plot(
            r_sun * np.cos(phi_c),
            r_sun * np.sin(phi_c),
            np.zeros(300),
            color="orange",
            lw=0.8,
            ls="--",
            alpha=0.60,
            zorder=12,
        )

        # ---- Sun ----
        sun_pos, _ = self.get_sun_galactocentric()
        sp = sun_pos.to(unit.kpc).value
        ax.scatter(
            [sp[0]],
            [sp[1]],
            [sp[2]],
            c="yellow",
            s=100,
            zorder=12,
            edgecolors="orange",
            linewidths=0.8,
            label="Sun",
        )

        # # ---- v_lab arrow ----
        # v = self.get_v_lab()
        # v_hat = (v / np.sqrt(np.sum(v**2))).to(unit.one).value
        # alen = 1.8  # kpc (schematic)
        # tip = sp + v_hat * alen
        # _draw_3d_arrow(ax, sp, tip, color="tomato", shaft_lw=2.0,
        #                head_length=0.30, head_radius=0.10, zorder=10)
        # ax.plot(
        #     [],
        #     [],
        #     color="tomato",
        #     lw=1.8,
        #     label=f"v_lab  ({self.get_v_lab_magnitude().value:.0f} km/s)",
        # )

        ax.text2D(
            0.03,
            0.03,
            "★ Not to scale",
            transform=ax.transAxes,
            color="gray",
            fontsize=7,
        )

        # ---- style ----
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_zlim(-2.5, 2.5)
        ax.set_xlabel("x (kpc)", color="white", fontsize=8, labelpad=1)
        ax.set_ylabel("y (kpc)", color="white", fontsize=8, labelpad=1)
        ax.set_zlabel("z (kpc)", color="white", fontsize=8, labelpad=1)
        ax.set_title("Milky Way", color="white", fontsize=9, pad=6)
        ax.set_facecolor("#0d0d0d")
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#404040")
        ax.tick_params(colors="gray", labelsize=7)
        ax.view_init(elev=28, azim=200)
        ax.legend(
            loc="upper right",
            fontsize=7,
            labelcolor="white",
            facecolor="#1a1a1a",
            edgecolor="#404040",
            framealpha=0.7,
            markerscale=0.5,
            scatterpoints=1,
        )

    def _plot_solar_system(self, ax) -> None:
        """Populate the Earth/solar-system panel (internal)."""
        # ---- Earth's current orbital position (ecliptic approx. from ICRS) ----
        pos, _ = get_body_barycentric_posvel("earth", self.time)
        lam_e = float(np.arctan2(pos.y.to(unit.AU).value, pos.x.to(unit.AU).value))
        r_orb = 1.0  # AU
        ex, ey, ez = r_orb * np.cos(lam_e), r_orb * np.sin(lam_e), 0.0

        # ---- faint ecliptic disk ----
        phi_g = np.linspace(0, 2 * np.pi, 60)
        r_g = np.linspace(0, 1.45, 8)
        RG, PG = np.meshgrid(r_g, phi_g)
        ax.plot_surface(
            RG * np.cos(PG),
            RG * np.sin(PG),
            np.zeros_like(RG),
            alpha=0.1,
            color="cyan",
            linewidth=0,
            antialiased=False,
        )

        # ---- Earth's orbit ----
        phi_o = np.linspace(0, 2 * np.pi, 300)
        ax.plot(
            r_orb * np.cos(phi_o),
            r_orb * np.sin(phi_o),
            np.zeros(300),
            color="deepskyblue",
            lw=1.2,
            ls="--",
            alpha=0.55,
            label="Earth orbit",
        )

        # ---- Sun–Earth direction line ----
        ax.plot([0, ex], [0, ey], [0, ez], color="cyan", lw=1.2, ls=":", alpha=1)

        # ---- orbital velocity arrow (perpendicular to radius) ----
        v_orb = np.array([-np.sin(lam_e), np.cos(lam_e), 0.0])
        v_orb_end = np.array([ex, ey, ez]) + v_orb * 0.22
        _draw_3d_arrow(ax, [ex, ey, ez], v_orb_end,
                       color="deepskyblue", shaft_lw=1.5,
                       head_length=0.06, head_radius=0.025, zorder=9)
        ax.plot([], [], color="deepskyblue", lw=1.5, label="Orbital velocity")

        # ---- Sun ----
        ax.scatter(
            [0],
            [0],
            [0],
            c="yellow",
            s=400,
            zorder=10,
            edgecolors="orange",
            linewidths=0.8,
            label="Sun",
        )

        # ---- Earth ----
        ax.scatter(
            [ex],
            [ey],
            [ez],
            c="#3399ff",
            s=130,
            zorder=10,
            edgecolors="cyan",
            linewidths=0.8,
            label="Earth",
        )

        # ---- rotation axis (N and S pole arrows) ----
        eps = math.radians(23.44)
        # N pole direction in ecliptic Cartesian: (0, +sin ε, cos ε)
        # (celestial N pole is at ecliptic longitude 90°, latitude 90°-ε)
        ax_n = np.array([0.0, math.sin(eps), math.cos(eps)])
        alen = 0.42  # AU (schematic)
        earth = np.array([ex, ey, ez])
        _draw_3d_arrow(ax, earth, earth + ax_n * alen,
                       color="tomato", shaft_lw=2.2,
                       head_length=0.07, head_radius=0.028, zorder=11)
        _draw_3d_arrow(ax, earth, earth - ax_n * alen,
                       color="tomato", shaft_lw=2.2,
                       head_length=0.07, head_radius=0.028, zorder=11)
        ax.plot([], [], color="tomato", lw=2.2,
                label=f"Rotation axis  (ε = {math.degrees(eps):.1f}°)")
        # N / S labels
        ax.text(
            ex + ax_n[0] * (alen + 0.06),
            ey + ax_n[1] * (alen + 0.06),
            ez + ax_n[2] * (alen + 0.06),
            "N",
            color="tomato",
            fontsize=8,
        )
        ax.text(
            ex - ax_n[0] * (alen + 0.1),
            ey - ax_n[1] * (alen + 0.1),
            ez - ax_n[2] * (alen + 0.1),
            "S",
            color="tomato",
            fontsize=8,
        )

        # ---- Earth's equatorial circle ----
        e1 = np.array([1.0, 0.0, 0.0])
        e1 -= e1.dot(ax_n) * ax_n
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ax_n, e1)
        phi_eq = np.linspace(0, 2 * np.pi, 80)
        r_eq = 0.2  # AU (schematic)
        ax.plot(
            ex + r_eq * (np.cos(phi_eq) * e1[0] + np.sin(phi_eq) * e2[0]),
            ey + r_eq * (np.cos(phi_eq) * e1[1] + np.sin(phi_eq) * e2[1]),
            ez + r_eq * (np.cos(phi_eq) * e1[2] + np.sin(phi_eq) * e2[2]),
            color="lightgreen",
            lw=1.0,
            alpha=0.70,
            label="Equator",
        )

        # ---- station sensitive axis ----
        if self.station is not None:
            n_gcrs = self.get_nvec_gcrs()
            # rotate GCRS equatorial → ecliptic (tilt z-axis by ε around x-axis)
            rot = np.array(
                [
                    [1, 0, 0],
                    [0, math.cos(eps), math.sin(eps)],
                    [0, -math.sin(eps), math.cos(eps)],
                ]
            )
            n_ecl = rot @ n_gcrs
            sa = 0.30
            _draw_3d_arrow(ax, earth, earth + n_ecl * sa,
                           color="w", shaft_lw=1.8,
                           head_length=0.07, head_radius=0.028, zorder=12)
            ax.plot([], [], color="w", lw=1.8,
                    label=f"{self.station.name}")

        # ---- date label ----
        # month = self.time.to_datetime().strftime("%b %d")
        # ax.text(ex * 1.12, ey * 1.12, 0.08, month, color="cyan", fontsize=8, zorder=12)
        ax.text2D(
            0.02,
            0.02,
            "★ Not to scale",
            transform=ax.transAxes,
            color="gray",
            fontsize=7,
        )

        # ---- style ----
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_zlim(-0.65, 0.65)
        ax.set_xlabel("x (Astro. Unit)", color="white", fontsize=8, labelpad=1)
        ax.set_ylabel("y (Astro. Unit)", color="white", fontsize=8, labelpad=1)
        ax.set_zlabel("z (Astro. Unit)", color="white", fontsize=8, labelpad=1)
        ax.set_title("Solar System", color="white", fontsize=9, pad=6)
        ax.set_facecolor("#0d0d0d")
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#404040")
        ax.tick_params(colors="gray", labelsize=7)
        # ax.view_init(elev=25, azim=50)
        ax.view_init(elev=28, azim=50)
        ax.legend(
            loc="upper right",
            fontsize=7,
            labelcolor="white",
            facecolor="#1a1a1a",
            edgecolor="#404040",
            framealpha=0.7,
            markerscale=0.5,
            scatterpoints=1,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_earth_location(self) -> EarthLocation:
        """Build an :class:`~astropy.coordinates.EarthLocation` from the station."""
        lat_sign = 1 if self.station.NSsemisphere == "N" else -1
        lon_sign = 1 if self.station.EWsemisphere == "E" else -1
        return EarthLocation(
            lat=(lat_sign * self.station.latitude).to(unit.deg),
            lon=(lon_sign * self.station.longitude).to(unit.deg),
            height=self.station.elevation.to(unit.m),
        )
