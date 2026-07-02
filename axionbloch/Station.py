"""Geographic station definitions for axion-wind calculations.

A :class:`Station` stores the location of an experimental site on Earth and
exposes the corresponding unit vector and distance from the Earth's centre.
"""

from astropy.coordinates import (GCRS, EarthLocation,
                                 get_body_barycentric_posvel)

from axionbloch.dependency import Quantity, np, unit


class Station:
    """A geographic location on Earth's surface.

    Converts latitude / longitude / elevation to spherical coordinates
    (theta, phi) with the following convention:

    - theta = 0 at the north pole, increases towards the south pole.
    - phi = 0 on the prime meridian (Greenwich), increases eastward.
    - ``nvec`` is the outward unit normal at the station's location.
    - ``R`` is the distance from Earth's centre to the station.

    Attributes
    ----------
    name : str
    NSsemisphere : str
        ``'N'`` or ``'S'``.
    EWsemisphere : str
        ``'E'`` or ``'W'``.
    latitude, longitude : Quantity
        Absolute values; hemisphere is tracked via ``NSsemisphere`` /
        ``EWsemisphere``.
    elevation : Quantity
        Height above sea level.
    theta, phi : Quantity
        Spherical coordinates (rad).
    nvec : ndarray, shape (3,)
        Outward unit normal vector in Cartesian coordinates.
    R : Quantity
        Distance from Earth's centre.
    """

    def __init__(
        self,
        name: str,
        *,
        NSsemisphere: str,  # 'N' or 'S'
        EWsemisphere: str,  # 'E' or 'W'
        latitude: Quantity,
        longitude: Quantity,
        elevation: Quantity,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        name : str
            Human-readable label for the station.
        NSsemisphere : str
            ``'N'`` for northern hemisphere, ``'S'`` for southern.
        EWsemisphere : str
            ``'E'`` for east of prime meridian, ``'W'`` for west.
        latitude, longitude : Quantity
            Geographic coordinates (positive magnitude, direction given by
            the hemisphere flags).
        elevation : Quantity
            Height above mean sea level.
        verbose : bool
            Print derived spherical coordinates after construction.
        """

        logPrefix = f"[{self.__class__.__name__}.{self.__init__.__name__}]"
        self.name = name
        self.NSsemisphere = NSsemisphere
        self.EWsemisphere = EWsemisphere
        self.latitude = np.abs(latitude)
        self.longitude = np.abs(longitude)

        required_params = {
            "name": name,
            "NSsemisphere": NSsemisphere,
            "EWsemisphere": EWsemisphere,
            "latitude": latitude,
            "longitude": longitude,
            "elevation": elevation,
        }
        missing_params = [k for k, v in required_params.items() if v is None]
        if missing_params:
            raise ValueError(
                "Missing required parameter(s): " + ", ".join(missing_params)
            )

        if NSsemisphere == "N":
            self.theta = (np.pi / 2) * unit.rad - self.latitude
        elif NSsemisphere == "S":
            self.theta = (np.pi / 2) * unit.rad + self.latitude
        else:
            raise ValueError("NSsemisphere must be 'N' or 'S'")

        if EWsemisphere == "E":
            self.phi = self.longitude
        elif EWsemisphere == "W":
            self.phi = (-1.0) * self.longitude
        else:
            raise ValueError("EWsemisphere must be 'E' or 'W'")
        self.theta = self.theta
        self.phi = self.phi
        self.nvec = np.array(
            [
                np.sin(self.theta) * np.cos(self.phi),
                np.sin(self.theta) * np.sin(self.phi),
                np.cos(self.theta),
            ]
        )
        self.elevation = elevation
        self.R = (
            self.elevation + 1 * unit.earthRad
        )  # distance from Earth center to station

        # signed geographic coordinates for astropy
        _lat = self.latitude if NSsemisphere == "N" else -self.latitude
        _lon = self.longitude if EWsemisphere == "E" else -self.longitude
        self.location = EarthLocation(lat=_lat, lon=_lon, height=elevation)

    def in_solarZ_frame(self, meas_time, output: str = "spherical"):
        """Station position expressed in the Sun–Earth-velocity frame.

        Builds a right-handed orthonormal frame at the given epoch:

        * **ẑ** — Earth → Sun direction.
        * **x̂** — Earth's heliocentric velocity, orthogonalized against ẑ.
        * **ŷ = ẑ × x̂** — completes the right-hand triad.

        The station's geocentric Cartesian position (GCRS) is projected onto
        this triad.

        Parameters
        ----------
        time : :class:`astropy.time.Time`
            Observation epoch.
        output : str
            ``'cartesian'`` → ``(x, y, z)`` each a Quantity [m].
            ``'spherical'`` → ``(r, theta, phi)`` where *theta* ∈ [0, π] is
            the polar angle from ẑ and *phi* ∈ (−π, π] is the azimuthal
            angle from x̂.

        Returns
        -------
        tuple of Quantity
        """

        R = self._solarZ_basis(meas_time)

        # station geocentric position in GCRS (inertial, aligned with ICRS)
        gcrs = self.location.get_itrs(obstime=meas_time).transform_to(
            GCRS(obstime=meas_time)
        )
        r_gcrs = np.array(
            [
                gcrs.cartesian.x.to(unit.m).value,
                gcrs.cartesian.y.to(unit.m).value,
                gcrs.cartesian.z.to(unit.m).value,
            ]
        )

        # project: solar-frame coords = R^T @ r_gcrs
        xyz = (R.T @ r_gcrs) * unit.m

        if output == "cartesian":
            return xyz[0], xyz[1], xyz[2]

        r = np.sqrt(np.sum(xyz**2))
        theta = (
            np.arccos(np.clip((xyz[2] / r).to_value(unit.one), -1.0, 1.0)) * unit.rad
        )
        phi = np.arctan2(xyz[1].value, xyz[0].value) * unit.rad
        return r, theta, phi

    @staticmethod
    def _solarZ_basis(meas_time) -> np.ndarray:
        """Return solar-Z basis vectors as columns in GCRS/ICRS orientation."""
        earth_pos, earth_vel = get_body_barycentric_posvel("earth", meas_time)
        sun_pos, _ = get_body_barycentric_posvel("sun", meas_time)

        # ẑ: Earth → Sun
        earth_to_sun = (sun_pos.xyz - earth_pos.xyz).to(unit.m).value
        z_hat = earth_to_sun / np.linalg.norm(earth_to_sun)

        # x̂: Earth heliocentric velocity, orthogonalized against ẑ
        v = earth_vel.xyz.to(unit.m / unit.s).value
        v_perp = v - np.dot(v, z_hat) * z_hat
        x_hat = v_perp / np.linalg.norm(v_perp)

        # ŷ = ẑ × x̂; columns express the solar-Z basis in GCRS/ICRS.
        y_hat = np.cross(z_hat, x_hat)
        return np.column_stack([x_hat, y_hat, z_hat])

    def rotation_velocity_in_solarZ_frame(self, meas_time) -> Quantity:
        """Station velocity relative to a nonrotating geocentric halo.

        Astropy supplies the velocity caused by Earth's rotation in GCRS.
        This method projects it into the same solar-Z Cartesian basis used by
        :meth:`in_solarZ_frame`.

        Parameters
        ----------
        meas_time : :class:`astropy.time.Time`
            Observation epoch.

        Returns
        -------
        Quantity, shape (3,) [m/s]
            Cartesian velocity ``(v_x, v_y, v_z)`` in the solar-Z frame.
        """
        _, velocity_gcrs = self.location.get_gcrs_posvel(meas_time)
        rotation = self._solarZ_basis(meas_time)
        return (
            rotation.T @ velocity_gcrs.xyz.to_value(unit.m / unit.s)
        ) * unit.m / unit.s

    @staticmethod
    def direction_to_spherical(direction) -> tuple[Quantity, Quantity]:
        """Spherical coordinates (theta, phi) of a Cartesian direction in the Earth frame.

        Uses the same convention as :class:`Station`:
        theta = 0 at north pole, phi = 0 on prime meridian increasing eastward.

        Parameters
        ----------
        direction : array-like, shape (3,)
            Direction vector ``(x, y, z)`` in ECEF Cartesian coordinates.
            Need not be normalized.

        Returns
        -------
        theta : Quantity [rad]
            Polar angle from north pole in ``[0, π]``.
        phi : Quantity [rad]
            Azimuthal angle in ``(-π, π]``.
        """
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        theta = np.arccos(np.clip(d[2], -1.0, 1.0)) * unit.rad
        phi = np.arctan2(d[1], d[0]) * unit.rad
        return theta, phi


Mainz = Station(
    name="Mainz",
    NSsemisphere="N",  # 'N' or 'S'
    EWsemisphere="E",  # 'E' or 'W'
    latitude=49.991247363525154 * unit.deg,
    longitude=8.235360426933486 * unit.deg,
    elevation=100.0 * unit.m,
    verbose=False,
)

Geneva = Station(
    name="Geneva",
    NSsemisphere="N",
    EWsemisphere="E",
    latitude=46.2044 * unit.deg,
    longitude=6.1432 * unit.deg,
    elevation=375.0 * unit.m,
    verbose=False,
)

Baltimore = Station(
    name="Baltimore",
    NSsemisphere="N",  # 'N' or 'S'
    EWsemisphere="W",  # 'E' or 'W'
    latitude=39.32948159004821 * unit.deg,
    longitude=76.62023874324737 * unit.deg,
    elevation=35.0 * unit.m,
    verbose=False,
)

Tokyo = Station(
    name="Tokyo",
    NSsemisphere="N",
    EWsemisphere="E",
    latitude=35.6762 * unit.deg,
    longitude=139.6503 * unit.deg,
    elevation=44.0 * unit.m,
    verbose=False,
)

Mumbai = Station(
    name="Mumbai",
    NSsemisphere="N",
    EWsemisphere="E",
    latitude=19.0760 * unit.deg,
    longitude=72.8777 * unit.deg,
    elevation=14.0 * unit.m,
    verbose=False,
)


Sanya = Station(
    name="Sanya",
    NSsemisphere="N",  # 'N' or 'S'
    EWsemisphere="E",  # 'E' or 'W'
    latitude=18.2546815 * unit.deg,
    longitude=109.5076269 * unit.deg,
    elevation=168.0 * unit.m,
    verbose=False,
)

# TODO check if the locations are correctly converted to spherical coordinates

Sydney = Station(
    name="Sydney",
    NSsemisphere="S",
    EWsemisphere="E",
    latitude=33.8688 * unit.deg,
    longitude=151.2093 * unit.deg,
    elevation=39.0 * unit.m,
    verbose=False,
)

CapeTown = Station(
    name="CapeTown",
    NSsemisphere="S",
    EWsemisphere="E",
    latitude=33.9249 * unit.deg,
    longitude=18.4241 * unit.deg,
    elevation=42.0 * unit.m,
    verbose=False,
)

BuenosAires = Station(
    name="BuenosAires",
    NSsemisphere="S",
    EWsemisphere="W",
    latitude=34.6037 * unit.deg,
    longitude=58.3816 * unit.deg,
    elevation=25.0 * unit.m,
    verbose=False,
)
