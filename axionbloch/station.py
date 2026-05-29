"""Geographic station definitions for axion-wind calculations.

A :class:`Station` stores the location of an experimental site on Earth and
exposes the corresponding unit vector and distance from the Earth's centre.
Three pre-built instances are provided: :data:`Mainz`, :data:`Baltimore`, and
:data:`Sanya`.
"""
from axionbloch.dependency import np, unit, Quantity
from axionbloch.constants import earth_radius


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

        logPrefix = f"[{self.__class__.__name__}.__init__]"
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
        self.R = self.elevation + earth_radius  # distance from Earth center to station


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

# print(Mainz.name, Mainz.theta.to("deg"), Mainz.phi.to("deg"), Mainz.nvec, Mainz.R)
# print(Baltimore.name, Baltimore.theta.to("deg"), Baltimore.phi.to("deg"), Baltimore.nvec, Baltimore.R)
# print(Sanya.name, Sanya.theta.to("deg"), Sanya.phi.to("deg"), Sanya.nvec, Sanya.R)
