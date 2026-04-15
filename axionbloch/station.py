# axionbloch/Station.py
import numpy as np
from axionbloch.enphylope import PhysicalQuantity as PQ
from axionbloch.constants import earth_radius


class Station:
    def __init__(
        self,
        name: str,
        *,
        NSsemisphere: str,  # 'N' or 'S'
        EWsemisphere: str,  # 'E' or 'W'
        latitude: PQ,
        longitude: PQ,
        elevation: PQ,
        verbose: bool = False,
    ):
        """
        A station on Earth. 
        The station's position is specified by its latitude, longitude, and elevation.
        The latitude and longitude are specified by PhysicalQuantity objects, with the N/S and E/W semispheres indicated. 
        The loaction is converted to spherical coordinates (theta, phi). phi=0 is the prime meridian, and phi increases towards the east. theta=0 is the north pole, and theta increases towards the south pole.
        """

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
            self.theta = PQ(np.pi / 2, "rad") - self.latitude
        elif NSsemisphere == "S":
            self.theta = PQ(np.pi / 2, "rad") + self.latitude
        else:
            raise ValueError("NSsemisphere must be 'N' or 'S'")

        if EWsemisphere == "E":
            self.phi = self.longitude
        elif EWsemisphere == "W":
            self.phi = (-1.0) * self.longitude
        else:
            raise ValueError("EWsemisphere must be 'E' or 'W'")
        self.theta = self.theta.to("rad")
        self.phi = self.phi.to("rad")
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
    latitude=PQ(49.991247363525154, "deg"),
    longitude=PQ(8.235360426933486, "deg"),
    elevation=PQ(100.0, "m"),
    verbose=False,
)

Baltimore = Station(
    name="Baltimore",
    NSsemisphere="N",  # 'N' or 'S'
    EWsemisphere="W",  # 'E' or 'W'
    latitude=PQ(39.32948159004821, "deg"),
    longitude=PQ(76.62023874324737, "deg"),
    elevation=PQ(35.0, "m"),
    verbose=False,
)

Sanya = Station(
    name="Sanya",
    NSsemisphere="N",  # 'N' or 'S'
    EWsemisphere="E",  # 'E' or 'W'
    latitude=PQ(18.2546815, "deg"),
    longitude=PQ(109.5076269, "deg"),
    elevation=PQ(168.0, "m"),
    verbose=False,
)

# print(Mainz.name, Mainz.theta.to("deg"), Mainz.phi.to("deg"), Mainz.nvec, Mainz.R)
# print(Baltimore.name, Baltimore.theta.to("deg"), Baltimore.phi.to("deg"), Baltimore.nvec, Baltimore.R)
# print(Sanya.name, Sanya.theta.to("deg"), Sanya.phi.to("deg"), Sanya.nvec, Sanya.R)
