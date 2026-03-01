# axionbloch/station.py

class Station:
    def __init__(
        self,
        name="Station name",
        NSsemisphere=None,  # 'N' or 'S'
        EWsemisphere=None,  # 'E' or 'W'
        unit="deg",
        latitude_deg=None,  # in [deg]
        longitude_deg=None,  # in [deg]
        elevation=None,  # in [m]
        verbose=False,
    ):
        """
        initialize a station on Earth
        """
        if name is None:
            raise ValueError("name is None")

        self.name = name
        self.NSsemisphere = NSsemisphere
        self.EWsemisphere = EWsemisphere
        # if latitude is None:
        #     raise ValueError('latitude is None')
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        # if NSsemisphere == 'N':
        #     self.theta = np.pi/2 - self.latitude
        # elif NSsemisphere == 'S':
        #     self.theta = np.pi/2 + self.latitude
        # else:
        #     raise ValueError('NSsemisphere != \'N\' nor \'S\'')

        # if EWsemisphere == 'E':
        #     self.phi = self.longitude
        # elif EWsemisphere == 'W':
        #     self.phi = (-1.)* self.longitude
        # else:
        #     raise ValueError('NSsemisphere != \'N\' nor \'S\'')

        # self.nvec = np.array([np.sin(self.theta)*np.cos(self.phi), np.sin(self.theta)*np.sin(self.phi), np.cos(self.theta)])
        self.elevation = elevation
        self.R = self.elevation + 6356.7523e3  # radius
        # self.rvec = self.R * self.nvec


Mainz = Station(
    name="Mainz",
    NSsemisphere="N",  # 'N' or 'S'
    EWsemisphere="E",  # 'E' or 'W'
    unit="deg",
    latitude_deg=49.9916,  # in [deg]
    longitude_deg=(8.0 + 16.0 / 60.0 + 26.2056 / 3600),  # in [deg]
    elevation=130.0,
    verbose=False,
)
