import numpy as np

from astropy import units as unit
from astropy.constants import codata2018 as const
from astropy.units import Quantity

from axionbloch.EarthBoundAxionHalo import earth_grav_potential_earth_center


def test_earth_grav_potential_earth_center_returns_valid_objects():
    phi_func, r_unit, phi_unit = earth_grav_potential_earth_center()

    assert callable(phi_func)
    assert r_unit == unit.meter
    assert phi_unit == unit.joule / unit.kilogram


def test_earth_grav_potential_earth_center_is_symmetric_and_finite():
    phi_func, _, _ = earth_grav_potential_earth_center()
    
    test_radius = 1e3 * unit.R_earth
    radii_m = np.linspace(-1 * test_radius.to_value(unit.meter), test_radius.to_value(unit.meter), num=1000)
    phi_pos = phi_func(radii_m)
    phi_neg = phi_func(-radii_m)

    assert np.all(np.isfinite(phi_pos))
    assert np.all(np.isfinite(phi_neg))
    assert np.allclose(phi_pos, phi_neg, rtol=1e-10, atol=1e-10)
