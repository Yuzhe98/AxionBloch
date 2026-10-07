import numpy as np
from astropy import units as unit
from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo

def test_partial_cell_integral_and_complex_amplitude():
    r = np.array([0., .3, .8, 2.]) * unit.km
    u = (1+2j) * r / unit.km**1.5
    actual = EarthBoundAxionHalo._integrateRadialProbability(
        r, u, 200 * unit.m, 1300 * unit.m
    )
    assert isinstance(actual, unit.Quantity)
    assert actual.unit.is_equivalent(unit.one)
    # Trapezoids at 0.2, 0.3, 0.8, 1.3 km, including both interpolated ends.
    np.testing.assert_allclose(actual, 3.8575 * unit.one)


def test_inferred_density_is_independent_of_wavefunction_scale():
    # A triangular wavefunction sampled on 100 points, avoiding a sparse-shell
    # warning while retaining the exact piecewise-linear profile.
    halo = EarthBoundAxionHalo.__new__(EarthBoundAxionHalo)
    halo.r = np.r_[
        np.linspace(.5, 1.0, 50),
        np.linspace(1.0, 1.5, 51)[1:],
    ] * unit.R_earth
    halo.extent = 2 * unit.R_earth
    halo.states = {
        "1s": {
            "u_r": np.interp(halo.r.value, [.5, 1.0, 1.5], [.5, 1.0, .5])
            * unit.R_earth**-.5
        }
    }
    radii = np.array([.5, 1., 1.7]) * unit.R_earth
    halo.inferHaloMass(2e-9 * unit.M_earth, [.7, 1.7] * unit.R_earth, ["1s"])
    density = halo.getDensity(radii)["1s"]["density"]
    total_mass = halo.getEnclosedMass()["1s"]["mass"]
    halo.states["1s"]["u_r"] *= -17
    halo.inferHaloMass(2e-9 * unit.M_earth, [.7, 1.7] * unit.R_earth, ["1s"])
    scaled_density = halo.getDensity(radii)["1s"]["density"]
    scaled_total_mass = halo.getEnclosedMass()["1s"]["mass"]
    np.testing.assert_allclose(density, scaled_density)
    np.testing.assert_allclose(total_mass, scaled_total_mass)
    assert density.unit.is_equivalent(unit.g / unit.cm**3)


def test_tiny_shell_is_integrated_without_cdf_subtraction():
    r = np.array([0., 1., 2., 3., 4.]) * unit.m
    u = np.array([0., 1., 1e-15, 1e-15, 0.]) * unit.m**-.5
    actual = EarthBoundAxionHalo._integrateRadialProbability(
        r, u, 2 * unit.m, 3 * unit.m
    )
    np.testing.assert_allclose(actual, 1e-30 * unit.one, rtol=1e-14, atol=0)


def test_trapezoidal_probability_converges_for_curved_wavefunction():
    errors = []
    # u=r^2 gives integral_0^1 |u|^2 dr = 1/5.
    for n in (101, 201):
        r = np.linspace(0, 1, n) * unit.m
        u = r**2 / unit.m**2.5
        result = EarthBoundAxionHalo._integrateRadialProbability(
            r, u, 0 * unit.m, 1 * unit.m
        )
        errors.append(abs(result - .2 * unit.one))
    assert 3.9 < errors[0] / errors[1] < 4.1
