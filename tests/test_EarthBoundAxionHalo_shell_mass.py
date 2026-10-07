"""Independent checks of the shell-mass and density class API."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import warnings
from astropy import units as unit

from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo


@pytest.mark.parametrize("count", [0, 9, 10])
def test_sparse_shell_warning_threshold(count):
    halo = EarthBoundAxionHalo.__new__(EarthBoundAxionHalo)
    halo.r = np.arange(1, 21) * unit.m
    halo.extent = 21 * unit.m
    halo.states = {"1s": {"u_r": np.ones(20) * unit.m**-.5}}
    bounds = [.1, .9] * unit.m if count == 0 else [1, count] * unit.m
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = halo.inferHaloMass(1 * unit.kg, bounds,
                                    maxEarthEnclosedMass=None)
    assert len(caught) == (1 if count < 10 else 0)
    if count < 10:
        assert f"{count} radial grid points" in str(caught[0].message)
    assert np.isfinite(result["1s"]["total_mass"])


@pytest.fixture
def halo():
    model = EarthBoundAxionHalo.__new__(EarthBoundAxionHalo)
    # Use a well-resolved radial grid so ordinary API tests do not emit the
    # sparse-shell warning.  Keep the original piecewise-linear profile and
    # its exact integral while sampling it with 100 points.
    model.r = np.r_[
        np.linspace(.5, 1.0, 50),
        np.linspace(1.0, 1.5, 51)[1:],
    ] * unit.m
    model.extent = 2 * unit.m
    model.states = {
        "1s": {
            "u_r": np.interp(model.r.value, [.5, 1.0, 1.5], [.5, 1.0, .5])
            * unit.m**-.5
        },
        "2s": {
            "u_r": np.interp(model.r.value, [.5, 1.0, 1.5], [-.5, -1.0, -.5])
            * 17
            * unit.m**-.5
        },
    }
    return model


@pytest.mark.parametrize("amplitude, expect_warning", [(1e-6, False), (1e-15, True), (1e-31, True)])
def test_small_shell_fraction_warning(amplitude, expect_warning):
    # Resolve a weak shell with ten points, independently of the sparse-grid warning.
    model = EarthBoundAxionHalo.__new__(EarthBoundAxionHalo)
    model.r = np.arange(1, 21) * unit.m
    model.extent = 21 * unit.m
    model.states = {"1s": {"u_r": np.r_[np.full(10, amplitude), np.ones(10)] * unit.m**-.5}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.inferHaloMass(1 * unit.kg, [1, 10] * unit.m,
                                     maxEarthEnclosedMass=None)
    assert len(caught) == int(expect_warning)
    if expect_warning:
        assert "1s: shell_fraction=" in str(caught[0].message)
        assert "does not necessarily imply an inaccurate probability" in str(caught[0].message)
    # Tiny fractions still yield finite masses and reproduce the shell-mass limit.
    inferred = result["1s"]
    assert np.isfinite(inferred["total_mass"])
    np.testing.assert_allclose(inferred["total_mass"] * inferred["shell_fraction"], 1 * unit.kg)


def test_triangle_mass_and_density(halo):
    # Trapezoidal total is 3/4, shell [1,2] is 3/8, hence scale = 8 kg.
    result = halo.inferHaloMass(3 * unit.kg, [100, 200] * unit.cm,
                                maxEarthEnclosedMass=None)
    for state in result:
        np.testing.assert_allclose(result[state]["total_mass"].to(unit.kg), 6 * unit.kg, rtol=1e-5)
        np.testing.assert_allclose(result[state]["enclosed_mass"].to(unit.kg), 6 * unit.kg, rtol=1e-5)
        inner = halo.getEnclosedMass(1 * unit.m)[state]
        np.testing.assert_allclose(inner["mass"].to(unit.kg), 3 * unit.kg, rtol=1e-5)
        density = halo.getDensity(100 * unit.cm)[state]
        u_at_one = np.interp(1.0, halo.r.value, halo.states[state]["u_r"].value) * unit.m**-.5
        expected_density = (
            halo._shell_mass_models[state]["scale"] * abs(u_at_one) ** 2
            / (4 * np.pi * (1 * unit.m) ** 2)
        ).to(unit.g / unit.cm**3)
        np.testing.assert_allclose(density["density"], expected_density, rtol=1e-5)
    np.testing.assert_allclose(halo.getEnclosedMass()["1s"]["mass"].to(unit.kg), 6 * unit.kg, rtol=1e-5)


def test_mass_limit_sets_density_and_plots(halo):
    halo.inferHaloMass(3 * unit.kg, [1, 2] * unit.m, ["1s"],
                       maxEarthEnclosedMass=None)
    radii = np.linspace(.1, 2, 30) * unit.m
    fig, ax, profiles = halo.plotDMdensity(radii, showPlot=False, radius_unit=unit.m)
    assert np.any(profiles["1s"]["density"] > 0 * unit.g/unit.cm**3)
    assert ax.get_yscale() == "log"
    plt.close(fig)


def test_density_plot_uses_shm_units_and_requested_size(halo):
    halo.rho_M_DM_SHM = 2 * unit.kg / unit.m**3
    halo.inferHaloMass(3 * unit.kg, [1, 2] * unit.m,
                       maxEarthEnclosedMass=None)
    fig, ax, profiles = halo.plotDMdensity(np.array([.5, 1., 1.5]) * unit.m, showPlot=False)
    assert np.isclose(fig.get_figwidth() * 2.54, 13)
    assert fig.dpi == 300
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(),
        (profiles["1s"]["density"] / halo.rho_M_DM_SHM).to_value(unit.one),
    )
    assert halo.getEnclosedMass()["1s"]["mass"].unit.is_equivalent(unit.M_earth)
    plt.close(fig)


def test_normalization_snapshots_survive_solver_state_changes(halo):
    halo.inferHaloMass(0 * unit.kg, [1, 2] * unit.m,
                       maxEarthEnclosedMass=None)
    before = halo.getDensity(.7 * unit.m)["1s"]["density"]
    halo.states["1s"]["u_r"] *= 9
    np.testing.assert_allclose(halo.getDensity(.7 * unit.m)["1s"]["density"], before)


def test_invalid_inputs_and_missing_normalization(halo):
    with pytest.raises(ValueError, match="inferHaloMass"):
        halo.getDensity(1 * unit.m)
    with pytest.raises(TypeError, match="mass Quantity"):
        halo.inferHaloMass(3, [1, 2] * unit.m,
                           maxEarthEnclosedMass=None)
    with pytest.raises(ValueError, match="inner"):
        halo.inferHaloMass(3 * unit.kg, [2, 1] * unit.m,
                           maxEarthEnclosedMass=None)
    with pytest.raises(ValueError, match="not been solved"):
        halo.inferHaloMass(3 * unit.kg, [1, 2] * unit.m, ["3s"],
                           maxEarthEnclosedMass=None)
    halo.inferHaloMass(3 * unit.kg, [1, 2] * unit.m,
                       maxEarthEnclosedMass=None)
    with pytest.raises(ValueError, match="positive"):
        halo.getDensity(0 * unit.m)
    with pytest.raises(ValueError, match="exceed"):
        halo.getDensity(3 * unit.m)
    with pytest.raises(ValueError, match="inside the grid"):
        halo.getEnclosedMass(3 * unit.m)


def test_earth_surface_density_selects_inferred_states_and_returns_shm_ratios():
    halo = EarthBoundAxionHalo(nu_a=1 * unit.MHz, N=256, extent=64 * unit.R_earth)
    with pytest.raises(ValueError, match="inferHaloMass"):
        halo.getDensityAtEarthSurface()
    halo.solve_TISE_3D(l_vals=[0], max_n_r=2)
    halo.inferHaloMass(4e-9 * unit.M_earth, [12300, 384000] * unit.km)
    results = halo.getDensityAtEarthSurface(["2s"])
    assert list(results) == ["2s"]
    assert set(halo.getDensityAtEarthSurface()) == {"1s", "2s"}
    reference = halo.rho_M_DM_SHM.to(unit.g / unit.cm**3, equivalencies=unit.mass_energy())
    expected = halo.getDensity(1 * unit.R_earth)["2s"]
    np.testing.assert_allclose(results["2s"]["density"], expected["density"])
    np.testing.assert_allclose(results["2s"]["density_ratio"] * reference,
                               expected["density"])
