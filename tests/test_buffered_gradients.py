"""Check cutoff derivatives independently of the eigenvalue solver."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as unit
from astropy.time import Time

from axionbloch.GravBoundAxionHalo import GravBoundAxionHalo


def synthetic_halo(r, power):
    halo = GravBoundAxionHalo.__new__(GravBoundAxionHalo)
    halo.r = r * unit.m
    halo.N = len(r)
    halo.m_a = 1 * unit.kg
    halo.states = {"1s": {
        "R_r": (1 + 0.5j) * r**power * unit.m**-1.5,
        "n_r": 0, "l": 0,
        "eigenE": 0 * unit.J, "eigenE_expect": 0 * unit.J,
    }}
    return halo


def evaluate(halo, cutoff):
    station = SimpleNamespace(in_solarZ_frame=lambda **kwargs:
                              (1 * unit.m, 1 * unit.rad, 2 * unit.rad))
    return halo.findGradientsAtDirection(
        {"1s": 1}, station=station, meas_time=Time("2022-12-14"),
        truncRadius=cutoff, include_lorentz_boost=False, showPlot=False,
    )


@pytest.mark.parametrize("nonuniform", [False, True])
@pytest.mark.parametrize("fraction", [0.0, 0.4])
def test_cutoff_matches_full_grid_stencil(nonuniform, fraction):
    r = np.linspace(0.1, 2, 24)
    if nonuniform:
        r = r**1.5
    halo = synthetic_halo(r, power=3)
    cutoff = (r[8] + fraction * (r[9] - r[8])) * unit.m
    result = evaluate(halo, cutoff)
    # Reference derivative uses the full grid, not a truncated array.
    derivative = np.gradient(r**3, r, edge_order=2)
    expected = np.interp(result[2].to_value(unit.m), r, derivative)
    expected = expected * (1 + 0.5j) / np.sqrt(4 * np.pi)
    np.testing.assert_allclose(result[3].to_value(unit.m**-2.5), expected, rtol=1e-12)
    assert len(result[0]) == 9
    assert len(result[1]) == 9
    assert np.all(result[0] <= cutoff)
    assert np.isclose(result[2][-1].to_value(unit.m), cutoff.to_value(unit.m))


@pytest.mark.parametrize("cutoff_index", [0, 1, -1, None])
def test_quadratic_derivative_is_exact_at_edges(cutoff_index):
    r = np.linspace(0.1, 2, 12)
    halo = synthetic_halo(r, power=2)
    cutoff = None if cutoff_index is None else halo.r[cutoff_index]
    result = evaluate(halo, cutoff)
    expected = 2 * result[2].to_value(unit.m) * (1 + 0.5j) / np.sqrt(4 * np.pi)
    np.testing.assert_allclose(result[3].to_value(unit.m**-2.5), expected, atol=1e-12)


def test_cutoff_outside_sampled_domain_is_rejected():
    halo = synthetic_halo(np.linspace(0.1, 2, 12), power=2)
    for cutoff in (0 * unit.m, 3 * unit.m):
        with pytest.raises(ValueError, match="sampled radial grid"):
            evaluate(halo, cutoff)
