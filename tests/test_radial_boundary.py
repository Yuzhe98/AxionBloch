"""Physical and discrete checks for the positive-radius Dirichlet solver."""

import numpy as np
import pytest
from astropy import constants as const, units as unit

from axionbloch.GravBoundAxionHalo import GravBoundAxionHalo
from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo


def free_radial_box(n):
    return GravBoundAxionHalo(
        m_a=1 * unit.kg, N=n, extent=1 * unit.m,
        getPot=lambda: (lambda r: np.zeros_like(r), unit.m, unit.m**2 / unit.s**2),
    )


def test_box_spectrum_shapes_and_second_order_convergence():
    """Sine eigenfunctions test both endpoints and absence of extra even modes."""
    errors = []
    for n in (63, 127):
        halo = free_radial_box(n)
        halo.solve_TISE_3D(l_vals=[0], max_n_r=3)
        r = halo.r.to_value(unit.m)
        assert halo.r_max == halo.extent
        assert np.isclose((halo.r[-1] + halo.dr).to_value(unit.m), halo.extent.to_value(unit.m))
        assert np.isclose(r[0], 1 / (n + 1))
        assert np.isclose(r[-1], 1 - 1 / (n + 1))
        np.testing.assert_allclose(np.diff(r), halo.dr.to_value(unit.m))
        energies = []
        expected = []
        for k in range(1, 4):
            state = halo.states[f"{k}s"]
            u = state["u_r"].to_value(unit.m**-0.5)
            np.testing.assert_allclose(u, np.sqrt(2) * np.sin(k * np.pi * r), atol=1e-11)
            energies.append(state["eigenE"].to_value(unit.J))
            expected.append((const.hbar**2 * (k * np.pi)**2 / (2 * halo.m_a * halo.r_max**2)).to_value(unit.J))
            assert state["n_r"] == k - 1
        errors.append(np.max(np.abs(np.array(energies) / expected - 1)))
    assert 3.9 < errors[0] / errors[1] < 4.1


@pytest.mark.parametrize("n", [128, 129])
def test_all_channels_normalized_orthogonal_and_energy_consistent(n):
    halo = EarthBoundAxionHalo(nu_a=1 * unit.MHz, N=n, extent=16 * unit.R_earth)
    halo.solve_TISE_3D(l_vals=[0, 1, 2, 3], max_n_r=3)
    for l, label in enumerate("spdf"):
        wavefunctions = []
        for i in range(3):
            state = halo.states[f"{i+l+1}{label}"]
            u = state["u_r"]
            wavefunctions.append(u.to_value(halo.r.unit**-0.5))
            assert np.all(np.isfinite(state["R_r"]))
            assert np.isclose((halo.dr * np.sum(abs(u)**2)).to_value(unit.one), 1, atol=1e-13)
            ratio = (state["eigenE_expect"] / state["eigenE"]).to_value(unit.one)
            assert np.isclose(ratio, 1, rtol=1e-10)
        matrix = np.asarray(wavefunctions)
        np.testing.assert_allclose(matrix @ matrix.T * halo.dr.value, np.eye(3), atol=1e-12)


def test_p_channel_keeps_consecutive_radial_states():
    # First three zeros of the spherical Bessel function j_1.
    roots = np.array([4.493409457909064, 7.725251836937707, 10.904121659428899])
    halo = free_radial_box(511)
    halo.solve_TISE_3D(l_vals=[1], max_n_r=3)
    scale = const.hbar**2 / (2 * halo.m_a * halo.r_max**2)
    energies = [(halo.states[f"{n}p"]["eigenE"] / scale).to_value(unit.one) for n in (2, 3, 4)]
    np.testing.assert_allclose(energies, roots**2, rtol=5e-5)


def test_earth_s_states_approach_zero_under_refinement():
    samples = []
    for n in (1023, 2047):
        halo = EarthBoundAxionHalo(nu_a=1 * unit.MHz, N=n, extent=16 * unit.R_earth)
        halo.solve_TISE_3D(l_vals=[0], max_n_r=3)
        samples.append([abs(halo.states[f"{k}s"]["u_r"][0].value) for k in (1, 2, 3)])
    # Halving r_min halves each regular s-wave amplitude to leading order.
    np.testing.assert_allclose(np.array(samples[1]) / samples[0], 0.5, rtol=0.01)


def test_eigenstate_plot_includes_first_positive_sample():
    import matplotlib.pyplot as plt

    halo = free_radial_box(63)
    halo.solve_TISE_3D(l_vals=[0], max_n_r=1)
    halo.plotEigenStates(numStates=1, showPlot=False)
    line = plt.gcf().axes[0].lines[0]
    assert len(line.get_xdata()) == halo.N
    assert np.isclose(line.get_xdata(orig=False)[0], halo.r[0].value)
    plt.close("all")
