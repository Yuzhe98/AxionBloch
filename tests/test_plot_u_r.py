import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as unit

from axionbloch.EarthBoundAxionHalo import EarthBoundAxionHalo
from axionbloch.utils import linestyles


def test_plot_u_r_solves_and_matches_example():
    halo = EarthBoundAxionHalo(nu_a=1 * unit.MHz, N=64, extent=16 * unit.R_earth)
    assert not halo.states
    fig, ax = halo.plot_u_r(showPlot=False)
    names = ["1s", "2s", "3s", "2p", "4s", "3d", "5s", "3p", "6s", "4f"]
    assert [line.get_label() for line in ax.lines] == names
    assert len(halo.states) == 80
    np.testing.assert_allclose(fig.get_size_inches() * 2.54, [8.5, 5.5])
    assert fig.dpi == 300
    assert ax.get_legend() is None
    for i, (line, name) in enumerate(zip(ax.lines, names)):
        np.testing.assert_allclose(line.get_ydata(), halo.states[name]["u_r"].real.to_value(unit.R_earth**-.5))
        np.testing.assert_allclose(line.get_xdata(), halo.r.to_value(unit.R_earth))
        assert line.get_linewidth() == 1.4
        expected, = ax.plot([], [], linestyle=linestyles[i % len(linestyles)])
        assert line.get_linestyle() == expected.get_linestyle()
        expected.remove()
    plt.close(fig)


def test_plot_u_r_reuses_states_and_fills_missing_channels(monkeypatch):
    halo = EarthBoundAxionHalo(nu_a=1 * unit.MHz, N=64, extent=16 * unit.R_earth)
    halo.solve_TISE_3D(l_vals=[0], max_n_r=1)
    original = halo.solve_TISE_3D
    calls = []

    def tracked(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(halo, "solve_TISE_3D", tracked)
    fig, _ = halo.plot_u_r(["1s"], showPlot=False)
    assert calls == []
    plt.close(fig)
    fig, _ = halo.plot_u_r(["1s", "2p"], showPlot=False)
    assert len(calls) == 1
    assert calls[0]["l_vals"] == [1]
    plt.close(fig)
    with pytest.raises(ValueError, match="invalid"):
        halo.plot_u_r(["1p"], showPlot=False)
