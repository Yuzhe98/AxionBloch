"""Tests for axionbloch.Apparatus.Magnet.

Environment variables
---------------------
PRINT_RESULTS=1
    Print diagnostic output (e.g. histograms) during the test run.
SEED
    Integer random seed (default 42); reserved for future stochastic tests.

Run with::

    pytest tests/test_Magnet.py -q
    $env:PRINT_RESULTS="1"; pytest tests/test_Magnet.py -s
"""

import os

import pytest

from axionbloch.Apparatus import Magnet
from axionbloch.dependency import *

PRINT_RESULTS = os.getenv("PRINT_RESULTS", "0") == "1"
SEED = int(os.getenv("SEED", "42"))


def test_magnet_uses_astropy_quantities_by_default():
    """Magnet stores B0 in Tesla and FWHM as dimensionless; homogeneous by default (numPt=1)."""
    magnet = Magnet(B0=1.5 * unit.T, FWHM=2.0 * ppm, nFWHM=10, verbose=PRINT_RESULTS)

    assert magnet.B0.unit == unit.T
    assert magnet.B0.to_value(unit.T) == pytest.approx(1.5)
    assert magnet.FWHM.unit.is_equivalent(unit.one)
    assert magnet.FWHM.to_value(unit.one) == pytest.approx(2.0e-6)
    assert magnet.ratios.shape == (1,)
    assert magnet.ratios[0] == pytest.approx(1.0)


def test_magnet_homogeneity_sampling_normalizes_weights():
    """setHomogeneity produces non-negative weights that sum to exactly 1."""
    magnet = Magnet(
        B0=2.0 * unit.T, FWHM=5.0 * ppm, nFWHM=10, numPt=100, verbose=PRINT_RESULTS
    )
    magnet.setHomogeneity(numPt=100, showPlot=False)

    assert magnet.ratios.shape == (100,)
    assert np.all(magnet.ratios >= 0)
    assert np.sum(magnet.ratios) == pytest.approx(1.0)
