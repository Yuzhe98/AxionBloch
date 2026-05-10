import os
import pytest

from axionbloch.dependency import *
from axionbloch.Sample import Sample
from axionbloch.constants import gamma_p, mu_p, gamma_Xe129, mu_Xe129
from axionbloch.Apparatus import Magnet

PRINT_RESULTS = os.getenv("PRINT_RESULTS", "0") == "1"
SEED = int(os.getenv("SEED", "42"))
# NUM_FIELD = int(os.getenv("NUM_FIELD", "1000"))


def test_magnet_uses_astropy_quantities_by_default():
    magnet = Magnet(B0=1.5 * unit.T, FWHM=2.0 * ppm, nFWHM=10, verbose=PRINT_RESULTS)

    assert magnet.B0.unit == unit.T
    assert magnet.B0.to_value(unit.T) == pytest.approx(1.5)
    assert magnet.FWHM.unit.is_equivalent(unit.one)
    assert magnet.FWHM.to_value(unit.one) == pytest.approx(2.0e-6)
    assert magnet.ratios.shape == (1,)
    assert magnet.ratios[0] == pytest.approx(1.0)


def test_magnet_homogeneity_sampling_normalizes_weights():
	magnet = Magnet(B0=2.0 * unit.T, FWHM=5.0 * ppm, nFWHM=10, numPt=100, verbose=PRINT_RESULTS)
	magnet.setHomogeneity(numPt=100, showPlot=PRINT_RESULTS)

	assert magnet.ratios.shape == (100,)
	assert np.all(magnet.ratios >= 0)
	assert np.sum(magnet.ratios) == pytest.approx(1.0)
