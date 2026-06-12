"""Tests for axionbloch.Sample.Sample.

Covers three representative samples via a parametrised fixture:

- ``methanol``    – proton-bearing liquid (¹H NMR)
- ``liquid Xe-129`` – spin-½ noble-gas NMR
- ``negative temperature Xe-129`` – verifies the ValueError guard in
  :meth:`~axionbloch.Sample.Sample.getThermalPol` for unphysical temperatures

Test functions
--------------
test_sample_initialization_runs
    Checks that spin number density and total spin count are finite and have
    the correct units after construction.
test_getThermalPol_runs
    Checks that thermal polarization is finite and dimensionless.
test_getThermalPol_raises_below_absolute_zero
    Verifies that a negative temperature raises ``ValueError``.
test_getM0_runs
    Checks that magnetization M₀ is finite and has units of A/m.
test_getM0eqb_runs
    Checks the equilibrium magnetization end-to-end pipeline.
"""

import numpy as np
import pytest
from astropy import units as unit

from axionbloch.constants import gamma_p, gamma_Xe129, mu_p, mu_Xe129
from axionbloch.Sample import Sample

# def _to_angular_gamma(gamma: unit.Quantity) -> unit.Quantity:
#     return gamma.to(1 / (unit.s * unit.T)) * unit.rad


SAMPLE_CASES = [
    {
        "name": "methanol",
        "gamma": gamma_p,
        "massDensity": 0.792 * unit.g * unit.cm ** (-3),
        "molarMass": 32.04 * unit.g / unit.mol,
        "numOfSpinsPerMolecule": 4 * unit.one,
        "T2": 1 * unit.s,
        "T1": 5 * unit.s,
        "vol": 1 * unit.cm**3,
        "mu": mu_p,
        "temp": 300.0 * unit.K,
        "pol": None,
    },
    {
        "name": "liquid Xe-129",
        "gamma": gamma_Xe129,
        "massDensity": 3.1 * unit.g * unit.cm ** (-3),
        "molarMass": 131.29 * unit.g / unit.mol,
        "numOfSpinsPerMolecule": 1 * unit.one,
        "T2": 10 * unit.minute,
        "T1": 15 * unit.minute,
        "vol": 1 * unit.cm**3,
        "mu": mu_Xe129,
        "temp": 165.0 * unit.K,
        "pol": 1e-6 * unit.one,
    },
    {
        "name": "negative temperature Xe-129",
        "gamma": gamma_Xe129,
        "massDensity": 3.1 * unit.g * unit.cm ** (-3),
        "molarMass": 131.29 * unit.g / unit.mol,
        "numOfSpinsPerMolecule": 1 * unit.one,
        "T2": 10 * unit.minute,
        "T1": 15 * unit.minute,
        "vol": 1 * unit.cm**3,
        "mu": mu_Xe129,
        "temp": -300.0 * unit.K,
        "pol": 1e-6 * unit.one,
    },
]


@pytest.fixture(params=SAMPLE_CASES, ids=lambda case: case["name"])
def sample(request) -> Sample:
    """Parametrised fixture that yields one Sample per entry in SAMPLE_CASES."""
    case = request.param
    return Sample(
        name=case["name"],
        gamma=case["gamma"],
        massDensity=case["massDensity"],
        molarMass=case["molarMass"],
        numOfSpinsPerMolecule=case["numOfSpinsPerMolecule"],
        T2=case["T2"],
        T1=case["T1"],
        vol=case["vol"],
        mu=case["mu"],
        temp=case["temp"],
        pol=case["pol"],
        verbose=False,
    )


def test_sample_initialization_runs(sample: Sample):
    """Sample construction derives spinNumDensity and totalNumOfSpins correctly."""
    assert sample.spinNumDensity.unit.is_equivalent(unit.cm ** (-3))
    assert np.isfinite(sample.spinNumDensity.value)

    assert sample.totalNumOfSpins.unit == unit.one
    assert np.isfinite(sample.totalNumOfSpins.value)


def test_getThermalPol_runs(sample: Sample):
    """getThermalPol returns a finite dimensionless polarization for T > 0."""
    pol = sample.getThermalPol(B_pol=1.0 * unit.T, temp=300.0 * unit.K)

    assert pol.unit == unit.one
    assert np.isfinite(pol.value)


def test_getThermalPol_raises_below_absolute_zero(sample: Sample):
    """getThermalPol raises ValueError for temperatures below absolute zero."""
    with pytest.raises(ValueError, match=r">= 0 K"):
        sample.getThermalPol(B_pol=1.0 * unit.T, temp=-1.0 * unit.K)


def test_getM0_runs(sample: Sample):
    """getM0 returns a finite magnetization with units of A/m."""
    M0 = sample.getM0(pol=1e-6 * unit.one)

    assert M0.unit.is_equivalent(unit.A / unit.m)
    assert np.isfinite(M0.value)


def test_getM0eqb_runs(sample: Sample):
    """getM0eqb chains getThermalPol → getM0 and returns a finite A/m value."""
    M0eqb = sample.getM0eqb(B_pol=1.0 * unit.T, temp=300.0 * unit.K)

    assert M0eqb.unit.is_equivalent(unit.A / unit.m)
    assert np.isfinite(M0eqb.value)


def test_getM0_SPN_runs(sample: Sample):
    """getM0_SPN returns a finite spin-projection-noise magnetization in A/m."""
    M0_SPN = sample.getM0_SPN()

    assert M0_SPN.unit.is_equivalent(unit.A / unit.m)
    assert np.isfinite(M0_SPN.value)


def test_getM0_SPN_scales_as_sqrt_N(sample: Sample):
    """M0_SPN ∝ √N: doubling the volume doubles √N and M0_SPN stays ∝ √(vol)."""
    import copy

    s2 = copy.copy(sample)
    s2.vol = sample.vol * 4
    s2.totalNumOfSpins = sample.totalNumOfSpins * 4  # N ∝ vol

    ratio = s2.getM0_SPN() / sample.getM0_SPN()
    # √(4N)/V2 / (√N/V1) = 2√N / (4V) * V = 0.5  →  ratio = 0.5
    assert abs(ratio.to_value(unit.one) - 0.5) < 1e-9
