"""Tests for axionbloch.MilkyWayAxionHalo and related utilities.

Also covers :func:`~axionbloch.utils.check_norm` and the end-to-end
``Simulations`` pipeline that ties together an axion model, an NMR sample,
and a magnet.

Environment variables
---------------------
PRINT_RESULTS=1
    Print intermediate values and call ``displayTrjries()`` after simulation.
SEED
    Integer random seed (default 42).
NUM_FIELD
    Number of stochastic field realisations (default 1000).

Useful run commands::

    pytest tests/test_MilkyWayAxionHalo.py -q -s
    pytest tests/test_MilkyWayAxionHalo.py::test_MilkyWayAxionHalo_initialization -q -s
    pytest tests/test_MilkyWayAxionHalo.py -k "initialization or RabiFreq" -q -s
    $env:PRINT_RESULTS="1"; pytest tests/test_MilkyWayAxionHalo.py -s
"""
import os
import warnings
import pytest

from axionbloch.dependency import *
from axionbloch.utils import check_norm, check

# Gyromagnetic ratio and magnetic dipole moment of Xe-129
from axionbloch.constants import gamma_Xe129, mu_Xe129, gamma_p, mu_p

# classes for simulations
from axionbloch.SimuTools import MagField, Simulations
from axionbloch.SimuTypes import SimuParams
from axionbloch.Sample import Sample
from axionbloch.Apparatus import Magnet
from axionbloch.MilkyWayAxionHalo import MilkyWayAxionHalo

PRINT_RESULTS = os.getenv("PRINT_RESULTS", "0") == "1"
SEED = int(os.getenv("SEED", "42"))
NUM_FIELD = int(os.getenv("NUM_FIELD", "1000"))


LXe = Sample(
    name="Liquid Xe-129",
    gamma=gamma_Xe129,
    massDensity=3.1 * unit.g * unit.cm ** (-3),
    molarMass=131.29 * unit.g / unit.mol,
    numOfSpinsPerMolecule=1 * unit.one,
    T2=10 * unit.minute,
    T1=15 * unit.minute,
    vol=1 * unit.cm**3,
    mu=mu_Xe129,
    temp=163 * unit.K,
    verbose=False,
)

# CH3OH
methanol = Sample(
    name="C-12 Methanol",
    gamma=gamma_p,
    massDensity=0.792 * unit.g * unit.cm ** (-3),
    molarMass=32.04 * unit.g / unit.mol,
    numOfSpinsPerMolecule=4 * unit.one,
    T2=1 * unit.s,
    T1=5 * unit.s,
    vol=1 * unit.cm**3,
    mu=mu_p,
    temp=300 * unit.K,
    verbose=False,
)

# CH3CH2OH
ethanol = Sample(
    name="Ethanol",
    gamma=gamma_p,
    massDensity=0.78945 * unit.g * unit.cm ** (-3),
    molarMass=46.069 * unit.g / unit.mol,
    numOfSpinsPerMolecule=6 * unit.one,
    T2=1 * unit.s,
    T1=5 * unit.s,
    vol=1 * unit.cm**3,
    mu=mu_p,
    temp=300 * unit.K,
    verbose=False,
)

samples = []

magnet_2ppm = Magnet(B0=1.5 * unit.T, FWHM=2.0 * ppm, nFWHM=100, verbose=PRINT_RESULTS)
magnet_2ppb = Magnet(
    B0=2.0 * unit.T, FWHM=5.0 * ppb, nFWHM=10, numPt=100, verbose=PRINT_RESULTS
)

magnets = []


def test_MilkyWayAxionHalo_initialization():
    """MilkyWayAxionHalo initialises with nu_a or m_a alone, and raises when neither is given.

    Verified cases:
    1. ``nu_a`` only — minimum valid input.
    2. ``m_a`` only — alternative mass-based input.
    3. All optional parameters explicit — full construction path.
    4. No ``nu_a`` / ``m_a`` — must raise an AssertionError (counted as the
       expected single error).

    TODO: test illegal units for parameters.
    TODO: test inconsistent nu_a / m_a pair.
    """
    # TODO: test illegal units for parameters
    # TODO: test inconsistent parameters (e.g. nu_a and m_a that do not match)
    msgPrefix = f"{test_MilkyWayAxionHalo_initialization.__name__} "
    errors = []

    # minimum information
    try:
        axion = MilkyWayAxionHalo(
            nu_a=1 * unit.MHz,
            verbose=False,
        )
    except Exception as exc:
        msg = msgPrefix + f"failed: {exc}"
        if PRINT_RESULTS:
            print(f"failed: {exc}")
            print(msg)
        errors.append(msg)

    # m_a=1 * unit.kg,
    try:
        axion = MilkyWayAxionHalo(
            m_a=1 * unit.kg,
            verbose=False,
        )
    except Exception as exc:
        msg = msgPrefix + f"failed: {exc}"
        if PRINT_RESULTS:
            print(f"failed: {exc}")
            print(msg)
        errors.append(msg)

    try:
        axion = MilkyWayAxionHalo(
            name="Milky Way Axion Halo",
            nu_a=1 * unit.MHz,
            g_aNN=None,
            Qa=None,
            v_0=220.0 * unit.km / unit.s,
            v_lab=233.0 * unit.km / unit.s,
            rho_E_DM=0.3 * unit.GeV / unit.cm**3,
            verbose=False,
        )
    except Exception as exc:
        msg = msgPrefix + f"failed: {exc}"
        if PRINT_RESULTS:
            print(f"failed: {exc}")
            print(msg)
        errors.append(msg)

    # When no axion Compton frequency is provided, it should raise an error
    try:
        axion = MilkyWayAxionHalo()
    except Exception as exc:
        msg = msgPrefix + f"(no nu_a or m_a, expected error): {exc}"
        if PRINT_RESULTS:
            print(msg)
        errors.append(msg)

    assert (
        len(errors) == 1
    ), f"Expected 1 error due to missing nu_a and m_a, but got {len(errors)} errors: {errors}"


def test_getRabiFreq():
    """getRabiFreq returns a finite quantity with units of Hz."""
    rabi_freq = MilkyWayAxionHalo.getRabiFreq(
        gaNN=1e-9 * unit.GeV ** (-1), verbose=PRINT_RESULTS
    )
    assert rabi_freq.unit.is_equivalent(
        unit.Hz
    ), f"Expected Rabi frequency to have units of Hz, but got {rabi_freq.unit}"
    assert np.isfinite(
        rabi_freq.value
    ), f"Expected Rabi frequency to be finite, but got {rabi_freq.value}"


def test_check_norm_with_quantities():
    """check_norm is silent for a unit-normalised Quantity array and warns when integral ≠ 1."""
    x = np.array([0.0, 1.0, 2.0]) * unit.Hz
    y = np.array([0.0, 1.0, 0.0]) / unit.Hz

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_norm(x, y)

    assert caught == []

    with pytest.warns(UserWarning):
        check_norm(x, 2.0 * y)


def test_check_norm_with_values():
    """check_norm works identically with plain ndarrays (no units)."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_norm(x, y)

    assert caught == []

    with pytest.warns(UserWarning):
        check_norm(x, 2.0 * y)


def test_axion_lineshape():
    """
    For testing axion_lineshape().
    """
    nu_a = 1 * unit.MHz
    v_0 = 220.0 * unit.km / unit.s
    v_lab = 233.0 * unit.km / unit.s

    frequency_arrays = [
        # case 0: frequencies < nu_a
        np.linspace(0.9e6, 0.99e6, 300) * unit.Hz,
        # case 1: fine RBW = 0.05 Hz
        np.linspace(1.0e6 - 10.0, 1.0e6 + 10.0, 1000) * unit.Hz,
        # case 2: coarse RBW = 0.5 Hz
        np.linspace(1.0e6 - 10.0, 1.0e6 + 10.0, 40) * unit.Hz,
        # case 3: max frequencies >> 10 axion linewidths
        np.linspace(1.0e6 - 10.0, 1.0e6 + 1.0e3, 500) * unit.Hz,
    ]

    spectra = [
        MilkyWayAxionHalo.axion_lineshape(
            v_0=v_0,
            v_lab=v_lab,
            nu_a=nu_a,
            nu=frequencies,
            case="grad_perp",
            alpha=0.0 * unit.rad,
            verbose=PRINT_RESULTS,
        )
        for frequencies in frequency_arrays
    ]

    for frequencies, spectrum in zip(frequency_arrays, spectra):
        assert spectrum.shape == frequencies.shape
        assert spectrum.unit.is_equivalent(unit.Hz**-1)
        assert np.all(np.isfinite(spectrum.value))
        # the spectrum should be zero for frequencies < nu_a
        # (allow the boundary frequency closest to nu_a to be non-zero)
        mask = frequencies <= nu_a
        if np.any(mask):
            # Set the last True (boundary frequency) to False
            mask[np.where(mask)[0][-1]] = False
        # Assert spectrum is zero where mask is True (all frequencies < nu_a except boundary)
        assert np.all(spectrum[mask] == 0 / unit.Hz)

    assert np.all(spectra[0] == 0)


def test_getAmpSpectra_stochastic():
    """Stochastic amplitude spectra have the right shape, units, and mean integral ≈ 1."""

    nu_a = 1 * unit.MHz
    v_0 = 220.0 * unit.km / unit.s
    v_lab = 233.0 * unit.km / unit.s
    frequencies = np.linspace(-10, 100, 2000) * unit.Hz + 1 * unit.MHz

    axion = MilkyWayAxionHalo(
        name="Milky Way Axion Halo",
        nu_a=nu_a,
        g_aNN=1.0e-9 * unit.GeV ** (-1),
        Qa=None,
        v_0=v_0,
        v_lab=v_lab,
        windAngle=None,
        rho_E_DM=0.3 * unit.GeV / unit.cm**3,
        verbose=False,
    )

    # Get amplitude spectra without stochasticity
    ampSpectra = axion.getAmpSpectra(
        frequencies=frequencies,
        numSpectra=NUM_FIELD,
        use_stoch=True,
        rand_seed=SEED,
        verbose=PRINT_RESULTS,
    )

    # ampSpectra has shape (numSpectra, len(frequencies));
    # each row is one spectrum.

    assert ampSpectra.shape == (NUM_FIELD, frequencies.shape[0])
    assert ampSpectra.unit.is_equivalent(unit.Hz ** (-0.5))
    assert np.all(np.isfinite(ampSpectra.value))

    # Compute intensity |ampSpectra|^2
    PSD = np.abs(ampSpectra) ** 2

    # Integrate each spectrum over frequency, then average the results.
    integral = np.trapezoid(PSD, frequencies, axis=1)
    integral_mean = np.mean(integral)
    if PRINT_RESULTS:
        print(f"mean integral of |ampSpectra|^2: {integral_mean:.6f}")

    # Assert integral is close to 1 with reasonable tolerance
    assert np.isclose(integral_mean.to_value(unit.one), 1.0, rtol=1e-2)


def test_getAmpSpectra_deterministic():
    """Deterministic amplitude spectrum (use_stoch=False) has |A|² integral ≈ 1."""
    nu_a = 1 * unit.MHz
    v_0 = 220.0 * unit.km / unit.s
    v_lab = 233.0 * unit.km / unit.s
    frequencies = np.linspace(-10, 100, 2000) * unit.Hz + 1 * unit.MHz

    axion = MilkyWayAxionHalo(
        name="Milky Way Axion Halo",
        nu_a=nu_a,
        g_aNN=1.0e-9 * unit.GeV ** (-1),
        Qa=None,
        v_0=v_0,
        v_lab=v_lab,
        windAngle=None,
        rho_E_DM=0.3 * unit.GeV / unit.cm**3,
        verbose=False,
    )

    # Get amplitude spectra without stochasticity
    ampSpectra = axion.getAmpSpectra(
        frequencies=frequencies,
        numSpectra=1,
        use_stoch=False,
        rand_seed=SEED,
        verbose=PRINT_RESULTS,
    )

    # Note: ampSpectra has shape (numSpectra, len(frequencies)) = (1, len(frequencies))
    # Extract the first spectrum to get shape (len(frequencies),)
    ampSpectra = ampSpectra[0]

    assert ampSpectra.shape == frequencies.shape
    assert ampSpectra.unit.is_equivalent(unit.Hz ** (-0.5))
    assert np.all(np.isfinite(ampSpectra.value))

    # Compute intensity |ampSpectra|^2
    PSD = np.abs(ampSpectra) ** 2

    # Integrate intensity over frequency (should equal 1 since PSD integrates to 1)
    integral = np.trapezoid(PSD, frequencies)

    if PRINT_RESULTS:
        print(f"Integral of |ampSpectra|^2: {integral:.6f}")

    # Assert integral is close to 1 with reasonable tolerance
    assert np.isclose(integral.to_value(unit.one), 1.0, rtol=1e-3)


@pytest.mark.parametrize(
    "sample", [LXe, methanol, ethanol], ids=["LXe", "methanol", "ethanol"]
)
@pytest.mark.parametrize(
    "magnet_kwargs",
    [
        {"FWHM": 1 * ppb, "numPt": 1},
        {"FWHM": 2 * ppm, "numPt": 100},
        {"FWHM": 10 * ppm, "numPt": 5000},
    ],
    ids=["1ppb", "2ppm", "10ppm"],
)
def test_Simulation(sample: Sample, magnet_kwargs: dict):
    """End-to-end Bloch simulation: axion + sample + magnet → Simulations.run().

    Parametrised over three samples (LXe, methanol, ethanol) and three
    magnet homogeneities (1 ppb / 2 ppm / 10 ppm).  Verifies that exactly one
    simulation entry is created and that the run completes without error.
    TODO test all scenarios (axion / magnet / sample and test if theory agrees with the simulation)
    """
    g_aNN = 1.0e-9 * unit.GeV ** (-1)

    axion = MilkyWayAxionHalo(
        name="Milky Way Axion Halo",
        nu_a=1 * unit.kHz,
        g_aNN=g_aNN,
        verbose=False,
    )

    # B0 is always determined by the sample's gamma and the axion frequency
    B0 = (axion.nu_a_eff / (sample.gamma / (2 * np.pi))).to(
        unit.T, equivalencies=unit.dimensionless_angles()
    )
    magnet = Magnet(
        B0=B0,
        direction=[0, 0, 1],
        **magnet_kwargs,
    )

    B_a_rms = (axion.getRabiFreq() / (sample.gamma / (2 * np.pi))).to(
        unit.T, equivalencies=unit.dimensionless_angles()
    )

    params: SimuParams = {
        "key_info": {"nu_a": axion.nu_a},
        "axion": axion,
        "sample": sample,
        "magnet": magnet,
        "excField": MagField(),
        "B_a_rms": B_a_rms,
        # Number of random field realizations.
        "numFields": NUM_FIELD,
        "rand_seed": SEED,  # random seed
        # amplitude, polar and azimuthal angle
        # of the initial magnetization
        "init_M": 1 * unit.one,
        "init_M_theta": 0 * unit.rad,
        "init_M_phi": 0 * unit.rad,
        # sampling rate and duration of the time series
        "rate": 1 * unit.Hz,
        "duration": 4000 * unit.s,
    }

    # Create and execute the simulation job collection
    simulations = Simulations(all_params=[params])

    # run the simulation
    simulations.run(verbose=PRINT_RESULTS)

    assert len(simulations.pool) == 1

    # Post-process results with summary stats and plotting
    if PRINT_RESULTS:
        for i, item in enumerate(simulations.pool):
            item.simu.keepMeanStd()
            item.simu.displayTrjries()
