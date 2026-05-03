# COMMANDS TO RUN THIS TEST:
# $env:NUMPY_PERF_PRINT_RESULTS="0"; pytest tests/test_MilkyWayAxionHalo.py -s -q --tb=no
# $env:NUMPY_PERF_PRINT_RESULTS="1"; pytest tests/test_MilkyWayAxionHalo.py -s
# pytest tests/test_MilkyWayAxionHalo.py -q -s
# pytest tests/test_MilkyWayAxionHalo.py::test_MilkyWayAxionHalo_initialization -q -s
# pytest tests/test_MilkyWayAxionHalo.py -k "initialization or RabiFreq" -q -s
import os
import time

import pytest

# # numerical computing
# import numpy as np

# # plotting
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec  # for creating subplots
# import matplotlib.ticker as mticker
# from matplotlib.axes import Axes

# # physical units and constants
# from astropy import units as unit
# from astropy.units import Quantity, CompositeUnit
# from astropy.constants import codata2018 as const

from axionbloch.dependency import *
from axionbloch.MilkyWayAxionHalo import MilkyWayAxionHalo, axion_lineshape

PRINT_RESULTS = os.getenv("NUMPY_PERF_PRINT_RESULTS", "0") == "1"
SEED = int(os.getenv("NUMPY_PERF_SEED", "42"))


def test_MilkyWayAxionHalo_initialization():
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

    assert len(errors) == 1, f"Expected 1 error due to missing nu_a and m_a, but got {len(errors)} errors: {errors}"


# def test_MilkyWayAxionHalo_initialization_minimum_info():
#     axion = MilkyWayAxionHalo(
#         name="Milky Way Axion Halo",
#         nu_a=PhysicalQuantity(1.0e6, "Hz"),
#         g_aNN=PhysicalQuantity(1.0e-9, "GeV**(-1)"),
#         Qa=None,
#         v_0=PhysicalQuantity(220.0, "km/s"),
#         v_lab=PhysicalQuantity(233.0, "km/s"),
#         windAngle=None,
#         rho_E_DM=PhysicalQuantity(0.3, "GeV/cm**3"),
#         verbose=PRINT_RESULTS,
#     )


# def test_MilkyWayAxionHalo_initialization():
#     # parameters_all = []
#     # param_minimum_infor =
#     # parameters_all.append(
#     #     {
#     #         "nu_a": 1.0e6 * unit.Hz,
#     #         "g_aNN": 1.0e-9 * unit.GeV**(-1),
#     #         "Qa": None,
#     #         "v_0": 220.0 * unit.km / unit.s,
#     #         "v_lab": 233.0 * unit.km / unit.s,
#     #         "windAngle": None,
#     #         "rho_E_DM": 0.3 * unit.GeV / unit.cm**3,
#     #     }
#     # )
#     axion = MilkyWayAxionHalo(
#         name="Milky Way Axion Halo",
#         nu_a=PhysicalQuantity(1.0e6, "Hz"),
#         g_aNN=PhysicalQuantity(1.0e-9, "GeV**(-1)"),
#         Qa=None,
#         v_0=PhysicalQuantity(220.0, "km/s"),
#         v_lab=PhysicalQuantity(233.0, "km/s"),
#         windAngle=None,
#         rho_E_DM=PhysicalQuantity(0.3, "GeV/cm**3"),
#         verbose=PRINT_RESULTS,
#     )

#     axion.getRabiFreq(verbose=True)
#     frequencies = np.linspace(0.9e6, 1.1e6, 1000)  # in Hz
#     spec = axion.getAmpSpectra(frequencies=frequencies, verbose=True)
#     print(spec.shape)

def test_axion_lineshape():
    """
    For testing axion_lineshape(). 
    """
    
    pass
