"""Type definitions shared across the simulation layer.

:class:`SimuParams` (TypedDict) describes the full set of parameters passed to
a Bloch-equation simulation run. :class:`SimuEntry` (dataclass) pairs a live
:class:`~axionbloch.SimuTools.Simulation` object with its :class:`SimuParams`
for bookkeeping across multiple runs.
"""

# Enable forward references for type hints (Python 3.7+)

# This allows us to reference classes that are defined later or imported only during type checking.
from __future__ import annotations

from dataclasses import dataclass
# Standard typing utilities
from typing import TYPE_CHECKING, TypedDict

# Only import these for type checking to avoid circular imports or runtime overhead
if TYPE_CHECKING:
    from axionbloch.SimuTools import (  # Simulation engine and magnetic field type
        MagField, Simulation)

# Import physical quantities and modules used in simulation
from axionbloch.Apparatus import Magnet  # magnet
from axionbloch.dependency import *  # physical quantity with units
from axionbloch.MilkyWayAxionHalo import \
    MilkyWayAxionHalo  # axion field information
from axionbloch.Sample import Sample  # NMR sample


# -------------------------------------------------------------------
# TypedDict for simulation parameters
# -------------------------------------------------------------------
# This defines the **structure of a parameter dictionary** passed to a simulation.
# TypedDict allows static type checking for keys and value types.
class SimuParams(TypedDict):
    """Typed dictionary describing all parameters for a single simulation run.

    Pass an instance of this dict to :class:`~axionbloch.SimuTools.Simulations`
    as one element of ``all_params``.

    Attributes
    ----------
    key_info : dict
        Arbitrary metadata (e.g. ``{"nu_a": axion.nu_a}``), printed during
        verbose runs and stored alongside results.
    axion : MilkyWayAxionHalo
        Axion field model object.
    sample : Sample
        NMR sample object.
    magnet : Magnet
        Static bias-field object.
    excField : MagField
        Excitation / pseudomagnetic field object.
    B_a_rms : Quantity, optional
        RMS axion-induced pseudomagnetic field amplitude (T).
    numFields : int
        Number of independent stochastic field realizations.
    rand_seed : int
        Random seed for reproducible field realizations.
    init_M : Quantity
        Initial magnetization magnitude (dimensionless, normalized by M0).
    init_M_theta : Quantity [rad]
        Initial polar angle of the magnetization vector.
    init_M_phi : Quantity [rad]
        Initial azimuthal angle of the magnetization vector.
    rate : Quantity [Hz]
        Output sampling rate of the simulation.
    duration : Quantity [s]
        Total duration of the simulation.
    """

    key_info: object | None = {}  # key information for the simulation
    axion: MilkyWayAxionHalo  # axion field object
    sample: Sample  # NMR sample
    magnet: Magnet  # magnetic field apparatus
    excField: MagField  # excitation (magnetic) field
    B_a_rms: Quantity | None  # RMS axion-induced magnetic field
    numFields: int  # number of magnetic fields
    rand_seed: int  # random seed for reproducibility
    init_M: Quantity | None  # initial magnetization magnitude
    init_M_theta: Quantity  # initial polar angle of magnetization
    init_M_phi: Quantity  # initial azimuthal angle of magnetization
    rate: Quantity | None  # simulation rate
    duration: Quantity | None  # simulation duration


# -------------------------------------------------------------------
# Dataclass to store a simulation instance and its parameters
# -------------------------------------------------------------------
# `SimuEntry` allows pairing a Simulation object with the parameters
# used to initialize it. Useful for keeping track of multiple runs.
@dataclass
class SimuEntry:
    """Pair of a completed :class:`~axionbloch.SimuTools.Simulation` and its parameters.

    Stored in :attr:`~axionbloch.SimuTools.Simulations.pool` after a run.

    Attributes
    ----------
    simu : Simulation
        The executed simulation instance with result trajectories.
    params : SimuParams
        The parameter dictionary used to configure and run ``simu``.
    """

    simu: Simulation  # The actual simulation instance (C++/Python backend)
    params: SimuParams  # Parameters used for this simulation
