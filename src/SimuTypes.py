# Enable forward references for type hints (Python 3.7+)

# This allows us to reference classes that are defined later or imported only during type checking.
from __future__ import annotations

# Standard typing utilities
from typing import TypedDict, TYPE_CHECKING
from dataclasses import dataclass

# Only import these for type checking to avoid circular imports or runtime overhead
if TYPE_CHECKING:
    from src.SimuTools import (
        Simulation,
        MagField,
    )  # Simulation engine and magnetic field type

# Import physical quantities and modules used in simulation
from src.Envelope import PhysicalQuantity  # physical quantity with units
from src.AxionWind import AxionWind  # axion field information
from src.Sample import Sample  # NMR sample
from src.Apparatus import Magnet  # magnet


# -------------------------------------------------------------------
# TypedDict for simulation parameters
# -------------------------------------------------------------------
# This defines the **structure of a parameter dictionary** passed to a simulation.
# TypedDict allows static type checking for keys and value types.
class SimuParams(TypedDict):
    key_info: object  # key information for the simulation
    axion: AxionWind  # axion field object
    sample: Sample  # NMR sample
    magnet: Magnet  # magnetic field apparatus
    excField: MagField  # excitation (magnetic) field
    B_a_rms: PhysicalQuantity  # RMS axion-induced magnetic field
    numFields: int  # number of magnetic fields
    rand_seed: int  # random seed for reproducibility
    init_M: PhysicalQuantity  # initial magnetization magnitude
    init_M_theta: PhysicalQuantity  # initial polar angle of magnetization
    init_M_phi: PhysicalQuantity  # initial azimuthal angle of magnetization
    rate: PhysicalQuantity  # simulation rate
    duration: PhysicalQuantity  # simulation duration


# -------------------------------------------------------------------
# Dataclass to store a simulation instance and its parameters
# -------------------------------------------------------------------
# `SimuEntry` allows pairing a Simulation object with the parameters
# used to initialize it. Useful for keeping track of multiple runs.
@dataclass
class SimuEntry:
    simu: Simulation  # The actual simulation instance (C++/Python backend)
    params: SimuParams  # Parameters used for this simulation
