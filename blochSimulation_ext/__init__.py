# __init__.py for blochSimulation package

# Import the compiled pybind11 module
from .blochsimulation import (
    add_3d_arrays_3loops,
    add_3d_arrays_parallel,
    add_3d_arrays_flattern_parallel,
    add_3d_arrays_flattern_SIMD,
    generateTrajectories,
    # burkert_potential,  
    burkert_potential_vector
)

# Optional: define __all__ for IDEs and wildcard imports
__all__ = [
    "add_3d_arrays_3loops",
    "add_3d_arrays_parallel",
    "add_3d_arrays_flattern_parallel",
    "add_3d_arrays_flattern_SIMD",
    "generateTrajectories",
    # "burkert_potential",
    "burkert_potential_vector"
]
