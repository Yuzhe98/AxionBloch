"""Check hard-pulse amplitude calibration against a free-decay measurement."""

import importlib.util
from pathlib import Path

import numpy as np

from axionbloch.dependency import unit


# The example filename uses a hyphen, so load it by path instead of importing
# it as a dotted Python module.
_EXAMPLE_PATH = (
    Path(__file__).parents[1] / "examples" / "convergency" / "calibration-excitation.py"
)
_SPEC = importlib.util.spec_from_file_location("calibration_excitation", _EXAMPLE_PATH)
_EXAMPLE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EXAMPLE)
free_decay_calibrated_amplitude = _EXAMPLE.free_decay_calibrated_amplitude
helper_amplitude = _EXAMPLE.helper_amplitude
simulate_free_decay = _EXAMPLE.simulate_free_decay


def test_ninety_degree_helper_amplitude_matches_free_decay_calibration():
    """The independent free-decay fit should measure the helper's 90-degree tip."""
    helper_b1 = helper_amplitude(90.0 * unit.deg)
    free_decay_b1 = free_decay_calibrated_amplitude(90.0 * unit.deg)
    measured = simulate_free_decay(helper_b1)

    # A small T2 relaxation is present during the finite pulse; the helper
    # intentionally calibrates the no-relaxation RK4 rotation.
    assert abs(np.rad2deg(measured["angle_rad"]) - 90.0) < 0.002
    assert abs(free_decay_b1 / helper_b1 - 1.0) < 3e-5
