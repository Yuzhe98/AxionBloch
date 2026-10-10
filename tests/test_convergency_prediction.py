"""Unit test for the pulse-aware free-decay theoretical predictor."""

import numpy as np

from axionbloch.Apparatus import Magnet
from examples.convergency.inhomogeneity_numpt_convergence import (
    FWHM,
    N_FWHM,
    PULSE_STEPS,
    RATE,
    RCF_FREQ,
    T2,
    pulse_aware_prediction,
)
from axionbloch.constants import gamma_p
from axionbloch.dependency import PI, unit


def test_pulse_aware_prediction_gives_a_quarter_turn_for_resonant_x_pulse():
    """A constant resonant x pulse should tip equilibrium magnetization to +y."""
    dt_s = (1 / RATE).to_value(unit.s)
    pulse_duration_s = PULSE_STEPS * dt_s
    gamma = gamma_p.to_value(unit.rad * unit.Hz / unit.T)
    # A continuous-time quarter-turn amplitude provides a clear closed-form
    # check of the exact matrix-exponential pulse propagator.
    bx_t = (PI / 2).to_value(unit.rad) / (gamma * pulse_duration_s)
    pulse_b_t = np.zeros((PULSE_STEPS, 2))
    pulse_b_t[:, 0] = bx_t

    magnet = Magnet(
        name="resonant_theory_test",
        B0=RCF_FREQ / (gamma_p / (2 * PI)),
        FWHM=0 * FWHM,
        numPt=1,
        nFWHM=N_FWHM,
    )
    magnet.setHomogeneity(numPt=1)
    predicted = pulse_aware_prediction(
        magnet,
        pulse_b_t,
        np.array([0.0]),
        dt_s,
    )[0]

    # During a quarter-turn, the magnetization spends about half the pulse
    # interval in the transverse plane, where T2 attenuation applies.
    expected_my = np.exp(-pulse_duration_s / (2 * T2.to_value(unit.s)))
    assert abs(predicted.real) < 2e-6
    assert abs(predicted.imag - expected_my) < 2e-5
