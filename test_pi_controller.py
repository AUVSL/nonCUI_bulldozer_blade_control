"""Separate PI control, time integration, and actuator feedback checks."""
import numpy as np
import pytest
from util import Control, PIControl, DozerSimulation


def test_pi_accumulates_and_resets():
    controller = PIControl(1.2, .1, 1.)
    error = controller.blade_deformation_errors(0., 0.)[0]
    for n in range(1, 4):
        output = controller.proportional_blade_controller(0., 0.)
        np.testing.assert_allclose(controller.integral_error, error * n * .1)
        np.testing.assert_allclose(output, controller.Kp * error + controller.Ki * error * n * .1)
    controller.reset_integrator()
    np.testing.assert_array_equal(controller.integral_error, 0.)


@pytest.mark.parametrize("limit", ["rate", "angle"])
def test_pi_limits_prevent_windup(limit):
    sim = DozerSimulation(controller_type="pi", blade_roll_pitch_yaw=np.zeros(3))
    if limit == "rate":
        sim.blade_roll_pitch_yaw_rate_limits[:] = 0.
    else:
        sim.blade_roll_pitch_yaw_limits[:] = 0.
    for _ in range(3):
        sim._blade_update()
    np.testing.assert_array_equal(sim.controller.integral_error, 0.)


def test_pi_can_unwind_at_limit():
    controller = PIControl(1.2, .1, 1.)
    controller._previous_integral[1] = .5
    controller.integral_error[1] = .4
    controller.apply_actuator_feedback(np.array([0., .2, 0.]), np.array([0., .1, 0.]))
    assert controller.integral_error[1] == .4


def test_controller_selection_and_geometry_only_update():
    assert type(DozerSimulation().controller) is Control
    sim = DozerSimulation(controller_type="pi")
    assert type(sim.controller) is PIControl
    sim._blade_update(deform=False)
    np.testing.assert_array_equal(sim.controller.integral_error, 0.)
    with pytest.raises(ValueError):
        DozerSimulation(controller_type="unknown")
