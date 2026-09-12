"""Derivative error timing, initialization, and actuator checks."""
import numpy as np
import pytest
from util import Control, DozerSimulation


def test_first_sample_and_reset_have_no_derivative_kick():
    controller = Control(1.2, 0.1, 1.0)
    error = controller.blade_deformation_errors(0., 0.)[0]
    np.testing.assert_allclose(controller.proportional_blade_controller(0., 0.), controller.Kp * error)
    np.testing.assert_array_equal(controller.derivative_error, 0.)
    controller.proportional_blade_controller(0., -0.1)
    controller.reset_derivative()
    np.testing.assert_allclose(controller.proportional_blade_controller(0., 0.), controller.Kp * error)
    np.testing.assert_array_equal(controller.derivative_error, 0.)


@pytest.mark.parametrize("dt", [0.01, 0.1])
def test_derivative_uses_error_difference_over_timestep(dt):
    controller = Control(1.2, dt, 1.0)
    previous = controller.blade_deformation_errors(0., 0.)[0]
    controller.proportional_blade_controller(0., 0.)
    current = controller.blade_deformation_errors(0., -0.1)[0]
    command = controller.proportional_blade_controller(0., -0.1)
    expected = (current - previous) / dt
    np.testing.assert_allclose(controller.derivative_error, expected)
    np.testing.assert_allclose(command, controller.Kp * current + controller.Kd * expected)
    assert command[1] < (controller.Kp * current)[1]
    controller.proportional_blade_controller(0., -0.1)
    np.testing.assert_array_equal(controller.derivative_error, 0.)


def test_zero_derivative_gain_preserves_proportional_output():
    controller = Control(1.2, 0.1, 1.0)
    controller.Kd[:] = 0.
    controller.proportional_blade_controller(0., 0.)
    command = controller.proportional_blade_controller(0., -0.1)
    np.testing.assert_allclose(command, controller.Kp * controller.blade_deformation_errors(0., -0.1)[0])


@pytest.mark.parametrize("limit", ["rate", "angle"])
def test_actuator_limits_still_apply(limit):
    sim = DozerSimulation(blade_roll_pitch_yaw=np.zeros(3))
    if limit == "rate":
        sim.blade_roll_pitch_yaw_rate_limits[:] = 0.
    else:
        sim.blade_roll_pitch_yaw_limits[:] = 0.
    sim._blade_update()
    sim.controller.desired_depth = 1.0
    sim._blade_update()
    np.testing.assert_array_equal(sim.blade_roll_pitch_yaw, 0.)


def test_geometry_refresh_does_not_update_derivative_history():
    sim = DozerSimulation(blade_roll_pitch_yaw=np.zeros(3))
    sim._blade_update(deform=False)
    assert sim.controller.previous_blade_error is None
    sim._blade_update()
    previous = sim.controller.previous_blade_error.copy()
    sim._blade_update(deform=False)
    np.testing.assert_array_equal(sim.controller.previous_blade_error, previous)
