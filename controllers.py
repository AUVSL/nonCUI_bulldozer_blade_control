"""Controllers for blade and track commands. Add new controller classes here."""

import numpy as np


class Control():
    """PD or PI blade control, including actuator feedback for PI anti-windup."""

    def __init__(self, L, dt, zero_to_max_angle_time, controller_type="pd"):
        if controller_type not in ("pd", "pi"):
            raise ValueError("controller_type must be 'pd' or 'pi'")
        self.controller_type = controller_type
        # controller reference
        self.L = L
        self.desired_depth = 0.3
        self.desired_roll_pitch_yaw = np.array([ 0, 1, 0])
        
        # Blade proportional and derivative controller gains
        self.Kp = np.ones(3) * dt * zero_to_max_angle_time
        self.Kp[1] = 1.28  # tuned pitch gain, assigned directly (no dt scaling)
        self.dt = dt
        self.Kd = np.ones(3) * 0.1 * dt * zero_to_max_angle_time
        self.Kd[1] = 0.32  # tuned pitch derivative gain; zero gives P-only
        self.previous_blade_error = None
        self.derivative_error = np.zeros(3)  # angular error change (rad/s)
        self.Ki = np.zeros(3)
        if controller_type == "pi":
            self.Kp = np.array([0.01, 0.00025, 0.01])
            self.Ki = np.array([0.0, 0.0001, 0.0])
        self.integral_error = np.zeros(3)
        self._previous_integral = self.integral_error.copy()
        
    def blade_controller_errors(self, blade_roll_pitch_yaw):
        """
        Computes blade roll, pitch, yaw errors relative to desired surface and depth.
        """
        roll, pitch, yaw                                    = blade_roll_pitch_yaw
        desired_roll, desired_pitch_multiplier, desired_yaw = self.desired_roll_pitch_yaw

        # TODO: update blade angle limits from -1 to 1 to something more realistic, and update the test cases accordingly
        desired_pitch = desired_pitch_multiplier * np.arcsin(np.clip(self.desired_depth / self.L, -1.0, 1.0))

        errors   = np.array([desired_roll  - roll,       desired_pitch - pitch, desired_yaw - yaw])
        plot_out = np.array(            [errors[0], np.sin(errors[1]) * self.L,         errors[2]])

        return errors, plot_out
    
    def blade_deformation_errors(self, starting_surface_height, blade_bottom_height):
        """ Positive deformation is below the original surface; positive depth
        error requests more cut. The pitch gain maps metres to angle commands.
        """
        deformation   =  starting_surface_height - blade_bottom_height
        AAA           = self.desired_depth - deformation
        desired_pitch = np.arcsin(np.clip(AAA / self.L, -1.0, 1.0))
        errors        = np.array([0.0, desired_pitch, 0.0])
        return errors, errors.copy()

    def proportional_blade_controller(self, starting_surface_height, blade_bottom_height):
        """Return the selected PD or PI angle increment from angular depth error."""
        errors = self.blade_deformation_errors(starting_surface_height, blade_bottom_height)[0]
        if self.controller_type == "pi":
            self._previous_integral = self.integral_error.copy()
            self.integral_error += np.where(self.Ki != 0., errors * self.dt, 0.)
            return self.Kp * errors + self.Ki * self.integral_error
        # No previous sample exists on startup or after a reset.
        self.derivative_error = (np.zeros(3) if self.previous_blade_error is None
                                 else (errors - self.previous_blade_error) / self.dt)
        self.previous_blade_error = errors.copy()
        return self.Kp * errors + self.Kd * self.derivative_error

    def dummy_track_controlller(self, dozer_position_and_orientation):
        F_track_base = 600000.0
        return np.array([F_track_base, F_track_base])


    def apply_actuator_feedback(self, requested_increment, applied_increment):
        """Freeze integration into a limit, but permit unwinding out of it."""
        if self.controller_type != "pi":
            return
        blocked_increment = requested_increment - applied_increment
        integral_increment = self.Ki * (self.integral_error - self._previous_integral)
        blocked = (np.abs(blocked_increment) > 1e-12) & (blocked_increment * integral_increment > 0.)
        self.integral_error[blocked] = self._previous_integral[blocked]
