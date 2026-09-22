"""Controllers for blade and track commands. Add new controller classes here."""
from __future__ import annotations
from typing import cast, Optional
from numpy.typing import ArrayLike
from sim_types import FloatArray, IntArray, Scalar, VectorLike


import numpy as np


class Control():
    """PD or PI blade control, including actuator feedback for PI anti-windup."""

    def __init__(self, L: float, dt: float, zero_to_max_angle_time: float, controller_type: str="pd") -> None:
        if controller_type not in ("pd", "pi"):
            raise ValueError("controller_type must be 'pd' or 'pi'")
        self.controller_type: str = controller_type
        # controller reference
        self.L: float = L
        self.desired_depth: float = 0.3
        self.desired_roll_pitch_yaw: IntArray = np.array([ 0, 1, 0])

        # Blade proportional and derivative controller gains
        self.Kp: FloatArray = np.ones(3) * dt * zero_to_max_angle_time
        self.Kp[1] = 1.28  # tuned pitch gain, assigned directly (no dt scaling)
        self.dt: float = dt
        self.Kd: FloatArray = np.ones(3) * 0.1 * dt * zero_to_max_angle_time
        self.Kd[1] = 0.32  # tuned pitch derivative gain; zero gives P-only
        self.previous_blade_error: Optional[FloatArray] = None
        self.derivative_error: FloatArray = np.zeros(3)  # angular error change (rad/s)
        self.Ki: FloatArray = np.zeros(3)
        if controller_type == "pi":
            self.Kp = np.array([0.01, 0.00025, 0.01])
            self.Ki = np.array([0.0, 0.0001, 0.0])
        self.integral_error: FloatArray = np.zeros(3)
        self._previous_integral: FloatArray = self.integral_error.copy()
        # Offline 4-segment piecewise-linear fit copied from main.py.
        self._4pl_coeffs: FloatArray = np.array([0.1499756, 1.87300871, 3.47736035, 5.59836133])
        self._4pl_knots: tuple[float, float, float] = (1.0593712690788024, 1.1706408294732875, 1.2077716115263897)
        self._4pl_threshold: float = 0.799
        self._4pl_ang_max: float = 1.2256907107093555
        self.configure_path()

    def blade_controller_errors(self, blade_roll_pitch_yaw: VectorLike) -> tuple[FloatArray, FloatArray]:
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

    def blade_deformation_errors(self, starting_surface_height: Scalar, blade_bottom_height: Scalar) -> tuple[FloatArray, FloatArray]:
        """ Positive deformation is below the original surface; positive depth
        error requests more cut. The pitch gain maps metres to angle commands.
        """
        deformation   =  starting_surface_height - blade_bottom_height
        AAA           = self.desired_depth - deformation
        desired_pitch = np.arcsin(np.clip(AAA / self.L, -1.0, 1.0))
        errors        = np.array([0.0, desired_pitch, 0.0])
        return errors, errors.copy()

    def proportional_blade_controller(self, starting_surface_height: Scalar, blade_bottom_height: Scalar) -> FloatArray:
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

    def dummy_track_controlller(self, dozer_position_and_orientation: VectorLike) -> FloatArray:
        F_track_base = 600000.0
        return np.array([F_track_base, F_track_base])


    def apply_actuator_feedback(self, requested_increment: FloatArray, applied_increment: FloatArray) -> None:
        """Freeze integration into a limit, but permit unwinding out of it."""
        if self.controller_type != "pi":
            return
        blocked_increment = requested_increment - applied_increment
        integral_increment = self.Ki * (self.integral_error - self._previous_integral)
        blocked = (np.abs(blocked_increment) > 1e-12) & (blocked_increment * integral_increment > 0.)
        self.integral_error[blocked] = self._previous_integral[blocked]


    def configure_path(self, path_points: Optional[ArrayLike]=None, surface_rotation: Optional[ArrayLike]=None, lookahead_dist: float=1.5) -> None:
        """Set a closed reference path (Nx2 or Nx3), plane, and lookahead in metres."""
        if not np.isfinite(lookahead_dist) or lookahead_dist <= 0:
            raise ValueError("lookahead_dist must be finite and positive")
        rotation = np.eye(3) if surface_rotation is None else np.asarray(surface_rotation, dtype=float)
        if (rotation.shape != (3, 3) or not np.all(np.isfinite(rotation))
                or not np.allclose(rotation.T @ rotation, np.eye(3))
                or not np.isclose(np.linalg.det(rotation), 1.0)):
            raise ValueError("surface_rotation must be a proper 3x3 rotation matrix")
        points = self.figure8_path(surface_rotation=rotation) if path_points is None else np.asarray(path_points, dtype=float)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] not in (2, 3) or not np.all(np.isfinite(points)):
            raise ValueError("path_points must contain at least three finite 2D or 3D waypoints")
        if points.shape[1] == 2:
            points = np.column_stack((points, np.zeros(len(points))))
        if len(np.unique(points, axis=0)) < 3:
            raise ValueError("path_points must contain at least three distinct waypoints")
        self.surface_rotation: FloatArray = rotation.copy()
        self.path_points: FloatArray = points.copy()
        self._lookahead_dist: float = float(lookahead_dist)
        self._nearest_path_idx = self._max_path_idx = 0
        self.path_complete: bool = False
        self.heading_err = self.cross_track_err = 0.0
        self.lookahead_point: FloatArray = self.path_points[0].copy()

    def track_force_fraction(self, heading_error: Scalar) -> float:
        """Map heading-error magnitude to the reduced track's force fraction.

        This is the original yaw-angle calibration, not a yaw-rate setpoint.
        Vehicle dynamics determine angular velocity from the resulting forces.
        """
        if not np.isfinite(heading_error):
            raise ValueError("heading_error must be finite")
        angle = min(abs(float(heading_error)), self._4pl_ang_max)
        return float(np.clip(self._eval_4pl(angle), 0.0, 1.0))

    @staticmethod
    def figure8_path(A: float=5.0, B: float=2.5, n_points: int=2000, surface_rotation: Optional[ArrayLike]=None) -> FloatArray:
        """Dense (x, y, z) waypoints for a figure-8 on the surface plane defined by surface_abg."""
        R_surf = np.eye(3) if surface_rotation is None else np.asarray(surface_rotation, dtype=float)
        e1, e2 = R_surf[:, 0], R_surf[:, 1]
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        xs, ys = A * np.sin(t), B * np.sin(2 * t)
        return cast(FloatArray, np.outer(xs, e1) + np.outer(ys, e2))

    def signed_cross_track_error(self, pos_xyz: VectorLike) -> float:
        """Signed perpendicular distance from pos_xyz to self.path_points on the surface plane.
        Positive when the vehicle is to the left of the path tangent direction."""
        position = np.asarray(pos_xyz, dtype=float)
        diffs = self.path_points - position
        idx      = int(np.argmin(np.linalg.norm(diffs, axis=1)))
        next_idx = (idx + 1) % len(self.path_points)
        tangent  = self.path_points[next_idx] - self.path_points[idx]
        norm     = np.linalg.norm(tangent)
        if norm < 1e-10:
            return 0.0
        tangent /= norm
        n_surf      = self.surface_rotation[:, 2]
        left_normal = np.cross(n_surf, tangent)  # left of tangent within surface plane
        return float(np.dot(position - self.path_points[idx], left_normal))

    def _eval_4pl(self, ang: Scalar) -> Scalar:
        k1, k2, k3 = self._4pl_knots
        c = self._4pl_coeffs
        T = c[0]*ang + c[1]*max(ang-k1, 0) + c[2]*max(ang-k2, 0) + c[3]*max(ang-k3, 0)
        return self._4pl_threshold - T

    def pure_pursuit_heading_error(self, pose: VectorLike, body_rotation: FloatArray) -> float:
        """
        Pure-pursuit: find the lookahead point on the path at distance
        self._lookahead_dist from the vehicle, return the signed angle
        from current heading to that point, measured within the surface plane.
        """
        pos = np.asarray(pose)[:3]
        L   = self._lookahead_dist
        n   = len(self.path_points)

        dists      = np.linalg.norm(self.path_points - pos, axis=1)
        nearest    = int(np.argmin(dists))
        self._nearest_path_idx = nearest
        lookahead_pt = None

        # Walk forward along path segments looking for sphere intersection
        for k in range(n):
            i   = (nearest + k) % n
            j   = (i + 1) % n
            p1  = self.path_points[i]
            p2  = self.path_points[j]
            d   = p2 - p1
            f   = p1 - pos
            a   = float(np.dot(d, d))
            if a < 1e-20:  # Repeated waypoint: skip a zero-length segment.
                continue
            b   = 2.0 * float(np.dot(f, d))
            c   = float(np.dot(f, f)) - L * L
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            t2 = (-b + np.sqrt(disc)) / (2 * a)   # forward intersection
            if 0.0 <= t2 <= 1.0:
                lookahead_pt = p1 + t2 * d
                break

        if lookahead_pt is None:                    # fallback: nearest point
            lookahead_pt = self.path_points[nearest]

        self.lookahead_point = lookahead_pt.copy()
        R_surf  = self.surface_rotation
        n_surf  = R_surf[:, 2]
        e1_surf = R_surf[:, 0]
        e2_surf = R_surf[:, 1]
        d_vec   = lookahead_pt - pos
        d_proj  = d_vec - np.dot(d_vec, n_surf) * n_surf   # project onto surface plane
        angle   = np.arctan2(np.dot(d_proj, e2_surf), np.dot(d_proj, e1_surf))
        fwd     = np.asarray(body_rotation)[:, 0]
        heading = np.arctan2(np.dot(fwd, e2_surf), np.dot(fwd, e1_surf))
        err     = angle - heading
        return float((err + np.pi) % (2 * np.pi) - np.pi)

    def angular_path_controller(self, pose: VectorLike, body_rotation: FloatArray, force_base: float=600000.0) -> FloatArray:
        """Assign track forces via 4PL lookup on pure-pursuit heading error."""
        self.heading_err = self.pure_pursuit_heading_error(pose, body_rotation)
        self.cross_track_err = self.signed_cross_track_error(np.asarray(pose)[:3])
        self._max_path_idx = max(self._max_path_idx, self._nearest_path_idx)
        self.path_complete = (self._max_path_idx > len(self.path_points) * 0.9
                              and self._nearest_path_idx < len(self.path_points) * 0.1)
        fraction = self.track_force_fraction(self.heading_err)
        forces = np.full(2, float(force_base))
        if self.heading_err > 0:          # need to turn left  → weaken left track
            forces[0] = fraction * force_base
            forces[1] = force_base
        else:                              # need to turn right → weaken right track
            forces[0] = force_base
            forces[1] = fraction * force_base
        return forces
