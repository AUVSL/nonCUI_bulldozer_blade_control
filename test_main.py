"""Unit tests for BulldozerSimulation in main.py."""
import numpy as np
import pytest
from main import BulldozerSimulation
# to run: pytest test_main.py

# ───────────────── Fixtures ─────────────────
@pytest.fixture
def sim():
    return BulldozerSimulation()


# ───────────────── Static helpers ─────────────────
class TestSaturation:
    def test_within_limits(self):
        assert BulldozerSimulation.saturation(1.0, 5.0) == pytest.approx(1.0)

    def test_above_limit(self):
        assert BulldozerSimulation.saturation(10.0, 5.0) == pytest.approx(5.0)

    def test_below_negative_limit(self):
        assert BulldozerSimulation.saturation(-10.0, 5.0) == pytest.approx(-5.0)

    def test_exactly_at_limit(self):
        assert BulldozerSimulation.saturation(5.0, 5.0) == pytest.approx(5.0)

    def test_zero(self):
        assert BulldozerSimulation.saturation(0.0, 3.0) == pytest.approx(0.0)

    def test_negative_limit(self):
        assert BulldozerSimulation.saturation(2.0, -3.0) == pytest.approx(2.0)

    def test_returns_float(self):
        result = BulldozerSimulation.saturation(np.float64(2.0), 5.0)
        assert isinstance(result, float)


class TestWrapAngles:
    def test_zero(self):
        assert BulldozerSimulation.wrap_angles(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_pi_wraps_to_negative_pi(self):
        # (π + π) % (2π) - π = -π, numpy convention
        result = BulldozerSimulation.wrap_angles(np.array([np.pi]))
        assert abs(result[0]) == pytest.approx(np.pi)

    def test_three_pi_wraps(self):
        result = BulldozerSimulation.wrap_angles(np.array([3 * np.pi]))
        assert result[0] == pytest.approx(-np.pi, abs=1e-10) or result[0] == pytest.approx(np.pi, abs=1e-10)

    def test_already_in_range(self):
        angles = np.array([0.5, -0.5, 1.0])
        np.testing.assert_allclose(BulldozerSimulation.wrap_angles(angles), angles)

    def test_two_pi_wraps_to_zero(self):
        result = BulldozerSimulation.wrap_angles(np.array([2 * np.pi]))
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_array_multiple(self):
        angles = np.array([0.0, np.pi / 2, -np.pi / 2, 2.5 * np.pi])
        result = BulldozerSimulation.wrap_angles(angles)
        assert all(-np.pi <= r <= np.pi for r in result)


class TestGFunction:
    def test_positive_dx_returns_negative_f(self):
        assert BulldozerSimulation.G(5.0, 2.0, 1.0) == pytest.approx(-2.0)

    def test_negative_dx_returns_positive_f(self):
        assert BulldozerSimulation.G(5.0, 2.0, -1.0) == pytest.approx(2.0)

    def test_zero_dx_small_F_returns_negative_F(self):
        assert BulldozerSimulation.G(1.0, 2.0, 0.0) == pytest.approx(-1.0)

    def test_zero_dx_large_positive_F_returns_negative_f(self):
        assert BulldozerSimulation.G(5.0, 2.0, 0.0) == pytest.approx(-2.0)

    def test_zero_dx_large_negative_F_returns_positive_f(self):
        assert BulldozerSimulation.G(-5.0, 2.0, 0.0) == pytest.approx(2.0)

    def test_zero_dx_F_equals_f(self):
        # |F| == f → still returns -F (≤ condition)
        assert BulldozerSimulation.G(2.0, 2.0, 0.0) == pytest.approx(-2.0)

    def test_tiny_dx_treated_as_nonzero(self):
        # abs(dx) = 5e-10 > 1e-10 threshold → sliding branch
        assert BulldozerSimulation.G(5.0, 2.0, 5e-10) == pytest.approx(-2.0)

    def test_dx_at_threshold_treated_as_zero(self):
        # abs(dx) = 1e-10 is NOT > 1e-10, so stiction branch
        result = BulldozerSimulation.G(1.0, 2.0, 1e-10)
        assert result == pytest.approx(-1.0)


class TestYc:
    def test_both_zero_returns_zero(self):
        assert BulldozerSimulation.yc(0.0, 0.0, 2.0) == pytest.approx(0.0)

    def test_symmetric_trapezoid(self):
        # D1 == D2 → centroid at (2*D1 + D2) / (3*(D1+D2)) * B1 - B1/2
        D1, D2, B1 = 1.0, 1.0, 4.0
        assert BulldozerSimulation.yc(D1, D2, B1) == pytest.approx(0.0)

    def test_triangle_D1_right(self):
        D1, D2, B1 = 0.0, 2.0, 3.0
        assert BulldozerSimulation.yc(D1, D2, B1) < 0

    def test_triangle_D1_left(self):
        D1, D2, B1 = 2.0, 0.0, 3.0
        assert BulldozerSimulation.yc(D1, D2, B1) > 0

    def test_result_in_valid_range(self):
        # Centroid must be within [-B1/2, B1/2]
        B1 = 5.0
        result = BulldozerSimulation.yc(1.0, 3.0, B1)
        assert -B1 / 2 <= result <= B1 / 2


# ───────────────── Kinematics ─────────────────
class TestRotationMatrices:
    def test_rotation_gl_identity_at_zero(self, sim):
        R = sim.rotation_gl(0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_rotation_lg_identity_at_zero(self, sim):
        R = sim.rotation_lg(0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_rotation_lg_is_transpose_of_rotation_gl(self, sim):
        a, B, g = 0.2, 0.1, 0.3
        Rgl = sim.rotation_gl(a, B, g)
        Rlg = sim.rotation_lg(a, B, g)
        np.testing.assert_allclose(Rlg, Rgl.T, atol=1e-12)

    def test_rotation_gl_is_orthogonal(self, sim):
        a, B, g = 0.4, -0.2, 0.5
        R = sim.rotation_gl(a, B, g)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_rotation_lg_is_orthogonal(self, sim):
        a, B, g = -0.3, 0.15, -0.4
        R = sim.rotation_lg(a, B, g)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_rotation_gl_determinant_is_one(self, sim):
        a, B, g = 0.1, 0.2, 0.3
        R = sim.rotation_gl(a, B, g)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


class TestRotationDerivatives:
    def test_shapes(self, sim):
        J_gl, J_lg = sim.rotation_derivatives(0.0, 0.0)
        assert J_gl.shape == (3, 3)
        assert J_lg.shape == (3, 3)

    def test_zero_angles_J_gl(self, sim):
        J_gl, _ = sim.rotation_derivatives(0.0, 0.0)
        expected = np.array([
            [1, 0,  0],
            [0, 1,  0],
            [0, 0,  1]
        ])
        np.testing.assert_allclose(J_gl, expected, atol=1e-12)

    def test_zero_angles_J_lg(self, sim):
        _, J_lg = sim.rotation_derivatives(0.0, 0.0)
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        np.testing.assert_allclose(J_lg, expected, atol=1e-12)

    def test_J_gl_J_lg_are_inverses(self, sim):
        a, B = 0.2, 0.15
        J_gl, J_lg = sim.rotation_derivatives(a, B)
        np.testing.assert_allclose(J_gl @ J_lg, np.eye(3), atol=1e-10)

# ───────────────── S Matrices ─────────────────
class TestSMatrix:
    def test_shape(self, sim):
        S = sim.S_matrix()
        assert S.shape == (6, 2)

    def test_S_at_zero_angles(self, sim):
        S = sim.S_matrix()
        expected = np.zeros((6, 2))
        expected[0, 0] = 1.0
        expected[5, 1] = 1.0
        np.testing.assert_allclose(S, expected, atol=1e-12)
    
    def test_S_at_zero_angles_nonzero_x_icr(self, sim):
        sim.x_ICR = 1.0
        S = sim.S_matrix()
        expected = np.zeros((6, 2))
        expected[0, 0] = 1.0
        expected[1, 1] = -1.0  # due to x_ICR
        expected[5, 1] = 1.0
        np.testing.assert_allclose(S, expected, atol=1e-12)

    def test_S_in_nullspace(self, sim):
        # Equivalent to the matlab file test
        sim.x_ICR = 1.0
        a, B, g = -0.3, 0.15, -0.4
        sim.R_lg    = sim.rotation_lg(a, B, g)
        R_gl    = sim.rotation_gl(a, B, g)
        J_gl, sim.J_lg = sim.rotation_derivatives(a, B)
        A = np.array([R_gl[1, :],  sim.x_ICR * J_gl[2,:]]).flatten()
        S = sim.S_matrix()
        expected = np.zeros((1, 2))
        np.testing.assert_allclose(A@S, expected[0], atol=1e-12)

# ───────────────── ICR helper ─────────────────
class TestGetXIcr:
    def test_near_zero_yaw_rate_returns_zero(self, sim):
        sim.daBg = np.array([0.0, 0.0, 1e-4])  # below eps=1e-3
        result = sim.get_x_icr()
        assert result == pytest.approx(0.0)

    def test_computes_ratio(self, sim):
        sim.dxyz  = np.array([1.0, 0.5, 0.0])
        sim.daBg  = np.array([0.0, 0.0, 1.0])
        result = sim.get_x_icr()
        expected = np.clip(-0.5 / 1.0, -sim.l / 2, sim.l / 2)
        assert result == pytest.approx(float(expected))

    def test_clamped_to_half_length(self, sim):
        sim.dxyz = np.array([0.0, 100.0, 0.0])
        sim.daBg = np.array([0.0, 0.0, 0.01])
        result = sim.get_x_icr()
        assert abs(result) <= sim.l / 2 + 1e-10

# ───────────────── Blade terrain interaction ─────────────────
class TestBladeTerrainInteraction:
    def test_zero_angles_at_origin_no_force(self, sim):
        sim.bld_ang = np.zeros(3)
        sim.q[:3]   = np.zeros(3)
        sim.blade_terrain_interaction()
        assert sim.Fb == pytest.approx(0.0)
        assert sim.Mb == pytest.approx(0.0)

    def test_pitch_produces_negative_force(self, sim):
        sim.bld_ang = np.array([0.0, -0.1, 0.0])
        sim.q[:3]   = np.zeros(3)
        sim.blade_terrain_interaction()
        assert sim.Fb < 0

    def test_pure_pitch_zero_moment(self, sim):
        # Symmetric contact (no roll) → zero net moment
        sim.bld_ang = np.array([0.0, -0.1, 0.0])
        sim.q[:3]   = np.zeros(3)
        sim.blade_terrain_interaction()
        assert sim.Mb == pytest.approx(0.0, abs=1e-8)

    def test_positive_roll_positive_moment(self, sim):
        sim.bld_ang = np.array([-0.1, 0.0, 0.0])
        sim.q[:3]   = np.zeros(3)
        sim.blade_terrain_interaction()
        assert sim.Mb > 0

    def test_roll_moment_sign_flips_with_roll_sign(self, sim):
        sim.bld_ang = np.array([-0.1, 0.0, 0.0])
        sim.q[:3]   = np.zeros(3)
        sim.blade_terrain_interaction()
        Mb_pos = sim.Mb

        sim.bld_ang = np.array([0.1, 0.0, 0.0])
        sim.blade_terrain_interaction()
        Mb_neg = sim.Mb

        assert Mb_pos > 0
        assert Mb_neg < 0

    def test_Fb_always_nonpositive(self, sim):
        for roll, pitch in [(0.0, 0.0), (0.1, 0.0), (-0.1, 0.0), (0.0, -0.1), (0.1, -0.05)]:
            sim.bld_ang = np.array([roll, pitch, 0.0])
            sim.q[:3]   = np.zeros(3)
            sim.blade_terrain_interaction()
            assert sim.Fb <= 0.0

    def test_larger_pitch_larger_force(self, sim):
        sim.bld_ang = np.array([0.0, -0.05, 0.0])
        sim.q[:3]   = np.zeros(3)
        sim.blade_terrain_interaction()
        Fb_small = sim.Fb

        sim.bld_ang = np.array([0.0, -0.15, 0.0])
        sim.blade_terrain_interaction()
        Fb_large = sim.Fb

        assert Fb_large < Fb_small  # both ≤ 0; more pitch → more negative

# ───────────────── Track terrain interaction ─────────────────
class TestTrackTerrainInteraction:
    def test_symmetric_velocities_at_rest(self, sim):
        sim.dxyz  = np.zeros(3)
        sim.daBg  = np.zeros(3)
        sim.track_terrain_interaction()
        assert sim.vtL == pytest.approx(0.0)
        assert sim.vtR == pytest.approx(0.0)

    def test_track_velocities_saturated(self, sim):
        sim.dxyz  = np.array([100.0, 0.0, 0.0])
        sim.daBg  = np.zeros(3)
        sim.track_terrain_interaction()
        assert abs(sim.vtL) <= sim.velocity_limit + 1e-10
        assert abs(sim.vtR) <= sim.velocity_limit + 1e-10

    def test_Rl_shape(self, sim):
        sim.track_terrain_interaction()
        assert sim.Rl.shape == (2,)

    def test_Rl_nonpositive(self, sim):
        sim.dxyz  = np.array([100.0, 0.0, 0.0])
        sim.daBg  = np.zeros(3)
        sim.track_terrain_interaction()
        assert sim.Rl[0] <= 0
        assert sim.Rl[1] <= 0

    def test_Fy_and_Mr_zero_when_stationary(self, sim):
        sim.dxyz  = np.zeros(3)
        sim.daBg  = np.zeros(3)
        sim.track_terrain_interaction()
        assert sim.Fy == pytest.approx(0.0)
        assert sim.Mr == pytest.approx(0.0)

    def test_Fy_and_Mr_sign_when_turning(self, sim):
        sim.F_track  = np.array([sim.F_track_base, 0.0])
        sim.x_ICR    = -1.0  # nonzero ICR → turning → should produce lateral forces
        sim.dxyz      = np.array([1.0, -1.0, 0.0])
        sim.daBg      = np.array([0.0, 0.0, 1.0])
        sim.track_terrain_interaction()
        assert sim.Fy > 0.0
        assert sim.Mr < 0.0


# ───────────────── Sd Matrix ─────────────────
def _sd_from_formula(a, B, g, Ad, Bd, Gd, x_ICR, x_ICR_dot):
    """Direct transcription of the closed-form Sd expression.

    Variables map to the LaTeX notation as:
      A=alpha(a), B=beta(B), C=gamma(g);  dot → time derivative;
      x = x_ICR,  ẋ = x_ICR_dot
    """
    sa, ca = np.sin(a), np.cos(a)
    sB, cB = np.sin(B), np.cos(B)
    sg, cg = np.sin(g), np.cos(g)
    tB = np.tan(B)
    x  = x_ICR
    xd = x_ICR_dot

    # Column 1
    s11 = -sB * cg * Bd - cB * sg * Gd
    s21 =  cB * cg * Gd - sB * sg * Bd
    s31 = -cB * Bd

    # Column 2
    s12 = xd * (ca * sg - sa * sB * cg) + x * (
            -sa * cB * cg * Bd
            - (ca * sB * cg + sa * sg) * Ad
            + (sa * sB * sg + ca * cg) * Gd)
    s22 = x * (
            -sa * cB * sg * Bd
            + (ca * sg - sa * sB * cg) * Gd
            + (sa * cg - ca * sB * sg) * Ad) - xd * (sa * sB * sg + ca * cg)
    s32 = x * sa * sB * Bd - cB * (x * ca * Ad + xd * sa)

    s42 =  ca / cB**2 * Bd - sa * tB * Ad
    s52 = -ca * Ad
    s62 = (1 / cB) * (ca * tB * Bd - sa * Ad)

    return np.array([
        [s11, s12],
        [s21, s22],
        [s31, s32],
        [0.0, s42],
        [0.0, s52],
        [0.0, s62]
    ])


def _configure_sim_for_sd(sim, a, B, g, Ad, Bd, Gd, x_ICR, x_ICR_dot):
    sim.q[3:6]     = [a, B, g]
    sim.q_dot[3:6] = [Ad, Bd, Gd]
    sim.R_lg       = sim.rotation_lg(a, B, g)
    sim.x_ICR      = x_ICR
    sim.x_ICR_dot  = x_ICR_dot


class TestSdMatrix:
    def test_shape(self, sim):
        Sd = sim.Sd_matrix()
        assert Sd.shape == (6, 2)

    def test_matches_formula_at_zero(self, sim):
        Sd_sim     = sim.Sd_matrix()
        Sd_formula = _sd_from_formula(0, 0, 0, 0, 0, 0, 0, 0)
        np.testing.assert_allclose(Sd_sim, Sd_formula, atol=1e-12)

    def test_matches_formula_nonzero(self, sim):
        a, B, g         = 0.2, 0.15, -0.1
        Ad, Bd, Gd      = 0.3, -0.2, 0.1
        x, xd           = 0.5, 0.05
        _configure_sim_for_sd(sim, a, B, g, Ad, Bd, Gd, x, xd)
        Sd_sim     = sim.Sd_matrix()
        Sd_formula = _sd_from_formula(a, B, g, Ad, Bd, Gd, x, xd)
        np.testing.assert_allclose(Sd_sim, Sd_formula, atol=1e-12)

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_matches_formula_random(self, sim, seed):
        rng             = np.random.default_rng(seed)
        a, B, g         = rng.uniform(-0.3, 0.3, 3)
        Ad, Bd, Gd      = rng.uniform(-1.0, 1.0, 3)
        x               = float(rng.uniform(-1.0, 1.0))
        xd              = float(rng.uniform(-0.5, 0.5))
        _configure_sim_for_sd(sim, a, B, g, Ad, Bd, Gd, x, xd)
        Sd_sim     = sim.Sd_matrix()
        Sd_formula = _sd_from_formula(a, B, g, Ad, Bd, Gd, x, xd)
        np.testing.assert_allclose(Sd_sim, Sd_formula, atol=1e-12)



# ───────────────── Cross-track error ─────────────────
class TestSignedCrossTrackError:
    def test_on_path_returns_near_zero(self, sim):
        # Place vehicle exactly on a path point
        pos = sim.path_points[0].copy()
        err = sim.signed_cross_track_error(pos)
        assert abs(err) < 0.5  # within half a metre of path

    def test_symmetry(self, sim):
        # A displacement perpendicular to path should flip sign
        tangent_idx = len(sim.path_points) // 4
        tangent = sim.path_points[tangent_idx + 1] - sim.path_points[tangent_idx]
        tangent /= np.linalg.norm(tangent)
        n_surf = sim.rotation_lg(*sim.surface_abg)[:, 2]
        left_normal = np.cross(n_surf, tangent)
        left_normal /= np.linalg.norm(left_normal)
        base = sim.path_points[tangent_idx].copy()
        err_left  = sim.signed_cross_track_error(base + 0.5 * left_normal)
        err_right = sim.signed_cross_track_error(base - 0.5 * left_normal)
        assert err_left > 0
        assert err_right < 0


# ───────────────── Pure pursuit heading error ─────────────────
class TestPurePursuitHeadingError:
    def test_output_in_range(self, sim):
        err = sim.pure_pursuit_heading_error()
        assert -np.pi <= err <= np.pi

    def test_nearest_path_idx_in_bounds(self, sim):
        sim.pure_pursuit_heading_error()
        assert 0 <= sim._nearest_path_idx < len(sim.path_points)

    def test_facing_path_small_error(self, sim):
        # Vehicle on path facing the path tangent → small heading error
        idx = len(sim.path_points) // 4
        sim.q[:3] = sim.path_points[idx].copy()
        tangent   = sim.path_points[idx + 1] - sim.path_points[idx]
        sim.q[5]  = float(np.arctan2(tangent[1], tangent[0]))
        err = sim.pure_pursuit_heading_error()
        assert abs(err) < np.pi / 2

    def test_turned_right_of_path_positive_error(self, sim):
        # Vehicle rotated right of path tangent → lookahead is to the left → err > 0
        sim.q[:3] = sim.path_points[0].copy()
        tangent   = sim.path_points[1] - sim.path_points[0]
        path_hdg  = float(np.arctan2(tangent[1], tangent[0]))
        sim.q[5]  = path_hdg - np.pi / 4
        err = sim.pure_pursuit_heading_error()
        assert err > 0

    def test_turned_left_of_path_negative_error(self, sim):
        # Vehicle rotated left of path tangent → lookahead is to the right → err < 0
        sim.q[:3] = sim.path_points[0].copy()
        tangent   = sim.path_points[1] - sim.path_points[0]
        path_hdg  = float(np.arctan2(tangent[1], tangent[0]))
        sim.q[5]  = path_hdg + np.pi / 4
        err = sim.pure_pursuit_heading_error()
        assert err < 0


# ───────────────── Angular path controller ─────────────────
class TestAngularPathController:
    def test_forces_in_valid_range(self, sim):
        sim.angular_path_controller()
        assert 0.0 <= sim.F_track[0] <= sim.F_track_base
        assert 0.0 <= sim.F_track[1] <= sim.F_track_base

    def test_sets_finite_errors(self, sim):
        sim.angular_path_controller()
        assert np.isfinite(sim.heading_err)
        assert np.isfinite(sim.cross_track_err)

    def test_positive_heading_error_weakens_left_track(self, sim):
        # Turned right of path → heading_err > 0 → left track should be weakened
        sim.q[:3] = sim.path_points[0].copy()
        tangent   = sim.path_points[1] - sim.path_points[0]
        path_hdg  = float(np.arctan2(tangent[1], tangent[0]))
        sim.q[5]  = path_hdg - np.pi / 4
        sim.angular_path_controller()
        assert sim.heading_err > 0
        assert sim.F_track[0] < sim.F_track_base
        assert sim.F_track[1] == pytest.approx(sim.F_track_base)

    def test_negative_heading_error_weakens_right_track(self, sim):
        # Turned left of path → heading_err < 0 → right track should be weakened
        sim.q[:3] = sim.path_points[0].copy()
        tangent   = sim.path_points[1] - sim.path_points[0]
        path_hdg  = float(np.arctan2(tangent[1], tangent[0]))
        sim.q[5]  = path_hdg + np.pi / 4
        sim.angular_path_controller()
        assert sim.heading_err < 0
        assert sim.F_track[1] < sim.F_track_base
        assert sim.F_track[0] == pytest.approx(sim.F_track_base)


# ───────────────── Integration smoke test ─────────────────
class TestRun:
    def test_log_populated(self, sim):
        sim.stop_time = 0.05
        sim.run()
        assert len(sim.log) > 0

    def test_log_entry_length(self, sim):
        sim.stop_time = 0.05
        sim.run()
        assert len(sim.log[0]) == 21  # t + 6 q + cross_track + heading + Mb + Fb + RlL + RlR + Fy + Mr + v0 + v1 + bld_ang(3)

    def test_position_changes_when_running(self, sim):
        sim.stop_time = 0.2
        sim.run()
        log_arr = np.array(sim.log)
        # At least one position coordinate must have changed
        assert not np.allclose(log_arr[-1, 1:4], np.zeros(3))
