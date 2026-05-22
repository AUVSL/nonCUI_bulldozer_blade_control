"""Unit tests for BulldozerSimulation in main.py."""
import numpy as np
import pytest
from main import BulldozerSimulation


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
        # D1 == D2 → centroid at (D1 + 2*D2) / (3*(D1+D2)) * B1 - B1/2
        D1, D2, B1 = 1.0, 1.0, 4.0
        assert BulldozerSimulation.yc(D1, D2, B1) == pytest.approx(0.0)

    def test_triangle_D1_zero(self):
        D1, D2, B1 = 0.0, 2.0, 3.0
        expected = (0 + 2 * D2) / (3 * (0 + D2)) * B1 - B1 / 2
        assert BulldozerSimulation.yc(D1, D2, B1) == pytest.approx(expected)

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


# ───────────────── Integration smoke test ─────────────────

class TestRun:
    def test_log_populated(self, sim):
        sim.stop_time = 0.05
        sim.run()
        assert len(sim.log) > 0

    def test_log_entry_length(self, sim):
        sim.stop_time = 0.05
        sim.run()
        assert len(sim.log[0]) == 17  # t + 6 q + cross_track + heading + Mb + Fb + RlL + RlR + Fy + Mr + v0 + v1

    def test_position_changes_when_running(self, sim):
        sim.stop_time = 0.2
        sim.run()
        log_arr = np.array(sim.log)
        # At least one position coordinate must have changed
        assert not np.allclose(log_arr[-1, 1:4], np.zeros(3))

    def test_backward_velocity_is_nonpositive(self, sim):
        sim.backward   = True
        sim.stop_time  = 0.2
        sim.run()
        log_arr = np.array(sim.log)
        assert (log_arr[:, 15] <= 1e-10).all()
