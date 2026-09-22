"""Unit tests for the bulldozer simulation, controllers, terrain, and visualization."""
from __future__ import annotations
from typing import Any, NoReturn, Sequence
from numpy.typing import ArrayLike
from sim_types import ContactMap, FloatArray, GridCell, Point3, Scalar
from pathlib import Path
from matplotlib.artist import Artist

import csv

import matplotlib
matplotlib.use("Agg")
import networkx as nx
import numpy as np
import pytest
from PIL import Image

import tuning
import main as simulation_module
import visualization as visualization_module
from controllers import Control
from main import DozerSimulation
from visualization import Visualization
# to run: pytest test_main.py

# ───────────────── Fixtures ─────────────────
@pytest.fixture
def sim() -> DozerSimulation:
    simulation = DozerSimulation(enable_blade_control=False)
    simulation.q[:] = 0.0
    simulation.R_lg = simulation._rotation_lg(*simulation.q[3:6])
    _, simulation.J_lg = simulation._rotation_derivatives()
    return simulation


# ───────────────── Static helpers ─────────────────
class TestSaturation:
    def test_within_limits(self, sim: DozerSimulation) -> None:
        assert sim._saturation(1.0, 5.0) == pytest.approx(1.0)

    def test_above_limit(self, sim: DozerSimulation) -> None:
        assert sim._saturation(10.0, 5.0) == pytest.approx(5.0)

    def test_below_negative_limit(self, sim: DozerSimulation) -> None:
        assert sim._saturation(-10.0, 5.0) == pytest.approx(-5.0)

    def test_exactly_at_limit(self, sim: DozerSimulation) -> None:
        assert sim._saturation(5.0, 5.0) == pytest.approx(5.0)

    def test_zero(self, sim: DozerSimulation) -> None:
        assert sim._saturation(0.0, 3.0) == pytest.approx(0.0)

    def test_negative_limit(self, sim: DozerSimulation) -> None:
        assert sim._saturation(2.0, -3.0) == pytest.approx(2.0)

    def test_returns_float(self, sim: DozerSimulation) -> None:
        result = sim._saturation(np.float64(2.0), 5.0)
        assert isinstance(result, float)


@pytest.mark.parametrize("yaw, expected", [
    (0.0, 0.0), (np.pi, -np.pi), (3 * np.pi, -np.pi),
    (0.5, 0.5), (2 * np.pi, 0.0), (2.5 * np.pi, np.pi / 2),
])
def test_body_update_wraps_yaw(sim: DozerSimulation, monkeypatch: pytest.MonkeyPatch, yaw: float, expected: float) -> None:
    sim.q[5] = yaw
    monkeypatch.setattr(sim, "_update_q_dot", lambda: None)
    monkeypatch.setattr(sim, "_settle_tracks", lambda orientation: None)
    sim._body_update()
    assert sim.q[5] == pytest.approx(expected, abs=1e-12)


class TestGFunction:
    def test_positive_dx_returns_negative_f(self, sim: DozerSimulation) -> None:
        assert sim._G(5.0, 2.0, 1.0) == pytest.approx(-2.0)

    def test_negative_dx_returns_positive_f(self, sim: DozerSimulation) -> None:
        assert sim._G(5.0, 2.0, -1.0) == pytest.approx(2.0)

    def test_zero_dx_small_F_returns_negative_F(self, sim: DozerSimulation) -> None:
        assert sim._G(1.0, 2.0, 0.0) == pytest.approx(-1.0)

    def test_zero_dx_large_positive_F_returns_negative_f(self, sim: DozerSimulation) -> None:
        assert sim._G(5.0, 2.0, 0.0) == pytest.approx(-2.0)

    def test_zero_dx_large_negative_F_returns_positive_f(self, sim: DozerSimulation) -> None:
        assert sim._G(-5.0, 2.0, 0.0) == pytest.approx(2.0)

    def test_zero_dx_F_equals_f(self, sim: DozerSimulation) -> None:
        # |F| == f → still returns -F (≤ condition)
        assert sim._G(2.0, 2.0, 0.0) == pytest.approx(-2.0)

    def test_tiny_dx_treated_as_nonzero(self, sim: DozerSimulation) -> None:
        # abs(dx) = 5e-10 > 1e-10 threshold → sliding branch
        assert sim._G(5.0, 2.0, 5e-10) == pytest.approx(-2.0)

    def test_dx_at_threshold_treated_as_zero(self, sim: DozerSimulation) -> None:
        # abs(dx) = 1e-10 is NOT > 1e-10, so stiction branch
        result = sim._G(1.0, 2.0, 1e-10)
        assert result == pytest.approx(-1.0)


class TestYc:
    def test_both_zero_returns_zero(self, sim: DozerSimulation) -> None:
        assert sim._yc(0.0, 0.0, 2.0) == pytest.approx(0.0)

    def test_symmetric_trapezoid(self, sim: DozerSimulation) -> None:
        # D1 == D2 → centroid at (2*D1 + D2) / (3*(D1+D2)) * B1 - B1/2
        D1, D2, B1 = 1.0, 1.0, 4.0
        assert sim._yc(D1, D2, B1) == pytest.approx(0.0)

    def test_triangle_D1_right(self, sim: DozerSimulation) -> None:
        D1, D2, B1 = 0.0, 2.0, 3.0
        assert sim._yc(D1, D2, B1) < 0

    def test_triangle_D1_left(self, sim: DozerSimulation) -> None:
        D1, D2, B1 = 2.0, 0.0, 3.0
        assert sim._yc(D1, D2, B1) > 0

    def test_result_in_valid_range(self, sim: DozerSimulation) -> None:
        # Centroid must be within [-B1/2, B1/2]
        B1 = 5.0
        result = sim._yc(1.0, 3.0, B1)
        assert -B1 / 2 <= result <= B1 / 2


# ───────────────── Kinematics ─────────────────
class TestRotationMatrices:
    def test_rotation_gl_identity_at_zero(self, sim: DozerSimulation) -> None:
        R = sim._rotation_gl(0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_rotation_lg_identity_at_zero(self, sim: DozerSimulation) -> None:
        R = sim._rotation_lg(0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_rotation_lg_is_transpose_of_rotation_gl(self, sim: DozerSimulation) -> None:
        a, B, g = 0.2, 0.1, 0.3
        Rgl = sim._rotation_gl(a, B, g)
        Rlg = sim._rotation_lg(a, B, g)
        np.testing.assert_allclose(Rlg, Rgl.T, atol=1e-12)

    def test_rotation_gl_is_orthogonal(self, sim: DozerSimulation) -> None:
        a, B, g = 0.4, -0.2, 0.5
        R = sim._rotation_gl(a, B, g)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_rotation_lg_is_orthogonal(self, sim: DozerSimulation) -> None:
        a, B, g = -0.3, 0.15, -0.4
        R = sim._rotation_lg(a, B, g)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_rotation_gl_determinant_is_one(self, sim: DozerSimulation) -> None:
        a, B, g = 0.1, 0.2, 0.3
        R = sim._rotation_gl(a, B, g)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


class TestRotationDerivatives:
    def test_shapes(self, sim: DozerSimulation) -> None:
        J_gl, J_lg = sim._rotation_derivatives()
        assert J_gl.shape == (3, 3)
        assert J_lg.shape == (3, 3)

    def test_zero_angles_J_gl(self, sim: DozerSimulation) -> None:
        J_gl, _ = sim._rotation_derivatives()
        expected = np.array([
            [1, 0,  0],
            [0, 1,  0],
            [0, 0,  1]
        ])
        np.testing.assert_allclose(J_gl, expected, atol=1e-12)

    def test_zero_angles_J_lg(self, sim: DozerSimulation) -> None:
        _, J_lg = sim._rotation_derivatives()
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        np.testing.assert_allclose(J_lg, expected, atol=1e-12)

    def test_J_gl_J_lg_are_inverses(self, sim: DozerSimulation) -> None:
        a, B = 0.2, 0.15
        sim.q[3:5] = [a, B]
        J_gl, J_lg = sim._rotation_derivatives()
        np.testing.assert_allclose(J_gl @ J_lg, np.eye(3), atol=1e-10)

# ───────────────── S Matrices ─────────────────
class TestSMatrix:
    def test_shape(self, sim: DozerSimulation) -> None:
        S = sim._S_matrix()
        assert S.shape == (6, 2)

    def test_S_at_zero_angles(self, sim: DozerSimulation) -> None:
        S = sim._S_matrix()
        expected = np.zeros((6, 2))
        expected[0, 0] = 1.0
        expected[5, 1] = 1.0
        np.testing.assert_allclose(S, expected, atol=1e-12)

    def test_S_at_zero_angles_nonzero_x_icr(self, sim: DozerSimulation) -> None:
        sim.x_ICR = 1.0
        S = sim._S_matrix()
        expected = np.zeros((6, 2))
        expected[0, 0] = 1.0
        expected[1, 1] = -1.0  # due to x_ICR
        expected[5, 1] = 1.0
        np.testing.assert_allclose(S, expected, atol=1e-12)

    def test_S_in_nullspace(self, sim: DozerSimulation) -> None:
        # Equivalent to the matlab file test
        sim.x_ICR = 1.0
        a, B, g = -0.3, 0.15, -0.4
        sim.R_lg    = sim._rotation_lg(a, B, g)
        R_gl    = sim._rotation_gl(a, B, g)
        sim.q[3:5] = [a, B]
        J_gl, sim.J_lg = sim._rotation_derivatives()
        A = np.array([R_gl[1, :],  sim.x_ICR * J_gl[2,:]]).flatten()
        S = sim._S_matrix()
        expected = np.zeros((1, 2))
        np.testing.assert_allclose(A@S, expected[0], atol=1e-12)

# ───────────────── ICR helper ─────────────────
class TestGetXIcr:
    def test_near_zero_yaw_rate_returns_zero(self, sim: DozerSimulation) -> None:
        sim.daBg = np.array([0.0, 0.0, 1e-4])  # below eps=1e-3
        result = sim._get_x_icr()
        assert result == pytest.approx(0.0)

    def test_computes_ratio(self, sim: DozerSimulation) -> None:
        sim.dxyz  = np.array([1.0, 0.5, 0.0])
        sim.daBg  = np.array([0.0, 0.0, 1.0])
        result = sim._get_x_icr()
        expected = np.clip(-0.5 / 1.0, -sim.l / 2, sim.l / 2)
        assert result == pytest.approx(float(expected))

    def test_clamped_to_half_length(self, sim: DozerSimulation) -> None:
        sim.dxyz = np.array([0.0, 100.0, 0.0])
        sim.daBg = np.array([0.0, 0.0, 0.01])
        result = sim._get_x_icr()
        assert abs(result) <= sim.l / 2 + 1e-10

# ───────────────── Blade terrain interaction ─────────────────
class TestBladeTerrainInteraction:
    def test_zero_angles_at_origin_no_force(self, sim: DozerSimulation) -> None:
        sim.blade_roll_pitch_yaw = np.zeros(3)
        sim.q[:3]   = np.zeros(3)
        sim._blade_terrain_interaction()
        assert sim.Fb == pytest.approx(0.0)
        assert sim.Mb == pytest.approx(0.0)

    def test_pitch_produces_negative_force(self, sim: DozerSimulation) -> None:
        sim.blade_roll_pitch_yaw = np.array([0.0, -0.1, 0.0])
        sim.q[:3]   = np.zeros(3)
        sim._blade_terrain_interaction()
        assert sim.Fb < 0

    def test_pure_pitch_zero_moment(self, sim: DozerSimulation) -> None:
        # Symmetric contact (no roll) → zero net moment
        sim.blade_roll_pitch_yaw = np.array([0.0, -0.1, 0.0])
        sim.q[:3]   = np.zeros(3)
        sim._blade_terrain_interaction()
        assert sim.Mb == pytest.approx(0.0, abs=1e-8)

    def test_positive_roll_positive_moment(self, sim: DozerSimulation) -> None:
        sim.blade_roll_pitch_yaw = np.array([-0.1, 0.0, 0.0])
        sim.q[:3]   = np.zeros(3)
        sim._blade_terrain_interaction()
        assert sim.Mb > 0

    def test_roll_moment_sign_flips_with_roll_sign(self, sim: DozerSimulation) -> None:
        sim.blade_roll_pitch_yaw = np.array([-0.1, 0.0, 0.0])
        sim.q[:3]   = np.zeros(3)
        sim._blade_terrain_interaction()
        Mb_pos = sim.Mb

        sim.blade_roll_pitch_yaw = np.array([0.1, 0.0, 0.0])
        sim._blade_terrain_interaction()
        Mb_neg = sim.Mb

        assert Mb_pos > 0
        assert Mb_neg < 0

    def test_Fb_always_nonpositive(self, sim: DozerSimulation) -> None:
        for roll, pitch in [(0.0, 0.0), (0.1, 0.0), (-0.1, 0.0), (0.0, -0.1), (0.1, -0.05)]:
            sim.blade_roll_pitch_yaw = np.array([roll, pitch, 0.0])
            sim.q[:3]   = np.zeros(3)
            sim._blade_terrain_interaction()
            assert sim.Fb <= 0.0

    def test_larger_pitch_larger_force(self, sim: DozerSimulation) -> None:
        sim.blade_roll_pitch_yaw = np.array([0.0, -0.05, 0.0])
        sim.q[:3]   = np.zeros(3)
        sim._blade_terrain_interaction()
        Fb_small = sim.Fb

        sim.blade_roll_pitch_yaw = np.array([0.0, -0.15, 0.0])
        sim._blade_terrain_interaction()
        Fb_large = sim.Fb

        assert Fb_large < Fb_small  # both ≤ 0; more pitch → more negative

# ───────────────── Track terrain interaction ─────────────────
class TestTrackTerrainInteraction:
    def test_symmetric_velocities_at_rest(self, sim: DozerSimulation) -> None:
        sim.dxyz  = np.zeros(3)
        sim.daBg  = np.zeros(3)
        sim._track_terrain_interaction()
        assert sim.vtL == pytest.approx(0.0)
        assert sim.vtR == pytest.approx(0.0)

    def test_track_velocities_saturated(self, sim: DozerSimulation) -> None:
        sim.dxyz  = np.array([100.0, 0.0, 0.0])
        sim.daBg  = np.zeros(3)
        sim._track_terrain_interaction()
        assert abs(sim.vtL) <= sim.velocity_limit + 1e-10
        assert abs(sim.vtR) <= sim.velocity_limit + 1e-10

    def test_Rl_shape(self, sim: DozerSimulation) -> None:
        sim._track_terrain_interaction()
        assert sim.Rl.shape == (2,)

    def test_Rl_nonpositive(self, sim: DozerSimulation) -> None:
        sim.dxyz  = np.array([100.0, 0.0, 0.0])
        sim.daBg  = np.zeros(3)
        sim._track_terrain_interaction()
        assert sim.Rl[0] <= 0
        assert sim.Rl[1] <= 0

    def test_Fy_and_Mr_zero_when_stationary(self, sim: DozerSimulation) -> None:
        sim.dxyz  = np.zeros(3)
        sim.daBg  = np.zeros(3)
        sim._track_terrain_interaction()
        assert sim.Fy == pytest.approx(0.0)
        assert sim.Mr == pytest.approx(0.0)

    def test_Fy_and_Mr_sign_when_turning(self, sim: DozerSimulation) -> None:
        sim.F_track  = np.array([sim.F_track_base, 0.0])
        sim.x_ICR    = -1.0  # nonzero ICR → turning → should produce lateral forces
        sim.dxyz      = np.array([1.0, -1.0, 0.0])
        sim.daBg      = np.array([0.0, 0.0, 1.0])
        sim._track_terrain_interaction()
        assert sim.Fy > 0.0
        assert sim.Mr < 0.0


# ───────────────── Sd Matrix ─────────────────
def _sd_from_formula(a: float, B: float, g: float, Ad: float, Bd: float, Gd: float, x_ICR: float, x_ICR_dot: float) -> FloatArray:
    """Direct transcription of the closed-form Sd expression.
    (Output of wolfram file s_derivative.nb)
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


def _configure_sim_for_sd(sim: DozerSimulation, a: float, B: float, g: float, Ad: float, Bd: float, Gd: float, x_ICR: float, x_ICR_dot: float) -> None:
    sim.q[3:6]     = [a, B, g]
    sim.q_dot[3:6] = [Ad, Bd, Gd]
    sim.R_lg       = sim._rotation_lg(a, B, g)
    sim.x_ICR      = x_ICR
    sim.x_ICR_dot  = x_ICR_dot


class TestSdMatrix:
    def test_shape(self, sim: DozerSimulation) -> None:
        Sd = sim._Sd_matrix()
        assert Sd.shape == (6, 2)

    def test_matches_formula_at_zero(self, sim: DozerSimulation) -> None:
        Sd_sim     = sim._Sd_matrix()
        Sd_formula = _sd_from_formula(0, 0, 0, 0, 0, 0, 0, 0)
        np.testing.assert_allclose(Sd_sim, Sd_formula, atol=1e-12)

    def test_matches_formula_nonzero(self, sim: DozerSimulation) -> None:
        a, B, g         = 0.2, 0.15, -0.1
        Ad, Bd, Gd      = 0.3, -0.2, 0.1
        x, xd           = 0.5, 0.05
        _configure_sim_for_sd(sim, a, B, g, Ad, Bd, Gd, x, xd)
        Sd_sim     = sim._Sd_matrix()
        Sd_formula = _sd_from_formula(a, B, g, Ad, Bd, Gd, x, xd)
        np.testing.assert_allclose(Sd_sim, Sd_formula, atol=1e-12)

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_matches_formula_random(self, sim: DozerSimulation, seed: int) -> None:
        rng             = np.random.default_rng(seed)
        a, B, g         = rng.uniform(-0.3, 0.3, 3)
        Ad, Bd, Gd      = rng.uniform(-1.0, 1.0, 3)
        x               = float(rng.uniform(-1.0, 1.0))
        xd              = float(rng.uniform(-0.5, 0.5))
        _configure_sim_for_sd(sim, a, B, g, Ad, Bd, Gd, x, xd)
        Sd_sim     = sim._Sd_matrix()
        Sd_formula = _sd_from_formula(a, B, g, Ad, Bd, Gd, x, xd)
        np.testing.assert_allclose(Sd_sim, Sd_formula, atol=1e-12)


    @pytest.mark.parametrize("icr_rate", [-0.2, 0.0, 0.2])
    def test_matches_numerical_derivative_of_s(self, sim: DozerSimulation, icr_rate: float) -> None:
        angles = np.array([0.2, 0.15, -0.1])
        angle_rates = np.array([0.3, -0.2, 0.1])
        icr = 0.5
        step = 1e-6

        def mapping_at(time: float) -> FloatArray:
            sim.q[3:6] = angles + time * angle_rates
            sim.R_lg = sim._rotation_lg(*sim.q[3:6])
            _, sim.J_lg = sim._rotation_derivatives()
            sim.x_ICR = icr + time * icr_rate
            return sim._S_matrix()

        numerical = (mapping_at(step) - mapping_at(-step)) / (2 * step)
        mapping_at(0.0)
        sim.q_dot[3:6] = angle_rates
        sim.x_ICR_dot = icr_rate
        np.testing.assert_allclose(sim._Sd_matrix(), numerical, rtol=1e-7, atol=1e-9)


# ───────────────── Vehicle Dynamics ─────────────────
class TestVehicleDynamics:
    def test_shape(self, sim: DozerSimulation) -> None:
        sim._vehicle_dynamics()
        assert sim.v_dot.shape == (2,)
        assert np.all(np.isfinite(sim.v_dot))

# ───────────────── Cross-track error ─────────────────
class TestSignedCrossTrackError:
    def test_on_path_returns_near_zero(self, sim: DozerSimulation) -> None:
        # Place vehicle exactly on a path point
        pos = sim.controller.path_points[0].copy()
        err = sim.controller.signed_cross_track_error(pos)
        assert abs(err) < 0.5  # within half a metre of path

    def test_symmetry(self, sim: DozerSimulation) -> None:
        # A displacement perpendicular to path should flip sign
        tangent_idx = len(sim.controller.path_points) // 4
        tangent = sim.controller.path_points[tangent_idx + 1] - sim.controller.path_points[tangent_idx]
        tangent /= np.linalg.norm(tangent)
        n_surf = sim._rotation_lg(*sim.surface_abg)[:, 2]
        left_normal = np.cross(n_surf, tangent)
        left_normal /= np.linalg.norm(left_normal)
        base = sim.controller.path_points[tangent_idx].copy()
        err_left  = sim.controller.signed_cross_track_error(base + 0.5 * left_normal)
        err_right = sim.controller.signed_cross_track_error(base - 0.5 * left_normal)
        assert err_left > 0
        assert err_right < 0


# ───────────────── Integration smoke test ─────────────────
class TestRun:
    def test_log_populated(self, sim: DozerSimulation) -> None:
        sim.stop_time = 0.05
        sim.run()
        assert len(sim.log) > 0

    def test_log_entry_length(self, sim: DozerSimulation) -> None:
        sim.stop_time = 0.05
        sim.run()
        assert np.asarray(sim.log).shape == (6, 7)  # time + six pose coordinates
        assert len(sim.force_log) == len(sim.log)

    def test_position_changes_when_running(self, sim: DozerSimulation) -> None:
        sim.stop_time = 0.2
        sim.run()
        log_arr = np.array(sim.log)
        # At least one position coordinate must have changed
        assert not np.allclose(log_arr[-1, 1:4], np.zeros(3))


# Surface utility and deformation tests
def _grid_heights(surface: DozerSimulation) -> FloatArray:
    return np.array([[point[2] for point in column] for column in surface.grid_pts])


def _body_simulation(**kwargs: Any) -> DozerSimulation:
    return DozerSimulation(enable_blade=False, **kwargs)


def test_dozer_simulation_combines_body_and_blade_modes() -> None:
    enabled = DozerSimulation()
    disabled = _body_simulation()

    assert enabled.enable_blade is True
    assert disabled.enable_blade is False
    assert hasattr(enabled, "blade_log")
    assert len(enabled.blade_log) == len(disabled.blade_log) == 1
    assert "run" in DozerSimulation.__dict__
    assert "run_and_plot" in DozerSimulation.__dict__


@pytest.mark.parametrize("enable_blade", [False, True])
def test_subdivision_uses_configurable_grid_resolution(enable_blade: bool) -> None:
    sim = DozerSimulation(enable_blade=enable_blade)
    assert sim.subdivision == pytest.approx(1.75 / 4)
    sim.division_factor = 8
    assert sim.subdivision == pytest.approx(1.75 / 8)


def test_body_grid_and_initial_logs() -> None:
    body = _body_simulation()
    heights = _grid_heights(body)
    assert heights.shape == (len(body.us), len(body.vs))
    # The default transition starts beyond the grid, leaving a flat surface.
    np.testing.assert_array_equal(heights, 0.0)
    assert body.is_initialized
    assert np.asarray(body.log).shape == (1, 7)
    np.testing.assert_allclose(body.log[0], [0., *body.q])
    assert np.asarray(body.neighbor_log).shape == (1, 4, 3)
    for (i, j), node in body.surf_grid.nodes(data=True):
        np.testing.assert_allclose([node["x"], node["y"], node["z"]], body.grid_pts[i][j])


def test_body_grid_adds_cross_slope_only_when_requested() -> None:
    unrolled = _body_simulation(is_surface_rolled=False)
    rolled = _body_simulation(is_surface_rolled=True)
    grids = []
    for sim in (unrolled, rolled):
        sim.u_split = sim.v_split = 0.0
        grid = sim._surface_grid()
        grids.append(np.array([[grid.nodes[i, j]["z"] for j in range(len(sim.vs))]
                               for i in range(len(sim.us))]))
    heights, rolled_heights = grids
    np.testing.assert_allclose(heights, np.broadcast_to(heights[0], heights.shape))
    assert heights.max() == pytest.approx(unrolled.b)
    assert np.ptp(rolled_heights[:, -1]) == pytest.approx(rolled.b)


@pytest.mark.parametrize("enable_blade", [False, True])
def test_track_settling_uses_terrain_roll(enable_blade: bool) -> None:
    sim = DozerSimulation(enable_blade=enable_blade)
    # A plane rising in world X rolls a vehicle facing world +Y.
    sim.grid_pts = [[(x, y, 0.2 * x) for x, y, _ in column] for column in sim.grid_pts]
    sim.starting_grid_heights = sim._grid_heights().copy()
    sim.q[:] = [1., 2., 0., 0., 0., np.pi / 2]
    sim._settle_tracks(sim.q[3:6].copy())
    assert sim.q[3] == pytest.approx(-np.arctan(0.2), abs=sim.contact_tol)
    assert sim.q[4] == pytest.approx(0., abs=sim.contact_tol)


def test_body_update_advances_and_settles_without_building_track_points() -> None:
    body = _body_simulation()
    previous_xy = body.q[:2].copy()

    neighbor_points = body._body_update()

    np.testing.assert_allclose(
        body.q[:2], previous_xy + body.dt * body.q_dot[:2]
    )
    assert np.asarray(neighbor_points).shape == (4, 3)
    np.testing.assert_allclose(neighbor_points, body._get_neighbor_points(body.q))

    orient = body.q[3:6].copy()
    rotation = body._rotation_lg(*orient)
    assert body.q[2] == pytest.approx(body._resting_height(rotation))


def test_track_geometry_helper_uses_the_logged_rigid_body_pose() -> None:
    surface = DozerSimulation(is_uphill=False)
    pose = np.array([1.2, -0.7, 0.4, 0.17, -0.23, 0.41])

    tracks = surface._track_xyz(pose)
    local = np.einsum(
        "ij,skj->ski",
        surface._rotation_gl(*pose[3:6]),
        tracks - pose[:3],
    )
    expected = np.array([
        [
            [surface.l / 2, -surface.b / 2, 0.0],
            [0.0, -surface.b / 2, 0.0],
            [-surface.l / 2, -surface.b / 2, 0.0],
        ],
        [
            [surface.l / 2, surface.b / 2, 0.0],
            [0.0, surface.b / 2, 0.0],
            [-surface.l / 2, surface.b / 2, 0.0],
        ],
    ])

    assert tracks.shape == (2, 3, 3)
    np.testing.assert_allclose(local, expected, atol=1e-12)


def test_visualization_draws_and_updates_tracks_in_every_view(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    surface = DozerSimulation(is_uphill=False, blade_roll_pitch_yaw=np.array([0.25, 0., 0.]))
    surface.stop_time = 3 * surface.dt + 1e-12
    surface.run()
    captured = {}

    monkeypatch.setattr(
        visualization_module.Image.Image,
        "save",
        lambda *args, **kwargs: captured.setdefault("save_kwargs", kwargs),
    )
    real_close = visualization_module.plt.close
    monkeypatch.setattr(
        visualization_module.plt,
        "close",
        lambda fig: captured.setdefault("figure", fig),
    )

    meshes = {}
    real_set_verts = visualization_module.Poly3DCollection.set_verts

    def record_vertices(artist: Artist, vertices: Sequence[FloatArray], *args: Any, **kwargs: Any) -> None:
        meshes[artist] = [np.asarray(face).copy() for face in vertices]
        return real_set_verts(artist, vertices, *args, **kwargs)

    monkeypatch.setattr(visualization_module.Poly3DCollection, "set_verts", record_vertices)
    monkeypatch.chdir(tmp_path)
    Visualization(surface).visualization()
    figure = captured["figure"]
    pose = np.asarray(surface.log)[::2][-1, 1:7]
    track_meshes = [faces for artist, faces in meshes.items()
                    if artist.get_label() == "dozer" and len(faces) > 6]
    assert len(track_meshes) == 2
    for side, faces in zip((-1., 1.), track_meshes):
        local = (np.vstack(faces) - pose[:3]) @ surface._rotation_lg(*pose[3:6])
        np.testing.assert_allclose(local.min(axis=0),
                                   [-surface.l / 2, side * surface.b / 2 - surface.track_width / 2, 0.], atol=1e-10)
        np.testing.assert_allclose(local.max(axis=0),
                                   [surface.l / 2, side * surface.b / 2 + surface.track_width / 2, surface.track_height], atol=1e-10)
    expected = np.vstack([face for faces in track_meshes for face in faces])
    projections = {
        ("X (m)", "Y (m)"): (0, 1),
        ("Y (m)", "Z (m)"): (1, 2),
        ("X (m)", "Z (m)"): (0, 2),
    }

    try:
        for axis in figure.axes:
            if axis.name == "3d":
                assert sum(artist.get_label() == "dozer" for artist in axis.collections) >= 4
                continue
            projection = projections.get((axis.get_xlabel(), axis.get_ylabel()))
            if projection is None:
                continue
            tracks, = [artist for artist in axis.collections if artist.get_label() == "tracks"]
            actual = np.vstack([path.vertices[:-1] for path in tracks.get_paths()])
            np.testing.assert_allclose(actual, expected[:, projection], atol=1e-10)

        side_axis = next(
            axis
            for axis in figure.axes
            if (axis.get_xlabel(), axis.get_ylabel()) == ("X (m)", "Z (m)")
        )
        labels = [text.get_text() for text in side_axis.get_legend().get_texts()]
        assert "tracks" in labels
        assert "blade face" in labels
        assert "blade contact points" in labels
        assert captured["save_kwargs"]["duration"] == int(1_000 / 30)

        blade = np.asarray(surface.blade_log)[::2][-1]
        blade_top = (
            blade[[0, -1], :3]
            + surface._blade_rotation_lg(blade[0, 3:6])[:, 2] * surface.H
        )
        expected_face = np.vstack([
            blade[0, :3], blade[-1, :3], blade_top[-1], blade_top[0]
        ])
        blade_face = next(
            patch
            for patch in side_axis.patches
            if patch.get_label() == "blade face"
        )
        np.testing.assert_allclose(
            blade_face.get_xy()[:4],
            expected_face[:, [0, 2]],
        )
    finally:
        real_close(figure)


def test_body_visualization_does_not_require_blade_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    surface = DozerSimulation(enable_blade=False, is_uphill=False)
    surface.stop_time = 3 * surface.dt + 1e-12
    surface.run()
    del surface.blade_log
    del surface.grid_log
    captured = {}

    monkeypatch.setattr(
        visualization_module.Image.Image,
        "save",
        lambda *args, **kwargs: captured.setdefault("save_kwargs", kwargs),
    )
    real_close = visualization_module.plt.close
    monkeypatch.setattr(
        visualization_module.plt,
        "close",
        lambda figure: captured.setdefault("figure", figure),
    )
    monkeypatch.chdir(tmp_path)

    Visualization(surface).visualization(show_neighbors=True)
    figure = captured["figure"]

    try:
        assert (tmp_path / "figures").is_dir()
        side_axis = next(
            axis
            for axis in figure.axes
            if (axis.get_xlabel(), axis.get_ylabel()) == ("X (m)", "Z (m)")
        )
        labels = [text.get_text() for text in side_axis.get_legend().get_texts()]
        collection_labels = {
            collection.get_label() for collection in side_axis.collections
        }
        assert "tracks" in labels
        assert "grid neighbors" in labels
        assert "blade face" not in labels
        assert "blade contact points" not in labels
        assert "track orientations" in collection_labels
        assert "track offsets" in collection_labels
        assert captured["save_kwargs"]["duration"] == int(1_000 / 20)
    finally:
        real_close(figure)


@pytest.mark.parametrize(
    "body_kwargs",
    ({}, {"is_surface_pitched": True}, {"is_backwards": True}),
)
def test_body_short_run_keeps_logs_aligned_without_deformation(body_kwargs: dict[str, bool]) -> None:
    body = _body_simulation(**body_kwargs)
    starting_heights = body._grid_heights().copy()
    body.stop_time = 3 * body.dt + 1e-12

    body.run()

    assert len(body.log) == len(body.neighbor_log) == 4
    assert np.asarray(body.log).shape == (4, 7)
    assert np.asarray(body.neighbor_log).shape == (4, 4, 3)
    np.testing.assert_allclose(body._grid_heights(), starting_heights)


def test_blade_enabled_short_run_keeps_logs_aligned_and_deforms() -> None:
    surface = DozerSimulation(is_uphill=False)
    starting_heights = surface._grid_heights().copy()
    surface.stop_time = 3 * surface.dt + 1e-12

    surface.run()

    assert len(surface.log) == len(surface.blade_log) == 4
    assert len(surface.grid_log) == len(surface.neighbor_log) == 4
    assert np.asarray(surface.log).shape == (4, 7)
    point_count = int(np.ceil(surface.B1 / surface.subdivision)) + 1
    assert np.asarray(surface.blade_log).shape == (4, point_count, 6)
    assert np.asarray(surface.grid_log).shape == (4, len(surface.us), len(surface.vs))
    np.testing.assert_allclose(surface.grid_log[-1], surface._grid_heights())
    assert np.max(starting_heights - surface._grid_heights()) > 0.0


@pytest.mark.parametrize("enable_blade", (True, False))
def test_run_and_plot_uses_one_mode_aware_visualization(monkeypatch: pytest.MonkeyPatch, enable_blade: bool) -> None:
    surface = DozerSimulation(enable_blade=enable_blade)
    calls = []
    monkeypatch.setattr(surface, "run", lambda: calls.append("run"))

    def visualization(instance: Visualization, show_neighbors: bool=False, show_desired_depth: bool=False) -> None:
        calls.append((
            "visualization", instance.simulation.enable_blade, show_neighbors, show_desired_depth
        ))

    monkeypatch.setattr(
        simulation_module.Visualization, "visualization", visualization
    )

    monkeypatch.setattr(simulation_module.Visualization, "forces_visualization",
                        lambda instance: calls.append("forces"))
    surface.run_and_plot(show_neighbors=True, show_desired_depth=True)

    assert calls == ["run", ("visualization", enable_blade, True, True), "forces"]


def _reached_contact(surface: DozerSimulation, tile: GridCell=(1, 1), depth: float=-10.0) -> FloatArray:
    """A contact near the tile's leading edge for the default +Y motion."""
    i, j = tile
    x0, y0, _ = surface.grid_pts[i][j]
    x1, y1, _ = surface.grid_pts[i + 1][j + 1]
    return np.array([(x0 + x1) / 2, y1 - surface.subdivision * 0.01, depth])


def test_duplicate_blade_contacts_cut_a_tile_once_per_frame() -> None:
    surface = DozerSimulation(is_uphill=False)
    surface.max_dig_depth_per_meter = 2.0
    surface.q_dot[:3] = [0.0, 10.0, 0.0]  # 0.1 m per step
    contact = _reached_contact(surface)
    before = surface.grid_pts[1][2][2]

    surface._deform_blade_tiles({(1, 1): [contact, contact.copy()]})

    assert surface.grid_pts[1][2][2] < before


def test_dig_depth_depends_on_distance_not_step_count() -> None:
    one_step = DozerSimulation(is_uphill=False)
    split_steps = DozerSimulation(is_uphill=False)
    for surface in (one_step, split_steps):
        surface.max_dig_depth_per_meter = 2.0
        surface.q_dot[:3] = [0.0, 10.0, 0.0]

    one_contact = _reached_contact(one_step)
    split_contact = _reached_contact(split_steps)
    one_step._deform_blade_tiles({(1, 1): [one_contact]})
    split_steps.dt = 0.005
    split_steps._deform_blade_tiles({(1, 1): [split_contact]})
    split_steps._deform_blade_tiles({(1, 1): [split_contact]})

    assert split_steps.grid_pts[1][1][2] == pytest.approx(one_step.grid_pts[1][1][2])


def test_initial_blade_geometry_does_not_deform_the_surface() -> None:
    surface = DozerSimulation(is_uphill=False)

    np.testing.assert_allclose(surface._grid_heights(), surface.grid_log[0])


def test_blade_local_yaw_and_roll_are_composed_after_body_rotation() -> None:
    local_yaw = 0.31
    local_roll = -0.22
    surface = DozerSimulation(
        is_uphill=False,
        blade_roll_pitch_yaw=np.array([local_roll, 0., local_yaw]),
    )
    surface.q[3:6] = [0.17, -0.13, 0.41]

    blade_points, _ = surface._blade_update(deform=False)
    body_R = surface._rotation_lg(*surface.q[3:6])
    local_R = surface._rotation_lg(local_roll, 0.0, local_yaw)
    expected_p0 = surface.q[:3] + body_R @ (surface.blade_arm_offset + local_R @ np.array(
        [surface.L, -surface.B1 / 2, -surface.blade_cut_depth()]
    ))
    expected_p1 = surface.q[:3] + body_R @ (surface.blade_arm_offset + local_R @ np.array(
        [surface.L, surface.B1 / 2, -surface.blade_cut_depth()]
    ))

    np.testing.assert_allclose(blade_points[0][:3], expected_p0, atol=1e-12)
    np.testing.assert_allclose(blade_points[-1][:3], expected_p1, atol=1e-12)


def test_blade_pitch_sets_cut_depth_without_tilting_local_rotation() -> None:
    blade_pitch = 0.27
    surface = DozerSimulation(is_uphill=False, blade_roll_pitch_yaw=np.array([0., blade_pitch, 0.]))
    surface.q[3:6] = [0.0, 0.0, 0.0]

    blade_points, _ = surface._blade_update(deform=False)
    p0, p1 = np.asarray(blade_points)[[0, -1], :3]
    expected_depth = surface.L * np.sin(blade_pitch)

    assert surface.blade_cut_depth() == pytest.approx(expected_depth)
    np.testing.assert_allclose(p0 - surface.q[:3], surface.blade_arm_offset + [surface.L, -surface.B1 / 2, -expected_depth])
    np.testing.assert_allclose(p1 - surface.q[:3], surface.blade_arm_offset + [surface.L, surface.B1 / 2, -expected_depth])
    assert p0[2] == pytest.approx(p1[2])


def test_only_forward_vertices_of_a_contacted_tile_deform() -> None:
    surface = DozerSimulation(is_uphill=False)
    surface.q_dot[:3] = [0.0, 10.0, 0.0]
    contact = _reached_contact(surface)
    before = surface._grid_heights()

    surface._deform_blade_tiles({(1, 1): [contact]})

    # +Y motion: j=2 is directly cut; j=1 remains untouched.
    assert surface.grid_pts[1][1][2] == pytest.approx(before[1, 1])
    assert surface.grid_pts[2][1][2] == pytest.approx(before[2, 1])
    assert surface.grid_pts[1][2][2] < before[1, 2]
    assert surface.grid_pts[2][2][2] < before[2, 2]


def test_repeated_cuts_stop_at_each_vertex_starting_height_limit() -> None:
    surface = DozerSimulation(is_uphill=False)
    surface.max_dig_depth_per_meter = 100.0
    surface.max_world_cut_depth = 0.25
    surface.q_dot[:3] = [0.0, 10.0, 0.0]
    contact = _reached_contact(surface)
    starting_heights = surface.starting_grid_heights.copy()

    for _ in range(3):
        surface._deform_blade_tiles({(1, 1): [contact]})

    for i, j in ((1, 2), (2, 2)):
        assert surface.grid_pts[i][j][2] == pytest.approx(
            starting_heights[i, j] - surface.max_world_cut_depth
        )


def test_blade_points_remain_interpolated_between_rigid_endpoints() -> None:
    surface = DozerSimulation(is_uphill=False)
    surface.max_world_cut_depth = 0.1
    surface.q[2] = -10.0

    blade_points, _ = surface._blade_update(deform=False)
    blade = np.asarray(blade_points)[:, :3]
    expected = np.array([
        (1.0 - t) * blade[0] + t * blade[-1]
        for t in np.linspace(0.0, 1.0, len(blade))
    ])

    np.testing.assert_allclose(blade, expected, atol=1e-12)


def test_rolled_blade_points_interpolate_between_rolled_endpoints() -> None:
    surface = DozerSimulation(is_uphill=False, blade_roll_pitch_yaw=np.array([0.25, 0., 0.]))
    surface.max_world_cut_depth = 0.1
    surface.q[2] = -1.0

    blade_points, _ = surface._blade_update(deform=False)
    blade = np.asarray(blade_points)[:, :3]
    expected = np.array([
        (1.0 - t) * blade[0] + t * blade[-1]
        for t in np.linspace(0.0, 1.0, len(blade))
    ])

    np.testing.assert_allclose(blade, expected, atol=1e-12)
    assert np.ptp(blade[:, 2]) > 0.0


def test_whole_track_contact_updates_pose_on_forward_slope() -> None:
    surface = DozerSimulation(is_uphill=False)
    slope = 0.2
    for i in range(len(surface.us)):
        for j in range(len(surface.vs)):
            x, y, _ = surface.grid_pts[i][j]
            z = slope * y
            surface.grid_pts[i][j] = (x, y, z)
            surface.surf_grid.nodes[(i, j)]["z"] = z
    surface.starting_grid_heights = surface._grid_heights().copy()
    surface.is_initialized = False
    surface.q[3] = 0.17
    surface.q[5] = np.pi / 2

    surface._body_update()

    R = surface._rotation_lg(*surface.q[3:6])
    assert surface.q[2] == pytest.approx(surface._resting_height(R), abs=1e-8)
    assert surface.q[4] != pytest.approx(0.0)
    assert surface.q[3] == pytest.approx(0.0, abs=1e-9)
    assert surface.q[5] == pytest.approx(np.pi / 2)


def test_blade_keeps_local_offset_on_forward_slope() -> None:
    surface = DozerSimulation(is_uphill=False)
    slope = 0.2
    for i in range(len(surface.us)):
        for j in range(len(surface.vs)):
            x, y, _ = surface.grid_pts[i][j]
            z = slope * y
            surface.grid_pts[i][j] = (x, y, z)
            surface.surf_grid.nodes[(i, j)]["z"] = z
    surface.starting_grid_heights = surface._grid_heights().copy()
    surface._body_update()

    blade_points, _ = surface._blade_update(deform=False)

    blade = np.asarray(blade_points)
    midpoint = (blade[0, :3] + blade[-1, :3]) / 2
    local_midpoint = surface._rotation_gl(*surface.q[3:6]) @ (midpoint - surface.q[:3])
    np.testing.assert_allclose(
        local_midpoint, surface.blade_arm_offset + [surface.L, 0.0, -surface.blade_cut_depth()], atol=1e-12
    )
    R = surface._rotation_lg(*surface.q[3:6])
    assert surface.q[2] == pytest.approx(surface._resting_height(R), abs=1e-8)


def test_blade_enabled_pose_uses_complete_tracks_instead_of_q_contact() -> None:
    surface = DozerSimulation(is_uphill=False)

    # Flatten the terrain, then raise only the small patch beneath q. Both
    # track centerlines are laterally clear of it, so q must not act as a
    # third, artificial support point.
    for i in range(len(surface.us)):
        for j in range(len(surface.vs)):
            x, y, _ = surface.grid_pts[i][j]
            surface.grid_pts[i][j] = (x, y, 0.0)
            surface.surf_grid.nodes[(i, j)]["z"] = 0.0

    i, j = surface._grid_cell(surface.q)
    for ci, cj in ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)):
        x, y, _ = surface.grid_pts[ci][cj]
        surface.grid_pts[ci][cj] = (x, y, 1.0)
        surface.surf_grid.nodes[(ci, cj)]["z"] = 1.0

    surface.is_initialized = False
    surface._body_update()

    assert surface._bilinear_height(surface.q, surface._get_neighbor_points(surface.q)) == pytest.approx(1.0)
    assert surface.q[2] == pytest.approx(0.0, abs=surface.contact_tol)
    R = surface._rotation_lg(*surface.q[3:6])
    assert surface.q[2] == pytest.approx(surface._resting_height(R))


def test_blade_enabled_pose_balances_and_clears_entire_tracks() -> None:
    surface = DozerSimulation(is_uphill=True)
    R = surface._rotation_lg(*surface.q[3:6])
    support = surface._support_heights(R)

    assert surface.l == pytest.approx(2.349)
    assert support.max() <= surface.q[2] + surface.contact_tol
    assert surface.q[2] == pytest.approx(surface._resting_height(R))


def test_commanded_blade_pitch_does_not_replace_whole_track_balance() -> None:
    shallow = DozerSimulation(is_uphill=True, blade_roll_pitch_yaw=np.array([0., 0.0, 0.]))
    deep = DozerSimulation(is_uphill=True, blade_roll_pitch_yaw=np.array([0., 0.3, 0.]))

    np.testing.assert_allclose(shallow.q, deep.q, atol=1e-12)
    assert shallow.blade_cut_depth() != pytest.approx(deep.blade_cut_depth())


def test_blade_descends_with_q_after_surface_is_cut() -> None:
    surface = DozerSimulation(is_uphill=False)
    surface.max_world_cut_depth = 1.0
    before, _ = surface._blade_update(deform=False)

    surface.q[2] -= 0.1
    after, _ = surface._blade_update(deform=False)

    np.testing.assert_allclose(
        np.asarray(after)[:, 2], np.asarray(before)[:, 2] - 0.1
    )


def test_blade_midpoint_keeps_local_offsets_from_q_until_depth_limit() -> None:
    surface = DozerSimulation(is_uphill=False)
    surface.max_world_cut_depth = 10.0
    surface.q[2] = -0.3
    surface.q[4] = 0.4

    blade, _ = surface._blade_update(deform=False)
    blade = np.asarray(blade)
    midpoint = (blade[0, :3] + blade[-1, :3]) / 2

    local_midpoint = surface._rotation_gl(*surface.q[3:6]) @ (midpoint - surface.q[:3])
    np.testing.assert_allclose(
        local_midpoint, surface.blade_arm_offset + [surface.L, 0.0, -surface.blade_cut_depth()], atol=1e-12
    )


def test_rigid_blade_vertical_motion_continues_below_soil_cut_limit() -> None:
    surface = DozerSimulation(is_uphill=False)
    surface.max_world_cut_depth = 0.25
    surface.q[2] = -1.0

    before, _ = surface._blade_update(deform=False)
    surface.q[2] -= 0.2
    after, _ = surface._blade_update(deform=False)

    np.testing.assert_allclose(
        np.asarray(after)[:, 2], np.asarray(before)[:, 2] - 0.2
    )


def test_normal_run_deepens_with_q_but_stops_at_maximum_depth() -> None:
    surface = DozerSimulation(is_uphill=False)

    surface.run()

    depths = surface.starting_grid_heights - surface._grid_heights()
    assert depths.max() > surface.blade_cut_depth()
    assert depths.max() <= surface.max_world_cut_depth + 2e-7


def test_unrolled_transition_and_cut_are_uniform_across_blade_width() -> None:
    surface = DozerSimulation(is_uphill=False, is_surface_rolled=False)
    np.testing.assert_allclose(
        np.ptp(surface.starting_grid_heights, axis=0), 0.0, atol=1e-12
    )

    surface.run()

    depths = surface.starting_grid_heights - surface._grid_heights()
    cut_columns = np.flatnonzero(depths.max(axis=0) > 1e-9)
    for j in cut_columns:
        # The widened terrain extends beyond the blade's swept strip.
        cut_depths = depths[:, j][depths[:, j] > 1e-9]
        assert len(cut_depths) > 1
        np.testing.assert_allclose(cut_depths, cut_depths[0], atol=2e-7)
    assert len(cut_columns) > 0
    assert np.all(depths[surface.us > surface.B1 / 2 + surface.subdivision] == 0.)


def test_contact_pitch_still_balances_tracks_after_cutoff_depth() -> None:
    surface = DozerSimulation(is_uphill=False)
    cutoff_q_depth = (
        surface.max_world_cut_depth - surface.blade_cut_depth()
    )
    surface.grid_pts = [
        [(x, y, z - cutoff_q_depth) for x, y, z in column]
        for column in surface.grid_pts
    ]
    for i, column in enumerate(surface.grid_pts):
        for j, (_, _, z) in enumerate(column):
            surface.surf_grid.nodes[(i, j)]["z"] = z
    surface.q[2] = surface._bilinear_height(surface.q, surface._get_neighbor_points(surface.q))
    orient = surface.q[3:6].copy()

    pitch = surface._body_contact_pitch(orient)
    R = surface._rotation_lg(orient[0], pitch, orient[2])
    front, back = surface._half_resting_heights(R)
    assert front == pytest.approx(back, abs=surface.contact_tol)


def test_rolled_soil_is_never_cut_below_the_blade_plane() -> None:
    surface = DozerSimulation(is_uphill=False, is_surface_rolled=True)
    surface.q_dot[:3] = [0.0, 10.0, 0.0]
    contact = _reached_contact(surface, tile=(4, 1), depth=-0.3)

    surface._deform_blade_tiles({(4, 1): [contact]})

    for i, j in ((4, 2), (5, 2)):
        assert surface.grid_pts[i][j][2] == pytest.approx(contact[2])


def test_rolled_soil_keeps_local_depth_limits_across_the_blade() -> None:
    surface = DozerSimulation(is_uphill=False, is_surface_rolled=True)

    surface.run()

    heights = surface._grid_heights()
    deformed = np.abs(heights - surface.starting_grid_heights) > 1e-9
    for i, j in np.argwhere(deformed):
        local_floor = (
            surface.starting_grid_heights[i, j] - surface.max_world_cut_depth
        )
        assert heights[i, j] >= local_floor - 2e-7


# Blade deformation
@pytest.fixture
def simulation() -> DozerSimulation:
    sim = DozerSimulation(is_surface_pitched=True, blade_roll_pitch_yaw=np.array([0., .2, 0.]))
    sim.division_factor = 7  # 0.25 m grid
    sim.us = np.arange(-3., 4.01, sim.subdivision)
    sim.vs = np.arange(-3., 3.01, sim.subdivision)
    sim.surf_grid = nx.grid_2d_graph(len(sim.us), len(sim.vs))
    sim.grid_pts = [[(x, y, 0.) for y in sim.vs] for x in sim.us]
    nx.set_node_attributes(sim.surf_grid, 0., "z")
    sim.starting_grid_heights = np.zeros((len(sim.us), len(sim.vs)))
    sim.controller.desired_depth = 0.2
    sim.q[:] = 0.
    sim.q_dot[:] = 0.
    sim.blade_roll_pitch_yaw[:] = 0.
    return sim


def assert_clearance(sim: DozerSimulation) -> None:
    rotation = sim._rotation_lg(*sim.q[3:6])
    edge = sim.q[:3] + sim._blade_edge_local(top=True) @ rotation.T
    for t in np.linspace(0., 1., 1001):
        point = edge[0] + t * (edge[1] - edge[0])
        assert point[2] >= sim._undeformed_height(point) - 1e-7
    # Tracks still cannot pass below the current flat surface.
    assert sim._track_xyz(sim.q)[:, :, 2].min() >= -1e-7


def test_blade_with_clearance_does_not_lift_body(simulation: DozerSimulation) -> None:
    simulation._settle_tracks(simulation.q[3:6].copy())
    np.testing.assert_allclose(simulation.q, 0., atol=1e-7)
    assert_clearance(simulation)


def test_original_soil_supports_blade_after_surface_is_cut(simulation: DozerSimulation) -> None:
    # Current surface is already cut to zero, original soil remains at 1.4 m.
    simulation.starting_grid_heights[:] = 1.4
    simulation._settle_tracks(simulation.q[3:6].copy())
    assert simulation.q[2] > 0.1
    assert simulation.q[4] < -0.1  # blade/front lifts; rear track ends support it
    assert abs(simulation.q[3]) < 1e-7
    assert abs(simulation._track_xyz(simulation.q)[:, -1, 2].min()) < 1e-7
    assert_clearance(simulation)


def test_asymmetric_soil_at_blade_changes_roll(simulation: DozerSimulation) -> None:
    for i, x in enumerate(simulation.us):
        for j, y in enumerate(simulation.vs):
            if x > 1.25 and y > .25:
                simulation.starting_grid_heights[i, j] = 1.4
    simulation._settle_tracks(simulation.q[3:6].copy())
    assert simulation.q[3] > .1
    assert simulation.q[5] == 0.  # contact does not steer the vehicle
    assert_clearance(simulation)


@pytest.mark.parametrize("blade_angles", [[.06, .3, .3], [-.06, .3, -.3]])
def test_rotated_offset_blade_clears_soil(simulation: DozerSimulation, blade_angles: Sequence[float]) -> None:
    simulation.blade_roll_pitch_yaw[:] = blade_angles
    simulation.blade_arm_offset = np.array([.7, .2, .1])
    simulation.starting_grid_heights[:] = 1.4
    simulation.q[5] = .4
    simulation._settle_tracks(simulation.q[3:6].copy())
    assert simulation.q[5] == .4
    assert_clearance(simulation)


def test_disabled_blade_does_not_support_vehicle(simulation: DozerSimulation) -> None:
    simulation.enable_blade = False
    simulation.starting_grid_heights[:] = 1.4
    simulation._settle_tracks(simulation.q[3:6].copy())
    np.testing.assert_allclose(simulation.q, 0., atol=1e-7)


def test_blade_pose_is_valid_before_cutting(simulation: DozerSimulation, monkeypatch: pytest.MonkeyPatch) -> None:
    simulation.starting_grid_heights[:] = 1.4
    simulation.q_dot[0] = 1.
    called = []

    def deform(contacts: ContactMap) -> None:
        assert contacts
        assert_clearance(simulation)
        called.append(True)

    monkeypatch.setattr(simulation, "_deform_blade_tiles", deform)
    simulation._blade_update()
    assert called == [True]


def test_peak_between_blade_endpoints_supports_vehicle(simulation: DozerSimulation) -> None:
    # A narrow ridge in the middle is missed by endpoint-only contact tests.
    simulation.starting_grid_heights[:, len(simulation.vs) // 2] = 1.4
    simulation._settle_tracks(simulation.q[3:6].copy())
    assert simulation.q[2] > .1
    assert_clearance(simulation)


def test_bilinear_peak_inside_tile_is_included(simulation: DozerSimulation) -> None:
    # A diagonal edge crosses a saddle tile: the largest required height is
    # inside the tile, away from either endpoint or any grid-line crossing.
    simulation.us = np.array([0., 1.])
    simulation.vs = np.array([0., 1.])
    simulation.grid_pts = [[(float(i), float(j), 0.) for j in range(2)] for i in range(2)]
    simulation.starting_grid_heights = np.array([[0., 2.], [2., 0.]])
    simulation._blade_edge_local = lambda top=False: np.array([[0., 0., 0.], [1., 1., 0.]])
    support = simulation._blade_support_heights(np.eye(3))
    assert support.max() == pytest.approx(1.)


@pytest.mark.parametrize("offset, stopped", [(0.01, False), (0.0, True), (-0.01, True)])
def test_track_cut_limit_before_blade_support(simulation: DozerSimulation, monkeypatch: pytest.MonkeyPatch, offset: float, stopped: bool) -> None:
    sim = simulation
    sim.blade_roll_pitch_yaw[:] = [0.1, 0.2, 0.0]
    R = sim._rotation_lg(0.15, 0.1, 0.0)
    local_R = sim._rotation_lg(0.1, 0.0, 0.0)
    uncut_center = sim.blade_arm_offset + local_R @ np.array([sim.L, 0., 0.])
    relative_edge = (sim._blade_edge_local() - uncut_center) @ R.T
    deepest_depth = -relative_edge[:, 2].min()
    # Measure track-only support for this orientation on the flat terrain.
    sim.enable_blade = False
    track_height = sim._support_heights(R).max()
    sim.enable_blade = True
    sim.starting_grid_heights[:] = (
        track_height + sim.max_world_cut_depth
        - deepest_depth - offset)

    def blade_support(rotation: FloatArray) -> FloatArray:
        assert sim.cut_limit_reached == stopped
        return np.full((2, 2), 10.0)

    monkeypatch.setattr(sim, "_blade_support_heights", blade_support)
    sim._support_heights(R, check_cut_limit=True)
    sim._blade_terrain_interaction()
    assert (sim.Fb == -1e9) == stopped


def test_cut_limit_force_stops_and_holds_vehicle(simulation: DozerSimulation) -> None:
    sim = simulation
    sim.starting_grid_heights[:] = sim.max_world_cut_depth
    sim._settle_tracks(sim.q[3:6].copy())
    assert sim.cut_limit_reached
    sim.v[0] = sim.velocity_limit
    sim.q_dot[0] = sim.velocity_limit
    for _ in range(2):
        sim._update_q_dot()
        assert sim.Fb == -1e9
        assert sim.v[0] == 0.0


def test_disabled_blade_does_not_stop_at_cut_limit(simulation: DozerSimulation) -> None:
    sim = simulation
    sim.enable_blade = False
    sim.starting_grid_heights[:] = sim.max_world_cut_depth
    sim._settle_tracks(sim.q[3:6].copy())
    assert not sim.cut_limit_reached


def test_excessive_cut_command_stops_at_cut_limit() -> None:
    sim = DozerSimulation(blade_roll_pitch_yaw=np.zeros(3))
    # Normal 0.2 m feedback can finish the course; deliberately request a cut
    # beyond the terrain limit to exercise the independent stop protection.
    sim.controller.desired_depth = sim.max_world_cut_depth + sim.H
    sim.stop_time = 5.0
    sim.run()
    assert sim.total_distance < sim.stop_distance
    assert sim.cut_limit_reached
    assert sim.Fb == -1e9
    assert sim.v[0] == 0.0


@pytest.mark.parametrize("roll", [-0.2, 0.0, 0.2])
def test_cut_limit_roll_depth_excludes_arm_lift(simulation: DozerSimulation, roll: float) -> None:
    sim = simulation
    sim.blade_roll_pitch_yaw[:] = [roll, 0.2, 0.0]
    pitch = 0.15
    R = sim._rotation_lg(0.0, pitch, 0.0)
    sim.enable_blade = False
    track_height = sim._support_heights(R).max()
    sim.enable_blade = True
    # Analytic depth below the uncut center; either roll sign lowers an end.
    depth = np.cos(pitch) * (
        np.cos(roll) * sim.blade_cut_depth() + abs(np.sin(roll)) * sim.B1 / 2)
    sim.starting_grid_heights[:] = track_height + sim.max_world_cut_depth - depth
    for arm_height in (0.0, 10.0):
        sim.blade_arm_offset = np.array([5.0, 0.0, arm_height])
        sim._support_heights(R, check_cut_limit=True)
        assert sim.cut_limit_reached
        sim.starting_grid_heights[:] -= 0.01
        sim._support_heights(R, check_cut_limit=True)
        assert not sim.cut_limit_reached
        sim.starting_grid_heights[:] += 0.01


@pytest.mark.parametrize("depth, expected", [(-0.1, 0.3), (0.1, 0.1), (0.2, 0.0), (0.3, -0.1)])
def test_deformation_controller_signed_depth_error(simulation: DozerSimulation, depth: float, expected: float) -> None:
    controller = simulation.controller
    errors, plot_errors = controller.blade_deformation_errors(2.0, 2.0 - depth)
    np.testing.assert_allclose(errors, [0., np.arcsin(expected / simulation.L), 0.], atol=1e-12)
    np.testing.assert_allclose(plot_errors, errors)
    np.testing.assert_allclose(
        controller.proportional_blade_controller(2.0, 2.0 - depth),
        controller.Kp * errors)


def test_blade_bottom_center_includes_body_and_offset(simulation: DozerSimulation) -> None:
    sim = simulation
    angle = np.pi / 6
    depth = 0.2
    sim.q[:] = [1.0, -1.0, 2.0, angle, 0.0, np.pi / 2]
    sim.blade_arm_offset = np.array([0.4, 0.3, 0.1])
    sim.blade_roll_pitch_yaw[:] = [angle, np.arcsin(depth / sim.L), 0.0]
    expected = [
        1.0 - (0.3 * np.cos(angle) - 0.1 * np.sin(angle) + depth * np.sin(2 * angle)),
        -1.0 + 0.4 + sim.L,
        2.0 + 0.3 * np.sin(angle) + 0.1 * np.cos(angle) - depth * np.cos(2 * angle),
    ]
    np.testing.assert_allclose(sim._blade_bottom_center(), expected, atol=1e-12)


def test_controller_samples_original_surface_at_blade_center(simulation: DozerSimulation, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = simulation
    sim.starting_grid_heights[:] = 0.3 * sim.us[:, None] + 0.1 * sim.vs[None, :]
    sim.blade_arm_offset = np.array([0.4, 0.2, 0.1])
    observed = []

    def capture(starting_height: Scalar, blade_height: Scalar) -> FloatArray:
        observed.append((starting_height, blade_height))
        return np.zeros(3)

    monkeypatch.setattr(sim.controller, "proportional_blade_controller", capture)
    sim._blade_update()
    np.testing.assert_allclose(observed, [[0.3 * (sim.L + 0.4) + 0.1 * 0.2, 0.1]])


@pytest.mark.parametrize("initial_depth, direction", [(0.1, 1), (0.3, -1)])
def test_deformation_feedback_moves_pitch_toward_target(simulation: DozerSimulation, initial_depth: float, direction: int) -> None:
    sim = simulation
    sim.blade_roll_pitch_yaw[1] = np.arcsin(initial_depth / sim.L)
    before = sim.blade_roll_pitch_yaw[1]
    sim._blade_update()
    change = sim.blade_roll_pitch_yaw[1] - before
    assert direction * change > 0
    assert abs(change) <= sim.blade_roll_pitch_yaw_rate_limits[1] * sim.dt + 1e-12


def test_deformation_controller_converges_on_flat_surface(simulation: DozerSimulation) -> None:
    sim = simulation
    # Verify feedback convergence with explicit gains, independent of tuning defaults.
    sim.controller.Kp[1] = 0.1
    sim.controller.Kd[:] = 0.0
    for _ in range(300):
        sim._blade_update()
    measured_depth = sim._undeformed_height(sim._blade_bottom_center()) - sim._blade_bottom_center()[2]
    assert measured_depth == pytest.approx(0.2, abs=1e-3)


# Blade derivative
def test_fresh_controller_has_no_derivative_kick() -> None:
    controller = Control(1.2, 0.1, 1.0)
    error = controller.blade_deformation_errors(0., 0.)[0]
    np.testing.assert_allclose(controller.proportional_blade_controller(0., 0.), controller.Kp * error)
    np.testing.assert_array_equal(controller.derivative_error, 0.)
    controller.proportional_blade_controller(0., -0.1)
    controller = Control(1.2, 0.1, 1.0)
    np.testing.assert_allclose(controller.proportional_blade_controller(0., 0.), controller.Kp * error)
    np.testing.assert_array_equal(controller.derivative_error, 0.)


@pytest.mark.parametrize("dt", [0.01, 0.1])
def test_derivative_uses_error_difference_over_timestep(dt: float) -> None:
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


def test_zero_derivative_gain_preserves_proportional_output() -> None:
    controller = Control(1.2, 0.1, 1.0)
    controller.Kd[:] = 0.
    controller.proportional_blade_controller(0., 0.)
    command = controller.proportional_blade_controller(0., -0.1)
    np.testing.assert_allclose(command, controller.Kp * controller.blade_deformation_errors(0., -0.1)[0])


@pytest.mark.parametrize("limit", ["rate", "angle"])
def test_actuator_limits_still_apply(limit: str) -> None:
    sim = DozerSimulation(blade_roll_pitch_yaw=np.zeros(3))
    if limit == "rate":
        sim.blade_roll_pitch_yaw_rate_limits[:] = 0.
    else:
        sim.blade_roll_pitch_yaw_limits[:] = 0.
    sim._blade_update()
    sim.controller.desired_depth = 1.0
    sim._blade_update()
    np.testing.assert_array_equal(sim.blade_roll_pitch_yaw, 0.)


def test_geometry_refresh_does_not_update_derivative_history() -> None:
    sim = DozerSimulation(blade_roll_pitch_yaw=np.zeros(3))
    sim._blade_update(deform=False)
    assert sim.controller.previous_blade_error is None
    sim._blade_update()
    previous = sim.controller.previous_blade_error.copy()
    sim._blade_update(deform=False)
    np.testing.assert_array_equal(sim.controller.previous_blade_error, previous)


# Blade rmse
def test_rmse_uses_each_reference_and_includes_terminal_pose(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    sim = DozerSimulation(blade_roll_pitch_yaw=np.zeros(3))
    sim.controller.desired_depth = 0.2
    sim.stop_time = 0.1
    sim.stop_distance = 0.01
    sim.q_dot[:3] = [1., 0., 0.]
    monkeypatch.setattr(sim, "_blade_bottom_center", lambda: np.array([0., 0., 0.]))
    monkeypatch.setattr(sim, "_undeformed_height", lambda _: 0.)
    monkeypatch.setattr(sim, "_blade_update", lambda **_: (np.zeros((2, 3)), []))
    def move() -> list[Point3]:
        sim.controller.desired_depth = 0.4
        return []
    monkeypatch.setattr(sim, "_body_update", move)
    sim.run()
    assert sim.blade_depth_rmse == pytest.approx(np.sqrt((0.2**2 + 0.4**2) / 2))
    assert len(sim.force_log) == len(sim.log) == 2
    assert sim.force_log[-1]["desired_depth"] == 0.4
    assert sim.force_log[-1]["blade_depth_error"] == -0.4
    assert "Blade depth RMSE:" in capsys.readouterr().out


def test_zero_step_run_uses_current_initial_depth() -> None:
    sim = DozerSimulation(blade_roll_pitch_yaw=np.zeros(3))
    sim.stop_time = 0.
    center = sim._blade_bottom_center()
    sim.controller.desired_depth = sim._undeformed_height(center) - center[2]
    sim.run()
    assert sim.blade_depth_rmse == pytest.approx(0.)


def test_disabled_blade_has_no_depth_rmse(capsys: pytest.CaptureFixture[str]) -> None:
    sim = DozerSimulation(enable_blade=False)
    sim.stop_time = 0.02
    sim.run()
    assert sim.blade_depth_rmse is None
    assert "Blade depth RMSE:" not in capsys.readouterr().out


# Forces
@pytest.mark.parametrize("enable_blade", [True, False])
def test_force_history_matches_logged_steps(enable_blade: bool) -> None:
    simulation = DozerSimulation(enable_blade=enable_blade)
    simulation.stop_time = 0.04
    simulation.run()
    assert len(simulation.force_log) == len(simulation.log)
    np.testing.assert_allclose(
        [row["time"] for row in simulation.force_log],
        np.asarray(simulation.log)[:, 0],
    )
    last = simulation.force_log[-1]
    assert last["Fb"] == simulation.Fb
    assert last["Mr"] == simulation.Mr
    assert last["v_forward"] == simulation.v[0]
    center = simulation._blade_bottom_center()
    assert last["starting_surface_height"] == pytest.approx(simulation._undeformed_height(center))
    assert last["blade_bottom_height"] == pytest.approx(center[2])
    simulation.q[2] += 1.0
    assert last["blade_bottom_height"] == pytest.approx(center[2])
    original = last["drive_left"]
    simulation.F_track[0] = -1
    assert last["drive_left"] == original


def test_force_gif_includes_final_sample_and_all_panels(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    simulation = DozerSimulation()
    for index in range(1, 4):
        simulation.Fb = -100.0 * index
        simulation.Mb = 20.0 * index
        simulation.requested_blade_rates = np.array([index, -2 * index, 3 * index], dtype=float)
        simulation.q[2] += 0.1
        simulation._log_forces(index * simulation.dt)
    captured = []
    real_close = visualization_module.plt.close
    monkeypatch.setattr(visualization_module.plt, "close", lambda fig: captured.append(fig))
    try:
        output = Visualization(simulation).forces_visualization(tmp_path / "forces.gif")
        with Image.open(output) as gif:
            assert gif.n_frames == 3
            assert gif.size == (1200, 1300)
        figure = captured[-1]
        assert len(figure.axes) == 10
        height_axis = figure.axes[-2]
        assert height_axis.get_title() == "Blade-center heights"
        for line, key in zip(height_axis.lines[:2], ("starting_surface_height", "blade_bottom_height")):
            np.testing.assert_allclose(line.get_ydata(), [row[key] for row in simulation.force_log])
            np.testing.assert_allclose(line.get_xdata(), [0, .01, .02, .03])
        rate_axis = figure.axes[-1]
        assert rate_axis.get_title() == "Requested blade rates (before limiting)"
        for line, scale in zip(rate_axis.lines[:3], (1, -2, 3)):
            np.testing.assert_allclose(line.get_ydata(), np.arange(4) * scale)
            np.testing.assert_allclose(line.get_xdata(), [0, .01, .02, .03])
        assert np.ptp(height_axis.lines[1].get_ydata()) > 0.2
        np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [0, -100, -200, -300])
        np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(), [0, .01, .02, .03])
    finally:
        for figure in captured:
            real_close(figure)


def test_single_sample_force_gif(tmp_path: Path) -> None:
    output = Visualization(DozerSimulation()).forces_visualization(tmp_path / "initial.gif")
    with Image.open(output) as gif:
        assert gif.n_frames == 1


def test_requested_rates_are_saved_before_limiting(monkeypatch: pytest.MonkeyPatch) -> None:
    simulation = DozerSimulation()
    previous = simulation.blade_roll_pitch_yaw.copy()
    command = np.array([2., -3., 4.])
    monkeypatch.setattr(simulation.controller, "proportional_blade_controller", lambda *_: command)
    simulation._blade_update()
    expected = command / simulation.dt
    np.testing.assert_allclose(simulation.requested_blade_rates, expected)
    assert np.all(np.abs(expected) > simulation.blade_roll_pitch_yaw_rate_limits)
    assert np.all(np.abs(simulation.blade_roll_pitch_yaw - previous)
                  <= simulation.blade_roll_pitch_yaw_rate_limits * simulation.dt + 1e-12)
    simulation._blade_update(deform=False)
    simulation._log_forces(simulation.dt)
    simulation.requested_blade_rates[:] = 0
    for key, value in zip(("requested_roll_rate", "requested_pitch_rate", "requested_yaw_rate"), expected):
        assert simulation.force_log[-1][key] == pytest.approx(value)
        assert simulation.force_log[0][key] == 0


# Pd tuning
def test_pd_grid_search_uses_fresh_simulations_and_actual_gains(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    trials = []
    class Simulation:
        def __init__(self, blade_roll_pitch_yaw: FloatArray, controller_type: str) -> None:
            assert controller_type == "pd"
            np.testing.assert_array_equal(blade_roll_pitch_yaw, 0.)
            self.controller = type("Controller", (), {"Kp": np.zeros(3), "Kd": np.zeros(3)})()
            self.total_distance = 1.
            self.cut_limit_reached = False
            trials.append(self)
        def run(self) -> None:
            assert np.isinf(self.stop_distance)
            assert self.controller.desired_depth == 0.3
            assert self.stop_time == 1.0
            self.blade_depth_rmse = self.controller.Kp[1] + 10 * self.controller.Kd[1]
            self.force_log = [{"time": self.stop_time}]
    monkeypatch.setattr(tuning, "DozerSimulation", Simulation)
    result = tuning.Tuning("pd").grid_search([1., 2.], [0., 0.1], duration=1., output_dir=tmp_path)
    np.testing.assert_allclose(result, [[1., 2.], [2., 3.]])
    assert len(trials) == 4
    assert (tmp_path / "rmse_heatmap.png").is_file()
    with (tmp_path / "results.csv").open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["kp", "kd", "rmse_m", "desired_depth_m",
                                    "duration_s", "distance_m", "cut_limit_reached"]
        rows = list(reader)
        np.testing.assert_allclose([float(row["rmse_m"]) for row in rows], result.ravel())
        np.testing.assert_allclose([float(row["kp"]) for row in rows], [1., 2., 1., 2.])
        np.testing.assert_allclose([float(row["kd"]) for row in rows], [0., 0., .1, .1])


# Pi controller
def test_pi_accumulates_and_fresh_controller_starts_empty() -> None:
    controller = Control(1.2, .1, 1., controller_type="pi")
    error = controller.blade_deformation_errors(0., 0.)[0]
    for n in range(1, 4):
        output = controller.proportional_blade_controller(0., 0.)
        np.testing.assert_allclose(controller.integral_error, error * n * .1)
        np.testing.assert_allclose(output, controller.Kp * error + controller.Ki * error * n * .1)
    controller = Control(1.2, .1, 1., controller_type="pi")
    np.testing.assert_array_equal(controller.integral_error, 0.)


@pytest.mark.parametrize("limit", ["rate", "angle"])
def test_pi_limits_prevent_windup(limit: str) -> None:
    sim = DozerSimulation(controller_type="pi", blade_roll_pitch_yaw=np.zeros(3))
    if limit == "rate":
        sim.blade_roll_pitch_yaw_rate_limits[:] = 0.
    else:
        sim.blade_roll_pitch_yaw_limits[:] = 0.
    for _ in range(3):
        sim._blade_update()
    np.testing.assert_array_equal(sim.controller.integral_error, 0.)


def test_pi_can_unwind_at_limit() -> None:
    controller = Control(1.2, .1, 1., controller_type="pi")
    controller._previous_integral[1] = .5
    controller.integral_error[1] = .4
    controller.apply_actuator_feedback(np.array([0., .2, 0.]), np.array([0., .1, 0.]))
    assert controller.integral_error[1] == .4


def test_controller_selection_and_geometry_only_update() -> None:
    assert type(DozerSimulation().controller) is Control
    sim = DozerSimulation(controller_type="pi")
    assert type(sim.controller) is Control
    assert sim.controller.controller_type == "pi"
    sim._blade_update(deform=False)
    np.testing.assert_array_equal(sim.controller.integral_error, 0.)
    with pytest.raises(ValueError):
        DozerSimulation(controller_type="unknown")


# Pi tuning
def test_pi_grid_search_uses_fresh_simulations_and_actual_gains(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    trials = []
    class Simulation:
        def __init__(self, blade_roll_pitch_yaw: FloatArray, controller_type: str) -> None:
            assert controller_type == "pi"
            np.testing.assert_array_equal(blade_roll_pitch_yaw, 0.)
            self.controller = type("Controller", (), {"Kp": np.zeros(3), "Ki": np.zeros(3)})()
            self.total_distance = 1.
            self.cut_limit_reached = False
            trials.append(self)
        def run(self) -> None:
            assert np.isinf(self.stop_distance)
            assert self.controller.desired_depth == 0.3
            assert self.stop_time == 1.0
            self.blade_depth_rmse = self.controller.Kp[1] + 10 * self.controller.Ki[1]
            self.force_log = [{"time": self.stop_time}]
    monkeypatch.setattr(tuning, "DozerSimulation", Simulation)
    result = tuning.Tuning("pi").grid_search([1., 2.], [0., 0.1], duration=1., output_dir=tmp_path)
    np.testing.assert_allclose(result, [[1., 2.], [2., 3.]])
    assert len(trials) == 4
    assert (tmp_path / "rmse_heatmap.png").is_file()
    with (tmp_path / "results.csv").open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["kp", "ki", "rmse_m", "desired_depth_m",
                                    "duration_s", "distance_m", "cut_limit_reached"]
        rows = list(reader)
        np.testing.assert_allclose([float(row["rmse_m"]) for row in rows], result.ravel())
        np.testing.assert_allclose([float(row["kp"]) for row in rows], [1., 2., 1., 2.])
        np.testing.assert_allclose([float(row["ki"]) for row in rows], [0., 0., .1, .1])


# Visualization
@pytest.mark.parametrize("offset", ([0.5, 0, 0], [0, 0, 0], [0.4, -0.3, 8.0]))
def test_offset_blade_renders_without_arm_overlay(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, offset: Sequence[float]) -> None:
    simulation = DozerSimulation(blade_roll_pitch_yaw=np.array([0.2, 0.1, -0.3]))
    simulation.blade_arm_offset = np.array(offset)
    simulation.log = []
    simulation.blade_log = []
    simulation.grid_log = []
    simulation.neighbor_log = []
    for i in range(3):
        simulation.q = np.array([0.1 * i, 0.2 * i, 0.3, 0.1 * i, -0.05 * i, 0.4 * i])
        simulation.log.append(np.r_[i * simulation.dt, simulation.q])
        blade, _ = simulation._blade_update(deform=False)
        simulation.blade_log.append(blade)
        simulation.grid_log.append(simulation._grid_heights())
        simulation.neighbor_log.append([])
    captured = {}
    real_close = visualization_module.plt.close
    monkeypatch.setattr(visualization_module.plt, "close", lambda fig: captured.setdefault("figure", fig))
    monkeypatch.chdir(tmp_path)
    try:
        visualization_module.Visualization(simulation).visualization()
        figure = captured["figure"]
        blade = np.asarray(simulation.blade_log[-1])
        top = blade[[0, -1], :3] + simulation._blade_rotation_lg(blade[0, 3:6])[:, 2] * simulation.H
        expected = np.vstack([blade[0, :3], blade[-1, :3], top[-1], top[0]])
        projections = {("X (m)", "Y (m)"): [0, 1],
                       ("Y (m)", "Z (m)"): [1, 2],
                       ("X (m)", "Z (m)"): [0, 2]}
        for axis in figure.axes:
            assert all(line.get_label() != "offset blade arm" for line in axis.lines)
            if axis.name == "3d":
                continue
            projection = projections[(axis.get_xlabel(), axis.get_ylabel())]
            face, = axis.patches
            np.testing.assert_allclose(face.get_xy()[:4], expected[:, projection])
        assert (tmp_path / "figures" / "simulation.gif").is_file()
    finally:
        if "figure" in captured:
            real_close(captured["figure"])


def test_body_only_render_does_not_need_arm_offset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    simulation = DozerSimulation(enable_blade=False)
    del simulation.blade_arm_offset
    captured = {}
    real_close = visualization_module.plt.close
    monkeypatch.setattr(visualization_module.plt, "close", lambda fig: captured.setdefault("figure", fig))
    monkeypatch.chdir(tmp_path)
    try:
        visualization_module.Visualization(simulation).visualization()
        assert all(line.get_label() != "offset blade arm"
                   for axis in captured["figure"].axes for line in axis.lines)
        assert (tmp_path / "figures" / "simulation.gif").is_file()
    finally:
        if "figure" in captured:
            real_close(captured["figure"])


# Soil spill-over while turning
@pytest.mark.parametrize("speed_ratio, expected", [
    (0.0, 1.0), (0.5, 0.5), (-0.5, 0.5),
    (1.0, 0.0), (-1.0, 0.0), (1.5, 0.0), (-1.5, 0.0),
])
def test_soil_spill_scales_stored_fill(speed_ratio: float, expected: float) -> None:
    sim = DozerSimulation()
    angles = np.array([0.05, 0.1, 0.0])
    sim.blade_roll_pitch_yaw = angles
    interact = sim._blade_terrain_interaction
    interact()
    empty_load = np.array([sim.Fb, sim.Mb])
    sim.total_distance = 100.0
    sim.fill_progress_distance = sim.fill_distance
    interact()
    full_load = np.array([sim.Fb, sim.Mb])
    full_heights = np.array([sim.H3, sim.H4])
    sim.dt = sim.spill_reference_seconds
    sim.v[1] = speed_ratio * sim.angular_velocity_limit
    sim._update_fill_progress(0.0)
    assert sim.fill_progress_distance == pytest.approx(sim.fill_distance * expected)
    assert sim.total_distance == 100.0
    expected_load = empty_load + (full_load - empty_load) * expected
    for _ in range(2):
        interact()
        np.testing.assert_allclose([sim.Fb, sim.Mb], expected_load)
        np.testing.assert_allclose([sim.H3, sim.H4], full_heights * expected)
    # Straightening does not restore spilled soil; additional travel refills it.
    sim.v[1] = 0.0
    sim._update_fill_progress(0.0)
    assert sim.fill_progress_distance == pytest.approx(sim.fill_distance * expected)
    sim._update_fill_progress(sim.fill_distance * 2)
    assert sim.fill_progress_distance == sim.fill_distance


@pytest.mark.parametrize("dt", [0.01, 0.1, 0.25])
def test_soil_spill_retention_over_reference_duration(dt: float) -> None:
    sim = DozerSimulation()
    sim.dt = dt
    sim.spill_reference_seconds = 2.0
    sim.fill_progress_distance = sim.fill_distance
    sim.v[1] = sim.angular_velocity_limit / 2
    for _ in range(round(sim.spill_reference_seconds / dt)):
        sim._update_fill_progress(0.0)
    assert sim.fill_progress_distance == pytest.approx(sim.fill_distance / 2)


def test_soil_fill_caps_before_spill_and_retains_when_turning_disabled() -> None:
    sim = DozerSimulation()
    sim.dt = sim.spill_reference_seconds
    sim.fill_progress_distance = sim.fill_distance
    sim.v[1] = sim.angular_velocity_limit / 2
    sim._update_fill_progress(100.0)
    assert sim.fill_progress_distance == pytest.approx(sim.fill_distance / 2)
    sim.angular_velocity_limit = 0.0
    sim._update_fill_progress(0.0)
    assert sim.fill_progress_distance == pytest.approx(sim.fill_distance / 2)


@pytest.mark.parametrize("backwards", [False, True])
def test_soil_spill_preserves_cut_depth_stop(backwards: bool) -> None:
    sim = DozerSimulation()
    sim.is_backwards = backwards
    sim.cut_limit_reached = True
    sim.v[1] = sim.angular_velocity_limit
    sim._blade_terrain_interaction()
    assert sim.Fb == (1e9 if backwards else -1e9)
    assert sim.Mb == 0.0


def test_simulation_run_advances_fill_once_per_step() -> None:
    sim = DozerSimulation()
    sim.stop_time = 0.04
    updates = []
    update = sim._update_fill_progress

    def record_update(distance_step: Scalar) -> None:
        updates.append(distance_step)
        update(distance_step)

    sim._update_fill_progress = record_update
    sim.run()
    assert len(updates) == round(sim.stop_time / sim.dt)
    assert sum(updates) == pytest.approx(sim.total_distance)
    assert sim.total_distance > 0.0
    assert 0.0 <= sim.fill_progress_distance <= min(sim.fill_distance, sim.total_distance)


@pytest.mark.parametrize("mode", ["pd", "pi"])
def test_fixed_rolled_blade_bypasses_control(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    angles = np.array([0.05, 0.0, 0.0])

    def unexpected_control(*args: Any) -> NoReturn:
        pytest.fail("Fixed blade must not invoke its controller")

    sim = DozerSimulation(blade_roll_pitch_yaw=angles.copy(),
                          controller_type=mode, enable_blade_control=False)
    monkeypatch.setattr(sim.controller, "proportional_blade_controller", unexpected_control)
    cuts = []
    deform = sim._deform_blade_tiles

    def record_cut(contacts: ContactMap) -> None:
        cuts.append(contacts)
        return deform(contacts)

    monkeypatch.setattr(sim, "_deform_blade_tiles", record_cut)
    sim.stop_time = 0.03
    sim.run()
    np.testing.assert_array_equal(sim.blade_roll_pitch_yaw, angles)
    assert cuts
    np.testing.assert_array_equal(sim.requested_blade_rates, 0.0)
    assert sim.controller.previous_blade_error is None
    if mode == "pi":
        np.testing.assert_array_equal(sim.controller.integral_error, 0.0)


# Soil pile visualization
@pytest.mark.parametrize("roll", [0.0, 0.05, -0.05])
def test_pile_geometry_follows_blade_and_snapshots_are_independent(roll: float) -> None:
    sim = DozerSimulation(enable_blade_control=False)
    assert sim.pile_log[0][:2] == (0.0, 0.0)
    sim.H3, sim.H4 = 2.0, 2.0  # Must be capped at the exposed blade height.
    sim.q[:] = [1., 2., 0., 0., 0., 0.4]
    sim.blade_roll_pitch_yaw = np.array([roll, np.arcsin(0.2 / sim.L), -0.2])
    vertices = Visualization(sim)._pile_vertices(
        sim._blade_update(deform=False)[0],
        (sim.H3, sim.H4, *sim.blade_roll_pitch_yaw[[0, 2]]))
    assert vertices.shape[0] > 6 and vertices.shape[1] == 3
    for base, crest, toe in vertices.reshape(-1, 3, 3):
        assert base[2] == pytest.approx(sim._undeformed_height(base), abs=1e-10)
        assert toe[2] == pytest.approx(sim._undeformed_height(toe), abs=1e-10)
        assert crest[2] > base[2]
    local = (vertices - sim._blade_bottom_center()) @ sim._blade_rotation_lg(sim.q[3:6])
    np.testing.assert_allclose(local.reshape(-1, 3, 3)[:, :2, 0], 0., atol=1e-10)
    np.testing.assert_allclose(local[1::3, 2], sim.H, atol=1e-10)
    assert np.all(local[::3, 2] > 0.)  # Base is above the buried cutting edge.
    saved = vertices.copy()
    sim.H3 = 0.8
    sim.q[0] += 1.
    np.testing.assert_array_equal(vertices, saved)
    sim.H3 = sim.H4 = 0.
    assert Visualization(sim)._pile_vertices(
        sim._blade_update(deform=False)[0],
        (sim.H3, sim.H4, *sim.blade_roll_pitch_yaw[[0, 2]])).shape == (0, 3)


def test_pile_hidden_when_blade_is_buried() -> None:
    sim = DozerSimulation(enable_blade_control=False)
    sim.H3 = sim.H4 = 0.5
    sim.q[2] = -2.0
    assert Visualization(sim)._pile_vertices(
        sim._blade_update(deform=False)[0],
        (sim.H3, sim.H4, *sim.blade_roll_pitch_yaw[[0, 2]])).shape == (0, 3)


def test_pile_renders_in_all_views(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sim = DozerSimulation(blade_roll_pitch_yaw=np.array([0.05, 0., 0.]),
                          enable_blade_control=False)
    sim.fill_distance = 0.01
    sim.stop_time = 0.04
    sim.run()
    assert len(sim.pile_log) == len(sim.log)
    assert sim.pile_log[0][:2] == (0.0, 0.0)
    assert len(sim.pile_log[-1]) == 4
    assert max(sim.pile_log[-1][:2]) > 0.0
    captured = []
    real_close = visualization_module.plt.close
    monkeypatch.setattr(visualization_module.plt, "close", lambda fig: captured.append(fig))
    monkeypatch.chdir(tmp_path)
    try:
        Visualization(sim).visualization()
        figure = captured[-1]
        for axis in figure.axes:
            pile, = [artist for artist in axis.collections if artist.get_label() == "soil pile"]
            assert len(pile.get_paths()) > 8
            tooth, = [artist for artist in axis.collections if artist.get_label() == "surface tooth"]
            assert len(tooth.get_paths()) > 0
            np.testing.assert_allclose(np.asarray(tooth.get_facecolor())[0],
                                       visualization_module.to_rgba("burlywood"))
        with Image.open(tmp_path / "figures" / "simulation.gif") as gif:
            assert gif.n_frames == 3
    finally:
        for figure in captured:
            real_close(figure)


def test_pile_visualizer_uses_logged_state_after_simulation_changes() -> None:
    sim = DozerSimulation(enable_blade_control=False)
    sim.blade_roll_pitch_yaw = np.array([0.05, np.arcsin(0.2 / sim.L), 0.1])
    blade = sim._blade_update(deform=False)[0]
    state = (0.3, 0.2, 0.05, 0.1)
    visualizer = Visualization(sim)
    expected = visualizer._pile_vertices(blade, state)
    assert expected.shape[0] > 6 and expected.shape[1] == 3
    sim.q[:] = 10.
    sim.blade_roll_pitch_yaw[:] = 0.
    sim.H3 = sim.H4 = 0.
    np.testing.assert_array_equal(visualizer._pile_vertices(blade, state), expected)


def test_surface_tooth_connects_blade_to_soil_without_pile() -> None:
    sim = DozerSimulation(enable_blade_control=False)
    visualizer = Visualization(sim)
    blade = np.array([[0.1, -0.5, -0.2, 0., 0., 0.],
                      [0.1, 0., -0.2, 0., 0., 0.],
                      [0.1, 0.5, -0.2, 0., 0., 0.]])
    heights = np.zeros_like(sim.starting_grid_heights)
    faces = np.asarray(visualizer._surface_tooth_faces(blade, heights))
    assert faces.shape == (4, 3, 3)
    np.testing.assert_allclose(faces[::2, 0], blade[:-1, :3])
    np.testing.assert_allclose(faces[::2, 1:, 2], 0.)
    assert np.all(faces[::2, 1:, 0] > 0.1)
    assert visualizer._pile_faces(np.empty((0, 3))) == []
    lowered = np.asarray(visualizer._surface_tooth_faces(blade, heights - 0.3))
    np.testing.assert_allclose(lowered[:, :, 2], -0.3)


@pytest.mark.parametrize("yaw", [0., 0.6, 1.8])
def test_pile_drawing_order_follows_camera_side(yaw: float) -> None:
    sim = DozerSimulation(enable_blade_control=False)
    rotation = sim._rotation_lg(0.1, 0.2, yaw)
    face = np.array([[1., -1., 0.], [1., 1., 0.],
                     [1., 1., 1.], [1., -1., 1.]]) @ rotation.T
    body_center = rotation @ np.array([0., 0., 0.5])
    front_eye = rotation @ np.array([5., 0., 2.])
    rear_eye = rotation @ np.array([-5., 0., 2.])
    assert Visualization._pile_in_front_of_blade(face, front_eye, body_center)
    assert not Visualization._pile_in_front_of_blade(face, rear_eye, body_center)


def test_pile_contact_boundary_grows_continuously() -> None:
    sim = DozerSimulation(enable_blade_control=False)
    visualizer = Visualization(sim)
    sim.blade_roll_pitch_yaw = np.array([0.05, 0., 0.])
    state = (0.8, 0.8, 0.05, 0.)

    def geometry(body_height: float) -> FloatArray:
        sim.q[:] = [0., 0., body_height, 0., 0., 0.]
        blade = sim._blade_update(deform=False)[0]
        return visualizer._pile_vertices(blade, state)

    before = geometry(1e-6)
    after = geometry(-1e-6)
    assert before.shape == after.shape
    assert np.max(np.abs(after - before)) < 1e-4
    strips = after.reshape(-1, 3, 3)
    heights = np.linalg.norm(strips[:, 1] - strips[:, 0], axis=1)
    np.testing.assert_allclose(heights, 0.8)  # No sideways contact taper.
    # Ground contact must not suddenly reveal a previously hidden endpoint.
    end_height = -sim.B1 / 2 * np.sin(0.05)
    before = geometry(end_height + 1e-6)
    after = geometry(end_height - 1e-6)
    assert before.shape == after.shape
    assert np.max(np.abs(after - before)) < 1e-4
    faces = np.asarray(visualizer._pile_faces(after))
    assert faces.shape[1:] == (3, 3)
    assert len(faces) == 2 + 6 * (len(after) // 3 - 1)


def test_pile_grows_outward_across_full_blade_width() -> None:
    sim = DozerSimulation(enable_blade_control=False)
    sim.blade_roll_pitch_yaw = np.array([0.05, 0., 0.])
    blade = sim._blade_update(deform=False)[0]
    visualizer = Visualization(sim)
    small = visualizer._pile_vertices(blade, (0.1, 0.1, 0.05, 0.)).reshape(-1, 3, 3)
    large = visualizer._pile_vertices(blade, (0.2, 0.2, 0.05, 0.)).reshape(-1, 3, 3)
    np.testing.assert_allclose(small[:, 0], large[:, 0])
    small_reach = np.linalg.norm(small[:, 2, :2] - small[:, 0, :2], axis=1)
    large_reach = np.linalg.norm(large[:, 2, :2] - large[:, 0, :2], axis=1)
    assert np.all(small_reach > 0.)
    np.testing.assert_allclose(large_reach, 2 * small_reach)
    np.testing.assert_allclose(small[:, 0, 2], 0., atol=1e-10)


def test_control_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="controller_type"):
        Control(1.2, 0.1, 1.0, controller_type="pid")


def test_control_pd_does_not_accumulate_pi_state() -> None:
    controller = Control(1.2, 0.1, 1.0)
    for _ in range(3):
        controller.proportional_blade_controller(0., 0.)
        controller.apply_actuator_feedback(np.ones(3), np.zeros(3))
    np.testing.assert_array_equal(controller.integral_error, 0.)
    assert controller.controller_type == "pd"


# Pure pursuit and calibrated track-force mapping
@pytest.mark.parametrize("yaw", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("tilted", [False, True])
def test_body_update_uses_path_controller(yaw: float, tilted: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = DozerSimulation(use_path_controller=True, enable_blade_control=False)
    rotation = sim._rotation_lg(0.1, -0.05, 0.2) if tilted else np.eye(3)
    sim.controller.configure_path(surface_rotation=rotation)
    sim.q[:] = [1., 0.3, 0., 0.1, -0.05, yaw]
    control = Control(sim.L, sim.dt, 1.)
    control.configure_path(surface_rotation=rotation)
    expected = control.angular_path_controller(sim.q, sim._rotation_lg(*sim.q[3:6]), sim.F_track_base)
    monkeypatch.setattr(sim, "_update_q_dot", lambda: None)
    monkeypatch.setattr(sim, "_settle_tracks", lambda orientation: None)
    sim._body_update()
    np.testing.assert_allclose(sim.F_track, expected)
    assert sim.controller.heading_err == pytest.approx(control.heading_err)
    assert sim.controller.cross_track_err == pytest.approx(control.cross_track_err)


def test_path_mapping_is_bounded_and_symmetric() -> None:
    control = Control(1.2, .01, 1.)
    assert control.track_force_fraction(0.) == pytest.approx(.799)
    values = [control.track_force_fraction(angle) for angle in np.linspace(0., np.pi, 30)]
    assert np.all(np.diff(values) <= 0.)
    assert min(values) >= 0. and max(values) <= 1.
    assert control.track_force_fraction(-1.) == control.track_force_fraction(1.)
    assert control.track_force_fraction(20.) == control.track_force_fraction(control._4pl_ang_max)


@pytest.mark.parametrize("heading, weakened", [(-0.4, 0), (0.4, 1)])
def test_path_controller_turn_direction_and_lookahead(heading: float, weakened: int) -> None:
    control = Control(1.2, .01, 1.)
    control.configure_path([[0., 0.], [10., 0.], [10., 10.], [0., 10.]], lookahead_dist=1.)
    pose = np.array([2., 0., 0., 0., 0., heading])
    rotation = DozerSimulation()._rotation_lg(0., 0., heading)
    forces = control.angular_path_controller(pose, rotation, 100.)
    np.testing.assert_allclose(control.lookahead_point, [3., 0., 0.])
    assert forces[weakened] < 100.
    assert forces[1 - weakened] == 100.


def test_path_repeated_waypoints_and_far_fallback() -> None:
    control = Control(1.2, .01, 1.)
    control.configure_path([[0., 0.], [0., 0.], [10., 0.], [10., 10.]], lookahead_dist=1.)
    assert np.isfinite(control.pure_pursuit_heading_error(np.zeros(6), np.eye(3)))
    assert np.isfinite(control.pure_pursuit_heading_error([100., 100., 0.], np.eye(3)))
    np.testing.assert_array_equal(control.lookahead_point, [10., 10., 0.])


@pytest.mark.parametrize("points", [[], [[0., 0.]], [[0., 0.]] * 3,
                                    [[0., 0.], [1., 0.], [np.nan, 1.]]])
def test_path_rejects_invalid_references(points: ArrayLike) -> None:
    with pytest.raises(ValueError):
        Control(1.2, .01, 1.).configure_path(points)


def test_path_progress_resets_with_reference() -> None:
    control = Control(1.2, .01, 1.)
    for index in (1950, 5):
        pose = np.r_[control.path_points[index], np.zeros(3)]
        control.angular_path_controller(pose, np.eye(3))
    assert control.path_complete
    control.configure_path()
    assert not control.path_complete
    assert control._nearest_path_idx == 0


@pytest.mark.parametrize("mode", ["pd", "pi"])
def test_path_surface_run_and_reference_rendering(mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sim = DozerSimulation(controller_type=mode, use_path_controller=True,
                          enable_blade_control=False, lookahead_dist=.9)
    assert np.isinf(sim.stop_distance)
    assert sim.us[0] < sim.controller.path_points[:, 0].min()
    assert sim.vs[0] < sim.controller.path_points[:, 1].min()
    sim.stop_time = .04
    sim.run()
    assert len(sim.log) == len(sim.force_log) == 5
    assert any(row["drive_left"] != row["drive_right"] for row in sim.force_log)
    assert np.all(np.isfinite([row["heading_error"] for row in sim.force_log]))
    assert abs(sim.v[1]) <= sim.angular_velocity_limit
    captured = []
    real_close = visualization_module.plt.close
    monkeypatch.setattr(visualization_module.plt, "close", lambda fig: captured.append(fig))
    monkeypatch.chdir(tmp_path)
    try:
        Visualization(sim).visualization()
        for axis in captured[-1].axes:
            assert any(artist.get_label() == "reference path"
                       for artist in list(axis.lines) + list(axis.collections))
    finally:
        for figure in captured:
            real_close(figure)
