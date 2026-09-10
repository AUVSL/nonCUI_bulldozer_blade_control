"""Blade contact constrains the vehicle pose before terrain is deformed."""
import numpy as np
import networkx as nx
import pytest
from util import DozerSimulation


@pytest.fixture
def simulation():
    sim = DozerSimulation(is_surface_pitched=True, blade_roll_pitch_yaw=np.array([0., .2, 0.]))
    sim.division_factor = 7  # 0.25 m grid
    sim.us = np.arange(-3., 4.01, sim.subdivision)
    sim.vs = np.arange(-3., 3.01, sim.subdivision)
    sim.surf_grid = nx.grid_2d_graph(len(sim.us), len(sim.vs))
    sim.grid_pts = [[(x, y, 0.) for y in sim.vs] for x in sim.us]
    nx.set_node_attributes(sim.surf_grid, 0., "z")
    sim.starting_grid_heights = np.zeros((len(sim.us), len(sim.vs)))
    sim.q[:] = 0.
    sim.q_dot[:] = 0.
    sim.blade_roll_pitch_yaw[:] = 0.
    return sim


def assert_clearance(sim):
    rotation = sim._rotation_lg(*sim.q[3:6])
    edge = sim.q[:3] + sim._blade_edge_local(top=True) @ rotation.T
    for t in np.linspace(0., 1., 1001):
        point = edge[0] + t * (edge[1] - edge[0])
        assert point[2] >= sim._undeformed_height(point) - 1e-7
    # Tracks still cannot pass below the current flat surface.
    assert sim._track_xyz(sim.q)[:, :, 2].min() >= -1e-7


def test_blade_with_clearance_does_not_lift_body(simulation):
    simulation._settle_tracks(simulation.q[3:6].copy())
    np.testing.assert_allclose(simulation.q, 0., atol=1e-7)
    assert_clearance(simulation)


def test_original_soil_supports_blade_after_surface_is_cut(simulation):
    # Current surface is already cut to zero, original soil remains at 1.4 m.
    simulation.starting_grid_heights[:] = 1.4
    simulation._settle_tracks(simulation.q[3:6].copy())
    assert simulation.q[2] > 0.1
    assert simulation.q[4] < -0.1  # blade/front lifts; rear track ends support it
    assert abs(simulation.q[3]) < 1e-7
    assert abs(simulation._track_xyz(simulation.q)[:, -1, 2].min()) < 1e-7
    assert_clearance(simulation)


def test_asymmetric_soil_at_blade_changes_roll(simulation):
    for i, x in enumerate(simulation.us):
        for j, y in enumerate(simulation.vs):
            if x > 1.25 and y > .25:
                simulation.starting_grid_heights[i, j] = 1.4
    simulation._settle_tracks(simulation.q[3:6].copy())
    assert simulation.q[3] > .1
    assert simulation.q[5] == 0.  # contact does not steer the vehicle
    assert_clearance(simulation)


@pytest.mark.parametrize("blade_angles", [[.06, .3, .3], [-.06, .3, -.3]])
def test_rotated_offset_blade_clears_soil(simulation, blade_angles):
    simulation.blade_roll_pitch_yaw[:] = blade_angles
    simulation.blade_arm_offset = np.array([.7, .2, .1])
    simulation.starting_grid_heights[:] = 1.4
    simulation.q[5] = .4
    simulation._settle_tracks(simulation.q[3:6].copy())
    assert simulation.q[5] == .4
    assert_clearance(simulation)


def test_disabled_blade_does_not_support_vehicle(simulation):
    simulation.enable_blade = False
    simulation.starting_grid_heights[:] = 1.4
    simulation._settle_tracks(simulation.q[3:6].copy())
    np.testing.assert_allclose(simulation.q, 0., atol=1e-7)


def test_blade_pose_is_valid_before_cutting(simulation, monkeypatch):
    simulation.starting_grid_heights[:] = 1.4
    simulation.q_dot[0] = 1.
    called = []

    def deform(contacts):
        assert contacts
        assert_clearance(simulation)
        called.append(True)

    monkeypatch.setattr(simulation, "_deform_blade_tiles", deform)
    simulation._blade_update()
    assert called == [True]


def test_peak_between_blade_endpoints_supports_vehicle(simulation):
    # A narrow ridge in the middle is missed by endpoint-only contact tests.
    simulation.starting_grid_heights[:, len(simulation.vs) // 2] = 1.4
    simulation._settle_tracks(simulation.q[3:6].copy())
    assert simulation.q[2] > .1
    assert_clearance(simulation)


def test_bilinear_peak_inside_tile_is_included(simulation):
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
def test_track_cut_limit_before_blade_support(simulation, monkeypatch, offset, stopped):
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

    def blade_support(rotation):
        assert sim.cut_limit_reached == stopped
        return np.full((2, 2), 10.0)

    monkeypatch.setattr(sim, "_blade_support_heights", blade_support)
    sim._support_heights(R, check_cut_limit=True)
    sim._blade_terrain_interaction()
    assert (sim.Fb == -1e9) == stopped


def test_cut_limit_force_stops_and_holds_vehicle(simulation):
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


def test_disabled_blade_does_not_stop_at_cut_limit(simulation):
    sim = simulation
    sim.enable_blade = False
    sim.starting_grid_heights[:] = sim.max_world_cut_depth
    sim._settle_tracks(sim.q[3:6].copy())
    assert not sim.cut_limit_reached


def test_default_run_stops_at_cut_limit():
    sim = DozerSimulation(blade_roll_pitch_yaw=np.zeros(3))
    sim.stop_time = 5.0
    sim.run()
    assert sim.total_distance < sim.stop_distance
    assert sim.cut_limit_reached
    assert sim.Fb == -1e9
    assert sim.v[0] == 0.0


@pytest.mark.parametrize("roll", [-0.2, 0.0, 0.2])
def test_cut_limit_roll_depth_excludes_arm_lift(simulation, roll):
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
