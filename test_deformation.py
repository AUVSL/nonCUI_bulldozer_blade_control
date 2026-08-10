import numpy as np
import pytest

from deformation import Surface


def _reached_contact(surface, tile=(1, 1), depth=-10.0):
    """A contact near the tile's leading edge for the default +Y motion."""
    i, j = tile
    x0, y0, _ = surface.grid_pts[i][j]
    x1, y1, _ = surface.grid_pts[i + 1][j + 1]
    return np.array([(x0 + x1) / 2, y1 - surface.subdivision * 0.01, depth])


def test_duplicate_blade_contacts_cut_a_tile_once_per_frame():
    surface = Surface(is_uphill=False)
    surface.max_dig_depth_per_meter = 2.0
    surface.q_dot[:3] = [0.0, 10.0, 0.0]  # 0.1 m per step
    contact = _reached_contact(surface)
    before = surface.grid_pts[1][2][2]

    surface._deform_blade_tiles({(1, 1): [contact, contact.copy()]})

    assert surface.grid_pts[1][2][2] < before


def test_dig_depth_depends_on_distance_not_step_count():
    one_step = Surface(is_uphill=False)
    split_steps = Surface(is_uphill=False)
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


def test_initial_blade_geometry_does_not_deform_the_surface():
    surface = Surface(is_uphill=False)

    np.testing.assert_allclose(surface._grid_heights(), surface.grid_log[0])


def test_only_forward_vertices_of_a_contacted_tile_deform():
    surface = Surface(is_uphill=False)
    surface.q_dot[:3] = [0.0, 10.0, 0.0]
    contact = _reached_contact(surface)
    before = surface._grid_heights()

    surface._deform_blade_tiles({(1, 1): [contact]})

    # +Y motion: j=2 is directly cut; j=1 remains untouched.
    assert surface.grid_pts[1][1][2] == pytest.approx(before[1, 1])
    assert surface.grid_pts[2][1][2] == pytest.approx(before[2, 1])
    assert surface.grid_pts[1][2][2] < before[1, 2]
    assert surface.grid_pts[2][2][2] < before[2, 2]


def test_repeated_cuts_stop_at_each_vertex_starting_height_limit():
    surface = Surface(is_uphill=False)
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


def test_blade_stops_at_maximum_soil_deformation_depth():
    surface = Surface(is_uphill=False)
    surface.max_world_cut_depth = 0.1
    surface.q[2] = -10.0

    blade_points, _ = surface._blade_update(deform=False)

    for point in blade_points:
        i, j = surface._grid_cell(point)
        starting_corners = (
            (*surface.grid_pts[i][j][:2], surface.starting_grid_heights[i, j]),
            (*surface.grid_pts[i + 1][j][:2], surface.starting_grid_heights[i + 1, j]),
            (*surface.grid_pts[i][j + 1][:2], surface.starting_grid_heights[i, j + 1]),
            (*surface.grid_pts[i + 1][j + 1][:2], surface.starting_grid_heights[i + 1, j + 1]),
        )
        minimum_z = surface._bilinear_height(point, starting_corners) - surface.max_world_cut_depth
        assert point[2] >= minimum_z - 1e-12


def test_body_contact_updates_only_pitch_on_forward_slope():
    surface = Surface(is_uphill=False)
    slope = 0.2
    for i in range(len(surface.us)):
        for j in range(len(surface.vs)):
            x, y, _ = surface.grid_pts[i][j]
            z = slope * y
            surface.grid_pts[i][j] = (x, y, z)
            surface.surf_grid.nodes[(i, j)]["z"] = z
    surface.starting_grid_heights = surface._grid_heights().copy()
    surface.is_initalization = True
    surface.q[3] = 0.17
    surface.q[5] = np.pi / 2

    surface._body_update()

    assert surface.q[2] == pytest.approx(slope * surface.q[1])
    assert surface.q[4] != pytest.approx(0.0)
    assert surface.q[3] == pytest.approx(0.17)
    assert surface.q[5] == pytest.approx(np.pi / 2)


def test_blade_keeps_local_offset_on_forward_slope():
    surface = Surface(is_uphill=False)
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
        local_midpoint, [surface.L, 0.0, -surface.starting_vertical_cut_offset], atol=1e-12
    )
    assert midpoint[2] == pytest.approx(
        surface._point_height(midpoint)[1] - surface.starting_vertical_cut_offset,
        abs=1e-6,
    )


def test_blade_descends_with_q_after_surface_is_cut():
    surface = Surface(is_uphill=False)
    surface.max_world_cut_depth = 1.0
    before, _ = surface._blade_update(deform=False)

    surface.q[2] -= 0.1
    after, _ = surface._blade_update(deform=False)

    np.testing.assert_allclose(
        np.asarray(after)[:, 2], np.asarray(before)[:, 2] - 0.1
    )


def test_blade_midpoint_keeps_local_offsets_from_q_until_depth_limit():
    surface = Surface(is_uphill=False)
    surface.max_world_cut_depth = 10.0
    surface.q[2] = -0.3
    surface.q[4] = 0.4

    blade, _ = surface._blade_update(deform=False)
    blade = np.asarray(blade)
    midpoint = (blade[0, :3] + blade[-1, :3]) / 2

    local_midpoint = surface._rotation_gl(*surface.q[3:6]) @ (midpoint - surface.q[:3])
    np.testing.assert_allclose(
        local_midpoint, [surface.L, 0.0, -surface.starting_vertical_cut_offset], atol=1e-12
    )


def test_blade_vertical_motion_stops_at_maximum_cut_depth():
    surface = Surface(is_uphill=False)
    surface.max_world_cut_depth = 0.25
    surface.q[2] = -1.0

    before, _ = surface._blade_update(deform=False)
    surface.q[2] -= 0.2
    after, _ = surface._blade_update(deform=False)

    np.testing.assert_allclose(np.asarray(after)[:, 2], np.asarray(before)[:, 2])


def test_normal_run_deepens_with_q_but_stops_at_maximum_depth():
    surface = Surface(is_uphill=False)

    surface._run()

    depths = surface.starting_grid_heights - surface._grid_heights()
    assert depths.max() > surface.starting_vertical_cut_offset
    assert depths.max() <= surface.max_world_cut_depth + 2e-7


def test_unrolled_transition_and_cut_are_uniform_across_blade_width():
    surface = Surface(is_uphill=False, is_surface_rolled=False)
    np.testing.assert_allclose(
        np.ptp(surface.starting_grid_heights, axis=0), 0.0, atol=1e-12
    )

    surface._run()

    depths = surface.starting_grid_heights - surface._grid_heights()
    cut_columns = np.flatnonzero(depths.max(axis=0) > 1e-9)
    for j in cut_columns:
        np.testing.assert_allclose(depths[:, j], depths[0, j], atol=2e-7)


def test_contact_pitch_uses_cut_floor_pitch_after_cutoff_depth():
    surface = Surface(is_uphill=False)
    cutoff_q_depth = (
        surface.max_world_cut_depth - surface.starting_vertical_cut_offset
    )
    surface.grid_pts = [
        [(x, y, z - cutoff_q_depth) for x, y, z in column]
        for column in surface.grid_pts
    ]
    for i, column in enumerate(surface.grid_pts):
        for j, (_, _, z) in enumerate(column):
            surface.surf_grid.nodes[(i, j)]["z"] = z
    surface.q[2] = surface._point_height(surface.q)[1]
    surface_pitch = surface._point_orientation(surface._get_neighbor_points(surface.q))[1]
    orient = surface.q[3:6].copy()
    orient[1] = surface_pitch

    R = surface._rotation_lg(*orient)
    blade_midpoint = surface.q[:3] + R @ np.array(
        [surface.L, 0.0, -surface.starting_vertical_cut_offset]
    )
    assert surface._contact_pitch(orient) == pytest.approx(
        surface._starting_pitch(blade_midpoint)
    )


def test_rolled_soil_is_never_cut_below_the_blade_plane():
    surface = Surface(is_uphill=False, is_surface_rolled=True)
    surface.q_dot[:3] = [0.0, 10.0, 0.0]
    contact = _reached_contact(surface, tile=(4, 1), depth=-0.3)

    surface._deform_blade_tiles({(4, 1): [contact]})

    for i, j in ((4, 2), (5, 2)):
        assert surface.grid_pts[i][j][2] == pytest.approx(contact[2])


def test_rolled_soil_is_graded_flat_across_the_blade():
    surface = Surface(is_uphill=False, is_surface_rolled=True)

    surface._run()

    heights = surface._grid_heights()
    deformed = np.abs(heights - surface.starting_grid_heights) > 1e-9
    columns = np.flatnonzero(deformed.any(axis=0))
    for j in columns:
        np.testing.assert_allclose(heights[:, j], heights[0, j], atol=2e-7)
