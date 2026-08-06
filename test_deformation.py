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
