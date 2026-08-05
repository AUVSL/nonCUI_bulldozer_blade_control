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
    contact = _reached_contact(surface)
    before = surface.grid_pts[1][1][2]

    surface._deform_blade_tiles({(1, 1): [contact, contact.copy()]}, 0.1)

    assert surface.grid_pts[1][1][2] == pytest.approx(before - 0.2)


def test_dig_depth_depends_on_distance_not_step_count():
    one_step = Surface(is_uphill=False)
    split_steps = Surface(is_uphill=False)
    for surface in (one_step, split_steps):
        surface.max_dig_depth_per_meter = 2.0

    one_contact = _reached_contact(one_step)
    split_contact = _reached_contact(split_steps)
    one_step._deform_blade_tiles({(1, 1): [one_contact]}, 0.1)
    split_steps._deform_blade_tiles({(1, 1): [split_contact]}, 0.05)
    split_steps._deform_blade_tiles({(1, 1): [split_contact]}, 0.05)

    assert split_steps.grid_pts[1][1][2] == pytest.approx(one_step.grid_pts[1][1][2])
