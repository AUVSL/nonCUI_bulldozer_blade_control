import numpy as np
import pytest

from util import Blade, Body, _SurfaceBase


SHARED_METHODS = (
    "subdivision",
    "_surface_grid",
    "_rotation_lg",
    "_rotation_gl",
    "_point_orientation",
    "_bilinear_gradient",
    "_point_height",
    "_grid_cell",
    "_get_neighbor_points",
    "_bilinear_height",
    "_contact_angle",
)


def _grid_heights(surface):
    return np.array([[point[2] for point in column] for column in surface.grid_pts])


def test_body_and_blade_inherit_the_shared_surface_implementation():
    assert issubclass(Body, _SurfaceBase)
    assert issubclass(Blade, _SurfaceBase)
    for method_name in SHARED_METHODS:
        assert method_name in _SurfaceBase.__dict__
        assert method_name not in Body.__dict__
        assert method_name not in Blade.__dict__
        assert getattr(Body, method_name) is getattr(Blade, method_name)


def test_subdivision_preserves_each_class_output():
    body = Body()
    blade = Blade()

    assert body.subdivision == pytest.approx(1.75 / 2)
    assert blade.subdivision == pytest.approx(1.75 / 8)

    # Body's old property always used a fixed factor of two, while Blade's
    # implementation used the mutable division_factor attribute.
    body.division_factor = blade.division_factor = 4
    assert body.subdivision == pytest.approx(1.75 / 2)
    assert blade.subdivision == pytest.approx(1.75 / 4)


def test_body_grid_and_initial_logs_keep_their_previous_shapes_and_values():
    body = Body()
    heights = _grid_heights(body)

    assert heights.shape == (3, 11)
    assert heights.max() == pytest.approx(2.625)
    assert heights.sum() == pytest.approx(38.5)
    assert body.q[3] == pytest.approx(-0.4636476080008061)
    assert not body.is_initalization
    assert np.asarray(body.log).shape == (1, 7)
    assert np.asarray(body.point_log).shape == (1, 6, 6)
    assert np.asarray(body.neighbor_log).shape == (1, 28, 3)
    assert all(
        node["visited_last"] is False
        for _, node in body.surf_grid.nodes(data=True)
    )


def test_body_grid_keeps_two_axis_ramp_and_sigmoid_node_metadata():
    unrolled = Body(is_surface_rolled=False)
    rolled = Body(is_surface_rolled=True)
    np.testing.assert_allclose(_grid_heights(unrolled), _grid_heights(rolled))

    unrolled.is_surface_sigmoid = True
    sigmoid_grid = unrolled._surface_grid()
    for _, node in sigmoid_grid.nodes(data=True):
        assert node["z"] == pytest.approx(
            np.sin(node["x"] / 3) * np.cos(node["y"] * 4)
        )
        assert node["visited_last"] is False


def test_shared_orientation_uses_blade_default_and_body_roll_hook():
    body = Body(is_uphill=False)
    blade = Blade(is_uphill=False)
    corners = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.2),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.2),
    ]
    body.q[:2] = blade.q[:2] = [0.5, 0.5]

    body_orientation = body._point_orientation(corners)
    blade_orientation = blade._point_orientation(corners)

    assert body_orientation[0] == pytest.approx(np.arctan2(-0.2, 1.0))
    assert blade_orientation[0] == pytest.approx(0.0)
    np.testing.assert_allclose(body_orientation[1:], blade_orientation[1:])


def test_body_update_keeps_return_contract_and_advances_position():
    body = Body()
    previous_xy = body.q[:2].copy()

    points, neighbor_groups = body._body_update()

    np.testing.assert_allclose(
        body.q[:2], previous_xy + body.dt * body.q_dot[:2]
    )
    assert len(points) == 7
    assert len(neighbor_groups) == 7
    assert all(np.asarray(point).shape == (6,) for point in points)
    assert all(np.asarray(group).shape == (4, 3) for group in neighbor_groups)

    rotation = body._rotation_lg(*body.q[3:6])
    half_width = rotation @ np.array([0.0, body.b / 2, 0.0])
    half_track = rotation @ np.array([body.l / 2, 0.0, 0.0])
    right_center = body.q[:3] - half_width
    left_center = body.q[:3] + half_width
    expected_xyz = [
        right_center + half_track,
        right_center,
        right_center - half_track,
        left_center + half_track,
        left_center,
        left_center - half_track,
    ]
    np.testing.assert_allclose(points[0], body.q)
    np.testing.assert_allclose(neighbor_groups[0], body._get_neighbor_points(body.q))
    for point, xyz, neighbors in zip(points[1:], expected_xyz, neighbor_groups[1:]):
        np.testing.assert_allclose(point[:3], xyz)
        np.testing.assert_allclose(point[3:], body.q[3:6])
        np.testing.assert_allclose(neighbors, body._get_neighbor_points(xyz))

    orient = body.q[3:6].copy()
    contact_pitch = body._contact_pitch(orient)
    contact_rotation = body._rotation_lg(orient[0], contact_pitch, orient[2])
    front_height, back_height = body._half_resting_heights(contact_rotation)
    assert front_height - back_height == pytest.approx(0.0, abs=body.contact_tol)
    assert Body._contact_pitch is not Blade._contact_pitch


def test_body_short_run_keeps_logs_aligned():
    body = Body()
    body.stop_time = 3 * body.dt + 1e-12

    body._run()

    assert len(body.log) == len(body.point_log) == len(body.neighbor_log) == 4
    assert np.asarray(body.log).shape == (4, 7)
    assert np.asarray(body.point_log).shape == (4, 6, 6)
    assert np.asarray(body.neighbor_log).shape == (4, 28, 3)
