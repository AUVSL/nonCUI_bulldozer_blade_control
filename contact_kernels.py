"""Compiled terrain-contact kernels used by :mod:`main`.

The functions remain valid Python when Numba is unavailable, which keeps the
simulation importable in minimal environments. Installing ``requirements.txt``
enables cached machine-code compilation automatically.
"""
from __future__ import annotations

from typing import Callable, TypeVar

import numpy as np

F = TypeVar("F", bound=Callable[..., object])

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - only used in minimal installations
    NUMBA_AVAILABLE = False

    def njit(*args: object, **kwargs: object) -> Callable[[F], F]:
        def decorate(function: F) -> F:
            return function
        return decorate


@njit(cache=True)
def _grid_cell(x: float, y: float, u0: float, v0: float,
               spacing: float, ni: int, nj: int) -> tuple[int, int]:
    i = int(np.floor((x - u0) / spacing))
    j = int(np.floor((y - v0) / spacing))
    i = 0 if i < 0 else ni - 2 if i > ni - 2 else i
    j = 0 if j < 0 else nj - 2 if j > nj - 2 else j
    return i, j


@njit(cache=True)
def _height_from_coefficients(x: float, y: float, geometry: np.ndarray,
                              coefficients: np.ndarray, i: int, j: int) -> float:
    geom = geometry[i, j]
    dx, dy = x - geom[0], y - geom[1]
    s = (dx * geom[2] + dy * geom[3]) / geom[4]
    t = (dx * geom[5] + dy * geom[6]) / geom[7]
    s = 0.0 if s < 0.0 else 1.0 if s > 1.0 else s
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    c = coefficients[i, j]
    return c[0] + c[1] * s + c[2] * t + c[3] * s * t


@njit(cache=True)
def _current_height(x: float, y: float, geometry: np.ndarray,
                    grid_points: np.ndarray, i: int, j: int) -> float:
    """Interpolate mutable terrain without constructing Python corner lists."""
    geom = geometry[i, j]
    dx, dy = x - geom[0], y - geom[1]
    s = (dx * geom[2] + dy * geom[3]) / geom[4]
    t = (dx * geom[5] + dy * geom[6]) / geom[7]
    s = 0.0 if s < 0.0 else 1.0 if s > 1.0 else s
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    h00 = grid_points[i, j, 2]
    h10 = grid_points[i + 1, j, 2]
    h01 = grid_points[i, j + 1, 2]
    h11 = grid_points[i + 1, j + 1, 2]
    return (h00 + (h10 - h00) * s + (h01 - h00) * t
            + (h11 - h10 - h01 + h00) * s * t)


@njit(cache=True)
def track_support_heights(q: np.ndarray, rotation: np.ndarray, body_width: float,
                          track_length: float, us: np.ndarray, vs: np.ndarray,
                          spacing: float, geometry: np.ndarray,
                          grid_points: np.ndarray) -> np.ndarray:
    """Return [right, left] x [back, front] track support heights."""
    fwd = rotation[:, 0]
    lat = rotation[:, 1] * (body_width / 2.0)
    half_l = track_length / 2.0
    support = np.full((2, 2), -np.inf)
    max_samples = len(us) + len(vs) + 3

    for side_index in range(2):
        side = -1.0 if side_index == 0 else 1.0
        base = q[:3] + side * lat
        samples = np.empty(max_samples)
        count = 3
        samples[0], samples[1], samples[2] = -half_l, 0.0, half_l
        back = base - half_l * fwd
        front = base + half_l * fwd
        back_cell = _grid_cell(back[0], back[1], us[0], vs[0], spacing,
                               len(us), len(vs))
        front_cell = _grid_cell(front[0], front[1], us[0], vs[0], spacing,
                                len(us), len(vs))

        for axis in range(2):
            if abs(fwd[axis]) <= 1e-12:
                continue
            grid = us if axis == 0 else vs
            lo = min(back_cell[axis], front_cell[axis])
            hi = max(back_cell[axis], front_cell[axis])
            stop = min(hi + 2, len(grid))
            for grid_index in range(lo, stop):
                value = (grid[grid_index] - base[axis]) / fwd[axis]
                if -half_l < value < half_l:
                    duplicate = False
                    for k in range(count):
                        if samples[k] == value:
                            duplicate = True
                            break
                    if not duplicate:
                        samples[count] = value
                        count += 1

        for k in range(count):
            s = samples[k]
            point = base + s * fwd
            i, j = _grid_cell(point[0], point[1], us[0], vs[0], spacing,
                              len(us), len(vs))
            height = _current_height(point[0], point[1], geometry, grid_points, i, j)
            needed = height - point[2] + q[2]
            if s <= 0.0 and needed > support[side_index, 0]:
                support[side_index, 0] = needed
            if s >= 0.0 and needed > support[side_index, 1]:
                support[side_index, 1] = needed
    return support


@njit(cache=True)
def _blade_needed(t: float, start: np.ndarray, delta: np.ndarray,
                  edge_start_z: float, local_delta_z: float,
                  us: np.ndarray, vs: np.ndarray, spacing: float,
                  geometry: np.ndarray, coefficients: np.ndarray) -> float:
    x = start[0] + t * delta[0]
    y = start[1] + t * delta[1]
    i, j = _grid_cell(x, y, us[0], vs[0], spacing, len(us), len(vs))
    height = _height_from_coefficients(x, y, geometry, coefficients, i, j)
    return height - (edge_start_z + t * local_delta_z)


@njit(cache=True)
def blade_support_heights(q: np.ndarray, rotation: np.ndarray, edge: np.ndarray,
                          us: np.ndarray, vs: np.ndarray, spacing: float,
                          geometry: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Return blade-top support, reusing every shared interval endpoint."""
    local_delta = edge[1] - edge[0]
    rotated_start = np.empty(3)
    delta = np.empty(3)
    for row in range(3):
        rotated_start[row] = (rotation[row, 0] * edge[0, 0]
                              + rotation[row, 1] * edge[0, 1]
                              + rotation[row, 2] * edge[0, 2])
        delta[row] = (rotation[row, 0] * local_delta[0]
                      + rotation[row, 1] * local_delta[1]
                      + rotation[row, 2] * local_delta[2])
    start = q[:3] + rotated_start
    breaks = np.empty(len(us) + len(vs) + 3)
    breaks[0], breaks[1], breaks[2] = 0.0, 0.5, 1.0
    count = 3
    for axis in range(2):
        if abs(delta[axis]) <= 1e-12:
            continue
        grid = us if axis == 0 else vs
        lower = min(start[axis], start[axis] + delta[axis])
        upper = max(start[axis], start[axis] + delta[axis])
        for coordinate in grid:
            if lower < coordinate < upper:
                value = (coordinate - start[axis]) / delta[axis]
                duplicate = False
                for k in range(count):
                    if breaks[k] == value:
                        duplicate = True
                        break
                if not duplicate:
                    breaks[count] = value
                    count += 1
    ordered = np.sort(breaks[:count])

    support = np.full((2, 2), -np.inf)
    edge_start_z = rotated_start[2]
    local_delta_z = delta[2]
    left = _blade_needed(ordered[0], start, delta, edge_start_z,
                         local_delta_z, us, vs, spacing, geometry, coefficients)
    for k in range(count - 1):
        lo, hi = ordered[k], ordered[k + 1]
        middle_t = (lo + hi) / 2.0
        middle = _blade_needed(middle_t, start, delta, edge_start_z,
                               local_delta_z, us, vs, spacing, geometry, coefficients)
        right = _blade_needed(hi, start, delta, edge_start_z,
                              local_delta_z, us, vs, spacing, geometry, coefficients)
        maximum = max(left, right)
        quadratic = 2.0 * (left + right - 2.0 * middle)
        linear = right - left - quadratic
        if quadratic < -1e-12:
            fraction = -linear / (2.0 * quadratic)
            if 0.0 < fraction < 1.0:
                peak_t = lo + fraction * (hi - lo)
                peak = _blade_needed(peak_t, start, delta, edge_start_z,
                                     local_delta_z, us, vs, spacing,
                                     geometry, coefficients)
                maximum = max(maximum, peak)
        local_y = edge[0, 1] + middle_t * local_delta[1]
        if local_y <= 1e-12:
            support[0, 1] = max(support[0, 1], maximum)
        if local_y >= -1e-12:
            support[1, 1] = max(support[1, 1], maximum)
        left = right
    return support
