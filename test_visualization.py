"""Regression checks for the offset blade-arm rendering."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

import visualization as visual
from util import DozerSimulation


@pytest.mark.parametrize("offset", ([0.5, 0, 0], [0, 0, 0], [0.4, -0.3, 8.0]))
def test_offset_blade_renders_without_arm_overlay(monkeypatch, tmp_path, offset):
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
    real_close = visual.plt.close
    monkeypatch.setattr(visual.plt, "close", lambda fig: captured.setdefault("figure", fig))
    monkeypatch.chdir(tmp_path)
    try:
        visual.Visualization(simulation).visualization()
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


def test_body_only_render_does_not_need_arm_offset(monkeypatch, tmp_path):
    simulation = DozerSimulation(enable_blade=False)
    del simulation.blade_arm_offset
    captured = {}
    real_close = visual.plt.close
    monkeypatch.setattr(visual.plt, "close", lambda fig: captured.setdefault("figure", fig))
    monkeypatch.chdir(tmp_path)
    try:
        visual.Visualization(simulation).visualization()
        assert all(line.get_label() != "offset blade arm"
                   for axis in captured["figure"].axes for line in axis.lines)
        assert (tmp_path / "figures" / "simulation.gif").is_file()
    finally:
        if "figure" in captured:
            real_close(captured["figure"])
