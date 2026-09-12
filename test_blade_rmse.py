"""Run-level depth accuracy uses pointwise references and terminal samples."""
import numpy as np
import pytest
from util import DozerSimulation


def test_rmse_uses_each_reference_and_includes_terminal_pose(monkeypatch, capsys):
    sim = DozerSimulation(blade_roll_pitch_yaw=np.zeros(3))
    sim.controller.desired_depth = 0.2
    sim.stop_time = 0.1
    sim.stop_distance = 0.01
    sim.q_dot[:3] = [1., 0., 0.]
    monkeypatch.setattr(sim, "_blade_bottom_center", lambda: np.array([0., 0., 0.]))
    monkeypatch.setattr(sim, "_undeformed_height", lambda _: 0.)
    monkeypatch.setattr(sim, "_blade_update", lambda **_: (np.zeros((2, 3)), []))
    def move():
        sim.controller.desired_depth = 0.4
        return []
    monkeypatch.setattr(sim, "_body_update", move)
    sim.run()
    assert sim.blade_depth_rmse == pytest.approx(np.sqrt((0.2**2 + 0.4**2) / 2))
    assert len(sim.force_log) == len(sim.log) == 2
    assert sim.force_log[-1]["desired_depth"] == 0.4
    assert sim.force_log[-1]["blade_depth_error"] == -0.4
    assert "Blade depth RMSE:" in capsys.readouterr().out


def test_zero_step_run_uses_current_initial_depth():
    sim = DozerSimulation(blade_roll_pitch_yaw=np.zeros(3))
    sim.stop_time = 0.
    center = sim._blade_bottom_center()
    sim.controller.desired_depth = sim._undeformed_height(center) - center[2]
    sim.run()
    assert sim.blade_depth_rmse == pytest.approx(0.)


def test_disabled_blade_has_no_depth_rmse(capsys):
    sim = DozerSimulation(enable_blade=False)
    sim.stop_time = 0.02
    sim.run()
    assert sim.blade_depth_rmse is None
    assert "Blade depth RMSE:" not in capsys.readouterr().out
