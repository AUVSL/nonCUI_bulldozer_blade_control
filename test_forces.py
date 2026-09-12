"""Force history and dashboard regression checks."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
from PIL import Image

from util import DozerSimulation
from visualization import Visualization
import visualization as visual


@pytest.mark.parametrize("enable_blade", [True, False])
def test_force_history_matches_logged_steps(enable_blade):
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


def test_force_gif_includes_final_sample_and_all_panels(monkeypatch, tmp_path):
    simulation = DozerSimulation()
    for index in range(1, 4):
        simulation.Fb = -100.0 * index
        simulation.Mb = 20.0 * index
        simulation.requested_blade_rates = np.array([index, -2 * index, 3 * index], dtype=float)
        simulation.q[2] += 0.1
        simulation._log_forces(index * simulation.dt)
    captured = []
    real_close = visual.plt.close
    monkeypatch.setattr(visual.plt, "close", lambda fig: captured.append(fig))
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


def test_single_sample_force_gif(tmp_path):
    output = Visualization(DozerSimulation()).forces_visualization(tmp_path / "initial.gif")
    with Image.open(output) as gif:
        assert gif.n_frames == 1


def test_requested_rates_are_saved_before_limiting(monkeypatch):
    simulation = DozerSimulation()
    previous = simulation.blade_roll_pitch_yaw.copy()
    command = np.array([2., -3., 4.])
    monkeypatch.setattr(simulation.controller, "proportional_blade_controller", lambda *_: command)
    simulation._blade_update()
    expected = simulation.blade_angle_actuation_scaler * command / simulation.dt
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
