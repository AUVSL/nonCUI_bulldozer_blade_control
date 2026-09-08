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
    original = last["drive_left"]
    simulation.F_track[0] = -1
    assert last["drive_left"] == original


def test_force_gif_includes_final_sample_and_all_panels(monkeypatch, tmp_path):
    simulation = DozerSimulation()
    for index in range(1, 4):
        simulation.Fb = -100.0 * index
        simulation.Mb = 20.0 * index
        simulation._log_forces(index * simulation.dt)
    captured = []
    real_close = visual.plt.close
    monkeypatch.setattr(visual.plt, "close", lambda fig: captured.append(fig))
    try:
        output = Visualization(simulation).forces_visualization(tmp_path / "forces.gif")
        with Image.open(output) as gif:
            assert gif.n_frames == 3
            assert gif.size == (1200, 900)
        figure = captured[-1]
        assert len(figure.axes) == 8
        np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [0, -100, -200, -300])
        np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(), [0, .01, .02, .03])
    finally:
        for figure in captured:
            real_close(figure)


def test_single_sample_force_gif(tmp_path):
    output = Visualization(DozerSimulation()).forces_visualization(tmp_path / "initial.gif")
    with Image.open(output) as gif:
        assert gif.n_frames == 1
