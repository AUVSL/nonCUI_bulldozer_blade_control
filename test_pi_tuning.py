"""Grid orientation, gain assignment, fresh trials, and output checks."""
import csv
import numpy as np
import tune_pi


def test_grid_search_uses_fresh_simulations_and_actual_gains(monkeypatch, tmp_path):
    trials = []
    class Simulation:
        def __init__(self, blade_roll_pitch_yaw, controller_type):
            assert controller_type == "pi"
            np.testing.assert_array_equal(blade_roll_pitch_yaw, 0.)
            self.controller = type("Controller", (), {"Kp": np.zeros(3), "Ki": np.zeros(3)})()
            self.total_distance = 1.
            self.cut_limit_reached = False
            trials.append(self)
        def run(self):
            assert np.isinf(self.stop_distance)
            self.blade_depth_rmse = self.controller.Kp[1] + 10 * self.controller.Ki[1]
            self.force_log = [{"time": self.stop_time}]
    monkeypatch.setattr(tune_pi, "DozerSimulation", Simulation)
    result = tune_pi.grid_search([1., 2.], [0., 0.1], duration=1., output_dir=tmp_path)
    np.testing.assert_allclose(result, [[1., 2.], [2., 3.]])
    assert len(trials) == 4
    assert (tmp_path / "rmse_heatmap.png").is_file()
    with (tmp_path / "results.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 4
