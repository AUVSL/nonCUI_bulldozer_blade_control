"""Tune PD or PI pitch gains with a grid search and blade-depth RMSE heatmap.

Examples:
    python tuning.py pd --p-gains 0.16 0.32 --d-gains 0 0.16
    python tuning.py pi --p-gains 0.002 0.004 --i-gains 0 0.002
"""
from __future__ import annotations
from typing import Optional, Union
from numpy.typing import ArrayLike
from sim_types import FloatArray

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from main import DozerSimulation


class Tuning:
    """Search pitch Kp and Kd (PD) or Ki (PI) using fresh simulations."""

    def __init__(self, controller_type: str="pd") -> None:
        if controller_type not in ("pd", "pi"):
            raise ValueError("Controller type must be 'pd' or 'pi'.")
        self.controller_type: str = controller_type
        self.gain_attribute: str = "Kd" if controller_type == "pd" else "Ki"

    def grid_search(self, p_gains: ArrayLike, secondary_gains: ArrayLike, duration: float=5.0, desired_depth: float=0.3, output_dir: Optional[Union[str, Path]]=None) -> FloatArray:
        """Return RMSE[secondary_index, p_index]; save trial data and a heatmap.

        Gains are assigned directly to controller.Kp[1] and the selected controller.Kd[1] or controller.Ki[1],
        without multiplying by dt or by one another. Lower RMSE is better.
        """
        p_gains = np.asarray(p_gains, dtype=float)
        secondary_gains = np.asarray(secondary_gains, dtype=float)
        for gains in (p_gains, secondary_gains):
            if gains.ndim != 1 or gains.size == 0 or not np.all(np.isfinite(gains)) or np.any(gains < 0):
                raise ValueError("Gain lists must contain finite nonnegative values.")
            if np.any(np.diff(gains) <= 0):
                raise ValueError("Gain lists must be strictly increasing.")
        if not np.isfinite(duration) or duration < 0.01:
            raise ValueError("Duration must be at least one simulation step (0.01 s).")
        if not np.isfinite(desired_depth):
            raise ValueError("Desired depth must be finite.")
        output_dir = Path(output_dir) if output_dir is not None else Path("figures") / f"{self.controller_type}_tuning"
        output_dir.mkdir(parents=True, exist_ok=True)
        rmse = np.empty((len(secondary_gains), len(p_gains)))
        with (output_dir / "results.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["kp", self.gain_attribute.lower(), "rmse_m", "desired_depth_m", "duration_s", "distance_m", "cut_limit_reached"])
            for j, secondary_gain in enumerate(secondary_gains):
                for i, kp in enumerate(p_gains):
                    sim = DozerSimulation(blade_roll_pitch_yaw=np.zeros(3), controller_type=self.controller_type)
                    sim.controller.Kp[1] = kp
                    getattr(sim.controller, self.gain_attribute)[1] = secondary_gain
                    sim.controller.desired_depth = desired_depth
                    sim.stop_time = duration
                    # Score every candidate over the same horizon, including stalls.
                    sim.stop_distance = np.inf
                    print(f"Trial {j * len(p_gains) + i + 1}/{rmse.size}: Kp={kp:g}, {self.gain_attribute}={secondary_gain:g}", flush=True)
                    sim.run()
                    rmse[j, i] = sim.blade_depth_rmse
                    if not np.isfinite(rmse[j, i]):
                        raise ValueError(f"Nonfinite RMSE for Kp={kp}, {self.gain_attribute}={secondary_gain}")
                    writer.writerow([kp, secondary_gain, rmse[j, i], desired_depth, sim.force_log[-1]["time"], sim.total_distance, sim.cut_limit_reached])
                    stream.flush()
        best_j, best_i = np.unravel_index(np.argmin(rmse), rmse.shape)
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        # Uniform cells make each tested pair readable, even for uneven gain grids.
        heatmap = ax.imshow(rmse, origin="lower", aspect="auto", cmap="viridis_r")
        ax.set_xticks(np.arange(len(p_gains)), [f"{value:g}" for value in p_gains])
        ax.set_yticks(np.arange(len(secondary_gains)), [f"{value:g}" for value in secondary_gains])
        ax.set_xlabel("P gain (controller.Kp[1])")
        ax.set_ylabel(f"{self.gain_attribute[1].upper()} gain (controller.{self.gain_attribute}[1])")
        ax.set_title(f"{self.controller_type.upper()} blade-depth RMSE: {duration:g} s, desired depth {desired_depth:g} m")
        fig.colorbar(heatmap, ax=ax, label="Depth RMSE (m; lower is better)")
        ax.scatter(best_i, best_j, marker="*", s=220, facecolors="none", edgecolors="red", linewidths=1.8, label="Best tested pair")
        ax.legend(loc="upper left", bbox_to_anchor=(0., -0.13))
        fig.savefig(output_dir / "rmse_heatmap.png", dpi=160)
        plt.close(fig)
        print(f"Best tested: Kp={p_gains[best_i]:g}, {self.gain_attribute}={secondary_gains[best_j]:g}, RMSE={rmse[best_j, best_i]:.6f} m")
        print(f"Saved results and heatmap to {output_dir}")
        return rmse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="controller_type", required=True)
    for mode, p_defaults, secondary_defaults in (
        ("pd", [0.16, 0.32, 0.64, 1.28, 2.56, 5.12],
         [0.0, 0.16, 0.32, 0.64, 1.28, 2.56]),
        ("pi", [0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.024],
         [0.0, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512]),
    ):
        command = modes.add_parser(mode, help=f"Tune {mode.upper()} pitch gains")
        command.add_argument("--p-gains", nargs="+", type=float, default=p_defaults)
        command.add_argument("--d-gains" if mode == "pd" else "--i-gains",
                             dest="secondary_gains", nargs="+", type=float,
                             default=secondary_defaults)
        command.add_argument("--duration", type=float, default=5.0)
        command.add_argument("--desired-depth", type=float, default=0.3)
        command.add_argument("--output-dir", default=f"figures/{mode}_tuning")
    args = parser.parse_args()
    Tuning(args.controller_type).grid_search(
        args.p_gains, args.secondary_gains, args.duration, args.desired_depth, args.output_dir)


if __name__ == "__main__":
    main()
