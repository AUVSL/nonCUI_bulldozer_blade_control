"""Grid-search actual pitch Kp/Ki values and plot blade-depth RMSE.

Example: python tune_pi.py --p-gains 0.005 0.01 0.02 --i-gains 0 0.001 0.002
Each trial uses a fresh default simulation and the same simulated time horizon.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from util import DozerSimulation


def grid_search(p_gains, i_gains, duration=5.0, desired_depth=0.3, output_dir="figures/pi_tuning"):
    """Return RMSE[i_index, p_index]; save trial data and a heatmap.

    Gains are assigned directly to controller.Kp[1] and controller.Ki[1],
    without multiplying by dt or by one another. Lower RMSE is better.
    """
    p_gains = np.asarray(p_gains, dtype=float)
    i_gains = np.asarray(i_gains, dtype=float)
    for gains in (p_gains, i_gains):
        if gains.ndim != 1 or gains.size == 0 or not np.all(np.isfinite(gains)) or np.any(gains < 0):
            raise ValueError("Gain lists must contain finite nonnegative values.")
        if np.any(np.diff(gains) <= 0):
            raise ValueError("Gain lists must be strictly increasing.")
    if not np.isfinite(duration) or duration < 0.01:
        raise ValueError("Duration must be at least one simulation step (0.01 s).")
    if not np.isfinite(desired_depth):
        raise ValueError("Desired depth must be finite.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rmse = np.empty((len(i_gains), len(p_gains)))
    with (output_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["kp", "ki", "rmse_m", "desired_depth_m", "duration_s", "distance_m", "cut_limit_reached"])
        for j, ki in enumerate(i_gains):
            for i, kp in enumerate(p_gains):
                sim = DozerSimulation(blade_roll_pitch_yaw=np.zeros(3), controller_type="pi")
                sim.controller.Kp[1] = kp
                sim.controller.Ki[1] = ki
                sim.controller.desired_depth = desired_depth
                sim.stop_time = duration
                # Score every candidate over the same horizon, including stalls.
                sim.stop_distance = np.inf
                print(f"Trial {j * len(p_gains) + i + 1}/{rmse.size}: Kp={kp:g}, Ki={ki:g}", flush=True)
                sim.run()
                rmse[j, i] = sim.blade_depth_rmse
                if not np.isfinite(rmse[j, i]):
                    raise ValueError(f"Nonfinite RMSE for Kp={kp}, Ki={ki}")
                writer.writerow([kp, ki, rmse[j, i], desired_depth, sim.force_log[-1]["time"], sim.total_distance, sim.cut_limit_reached])
                stream.flush()
    best_j, best_i = np.unravel_index(np.argmin(rmse), rmse.shape)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    # Uniform cells make each tested pair readable, even for uneven gain grids.
    heatmap = ax.imshow(rmse, origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(p_gains)), [f"{value:g}" for value in p_gains])
    ax.set_yticks(np.arange(len(i_gains)), [f"{value:g}" for value in i_gains])
    ax.set_xlabel("P gain (controller.Kp[1])")
    ax.set_ylabel("I gain (controller.Ki[1])")
    ax.set_title(f"PI blade-depth RMSE: {duration:g} s, desired depth {desired_depth:g} m")
    fig.colorbar(heatmap, ax=ax, label="Depth RMSE (m; lower is better)")
    ax.scatter(best_i, best_j, marker="*", s=220, facecolors="none", edgecolors="red", linewidths=1.8, label="Best tested pair")
    ax.legend(loc="upper left", bbox_to_anchor=(0., -0.13))
    fig.savefig(output_dir / "rmse_heatmap.png", dpi=160)
    plt.close(fig)
    print(f"Best tested: Kp={p_gains[best_i]:g}, Ki={i_gains[best_j]:g}, RMSE={rmse[best_j, best_i]:.6f} m")
    print(f"Saved results and heatmap to {output_dir}")
    return rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p-gains", nargs="+", type=float, default=[0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.024])
    parser.add_argument("--i-gains", nargs="+", type=float, default=[0.0 , 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512])
    parser.add_argument("--duration", type=float, default=5.)
    parser.add_argument("--desired-depth", type=float, default=0.3)
    parser.add_argument("--output-dir", default="figures/pi_tuning")
    args = parser.parse_args()
    grid_search(args.p_gains, args.i_gains, args.duration, args.desired_depth, args.output_dir)
