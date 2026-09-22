# Surface-Aware Bulldozer Simulation

A Python simulation of a tracked bulldozer whose body and blade respond to a
3D height-field terrain. The model ports the original MATLAB/Simulink dynamics
to an explicit numerical loop built with NumPy.

The variable names follow the equations used by the associated research so the
implementation can be compared directly with the derivations in [`math/`](math/).

## Features

- Six-degree-of-freedom rigid-body dynamics for a tracked vehicle.
- Track contact sampled at track endpoints, midpoints, and terrain-grid crossings.
- Body pitch and roll settlement over slopes and surface transitions.
- Blade roll, pitch, and yaw with configurable rate and angle limits.
- Terrain deformation, maximum cut depth, soil accumulation, and spill while turning.
- PD and PI blade-depth controllers with actuator limiting and PI anti-windup.
- Pure-pursuit path following with a calibrated heading-error-to-track-force mapping.
- Headless animated visualization of the vehicle, terrain, blade, soil pile, and logged forces.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`main.py`](main.py) | `DozerSimulation`, including terrain generation, contact, dynamics, deformation, logging, and the runnable example. |
| [`controllers.py`](controllers.py) | PD/PI blade control and pure-pursuit track control. |
| [`visualization.py`](visualization.py) | Geometry and force-history GIF rendering. |
| [`tuning.py`](tuning.py) | PD/PI pitch-gain grid searches and result heatmaps. |
| [`test_main.py`](test_main.py) | Unit, integration, controller, terrain, and visualization tests. |
| [`sim_types.py`](sim_types.py) | Shared NumPy, terrain, contact, and log type aliases. |
| [`mypy.ini`](mypy.ini) | Static type-check configuration for the core simulation modules. |
| [`calibration.py`](calibration.py) | Historical track-force calibration experiments; see the limitation below. |
| [`math/`](math/) | Mathematica, MATLAB, and written derivations for the kinematic model. |

## Requirements

- Python 3.9 or newer
- NumPy
- Matplotlib
- NetworkX
- Pillow
- pytest

Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Matplotlib uses the noninteractive `Agg` backend, so simulations and tests can
run without a desktop display.


## Docker and Dev Containers

The repository includes a digest-pinned Ubuntu 24.04 image and a VS Code Dev
Container configuration. Build the image, then explicitly select the supported
entry point:

```bash
docker compose build
docker compose run --rm simulator python -m pytest test_main.py -q -p no:cacheprovider
docker compose run --rm simulator python main.py
```

The Compose service has networking disabled at runtime and mounts `figures/`
from the host. Rebuild the image after changing Python sources, dependencies, or
the Docker configuration. In a VS Code Dev Container, the repository is mounted
at `/workspaces/surface_aware_bulldozer_sim`, so normal source edits are visible
immediately.

## Run the simulation

```bash
python main.py
```

The default entry point enables the blade controller and figure-eight path
controller. It writes these animations under `figures/`:

- `simulation.gif` — terrain and vehicle views
- `forces.gif` — forces, moments, velocity, blade depth, and actuator commands

For programmatic use:

```python
import numpy as np

from main import DozerSimulation

simulation = DozerSimulation(
    is_uphill=True,
    is_surface_pitched=False,
    is_surface_rolled=False,
    is_backwards=False,
    enable_blade=True,
    blade_roll_pitch_yaw=np.array([0.0, 0.0, 0.0]),
    controller_type="pd",
    enable_blade_control=True,
    use_path_controller=False,
)
simulation.stop_time = 5.0
simulation.run_and_plot()
```

Set `enable_blade=False` to simulate and render the tracked body without blade
contact or terrain deformation. Set `enable_blade_control=False` to retain a
fixed blade pose while keeping blade contact enabled.

## Terrain configuration

`DozerSimulation` creates a NetworkX grid and stores its numerical points in
`grid_pts`. The main terrain options are:

| Argument | Meaning |
| --- | --- |
| `is_uphill` | Selects whether the transition rises or falls. |
| `is_surface_pitched` | Places the transition along the longitudinal terrain axis. |
| `is_surface_rolled` | Adds a transition across the second terrain axis. |
| `is_backwards` | Reverses the permitted direction of travel. |

The vehicle pose is stored in `q` as `[x, y, z, roll, pitch, yaw]`. Tracks
provide body support; commanded blade pitch changes cutting depth without
replacing the track-supported body pitch.

## Blade controllers

Select the controller when constructing the simulation:

```python
pd_simulation = DozerSimulation(controller_type="pd")
pi_simulation = DozerSimulation(controller_type="pi")
```

Both modes use `Control.proportional_blade_controller`. PD mode uses `Kp` and
`Kd`; PI mode uses `Kp` and `Ki`. The simulation applies blade angle and
rate limits and feeds the applied increment back to the PI controller to prevent
windup.

The desired cut depth is configured in metres:

```python
simulation.controller.desired_depth = 0.2
```

## Controller tuning

Run a small PD or PI grid search with:

```bash
python tuning.py pd --p-gains 0.16 0.32 --d-gains 0 0.16
python tuning.py pi --p-gains 0.002 0.004 --i-gains 0 0.002
```

Useful options are `--duration`, `--desired-depth`, and `--output-dir`. Each
trial starts with a fresh simulation. The output directory receives
`results.csv` and a heatmap. The returned array uses secondary gain (`Kd` or
`Ki`) rows and `Kp` columns.

The same interface is available from Python:

```python
from tuning import Tuning

rmse = Tuning("pi").grid_search(
    [0.002, 0.004],
    [0.0, 0.002],
    duration=5.0,
    desired_depth=0.2,
)
```

## Path following

Enable the pure-pursuit controller with `use_path_controller=True`. The default
reference is a closed figure-eight. A custom closed reference may be supplied as
an `N x 2` or `N x 3` NumPy-compatible array:

```python
simulation = DozerSimulation(
    controller_type="pi",
    use_path_controller=True,
    path_points=[[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]],
    lookahead_dist=0.9,
)
simulation.stop_time = 30.0
simulation.run_and_plot()
```

The controller writes heading and cross-track errors to `force_log`. A run ends
at the time or distance limit, or when the closed path is complete.

## Testing

Run the complete suite from the repository root:

```bash
python -m pytest test_main.py -q -p no:cacheprovider
```

The suite covers kinematics, rotation and velocity-mapping matrices, the
closed-form and numerical derivative of `S`, track and blade forces, terrain
contact, deformation, PD/PI behavior, path following, logging, and GIF
rendering. CI runs the suite on pushes and pull requests through
[`tests.yml`](.github/workflows/tests.yml).

## Type checking

Function parameters, return values, persistent simulation state, test fixtures,
and test callbacks are annotated. Shared array and terrain aliases live in
[`sim_types.py`](sim_types.py).

Install mypy in the active development environment, then run:

```bash
python -m pip install mypy
python -m mypy
```

The checked core includes `main.py`, `controllers.py`, `tuning.py`, and
`sim_types.py`. `visualization.py` is fully annotated, but its dynamically
constructed Matplotlib 3D artists are excluded from strict checking because the
available Matplotlib stubs do not represent those runtime APIs accurately.
Visualization behavior remains covered by the pytest suite.

## Design notes

- Terrain-contact loops are compiled with Numba and cached after their first use.
  The first simulation in a fresh environment includes a one-time compilation cost.
- The default time step is `0.01` seconds. Larger steps can make the integration
  and blade controller oscillate.
- Lateral velocity is clamped by `lateral_velocity_limit`, which defaults to zero.
- Soil resistance uses parameters for sandy loam; changing vehicle mass, track
  force, or soil coefficients changes the balance of the model.
- `main.py` exposes `_run` as a compatibility alias for `run`.

## License

Released under the GNU General Public License v3.0. See [`LICENSE`](LICENSE).
