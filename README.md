# surface_aware_bulldozer_sim

A Python simulation of a 3D bulldozer blade controller that is aware of the
terrain it drives over. It is a port of the original MATLAB/Simulink model —
the `sim('simulation_3d')` call is replaced with an explicit forward-Euler
integration loop, and the soil/track/blade physics are re-implemented in NumPy.

The variable names deliberately mirror the equations in the associated paper
rather than following typical software naming conventions, so the code can be
read side-by-side with the derivations.

## What it models

- **Rigid-body dynamics** of a tracked dozer in 6 DOF, driven by a
  configuration-dependent velocity-mapping matrix `S` (and its time
  derivative `Sd`, verified against the closed-form Mathematica result in
  [math/s_derivative.nb](math/s_derivative.nb)).
- **Blade–terrain interaction** — soil cutting forces and moments from the
  blade roll/pitch, plus a growing spoil pile whose fill fraction scales with
  distance travelled. Soil spill-over scales blade force and moment by
  `clip(1 - abs(yaw_speed) / angular_velocity_limit, 0, 1)` for either turn
  direction. With turning disabled (zero limit), resistance is unchanged.
  The surface simulation's cut-depth stop force remains independent of spill-over.
- **Track–terrain interaction** — per-track rolling resistance, lateral
  ground reaction, and turning moment, with an instantaneous-center-of-rotation
  (ICR) model saturated as in Ahmadi, Polotski & Hurteau (2000).
- **Path following** — a pure-pursuit controller drives a figure-8, using a
  piecewise-linear (4PL) lookup that maps heading error to a differential
  track-force fraction. The lookup is fitted offline in `calibration.py`.
- **Surface-transition awareness** (`util.py`, work in progress on the
  `track-surface-transitions` branch) — the surface is a NetworkX height-field
  grid; both complete track centerlines are sampled at their ends, midpoints,
  and grid crossings so the body pitches and rolls to conform to slopes, ramps,
  and up/down-hill transitions. In blade mode, `q` remains the rigid-body pose
  origin but is not treated as an additional terrain-contact point; commanded
  blade pitch controls cut depth without replacing the track-supported body
  pitch.

## Repository layout

| File | Purpose |
|------|---------|
| [main.py](main.py) | `BulldozerSimulation` — full dynamics, controllers, and the multi-panel GIF / force plots. |
| [util.py](util.py) | `Surface` — surface-transition tracking of the track-contact points over a height-field grid. |
| [controllers.py](controllers.py) | Blade and track controllers; `Control` supports PD and PI modes. |
| [calibration.py](calibration.py) | `BulldozerCalibration` — track-force sweeps, straight-line threshold search, and angle→torque curve fitting. |
| [test_main.py](test_main.py) | pytest unit tests for the kinematics, force models, and integration loop. |
| [math/](math/) | Mathematica / MATLAB derivations of the `S` matrix and its derivative. |

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/AUVSL/surface_aware_bulldozer_sim.git
cd surface_aware_bulldozer_sim
pip install -r requirements.txt
```

Dependencies: `numpy`, `matplotlib`, `networkx`, `pytest`.

## Docker (Ubuntu 24.04)

If you use VS Code and want to work inside the container, install Microsoft's
**Dev Containers** extension (`ms-vscode-remote.remote-containers`). It lets you
reopen this project inside a container using the configuration described below. The Docker commands below can also be
run in VS Code's terminal without the extension.

Review the base image before running the build commands below:

- Source: [Docker Official Ubuntu image](https://hub.docker.com/_/ubuntu), maintained by Canonical.
- Image: `docker.io/library/ubuntu:24.04`
- Pinned image index digest: `sha256:69cecf4bbf72d2d44a9eef1b71fb98c7fb973d78af11399deccef19beb008ad9`.

The Dockerfile pins this digest so changes to the tag do not silently replace
the reviewed base. Future base updates require reviewing and changing the digest.
It installs Python 3, venv, and Git from Ubuntu's signed package repositories, then
installs `requirements.txt` and its dependencies from PyPI into a virtual
environment. Only binary Python packages are accepted. These Python dependencies
are third-party packages, separate from the official base image; their versions
retain the existing requirements ranges and are not locked. Official image
provenance does not guarantee that all software is free of vulnerabilities.

After image approval, start Docker Desktop with Linux containers enabled, then
run from the repository root (PowerShell or a Linux shell):

```bash
docker compose build
docker compose run --rm simulator python -m pytest -q -p no:cacheprovider
docker compose run --rm simulator
```

Breaking down `docker compose run --rm simulator`:

- `docker compose` uses the services defined in `compose.yaml`.
- `run` creates and starts a new container for a one-time task.
- `--rm` removes that container when the task finishes. The built image stays available.
- `simulator` selects the service, which runs `python util.py` by default.

Results remain in the host's `figures/` folder after the container is removed
because that folder is mounted into the container.

The default command runs the surface-transition demo in `util.py`. Output is saved to the host `figures/`
directory through a bind mount. Existing files with the same output names may
be overwritten, as with running the scripts directly. The container runs as an
unprivileged user with networking disabled at runtime. The build needs network
access to Docker Hub, Ubuntu repositories, and PyPI.

The default command can also be written explicitly:

```bash
docker compose run --rm simulator python util.py
```

To run the dozer dynamics simulation in `main.py` or another entry point, override the default command:

```bash
docker compose run --rm simulator python main.py
docker compose run --rm simulator python calibration.py
docker compose run --rm simulator python tuning.py pd --p-gains 0.16 0.32 --d-gains 0 0.16
```

Run `docker compose build` again after editing the Dockerfile, code, or requirements so the image includes the changes. On Linux, create `figures/` first
and add `--user "$(id -u):$(id -g)"` after `run --rm` if needed for host output
permissions. Docker Desktop on Windows normally handles bind-mount permissions.

## Develop in VS Code with Dev Containers

The configuration is in `.devcontainer/devcontainer.json` (a file inside the
`.devcontainer` folder). It builds the existing Dockerfile using the same
approved, digest-pinned official Ubuntu 24.04 image.

1. Start Docker Desktop with Linux containers enabled.
2. Open this repository folder in VS Code and install Microsoft's **Dev Containers** extension.
3. Press **Ctrl+Shift+P** and select **Dev Containers: Reopen in Container**.
4. Wait for the build and VS Code setup to finish, then open a new terminal in VS Code.

The terminal now runs Bash inside Ubuntu, so VS Code's `source` command for
virtual environment activation works. Run Python directly:

```bash
python util.py
python main.py
python -m pytest test_main.py -v -p no:cacheprovider
```

The dev container stays running while you work; it does not launch the simulation
automatically. Your repository is mounted at
`/workspaces/surface_aware_bulldozer_sim`, so code edits take effect immediately
and generated `figures/` files remain on your host. You do not need to rebuild
for Python source edits in this workflow.

VS Code uses `/opt/venv/bin/python` and installs Microsoft's Python and Pylance
extensions inside the container. The development container has network access
for VS Code setup and extensions; the separate Compose simulation service still
has networking disabled. Docker commands should be run from a host terminal;
Docker is not installed inside the development container.

After changing requirements, the Dockerfile, or the dev container configuration,
use **Dev Containers: Rebuild Container**. To return to your host environment,
use **Dev Containers: Reopen Folder Locally**.

The test command runs the existing suite. If collection reports missing
`_DozerTrackSimulation` or `_Surface` imports, that is the existing mismatch
between `test_main.py` and `util.py`, not a Dev Containers setup error.

If reopening on Windows fails with an error mentioning a WSL distro mount
service or `wayland-0`, open VS Code's **User Settings**, search for
`dev.containers.mountWaylandSocket`, and disable **Dev Containers: Mount Wayland
Socket**. Then retry **Dev Containers: Reopen in Container**. This optional
Linux GUI socket is not needed for the simulation's headless plots. The setting
must be changed in User Settings, not workspace or container settings.

See the [official VS Code Dev Containers guide](https://code.visualstudio.com/docs/devcontainers/create-dev-container).

## Usage

Each script is runnable on its own and writes its output into a `figures/`
directory (created automatically). Note that `matplotlib` is set to the
headless `Agg` backend by default; remove that line if you want to view plots
interactively.

**Run the dozer dynamics simulation** — renders an animated GIF (3D + top +
side views with the body, blade, and spoil pile) and a force/moment panel:

```bash
python main.py          # → figures/simulation.gif, figures/forces.png, figures/surface_grid.png
```

`main()` runs with fixed track forces. To follow the figure-8 path instead,
call `run_and_plot(use_path_controller=True, stop_time=30, lookahead_dist=0.9)`,
which also prints cross-track and heading RMSE/MAE/max statistics.

**Run the surface-transition demo** — renders a 4-panel GIF (top, 3D, side,
back) of the track-contact points conforming to the terrain:

```bash
python util.py          # → figures/simulation.gif
```

The scenario is chosen via `Surface(is_uphill=...,
is_surface_pitched=..., is_backwards=..., enable_blade=...)`. The blade is
enabled by default; pass `enable_blade=False` to run and render the tracked body
without blade geometry or soil deformation.

```python
from util import Surface

simulation = Surface(enable_blade=False)
simulation.run()
```

**Fit the controller lookup** — sweeps track forces, finds the straight-line
threshold, and compares single-term / piecewise / Bezier fits for the
angle→torque mapping:

```bash
python calibration.py   # → figures/track_force_sweep.png, figures/torque_fits.png
```

## Testing

```bash
pytest test_main.py
```

Tests cover the saturation/wrap/friction helpers, rotation matrices, the `S`
and `Sd` matrices (against the closed-form formula), the blade and track force
models, and an integration smoke test. They run in CI on every push via
[.github/workflows/tests.yml](.github/workflows/tests.yml).

## Design notes

- The driving force is intentionally low relative to the vehicle mass so the
  forces stay consistent with the sandy-loam soil parameters from the
  literature; turning requires a large differential between the two track
  commands.
- Lateral velocity is clamped by `laterial_velocity_limit` (default 0) — the
  dozer does not "slide" sideways unless this is raised.
- The code assumes forward motion. When driving in reverse, set the blade
  force to zero (see the `is_backwards` handling in `util.py`).
- Keep the time step small (`dt` defaults to 1/100). A step that is too large
  produces oscillation in the integrators and thus in the position.

## License

Released under the GNU General Public License v3.0 — see [LICENSE](LICENSE).

### Controller tuning

`tuning.py` provides one `Tuning` class for PD and PI pitch-gain searches.
Each trial starts a fresh simulation and saves its depth RMSE to `results.csv`,
with a heatmap in `figures/pd_tuning` or `figures/pi_tuning` by default.

```bash
python tuning.py pd --p-gains 0.16 0.32 --d-gains 0 0.16
python tuning.py pi --p-gains 0.002 0.004 --i-gains 0 0.002
```

```python
from tuning import Tuning

rmse = Tuning("pi").grid_search([0.002, 0.004], [0.0, 0.002], duration=5.0)
```

The returned array has secondary gain (Kd or Ki) rows and Kp columns.
Use `--duration`, `--desired-depth`, and `--output-dir` to configure either mode.

The runnable examples (`python util.py` and `python main.py`) start with a fixed
blade roll, pitch, and yaw of zero, and blade
control disabled. For `DozerSimulation`, configure this with
`blade_roll_pitch_yaw=np.array([0.0, 0.0, 0.0]), enable_blade_control=False`.
Body motion and soil interaction continue normally. Blade control remains enabled
by default for programmatic simulations and tuning.

`Control` (imported with `from controllers import Control`) supports both blade controller modes via `controller_type="pd"` (default)
or `controller_type="pi"`. `DozerSimulation(controller_type="pi")` uses the same
class with PI gains, integral reset, and actuator anti-windup enabled.
