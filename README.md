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
