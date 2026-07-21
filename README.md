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
  distance travelled.
- **Track–terrain interaction** — per-track rolling resistance, lateral
  ground reaction, and turning moment, with an instantaneous-center-of-rotation
  (ICR) model saturated as in Ahmadi, Polotski & Hurteau (2000).
- **Path following** — a pure-pursuit controller drives a figure-8, using a
  piecewise-linear (4PL) lookup that maps heading error to a differential
  track-force fraction. The lookup is fitted offline in `calibration.py`.
- **Surface-transition awareness** (`util.py`, work in progress on the
  `track-surface-transitions` branch) — the surface is a NetworkX height-field
  grid; the dozer's front and back track-contact points are projected onto the
  surface via bilinear interpolation so the body pitches and rolls to conform to
  slopes, ramps, and up/down-hill transitions.

## Repository layout

| File | Purpose |
|------|---------|
| [main.py](main.py) | `BulldozerSimulation` — full dynamics, controllers, and the multi-panel GIF / force plots. |
| [util.py](util.py) | `Surface` — surface-transition tracking of the track-contact points over a height-field grid. |
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

The scenario is chosen via the `Surface(is_uphill=..., is_surface_pitched=...,
is_backwards=...)` flags at the bottom of the file.

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
