"""Mode-aware visualization for the surface-aware bulldozer simulation."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from PIL import Image


class Visualization:
    """Visualize a simulation with optional blade-specific layers.

    The simulation's ``enable_blade`` setting is the single source of truth for
    selecting blade geometry, deforming terrain, and output timing.
    """

    def __init__(self, simulation):
        self.simulation = simulation

    def _pile_vertices(self, blade, pile_state):
        """Build a full-width pile that grows outward from the blade with soil load."""
        simulation = self.simulation
        h3, h4, roll, yaw = pile_state
        blade = np.asarray(blade)
        if not simulation.enable_blade or max(h3, h4) <= 1e-6:
            return np.empty((0, 3))
        rotation = (simulation._rotation_lg(*blade[0, 3:6])
                    @ simulation._rotation_lg(roll, 0.0, yaw))
        center = (blade[0, :3] + blade[-1, :3]) / 2
        up = rotation[:, 2]
        forward = rotation[:, 0].copy()
        forward[2] = 0.0
        forward /= max(np.linalg.norm(forward), 1e-12)
        vertices = []
        visible = False
        sections = max(3, int(np.ceil(4 * simulation.B1 / simulation.subdivision)) + 1)
        for fraction in np.linspace(0.0, 1.0, sections):
            load_height = (1 - fraction) * h3 + fraction * h4
            bottom = center + (0.5 - fraction) * simulation.B1 * rotation[:, 1]

            def clearance(distance):
                point = bottom + distance * up
                return point[2] - simulation._undeformed_height(point)

            # Locate the soil/blade intersection along the actual tilted face.
            low, high = 0.0, simulation.H
            if clearance(simulation.H) <= 0.0:
                # Keep inactive sections collapsed to preserve mesh connectivity.
                base = bottom.copy()
                base[2] = simulation._undeformed_height(base)
                vertices.extend([base.copy(), base.copy(), base.copy()])
                continue
            if clearance(0.0) > 0.0:
                # Keep existing soil on the ground under a raised blade section.
                # Do not switch an entire side on when its cutting edge touches.
                base = bottom.copy()
                base[2] = simulation._undeformed_height(base)
                exposed_height = clearance(simulation.H) / max(up[2], 1e-12)
            else:
                for _ in range(40):
                    middle = (low + high) / 2
                    if clearance(middle) < 0.0:
                        low = middle
                    else:
                        high = middle
                base = bottom + high * up
                exposed_height = simulation.H - high
            height = min(max(load_height, 0.0), exposed_height)
            crest = base + height * up
            visible |= height > 1e-6
            toe = base + forward * max(crest[2] - base[2], 0.0) / np.tan(simulation.beta0)
            toe[2] = simulation._undeformed_height(toe)
            vertices.extend([base, crest, toe])
        return np.asarray(vertices) if visible else np.empty((0, 3))

    def _pile_faces(self, vertices, terrain_heights=None):
        """Triangulate only the above-ground brown pile."""
        if not len(vertices):
            return []
        strips = np.asarray(vertices).reshape(-1, 3, 3)
        faces = [strips[0, [0, 1, 2]], strips[-1, [0, 2, 1]]]
        for left, right in zip(strips[:-1], strips[1:]):
            # Join base-to-crest, crest-to-toe, and toe-to-base surfaces.
            for first, second in ((0, 1), (1, 2), (2, 0)):
                faces.append(np.array([left[first], left[second], right[second]]))
                faces.append(np.array([left[first], right[second], right[first]]))
        return faces

    def _surface_tooth_faces(self, blade, terrain_heights, blade_yaw=0.0):
        """Bridge the cutting edge to the next soil grid boundary ahead of it."""
        simulation = self.simulation
        blade = np.asarray(blade)
        rotation = (simulation._rotation_lg(*blade[0, 3:6])
                    @ simulation._rotation_lg(0.0, 0.0, blade_yaw))
        forward = rotation[:, 0].copy()
        forward[2] = 0.0
        forward /= max(np.linalg.norm(forward), 1e-12)

        def height(point):
            i, j = simulation._grid_cell(point)
            corners = [(*simulation.grid_pts[ci][cj][:2], terrain_heights[ci, cj])
                       for ci, cj in ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1))]
            return simulation._bilinear_height(point, corners)

        edge = blade[:, :3].copy()
        ground = edge.copy()
        for index, point in enumerate(edge):
            distances = []
            for axis, coordinates in ((0, simulation.us), (1, simulation.vs)):
                if abs(forward[axis]) > 1e-12:
                    crossings = (coordinates - point[axis]) / forward[axis]
                    distances.extend(crossings[crossings > 1e-8])
            distance = min(distances) if distances else simulation.subdivision
            ground[index] = point + distance * forward
            ground[index, 2] = height(ground[index])
            # A lifted blade must not pull a soil patch into the air.
            edge[index, 2] = min(point[2], height(point))
        faces = []
        for index in range(len(edge) - 1):
            faces.extend([
                np.array([edge[index], ground[index], ground[index + 1]]),
                np.array([edge[index], ground[index + 1], edge[index + 1]]),
            ])
        return faces

    @staticmethod
    def _pile_in_front_of_blade(face, eye, body_center):
        """The pile is on the opposite side of the blade plane from the body."""
        normal = np.cross(face[1] - face[0], face[3] - face[0])
        return np.dot(normal, body_center - face[0]) * np.dot(normal, eye - face[0]) < 0.0

    def forces_visualization(self, filename="figures/forces.gif"):
        """Animate logged forces, blade pitch, blade-center heights, and requested rates.

        Uses the same frame stride and playback rate as the geometry GIF.
        Dynamics values are snapshots from each simulation step, not recomputed
        from the final state.
        """
        if not self.simulation.force_log:
            raise ValueError("No force samples are available to visualize.")
        samples = self.simulation.force_log
        times = np.array([row["time"] for row in samples])
        panels = [
            ("Blade force", "Fb (N)", [("Fb", "Blade")]),
            ("Blade moment", "Mb (N m)", [("Mb", "Blade")]),
            ("Track rolling resistance", "Rl (N)", [("Rl_left", "Left"), ("Rl_right", "Right")]),
            ("Lateral track force", "Fy (N)", [("Fy", "Lateral")]),
            ("Track turning moment", "Mr (N m)", [("Mr", "Turning")]),
            ("Velocity", "m/s (forward), rad/s (turn)", [("v_forward", "Forward"), ("v_turn", "Turn")]),
            ("Blade pitch", "Pitch (rad)", [("blade_pitch", "Pitch")]),
            ("Track drive forces", "Force (N)", [("drive_left", "Left"), ("drive_right", "Right")]),
        ]
        panels.append(("Blade-center heights", "World height (m)", [
            ("starting_surface_height", "Original surface at blade center"),
            ("blade_bottom_height", "Blade bottom center"),
        ]))
        panels.append(("Requested blade rates (before limiting)", "Rate (rad/s)", [
            ("requested_roll_rate", "Roll"),
            ("requested_pitch_rate", "Pitch"),
            ("requested_yaw_rate", "Yaw"),
        ]))
        figure = plt.figure(figsize=(12, 13))
        layout = figure.add_gridspec(6, 2)
        axes = [figure.add_subplot(layout[row, column])
                for row in range(4) for column in range(2)]
        axes.append(figure.add_subplot(layout[4, :]))
        axes.append(figure.add_subplot(layout[5, :]))
        for axis in axes[1:]:
            axis.sharex(axes[0])
        title = figure.suptitle("Forces & Moments over Time")
        traces, cursors = [], []
        for axis, (name, units, series) in zip(axes, panels):
            values = []
            for key, label in series:
                history = np.array([row[key] for row in samples])
                line, = axis.plot([], [], label=label)
                traces.append((line, history))
                values.append(history)
            low, high = np.min(values), np.max(values)
            margin = 0.08 * (high - low) if high > low else max(abs(low) * 0.08, 1.0)
            axis.set_ylim(low - margin, high + margin)
            axis.set_xlim(times[0], max(times[-1], times[0] + self.simulation.dt))
            axis.set_title(name)
            axis.set_ylabel(units)
            axis.set_xlabel("Time (s)")
            axis.grid(True, linewidth=0.4)
            if len(series) > 1:
                axis.legend(loc="upper right", fontsize=8)
            cursors.append(axis.axvline(times[0], color="gray", linestyle="--", linewidth=0.8))
        figure.tight_layout(rect=(0, 0, 1, 0.96))

        def update(index):
            for line, history in traces:
                line.set_data(times[:index + 1], history[:index + 1])
            for cursor in cursors:
                cursor.set_xdata([times[index], times[index]])
            title.set_text(f"Forces & Moments over Time - t = {times[index]:.2f} s")
            return [line for line, _ in traces] + cursors + [title]

        indices = list(range(0, len(samples), 2))
        if indices[-1] != len(samples) - 1:
            indices.append(len(samples) - 1)
        fps = 30 if self.simulation.enable_blade else 20
        output = Path(filename)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            movie = FuncAnimation(figure, update, frames=indices, interval=1000 / fps, blit=False)
            movie.save(str(output), writer=PillowWriter(fps=fps))
        finally:
            plt.close(figure)
        print(f"Saved {output}")
        return output

    def visualization(self, show_neighbors: bool = False):
        """Write the logged simulation to ``figures/simulation.gif``."""
        simulation = self.simulation
        enable_blade = simulation.enable_blade

        print("Rendering GIF...")
        Path("figures").mkdir(parents=True, exist_ok=True)

        # Every second logged state is rendered to keep GIF generation quick.
        log_data = np.asarray(simulation.log)
        data = log_data[::2]
        track_path = np.array([
            simulation._track_xyz(row[1:7]) for row in log_data
        ])
        track_data = track_path[::2]
        neighbor_data = simulation.neighbor_log[::2]

        # Extrude a capsule-shaped side profile across the track width.
        # Inset the semicircle centers to preserve the overall length l;
        # the lower straight edge stays at local z=0.
        radius = simulation.track_height / 2
        end_center = simulation.l / 2 - radius
        half_width = simulation.track_width / 2
        front_angles = np.linspace(-np.pi / 2, np.pi / 2, 17)
        rear_angles = np.linspace(np.pi / 2, 3 * np.pi / 2, 17)
        profile_x = np.r_[end_center + radius * np.cos(front_angles),
                          -end_center + radius * np.cos(rear_angles)]
        profile_z = np.r_[radius + radius * np.sin(front_angles),
                          radius + radius * np.sin(rear_angles)]
        profile_size = len(profile_x)
        track_local = np.concatenate([
            np.column_stack((profile_x, np.full(profile_size, side * half_width), profile_z))
            for side in (-1.0, 1.0)
        ])
        tracks_local = np.array([
            track_local + [0.0, side * simulation.b / 2, 0.0]
            for side in (-1.0, 1.0)
        ])
        # Quads wrap around the tread; the two caps are yellow lateral faces.
        track_faces = [
            [i, (i + 1) % profile_size, (i + 1) % profile_size + profile_size,
             i + profile_size]
            for i in range(profile_size)
        ]
        track_faces.extend([
            list(range(profile_size - 1, -1, -1)),
            list(range(profile_size, 2 * profile_size)),
        ])
        track_mesh_path = np.array([
            row[1:4] + tracks_local @ simulation._rotation_lg(*row[4:7]).T
            for row in log_data
        ])
        track_mesh_data = track_mesh_path[::2]

        # Two blocks fill the gap between the inner track faces. Local +x
        # points toward the blade: a full-height rear and a lower front hood.
        half_body_length = simulation.l / 2
        body_split = -half_body_length + 3 * simulation.l / 5
        body_width_scale = 0.8  # Fraction of the gap between the tracks filled by the body.
        half_body_width = body_width_scale * (simulation.b - simulation.track_width) / 2
        body_bottom, body_top = simulation.track_height / 2, simulation.h
        front_top = body_bottom + (body_top - body_bottom) * 2 / 3
        # Keep the hood height at the body split and lower its nose by 30%.
        front_nose_top = front_top - 0.3 * (front_top - body_bottom)
        body_local = np.array([
            [
                [back, -half_body_width, body_bottom],
                [front, -half_body_width, body_bottom],
                [front, half_body_width, body_bottom],
                [back, half_body_width, body_bottom],
                [back, -half_body_width, back_top],
                [front, -half_body_width, nose_top],
                [front, half_body_width, nose_top],
                [back, half_body_width, back_top],
            ]
            for back, front, back_top, nose_top in (
                (-half_body_length, body_split, body_top, body_top),
                (body_split, half_body_length, front_top, front_nose_top),
            )
        ])
        body_faces = np.array([
            [0, 1, 2, 3], [4, 5, 6, 7], [1, 2, 6, 5],
            [0, 3, 7, 4], [0, 1, 5, 4], [2, 3, 7, 6],
        ])
        body_path = np.array([
            row[1:4] + body_local @ simulation._rotation_lg(*row[4:7]).T
            for row in log_data
        ])
        body_data = body_path[::2]

        # Surface x/y positions are fixed. Blade mode supplies a height snapshot
        # for each frame; body-only mode repeats the final static surface.
        grid_x = np.array([[point[0] for point in column]
                           for column in simulation.grid_pts])
        grid_y = np.array([[point[1] for point in column]
                           for column in simulation.grid_pts])

        if enable_blade:
            blade_path = np.asarray(simulation.blade_log)
            blade_data = blade_path[::2]
            blade_top_data = np.array([
                frame[[0, -1], :3]
                + simulation._blade_rotation_lg(frame[0, 3:6])[:, 2]
                * simulation.H
                for frame in blade_data
            ])
            grid_z_data = np.asarray(simulation.grid_log)[::2]
            # Compute geometry only for rendered frames, after the simulation run.
            history = getattr(simulation, "pile_log", [])
            pile_data = ([self._pile_vertices(blade, state)
                          for blade, state in zip(blade_data, history[::2])]
                         if len(history) == len(log_data)
                         else [np.empty((0, 3)) for _ in data])

            arm_thickness = 0.1  # Square cross-section in metres.
            arm_mesh_data = []
            for pose, blade, blade_top in zip(data[:, 1:7], blade_data, blade_top_data):
                rotation = simulation._rotation_lg(*pose[3:6])
                blade_up = (blade_top[0] - blade[0, :3]) / simulation.H
                arms = []
                for side in (-1.0, 1.0):
                    mount = np.array(simulation.blade_arm_offset, dtype=float, copy=True)
                    mount[1] += side * half_body_width
                    mount[2] = body_bottom
                    start = pose[:3] + rotation @ mount
                    fraction = 0.5 + side * half_body_width / simulation.B1
                    end = ((1 - fraction) * blade[0, :3] + fraction * blade[-1, :3]
                           + body_bottom * blade_up)
                    direction = end - start
                    direction /= max(np.linalg.norm(direction), 1e-12)
                    lateral = np.cross(rotation[:, 2], direction)
                    if np.linalg.norm(lateral) < 1e-8:
                        lateral = rotation[:, 1].copy()
                    lateral /= np.linalg.norm(lateral)
                    vertical = np.cross(direction, lateral)
                    cross_section = arm_thickness / 2 * np.array([
                        -lateral - vertical, lateral - vertical,
                        lateral + vertical, -lateral + vertical,
                    ])
                    arms.append(np.vstack([start + cross_section, end + cross_section]))
                arm_mesh_data.append(arms)
            arm_mesh_data = np.asarray(arm_mesh_data)
        else:
            static_grid_z = np.array([[point[2] for point in column]
                                      for column in simulation.grid_pts])
            grid_z_data = np.broadcast_to(
                static_grid_z, (len(data),) + static_grid_z.shape
            )

        # Give the terrain a fixed base below every logged cut, so the side
        # walls retain their thickness as the top surface deforms.
        soil_base_height = float(grid_z_data.min()) - simulation.subdivision

        # Include all geometry that can appear during the run when choosing
        # view bounds. Blade-only histories are never touched in body-only mode.
        bound_x = [grid_x.ravel(), track_path[:, :, :, 0].ravel()]
        bound_y = [grid_y.ravel(), track_path[:, :, :, 1].ravel()]
        bound_z = [grid_z_data.ravel(), track_path[:, :, :, 2].ravel(),
                   np.array([soil_base_height])]
        bound_x.append(track_mesh_path[:, :, :, 0].ravel())
        bound_y.append(track_mesh_path[:, :, :, 1].ravel())
        bound_z.append(track_mesh_path[:, :, :, 2].ravel())
        bound_x.append(body_path[:, :, :, 0].ravel())
        bound_y.append(body_path[:, :, :, 1].ravel())
        bound_z.append(body_path[:, :, :, 2].ravel())
        if enable_blade:
            bound_x.append(arm_mesh_data[:, :, :, 0].ravel())
            bound_y.append(arm_mesh_data[:, :, :, 1].ravel())
            bound_z.append(arm_mesh_data[:, :, :, 2].ravel())
            bound_x.extend([
                blade_path[:, :, 0].ravel(),
                blade_top_data[:, :, 0].ravel(),
            ])
            bound_y.extend([
                blade_path[:, :, 1].ravel(),
                blade_top_data[:, :, 1].ravel(),
            ])
            bound_z.extend([
                np.asarray(simulation.grid_log).ravel(),
                blade_path[:, :, 2].ravel(),
                blade_top_data[:, :, 2].ravel(),
            ])
        if enable_blade:
            for vertices in pile_data:
                if len(vertices):
                    bound_x.append(vertices[:, 0])
                    bound_y.append(vertices[:, 1])
                    bound_z.append(vertices[:, 2])
        if show_neighbors:
            neighbor_points = np.array([
                point
                for frame in simulation.neighbor_log
                for point in frame
            ])
            bound_x.append(neighbor_points[:, 0])
            bound_y.append(neighbor_points[:, 1])
            bound_z.append(neighbor_points[:, 2])

        all_x = np.concatenate([data[:, 1], *bound_x])
        all_y = np.concatenate([data[:, 2], *bound_y])
        all_z = np.concatenate([data[:, 3], *bound_z])

        margin = 0.5
        center_x = (all_x.max() + all_x.min()) / 2
        center_y = (all_y.max() + all_y.min()) / 2
        center_z = (all_z.max() + all_z.min()) / 2
        half_x = (all_x.max() - all_x.min()) / 2 + margin
        half_y = (all_y.max() - all_y.min()) / 2 + margin
        half_z = (all_z.max() - all_z.min()) / 2 + margin

        # 2x2 layout: top view (X-Y) | 3D view
        #             side view (X-Z)| back view (Y-Z)
        width, height = half_x + half_y, half_y + half_z
        scale = 9 / max(width, height)
        figure = plt.figure(
            figsize=(width * scale, height * scale), layout="constrained"
        )
        grid_spec = figure.add_gridspec(
            2,
            2,
            width_ratios=[half_x, half_y],
            height_ratios=[half_y, half_z],
        )
        axis_top = figure.add_subplot(grid_spec[0, 0])
        axis_3d = figure.add_subplot(grid_spec[0, 1], projection="3d")
        axis_side = figure.add_subplot(grid_spec[1, 0])
        axis_back = figure.add_subplot(grid_spec[1, 1])

        axis_3d.set_xlim(center_x - half_x, center_x + half_x)
        axis_3d.set_ylim(center_y - half_y, center_y + half_y)
        axis_3d.set_zlim(center_z - half_z, center_z + half_z)
        axis_top.set_xlim(center_x - half_x, center_x + half_x)
        axis_top.set_ylim(center_y - half_y, center_y + half_y)
        axis_back.set_xlim(center_y - half_y, center_y + half_y)
        axis_back.set_ylim(center_z - half_z, center_z + half_z)
        axis_side.set_xlim(center_x - half_x, center_x + half_x)
        axis_side.set_ylim(center_z - half_z, center_z + half_z)

        for axis_2d in (axis_top, axis_back, axis_side):
            axis_2d.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis_2d.yaxis.set_major_locator(MaxNLocator(integer=True))

        # An edge's grid indices stay fixed, so only its z coordinates need to
        # be gathered again when blade mode deforms the surface.
        edge_a = np.array([a for a, _ in simulation.surf_grid.edges()])
        edge_b = np.array([b for _, b in simulation.surf_grid.edges()])
        ia, ja = edge_a[:, 0], edge_a[:, 1]
        ib, jb = edge_b[:, 0], edge_b[:, 1]

        def grid_segments(z):
            return np.stack([
                np.column_stack([
                    grid_x[ia, ja], grid_y[ia, ja], z[ia, ja]
                ]),
                np.column_stack([
                    grid_x[ib, jb], grid_y[ib, jb], z[ib, jb]
                ]),
            ], axis=1)

        # Only perimeter edges get vertical walls; the top remains a mesh.
        last_i, last_j = np.array(grid_x.shape) - 1
        boundary_edges = (((ia == 0) & (ib == 0))
                          | ((ia == last_i) & (ib == last_i))
                          | ((ja == 0) & (jb == 0))
                          | ((ja == last_j) & (jb == last_j)))
        terrain_center = np.array([grid_x.mean(), grid_y.mean(), soil_base_height])

        def soil_wall_faces(z):
            upper = grid_segments(z)[boundary_edges]
            lower = upper.copy()
            lower[:, :, 2] = soil_base_height
            faces = np.stack([upper[:, 0], upper[:, 1], lower[:, 1], lower[:, 0]], axis=1)
            normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
            inward = np.sum(normals * (faces.mean(axis=1) - terrain_center), axis=1) < 0
            faces[inward] = faces[inward, ::-1]
            return faces

        def soil_top_faces(z):
            points = np.stack([grid_x, grid_y, z], axis=-1)
            return np.stack([
                points[:-1, :-1], points[1:, :-1],
                points[1:, 1:], points[:-1, 1:],
            ], axis=2).reshape(-1, 4, 3)

        soil_color = "saddlebrown"
        soil_top_color = "burlywood"
        initial_tiles = soil_top_faces(grid_z_data[0])
        soil_top_3d = Poly3DCollection(
            initial_tiles, facecolors=soil_top_color, edgecolors="none",
            alpha=1.0, zorder=0,
        )
        axis_3d.add_collection3d(soil_top_3d)
        # XY positions never change, so this fill can stay in the background.
        axis_top.add_collection(PolyCollection(
            initial_tiles[:, :, [0, 1]], facecolors=soil_top_color,
            edgecolors="none", alpha=1.0, zorder=-1,
        ))
        segments = grid_segments(grid_z_data[0])
        grid_3d = Line3DCollection(
            segments,
            colors=soil_color,
            linewidths=0.5,
            alpha=0.5,
            zorder=0,
        )
        grid_back = LineCollection(
            segments[:, :, [1, 2]],
            colors=soil_color,
            linewidths=0.5,
            alpha=0.5,
            zorder=0,
        )
        grid_side = LineCollection(
            segments[:, :, [0, 2]],
            colors=soil_color,
            linewidths=0.5,
            alpha=0.5,
            zorder=0,
        )
        axis_3d.add_collection3d(grid_3d)
        axis_back.add_collection(grid_back)
        axis_side.add_collection(grid_side)
        axis_top.add_collection(LineCollection(
            segments[:, :, [0, 1]],
            colors=soil_color,
            linewidths=0.5,
            alpha=0.5,
            zorder=0,
        ))

        initial_walls = soil_wall_faces(grid_z_data[0])
        soil_walls_3d = Poly3DCollection(
            initial_walls, facecolors=soil_color, edgecolors=soil_color,
            linewidths=0.2, alpha=1.0, zorder=1,
        )
        soil_walls_back = PolyCollection(
            initial_walls[:, :, [1, 2]], facecolors=soil_color,
            edgecolors=soil_color, linewidths=0.2, alpha=1.0, zorder=1,
        )
        soil_walls_side = PolyCollection(
            initial_walls[:, :, [0, 2]], facecolors=soil_color,
            edgecolors=soil_color, linewidths=0.2, alpha=1.0, zorder=1,
        )
        axis_3d.add_collection3d(soil_walls_3d)
        axis_back.add_collection(soil_walls_back)
        axis_side.add_collection(soil_walls_side)

        def set_grid(frame_index):
            soil_top_3d.set_verts(soil_top_faces(grid_z_data[frame_index]))
            frame_segments = grid_segments(grid_z_data[frame_index])
            grid_3d.set_segments(frame_segments)
            grid_back.set_segments(frame_segments[:, :, [1, 2]])
            grid_side.set_segments(frame_segments[:, :, [0, 2]])
            walls = soil_wall_faces(grid_z_data[frame_index])
            # Hide the far walls so they cannot cover the visible top mesh.
            x, y, _ = proj3d.proj_transform(
                walls[:, :, 0].ravel(), walls[:, :, 1].ravel(),
                walls[:, :, 2].ravel(), axis_3d.get_proj())
            x, y = x.reshape(-1, 4), y.reshape(-1, 4)
            visible = np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1) > 0
            soil_walls_3d.set_verts(walls[visible])
            soil_walls_back.set_verts(walls[:, :, [1, 2]])
            soil_walls_side.set_verts(walls[:, :, [0, 2]])

        axis_3d.set_box_aspect((half_x, half_y, half_z), zoom=1)
        axis_3d.grid(False)
        axis_3d.set_xlabel("X (m)")
        axis_3d.set_ylabel("Y (m)")
        axis_3d.set_zlabel("Z (m)")

        axis_top.set_aspect("equal")
        axis_top.grid(False)
        axis_top.set_xlabel("X (m)")
        axis_top.set_ylabel("Y (m)")

        axis_back.set_aspect("equal")
        axis_back.grid(False)
        axis_back.set_xlabel("Y (m)")
        axis_back.set_ylabel("Z (m)")

        axis_side.set_aspect("equal")
        axis_side.grid(False)
        axis_side.set_xlabel("X (m)")
        axis_side.set_ylabel("Z (m)")

        initial_body_faces = body_data[0][:, body_faces].reshape(-1, 4, 3)
        body_yellow = "gold"
        track_dark = "#444a50"
        body_colors = ["#b8860b", "#ffdf50", "#e8b923",
                       "#e8b923", body_yellow, body_yellow] * 2
        # The second block is the hood; faces 1 and 2 are its top and nose.
        body_colors[len(body_faces) + 1] = body_yellow
        body_colors[len(body_faces) + 2] = body_yellow
        body_projections = []
        for axis, projection in ((axis_top, [0, 1]), (axis_back, [1, 2]),
                                 (axis_side, [0, 2])):
            faces = PolyCollection(
                initial_body_faces[:, :, projection], facecolors=body_colors,
                edgecolors="darkgoldenrod", linewidths=0.8, zorder=4,
                label="dozer body",
            )
            axis.add_collection(faces)
            body_projections.append((projection, faces))

        def set_body(frame_index):
            faces = body_data[frame_index][:, body_faces].reshape(-1, 4, 3)
            for projection, artist in body_projections:
                artist.set_verts(faces[:, :, projection])

        # Keep the contact lines for diagnostics; render the rounded tracks.
        initial_tracks = track_data[0]

        def track_polygons(frame_index):
            return [vertices[face] for vertices in track_mesh_data[frame_index]
                    for face in track_faces]

        initial_faces = track_polygons(0)
        track_colors = ([track_dark] * profile_size + [body_yellow, body_yellow]) * 2
        track_top = PolyCollection(
            [face[:, [0, 1]] for face in initial_faces], facecolors=track_colors,
            edgecolors="#202326", linewidths=0.8, zorder=5, label="tracks",
        )
        track_back = PolyCollection(
            [face[:, [1, 2]] for face in initial_faces], facecolors=track_colors,
            edgecolors="#202326", linewidths=0.8, zorder=5, label="tracks",
        )
        track_side = PolyCollection(
            [face[:, [0, 2]] for face in initial_faces], facecolors=track_colors,
            edgecolors="#202326", linewidths=0.8, zorder=5, label="tracks",
        )
        axis_top.add_collection(track_top)
        axis_back.add_collection(track_back)
        axis_side.add_collection(track_side)

        def set_tracks(frame_index):
            faces = track_polygons(frame_index)
            track_top.set_verts([face[:, [0, 1]] for face in faces])
            track_back.set_verts([face[:, [1, 2]] for face in faces])
            track_side.set_verts([face[:, [0, 2]] for face in faces])

        green_3d = green_top = green_back = green_side = None
        if show_neighbors:
            green_3d = axis_3d.scatter([], [], [], color="green", s=40, zorder=5)
            green_top = axis_top.scatter([], [], color="green", s=40, zorder=5)
            green_back = axis_back.scatter([], [], color="green", s=40, zorder=5)
            green_side = axis_side.scatter([], [], color="green", s=40, zorder=5)

        arrow_length = simulation.subdivision * 0.6

        def forward(frame_index):
            return simulation._rotation_lg(*data[frame_index, 4:7])[:, 0]

        initial_forward = forward(0)
        q_arrow_top = axis_top.quiver(
            data[0, 1], data[0, 2], initial_forward[0], initial_forward[1],
            color="blue", scale=1 / arrow_length, scale_units="xy",
            angles="xy", zorder=7,
        )
        q_arrow_back = axis_back.quiver(
            data[0, 2], data[0, 3], initial_forward[1], initial_forward[2],
            color="blue", scale=1 / arrow_length, scale_units="xy",
            angles="xy", zorder=7,
        )
        q_arrow_side = axis_side.quiver(
            data[0, 1], data[0, 3], initial_forward[0], initial_forward[2],
            color="blue", scale=1 / arrow_length, scale_units="xy",
            angles="xy", zorder=7,
        )
        q_arrow_3d = [None]

        if not enable_blade:
            # Body-only mode keeps the diagnostic arrows from the original
            # visualization: forward orientation at all six track points, plus the
            # lateral offsets from the center of mass to both tracks.
            initial_track_points = initial_tracks.reshape(-1, 3)
            body_arrow_top = axis_top.quiver(
                initial_track_points[:, 0], initial_track_points[:, 1],
                np.full(6, initial_forward[0]),
                np.full(6, initial_forward[1]),
                color="darkorange", scale=1 / arrow_length, scale_units="xy",
                angles="xy", zorder=6,
            )
            body_arrow_back = axis_back.quiver(
                initial_track_points[:, 1], initial_track_points[:, 2],
                np.full(6, initial_forward[1]),
                np.full(6, initial_forward[2]),
                color="darkorange", scale=1 / arrow_length, scale_units="xy",
                angles="xy", zorder=6,
            )
            body_arrow_side = axis_side.quiver(
                initial_track_points[:, 0], initial_track_points[:, 2],
                np.full(6, initial_forward[0]),
                np.full(6, initial_forward[2]),
                color="darkorange", scale=1 / arrow_length, scale_units="xy",
                angles="xy", zorder=6, label="track orientations",
            )
            body_arrow_3d = [None]

            initial_lateral = (
                simulation._rotation_lg(*data[0, 4:7])
                @ np.array([0.0, simulation.b / 2, 0.0])
            )
            q_lateral_top = axis_top.quiver(
                [data[0, 1]] * 2, [data[0, 2]] * 2,
                [initial_lateral[0], -initial_lateral[0]],
                [initial_lateral[1], -initial_lateral[1]],
                color="purple", scale=1, scale_units="xy",
                angles="xy", zorder=7,
            )
            q_lateral_back = axis_back.quiver(
                [data[0, 2]] * 2, [data[0, 3]] * 2,
                [initial_lateral[1], -initial_lateral[1]],
                [initial_lateral[2], -initial_lateral[2]],
                color="purple", scale=1, scale_units="xy",
                angles="xy", zorder=7,
            )
            q_lateral_side = axis_side.quiver(
                [data[0, 1]] * 2, [data[0, 3]] * 2,
                [initial_lateral[0], -initial_lateral[0]],
                [initial_lateral[2], -initial_lateral[2]],
                color="purple", scale=1, scale_units="xy",
                angles="xy", zorder=7, label="track offsets",
            )
            q_lateral_3d = [None]

        def blade_face_color(face, projection=None):
            # The vertex order faces blade-local +x (toward the soil).
            # Counterclockwise screen winding exposes the front; clockwise
            # exposes the back. Projection also handles the isometric camera.
            if projection is None:
                x, y, _ = proj3d.proj_transform(*face.T, axis_3d.get_proj())
            else:
                x, y = face[:, projection].T
            signed_area = np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
            return "grey" if signed_area >= 0.0 else body_yellow

        if enable_blade:
            # Match the controller's reference at the blade bottom center.
            # Use each logged target, so later reference changes are preserved.
            depth_samples = simulation.force_log[::2]
            desired_cut_heights = np.array([
                sample["starting_surface_height"] - sample["desired_depth"]
                for sample in depth_samples
            ])
            desired_depth_line = axis_back.axhline(
                desired_cut_heights[0], color="red", linewidth=1.8, zorder=10,
                label="Desired cutting depth",
            )
            axis_back.legend(handles=[desired_depth_line], loc="upper right", fontsize=8)
            initial_blade = blade_data[0]
            blade_arrow_top = axis_top.quiver(
                initial_blade[:, 0], initial_blade[:, 1],
                np.full(len(initial_blade), initial_forward[0]),
                np.full(len(initial_blade), initial_forward[1]),
                color="grey", scale=1 / arrow_length, scale_units="xy",
                angles="xy", zorder=7,
            )
            blade_arrow_back = axis_back.quiver(
                initial_blade[:, 1], initial_blade[:, 2],
                np.full(len(initial_blade), initial_forward[1]),
                np.full(len(initial_blade), initial_forward[2]),
                color="grey", scale=1 / arrow_length, scale_units="xy",
                angles="xy", zorder=7,
            )
            blade_arrow_side = axis_side.quiver(
                initial_blade[:, 0], initial_blade[:, 2],
                np.full(len(initial_blade), initial_forward[0]),
                np.full(len(initial_blade), initial_forward[2]),
                color="grey", scale=1 / arrow_length, scale_units="xy",
                angles="xy", zorder=7,
            )
            blade_arrow_3d = [None]

            initial_blade_top = blade_top_data[0]
            initial_blade_face = np.vstack([
                initial_blade[0, :3],
                initial_blade[-1, :3],
                initial_blade_top[-1],
                initial_blade_top[0],
            ])
            blade_face_top = Polygon(
                initial_blade_face[:, [0, 1]], closed=True,
                facecolor="grey", edgecolor="#4a4a4a", alpha=1.0, zorder=6,
            )
            blade_face_back = Polygon(
                initial_blade_face[:, [1, 2]], closed=True,
                facecolor="grey", edgecolor="#4a4a4a", alpha=1.0, zorder=6,
            )
            blade_face_side = Polygon(
                initial_blade_face[:, [0, 2]], closed=True,
                facecolor="grey", edgecolor="#4a4a4a", alpha=1.0, zorder=6,
                label="blade face",
            )
            axis_top.add_patch(blade_face_top)
            axis_back.add_patch(blade_face_back)
            axis_side.add_patch(blade_face_side)

        pile_projections = []
        tooth_projections = []
        if enable_blade:
            pile_3d = Poly3DCollection([], facecolors="saddlebrown", edgecolors="none",
                                       alpha=1.0, linewidths=0, antialiased=False, label="soil pile")
            axis_3d.add_collection3d(pile_3d)
            tooth_3d = Poly3DCollection([], facecolors=soil_top_color, edgecolors="none",
                                        alpha=1.0, linewidths=0, antialiased=False,
                                        label="surface tooth")
            axis_3d.add_collection3d(tooth_3d)
            for axis, projection in ((axis_top, [0, 1]), (axis_back, [1, 2]),
                                     (axis_side, [0, 2])):
                artist = PolyCollection([], facecolors="saddlebrown", edgecolors="none",
                                        alpha=1.0, linewidths=0, antialiased=False, label="soil pile")
                axis.add_collection(artist)
                pile_projections.append((projection, artist))
                tooth = PolyCollection([], facecolors=soil_top_color, edgecolors="none",
                                       alpha=1.0, linewidths=0, antialiased=False,
                                       label="surface tooth")
                axis.add_collection(tooth)
                tooth_projections.append((projection, tooth))

            def set_pile(frame_index):
                vertices = pile_data[frame_index]
                faces = self._pile_faces(vertices)
                pile_3d.set_verts(faces)
                for projection, artist in pile_projections:
                    artist.set_verts([face[:, projection] for face in faces])
                yaw = history[frame_index * 2][3] if len(history) == len(log_data) else 0.0
                tooth_faces = self._surface_tooth_faces(
                    blade_data[frame_index], grid_z_data[frame_index], yaw)
                tooth_3d.set_verts(tooth_faces)
                for projection, artist in tooth_projections:
                    artist.set_verts([face[:, projection] for face in tooth_faces])

            set_pile(0)

        arm_projections = []
        if enable_blade:
            initial_arms = arm_mesh_data[0][:, body_faces].reshape(-1, 4, 3)
            for axis, projection in ((axis_top, [0, 1]), (axis_back, [1, 2]),
                                     (axis_side, [0, 2])):
                artist = PolyCollection(
                    initial_arms[:, :, projection], facecolors=track_dark,
                    edgecolors="#202326", linewidths=0.6, zorder=4,
                    label="blade arms",
                )
                axis.add_collection(artist)
                arm_projections.append((projection, artist))

        # Gather the vehicle faces and colors in rear/hood/right-track/left-track
        # order, followed by two arms and the blade when enabled.
        def vehicle_polygons(frame_index):
            faces = list(body_data[frame_index][:, body_faces].reshape(-1, 4, 3))
            faces.extend(track_polygons(frame_index))
            if enable_blade:
                faces.extend(arm_mesh_data[frame_index][:, body_faces].reshape(-1, 4, 3))
                blade, top = blade_data[frame_index], blade_top_data[frame_index]
                faces.append(np.vstack([blade[0, :3], blade[-1, :3], top[-1], top[0]]))
            return faces

        vehicle_colors = [to_rgba(color, 1.0) for color in body_colors + track_colors]
        vehicle_edges = ([to_rgba("darkgoldenrod")] * len(body_colors)
                         + [to_rgba("#202326")] * len(track_colors))
        if enable_blade:
            vehicle_colors.extend([to_rgba(track_dark)] * (2 * len(body_faces)))
            vehicle_edges.extend([to_rgba("#202326")] * (2 * len(body_faces)))
            vehicle_colors.append(to_rgba("grey", 1.0))
            vehicle_edges.append(to_rgba("#4a4a4a", 1.0))
        # Keep convex parts separate. Their separating planes give a reliable
        # draw order; average depths of long faces can wrongly cover a track.
        part_sizes = [len(body_faces), len(body_faces), len(track_faces), len(track_faces)]
        if enable_blade:
            part_sizes.extend([len(body_faces), len(body_faces), 1])
        part_offsets = np.r_[0, np.cumsum(part_sizes)]
        part_slices = [slice(start, end) for start, end in zip(part_offsets[:-1], part_offsets[1:])]
        initial_vehicle = vehicle_polygons(0)
        vehicle_parts = []
        for section in part_slices:
            artist = Poly3DCollection(
                initial_vehicle[section], facecolors=vehicle_colors[section],
                edgecolors=vehicle_edges[section], linewidths=0.8,
                zsort="average", zorder=5, label="dozer",
            )
            axis_3d.add_collection3d(artist)
            vehicle_parts.append(artist)

        def ordered_vehicle_parts(frame_index):
            # Recover the camera in world coordinates from the projection.
            eye = np.linalg.solve(axis_3d.get_proj(), [0.0, 0.0, -1.0, 0.0])
            pose = data[frame_index, 1:7]
            if abs(eye[3]) > 1e-12:
                eye_world = eye[:3] / eye[3]
            else:  # Orthographic camera: only the viewing direction matters.
                eye_world = pose[:3] + eye[:3] / np.linalg.norm(eye[:3]) * 1e6
            eye_local = simulation._rotation_lg(*pose[3:6]).T @ (eye_world - pose[:3])

            # Parts 2/3 are the -y/+y tracks. Paint the far track, then the
            # body, then the near track so the body cannot clip the near tread.
            far_track, near_track = (2, 3) if eye_local[1] >= 0.0 else (3, 2)
            rear, hood = (0, 1) if eye_local[0] >= body_split else (1, 0)
            order = [far_track, rear, hood, near_track]
            if enable_blade:
                far_arm, near_arm = (4, 5) if eye_local[1] >= 0.0 else (5, 4)
                order = [far_track, far_arm, rear, hood, near_arm, near_track]
                face = vehicle_polygons(frame_index)[-1]
                body_center = body_data[frame_index].mean(axis=(0, 1))
                pile_in_front = self._pile_in_front_of_blade(face, eye_world, body_center)
                if pile_in_front:
                    order.append(6)
                else:
                    order.insert(0, 6)
            artists = [vehicle_parts[index] for index in order]
            if enable_blade:
                # Render the soil-facing view over the blade; rear views behind it.
                if pile_in_front:
                    artists.extend([tooth_3d, pile_3d])
                else:
                    artists[0:0] = [tooth_3d, pile_3d]
            return artists

        def set_neighbors(frame_index):
            points = (
                np.asarray(neighbor_data[frame_index])
                if len(neighbor_data[frame_index])
                else np.empty((0, 3))
            )
            green_3d._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
            green_top.set_offsets(points[:, [0, 1]])
            green_back.set_offsets(points[:, [1, 2]])
            green_side.set_offsets(points[:, [0, 2]])

        def set_body_diagnostics(frame_index):
            points = track_data[frame_index].reshape(-1, 3)
            direction = forward(frame_index)
            body_arrow_top.set_offsets(points[:, [0, 1]])
            body_arrow_top.set_UVC(
                np.full(len(points), direction[0]),
                np.full(len(points), direction[1]),
            )
            body_arrow_back.set_offsets(points[:, [1, 2]])
            body_arrow_back.set_UVC(
                np.full(len(points), direction[1]),
                np.full(len(points), direction[2]),
            )
            body_arrow_side.set_offsets(points[:, [0, 2]])
            body_arrow_side.set_UVC(
                np.full(len(points), direction[0]),
                np.full(len(points), direction[2]),
            )
            if body_arrow_3d[0] is not None:
                body_arrow_3d[0].remove()
            body_arrow_3d[0] = axis_3d.quiver(
                points[:, 0], points[:, 1], points[:, 2],
                np.full(len(points), direction[0]),
                np.full(len(points), direction[1]),
                np.full(len(points), direction[2]),
                length=arrow_length, color="darkorange", zorder=6,
            )

            x, y, z = data[frame_index, 1:4]
            lateral = (
                simulation._rotation_lg(*data[frame_index, 4:7])
                @ np.array([0.0, simulation.b / 2, 0.0])
            )
            q_lateral_top.set_offsets([[x, y], [x, y]])
            q_lateral_top.set_UVC(
                [lateral[0], -lateral[0]], [lateral[1], -lateral[1]]
            )
            q_lateral_back.set_offsets([[y, z], [y, z]])
            q_lateral_back.set_UVC(
                [lateral[1], -lateral[1]], [lateral[2], -lateral[2]]
            )
            q_lateral_side.set_offsets([[x, z], [x, z]])
            q_lateral_side.set_UVC(
                [lateral[0], -lateral[0]], [lateral[2], -lateral[2]]
            )
            if q_lateral_3d[0] is not None:
                q_lateral_3d[0].remove()
            q_lateral_3d[0] = axis_3d.quiver(
                [x, x], [y, y], [z, z],
                [lateral[0], -lateral[0]],
                [lateral[1], -lateral[1]],
                [lateral[2], -lateral[2]],
                length=1, color="purple", zorder=7,
            )

        def set_q_and_blade(frame_index):
            x, y, z = data[frame_index, 1:4]
            direction = forward(frame_index)
            q_arrow_top.set_offsets([[x, y]])
            q_arrow_top.set_UVC(direction[0], direction[1])
            q_arrow_back.set_offsets([[y, z]])
            q_arrow_back.set_UVC(direction[1], direction[2])
            q_arrow_side.set_offsets([[x, z]])
            q_arrow_side.set_UVC(direction[0], direction[2])
            if q_arrow_3d[0] is not None:
                q_arrow_3d[0].remove()
            q_arrow_3d[0] = axis_3d.quiver(
                x, y, z, direction[0], direction[1], direction[2],
                length=arrow_length, color="blue", zorder=7,
            )

            if not enable_blade:
                return

            blade = blade_data[frame_index]
            blade_arrow_top.set_offsets(blade[:, [0, 1]])
            blade_arrow_top.set_UVC(
                np.full(len(blade), direction[0]),
                np.full(len(blade), direction[1]),
            )
            blade_arrow_back.set_offsets(blade[:, [1, 2]])
            blade_arrow_back.set_UVC(
                np.full(len(blade), direction[1]),
                np.full(len(blade), direction[2]),
            )
            blade_arrow_side.set_offsets(blade[:, [0, 2]])
            blade_arrow_side.set_UVC(
                np.full(len(blade), direction[0]),
                np.full(len(blade), direction[2]),
            )
            if blade_arrow_3d[0] is not None:
                blade_arrow_3d[0].remove()
            blade_arrow_3d[0] = axis_3d.quiver(
                blade[:, 0], blade[:, 1], blade[:, 2],
                np.full(len(blade), direction[0]),
                np.full(len(blade), direction[1]),
                np.full(len(blade), direction[2]),
                length=arrow_length, color="grey", zorder=7,
            )

            blade_top = blade_top_data[frame_index]
            blade_face = np.vstack([
                blade[0, :3], blade[-1, :3], blade_top[-1], blade_top[0]
            ])
            blade_face_top.set_xy(blade_face[:, [0, 1]])
            blade_face_back.set_xy(blade_face[:, [1, 2]])
            blade_face_side.set_xy(blade_face[:, [0, 2]])
            for patch, projection in ((blade_face_top, [0, 1]),
                                      (blade_face_back, [1, 2])):
                patch.set_facecolor(blade_face_color(blade_face, projection))
            # Show the grey working face in the bottom-left panel.
            blade_face_side.set_facecolor("grey")
            vehicle_colors[-1] = to_rgba(blade_face_color(blade_face), 1.0)
            vehicle_parts[-1].set_facecolor([vehicle_colors[-1]])

        if show_neighbors:
            set_neighbors(0)
        set_grid(0)
        set_tracks(0)
        if not enable_blade:
            set_body_diagnostics(0)
        set_q_and_blade(0)

        legend_handles = [body_projections[-1][1], track_side, q_arrow_side]
        legend_labels = ["dozer body", "tracks", "q (center of mass)"]
        if enable_blade:
            legend_handles.extend([
                blade_arrow_side, blade_face_side, pile_projections[-1][1]
            ])
            legend_labels.extend([
                "blade contact points", "blade face", "soil pile"
            ])
        if show_neighbors:
            legend_handles.append(green_side)
            legend_labels.append("grid neighbors")
        axis_side.legend(
            legend_handles, legend_labels, loc="upper right", fontsize=8
        )

        # Let constrained layout converge, then freeze the panel positions.
        axis_top.set_title(f"t = {data[0, 0]:.2f} s")
        figure.canvas.draw()
        axis_3d.set_xticks([
            tick for tick in axis_top.get_xticks()
            if center_x - half_x <= tick <= center_x + half_x
        ])
        axis_3d.set_yticks([
            tick for tick in axis_top.get_yticks()
            if center_y - half_y <= tick <= center_y + half_y
        ])
        axis_3d.set_zticks([
            tick for tick in axis_side.get_yticks()
            if center_z - half_z <= tick <= center_z + half_z
        ])
        for _ in range(2):
            figure.canvas.draw()
        figure.set_layout_engine("none")

        def update(frame_index):
            polygons = vehicle_polygons(frame_index)
            for artist, section in zip(vehicle_parts, part_slices):
                artist.set_verts(polygons[section])
            set_body(frame_index)
            set_grid(frame_index)
            set_tracks(frame_index)
            if show_neighbors:
                set_neighbors(frame_index)
            if not enable_blade:
                set_body_diagnostics(frame_index)
            set_q_and_blade(frame_index)
            if enable_blade:
                set_pile(frame_index)
                desired_depth_line.set_ydata([desired_cut_heights[frame_index]] * 2)
                arms = arm_mesh_data[frame_index][:, body_faces].reshape(-1, 4, 3)
                for projection, artist in arm_projections:
                    artist.set_verts(arms[:, :, projection])
            axis_top.set_title(f"t = {data[frame_index, 0]:.2f} s")

        flat_artists = [
            grid_back, grid_side, soil_walls_back, soil_walls_side,
            *[artist for _, artist in tooth_projections],
            *[artist for _, artist in pile_projections],
            *[artist for _, artist in arm_projections],
            *[artist for _, artist in body_projections],
            track_top, track_back, track_side,
        ]
        three_d_artists = [soil_top_3d, grid_3d, soil_walls_3d, *vehicle_parts]
        if enable_blade:
            three_d_artists.extend([tooth_3d, pile_3d])
        if show_neighbors:
            flat_artists.extend([green_back, green_side, green_top])
            three_d_artists.append(green_3d)
        if enable_blade:
            flat_artists.extend([
                blade_face_top, blade_face_back, blade_face_side,
            ])
        else:
            flat_artists.extend([
                body_arrow_top, body_arrow_back, body_arrow_side,
                q_lateral_top, q_lateral_back, q_lateral_side,
            ])
        flat_artists.extend([
            q_arrow_top, q_arrow_back, q_arrow_side,
        ])
        if enable_blade:
            flat_artists.extend([
                blade_arrow_top, blade_arrow_back, blade_arrow_side,
            ])

        if enable_blade:
            # Draw the reference last so soil and vehicle faces cannot hide it.
            flat_artists.append(desired_depth_line)

        # Cache all frame-invariant content once. 3D quivers are replaced per
        # frame because Matplotlib does not support updating them in place.
        q_arrow_3d[0].remove()
        q_arrow_3d[0] = None
        if enable_blade:
            blade_arrow_3d[0].remove()
            blade_arrow_3d[0] = None
        else:
            body_arrow_3d[0].remove()
            body_arrow_3d[0] = None
            q_lateral_3d[0].remove()
            q_lateral_3d[0] = None
        cached_artists = flat_artists + three_d_artists
        for artist in cached_artists:
            artist.set_visible(False)
        axis_top.set_title("")
        figure.canvas.draw()
        background = figure.canvas.copy_from_bbox(figure.bbox)
        for artist in cached_artists:
            artist.set_visible(True)

        fps = 30 if enable_blade else 20
        frame_size = figure.canvas.get_width_height()
        frames = []
        for frame_index in range(len(data)):
            update(frame_index)
            figure.canvas.restore_region(background)

            for artist in flat_artists:
                artist.axes.draw_artist(artist)

            dynamic_3d_artists = []
            if not enable_blade:
                dynamic_3d_artists.extend([
                    body_arrow_3d[0], q_lateral_3d[0]
                ])
            dynamic_3d_artists.append(q_arrow_3d[0])
            if enable_blade:
                dynamic_3d_artists.append(blade_arrow_3d[0])
            # Draw diagnostic arrows beneath the solid vehicle so they do not
            # appear through its opaque body and track faces.
            scene_artists = [artist for artist in three_d_artists
                             if artist not in vehicle_parts
                             and (not enable_blade or artist not in (pile_3d, tooth_3d))]
            for artist in scene_artists + dynamic_3d_artists + ordered_vehicle_parts(frame_index):
                artist.do_3d_projection()
                axis_3d.draw_artist(artist)
            axis_top.draw_artist(axis_top.title)

            frames.append(Image.frombuffer(
                "RGBA",
                frame_size,
                figure.canvas.buffer_rgba(),
                "raw",
                "RGBA",
                0,
                1,
            ).convert("RGB"))

        palette = frames[0].convert("P", palette=Image.ADAPTIVE, colors=128)
        frames = [
            frame.quantize(palette=palette, dither=Image.NONE)
            for frame in frames
        ]

        filename = "figures/simulation.gif"
        frames[0].save(
            filename,
            save_all=True,
            append_images=frames[1:],
            duration=int(1_000 / fps),
            loop=0,
        )
        plt.close(figure)
        print(f"Saved {filename}")
