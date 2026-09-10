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
from PIL import Image


class Visualization:
    """Visualize a simulation with optional blade-specific layers.

    The simulation's ``enable_blade`` setting is the single source of truth for
    selecting blade geometry, deforming terrain, and output timing.
    """

    def __init__(self, simulation):
        self.simulation = simulation

    def forces_visualization(self, filename="figures/forces.gif"):
        """Animate logged forces in the 4-by-2 dashboard used by main.py.

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
            ("Blade roll error", "Error (rad)", [("roll_error", "Roll")]),
            ("Track drive forces", "Force (N)", [("drive_left", "Left"), ("drive_right", "Right")]),
        ]
        figure, axes = plt.subplots(4, 2, figsize=(12, 9), sharex=True)
        title = figure.suptitle("Forces & Moments over Time")
        traces, cursors = [], []
        for axis, (name, units, series) in zip(axes.flat, panels):
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
        half_body_width = (simulation.b - simulation.track_width) / 2
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
        else:
            static_grid_z = np.array([[point[2] for point in column]
                                      for column in simulation.grid_pts])
            grid_z_data = np.broadcast_to(
                static_grid_z, (len(data),) + static_grid_z.shape
            )

        # Include all geometry that can appear during the run when choosing
        # view bounds. Blade-only histories are never touched in body-only mode.
        bound_x = [grid_x.ravel(), track_path[:, :, :, 0].ravel()]
        bound_y = [grid_y.ravel(), track_path[:, :, :, 1].ravel()]
        bound_z = [grid_z_data.ravel(), track_path[:, :, :, 2].ravel()]
        bound_x.append(track_mesh_path[:, :, :, 0].ravel())
        bound_y.append(track_mesh_path[:, :, :, 1].ravel())
        bound_z.append(track_mesh_path[:, :, :, 2].ravel())
        bound_x.append(body_path[:, :, :, 0].ravel())
        bound_y.append(body_path[:, :, :, 1].ravel())
        bound_z.append(body_path[:, :, :, 2].ravel())
        if enable_blade:
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

        segments = grid_segments(grid_z_data[0])
        grid_3d = Line3DCollection(
            segments,
            colors="saddlebrown",
            linewidths=0.5,
            alpha=0.5,
            zorder=0,
        )
        grid_back = LineCollection(
            segments[:, :, [1, 2]],
            colors="black",
            linewidths=0.5,
            alpha=0.5,
            zorder=0,
        )
        grid_side = LineCollection(
            segments[:, :, [0, 2]],
            colors="black",
            linewidths=0.5,
            alpha=0.5,
            zorder=0,
        )
        axis_3d.add_collection3d(grid_3d)
        axis_back.add_collection(grid_back)
        axis_side.add_collection(grid_side)
        axis_top.add_collection(LineCollection(
            segments[:, :, [0, 1]],
            colors="black",
            linewidths=0.5,
            alpha=0.5,
            zorder=0,
        ))

        def set_grid(frame_index):
            frame_segments = grid_segments(grid_z_data[frame_index])
            grid_3d.set_segments(frame_segments)
            grid_back.set_segments(frame_segments[:, :, [1, 2]])
            grid_side.set_segments(frame_segments[:, :, [0, 2]])

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
        body_colors = ["#b8860b", "#ffdf50", "#e8b923",
                       "#e8b923", body_yellow, body_yellow] * 2
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
        track_colors = (["#444a50"] * profile_size + [body_yellow, body_yellow]) * 2
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

        if enable_blade:
            initial_blade = blade_data[0]
            blade_arrow_top = axis_top.quiver(
                initial_blade[:, 0], initial_blade[:, 1],
                np.full(len(initial_blade), initial_forward[0]),
                np.full(len(initial_blade), initial_forward[1]),
                color="black", scale=1 / arrow_length, scale_units="xy",
                angles="xy", zorder=7,
            )
            blade_arrow_back = axis_back.quiver(
                initial_blade[:, 1], initial_blade[:, 2],
                np.full(len(initial_blade), initial_forward[1]),
                np.full(len(initial_blade), initial_forward[2]),
                color="black", scale=1 / arrow_length, scale_units="xy",
                angles="xy", zorder=7,
            )
            blade_arrow_side = axis_side.quiver(
                initial_blade[:, 0], initial_blade[:, 2],
                np.full(len(initial_blade), initial_forward[0]),
                np.full(len(initial_blade), initial_forward[2]),
                color="black", scale=1 / arrow_length, scale_units="xy",
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
                facecolor="black", edgecolor="black", alpha=1.0, zorder=6,
            )
            blade_face_back = Polygon(
                initial_blade_face[:, [1, 2]], closed=True,
                facecolor="black", edgecolor="black", alpha=1.0, zorder=6,
            )
            blade_face_side = Polygon(
                initial_blade_face[:, [0, 2]], closed=True,
                facecolor="black", edgecolor="black", alpha=1.0, zorder=6,
                label="blade face",
            )
            axis_top.add_patch(blade_face_top)
            axis_back.add_patch(blade_face_back)
            axis_side.add_patch(blade_face_side)

        # Sort all vehicle faces together by camera depth. Separate collections
        # draw whole parts over each other, making an opaque body look transparent.
        def vehicle_polygons(frame_index):
            faces = list(body_data[frame_index][:, body_faces].reshape(-1, 4, 3))
            faces.extend(track_polygons(frame_index))
            if enable_blade:
                blade, top = blade_data[frame_index], blade_top_data[frame_index]
                faces.append(np.vstack([blade[0, :3], blade[-1, :3], top[-1], top[0]]))
            return faces

        vehicle_colors = [to_rgba(color, 1.0) for color in body_colors + track_colors]
        vehicle_edges = ([to_rgba("darkgoldenrod")] * len(body_colors)
                         + [to_rgba("#202326")] * len(track_colors))
        if enable_blade:
            vehicle_colors.append(to_rgba("black", 1.0))
            vehicle_edges.append(to_rgba("black", 1.0))
        vehicle_3d = Poly3DCollection(
            vehicle_polygons(0), facecolors=vehicle_colors, edgecolors=vehicle_edges,
            linewidths=0.8, zsort="average", zorder=5, label="dozer",
        )
        axis_3d.add_collection3d(vehicle_3d)

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
                length=arrow_length, color="black", zorder=7,
            )

            blade_top = blade_top_data[frame_index]
            blade_face = np.vstack([
                blade[0, :3], blade[-1, :3], blade_top[-1], blade_top[0]
            ])
            blade_face_top.set_xy(blade_face[:, [0, 1]])
            blade_face_back.set_xy(blade_face[:, [1, 2]])
            blade_face_side.set_xy(blade_face[:, [0, 2]])

        if show_neighbors:
            set_neighbors(0)
        set_tracks(0)
        if not enable_blade:
            set_body_diagnostics(0)
        set_q_and_blade(0)

        legend_handles = [body_projections[-1][1], track_side, q_arrow_side]
        legend_labels = ["dozer body", "tracks", "q (center of mass)"]
        if enable_blade:
            legend_handles.extend([
                blade_arrow_side, blade_face_side
            ])
            legend_labels.extend([
                "blade contact points", "blade face"
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
            vehicle_3d.set_verts(vehicle_polygons(frame_index))
            set_body(frame_index)
            set_grid(frame_index)
            set_tracks(frame_index)
            if show_neighbors:
                set_neighbors(frame_index)
            if not enable_blade:
                set_body_diagnostics(frame_index)
            set_q_and_blade(frame_index)
            axis_top.set_title(f"t = {data[frame_index, 0]:.2f} s")

        flat_artists = [
            grid_back, grid_side,
            *[artist for _, artist in body_projections],
            track_top, track_back, track_side,
        ]
        three_d_artists = [grid_3d, vehicle_3d]
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
            scene_artists = [artist for artist in three_d_artists if artist is not vehicle_3d]
            for artist in scene_artists + dynamic_3d_artists + [vehicle_3d]:
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
