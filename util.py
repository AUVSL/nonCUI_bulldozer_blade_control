import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image
from matplotlib.patches import Polygon


class _SurfaceBase:
    """Terrain geometry shared by both modes of :class:`Surface`.

    The defaults follow blade-enabled behavior. ``_BodyMode`` customizes the
    few places where tracked-body terrain/contact behavior differs.
    """

    @property
    def subdivision(self):
        """Grid spacing, sized relative to the dozer width ``self.b``."""
        return self.b / self._subdivision_factor()

    def _subdivision_factor(self):
        """Blade-default number of grid divisions across the dozer width."""
        return self.division_factor

    def _surface_grid(self):
        R_surf = self._rotation_lg(*self.surface_abg)
        e1, e2, e3 = R_surf[:, 0], R_surf[:, 1], R_surf[:, 2]
        grid = nx.grid_2d_graph(len(self.us), len(self.vs))
        u_start = self.us[self.us <= self.u_split][-1]
        v_start = self.vs[self.vs <= self.v_split][-1]
        ramp_width = max(1, round(self.transition_tiles)) * self.subdivision

        for i, u in enumerate(self.us):
            for j, v in enumerate(self.vs):
                # Full offset before each split, followed by a linear ramp.
                u_clip = np.clip((u - u_start) / ramp_width, 0.0, 1.0)
                v_clip = np.clip((v - v_start) / ramp_width, 0.0, 1.0)
                weight = self._surface_weight(u_clip, v_clip)
                x, y, z = u * e1 + v * e2 + weight * self.offset * e3

                node = grid.nodes[(i, j)]
                node["x"], node["y"], node["z"] = float(x), float(y), float(z)
                self._initialize_surface_node(node)
        return grid

    def _surface_weight(self, u_clip, v_clip):
        """Blade-default blend for the transition in the travel direction."""
        weight = u_clip if self.is_surface_pitched else v_clip
        if self.is_surface_rolled:
            weight += v_clip if self.is_surface_pitched else u_clip
        return weight

    def _initialize_surface_node(self, node):
        """Add subclass-specific metadata to a newly created grid node."""
        return None

    def _rotation_lg(self, a, B, g):
        """Rotation matrix: local to global frame."""
        return self._rotation_gl(a, B, g).T

    def _rotation_gl(self, a, B, g):
        """Rotation matrix: global to local frame."""
        sa, ca = np.sin(a), np.cos(a)
        sB, cB = np.sin(B), np.cos(B)
        sg, cg = np.sin(g), np.cos(g)

        return np.array([
            [cB * cg, sa * sB * cg - ca * sg, ca * sB * cg + sa * sg],
            [cB * sg, sa * sB * sg + ca * cg, ca * sB * sg - sa * cg],
            [-sB, sa * cB, ca * cB],
        ]).T

    def _point_orientation(self, corners):
        """Return terrain-fitted roll, pitch, and yaw at ``self.q``.

        Roll and pitch follow the terrain gradient relative to the direction
        of travel; yaw follows that direction of travel.
        """
        grad_xy = self._bilinear_gradient(self.q, corners)

        vel = np.array(self.q_dot[:3])
        if self.is_backwards:
            vel *= -1
        speed = np.linalg.norm(vel[:2])
        if speed < 1e-9:
            return self.q[3:6].copy()

        fwd_xy = vel[:2] / speed
        s_f = np.dot(grad_xy, fwd_xy)
        roll = self._surface_roll(grad_xy, fwd_xy, s_f)
        pitch = np.arctan2(-s_f, 1.0)
        yaw = np.arctan2(vel[1], vel[0])
        return np.array([roll, pitch, yaw])

    def _surface_roll(self, grad_xy, fwd_xy, forward_slope):
        """Return terrain-induced roll about the direction of travel."""
        left_xy = np.array([-fwd_xy[1], fwd_xy[0]])
        lateral_slope = np.dot(grad_xy, left_xy)
        return np.arctan2(lateral_slope, np.sqrt(1.0 + forward_slope**2))

    def _bilinear_gradient(self, point, corners):
        """Gradient of the bilinear height patch in the global xy-plane."""
        (x1, y1, h1), (x2, y2, h2), (x3, y3, h3), (x4, y4, h4) = corners

        dx, dy = point[0] - x1, point[1] - y1
        e_s = np.array([x2 - x1, y2 - y1])
        e_t = np.array([x3 - x1, y3 - y1])
        es2, et2 = e_s @ e_s, e_t @ e_t

        s = min(max((dx * e_s[0] + dy * e_s[1]) / es2, 0.0), 1.0)
        t = min(max((dx * e_t[0] + dy * e_t[1]) / et2, 0.0), 1.0)

        dh_ds = (h2 - h1) * (1 - t) + (h4 - h3) * t
        dh_dt = (h3 - h1) * (1 - s) + (h4 - h2) * s
        return dh_ds * e_s / es2 + dh_dt * e_t / et2

    def _point_height(self, point):
        neighbor_points = self._get_neighbor_points(point)
        height_to_surface = self._bilinear_height(point, neighbor_points)
        return neighbor_points, height_to_surface

    def _grid_cell(self, point):
        """Index of the grid tile containing point's xy position, clipped."""
        i = int((point[0] - self.us[0]) // self.subdivision)
        j = int((point[1] - self.vs[0]) // self.subdivision)
        i_max, j_max = len(self.us) - 2, len(self.vs) - 2
        i = 0 if i < 0 else i_max if i > i_max else i
        j = 0 if j < 0 else j_max if j > j_max else j
        return i, j

    def _get_neighbor_points(self, point):
        i, j = self._grid_cell(point)
        col_i, col_i1 = self.grid_pts[i], self.grid_pts[i + 1]
        return [col_i[j], col_i1[j], col_i[j + 1], col_i1[j + 1]]

    def _bilinear_height(self, point, corners):
        """Bilinearly interpolate height from a tile's four corner vertices."""
        (x1, y1, h1), (x2, y2, h2), (x3, y3, h3), (x4, y4, h4) = corners

        dx, dy = point[0] - x1, point[1] - y1
        esx, esy = x2 - x1, y2 - y1
        etx, ety = x3 - x1, y3 - y1
        s = (dx * esx + dy * esy) / (esx * esx + esy * esy)
        t = (dx * etx + dy * ety) / (etx * etx + ety * ety)
        s = 0.0 if s < 0.0 else 1.0 if s > 1.0 else s
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t

        return (
            h1 * (1 - s) * (1 - t)
            + h2 * s * (1 - t)
            + h3 * (1 - s) * t
            + h4 * s * t
        )

    def _contact_angle(self, imbalance, fitted):
        """Solve for a balanced contact angle using Blade's root solver (the Illinois method of Regula Falsi)."""
        limit, span = np.pi / 2 - 1e-3, 0.25
        while True:
            lo, hi = max(fitted - span, -limit), min(fitted + span, limit)
            f_lo, f_hi = imbalance(lo), imbalance(hi)
            if f_lo * f_hi <= 0:
                break
            if lo <= -limit and hi >= limit:
                return fitted
            span *= 2

        angle = fitted
        for _ in range(60):
            if f_hi == f_lo:
                break
            angle = (lo * f_hi - hi * f_lo) / (f_hi - f_lo)
            value = imbalance(angle)
            if abs(value) < self.contact_tol or hi - lo < self.contact_tol:
                break
            if value * f_lo > 0:
                lo, f_lo = angle, value
                f_hi *= 0.5
            else:
                hi, f_hi = angle, value
                f_lo *= 0.5
        return angle


class _BodyMode(_SurfaceBase):
    def _initialize_body_mode(self, is_uphill=True, is_surface_pitched: bool = False,
                              is_surface_rolled: bool = False, is_backwards: bool = False):
        self.enable_blade = False
        # simulation parameters
        self.b                  = 1.75
        self.offset             = np.array([0, 0, self.b]) if is_uphill else np.array([0, 0, -self.b])
        self.surface_abg        = np.array([ 0.0, 0.0, 0.0])
        self.u_split            = 0  # u-value where the grid switches to surface_abg2
        self.v_split            = 2  # u-value where the grid switches to surface_abg2
        self.transition_tiles   = self.b                      # tiles over which the offset ramps down past u_split
        self.q                  = np.array([0.0, 0.0, 0.0, self.surface_abg[0], self.surface_abg[1], 0.0 if is_surface_pitched else np.pi / 2])
        self.q_dot              = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]) if is_surface_pitched else np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0]) 
        self.is_initalization   = True

        self.l                  = 2.349
        self.stop_time          = 300.0
        self.total_distance     = 0.0
        self.dt                 = 1/100  
        if is_backwards:
            self.q_dot   *= -1
            self.u_split *= self.q_dot[0] / np.linalg.norm(self.q_dot)
            self.v_split *= self.q_dot[1] / np.linalg.norm(self.q_dot)
        self.is_backwards       = is_backwards
        self.is_surface_pitched = is_surface_pitched
        self.is_surface_rolled = is_surface_rolled
        # contact-angle root-find tolerance (rad of angle / m of hang). Must stay
        # tighter than the settle loop's 1e-6 convergence check, or the pose it
        # returns is noisier than that check and the settle loop never converges
        # -- which burns *more* passes, so loosening this is a net slowdown.
        self.contact_tol        = 1e-7
        self.log                = []
        self.neighbor_points    = []
        self.neighbor_log       = []
        self.point_log          = []  # per frame: (3, 6) rows front, center, back
        
        # set up the surface grid
        self.u_range       = (-2* self.b/2, 4 * self.b) if is_surface_pitched else (-self.b/2,     self.b/2) 
        self.v_range       = (-self.b/2,   self.b/2) if is_surface_pitched else (-2* self.b/2, 4 * self.b)
        if is_backwards:
            self.u_range       = (-2.5 * self.b, self.b/2) if is_surface_pitched else (-self.b/2,     self.b/2) 
            self.v_range       = (-self.b/2,     self.b/2) if is_surface_pitched else (-2.5 * self.b, self.b/2)
        self.us            = np.arange(self.u_range[0], self.u_range[1] + self.subdivision, self.subdivision)
        self.vs            = np.arange(self.v_range[0], self.v_range[1] + self.subdivision, self.subdivision)
        
        stop_index         = 0 if self.is_backwards else -1 
        self.stop_distance = abs(self.us[stop_index]) if is_surface_pitched else abs(self.vs[stop_index])
        
        self.surf_grid     = self._surface_grid()

        # node (x, y, z) cached as plain Python floats, indexed [i][j], so the
        # hot height lookups skip the networkx attribute dicts entirely
        self.grid_pts = [[(self.surf_grid.nodes[(i, j)]['x'],
                           self.surf_grid.nodes[(i, j)]['y'],
                           self.surf_grid.nodes[(i, j)]['z'])
                          for j in range(len(self.vs))] for i in range(len(self.us))]

        # put track on the surface at the start of the simulation
        points, neighbor_points = self._body_update()
        self.log.append([0, *self.q])
        self.point_log.append(np.array(points[1:])) 
        self.neighbor_log.append(np.vstack(neighbor_points)) 

    def _subdivision_factor(self):
        if self.enable_blade:
            return super()._subdivision_factor()
        return 2.0

    def _surface_weight(self, u_clip, v_clip):
        """Preserve Body's historical two-axis transition ramp."""
        if self.enable_blade:
            return super()._surface_weight(u_clip, v_clip)
        return u_clip + v_clip

    def _initialize_surface_node(self, node):
        if self.enable_blade:
            return super()._initialize_surface_node(node)
        node["visited_last"] = False

    def _render_body_run(self, show_neighbors: bool = False):
        print("Rendering GIF...")
        
        # set axis limits based on the logged data (and neighbor points, if any)
        margin = 0.5
        data          = np.array(self.log)[::2] # every 2nd frame represented to speed up rendering
        point_data    = np.array(self.point_log)[::2]  # (frames, 6, 6): right front/center/back then left front/center/back
        front_data,  center_data,  back_data  = point_data[:, 0], point_data[:, 1], point_data[:, 2]  # right track
        lfront_data, lcenter_data, lback_data = point_data[:, 3], point_data[:, 4], point_data[:, 5]  # left track
        track_pts     = point_data.reshape(-1, point_data.shape[-1])  # (frames*6, 6): every logged track point
        neighbor_data = self.neighbor_log[::2]
        grid_pts = np.array([[self.surf_grid.nodes[n]['x'],
                              self.surf_grid.nodes[n]['y'],
                              self.surf_grid.nodes[n]['z']] for n in self.surf_grid.nodes])

        # bound the view with the track path and the surface grid (so it isn't
        # clipped flush at a panel edge), plus the grid-neighbor points only when
        # they will actually be drawn
        bound_pts = [track_pts[:, :3], grid_pts]
        if show_neighbors:
            bound_pts.append(np.array([pt for frame in self.neighbor_log for pt in frame])[:, :3])
        bound_pts = np.vstack(bound_pts)

        all_x = np.concatenate([data[:, 1], bound_pts[:, 0]])
        all_y = np.concatenate([data[:, 2], bound_pts[:, 1]])
        all_z = np.concatenate([data[:, 3], bound_pts[:, 2]])
        
        cx     = (all_x.max() + all_x.min()) / 2
        cy     = (all_y.max() + all_y.min()) / 2
        cz     = (all_z.max() + all_z.min()) / 2
        half_x = (all_x.max() - all_x.min()) / 2 + margin
        half_y = (all_y.max() - all_y.min()) / 2 + margin
        half_z = (all_z.max() - all_z.min()) / 2 + margin

        # set up the figure and axes for the animation
        # 2x2 layout: top view (X-Y) | 3D view
        #             side view (X-Z)| back view (Y-Z)
        # row/column ratios match the per-axis data spans so each equal-aspect
        # panel exactly fills its slot
        w, h  = half_x + half_y, half_y + half_z
        scale = 9 / max(w, h)
        fig   = plt.figure(figsize=(w * scale, h * scale), layout='constrained')
        gs    = fig.add_gridspec(2, 2, width_ratios=[half_x, half_y],
                                 height_ratios=[half_y, half_z])
        ax_top  = fig.add_subplot(gs[0, 0])
        ax      = fig.add_subplot(gs[0, 1], projection='3d')
        ax_side = fig.add_subplot(gs[1, 0])
        ax_back = fig.add_subplot(gs[1, 1])
        
        ax.set_xlim( cx - half_x, cx + half_x)
        ax.set_ylim( cy - half_y, cy + half_y)
        ax.set_zlim( cz - half_z, cz + half_z)
        ax_top.set_xlim(cx - half_x, cx + half_x)
        ax_top.set_ylim(cy - half_y, cy + half_y)
        ax_back.set_xlim(cy - half_y, cy + half_y)
        ax_back.set_ylim(cz - half_z, cz + half_z)
        ax_side.set_xlim(cx - half_x, cx + half_x)
        ax_side.set_ylim(cz - half_z, cz + half_z)

        # pin tick spacing to whole meters so panel resizes can't switch the
        # locators to fractional steps that crowd the foreshortened 3D axes
        for a2d in (ax_top, ax_back, ax_side):
            a2d.xaxis.set_major_locator(MaxNLocator(integer=True))
            a2d.yaxis.set_major_locator(MaxNLocator(integer=True))

        # draw the surface grid
        grid_segments = [
            [(self.surf_grid.nodes[u]['x'], self.surf_grid.nodes[u]['y'], self.surf_grid.nodes[u]['z']),
             (self.surf_grid.nodes[v]['x'], self.surf_grid.nodes[v]['y'], self.surf_grid.nodes[v]['z'])]
            for u, v in self.surf_grid.edges()
        ]
        grid_segments1 = [
            [(self.surf_grid.nodes[u]['x'], self.surf_grid.nodes[u]['y']),
             (self.surf_grid.nodes[v]['x'], self.surf_grid.nodes[v]['y'])]
            for u, v in self.surf_grid.edges()
        ]
        grid_segments2 = [
            [(self.surf_grid.nodes[u]['x'], self.surf_grid.nodes[u]['z']),
             (self.surf_grid.nodes[v]['x'], self.surf_grid.nodes[v]['z'])]
            for u, v in self.surf_grid.edges()
        ]
        grid_segments3 = [
            [(self.surf_grid.nodes[u]['y'], self.surf_grid.nodes[u]['z']),
             (self.surf_grid.nodes[v]['y'], self.surf_grid.nodes[v]['z'])]
            for u, v in self.surf_grid.edges()
        ]

        ax.add_collection3d(Line3DCollection(grid_segments, colors='saddlebrown', linewidths=0.5, alpha=0.5, zorder=0))
        ax_top.add_collection(LineCollection(grid_segments1, colors="black", linewidths=0.5, alpha=0.5, zorder=0))
        ax_back.add_collection(LineCollection(grid_segments3, colors="black", linewidths=0.5, alpha=0.5, zorder=0))
        ax_side.add_collection(LineCollection(grid_segments2, colors="black", linewidths=0.5, alpha=0.5, zorder=0))

        # remaining plot settings
        ax.set_box_aspect((half_x, half_y, half_z), zoom=1)
        ax.grid(False)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)") 
        
        ax_top.set_aspect('equal')
        ax_top.grid(False)
        ax_top.set_xlabel("X (m)")
        ax_top.set_ylabel("Y (m)")
        
        ax_back.set_aspect('equal')
        ax_back.grid(False)
        ax_back.set_xlabel("Y (m)")
        ax_back.set_ylabel("Z (m)")

        ax_side.set_aspect('equal')
        ax_side.grid(False)
        ax_side.set_xlabel("X (m)")
        ax_side.set_ylabel("Z (m)")

        # orange line connecting the current q and point positions, updated each frame
        link,      = ax.plot([], [], [], color='darkorange', linewidth=1.5)
        link_top,  = ax_top.plot([], [], color='darkorange', linewidth=1.5)
        link_back, = ax_back.plot([], [], color='darkorange', linewidth=1.5)
        link_side, = ax_side.plot([], [], color='darkorange', linewidth=1.5)
        
        link_lower,      = ax.plot([], [], [], color='darkorange', linewidth=1.5)
        link_top_lower,  = ax_top.plot([], [], color='darkorange', linewidth=1.5)
        link_back_lower, = ax_back.plot([], [], color='darkorange', linewidth=1.5)
        link_side_lower, = ax_side.plot([], [], color='darkorange', linewidth=1.5)

        # green scatter artists for grid vertices within one tile length of the
        # point, updated each frame (only created when show_neighbors is set)
        green_3d = green_top = green_back = green_side = None
        if show_neighbors:
            green_3d   = ax.scatter([], [], [], color='green', s=40, zorder=5)
            green_top  = ax_top.scatter([], [], color='green', s=40, zorder=5)
            green_back = ax_back.scatter([], [], color='green', s=40, zorder=5)
            green_side = ax_side.scatter([], [], color='green', s=40, zorder=5)

        # red arrow at the tracked point showing the center of mass's orientation,
        # i.e. the local forward axis (R_lg(*q[3:6])[:, 0]) for that frame's roll/pitch/yaw
        arrow_len = self.subdivision * 0.6

        def _forward(i):
            return self._rotation_lg(*data[i, 4:7])[:, 0]

        def _fwd(row):
            return self._rotation_lg(*row[3:])[:, 0]

        def _lateral(i):
            # q's local +y axis scaled to the half-width: R_lg(*q[3:6]) @ [0, b/2, 0]
            return self._rotation_lg(*data[i, 4:7]) @ np.array([0, self.b / 2, 0])
        fwd0 = _forward(0)
        lat0 = _lateral(0)
        # each track arrow carries two arrows (right, left); set_qdot repositions them
        qdot_top  = ax_top.quiver([data[0, 1]] * 2, [data[0, 2]] * 2, [fwd0[0]] * 2, [fwd0[1]] * 2,
                                   color='darkorange', scale=1 / arrow_len, scale_units='xy',
                                   angles='xy', zorder=6)
        qdot_back = ax_back.quiver([data[0, 2]] * 2, [data[0, 3]] * 2, [fwd0[1]] * 2, [fwd0[2]] * 2,
                                    color='darkorange', scale=1 / arrow_len, scale_units='xy',
                                    angles='xy', zorder=6)
        qdot_side = ax_side.quiver([data[0, 1]] * 2, [data[0, 3]] * 2, [fwd0[0]] * 2, [fwd0[2]] * 2,
                                    color='darkorange', scale=1 / arrow_len, scale_units='xy',
                                    angles='xy', zorder=6)
        qdot_3d   = [None]  # mplot3d quiver has no in-place update, so remove/recreate each frame

        # blue arrow marking the center of mass q, showing q's own orientation
        q_arrow_top  = ax_top.quiver(data[0, 1], data[0, 2], fwd0[0], fwd0[1],
                                      color='blue', scale=1 / arrow_len, scale_units='xy',
                                      angles='xy', zorder=7)
        q_arrow_back = ax_back.quiver(data[0, 2], data[0, 3], fwd0[1], fwd0[2],
                                       color='blue', scale=1 / arrow_len, scale_units='xy',
                                       angles='xy', zorder=7)
        q_arrow_side = ax_side.quiver(data[0, 1], data[0, 3], fwd0[0], fwd0[2],
                                       color='blue', scale=1 / arrow_len, scale_units='xy',
                                       angles='xy', zorder=7)
        q_arrow_3d   = [None]

        # purple arrows from q toward each track: q ± R_lg(*q[3:6]) @ [0, b/2, 0].
        # drawn at true length (scale=1) so each tip lands on a track center
        q_lat_top  = ax_top.quiver([data[0, 1]] * 2, [data[0, 2]] * 2,
                                    [lat0[0], -lat0[0]], [lat0[1], -lat0[1]],
                                    color='purple', scale=1, scale_units='xy',
                                    angles='xy', zorder=7)
        q_lat_back = ax_back.quiver([data[0, 2]] * 2, [data[0, 3]] * 2,
                                     [lat0[1], -lat0[1]], [lat0[2], -lat0[2]],
                                     color='purple', scale=1, scale_units='xy',
                                     angles='xy', zorder=7)
        q_lat_side = ax_side.quiver([data[0, 1]] * 2, [data[0, 3]] * 2,
                                     [lat0[0], -lat0[0]], [lat0[2], -lat0[2]],
                                     color='purple', scale=1, scale_units='xy',
                                     angles='xy', zorder=7)
        q_lat_3d   = [None]

        # orange arrows at the right and left front points (f shares q's orientation)
        front_arrow_top  = ax_top.quiver([front_data[0, 0], lfront_data[0, 0]], [front_data[0, 1], lfront_data[0, 1]],
                                          [fwd0[0]] * 2, [fwd0[1]] * 2,
                                          color='darkorange', scale=1 / arrow_len, scale_units='xy',
                                          angles='xy', zorder=6)
        front_arrow_back = ax_back.quiver([front_data[0, 1], lfront_data[0, 1]], [front_data[0, 2], lfront_data[0, 2]],
                                           [fwd0[1]] * 2, [fwd0[2]] * 2,
                                           color='darkorange', scale=1 / arrow_len, scale_units='xy',
                                           angles='xy', zorder=6)
        front_arrow_side = ax_side.quiver([front_data[0, 0], lfront_data[0, 0]], [front_data[0, 2], lfront_data[0, 2]],
                                           [fwd0[0]] * 2, [fwd0[2]] * 2,
                                           color='darkorange', scale=1 / arrow_len, scale_units='xy',
                                           angles='xy', zorder=6)
        front_arrow_3d = [None]

        # orange arrows at the right and left back points (b shares q's orientation)
        back_arrow_top  = ax_top.quiver([back_data[0, 0], lback_data[0, 0]], [back_data[0, 1], lback_data[0, 1]],
                                          [fwd0[0]] * 2, [fwd0[1]] * 2,
                                          color='darkorange', scale=1 / arrow_len, scale_units='xy',
                                          angles='xy', zorder=6)
        back_arrow_back = ax_back.quiver([back_data[0, 1], lback_data[0, 1]], [back_data[0, 2], lback_data[0, 2]],
                                           [fwd0[1]] * 2, [fwd0[2]] * 2,
                                           color='darkorange', scale=1 / arrow_len, scale_units='xy',
                                           angles='xy', zorder=6)
        back_arrow_side = ax_side.quiver([back_data[0, 0], lback_data[0, 0]], [back_data[0, 2], lback_data[0, 2]],
                                           [fwd0[0]] * 2, [fwd0[2]] * 2,
                                           color='darkorange', scale=1 / arrow_len, scale_units='xy',
                                           angles='xy', zorder=6)
        back_arrow_3d = [None]

        def set_front(i):
            # right and left center/front rows and their forward axes
            rc, rf = center_data[i], front_data[i]
            lc, lf = lcenter_data[i], lfront_data[i]
            rfwd, lfwd = _fwd(rf), _fwd(lf)

            # two disjoint center→front segments carried by one line artist (NaN breaks the line)
            xs = np.array([rc[0], rf[0], np.nan, lc[0], lf[0]], dtype=float)
            ys = np.array([rc[1], rf[1], np.nan, lc[1], lf[1]], dtype=float)
            zs = np.array([rc[2], rf[2], np.nan, lc[2], lf[2]], dtype=float)
            link.set_data_3d(xs, ys, zs)
            link_top.set_data(xs, ys)
            link_back.set_data(ys, zs)
            link_side.set_data(xs, zs)

            front_arrow_top.set_offsets([[rf[0], rf[1]], [lf[0], lf[1]]])
            front_arrow_top.set_UVC([rfwd[0], lfwd[0]], [rfwd[1], lfwd[1]])
            front_arrow_back.set_offsets([[rf[1], rf[2]], [lf[1], lf[2]]])
            front_arrow_back.set_UVC([rfwd[1], lfwd[1]], [rfwd[2], lfwd[2]])
            front_arrow_side.set_offsets([[rf[0], rf[2]], [lf[0], lf[2]]])
            front_arrow_side.set_UVC([rfwd[0], lfwd[0]], [rfwd[2], lfwd[2]])
            if front_arrow_3d[0] is not None:
                front_arrow_3d[0].remove()
            front_arrow_3d[0] = ax.quiver([rf[0], lf[0]], [rf[1], lf[1]], [rf[2], lf[2]],
                                           [rfwd[0], lfwd[0]], [rfwd[1], lfwd[1]], [rfwd[2], lfwd[2]],
                                           length=arrow_len, color='darkorange', zorder=6)


        def set_back(i):
            # right and left center/back rows and their forward axes
            rc, rb = center_data[i], back_data[i]
            lc, lb = lcenter_data[i], lback_data[i]
            rfwd, lfwd = _fwd(rb), _fwd(lb)

            # two disjoint back→center segments carried by one line artist (NaN breaks the line)
            xs = np.array([rb[0], rc[0], np.nan, lb[0], lc[0]], dtype=float)
            ys = np.array([rb[1], rc[1], np.nan, lb[1], lc[1]], dtype=float)
            zs = np.array([rb[2], rc[2], np.nan, lb[2], lc[2]], dtype=float)
            link_lower.set_data_3d(xs, ys, zs)
            link_top_lower.set_data(xs, ys)
            link_back_lower.set_data(ys, zs)
            link_side_lower.set_data(xs, zs)

            back_arrow_top.set_offsets([[rb[0], rb[1]], [lb[0], lb[1]]])
            back_arrow_top.set_UVC([rfwd[0], lfwd[0]], [rfwd[1], lfwd[1]])
            back_arrow_back.set_offsets([[rb[1], rb[2]], [lb[1], lb[2]]])
            back_arrow_back.set_UVC([rfwd[1], lfwd[1]], [rfwd[2], lfwd[2]])
            back_arrow_side.set_offsets([[rb[0], rb[2]], [lb[0], lb[2]]])
            back_arrow_side.set_UVC([rfwd[0], lfwd[0]], [rfwd[2], lfwd[2]])
            if back_arrow_3d[0] is not None:
                back_arrow_3d[0].remove()
            back_arrow_3d[0] = ax.quiver([rb[0], lb[0]], [rb[1], lb[1]], [rb[2], lb[2]],
                                          [rfwd[0], lfwd[0]], [rfwd[1], lfwd[1]], [rfwd[2], lfwd[2]],
                                          length=arrow_len, color='darkorange', zorder=6)
        def set_neighbors(i):
            pts = np.asarray(neighbor_data[i]) if len(neighbor_data[i]) else np.empty((0, 3))
            green_3d._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
            green_top.set_offsets(pts[:, [0, 1]])
            green_back.set_offsets(pts[:, [1, 2]])
            green_side.set_offsets(pts[:, [0, 2]])

        def set_qdot(i):
            # right and left center rows and their forward axes
            rc, lc = center_data[i], lcenter_data[i]
            rfwd, lfwd = _fwd(rc), _fwd(lc)
            qdot_top.set_offsets([[rc[0], rc[1]], [lc[0], lc[1]]])
            qdot_top.set_UVC([rfwd[0], lfwd[0]], [rfwd[1], lfwd[1]])
            qdot_back.set_offsets([[rc[1], rc[2]], [lc[1], lc[2]]])
            qdot_back.set_UVC([rfwd[1], lfwd[1]], [rfwd[2], lfwd[2]])
            qdot_side.set_offsets([[rc[0], rc[2]], [lc[0], lc[2]]])
            qdot_side.set_UVC([rfwd[0], lfwd[0]], [rfwd[2], lfwd[2]])
            if qdot_3d[0] is not None:
                qdot_3d[0].remove()
            qdot_3d[0] = ax.quiver([rc[0], lc[0]], [rc[1], lc[1]], [rc[2], lc[2]],
                                    [rfwd[0], lfwd[0]], [rfwd[1], lfwd[1]], [rfwd[2], lfwd[2]],
                                    length=arrow_len, color='darkorange', zorder=6)

        def set_q_point(i):
            x, y, z = data[i, 1], data[i, 2], data[i, 3]
            fwd = _forward(i)
            q_arrow_top.set_offsets([[x, y]])
            q_arrow_top.set_UVC(fwd[0], fwd[1])
            q_arrow_back.set_offsets([[y, z]])
            q_arrow_back.set_UVC(fwd[1], fwd[2])
            q_arrow_side.set_offsets([[x, z]])
            q_arrow_side.set_UVC(fwd[0], fwd[2])
            if q_arrow_3d[0] is not None:
                q_arrow_3d[0].remove()
            q_arrow_3d[0] = ax.quiver(x, y, z, fwd[0], fwd[1], fwd[2],
                                       length=arrow_len, color='blue', zorder=7)

            # lateral half-width arrows from q toward each track (true length)
            lat = _lateral(i)
            q_lat_top.set_offsets([[x, y], [x, y]])
            q_lat_top.set_UVC([lat[0], -lat[0]], [lat[1], -lat[1]])
            q_lat_back.set_offsets([[y, z], [y, z]])
            q_lat_back.set_UVC([lat[1], -lat[1]], [lat[2], -lat[2]])
            q_lat_side.set_offsets([[x, z], [x, z]])
            q_lat_side.set_UVC([lat[0], -lat[0]], [lat[2], -lat[2]])
            if q_lat_3d[0] is not None:
                q_lat_3d[0].remove()
            q_lat_3d[0] = ax.quiver([x, x], [y, y], [z, z],
                                     [lat[0], -lat[0]], [lat[1], -lat[1]], [lat[2], -lat[2]],
                                     length=1, color='purple', zorder=7)

        set_front(0)
        set_back(0)
        if show_neighbors:
            set_neighbors(0)
        set_qdot(0)
        set_q_point(0)

        legend_handles = [link_side, q_arrow_side]
        legend_labels  = ["track", "q (center of mass)"]
        if show_neighbors:
            legend_handles.insert(1, green_side)
            legend_labels.insert(1, "grid neighbors")
        ax_side.legend(legend_handles, legend_labels, loc="upper right", fontsize=8)

        # the constrained-layout solver converges over the first few draws,
        # visibly nudging the panels; converge it now, then freeze the layout
        # so every animation frame uses identical panel positions
        ax_top.set_title(f"t = {data[0, 0]:.2f} s")
        fig.canvas.draw()

        # match the 3D ticks to the flat views' ticks (the 3D locator
        # overcrowds foreshortened axes)
        ax.set_xticks([t for t in ax_top.get_xticks()
                       if cx - half_x <= t <= cx + half_x])
        ax.set_yticks([t for t in ax_top.get_yticks()
                       if cy - half_y <= t <= cy + half_y])
        ax.set_zticks([t for t in ax_side.get_yticks()
                       if cz - half_z <= t <= cz + half_z])

        for _ in range(2):
            fig.canvas.draw()
        fig.set_layout_engine('none')

        # animation update function
        def update(i):
            set_front(i)
            set_back(i)
            if show_neighbors:
                set_neighbors(i)
            set_qdot(i)
            set_q_point(i)

            ax_top.set_title(f"t = {data[i, 0]:.2f} s")
            artists = [link, link_top, link_back, link_side,
                       front_arrow_top, front_arrow_back, front_arrow_side, front_arrow_3d[0],
                       link_lower, link_top_lower, link_back_lower, link_side_lower,
                       back_arrow_top, back_arrow_back, back_arrow_side, back_arrow_3d[0],
                       qdot_top, qdot_back, qdot_side, qdot_3d[0],
                       q_arrow_top, q_arrow_back, q_arrow_side, q_arrow_3d[0]]
            if show_neighbors:
                artists += [green_3d, green_top, green_back, green_side]
            return artists

        anim = animation.FuncAnimation(
            fig, update, frames=len(data), blit=False, interval=50
        )
        fname = "figures/simulation.gif"
        anim.save(fname, writer=animation.PillowWriter(fps=20))
        plt.close(fig)
        print(f"Saved {fname}")

    def _tracked_body_update(self):
        if not self.is_initalization:
            self.q += self.dt * self.q_dot
        else:
            self.is_initalization = False

        # seed the body height + orientation from the tile under the center of mass
        neighbors_q, self.q[2] = self._point_height(self.q)
        self.q[3:6]            = self._point_orientation(neighbors_q)

        # rigid contact-averaged solve, then build both tracks as parallel offsets
        track_points, track_neighbors = self._body_update_iteration()

        points          = [self.q, *track_points]
        neighbor_points = [neighbors_q, *track_neighbors]
        return points, neighbor_points

    def _body_update_iteration(self):
        """
        Settle one rigid-body pose (height + roll/pitch/yaw) and return the six
        track points as [right, left] x [front, center, back]. The body is
        rigid, so both tracks share a single half_track vector and stay parallel
        by construction.

        _body_update has already snapped the body flush to the tile under the
        center of mass (height from _bilinear_height, roll/pitch from that
        tile's gradient). Keep that snap whenever it leaves every track point on
        or above the surface: crossing a crest or a uniform slope the body
        really does lie flush on the tile it is on. Only when the snap would
        submerge part of a track does the loop below solve, alternating
        _body_contact_roll and _contact_pitch to settle the pose onto the ground so
        no quarter of the body hangs, with the height resting on the highest
        support.
        """
        # deepest penetration of the snapped pose; 1e-15 absorbs floating point error
        is_under_ground = (self._resting_height(self._rotation_lg(*self.q[3:6])) - self.q[2]) > 1e-15
        if is_under_ground:
            self._settle_tracks(self.q[3:6])

        # Rebuild both tracks rigidly from the converged pose.
        orient = self.q[3:6].copy()
        track_xyz = self._track_xyz(self.q).reshape(-1, 3)

        track_points    = [np.concatenate((xyz, orient))           for xyz in track_xyz]
        track_neighbors = [np.array(self._get_neighbor_points(xyz)) for xyz in track_xyz]
        return track_points, track_neighbors

    def _track_xyz(self, pose):
        """Return [right, left] x [front, center, back] track points."""
        pose       = np.asarray(pose)
        R          = self._rotation_lg(*pose[3:6])
        half_width = R @ np.array([0.0, self.b / 2, 0.0])
        half_track = R @ np.array([self.l / 2, 0.0, 0.0])
        centers    = np.array([pose[:3] - half_width, pose[:3] + half_width])
        return np.stack(
            [centers + half_track, centers, centers - half_track], axis=1
        )

    def _settle_tracks(self, orient):
        """Set q's supported pose from both complete track contact lines."""
        self.q[3:6] = orient
        for _ in range(20):
            # Roll and pitch settle onto the ground so no quarter of the body
            # hangs. Yaw is the heading and cannot change during settling.
            new_orient = self.q[3:6].copy()
            new_orient[0] = self._body_contact_roll(new_orient)
            new_orient[1] = self._body_contact_pitch(new_orient)
            new_z = self._resting_height(self._rotation_lg(*new_orient))

            converged = (
                abs(new_z - self.q[2]) < 1e-6
                and np.all(np.abs(new_orient - self.q[3:6]) < 1e-6)
            )
            self.q[2], self.q[3:6] = new_z, new_orient
            if converged:
                break

    def _body_contact_roll(self, orient):
        """Roll (at orient's pitch and yaw) that leaves both tracks on the ground."""
        def imbalance(roll):
            z_left, z_right = self._track_resting_heights(
                self._rotation_lg(roll, orient[1], orient[2]))
            return z_left - z_right
        return self._contact_angle(imbalance, orient[0])

    def _track_resting_heights(self, R):
        """
        The lowest-height question asked of each track on its own, as
        (left, right). _body_contact_roll balances the pair.
        """
        support = self._support_heights(R)
        return float(support[1].max()), float(support[0].max())

    def _support_heights(self, R):
        """
        Lowest body height each quarter of the contact patch calls for, as a
        [right, left] x [back, front] array. The rigid body can only rest at the
        highest of the four, so any quarter asking for less hangs clear of the
        ground unless the pose balances them: _body_contact_roll balances left
        against right, _contact_pitch front against back.

        The halves overlap at the track midpoint, so a body pivoting on a peak
        directly beneath its center reads as balanced from every direction --
        which it is, since no rotation can bring anything else down onto the
        ground without driving that peak through the belly.

        The surface is piecewise planar with kinks only on the grid lines, so a
        straight track's deepest penetration always occurs at an endpoint or a
        grid-line crossing; between those the clearance varies linearly. Sampling
        the ends plus every grid crossing therefore finds the true deepest point
        exactly (a uniform scan would step over the kinks).
        """
        fwd     = R[:, 0]
        lat     = R[:, 1] * (self.b / 2)      # body center -> track lateral offset (local +y)
        half_l  = self.l / 2
        support = np.full((2, 2), -np.inf)    # [right, left] x [back, front]
        for i, side in enumerate((-1.0, 1.0)):   # right (-lat) and left (+lat) tracks
            base  = self.q[:3] + side * lat
            ss    = {-half_l, 0, half_l}         # track ends

            # the track only spans the tiles between its two ends, so search that
            # neighborhood of grid lines instead of the whole grid
            back_cell  = self._grid_cell(base - half_l * fwd)
            front_cell = self._grid_cell(base + half_l * fwd)
            # seach x axis tiles via (self.us, 0) then y axis tiles via (self.vs, 1)
            for grid, axis in ((self.us, 0), (self.vs, 1)):
                # skip if the track runs parallel to this axis' lines (fwd[axis] ~ 0) since it never crosses
                # plus avoid divide by zero error later
                if abs(fwd[axis]) > 1e-12:
                    # grid lines bounding the tiles the ends fall in (+1 stop, +1 for the far tile's upper line)
                    lo, hi = sorted((back_cell[axis], front_cell[axis]))
                    #TODO: could super sample for smaller grid sizes so that surfaces difference above a certain size 
                    # are treated as "real" surface differnces
                    near   = grid[lo:hi + 2]
                    # solve grid = base[axis] + s*fwd[axis] for every nearby grid value at once
                    for s in (near - base[axis]) / fwd[axis]:
                        if -half_l < s < half_l:   # keep only crossings within the track
                            ss.add(float(s))

            for s in ss:
                grid_crossing_point = base + s * fwd
                surface_height      = self._point_height(grid_crossing_point)[1]
                delta = surface_height - grid_crossing_point[2]
                needed = delta + self.q[2]
                # the midpoint (s == 0) is the last point of both halves
                if s <= 0:
                    support[i, 0] = max(support[i, 0], needed)
                if s >= 0:
                    support[i, 1] = max(support[i, 1], needed)
        return support

    def _body_contact_pitch(self, orient):
        """Pitch (at orient's roll and yaw) that leaves both ends on the ground."""
        def imbalance(pitch):
            z_front, z_back = self._half_resting_heights(
                self._rotation_lg(orient[0], pitch, orient[2]))
            return z_front - z_back
        return self._contact_angle(imbalance, orient[1])

    def _half_resting_heights(self, R):
        """
        The same question asked of each half of the body, as (front, back).
        _contact_pitch balances the pair.
        """
        support = self._support_heights(R)
        return float(support[:, 1].max()), float(support[:, 0].max())

    def _resting_height(self, R):
        """
        Lowest body height q[2] (for orientation R and the current q[:2]) that
        keeps every point of both rigid tracks on or above the surface, so the
        body rests on its highest contact and no track segment is submerged.
        """
        return float(self._support_heights(R).max())

class Surface(_BodyMode):
    """Surface-aware bulldozer with an optional deforming blade."""

    def __init__(self, is_uphill = True, is_surface_pitched: bool = False, is_surface_rolled: bool = False,
                 is_backwards: bool = False, blade_local_yaw: float = 0.0,
                 blade_local_roll: float = 0.0, blade_pitch: float = None,
                 enable_blade: bool = True):
        self.enable_blade = enable_blade
        if not enable_blade:
            self._initialize_body_mode(
                is_uphill=is_uphill,
                is_surface_pitched=is_surface_pitched,
                is_surface_rolled=is_surface_rolled,
                is_backwards=is_backwards,
            )
            return

        # simulation parameters
        self.division_factor    = 4*2
        self.b                  = 1.75
        self.offset             = np.array([0, 0, self.b]) if is_uphill else np.array([0, 0, -self.b])
        self.surface_abg        = np.array([ 0.0, 0.0, 0.0])
        self.u_split            = 0  # u-value where the grid switches to surface_abg2
        self.v_split            = 0  # u-value where the grid switches to surface_abg2
        self.transition_tiles   = 1/2 *self.b * self.division_factor # tiles over which the offset ramps down past u_split
        self.q                  = np.array([0.0, 0.0, 0.0, self.surface_abg[0], self.surface_abg[1], 0.0 if is_surface_pitched else np.pi / 2])
        self.q_dot              = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]) if is_surface_pitched else np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0]) 
        self.is_initalization   = True

        self.stop_time          = 300.0
        self.total_distance     = 0.0
        self.dt                 = 1/100 

        # Bulldozer blade parameters
        self.B1   = 2.921
        self.H    = 0.955
        self.L    = 1.2 
        self.l    = 2.349
        self.blade_local_yaw = blade_local_yaw
        self.blade_local_roll = blade_local_roll
        self.blade_pitch = (
            np.arcsin((self.H / 4) / self.L)
            if blade_pitch is None else blade_pitch
        )
        self.contact_tol = 1e-7

        # No vertex may be cut farther than this below its own starting height.
        self.max_world_cut_depth     = self.H


        if is_backwards:
            self.q_dot   *= -1
            self.u_split *= self.q_dot[0] / np.linalg.norm(self.q_dot)
            self.v_split *= self.q_dot[1] / np.linalg.norm(self.q_dot)
        self.is_backwards       = is_backwards
        self.is_surface_pitched = is_surface_pitched
        self.is_surface_rolled = is_surface_rolled
        self.log                = []
        self.blade_log          = []
        self.neighbor_log       = []
        self.grid_log           = []
        
        # set up the surface grid
        self.u_range       = (-2* self.b/2, 4 * self.b) if is_surface_pitched else (-self.b/2,     self.b/2) 
        self.v_range       = (-self.b/2,   self.b/2) if is_surface_pitched else (-2* self.b/2, 4 * self.b)
        if is_backwards:
            self.u_range       = (-2.5 * self.b, self.b/2) if is_surface_pitched else (-self.b/2,     self.b/2) 
            self.v_range       = (-self.b/2,     self.b/2) if is_surface_pitched else (-2.5 * self.b, self.b/2)
        self.us            = np.arange(self.u_range[0], self.u_range[1] + self.subdivision, self.subdivision)
        self.vs            = np.arange(self.v_range[0], self.v_range[1] + self.subdivision, self.subdivision)
        
        stop_index         = 0 if self.is_backwards else -1 
        self.stop_distance = abs(self.us[stop_index]) if is_surface_pitched else abs(self.vs[stop_index])
        
        self.surf_grid     = self._surface_grid()

        # node (x, y, z) cached as plain Python floats, indexed [i][j], so the
        # hot height lookups skip the networkx attribute dicts entirely.
        self.grid_pts = [[(self.surf_grid.nodes[(i, j)]['x'],
                           self.surf_grid.nodes[(i, j)]['y'],
                           self.surf_grid.nodes[(i, j)]['z'])
                          for j in range(len(self.vs))] for i in range(len(self.us))]
        self.starting_grid_heights = self._grid_heights().copy()

        # q is retained for visualization and as the rigid-body pose origin;
        # only the complete track model determines the supported pose.
        neighbor_points = self._get_neighbor_points(self.q)
        self._settle_tracks(self.q[3:6].copy())
        self.log.append([0, *self.q])
        self.grid_log.append(self._grid_heights())

        # seed the blade points the same way, so blade_log stays aligned frame
        # for frame with log/neighbor_log/grid_log
        blade_points, blade_neighbors = self._blade_update(deform=False)
        neighbor_points.extend(point for neighbors in blade_neighbors for point in neighbors)
        self.neighbor_log.append(np.array(neighbor_points))
        self.blade_log.append(blade_points)

    @property
    def blade_cut_depth(self):
        """Vertical cut depth commanded by blade pitch without tilting the blade."""
        return self.L * np.sin(self.blade_pitch)

    def _blade_rotation_lg(self, orient):
        """Blade-local roll/yaw composed on top of the body orientation."""
        body_R = self._rotation_lg(*orient)
        local_R = self._rotation_lg(
            self.blade_local_roll, 0.0, self.blade_local_yaw
        )
        return body_R @ local_R

    def _starting_height(self, point):
        """Undeformed surface height at point's horizontal position."""
        i, j = self._grid_cell(point)
        corners = (
            (*self.grid_pts[i][j][:2], self.starting_grid_heights[i, j]),
            (*self.grid_pts[i + 1][j][:2], self.starting_grid_heights[i + 1, j]),
            (*self.grid_pts[i][j + 1][:2], self.starting_grid_heights[i, j + 1]),
            (*self.grid_pts[i + 1][j + 1][:2], self.starting_grid_heights[i + 1, j + 1]),
        )
        return self._bilinear_height(point, corners)

    def _contact_pitch(self, orient):
        """Pitch that balances the front and rear halves of both tracks."""
        return self._body_contact_pitch(orient)

    def run_and_plot(self, show_neighbors: bool = False):
        """Run the simulation and render the active body/blade view."""
        self.run()
        if self.enable_blade:
            return self._render_blade_run(show_neighbors)
        return self._render_body_run(show_neighbors)

    def _render_blade_run(self, show_neighbors: bool = False):
        print("Rendering GIF...")
        
        # set axis limits based on the logged data (and neighbor points, if any)
        margin = 0.5
        log_data      = np.asarray(self.log)
        data          = log_data[::2]  # every 2nd frame represented to speed up rendering
        track_path    = np.array([self._track_xyz(row[1:7]) for row in log_data])
        track_data    = track_path[::2]
        # (frames, blade_points, 6), ordered from the right blade end to the left.
        # The point count follows the blade/grid resolution rather than being
        # fixed at the two blade endpoints.
        blade_data    = np.asarray(self.blade_log)[::2]
        blade_path    = np.asarray(self.blade_log)
        blade_top_data = np.array([
            frame[[0, -1], :3]
            + self._rotation_lg(*frame[0, 3:6])[:, 2] * self.H
            for frame in blade_data
        ])
        neighbor_data = self.neighbor_log[::2]
        grid_z_data   = self.grid_log[::2]

        # deformation only moves heights, so x/y are fixed for the whole run and
        # a frame's surface is these two arrays plus that frame's z snapshot
        grid_x = np.array([[pt[0] for pt in col] for col in self.grid_pts])
        grid_y = np.array([[pt[1] for pt in col] for col in self.grid_pts])

        # bound the view with every height the surface takes over the run (so
        # neither the q path, the blade paths, nor a fresh cut is clipped flush
        # at a panel edge), plus the grid-neighbor points only when they will
        # actually be drawn
        bound_x = [grid_x.ravel(), blade_data[:, :, 0].ravel(),
                   blade_top_data[:, :, 0].ravel(), track_path[:, :, :, 0].ravel()]
        bound_y = [grid_y.ravel(), blade_path[:, :, 1].ravel(),
                   blade_top_data[:, :, 1].ravel(), track_path[:, :, :, 1].ravel()]
        bound_z = [np.asarray(self.grid_log).ravel(), blade_path[:, :, 2].ravel(),
                   blade_top_data[:, :, 2].ravel(), track_path[:, :, :, 2].ravel()]
        if show_neighbors:
            neighbor_pts = np.array([pt for frame in self.neighbor_log for pt in frame])
            bound_x.append(neighbor_pts[:, 0])
            bound_y.append(neighbor_pts[:, 1])
            bound_z.append(neighbor_pts[:, 2])

        all_x = np.concatenate([data[:, 1], *bound_x])
        all_y = np.concatenate([data[:, 2], *bound_y])
        all_z = np.concatenate([data[:, 3], *bound_z])
        
        cx     = (all_x.max() + all_x.min()) / 2
        cy     = (all_y.max() + all_y.min()) / 2
        cz     = (all_z.max() + all_z.min()) / 2
        half_x = (all_x.max() - all_x.min()) / 2 + margin
        half_y = (all_y.max() - all_y.min()) / 2 + margin
        half_z = (all_z.max() - all_z.min()) / 2 + margin

        # set up the figure and axes for the animation
        # 2x2 layout: top view (X-Y) | 3D view
        #             side view (X-Z)| back view (Y-Z)
        # row/column ratios match the per-axis data spans so each equal-aspect
        # panel exactly fills its slot
        w, h  = half_x + half_y, half_y + half_z
        scale = 9 / max(w, h)
        fig   = plt.figure(figsize=(w * scale, h * scale), layout='constrained')
        gs    = fig.add_gridspec(2, 2, width_ratios=[half_x, half_y],
                                 height_ratios=[half_y, half_z])
        ax_top  = fig.add_subplot(gs[0, 0])
        ax      = fig.add_subplot(gs[0, 1], projection='3d')
        ax_side = fig.add_subplot(gs[1, 0])
        ax_back = fig.add_subplot(gs[1, 1])
        
        ax.set_xlim( cx - half_x, cx + half_x)
        ax.set_ylim( cy - half_y, cy + half_y)
        ax.set_zlim( cz - half_z, cz + half_z)
        ax_top.set_xlim(cx - half_x, cx + half_x)
        ax_top.set_ylim(cy - half_y, cy + half_y)
        ax_back.set_xlim(cy - half_y, cy + half_y)
        ax_back.set_ylim(cz - half_z, cz + half_z)
        ax_side.set_xlim(cx - half_x, cx + half_x)
        ax_side.set_ylim(cz - half_z, cz + half_z)

        # pin tick spacing to whole meters so panel resizes can't switch the
        # locators to fractional steps that crowd the foreshortened 3D axes
        for a2d in (ax_top, ax_back, ax_side):
            a2d.xaxis.set_major_locator(MaxNLocator(integer=True))
            a2d.yaxis.set_major_locator(MaxNLocator(integer=True))

        # draw the surface grid. the edges never change, only the heights they
        # hang off, so each edge's two endpoints are held as (i, j) index arrays
        # and a frame's segments are a straight gather from its z snapshot
        edge_a = np.array([a for a, _ in self.surf_grid.edges()])
        edge_b = np.array([b for _, b in self.surf_grid.edges()])
        ia, ja = edge_a[:, 0], edge_a[:, 1]
        ib, jb = edge_b[:, 0], edge_b[:, 1]

        def grid_segments(z):
            """(edges, 2, 3) endpoint array for one frame's heights."""
            return np.stack([np.column_stack([grid_x[ia, ja], grid_y[ia, ja], z[ia, ja]]),
                             np.column_stack([grid_x[ib, jb], grid_y[ib, jb], z[ib, jb]])], axis=1)

        segs0      = grid_segments(grid_z_data[0])
        grid_3d    = Line3DCollection(segs0, colors='saddlebrown', linewidths=0.5, alpha=0.5, zorder=0)
        grid_back  = LineCollection(segs0[:, :, [1, 2]], colors="black", linewidths=0.5, alpha=0.5, zorder=0)
        grid_side  = LineCollection(segs0[:, :, [0, 2]], colors="black", linewidths=0.5, alpha=0.5, zorder=0)

        ax.add_collection3d(grid_3d)
        ax_back.add_collection(grid_back)
        ax_side.add_collection(grid_side)
        # the top view is x/y only, which deformation never touches, so it is
        # drawn once and left alone
        ax_top.add_collection(LineCollection(segs0[:, :, [0, 1]], colors="black",
                                             linewidths=0.5, alpha=0.5, zorder=0))

        def set_grid(i):
            segs = grid_segments(grid_z_data[i])
            grid_3d.set_segments(segs)
            grid_back.set_segments(segs[:, :, [1, 2]])
            grid_side.set_segments(segs[:, :, [0, 2]])

        # remaining plot settings
        ax.set_box_aspect((half_x, half_y, half_z), zoom=1)
        ax.grid(False)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)") 
        
        ax_top.set_aspect('equal')
        ax_top.grid(False)
        ax_top.set_xlabel("X (m)")
        ax_top.set_ylabel("Y (m)")
        
        ax_back.set_aspect('equal')
        ax_back.grid(False)
        ax_back.set_xlabel("Y (m)")
        ax_back.set_ylabel("Z (m)")

        ax_side.set_aspect('equal')
        ax_side.grid(False)
        ax_side.set_xlabel("X (m)")
        ax_side.set_ylabel("Z (m)")

        # Both rigid track centerlines, reconstructed from each logged body pose.
        # Collections update in place and participate in the renderer's manual
        # background/blit lifecycle alongside the deforming grid and blade.
        track0 = track_data[0]
        track_3d = Line3DCollection(
            track0, colors="darkorange", linewidths=3.0, zorder=5, label="tracks"
        )
        track_top = LineCollection(
            track0[:, :, [0, 1]], colors="darkorange", linewidths=3.0,
            zorder=5, label="tracks"
        )
        track_back = LineCollection(
            track0[:, :, [1, 2]], colors="darkorange", linewidths=3.0,
            zorder=5, label="tracks"
        )
        track_side = LineCollection(
            track0[:, :, [0, 2]], colors="darkorange", linewidths=3.0,
            zorder=5, label="tracks"
        )
        ax.add_collection3d(track_3d)
        ax_top.add_collection(track_top)
        ax_back.add_collection(track_back)
        ax_side.add_collection(track_side)

        def set_tracks(i):
            tracks = track_data[i]
            track_3d.set_segments(tracks)
            track_top.set_segments(tracks[:, :, [0, 1]])
            track_back.set_segments(tracks[:, :, [1, 2]])
            track_side.set_segments(tracks[:, :, [0, 2]])

        # green scatter artists for grid vertices within one tile length of the
        # point, updated each frame (only created when show_neighbors is set)
        green_3d = green_top = green_back = green_side = None
        if show_neighbors:
            green_3d   = ax.scatter([], [], [], color='green', s=40, zorder=5)
            green_top  = ax_top.scatter([], [], color='green', s=40, zorder=5)
            green_back = ax_back.scatter([], [], color='green', s=40, zorder=5)
            green_side = ax_side.scatter([], [], color='green', s=40, zorder=5)

        # arrows show the center of mass's orientation, i.e. the local forward
        # axis (R_lg(*q[3:6])[:, 0]) for that frame's roll/pitch/yaw, drawn at a
        # fixed fraction of a tile so they stay readable as the view bounds change
        arrow_len = self.subdivision * 0.6

        def _forward(i):
            return self._rotation_lg(*data[i, 4:7])[:, 0]
        fwd0 = _forward(0)

        # blade_data is logged by _blade_update every frame, so it already
        # reflects that frame's surface height -- unlike recomputing it here
        # against self.grid_pts, which by plot time holds only the final,
        # fully-deformed grid
        blade0 = blade_data[0]

        # Blue arrow marking the center of mass q at the track-supported body
        # pose. mplot3d quiver has no in-place update, so the 3D twin of each
        # artist is held in a 1-element list and removed/recreated each frame.
        q_arrow_top  = ax_top.quiver(data[0, 1], data[0, 2], fwd0[0], fwd0[1],
                                      color='blue', scale=1 / arrow_len, scale_units='xy',
                                      angles='xy', zorder=7)
        q_arrow_back = ax_back.quiver(data[0, 2], data[0, 3], fwd0[1], fwd0[2],
                                       color='blue', scale=1 / arrow_len, scale_units='xy',
                                       angles='xy', zorder=7)
        q_arrow_side = ax_side.quiver(data[0, 1], data[0, 3], fwd0[0], fwd0[2],
                                       color='blue', scale=1 / arrow_len, scale_units='xy',
                                       angles='xy', zorder=7)
        q_arrow_3d   = [None]

        # Red arrows mark every deformation contact point across the blade.
        # They are offset forward and laterally, then snapped to
        # the surface, sharing q's orientation
        blade_arrow_top  = ax_top.quiver(blade0[:, 0], blade0[:, 1],
                                          np.full(len(blade0), fwd0[0]), np.full(len(blade0), fwd0[1]),
                                          color='red', scale=1 / arrow_len, scale_units='xy',
                                          angles='xy', zorder=7)
        blade_arrow_back = ax_back.quiver(blade0[:, 1], blade0[:, 2],
                                           np.full(len(blade0), fwd0[1]), np.full(len(blade0), fwd0[2]),
                                           color='red', scale=1 / arrow_len, scale_units='xy',
                                           angles='xy', zorder=7)
        blade_arrow_side = ax_side.quiver(blade0[:, 0], blade0[:, 2],
                                           np.full(len(blade0), fwd0[0]), np.full(len(blade0), fwd0[2]),
                                           color='red', scale=1 / arrow_len, scale_units='xy',
                                           angles='xy', zorder=7)
        blade_arrow_3d   = [None]

        # A semi-transparent face spans the two lower contact endpoints and the
        # two upper corners obtained along the blade's local +Z axis.
        blade_top0 = blade_top_data[0]
        blade_face0 = np.vstack([blade0[0, :3], blade0[-1, :3],
                                 blade_top0[-1], blade_top0[0]])
        blade_face_top = Polygon(blade_face0[:, [0, 1]], closed=True,
                                 facecolor="red", edgecolor="darkred", alpha=0.35, zorder=6)
        blade_face_back = Polygon(blade_face0[:, [1, 2]], closed=True,
                                  facecolor="red", edgecolor="darkred", alpha=0.35, zorder=6)
        blade_face_side = Polygon(blade_face0[:, [0, 2]], closed=True,
                                  facecolor="red", edgecolor="darkred", alpha=0.35, zorder=6)
        blade_face_3d = Poly3DCollection([blade_face0], facecolors="red",
                                         edgecolors="darkred", alpha=0.35, zorder=6)
        ax_top.add_patch(blade_face_top)
        ax_back.add_patch(blade_face_back)
        ax_side.add_patch(blade_face_side)
        ax.add_collection3d(blade_face_3d)

        def set_neighbors(i):
            pts = np.asarray(neighbor_data[i]) if len(neighbor_data[i]) else np.empty((0, 3))
            green_3d._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
            green_top.set_offsets(pts[:, [0, 1]])
            green_back.set_offsets(pts[:, [1, 2]])
            green_side.set_offsets(pts[:, [0, 2]])

        def set_q_point(i):
            x, y, z = data[i, 1], data[i, 2], data[i, 3]
            fwd = _forward(i)
            q_arrow_top.set_offsets([[x, y]])
            q_arrow_top.set_UVC(fwd[0], fwd[1])
            q_arrow_back.set_offsets([[y, z]])
            q_arrow_back.set_UVC(fwd[1], fwd[2])
            q_arrow_side.set_offsets([[x, z]])
            q_arrow_side.set_UVC(fwd[0], fwd[2])
            if q_arrow_3d[0] is not None:
                q_arrow_3d[0].remove()
            q_arrow_3d[0] = ax.quiver(x, y, z, fwd[0], fwd[1], fwd[2],
                                       length=arrow_len, color='blue', zorder=7)

            blade = blade_data[i]
            blade_arrow_top.set_offsets(blade[:, [0, 1]])
            blade_arrow_top.set_UVC(np.full(len(blade), fwd[0]), np.full(len(blade), fwd[1]))
            blade_arrow_back.set_offsets(blade[:, [1, 2]])
            blade_arrow_back.set_UVC(np.full(len(blade), fwd[1]), np.full(len(blade), fwd[2]))
            blade_arrow_side.set_offsets(blade[:, [0, 2]])
            blade_arrow_side.set_UVC(np.full(len(blade), fwd[0]), np.full(len(blade), fwd[2]))
            if blade_arrow_3d[0] is not None:
                blade_arrow_3d[0].remove()
            blade_arrow_3d[0] = ax.quiver(blade[:, 0], blade[:, 1], blade[:, 2],
                                           np.full(len(blade), fwd[0]), np.full(len(blade), fwd[1]), np.full(len(blade), fwd[2]),
                                           length=arrow_len, color='red', zorder=7)

            blade_top = blade_top_data[i]
            blade_face = np.vstack([blade[0, :3], blade[-1, :3],
                                    blade_top[-1], blade_top[0]])
            blade_face_top.set_xy(blade_face[:, [0, 1]])
            blade_face_back.set_xy(blade_face[:, [1, 2]])
            blade_face_side.set_xy(blade_face[:, [0, 2]])
            blade_face_3d.set_verts([blade_face])

        if show_neighbors:
            set_neighbors(0)
        set_tracks(0)
        set_q_point(0)

        legend_handles = [track_side, q_arrow_side, blade_arrow_side, blade_face_side]
        legend_labels  = ["tracks", "q (center of mass)", "blade contact points", "blade face"]
        if show_neighbors:
            legend_handles.append(green_side)
            legend_labels.append("grid neighbors")
        ax_side.legend(legend_handles, legend_labels, loc="upper right", fontsize=8)

        # the constrained-layout solver converges over the first few draws,
        # visibly nudging the panels; converge it now, then freeze the layout
        # so every animation frame uses identical panel positions
        ax_top.set_title(f"t = {data[0, 0]:.2f} s")
        fig.canvas.draw()

        # match the 3D ticks to the flat views' ticks (the 3D locator
        # overcrowds foreshortened axes)
        ax.set_xticks([t for t in ax_top.get_xticks()
                       if cx - half_x <= t <= cx + half_x])
        ax.set_yticks([t for t in ax_top.get_yticks()
                       if cy - half_y <= t <= cy + half_y])
        ax.set_zticks([t for t in ax_side.get_yticks()
                       if cz - half_z <= t <= cz + half_z])

        for _ in range(2):
            fig.canvas.draw()
        fig.set_layout_engine('none')

        # animation update function
        def update(i):
            set_grid(i)
            set_tracks(i)
            if show_neighbors:
                set_neighbors(i)
            set_q_point(i)
            ax_top.set_title(f"t = {data[i, 0]:.2f} s")

        # only the arrows, the grid collections and the title move between
        # frames. everything else - panes, ticks, tick labels, axis labels, the
        # legend - is identical throughout, so it is rasterized once here and
        # replayed under each frame. redrawing it per frame dominated the render,
        # because matplotlib re-measures every tick and axis label on each draw.
        # the artists listed per axes are ordered by zorder, since drawing them
        # by hand skips the sort a full draw would do
        flat_artists = [grid_back, grid_side, track_top, track_back, track_side]
        if show_neighbors:
            flat_artists += [green_back, green_side, green_top]
        flat_artists += [q_arrow_top, q_arrow_back, q_arrow_side,
                         blade_arrow_top, blade_arrow_back, blade_arrow_side,
                         blade_face_top, blade_face_back, blade_face_side]
        three_d       = [grid_3d, track_3d, blade_face_3d] + ([green_3d] if show_neighbors else [])

        # the background has to hold no frame-specific state, or frame 0's
        # arrows and grid ghost behind the whole animation. the 3D quivers are
        # replaced rather than updated each frame, so they are simply dropped
        # and the first update() rebuilds them
        q_arrow_3d[0].remove()
        q_arrow_3d[0] = None
        blade_arrow_3d[0].remove()
        blade_arrow_3d[0] = None
        for art in flat_artists + three_d:
            art.set_visible(False)
        ax_top.set_title("")
        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(fig.bbox)
        for art in flat_artists + three_d:
            art.set_visible(True)

        fps        = 30
        frame_size = fig.canvas.get_width_height()
        frames     = []
        for i in range(len(data)):
            update(i)
            fig.canvas.restore_region(background)

            for art in flat_artists:
                art.axes.draw_artist(art)
            # Axes3D.draw is what normally refreshes these projections, so
            # bypassing it means projecting by hand. that is only valid because
            # the view angle - and with it ax.M - is fixed for the whole run
            for art in three_d + [q_arrow_3d[0], blade_arrow_3d[0]]:
                art.do_3d_projection()
                ax.draw_artist(art)
            ax_top.draw_artist(ax_top.title)

            frames.append(Image.frombuffer("RGBA", frame_size, fig.canvas.buffer_rgba(),
                                           "raw", "RGBA", 0, 1).convert("RGB"))

        # one palette shared by every frame, rather than an adaptive palette per
        # frame: the frames differ only in where a few lines sit, so frame 0's
        # colors cover the run, and matching them is far cheaper than rebuilding
        palette = frames[0].convert("P", palette=Image.ADAPTIVE, colors=128)
        frames  = [f.quantize(palette=palette, dither=Image.NONE) for f in frames]

        fname = "figures/simulation.gif"
        frames[0].save(fname, save_all=True, append_images=frames[1:],
                       duration=int(1_000 / fps), loop=0)
        plt.close(fig)
        print(f"Saved {fname}")

    def run(self):
        """Advance either the blade-enabled or tracked-body simulation."""
        t = 0.0
        for _ in range(int(self.stop_time / self.dt)):
            t += self.dt

            if self.enable_blade:
                # Cut at the current pose, settle on the new terrain, then log
                # blade geometry without cutting the same frame twice.
                self._blade_update()
                neighbor_points = self._body_update()
                blade_points, _ = self._blade_update(deform=False)
            else:
                points, neighbor_points = self._body_update()

            self.total_distance += np.linalg.norm(self.q_dot[:3] * self.dt)

            if self.total_distance >= self.stop_distance:
                break

            self.log.append([t, *self.q])
            if self.enable_blade:
                self.neighbor_log.append(np.array(neighbor_points))
                self.blade_log.append(blade_points)
                self.grid_log.append(self._grid_heights())
            else:
                self.point_log.append(np.array(points[1:]))
                self.neighbor_log.append(np.vstack(neighbor_points))

    def _body_update(self):
        #TODO: see if this if should go after the q update
        if not self.enable_blade:
            return self._tracked_body_update()

        if not self.is_initalization:
            self.q += self.dt * self.q_dot
        else:
            self.is_initalization = False

        # q remains the pose origin and a logged visualization point; both
        # entire track lines alone determine height, roll, and pitch.
        neighbor_points = self._get_neighbor_points(self.q)
        self._settle_tracks(self.q[3:6].copy())
       
        return neighbor_points

    def _blade_update(self, deform=True):
        orient = self.q[3:6].copy()
        R      = self._blade_rotation_lg(orient)
        p0 = self.q[:3] + R @ np.array([self.L, -self.B1 / 2, -self.blade_cut_depth])
        p1 = self.q[:3] + R @ np.array([self.L,  self.B1 / 2, -self.blade_cut_depth])

        # TODO: when go back to optimize the code consider using the vector between p0 and p1 so not O(n) points
        length = np.linalg.norm(p1 - p0)
        n_segments = max(1, int(np.ceil(length / self.subdivision)))

        xyz = [
        (1 - t) * p0 + t * p1
            for t in np.linspace(0.0, 1.0, n_segments + 1)
        ]

        contacts_by_tile = {}
        contact_points = []
        for point in xyz:
            contacts_by_tile.setdefault(self._grid_cell(point), []).append(point)
            contact_points.append(point)

        if deform:
            self._deform_blade_tiles(contacts_by_tile)

        blade_points = [np.concatenate((point, orient)) for point in xyz]
        blade_neighbors = [self._get_neighbor_points(point) for point in contact_points]
        return blade_points, blade_neighbors

    def _deform_blade_tiles(self, contacts_by_tile):
        """Cut only the two vertices ahead of the blade in each contacted tile."""
        vel = np.array(self.q_dot[:3])
        if self.is_backwards:
            vel *= -1

        # Determine the direct footprint first. A shared vertex is still cut
        # only once when the blade contacts adjacent tiles in the same frame.
        direct_nodes = {}
        for (i, j), contacts in contacts_by_tile.items():
            local_blade_z = min(point[2] for point in contacts)
            corners = ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1))
            for ci, cj in corners:
                cx, cy, _ = self.grid_pts[ci][cj]
                if any((cx - point[0]) * vel[0] + (cy - point[1]) * vel[1] > 0.0
                       for point in contacts):
                    # Shared vertices take the deepest actual blade plane from
                    # their contacted tiles, never a relative depth copied from
                    # terrain at a different lateral height.
                    direct_nodes[(ci, cj)] = min(
                        direct_nodes.get((ci, cj), np.inf), local_blade_z
                    )

        if not direct_nodes:
            return

        for (i, j), local_blade_z in direct_nodes.items():
            x, y, height = self.grid_pts[i][j]

            min_height = self.starting_grid_heights[i, j] - self.max_world_cut_depth
            blade_z = max(local_blade_z, min_height)
            if blade_z < height :
                self.grid_pts[i][j] = (x, y, blade_z)
                self.surf_grid.nodes[(i, j)]['z'] = blade_z

    def _grid_heights(self):
        """Snapshot of every node height, indexed [i][j], for the GIF's surface."""
        return np.array([[pt[2] for pt in col] for col in self.grid_pts])


class Body(Surface):
    """Backward-compatible tracked-body wrapper."""

    def __init__(self, *args, **kwargs):
        kwargs["enable_blade"] = False
        super().__init__(*args, **kwargs)


class Blade(Surface):
    """Backward-compatible blade-enabled wrapper."""

    def __init__(self, *args, **kwargs):
        kwargs["enable_blade"] = True
        super().__init__(*args, **kwargs)


# Compatibility with the pre-unification private entry point.
Surface._run = Surface.run

if __name__ == "__main__":
    simulation = Surface(
        enable_blade       = True,
        is_backwards       = False,
        is_uphill          = True,
        is_surface_pitched = False,
        is_surface_rolled  = True,
        blade_local_roll   = -0.3, 
        blade_local_yaw    = 0.0, 
        blade_pitch        = 0.0
    )
    simulation.run_and_plot()
