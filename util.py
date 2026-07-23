import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class Surface:
    def __init__(self, is_uphill = True, is_surface_pitched: bool = False, is_surface_rolled: bool = False, is_backwards: bool = False):
        # simulation parameters
        self.b                  = 1.75
        self.offset             = np.array([0, 0, self.b]) if is_uphill else np.array([0, 0, -self.b])
        self.surface_abg        = np.array([ 0.0, 0.0, 0.0])
        self.u_split            = 2  # u-value where the grid switches to surface_abg2
        self.v_split            = 0  # u-value where the grid switches to surface_abg2
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
        self.is_surface_sigmoid = False
        self.is_surface_rolled = is_surface_rolled
        self.log                = []
        self.neighbor_points    = []
        self.neighbor_log       = []
        self.point_log          = []  # per frame: (3, 6) rows front, center, back
        
        # set up the surface grid
        self.u_range       = (-self.b/2, 3 * self.b) if is_surface_pitched else (-self.b/2,     self.b/2) 
        self.v_range       = (-self.b/2,   self.b/2) if is_surface_pitched else (-self.b/2, 3 * self.b)
        if is_backwards:
            self.u_range       = (-2.5 * self.b, self.b/2) if is_surface_pitched else (-self.b/2,     self.b/2) 
            self.v_range       = (-self.b/2,     self.b/2) if is_surface_pitched else (-2.5 * self.b, self.b/2)
        self.us            = np.arange(self.u_range[0], self.u_range[1] + self.subdivision, self.subdivision)
        self.vs            = np.arange(self.v_range[0], self.v_range[1] + self.subdivision, self.subdivision)
        
        stop_index         = 0 if self.is_backwards else -1 
        self.stop_distance = abs(self.us[stop_index]) if is_surface_pitched else abs(self.vs[stop_index])
        
        self.surf_grid     = self._surface_grid()

        # put track on the surface at the start of the simulation
        points, neighbor_points = self._body_update()
        self.log.append([0, *self.q])
        self.point_log.append(np.array(points[1:])) 
        self.neighbor_log.append(np.vstack(neighbor_points)) 

    @property
    def subdivision(self, division_factor: float = 2.0):
        """Grid spacing, sized relative to the dozer width self.b."""
        return self.b / division_factor

    def _surface_grid(self):
        R_surf     = self._rotation_lg(*self.surface_abg)
        e1, e2, e3 = R_surf[:, 0], R_surf[:, 1], R_surf[:, 2]
        G          = nx.grid_2d_graph(len(self.us), len(self.vs))
        u_start    = self.us[self.us <= self.u_split][-1]
        v_start    = self.vs[self.vs <= self.v_split][-1]
        ramp_width = max(1, round(self.transition_tiles)) * self.subdivision
        for i, u in enumerate(self.us):
            for j, v in enumerate(self.vs):
                if self.is_surface_sigmoid:
                    x = u
                    y = v
                    z = np.sin(u/3) * np.cos(y*4)
                else:
                    # full offset for * <= *_start, then a linear ramp to zero over ramp_width
                    u_clip  = np.clip((u - u_start) / ramp_width, 0.0, 1.0)
                    v_clip  = np.clip((v - v_start) / ramp_width, 0.0, 1.0)
                    
                    w = u_clip if self.is_surface_pitched else v_clip
                    if self.is_surface_rolled:
                        w = u_clip + v_clip
                    elif self.is_surface_rolled and not self.is_surface_pitched:
                        w = v_clip if self.is_surface_pitched else u_clip
                        
                    x, y, z = u * e1 + v * e2 + w * self.offset * e3

                node = G.nodes[(i, j)]
                node["x"], node["y"], node["z"] = float(x), float(y), float(z)
                node["visited_last"] = False
        return G
    
    def _rotation_lg(self, a, B, g):
        """Rotation matrix: local → global frame"""
        return self._rotation_gl(a, B, g).T

    def _rotation_gl(self, a, B, g):
        """Rotation matrix: global → local frame"""
        # change to accept input array
        sa, ca = np.sin(a), np.cos(a)
        sB, cB = np.sin(B), np.cos(B)
        sg, cg = np.sin(g), np.cos(g)

        return np.array([
            [cB * cg,   sa * sB * cg - ca * sg,   ca * sB * cg + sa * sg],
            [cB * sg,   sa * sB * sg + ca * cg,   ca * sB * sg - sa * cg],
            [-sB,                      sa * cB,                  ca * cB]
        ]).T
 
    def run_and_plot(self, show_neighbors: bool = False):
        self._run()

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
            xs = [rc[0], rf[0], np.nan, lc[0], lf[0]]
            ys = [rc[1], rf[1], np.nan, lc[1], lf[1]]
            zs = [rc[2], rf[2], np.nan, lc[2], lf[2]]
            link.set_data(xs, ys)
            link.set_3d_properties(zs)
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
            xs = [rb[0], rc[0], np.nan, lb[0], lc[0]]
            ys = [rb[1], rc[1], np.nan, lb[1], lc[1]]
            zs = [rb[2], rc[2], np.nan, lb[2], lc[2]]
            link_lower.set_data(xs, ys)
            link_lower.set_3d_properties(zs)
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

    def _run(self):
        t = 0.0
        for _ in range(int(self.stop_time / self.dt)):
            # update variables
            t += self.dt

            points, neighbor_points = self._body_update()
            self.total_distance  += np.linalg.norm(self.dt * self.q_dot[0:3])
            
            # loop termination check
            if self.total_distance >= self.stop_distance:
                break

            # log variables for plotting
            self.log.append([t, *self.q])
            self.point_log.append(np.array(points[1:]))  # (3, 6): rows front, center, back
            self.neighbor_log.append(np.vstack(neighbor_points))  # (16, 3): 4 points each for q, front, center, back
    
    def _body_update(self):
        if not self.is_initalization:
            self.q += self.dt * self.q_dot
        else:
            self.is_initalization = False

        # seed the body height + orientation from the tile under the center of mass
        neighbors_q, self.q[2] = self._point_height(self.q)
        self.q[3:6]            = self._point_orientation(self._get_neighbor_points(self.q))

        # rigid contact-averaged solve, then build both tracks as parallel offsets
        track_points, track_neighbors = self._body_update_iteration()

        points          = [self.q, *track_points]
        neighbor_points = [neighbors_q, *track_neighbors]
        return points, neighbor_points

    def _body_update_iteration(self):
        """
        Solve one rigid-body pose (height + roll/pitch/yaw) so the four track
        contact points settle onto the surface, then return the six track points
        as [right, left] x [front, center, back]. The body is rigid, so both
        tracks share a single half_track vector and stay parallel by
        construction. Orientation is contact-averaged: _point_orientation blends
        the four snapped corners, so pitch follows the front-vs-back heights and
        roll the left-vs-right heights.
        """
        for _ in range(20):
            R          = self._rotation_lg(*self.q[3:6])
            half_width = R @ np.array([         0, self.b / 2, 0])
            half_track = R @ np.array([self.l / 2,          0, 0])

            right_center = self.q[:3] - half_width
            left_center  = self.q[:3] + half_width
            rf, rb = right_center + half_track, right_center - half_track
            lf, lb = left_center  + half_track, left_center  - half_track

            # drop the four track-contact points onto the surface
            contacts = [rf, rb, lf, lb]
            heights  = [self._point_height(p)[1] for p in contacts]
            for p, h in zip(contacts, heights):
                p[2] = h

            # contact-averaged orientation from the settled corners
            # (_point_orientation reads [rb, rf, lb, lf] as one bilinear patch)
            new_orient = self._point_orientation([rb, rf, lb, lf])

            # rest the rigid body on its highest support: the lowest height that
            # keeps every point of both tracks on or above the surface, so no
            # track segment is ever submerged
            new_z = self._resting_height(self._rotation_lg(*new_orient))

            converged = (abs(new_z - self.q[2]) < 1e-6 and
                         np.all(np.abs(new_orient - self.q[3:6]) < 1e-6))
            self.q[2], self.q[3:6] = new_z, new_orient
            if converged:
                break

        # rebuild both tracks rigidly from the converged pose so they stay parallel
        R          = self._rotation_lg(*self.q[3:6])
        half_width = R @ np.array([         0, self.b / 2, 0])
        half_track = R @ np.array([self.l / 2,          0, 0])
        orient     = self.q[3:6].copy()

        right_center = self.q[:3] - half_width
        left_center  = self.q[:3] + half_width
        track_xyz    = [right_center + half_track, right_center, right_center - half_track,
                        left_center  + half_track, left_center,  left_center  - half_track]

        track_points    = [np.concatenate((xyz, orient))           for xyz in track_xyz]
        track_neighbors = [np.array(self._get_neighbor_points(xyz)) for xyz in track_xyz]
        return track_points, track_neighbors

    def _resting_height(self, R):
        """
        Lowest body height q[2] (for orientation R and the current q[:2]) that
        keeps every point of both rigid tracks on or above the surface, so the
        body rests on its highest contact and no track segment is submerged.

        The surface is piecewise planar with kinks only on the grid lines, so a
        straight track's deepest penetration always occurs at an endpoint or a
        grid-line crossing; between those the clearance varies linearly. Sampling
        the ends plus every grid crossing therefore finds the true deepest point
        exactly (a uniform scan would step over the kinks).
        """
        fwd     = R[:, 0]
        lat     = R[:, 1] * (self.b / 2)      # body center -> track lateral offset (local +y)
        half_l  = self.l / 2
        new_z   = -np.inf
        for side in (1.0, -1.0):              # left (+lat) and right (-lat) tracks
            base = self.q[:3] + side * lat
            ss   = {-half_l, half_l}          # track ends

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
                    near   = grid[lo:hi + 2]
                    # solve grid = base[axis] + s*fwd[axis] for every nearby grid value at once
                    for s in (near - base[axis]) / fwd[axis]:
                        if -half_l < s < half_l:   # keep only crossings within the track
                            ss.add(float(s))
            for s in ss:
                grid_crossing_point = base + s * fwd
                surface_height      = self._point_height(grid_crossing_point)[1]
                new_z               = max(new_z, (surface_height - grid_crossing_point[2]) + self.q[2])
        return new_z

    def _point_height(self, point):
        neighbor_points   = self._get_neighbor_points(point)
        height_to_surface = self._bilinear_height(point, neighbor_points)
        return neighbor_points, height_to_surface

    def _grid_cell(self, point):
        """(i, j) index of the grid tile containing point's (x, y), clipped to the grid."""
        i = int(np.clip((point[0] - self.us[0]) // self.subdivision, 0, len(self.us) - 2))
        j = int(np.clip((point[1] - self.vs[0]) // self.subdivision, 0, len(self.vs) - 2))
        return i, j

    def _get_neighbor_points(self, point):
        i, j = self._grid_cell(point)

        corners         = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
        neighbor_points = [(self.surf_grid.nodes[n]['x'], self.surf_grid.nodes[n]['y'], self.surf_grid.nodes[n]['z']) for n in corners]

        return neighbor_points
    
    def _bilinear_height(self, point, corners):
        """Bilinear height at point's (x,y) from 4 corner vertices [h1,h2,h3,h4] = [(i,j),(i+1,j),(i,j+1),(i+1,j+1)]."""
        p1, p2, p3, p4 = (np.array(c) for c in corners)
        h1, h2, h3, h4 = p1[2], p2[2], p3[2], p4[2]

        xy  = np.array(point[:2])
        e_s = (p2 - p1)[:2]
        e_t = (p3 - p1)[:2]

        s = np.clip(np.dot(xy - p1[:2], e_s) / np.dot(e_s, e_s), 0.0, 1.0)
        t = np.clip(np.dot(xy - p1[:2], e_t) / np.dot(e_t, e_t), 0.0, 1.0)

        return h1 * (1 - s) * (1 - t) + h2 * s * (1 - t) + h3 * (1 - s) * t + h4 * s * t
            
    def _point_orientation(self, corners):
        """
        Roll/pitch/yaw of the center of mass crossing the current tile: pitch and
        roll come from the tile's height-field gradient (the edges' angles,
        blended the same s,t weights as _bilinear_height) read off along and
        across the direction of travel; yaw is the global-frame heading of
        q_dot, arctan2(vel_y, vel_x), so (roll, pitch, yaw) form a consistent
        ZYX Euler triple for _rotation_lg.
        """
        grad_xy = self._bilinear_gradient(self.q, corners)

        vel   = np.array(self.q_dot[:3])
        if self.is_backwards:
            vel *= -1
        speed = np.linalg.norm(vel[:2])
        if speed < 1e-9:
            return self.q[3:6].copy()

        # the velocity in the the local body frame
        fwd_xy  = vel[:2] / speed
        left_xy = np.array([-fwd_xy[1], fwd_xy[0]])
        
        s_l = np.dot(grad_xy, left_xy)  # lateral slope
        s_f = np.dot(grad_xy, fwd_xy)   # forward slope
         
        roll  = np.arctan2(s_l, np.sqrt(1.0 + s_f**2))
        pitch = np.arctan2(-s_f, 1.0)                 
        yaw   = np.arctan2(vel[1], vel[0])

        return np.array([roll, pitch, yaw])
    
    def _bilinear_gradient(self, point, corners):
        """(dh/dx, dh/dy) of the same bilinear patch _bilinear_height blends,
        found by differentiating it w.r.t. the tile's (s,t) edge parameters
        and mapping back to the xy-plane via the edge vectors."""
        p1, p2, p3, p4 = (np.array(c) for c in corners)
        h1, h2, h3, h4 = p1[2], p2[2], p3[2], p4[2]

        xy  = np.array(point[:2])
        e_s = (p2 - p1)[:2]
        e_t = (p3 - p1)[:2]

        s = np.clip(np.dot(xy - p1[:2], e_s) / np.dot(e_s, e_s), 0.0, 1.0)
        t = np.clip(np.dot(xy - p1[:2], e_t) / np.dot(e_t, e_t), 0.0, 1.0)

        dh_ds = (h2 - h1) * (1 - t) + (h4 - h3) * t
        dh_dt = (h3 - h1) * (1 - s) + (h4 - h2) * s
        
        # [dh/dx, dh/dy]^T = dh_ds*[ds/dx, ds/dy]^T + dh_dt*[dt/dx, dt/dy]^T,
        # [ds/dx, ds/dy]^T = e_s / (e_s . e_s), 
        # [dt/dx, dt/dy]^T = e_t / (e_t . e_t).
        return dh_ds * e_s / np.dot(e_s, e_s) + dh_dt * e_t / np.dot(e_t, e_t)

if __name__ == "__main__":
    my_surface = Surface(is_uphill=True, is_surface_pitched=True, is_surface_rolled = True, is_backwards=False)
    my_surface.run_and_plot()   