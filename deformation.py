import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class Surface:
    def __init__(self, is_uphill = True, is_surface_pitched: bool = False, is_surface_rolled: bool = False, is_backwards: bool = False):
        # simulation parameters
        self.division_factor    = 4
        self.b                  = 1.75
        self.offset             = np.array([0, 0, self.b]) if is_uphill else np.array([0, 0, -self.b])
        self.surface_abg        = np.array([ 0.0, 0.0, 0.0])
        self.u_split            = 0  # u-value where the grid switches to surface_abg2
        self.v_split            = 2  # u-value where the grid switches to surface_abg2
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
        # Excavation is distance-based, not timestep-based. This default takes
        # roughly five grid-cell lengths to reach the blade's full depth.
        self.max_dig_depth_per_meter = self.H / (5 * self.subdivision)
        self.blade_blend_weight      = 0.0

        if is_backwards:
            self.q_dot   *= -1
            self.u_split *= self.q_dot[0] / np.linalg.norm(self.q_dot)
            self.v_split *= self.q_dot[1] / np.linalg.norm(self.q_dot)
        self.is_backwards       = is_backwards
        self.is_surface_pitched = is_surface_pitched
        self.is_surface_sigmoid = False
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
        # _neighbor_deformation writes new heights here (and through to the
        # graph), so this - not surf_grid - is the live surface
        self.grid_pts = [[(self.surf_grid.nodes[(i, j)]['x'],
                           self.surf_grid.nodes[(i, j)]['y'],
                           self.surf_grid.nodes[(i, j)]['z'])
                          for j in range(len(self.vs))] for i in range(len(self.us))]

        # put the body on the surface at the start of the simulation
        neighbor_points, self.q[2] = self._point_height(self.q)
        self.log.append([0, *self.q])
        self.neighbor_log.append(np.array(neighbor_points))
        self.grid_log.append(self._grid_heights())

        # seed the blade points the same way, so blade_log stays aligned frame
        # for frame with log/neighbor_log/grid_log
        blade_points, blade_neighbors = self._blade_update()
        neighbor_points.extend(point for neighbors in blade_neighbors for point in neighbors)
        self.neighbor_log.append(np.array(neighbor_points))
        self.blade_log.append(blade_points)

    @property
    def subdivision(self):
        """Grid spacing, sized relative to the dozer width self.b."""
        return self.b / self.division_factor

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
                    
                    # w = u_clip if self.is_surface_pitched else v_clip
                    # if self.is_surface_rolled and self.is_surface_pitched:
                    w = u_clip + v_clip
                    # elif self.is_surface_rolled:
                        # w = v_clip if self.is_surface_pitched else u_clip
                        
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
         
        # roll  = np.arctan2(s_l, np.sqrt(1.0 + s_f**2))
        roll = 0
        pitch = np.arctan2(-s_f, 1.0)                 
        yaw   = np.arctan2(vel[1], vel[0])

        return np.array([roll, pitch, yaw])
    
    def _bilinear_gradient(self, point, corners):
        """(dh/dx, dh/dy) of the same bilinear patch _bilinear_height blends,
        found by differentiating it w.r.t. the tile's (s,t) edge parameters
        and mapping back to the xy-plane via the edge vectors."""
        (x1, y1, h1), (x2, y2, h2), (x3, y3, h3), (x4, y4, h4) = corners

        dx,  dy  = point[0] - x1, point[1] - y1
        e_s = np.array([x2 - x1, y2 - y1])
        e_t = np.array([x3 - x1, y3 - y1])
        es2, et2 = e_s @ e_s, e_t @ e_t

        s = min(max((dx * e_s[0] + dy * e_s[1]) / es2, 0.0), 1.0)
        t = min(max((dx * e_t[0] + dy * e_t[1]) / et2, 0.0), 1.0)

        dh_ds = (h2 - h1) * (1 - t) + (h4 - h3) * t
        dh_dt = (h3 - h1) * (1 - s) + (h4 - h2) * s

        # [dh/dx, dh/dy]^T = dh_ds*[ds/dx, ds/dy]^T + dh_dt*[dt/dx, dt/dy]^T,
        # [ds/dx, ds/dy]^T = e_s / (e_s . e_s),
        # [dt/dx, dt/dy]^T = e_t / (e_t . e_t).
        return dh_ds * e_s / es2 + dh_dt * e_t / et2

    def _point_height(self, point):
        neighbor_points   = self._get_neighbor_points(point)
        height_to_surface = self._bilinear_height(point, neighbor_points)
        return neighbor_points, height_to_surface

    def _get_neighbor_points(self, point):
        i, j = self._grid_cell(point)
        col_i, col_i1 = self.grid_pts[i], self.grid_pts[i + 1]
        return [col_i[j], col_i1[j], col_i[j + 1], col_i1[j + 1]]
    
    def _grid_cell(self, point):
        """(i, j) index of the grid tile containing point's (x, y), clipped to the grid."""
        i = int((point[0] - self.us[0]) // self.subdivision)
        j = int((point[1] - self.vs[0]) // self.subdivision)
        i_max, j_max = len(self.us) - 2, len(self.vs) - 2
        i = 0 if i < 0 else i_max if i > i_max else i
        j = 0 if j < 0 else j_max if j > j_max else j
        return i, j

    def _bilinear_height(self, point, corners):
        """Bilinear height at point's (x,y) from 4 corner vertices [h1,h2,h3,h4] = [(i,j),(i+1,j),(i,j+1),(i+1,j+1)]."""
        (x1, y1, h1), (x2, y2, h2), (x3, y3, h3), (x4, y4, h4) = corners

        # same bilinear as before, but on plain floats so numpy's array-function
        # dispatch (np.dot / np.clip on scalars) never runs on the hot path
        dx,   dy   = point[0] - x1, point[1] - y1
        esx,  esy  = x2 - x1, y2 - y1
        etx,  ety  = x3 - x1, y3 - y1
        s = (dx * esx + dy * esy) / (esx * esx + esy * esy)
        t = (dx * etx + dy * ety) / (etx * etx + ety * ety)
        s = 0.0 if s < 0.0 else 1.0 if s > 1.0 else s
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t

        return h1 * (1 - s) * (1 - t) + h2 * s * (1 - t) + h3 * (1 - s) * t + h4 * s * t
            
    def run_and_plot(self, show_neighbors: bool = False):
        self._run()

        print("Rendering GIF...")
        
        # set axis limits based on the logged data (and neighbor points, if any)
        margin = 0.5
        data          = np.array(self.log)[::2]  # every 2nd frame represented to speed up rendering
        # (frames, blade_points, 6), ordered from the right blade end to the left.
        # The point count follows the blade/grid resolution rather than being
        # fixed at the two blade endpoints.
        blade_data    = np.asarray(self.blade_log)[::2]
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
        bound_x = [grid_x.ravel(), blade_data[:, :, 0].ravel()]
        bound_y = [grid_y.ravel(), blade_data[:, :, 1].ravel()]
        bound_z = [np.asarray(self.grid_log).ravel(), blade_data[:, :, 2].ravel()]
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

        # blue arrow marking the center of mass q, showing q's own orientation;
        # q sticks to the surface because q[2] already comes from _point_height
        # each frame. mplot3d quiver has no in-place update, so the 3D twin of
        # each artist is held in a 1-element list and removed/recreated each frame
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

        if show_neighbors:
            set_neighbors(0)
        set_q_point(0)

        legend_handles = [q_arrow_side, blade_arrow_side]
        legend_labels  = ["q (center of mass)", "blade contact points"]
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
        flat_artists = [grid_back, grid_side]
        if show_neighbors:
            flat_artists += [green_back, green_side, green_top]
        flat_artists += [q_arrow_top, q_arrow_back, q_arrow_side,
                         blade_arrow_top, blade_arrow_back, blade_arrow_side]
        three_d       = [grid_3d] + ([green_3d] if show_neighbors else [])

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

        fps        = 20
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
                       duration=int(1000 / fps), loop=0)
        plt.close(fig)
        print(f"Saved {fname}")

    def _run(self):
        t = 0.0
        for _ in range(int(self.stop_time / self.dt)):
            # update variables
            t += self.dt

            blade_points, _ = self._blade_update()
            neighbor_points = self._body_update()
            

            self.total_distance += np.linalg.norm(self.q_dot[:3] * self.dt)

            # loop termination check
            if self.total_distance >= self.stop_distance:
                break

            # log variables for plotting
            self.log.append([t, *self.q])
            self.neighbor_log.append(np.array(neighbor_points))
            self.blade_log.append(blade_points)
            self.grid_log.append(self._grid_heights())

    def _body_update(self):
        if not self.is_initalization:
            self.q += self.dt * self.q_dot
        else:
            self.is_initalization = False

        # seed the body height + orientation from the tile under the center of mass
        neighbor_points, self.q[2] = self._point_height(self.q)
        self.q[3:6]                = self._point_orientation(self._get_neighbor_points(self.q))

        return neighbor_points

    def _blade_update(self):
        """Deformation contact points sampled across the blade, offset from q
        by self.L forward (local +x), ±self.B1/2 laterally (local y -- right
        is -B1/2, its mirror across q left is +B1/2), and self.H/4 down
        (local -z), rotated into the global frame the same way util.py places
        the two tracks (R @ local_offset). The blade is a rigid straight edge,
        so both ends share whichever end's rotated offset sits lower, rather
        than each cutting its own (possibly shallower) depth. Each point cuts
        any ground behind it down to that height. Contact points that land in
        the same tile are aggregated, so one frame can cut each tile only once.

        Returns the right-to-left blade points, each a 6-vector
        [x, y, z, roll, pitch, yaw] (sharing q's orientation), plus each
        point's tile's 4 deformed corner neighbors.
        """
        R      = self._rotation_lg(*self.q[3:6])
        orient = self.q[3:6].copy()

        p0 = self.q[:3] + R @ np.array([self.L, -self.B1 / 2, -self.H / 4])
        p1 = self.q[:3] + R @ np.array([self.L,  self.B1 / 2, -self.H / 4])

        # TODO: when go back to optimize the code consider using the vector between p0 and p1 so not O(n) points
        length = np.linalg.norm(p1 - p0)
        n_segments = max(1, int(np.ceil(length / self.subdivision)))

        xyz = [
        (1 - t) * p0 + t * p1
            for t in np.linspace(0.0, 1.0, n_segments + 1)
        ]

        z   = min(p[2] for p in xyz)

        contacts_by_tile = {}
        contact_points = []
        for x, y, _ in xyz:
            point = np.array([x, y, z])
            contacts_by_tile.setdefault(self._grid_cell(point), []).append(point)
            contact_points.append(point)

        self._deform_blade_tiles(contacts_by_tile)

        blade_points = [np.concatenate(([x, y, z], orient)) for x, y, _ in xyz]
        blade_neighbors = [self._get_neighbor_points(point) for point in contact_points]
        return blade_points, blade_neighbors

    def _deform_blade_tiles(self, contacts_by_tile):
        """Apply one distance-limited cut per reached tile, then soften its edge."""
        travel_distance = np.linalg.norm(self.q_dot[:3] * self.dt)
        max_cut = self.max_dig_depth_per_meter * max(0.0, travel_distance)
        if max_cut == 0.0:
            return

        vel     = np.array(self.q_dot[:3])
        if self.is_backwards:
            vel *= -1

        # Determine the direct footprint first. A vertex shared by adjacent
        # touched tiles is still cut only once in this frame.
        direct_nodes = set()
        blade_z = min(point[2] for contacts in contacts_by_tile.values()
                      for point in contacts)
        for (i, j), contacts in contacts_by_tile.items():
            corners = ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1))
            reached = any((cx - point[0]) * vel[0] + (cy - point[1]) * vel[1] < 0.0
                          for point in contacts
                          for cx, cy, _ in (self.grid_pts[ci][cj] for ci, cj in corners))
            if reached:
                direct_nodes.update(corners)

        drops = {}
        for i, j in direct_nodes:
            x, y, height = self.grid_pts[i][j]
            new_height = max(blade_z, height - max_cut)
            if new_height < height:
                drops[(i, j)] = height - new_height
                self.grid_pts[i][j] = (x, y, new_height)
                self.surf_grid.nodes[(i, j)]['z'] = new_height

        # A single surrounding vertex ring gets a smaller, capped depression.
        # This removes the abrupt one-tile step without broadening the cut
        # indefinitely across repeated frames.
        blend_drops = {}
        i_max, j_max = len(self.us) - 1, len(self.vs) - 1
        for (i, j), drop in drops.items():
            for ni in range(max(0, i - 1), min(i_max, i + 1) + 1):
                for nj in range(max(0, j - 1), min(j_max, j + 1) + 1):
                    if (ni, nj) not in direct_nodes:
                        blend_drops[(ni, nj)] = max(
                            blend_drops.get((ni, nj), 0.0),
                            drop * self.blade_blend_weight,
                        )

        for (i, j), drop in blend_drops.items():
            x, y, height = self.grid_pts[i][j]
            new_height = max(blade_z, height - drop)
            if new_height < height:
                self.grid_pts[i][j] = (x, y, new_height)
                self.surf_grid.nodes[(i, j)]['z'] = new_height

    def _grid_heights(self):
        """Snapshot of every node height, indexed [i][j], for the GIF's surface."""
        return np.array([[pt[2] for pt in col] for col in self.grid_pts])

if __name__ == "__main__":
    my_surface = Surface(is_uphill=False, is_surface_pitched=False, is_surface_rolled = True, is_backwards=False)
    my_surface.run_and_plot()
