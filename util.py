import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class Surface:
    def __init__(self, is_uphill = True, is_surface_pitched: bool = False, is_backwards: bool = False):
        # simulation parameters
        self.b                  = 1.75
        self.offset             = np.array([0, 0, 2*self.b]) if is_uphill else np.array([0, 0, -2*self.b])
        self.surface_abg        = np.array([ 0.0, 0.0, 0.0])
        self.u_split            = 2  # u-value where the grid switches to surface_abg2
        self.v_split            = 2  # u-value where the grid switches to surface_abg2
        self.transition_tiles   = self.b                      # tiles over which the offset ramps down past u_split
        self.q                  = np.array([0.0, 0.0, 0.0, self.surface_abg[0], self.surface_abg[1], 0.0 if is_surface_pitched else np.pi / 2])
        self.q_dot              = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]) if is_surface_pitched else np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0]) 
        self.is_initalization   = True
        self.is_surface_sigmoid = False
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
        self.log                = []
        self.neighbor_points    = []
        self.neighbor_log       = []
        self.center_log         = []
        self.front_log          = []
        self.back_log           = []
        
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
        points, neighbor_points = self._track_update()
        self.log.append([0, *self.q])
        self.center_log.append(np.array(points[1]))
        self.front_log.append(np.array(points[2]))
        self.back_log.append(np.array(points[3]))
        self.neighbor_log.append(neighbor_points[0] + neighbor_points[1] + neighbor_points[2] + neighbor_points[3])
        
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
                    w       = u_clip if self.is_surface_pitched else v_clip
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
 
    def run_and_plot(self):
        self._run()

        print("Rendering GIF...")
        
        # set axis limits based on the logged data (and neighbor points, if any)
        margin = 0.5
        data          = np.array(self.log)[::2] # every 2nd frame represented to speed up rendering
        center_data   = np.array(self.center_log)[::2]
        front_data    = np.array(self.front_log)[::2]
        back_data     = np.array(self.back_log)[::2]
        neighbor_data = self.neighbor_log[::2]
        all_neighbor_pts = np.array([pt for frame in self.neighbor_log for pt in frame])
        grid_pts = np.array([[self.surf_grid.nodes[n]['x'],
                              self.surf_grid.nodes[n]['y'],
                              self.surf_grid.nodes[n]['z']] for n in self.surf_grid.nodes])

        # include the surface grid so it isn't clipped flush at a panel edge
        all_x = np.concatenate([data[:, 1], center_data[:, 0], front_data[:, 0], back_data[:, 0], all_neighbor_pts[:, 0], grid_pts[:, 0]])
        all_y = np.concatenate([data[:, 2], center_data[:, 1], front_data[:, 1], back_data[:, 1], all_neighbor_pts[:, 1], grid_pts[:, 1]])
        all_z = np.concatenate([data[:, 3], center_data[:, 2], front_data[:, 2], back_data[:, 2], all_neighbor_pts[:, 2], grid_pts[:, 2]])
        
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
        
        link_lower,      = ax.plot([], [], [], color='pink', linewidth=1.5)
        link_top_lower,  = ax_top.plot([], [], color='pink', linewidth=1.5)
        link_back_lower, = ax_back.plot([], [], color='pink', linewidth=1.5)
        link_side_lower, = ax_side.plot([], [], color='pink', linewidth=1.5)

        # green scatter artists for grid vertices within one tile length of the point, updated each frame
        green_3d   = ax.scatter([], [], [], color='green', s=40, zorder=5)
        green_top  = ax_top.scatter([], [], color='green', s=40, zorder=5)
        green_back = ax_back.scatter([], [], color='green', s=40, zorder=5)
        green_side = ax_side.scatter([], [], color='green', s=40, zorder=5)

        # blue star marking the center of mass q, drawn as its own point (no link to the track)
        q_point_3d   = ax.scatter([], [], [], color='blue', marker='*', s=120, zorder=7)
        q_point_top  = ax_top.scatter([], [], color='blue', marker='*', s=120, zorder=7)
        q_point_back = ax_back.scatter([], [], color='blue', marker='*', s=120, zorder=7)
        q_point_side = ax_side.scatter([], [], color='blue', marker='*', s=120, zorder=7)

        # red arrow at the tracked point showing the center of mass's orientation,
        # i.e. the local forward axis (R_lg(*q[3:6])[:, 0]) for that frame's roll/pitch/yaw
        arrow_len = self.subdivision * 0.6

        def _forward(i):
            return self._rotation_lg(*data[i, 4:7])[:, 0]

        def _center_forward(i):
            return self._rotation_lg(*center_data[i, 3:])[:, 0]

        def _front_forward(i):
            return self._rotation_lg(*front_data[i, 3:])[:, 0]

        def _back_forward(i):
            return self._rotation_lg(*back_data[i, 3:])[:, 0]
        fwd0 = _forward(0)
        qdot_top  = ax_top.quiver(data[0, 1], data[0, 2], fwd0[0], fwd0[1],
                                   color='red', scale=1 / arrow_len, scale_units='xy',
                                   angles='xy', zorder=6)
        qdot_back = ax_back.quiver(data[0, 2], data[0, 3], fwd0[1], fwd0[2],
                                    color='red', scale=1 / arrow_len, scale_units='xy',
                                    angles='xy', zorder=6)
        qdot_side = ax_side.quiver(data[0, 1], data[0, 3], fwd0[0], fwd0[2],
                                    color='red', scale=1 / arrow_len, scale_units='xy',
                                    angles='xy', zorder=6)
        qdot_3d   = [None]  # mplot3d quiver has no in-place update, so remove/recreate each frame

        # orange arrow at the front point, same forward axis (f shares q's orientation)
        front_arrow_top  = ax_top.quiver(front_data[0, 0], front_data[0, 1], fwd0[0], fwd0[1],
                                          color='darkorange', scale=1 / arrow_len, scale_units='xy',
                                          angles='xy', zorder=6)
        front_arrow_back = ax_back.quiver(front_data[0, 1], front_data[0, 2], fwd0[1], fwd0[2],
                                           color='darkorange', scale=1 / arrow_len, scale_units='xy',
                                           angles='xy', zorder=6)
        front_arrow_side = ax_side.quiver(front_data[0, 0], front_data[0, 2], fwd0[0], fwd0[2],
                                           color='darkorange', scale=1 / arrow_len, scale_units='xy',
                                           angles='xy', zorder=6)
        front_arrow_3d = [None]

        # pink arrow at the back point, same forward axis (b shares q's orientation)
        back_arrow_top  = ax_top.quiver(back_data[0, 0], back_data[0, 1], fwd0[0], fwd0[1],
                                          color='pink', scale=1 / arrow_len, scale_units='xy',
                                          angles='xy', zorder=6)
        back_arrow_back = ax_back.quiver(back_data[0, 1], back_data[0, 2], fwd0[1], fwd0[2],
                                           color='pink', scale=1 / arrow_len, scale_units='xy',
                                           angles='xy', zorder=6)
        back_arrow_side = ax_side.quiver(back_data[0, 0], back_data[0, 2], fwd0[0], fwd0[2],
                                           color='pink', scale=1 / arrow_len, scale_units='xy',
                                           angles='xy', zorder=6)
        back_arrow_3d = [None]

        def set_front(i):
            rx, ry, rz = center_data[i, 0], center_data[i, 1], center_data[i, 2]
            fx, fy, fz = front_data[i, 0], front_data[i,1] , front_data[i,2]
            front_fwd  = _front_forward(i)

            link.set_data([rx, fx], [ry, fy])
            link.set_3d_properties([rz, fz])
            link_top.set_data([rx, fx], [ry, fy])
            link_back.set_data([ry, fy], [rz, fz])
            link_side.set_data([rx, fx], [rz, fz])

            front_arrow_top.set_offsets([[fx, fy]])
            front_arrow_top.set_UVC(front_fwd[0], front_fwd[1])
            front_arrow_back.set_offsets([[fy, fz]])
            front_arrow_back.set_UVC(front_fwd[1], front_fwd[2])
            front_arrow_side.set_offsets([[fx, fz]])
            front_arrow_side.set_UVC(front_fwd[0], front_fwd[2])
            if front_arrow_3d[0] is not None:
                front_arrow_3d[0].remove()
            front_arrow_3d[0] = ax.quiver(fx, fy, fz, front_fwd[0], front_fwd[1], front_fwd[2],
                                           length=arrow_len, color='darkorange', zorder=6)
            

        def set_back(i):
            rx, ry, rz = center_data[i, 0], center_data[i, 1], center_data[i, 2]
            bx, by, bz = back_data[i, 0], back_data[i,1] , back_data[i,2]
            back_fwd   = _back_forward(i)

            link_lower.set_data([bx, rx], [by, ry])
            link_lower.set_3d_properties([bz, rz])
            link_top_lower.set_data([bx, rx], [by, ry])
            link_back_lower.set_data([by, ry], [bz, rz])
            link_side_lower.set_data([bx, rx], [bz, rz])

            back_arrow_top.set_offsets([[bx, by]])
            back_arrow_top.set_UVC(back_fwd[0], back_fwd[1])
            back_arrow_back.set_offsets([[by, bz]])
            back_arrow_back.set_UVC(back_fwd[1], back_fwd[2])
            back_arrow_side.set_offsets([[bx, bz]])
            back_arrow_side.set_UVC(back_fwd[0], back_fwd[2])
            if back_arrow_3d[0] is not None:
                back_arrow_3d[0].remove()
            back_arrow_3d[0] = ax.quiver(bx, by, bz, back_fwd[0], back_fwd[1], back_fwd[2],
                                           length=arrow_len, color='pink', zorder=6)
        def set_neighbors(i):
            pts = np.array(neighbor_data[i]) if neighbor_data[i] else np.empty((0, 3))
            green_3d._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
            green_top.set_offsets(pts[:, [0, 1]])
            green_back.set_offsets(pts[:, [1, 2]])
            green_side.set_offsets(pts[:, [0, 2]])

        def set_qdot(i):
            x, y, z = center_data[i, 0], center_data[i, 1], center_data[i, 2]
            fwd = _center_forward(i)
            qdot_top.set_offsets([[x, y]])
            qdot_top.set_UVC(fwd[0], fwd[1])
            qdot_back.set_offsets([[y, z]])
            qdot_back.set_UVC(fwd[1], fwd[2])
            qdot_side.set_offsets([[x, z]])
            qdot_side.set_UVC(fwd[0], fwd[2])
            if qdot_3d[0] is not None:
                qdot_3d[0].remove()
            qdot_3d[0] = ax.quiver(x, y, z, fwd[0], fwd[1], fwd[2],
                                    length=arrow_len, color='red', zorder=6)

        def set_q_point(i):
            x, y, z = data[i, 1], data[i, 2], data[i, 3]
            q_point_3d._offsets3d = ([x], [y], [z])
            q_point_top.set_offsets([[x, y]])
            q_point_back.set_offsets([[y, z]])
            q_point_side.set_offsets([[x, z]])

        set_front(0)
        set_back(0)
        set_neighbors(0)
        set_qdot(0)
        set_q_point(0)

        ax_side.legend([link_side, link_side_lower, green_side, q_point_side],
                       ["center–front link", "center–back link", "grid neighbors", "q (center of mass)"],
                       loc="upper right", fontsize=8)

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
            set_neighbors(i)
            set_qdot(i)
            set_q_point(i)

            ax_top.set_title(f"t = {data[i, 0]:.2f} s")
            return (link, link_top, link_back, link_side,
                    front_arrow_top, front_arrow_back, front_arrow_side, front_arrow_3d[0],
                    link_lower, link_top_lower, link_back_lower, link_side_lower,
                    back_arrow_top, back_arrow_back, back_arrow_side, back_arrow_3d[0],
                    green_3d, green_top, green_back, green_side,
                    qdot_top, qdot_back, qdot_side, qdot_3d[0],
                    q_point_3d, q_point_top, q_point_back, q_point_side,)

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

            points, neighbor_points = self._track_update()
            self.total_distance  += np.linalg.norm(self.dt * self.q_dot[0:3])
            
            # loop termination check
            if self.total_distance >= self.stop_distance:
                break

            # log variables for plotting
            self.log.append([t, *self.q])
            self.center_log.append(np.array(points[1]))
            self.front_log.append(np.array(points[2]))
            self.back_log.append(np.array(points[3]))
            self.neighbor_log.append(neighbor_points[0] + neighbor_points[1] + neighbor_points[2] + neighbor_points[3])
    
    def _track_update(self): 
        if not self.is_initalization:
            self.q += self.dt * self.q_dot
        else:
            self.is_initalization = False
        neighbors_q, self.q[2] = self._point_height(self.q)
        self.q[3:6]            = self._point_orientation()

        half_width   = self._rotation_lg(*self.q[3:6]) @ np.array([          0, self.b/2, 0])
        half_track   = self._rotation_lg(*self.q[3:6]) @ np.array([self.l / 2,         0, 0])
        right_center = self.q + np.concatenate((half_width, np.zeros(3)))
        front        = right_center + np.concatenate((half_track, np.zeros(3)))
        back         = right_center - np.concatenate((half_track, np.zeros(3)))

        neighbors_right_center, _ = self._point_height(right_center)
        right_center, front, back, neighbors_front, neighbors_back = self._track_surface_contact(right_center, front, back, examine_front = True)

        right_center, front, back, neighbors_front, neighbors_back = self._track_surface_contact(right_center, front, back, examine_front = False)
        points          = [self.q, right_center, front, back]
        neighbor_points = [neighbors_q, neighbors_right_center, neighbors_front, neighbors_back]
        return points, neighbor_points

    def _track_surface_contact(self, center, front, back, examine_front):

        # consider tracking front and back points as class vairables
        neighbors_front, surface_height_front = self._point_height(front)
        neighbors_back, surface_height_back   = self._point_height(back)
        
        # protect against floating point error with 1e-15
        front_is_under_ground = (examine_front and (surface_height_front - front[2]) > 1e-15) 
        back_is_under_ground  = ((not examine_front) and (surface_height_back - back[2]) > 1e-15)
        if front_is_under_ground or back_is_under_ground:
            for _ in range(20):
                front[2]         = surface_height_front
                back[2]          = surface_height_back
                center[2]        = (front[2] + back[2])/2
                angles           = self._track_orientation(center, front)
                half_track = self._rotation_lg(*angles)[:, 0] * self.l / 2
                front            = np.concatenate((center[:3] + half_track, angles))
                back             = np.concatenate((center[:3] - half_track, angles))
                
                neighbors_front, surface_height_front = self._point_height(front)
                neighbors_back, surface_height_back   = self._point_height(back)
                
                if front_is_under_ground and (abs(surface_height_front - front[2]) < 1e-9):
                    break
                if back_is_under_ground and (abs(surface_height_back - back[2]) < 1e-9):
                    break

        if front_is_under_ground:
            center[3:6] = front[3:6]
        if  back_is_under_ground:
            center[3:6] = back[3:6]
        print(
            [f"{x:.3f}" for x in center],
            [f"{x:.3f}" for x in front],
            [f"{x:.3f}" for x in back],
        )
        return center, front, back, neighbors_front, neighbors_back

    def _point_height(self, point):
        neighbor_points   = self._get_neighbor_points(point)
        height_to_surface = self._bilinear_height(point, neighbor_points)
        return neighbor_points, height_to_surface

    def _get_neighbor_points(self, point):
        i = int(np.clip((point[0] - self.us[0]) // self.subdivision, 0, len(self.us) - 2))
        j = int(np.clip((point[1] - self.vs[0]) // self.subdivision, 0, len(self.vs) - 2))
        
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
            
    def _track_orientation(self, center, front):
        """
        Roll/pitch/yaw with the front point in contact: pitch is the climb of
        the front point over its horizontal offset from the center, measured
        along the heading so it works for any travel direction; roll cannot be
        recovered from a point on the forward axis, so it set to zero for the time being.
        """
        vel   = np.array(self.q_dot[:3])
        if self.is_backwards:
            vel *= -1
        speed = np.linalg.norm(vel[:2])
        yaw   = center[5] if speed < 1e-9 else np.arctan2(vel[1], vel[0])

        d     = np.asarray(front[:3]) - center[:3]
        horiz = d[0] * np.cos(yaw) + d[1] * np.sin(yaw)
        pitch = -np.arctan2(d[2], horiz)
        
        #TODO: when adding in a second track compute the roll for real 
        roll = 0

        return np.array([roll, pitch, yaw])

    def _point_orientation(self):
        """
        Roll/pitch/yaw of the center of mass crossing the current tile: pitch and
        roll come from the tile's height-field gradient (the edges' angles,
        blended the same s,t weights as _bilinear_height) read off along and
        across the direction of travel; yaw is the global-frame heading of
        q_dot, arctan2(vel_y, vel_x), so (roll, pitch, yaw) form a consistent
        ZYX Euler triple for _rotation_lg.
        """
        corners = self._get_neighbor_points(self.q)
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
    my_surface = Surface(is_uphill=False, is_surface_pitched=True, is_backwards=False)
    my_surface.run_and_plot()   