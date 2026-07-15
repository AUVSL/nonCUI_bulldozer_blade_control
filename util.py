import os
from matplotlib.collections import LineCollection
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class surface:
    def __init__(self):
        # simulation parameters
        self.surface_abg      = np.array([ 0, 0.0, 0])
        self.b                = 1.75
        self.l                = 2.349
        self.dt               = 1/100  
        self.u_split          = 1  # u-value where the grid switches to surface_abg2
        self.transition_tiles = self.b  # tiles over which the offset ramps down past u_split
        self.offset           = np.array([0, 0, -2*self.b])
        self.stop_time        = 300.0
        self.total_distance   = 0.0
        self.front_contact    = False
        self.q                = np.array([0.0, 0, 0.0, self.surface_abg[0], self.surface_abg[1], 0.0])
        self.q_dot            = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.log              = []
        self.neighbor_points  = []
        self.neighbor_log     = []
        self.front_log        = []
        
        # set up the surface grid
        self.u_range         = (0, 2.5 * self.b) 
        self.v_range         = (-self.b/2, self.b/2)
        self.us              = np.arange(self.u_range[0], self.u_range[1] + self.subdivision, self.subdivision)
        self.vs              = np.arange(self.v_range[0], self.v_range[1] + self.subdivision, self.subdivision)
        self.stop_distance   = self.us[-1] - self.us[0]
        self.surf_grid       = self._surface_grid()

        # put particle on the surface at the start of the simulation
        points, neighbor_points = self._multi_particle_update()
        self.log.append([0, *self.q])
        self.front_log.append(np.array(points[1]))
        self.neighbor_log.append(neighbor_points[0] + neighbor_points[1])
        self.q_dot           = np.array([2, 0.0, 0.0, 0.0, 0.0, 0.0])

    @property
    def subdivision(self, division_factor: float = 2.0):
        """Grid spacing, sized relative to the dozer width self.b."""
        return self.b / division_factor

    def _surface_grid(self):
        R_surf  = self._rotation_lg(*self.surface_abg)
        e1, e2, e3 = R_surf[:, 0], R_surf[:, 1], R_surf[:, 2]

        G = nx.grid_2d_graph(len(self.us), len(self.vs))
        # snap the ramp to grid nodes (start at the last node <= u_split, span a whole
        # number of tiles) so every ramp segment has the same slope
        u_start    = self.us[self.us <= self.u_split][-1]
        ramp_width = max(1, round(self.transition_tiles)) * self.subdivision
        for i, u in enumerate(self.us):
            for j, v in enumerate(self.vs):
                # full offset for u <= u_start, then a linear ramp to zero over ramp_width
                w = np.clip((u - u_start) / ramp_width, 0.0, 1.0)
                x, y, z = u * e1 + v * e2 + (1 - w) * self.offset * e3
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
        
        # set up the figure and axes for the animation
        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot(1,3,3, projection='3d')
        ax_top  = fig.add_subplot(1,3,2)
        ax_side = fig.add_subplot(1,3,1)
        
        # set axis limits based on the logged data (and neighbor points, if any)
        margin = 1.0
        data          = np.array(self.log)[::2] # every 2nd frame represented to speed up rendering
        front_data    = np.array(self.front_log)[::2]
        neighbor_data = self.neighbor_log[::2]
        all_neighbor_pts = np.array([pt for frame in self.neighbor_log for pt in frame])

        all_x = np.concatenate([data[:, 1], front_data[:, 0], all_neighbor_pts[:, 0]])
        all_y = np.concatenate([data[:, 2], front_data[:, 1], all_neighbor_pts[:, 1]])
        all_z = np.concatenate([data[:, 3], front_data[:, 2], all_neighbor_pts[:, 2]])
        
        cx     = (all_x.max() + all_x.min()) / 2
        cy     = (all_y.max() + all_y.min()) / 2
        cz     = (all_z.max() + all_z.min()) / 2
        half   = max(all_x.max() - all_x.min(),
                     all_y.max() - all_y.min(),
                     all_z.max() - all_z.min()) / 2 + margin
        
        ax.set_xlim( cx - half, cx + half)
        ax.set_ylim( cy - half, cy + half)
        ax.set_zlim( cz - half, cz + half)
        ax_top.set_xlim(cx - half, cx + half)
        ax_top.set_ylim(cy - half, cy + half)
        ax_side.set_xlim(cx - half, cx + half)
        ax_side.set_ylim(cz - half, cz + half)        

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
        
        ax.add_collection3d(Line3DCollection(grid_segments, colors='saddlebrown', linewidths=0.5, alpha=0.5, zorder=0))
        ax_top.add_collection(LineCollection(grid_segments1, colors="black", linewidths=0.5, alpha=0.5, zorder=0))
        ax_side.add_collection(LineCollection(grid_segments2, colors="black", linewidths=0.5, alpha=0.5, zorder=0))

        # remaining plot settings
        ax.set_box_aspect([1, 1, 1])
        ax.grid(False)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)") 
        
        ax_top.set_aspect('equal')
        ax_top.grid(False)
        ax_top.set_xlabel("X (m)")
        ax_top.set_ylabel("Y (m)")
        
        ax_side.set_aspect('equal')
        ax_side.grid(False)
        ax_side.set_xlabel("X (m)")
        ax_side.set_ylabel("Z (m)")

        # orange line connecting the current q and f positions, updated each frame
        link,      = ax.plot([], [], [], color='darkorange', linewidth=1.5)
        link_top,  = ax_top.plot([], [], color='darkorange', linewidth=1.5)
        link_side, = ax_side.plot([], [], color='darkorange', linewidth=1.5)

        # green scatter artists for grid vertices within one tile length of the point, updated each frame
        green_3d   = ax.scatter([], [], [], color='green', s=40, zorder=5)
        green_top  = ax_top.scatter([], [], color='green', s=40, zorder=5)
        green_side = ax_side.scatter([], [], color='green', s=40, zorder=5)

        # red arrow at the tracked point showing the particle's orientation,
        # i.e. the local forward axis (R_lg(*q[3:6])[:, 0]) for that frame's roll/pitch/yaw
        arrow_len = self.subdivision * 0.6

        def _forward(i):
            return self._rotation_lg(*data[i, 4:7])[:, 0]
        
        def _front_forward(i):
            return self._rotation_lg(*front_data[i, 3:])[:, 0]

        fwd0 = _forward(0)
        qdot_top  = ax_top.quiver(data[0, 1], data[0, 2], fwd0[0], fwd0[1],
                                   color='red', scale=1 / arrow_len, scale_units='xy',
                                   angles='xy', zorder=6)
        qdot_side = ax_side.quiver(data[0, 1], data[0, 3], fwd0[0], fwd0[2],
                                    color='red', scale=1 / arrow_len, scale_units='xy',
                                    angles='xy', zorder=6)
        qdot_3d = [None]  # mplot3d quiver has no in-place update, so remove/recreate each frame

        # orange arrow at the front point, same forward axis (f shares q's orientation)
        front_arrow_top  = ax_top.quiver(front_data[0, 0], front_data[0, 1], fwd0[0], fwd0[1],
                                          color='darkorange', scale=1 / arrow_len, scale_units='xy',
                                          angles='xy', zorder=6)
        front_arrow_side = ax_side.quiver(front_data[0, 0], front_data[0, 2], fwd0[0], fwd0[2],
                                           color='darkorange', scale=1 / arrow_len, scale_units='xy',
                                           angles='xy', zorder=6)
        front_arrow_3d = [None]

        def set_front(i):
            qx, qy, qz = data[i, 1], data[i, 2], data[i, 3]
            fx, fy, fz = front_data[i, 0], front_data[i,1] , front_data[i,2]
            front_fwd = _front_forward(i)

            link.set_data([qx, fx], [qy, fy])
            link.set_3d_properties([qz, fz])
            link_top.set_data([qx, fx], [qy, fy])
            link_side.set_data([qx, fx], [qz, fz])

            front_arrow_top.set_offsets([[fx, fy]])
            front_arrow_top.set_UVC(front_fwd[0], front_fwd[1])
            front_arrow_side.set_offsets([[fx, fz]])
            front_arrow_side.set_UVC(front_fwd[0], front_fwd[2])
            if front_arrow_3d[0] is not None:
                front_arrow_3d[0].remove()
            front_arrow_3d[0] = ax.quiver(fx, fy, fz, front_fwd[0], front_fwd[1], front_fwd[2],
                                           length=arrow_len, color='darkorange', zorder=6)

        def set_neighbors(i):
            pts = np.array(neighbor_data[i]) if neighbor_data[i] else np.empty((0, 3))
            green_3d._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
            green_top.set_offsets(pts[:, [0, 1]])
            green_side.set_offsets(pts[:, [0, 2]])

        def set_qdot(i):
            x, y, z = data[i, 1], data[i, 2], data[i, 3]
            fwd = _forward(i)
            qdot_top.set_offsets([[x, y]])
            qdot_top.set_UVC(fwd[0], fwd[1])
            qdot_side.set_offsets([[x, z]])
            qdot_side.set_UVC(fwd[0], fwd[2])
            if qdot_3d[0] is not None:
                qdot_3d[0].remove()
            qdot_3d[0] = ax.quiver(x, y, z, fwd[0], fwd[1], fwd[2],
                                    length=arrow_len, color='red', zorder=6)

        set_front(0)
        set_neighbors(0)
        set_qdot(0)

        ax_top.legend([link_top, green_top],
                      ["q–f link", "grid neighbors"],
                      loc="upper right", fontsize=8)

        # animation update function
        def update(i):
            set_front(i)
            set_neighbors(i)
            set_qdot(i)

            ax_top.set_title(f"t = {data[i, 0]:.2f} s")
            return (link, link_top, link_side,
                    front_arrow_top, front_arrow_side, front_arrow_3d[0],
                    green_3d, green_top, green_side, qdot_top, qdot_side, qdot_3d[0],)

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

            points, neighbor_points = self._multi_particle_update()
            self.total_distance  += np.linalg.norm(self.dt * self.q_dot[0:3])

            # loop termination check
            if self.total_distance >= self.stop_distance:
                break

            # log variables for plotting
            self.log.append([t, *self.q])
            self.front_log.append(np.array(points[1]))
            self.neighbor_log.append(neighbor_points[0] + neighbor_points[1])
    
    def _multi_particle_update(self):
        self.q += self.dt * self.q_dot
        neighbor_points_q, self.q[2] = self._particle_height(self.q)
        
        forward_position = self._rotation_lg(*self.q[3:6])[:, 0] * self.l / 2
        f  = self.q + np.concatenate((forward_position, np.zeros(3)))

        neighbor_points_f, surface_height_f = self._particle_height(f)

        if surface_height_f - f[2] > 0:
            for _ in range(20):
                # 20 iterations gets us to about 1e-6 error in the height so length conservation is okay
                f[2]             = surface_height_f
                angles           = self._multi_particle_contact_orientation(f)
                forward_position = self._rotation_lg(*angles)[:, 0] * self.l / 2
                f                = np.concatenate((self.q[:3] + forward_position, angles))
                neighbor_points_f, surface_height_f = self._particle_height(f)
                if abs(surface_height_f - f[2]) < 1e-9:
                    break
            f[2]        = surface_height_f
            self.q[3:6] = f[3:6]
        else:
            self.q[3:6] = self._particle_orientation()
        
        points          = [self.q, f]
        neighbor_points = [neighbor_points_q, neighbor_points_f]
        
        return points, neighbor_points
    
    def _particle_height(self, point):
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
            
    def _multi_particle_contact_orientation(self, point):
        """
        Roll/pitch/yaw of the surface-snapped front point: pitch is the
        elevation of the line from the center of mass to the front point
        (measured against the horizontal distance along the heading), roll
        comes from the lateral slope of the tile under the front point, and
        yaw is the global-frame heading of q_dot — the same conventions as
        _particle_orientation.
        """
        vel   = np.array(self.q_dot[:3])
        # speed = np.linalg.norm(vel[:2])
        # if speed < 1e-9:
        #     return self.q[3:6].copy()

        # fwd_xy  = vel[:2] / speed
        # left_xy = np.array([-fwd_xy[1], fwd_xy[0]])

        # corners = self._get_neighbor_points(point)
        # grad_xy = self._bilinear_gradient(point, corners)
        # s_l     = np.dot(grad_xy, left_xy)  # lateral slope
        # s_f     = np.dot(grad_xy, fwd_xy)   # forward slope

        # run   = np.hypot(point[0] - self.q[0], point[1] - self.q[1])
        # roll  = np.arctan2(s_l, np.sqrt(1.0 + s_f**2))
        # pitch = -np.arctan2(point[2] - self.q[2], run)
        
        roll  = np.arctan2(point[2] - self.q[2], point[1] - self.q[1])        
        pitch = -np.arctan2(point[2] - self.q[2], point[0] - self.q[0])
        yaw   = np.arctan2(vel[1], vel[0])

        return np.array([roll, pitch, yaw])

    def _particle_orientation(self):
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
    my_surface = surface()
    my_surface.run_and_plot()   