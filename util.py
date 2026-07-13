import os
from matplotlib.collections import LineCollection
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class surface:
    def __init__(self):
        self.surface_abg    = np.array([ 0, 0.2, 0])
        self.u_split        = 5  # u-value where the grid switches to surface_abg2
        self.offset         = np.array([0, 0, 1])
        self.b              = 1.75
        self.l              = 2.349  
        self.u_range        = (0, 5 * self.b) 
        self.v_range        = (-self.b, self.b)
        spacing             = self.subdivision
        self.us             = np.arange(self.u_range[0], self.u_range[1] + spacing, spacing)
        self.vs             = np.arange(self.v_range[0], self.v_range[1] + spacing, spacing)
        self.q              = np.array([0.0, 0, 0.0, self.surface_abg[0], self.surface_abg[1], self.surface_abg[2]])
        self.q_dot          = np.array([5, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.stop_time      = 300.0
        self.dt             = 1/100
        self.stop_distance  = self.us[-1] - self.us[0]
        self.total_distance = 0.0
        self.log            = []
        self.neighbor_points = []
        self.neighbor_log   = []
        self.height_log     = []

    @property
    def subdivision(self, division_factor: float = 2.0):
        """Grid spacing, sized relative to the dozer width self.b."""
        return self.b / division_factor

    def _run(self):
        t = 0.0
        for _ in range(int(self.stop_time / self.dt)):
            self.q, neighbor_points, height_to_surface = self._particle_update(self.q)
            
            self.total_distance  += np.linalg.norm(self.dt * self.q_dot[0:3])
            if self.total_distance >= self.stop_distance:
                break

            self.log.append([t, *self.q])
            
            self.neighbor_log.append(neighbor_points)
            height_to_surface = self._bilinear_height(self.q, neighbor_points)
            self.height_log.append(height_to_surface)
            t += self.dt

    def _particle_update(self, point):
        point            += self.dt * self.q_dot
        point[3:6]        = self._particle_orientation(point, self.q_dot)
        neighbor_points   = self._get_neighbor_points(point)
        height_to_surface = self._bilinear_height(point, neighbor_points)
        point[2]          = height_to_surface
        return point, neighbor_points, height_to_surface

    def _particle_orientation(self, point, q_dot):
        """
        Roll/pitch/yaw of a particle crossing the current tile: pitch and
        roll come from the tile's height-field gradient (the edges' angles,
        blended the same s,t weights as _bilinear_height) read off along and
        across the direction of travel; yaw is the global-frame heading of
        q_dot, arctan2(vel_y, vel_x), so (roll, pitch, yaw) form a consistent
        ZYX Euler triple for _rotation_lg.
        """
        corners = self._get_neighbor_points(point)
        grad_xy = self._bilinear_gradient(point, corners)

        vel   = np.array(q_dot[:3])
        speed = np.linalg.norm(vel[:2])
        if speed < 1e-9:
            return self.q[3:6].copy()

        # the velocity in the the local body frame
        fwd_xy  = vel[:2] / speed
        left_xy = np.array([-fwd_xy[1], fwd_xy[0]])
        
        s_f = np.dot(grad_xy, fwd_xy)   # forward slope
        s_l = np.dot(grad_xy, left_xy)  # lateral slope
         
        pitch = np.arctan2(-s_f, 1.0)                 # exact as-is
        roll  = np.arctan2( s_l, np.sqrt(1.0 + s_f**2))  # generalized

        yaw = np.arctan2(vel[1], vel[0])

        return np.array([roll, pitch, yaw])
    
    def _get_neighbor_points(self, point):
        xyz       = np.array(point[:3])
        surf_grid = self.surface_grid()
        R_surf    = self._rotation_lg(*self.surface_abg)
        e1, e2    = R_surf[:, 0], R_surf[:, 1]

   
        # 1 x 3 times 3 x 1 with ones inserted via numpy e* ^2 [x,y,z]^T is just a row of R_gl * global x,y,z coordinates (u,v are the [local] flattened version of the global coordinates)
        u = xyz @ e1
        v = xyz @ e2
        spacing = self.subdivision

        i = int(np.clip((u - self.us[0]) // spacing, 0, len(self.us) - 2))
        j = int(np.clip((v - self.vs[0]) // spacing, 0, len(self.vs) - 2))

        corners = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
        return [(surf_grid.nodes[n]['x'], surf_grid.nodes[n]['y'], surf_grid.nodes[n]['z']) for n in corners]
    
    def surface_grid(self):
        R_surf  = self._rotation_lg(*self.surface_abg)
        e1, e2, e3 = R_surf[:, 0], R_surf[:, 1], R_surf[:, 2]

        G = nx.grid_2d_graph(len(self.us), len(self.vs))
        for i, u in enumerate(self.us):
            for j, v in enumerate(self.vs):
                if (u >= self.u_split): 
                    x, y, z = u * e1 + v* e2 + self.offset * e3
                else:
                    x, y, z = u * e1 + v * e2
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
        neighbor_data = self.neighbor_log[::2]
        height_data   = self.height_log[::2]
        all_neighbor_pts = np.array([pt for frame in self.neighbor_log for pt in frame])
        if all_neighbor_pts.size:
            all_x = np.concatenate([data[:, 1], all_neighbor_pts[:, 0]])
            all_y = np.concatenate([data[:, 2], all_neighbor_pts[:, 1]])
            all_z = np.concatenate([data[:, 3], all_neighbor_pts[:, 2]])
        else:
            all_x, all_y, all_z = data[:, 1], data[:, 2], data[:, 3]
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
        surf_grid = self.surface_grid()
        grid_segments = [
            [(surf_grid.nodes[u]['x'], surf_grid.nodes[u]['y'], surf_grid.nodes[u]['z']),
             (surf_grid.nodes[v]['x'], surf_grid.nodes[v]['y'], surf_grid.nodes[v]['z'])]
            for u, v in surf_grid.edges()
        ]
        grid_segments1 = [
            [(surf_grid.nodes[u]['x'], surf_grid.nodes[u]['y']),
             (surf_grid.nodes[v]['x'], surf_grid.nodes[v]['y'])]
            for u, v in surf_grid.edges()
        ]
        grid_segments2 = [
            [(surf_grid.nodes[u]['x'], surf_grid.nodes[u]['z']),
             (surf_grid.nodes[v]['x'], surf_grid.nodes[v]['z'])]
            for u, v in surf_grid.edges()
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

        ax.scatter(data[0, 1], data[0, 2], data[0, 3], color='blue', s=60, zorder=5)
        ax_top.scatter(data[0, 1],  data[0, 2], color='blue', s=60, zorder=5)
        ax_side.scatter(data[0, 1],  data[0, 3], color='blue', s=60, zorder=5)

        trail,      = ax.plot([], [], [], 'b-', linewidth=1.5)
        trail_top,  = ax_top.plot([], [], 'b-', linewidth=1.5)
        trail_side, = ax_side.plot([], [], 'b-', linewidth=1.5)

        # green scatter artists for grid vertices within one tile length of the point, updated each frame
        green_3d   = ax.scatter([], [], [], color='green', s=40, zorder=5)
        green_top  = ax_top.scatter([], [], color='green', s=40, zorder=5)
        green_side = ax_side.scatter([], [], color='green', s=40, zorder=5)

        # dotted line from the position point down to the bilinearly-interpolated surface height
        drop_3d,   = ax.plot([], [], [], 'k:', linewidth=1.2, zorder=4)
        drop_side, = ax_side.plot([], [], 'k:', linewidth=1.2, zorder=4)

        # red arrow at the tracked point showing the particle's orientation,
        # i.e. the local forward axis (R_lg(*q[3:6])[:, 0]) for that frame's roll/pitch/yaw
        arrow_len = self.subdivision * 0.6

        def _forward(i):
            return self._rotation_lg(*data[i, 4:7])[:, 0]

        fwd0 = _forward(0)
        qdot_top  = ax_top.quiver(data[0, 1], data[0, 2], fwd0[0], fwd0[1],
                                   color='red', scale=1 / arrow_len, scale_units='xy',
                                   angles='xy', zorder=6)
        qdot_side = ax_side.quiver(data[0, 1], data[0, 3], fwd0[0], fwd0[2],
                                    color='red', scale=1 / arrow_len, scale_units='xy',
                                    angles='xy', zorder=6)
        qdot_3d = [None]  # mplot3d quiver has no in-place update, so remove/recreate each frame

        def set_neighbors(i):
            pts = np.array(neighbor_data[i]) if neighbor_data[i] else np.empty((0, 3))
            green_3d._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
            green_top.set_offsets(pts[:, [0, 1]])
            green_side.set_offsets(pts[:, [0, 2]])

        def set_drop_line(i):
            x, y, z = data[i, 1], data[i, 2], data[i, 3]
            h = height_data[i]
            drop_3d.set_data([x, x], [y, y])
            drop_3d.set_3d_properties([z, h])
            drop_side.set_data([x, x], [z, h])

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

        set_neighbors(0)
        set_drop_line(0)
        set_qdot(0)

        # animation update function
        def update(i):
            trail.set_data(data[:i+1, 1], data[:i+1, 2])
            trail.set_3d_properties(data[:i+1, 3])
            trail_top.set_data(data[:i+1, 1], data[:i+1, 2])
            trail_side.set_data(data[:i+1, 1], data[:i+1, 3])
            set_neighbors(i)
            set_drop_line(i)
            set_qdot(i)

            ax_top.set_title(f"t = {data[i, 0]:.2f} s")
            return trail, trail_top, trail_side, green_3d, green_top, green_side, drop_3d, drop_side, qdot_top, qdot_side, qdot_3d[0],

        anim = animation.FuncAnimation(
            fig, update, frames=len(data), blit=False, interval=50
        )
        fname = "figures/simulation.gif"
        anim.save(fname, writer=animation.PillowWriter(fps=20))
        plt.close(fig)
        print(f"Saved {fname}")

if __name__ == "__main__":
    # Example usage
    my_surface = surface()
    my_surface.run_and_plot()
    
   