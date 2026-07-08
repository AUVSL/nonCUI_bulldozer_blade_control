import os
from matplotlib.collections import LineCollection
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class surface:
    def __init__(self):
        self.surface_abg    = np.array([ 0, 0.1, 0])
        self.u_split        = 5  # u-value where the grid switches to surface_abg2
        self.offset         = np.array([0, 0, 0.1])
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
    
    def _rotation_lg(self, a, B, g):
        """Rotation matrix: local → global frame"""
        return self._rotation_gl(a, B, g).T

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

    def _run(self):
        t = 0.0
        for _ in range(int(self.stop_time / self.dt)):
            self.q   += self.dt * self.q_dot
            self.total_distance += np.linalg.norm(self.dt * self.q_dot[0:3])
            neighbor_points = self._get_neighbor_points(self.q)
            height_to_surface = self._bilinear_height(self.q, neighbor_points)
            self.q[2] = height_to_surface

            if self.total_distance >= self.stop_distance:
                break

            self.log.append([t, *self.q])
            
            self.neighbor_log.append(neighbor_points)
            height_to_surface = self._bilinear_height(self.q, neighbor_points)
            self.height_log.append(height_to_surface)
            t += self.dt

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

    def _get_interpolated_height(self, point):
        corners = self._get_neighbor_points(point)
        return self._bilinear_height(point, corners)

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

        set_neighbors(0)
        set_drop_line(0)

        # animation update function
        def update(i):
            trail.set_data(data[:i+1, 1], data[:i+1, 2])
            trail.set_3d_properties(data[:i+1, 3])
            trail_top.set_data(data[:i+1, 1], data[:i+1, 2])
            trail_side.set_data(data[:i+1, 1], data[:i+1, 3])
            set_neighbors(i)
            set_drop_line(i)

            ax_top.set_title(f"t = {data[i, 0]:.2f} s")
            return trail, trail_top, trail_side, green_3d, green_top, green_side, drop_3d, drop_side,

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
    
   