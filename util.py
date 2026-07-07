import os
from matplotlib.collections import LineCollection
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class surface:
    def __init__(self):
        self.surface_abg    = np.array([ 0, 0, 0])
        self.u_split        = 5  # u-value where the grid switches to surface_abg2
        self.offset         = np.array([0, 0, -1.0])
        self.b              = 1.75
        self.l              = 2.349  
        self.u_range        = (0, 5 * self.b) 
        self.v_range        = (-self.b, self.b)
        spacing             = self.subdivision
        self.us             = np.arange(self.u_range[0], self.u_range[1] + spacing, spacing)
        self.vs             = np.arange(self.v_range[0], self.v_range[1] + spacing, spacing)
        self.q              = np.array([0.0, 0.0, 0.0, self.surface_abg[0], self.surface_abg[1], self.surface_abg[2]])
        self.q_dot          = np.array([5, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.stop_time      = 300.0
        self.dt             = 1/100
        self.stop_distance  = self.us[-1] - self.us[0]
        self.total_distance = 0.0
        self.log            = []

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

            if self.total_distance >= self.stop_distance:
                break

            self.log.append([t, *self.q])
            t += self.dt

    def run_and_plot(self):
        self._run()

        print("Rendering GIF...")
        
        # set up the figure and axes for the animation
        fig , (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # set axis limits based on the logged data
        margin = 1.0
        data   = np.array(self.log)[::5] # every 5th frame repesented to speed up rendering
        cx     = (data[:, 1].max() + data[:, 1].min()) / 2
        cy     = (data[:, 2].max() + data[:, 2].min()) / 2
        cz     = (data[:, 3].max() + data[:, 3].min()) / 2
        half   = max(data[:, 1].max() - data[:, 1].min(),
                     data[:, 2].max() - data[:, 2].min(),
                      data[:, 3].max() - data[:, 3].min()) / 2 + margin
        ax1.set_xlim(cx - half, cx + half)
        ax1.set_ylim(cy - half, cy + half)
        ax2.set_xlim(cx - half, cx + half)
        ax2.set_ylim(cz - half, cz + half)

        # draw the surface grid
        surf_grid = self.surface_grid()
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
        
        ax1.add_collection(LineCollection(grid_segments1, colors="black", linewidths=0.5, alpha=0.5, zorder=0))
        ax2.add_collection(LineCollection(grid_segments2, colors="black", linewidths=0.5, alpha=0.5, zorder=0))

        # remaining plot settings
        ax1.set_aspect('equal')
        ax1.grid(False)
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax2.set_aspect('equal')
        ax2.grid(False)
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Z (m)")

        ax1.scatter(data[0, 1],  data[0, 2], color='blue', s=60, zorder=5)
        trail1,      = ax1.plot([], [], 'b-', linewidth=1.5)
        
        ax2.scatter(data[0, 1],  data[0, 3], color='blue', s=60, zorder=5)
        trail2,      = ax2.plot([], [], 'b-', linewidth=1.5)

        # animation update function
        def update(i):
            trail1.set_data(data[:i+1, 1], data[:i+1, 2])
            ax1.set_title(f"t = {data[i, 0]:.2f} s")
            trail2.set_data(data[:i+1, 1], data[:i+1, 3])
            ax2.set_title(f"t = {data[i, 0]:.2f} s")
            return trail1, trail2,

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
    
   