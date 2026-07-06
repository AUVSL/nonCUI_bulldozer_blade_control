import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
class surface:
    def __init__(self):
        self.surface_abg    = np.array([ 0, 0, 0])
        self.u_split        = 5  # u-value where the grid switches to surface_abg2
        self.offset         = np.array([0, 0, -1.0])
        self.b              = 1.75
        self.l              = 2.349  
        self.u_range        = (0, 10)
        self.v_range        = (-1, 1)
        self.q              = np.array([0.0, 0.0, 0.0, self.surface_abg[0], self.surface_abg[1], self.surface_abg[2]])
        self.q_dot          = np.array([5, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.stop_time      = 300.0
        self.dt             = 1/100
        self.stop_distance  = (self.u_range[1]-1) - (self.u_range[0])
        self.total_distance = 0.0
        self.log            = []

    @property
    def subdivision(self, division_factor: float = 1.0):
        """Grid spacing, sized relative to the dozer width self.b."""
        return self.b / division_factor

    def rotation_gl(self, a, B, g):
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
    
    def rotation_lg(self, a, B, g):
        """Rotation matrix: local → global frame"""
        return self.rotation_gl(a, B, g).T

    def surface_grid(self, spacing = None):
        spacing = self.subdivision
        R_surf  = self.rotation_lg(*self.surface_abg)
        e1, e2  = R_surf[:, 0], R_surf[:, 1]
        
        us = np.arange(self.u_range[0], self.u_range[1] + spacing / 2, spacing)
        vs = np.arange(self.v_range[0], self.v_range[1] + spacing / 2, spacing)

        G = nx.grid_2d_graph(len(us), len(vs))
        for i, u in enumerate(us):
            for j, v in enumerate(vs):
                if (self.u_split is not None and u >= self.u_split): 
                    x, y, z = (u + self.offset[0]) * e1 + (v + self.offset[1]) * e2 + self.offset[2]
                else:
                    x, y, z = u * e1 + v * e2
                node = G.nodes[(i, j)]
                node["x"], node["y"], node["z"] = float(x), float(y), float(z)
                node["visited_last"] = False

        return G

    def plot_surface(self, save_path="figures/surface_3d.png", show=False):
        """Render the surface grid as a 3D height-colored mesh."""
        spacing = self.subdivision
        G = self.surface_grid(spacing=spacing)

        us = np.arange(self.u_range[0], self.u_range[1] + spacing / 2, spacing)
        vs = np.arange(self.v_range[0], self.v_range[1] + spacing / 2, spacing)

        X = np.array([[G.nodes[(i, j)]["x"] for j in range(len(vs))] for i in range(len(us))])
        Y = np.array([[G.nodes[(i, j)]["y"] for j in range(len(vs))] for i in range(len(us))])
        Z = np.array([[G.nodes[(i, j)]["z"] for j in range(len(vs))] for i in range(len(us))])

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="hsv", edgecolor="black",
                                linewidth=1, alpha=0.8, antialiased=True)
        plt.xticks(np.arange(np.floor(X.min()), np.ceil(X.max()) + 1, 2), fontsize=10)
        plt.yticks(np.arange(np.floor(Y.min()), np.ceil(Y.max()) + 1, 1), fontsize=10)
        ax.set_zticks(np.arange(np.floor(Z.min()), np.ceil(Z.max()) + 1, 1))
        ax.set_box_aspect([np.ptp(X), np.ptp(Y), max(np.ptp(Z), 1e-6)])
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=120)
            print(f"Saved {save_path}")
        if show:
            plt.show()
        plt.close(fig)

    def run(self):
        t = 0.0
        for _ in range(int(self.stop_time / self.dt)):
            self.q   += self.dt * self.q_dot
            self.total_distance += np.linalg.norm(self.dt * self.q_dot[0:3])

            if self.total_distance >= self.stop_distance:
                break

            self.log.append([t, *self.q])
            t += self.dt

    def run_and_plot(self):
        self.run()

        print("Rendering GIF...")
        data     = np.array(self.log)[::5]

        fig = plt.figure(figsize=(14, 7))
        gs  = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], hspace=0.35, wspace=0.3)
        ax      = fig.add_subplot(gs[:, 0], projection='3d')

        # Surface plane normal
        a_s, B_s, g_s = self.surface_abg
        sa, ca = np.sin(a_s), np.cos(a_s)
        sB, cB = np.sin(B_s), np.cos(B_s)
        sg, cg = np.sin(g_s), np.cos(g_s)
        n_x = ca * sB * cg + sa * sg
        n_y = ca * sB * sg - sa * cg
        n_z = ca * cB

        margin =1.0
        cx   = (data[:, 1].max() + data[:, 1].min()) / 2
        cy   = (data[:, 2].max() + data[:, 2].min()) / 2
        cz   = (data[:, 3].max() + data[:, 3].min()) / 2
        half = max(data[:, 1].max() - data[:, 1].min(),
                   data[:, 2].max() - data[:, 2].min(),
                   data[:, 3].max() - data[:, 3].min()) / 2 + margin

        xs = np.linspace(cx - half, cx + half, 30)
        ys = np.linspace(cy - half, cy + half, 30)
        Xs, Ys = np.meshgrid(xs, ys)
        Zs = -(n_x * Xs + n_y * Ys) / n_z
        ax.plot_surface(Xs, Ys, Zs, color='none', edgecolor='none', zorder=0)

        surf_grid = self.surface_grid()
        grid_segments = [
            [(surf_grid.nodes[u]['x'], surf_grid.nodes[u]['y'], surf_grid.nodes[u]['z']),
             (surf_grid.nodes[v]['x'], surf_grid.nodes[v]['y'], surf_grid.nodes[v]['z'])]
            for u, v in surf_grid.edges()
        ]
        ax.add_collection3d(Line3DCollection(grid_segments, colors='saddlebrown', linewidths=0.5, alpha=0.5, zorder=0))

        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_zlim(cz - half, cz + half)
        ax.set_box_aspect([1, 1, 1])
        ax.grid(False)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        ax.scatter(data[0, 1], data[0, 2], data[0, 3], color='blue', s=60, zorder=5)
        trail,      = ax.plot([], [], [], 'b-', linewidth=1.5)

        def update(i):
            trail.set_data(data[:i+1, 1], data[:i+1, 2])
            trail.set_3d_properties(data[:i+1, 3])
            ax.set_title(f"t = {data[i, 0]:.2f} s")
            return trail,

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
    grid_graph = my_surface.surface_grid()

    my_surface.plot_surface()
    my_surface.run_and_plot()
    
   