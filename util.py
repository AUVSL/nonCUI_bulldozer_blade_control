import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
class surface:
    def __init__(self):
        self.surface_abg    = np.array([ 0, 0, 0])
        self.u_split        = 5  # u-value where the grid switches to surface_abg2
        self.offset         = np.array([0, 0, -1.0])
        self.b              = 1.75
        self.l              = 2.349  
        self.u_range        = (-1, 10)
        self.v_range        = (-1, 2)
        self.q              = np.array([0.0, 0.0, 0.0, self.surface_abg[0], self.surface_abg[1], self.surface_abg[2]])
        self.q_dot          = np.array([1, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.stop_time      = 1.0
        self.dt             = 1/100
        self.stop_distance  = self.u_range[1] - self.u_range[0]
        self.total_distance = 0.0

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

if __name__ == "__main__":
    # Example usage
    my_surface = surface()
    grid_graph = my_surface.surface_grid()

    my_surface.plot_surface()