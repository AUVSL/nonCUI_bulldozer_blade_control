import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
class surface:
    def __init__(self):
        self.surface_abg   = np.array([ 0, 0, 0])
        self.surface_abg2  = np.array([ -0.2, 0, 0])  # optional roll angle applied beyond v_split
        self.v_split       = 5  # v-value where the grid switches to surface_abg2
        self.b             = 1.75
        self.l             = 2.349  
        self.u_range       = (0, 10)
        self.v_range       = (0, 10)
        self.q             = np.array([0.0, 0.0, 0.0, self.surface_abg[0], self.surface_abg[1], self.surface_abg[2]])
        self.q_dot         = np.zeros(6)

    @property
    def subdivision(self):
        """Grid spacing, sized relative to the dozer width self.b."""
        return self.b / 4

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

    def surface_grid(self, u_range=None, v_range=None, spacing=None, v_split=None, surface_abg2=None):
        u_range = self.u_range if u_range is None else u_range
        v_range = self.v_range if v_range is None else v_range
        spacing = self.subdivision if spacing is None else spacing
        R_surf = self.rotation_lg(*self.surface_abg)
        e1, e2 = R_surf[:, 0], R_surf[:, 1]

        v_split = self.v_split if v_split is None else v_split
        if v_split is not None:
            abg2 = self.surface_abg2 if surface_abg2 is None else surface_abg2
            R_surf2 = self.rotation_lg(*abg2)
            e1_2, e2_2 = R_surf2[:, 0], R_surf2[:, 1]

        us = np.arange(u_range[0], u_range[1] + spacing / 2, spacing)
        vs = np.arange(v_range[0], v_range[1] + spacing / 2, spacing)

        G = nx.grid_2d_graph(len(us), len(vs))
        for i, u in enumerate(us):
            for j, v in enumerate(vs):
                u_e1, u_e2 = (e1_2, e2_2) if (v_split is not None and v >= v_split) else (e1, e2)
                x, y, z = u * u_e1 + v * u_e2
                node = G.nodes[(i, j)]
                node["x"], node["y"], node["z"] = float(x), float(y), float(z)
                node["visited_last"] = False

        return G

    def plot_surface(self, u_range=None, v_range=None, spacing=None, v_split=None, surface_abg2=None,
                      save_path="figures/surface_3d.png", show=False):
        """Render the surface grid as a 3D height-colored mesh."""
        u_range = self.u_range if u_range is None else u_range
        v_range = self.v_range if v_range is None else v_range
        spacing = self.subdivision if spacing is None else spacing
        G = self.surface_grid(u_range, v_range, spacing, v_split=v_split, surface_abg2=surface_abg2)

        us = np.arange(u_range[0], u_range[1] + spacing / 2, spacing)
        vs = np.arange(v_range[0], v_range[1] + spacing / 2, spacing)

        X = np.array([[G.nodes[(i, j)]["x"] for j in range(len(vs))] for i in range(len(us))])
        Y = np.array([[G.nodes[(i, j)]["y"] for j in range(len(vs))] for i in range(len(us))])
        Z = np.array([[G.nodes[(i, j)]["z"] for j in range(len(vs))] for i in range(len(us))])

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="saddlebrown",
                                linewidth=0.3, alpha=0.9, antialiased=True)
        fig.colorbar(surf, ax=ax, shrink=0.6, label="Z (m)")

        ax.set_box_aspect([np.ptp(X), np.ptp(Y), max(np.ptp(Z), 1e-6)])
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Surface Grid")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=120)
            print(f"Saved {save_path}")
        if show:
            plt.show()
        plt.close(fig)

        return fig, ax

if __name__ == "__main__":
    # Example usage
    my_surface = surface()
    grid_graph = my_surface.surface_grid()

    my_surface.plot_surface()