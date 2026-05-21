"""
Python port of the 3D bulldozer blade control simulation.
Replaces the Simulink sim('simulation_3d') call with an explicit
forward-Euler integration loop.
"""
from xml.parsers.expat import errors

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
matplotlib.use("Agg")   # headless; remove if running interactively

class BulldozerSimulation:
    def __init__(self):
        # ───────────────── Parameters ───── ────────────
        #TODO: add comments with parameter descriptions and units (maybe change names to be more descriptive?)
        self.stop_time   = 3
        self.surface_abg = np.array([ 0, 0, 0])
        self.h           = 2.762 / 2   # scaled down by Sam
        self.l           = 2.349 /1.5 # scaled down by Sam
        self.b           = 1.75  /1.5 # scaled down by Sam
        self.dt          = 0.01
        velocity_limit   = 2.222
        turn_vel_limit   = 2 * velocity_limit / self.b
        self.dxyz        = np.zeros(3)
        self.daBg        = np.zeros(3)
        self.q = np.array([
            0.0, 0.0, 0.0,
            self.surface_abg[0],
            self.surface_abg[1],
            self.surface_abg[2]
        ])
        self.q_dot   = np.zeros(6)
        self.v       = np.array([velocity_limit, turn_vel_limit])
        self.x_ICR   = 0.0
        self.R_lg    = self.rotation_lg(self.q[3], self.q[4], self.q[5])
        _, self.J_lg = self.rotation_derivatives(self.q[3], self.q[4])
        self.log     = []

    @staticmethod
    def wrap_angles(angles):
        return (angles + np.pi) % (2 * np.pi) - np.pi

    # ───────────────── Kinematics ─────────────────
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

    def rotation_derivatives(self, a, B):
        """Rotation derivative matrices"""
        # change to accept input array
        sa, ca = np.sin(a), np.cos(a)
        sB, cB, tB = np.sin(B), np.cos(B), np.tan(B)

        J_gl = np.array([
            [1,   0,     -sB],
            [0,  ca, sa * cB],
            [0, -sa, ca * cB]
        ])

        J_lg = np.array([
            [1, sa * tB, ca * tB],
            [0,      ca,    - sa],
            [0, sa / cB,  ca / cB]
        ])

        return J_gl, J_lg

    def rotation_lg(self, a, B, g):
        """Rotation matrix: local → global frame"""
        return self.rotation_gl(a, B, g).T

    def S_matrix(self):
        """
        Configuration-dependent velocity mapping matrix.
        Recomputed every time step.

        Maps v = [v_forward, v_turn] to q_dot.
        """
        S = np.zeros((6, 2))
        S[0:3, 0] = self.R_lg[:, 0]                 # forward velocity
        S[0:3, 1] = self.R_lg[:, 1] * (self.x_ICR) # lateral/turning velocity
        S[3:6, 1] = self.J_lg[:, 2]                 # yaw contribution

        return S

    def get_x_icr(self, eps: float = 1e-3):
        if abs(self.daBg[2]) < eps:
            return 0.0
        return float(np.clip(self.dxyz[1] / self.daBg[2], -self.l / 2, self.l / 2))

    # ───────────────── Main Integration Loop ─────────────────
    def run(self):
        t = 0.0
        for _ in range(int(self.stop_time / self.dt)):
            self.q_dot = self.S_matrix() @ self.v

            # update global/local positions and orientations for next time step
            self.q   += self.dt * self.q_dot
            self.q[3:6] = self.wrap_angles(self.q[3:6])
            a, B, g         = self.q[3:6]
            self.R_lg       = self.rotation_lg(a, B, g)
            R_gl            = self.rotation_gl(a, B, g)
            J_gl, self.J_lg = self.rotation_derivatives(a, B)


            self.dxyz = R_gl @ self.q_dot[0:3]
            self.daBg = J_gl @ self.q_dot[3:6]

            self.x_ICR     = self.get_x_icr()

            self.log.append([t, *self.q, self.x_ICR, self.dxyz[1], self.daBg[2]])
            t += self.dt

    def run_and_plot(self, stop_time=30.0, first_frame_only=False):
        """Run the simulation and render a single GIF with the 3D trajectory
        and x_ICR / lateral-velocity / yaw-rate time series animated together."""
        sim = BulldozerSimulation()
        sim.stop_time = stop_time
        sim.run()

        print("Rendering GIF...")
        data     = np.array(sim.log)[::5]
        n_frames = 1 if first_frame_only else len(data)

        fig = plt.figure(figsize=(16, 9))
        gs  = fig.add_gridspec(3, 2, width_ratios=[1.4, 1], hspace=0.45, wspace=0.35)
        ax      = fig.add_subplot(gs[:, 0], projection='3d')
        ax_icr  = fig.add_subplot(gs[0, 1])
        ax_dy   = fig.add_subplot(gs[1, 1])
        ax_dyaw = fig.add_subplot(gs[2, 1])

        # ── Surface plane ──
        a_s, B_s, g_s = sim.surface_abg
        sa, ca = np.sin(a_s), np.cos(a_s)
        sB, cB = np.sin(B_s), np.cos(B_s)
        sg, cg = np.sin(g_s), np.cos(g_s)
        nx = ca * sB * cg + sa * sg
        ny = ca * sB * sg - sa * cg
        nz = ca * cB

        margin = 2.0
        cx   = (data[:, 1].max() + data[:, 1].min()) / 2
        cy   = (data[:, 2].max() + data[:, 2].min()) / 2
        cz   = (data[:, 3].max() + data[:, 3].min()) / 2
        half = max(data[:, 1].max() - data[:, 1].min(),
                   data[:, 2].max() - data[:, 2].min(),
                   data[:, 3].max() - data[:, 3].min()) / 2 + margin

        xs = np.linspace(cx - half, cx + half, 30)
        ys = np.linspace(cy - half, cy + half, 30)
        Xs, Ys = np.meshgrid(xs, ys)
        Zs = -(nx * Xs + ny * Ys) / nz
        ax.plot_surface(Xs, Ys, Zs, alpha=0.3, color='tan', zorder=0)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_zlim(cz - half, cz + half)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        arrow_len = half * 0.5
        ax.quiver(0, 0, 0, arrow_len, 0, 0, color='red',   linewidth=2, arrow_length_ratio=0.2)
        ax.quiver(0, 0, 0, 0, arrow_len, 0, color='green', linewidth=2, arrow_length_ratio=0.2)
        ax.quiver(0, 0, 0, 0, 0, arrow_len, color='blue',  linewidth=2, arrow_length_ratio=0.2)
        ax.text(arrow_len * 1.15, 0, 0, 'X', color='red',   fontsize=11, fontweight='bold')
        ax.text(0, arrow_len * 1.15, 0, 'Y', color='green', fontsize=11, fontweight='bold')
        ax.text(0, 0, arrow_len * 1.15, 'Z', color='blue',  fontsize=11, fontweight='bold')

        hl, hb = sim.l / 2, sim.b / 2
        c_local = np.array([
            [-hl, -hb,       0],
            [+hl, -hb,       0],
            [+hl, +hb,       0],
            [-hl, +hb,       0],
            [-hl, -hb, sim.h],
            [+hl, -hb, sim.h],
            [+hl, +hb, sim.h],
            [-hl, +hb, sim.h],
        ])
        box_edges = [(0,1),(1,2),(2,3),(3,0),
                     (4,5),(5,6),(6,7),(7,4),
                     (0,4),(1,5),(2,6),(3,7)]
        box_lines      = [None] * 12
        box_lines_top  = [None] * 12
        box_lines_side = [None] * 12

        ax.scatter(data[0, 1], data[0, 2], data[0, 3], color='blue', s=60, zorder=5)
        trail, = ax.plot([], [], [], 'b-', linewidth=1.5)

        # ── Time-series axes ──
        t_all     = data[:, 0]
        x_icr_all = data[:, 7]
        dy_all    = data[:, 8]
        dyaw_all  = data[:, 9]

        def _ylim(arr, margin=0.05):
            lo, hi = float(arr.min()), float(arr.max())
            span = hi - lo if abs(hi - lo) > 1e-9 else 0.2
            return lo - margin * span, hi + margin * span

        for ax_ts, arr, ylabel, _ in [
            (ax_icr,  x_icr_all, "x_ICR (m)",          'tab:blue'),
            (ax_dy,   dy_all,    "lateral vel (m/s)",   'tab:green'),
            (ax_dyaw, dyaw_all,  "yaw rate (rad/s)",    'tab:red'),
        ]:
            ax_ts.set_xlim(t_all[0], t_all[-1])
            ax_ts.set_ylim(*_ylim(arr))
            ax_ts.set_ylabel(ylabel)
            ax_ts.grid(True)

        ax_dyaw.set_xlabel("time (s)")

        line_icr,  = ax_icr.plot([], [], color='tab:blue',  linewidth=1.5)
        line_dy,   = ax_dy.plot([], [],  color='tab:green', linewidth=1.5)
        line_dyaw, = ax_dyaw.plot([], [], color='tab:red',  linewidth=1.5)

        def update(i):
            trail.set_data(data[:i+1, 1], data[:i+1, 2])
            trail.set_3d_properties(data[:i+1, 3])

            for line in box_lines:
                if line is not None:
                    line.remove()
            for line in box_lines_top:
                if line is not None:
                    line.remove()
            for line in box_lines_side:
                if line is not None:
                    line.remove()

            R   = sim.rotation_lg(data[i, 4], data[i, 5], data[i, 6])
            pos = data[i, 1:4]
            c_g = pos + (R @ c_local.T).T

            for j, (ia, ib) in enumerate(box_edges):
                p1, p2 = c_g[ia], c_g[ib]
                box_lines[j], = ax.plot(
                    [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color='red', linewidth=1.5)

            ax.set_title(f"t = {data[i, 0]:.2f} s")

            line_icr.set_data(t_all[:i+1],  x_icr_all[:i+1])
            line_dy.set_data(t_all[:i+1],   dy_all[:i+1])
            line_dyaw.set_data(t_all[:i+1], dyaw_all[:i+1])

            return trail,

        anim = animation.FuncAnimation(
            fig, update, frames=n_frames, blit=False, interval=50
        )
        fname = "simulation.gif"
        anim.save(fname, writer=animation.PillowWriter(fps=20))
        plt.close(fig)
        print(f"Saved {fname}")


def main():
    sim = BulldozerSimulation()
    sim.run_and_plot(stop_time=2)

if __name__ == "__main__":
    main()
