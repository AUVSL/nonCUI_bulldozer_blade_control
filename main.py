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
        #TODO: add comments with parameter descriptions and units (maybe change names to be more descriptive?)
        # simulation and control parameters
        self.dt            = 1/100
        self.stop_time     = 1
        grav               = 9.81
        self.surface_abg   = np.array([ 0, 0, 0])
        self.desired_depth = -0.05
        self.desired_abg   = np.array([ 0, 0, 0.0])
        self.fill_distance = 8.0
        self.Kp            = -3.0       # Controller gains

        # Dozer body parameters
        m                                = 10156.0 # scaled down by Sam
        self.h                           = 2.762/2 # scaled down by Sam
        self.l                           = 2.349   # scaled down by Sam
        self.b                           = 1.75    # scaled down by Sam
        self.longitudinal_velocity_limit = 2.222
        self.laterial_velocity_limit     = 0.0     # this governs how much the dozer can "slide" laterally
        self.angular_velocity_limit      = 2 * self.longitudinal_velocity_limit / self.b
        self.F_track_base                = 600000.0
        self.F_track                     = np.array([self.F_track_base, self.F_track_base*0.79])

        # Bulldozer blade parameters
        self.B1   = 2.921
        self.H    = 0.955
        self.L    = 1.2
        self.gain = 1/40

        # Soil parameters
        self.mu_l    = 0.1
        self.mu_t    = 0.9
        self.mu_ss   = 0.5
        self.kb      = 0.734e6
        self.beta0   = np.radians(38.0)
        self.gamma_g = 1640 * grav

        # Dynamic motion parameters
        self.rl   = self.mu_l * m * grav / 2
        self.fy   = self.mu_t * m * grav / self.l
        Ix        = m * (self.b**2 + self.h**2) / 12
        Iy        = m * (self.h**2 + self.l**2) / 12
        Iz        = m * (self.b**2 + self.l**2) / 12
        self.M    = np.diag([m, m, m, Ix, Iy, Iz])
        self.P    = np.array([0, 0, m * grav, 0, 0, 0])
        self.elim = np.diag([1, 1, 1, 0, 0, 1])

        # Initial conditions
        self.dxyz      = np.zeros(3)
        self.daBg      = np.zeros(3)
        self.bld_ang   = np.zeros(3)  # [roll, pitch, yaw]
        self.q         = np.array([0.0, 0.0, 0.0, self.surface_abg[0], self.surface_abg[1], self.surface_abg[2]])
        self.q_dot     = np.zeros(6)
        self.v         = np.zeros(2)
        self.R_lg      = self.rotation_lg(self.q[3], self.q[4], self.q[5])
        _, self.J_lg   = self.rotation_derivatives(self.q[3], self.q[4])
        self.x_ICR     = 0.0
        self.x_ICR_dot = 0.0
        self.Fb        = 0.0
        self.Mb        = 0.0
        self.Rl        = np.zeros(2)
        self.Fy        = 0.0
        self.Mr        = 0.0
        self.vtL       = 0.0
        self.vtR       = 0.0
        self.v_dot     = np.zeros(2)
        self.log       = []

    # ---------------- Helpers ----------------
    @staticmethod
    def saturation(value, limit):
        return float(np.clip(value, -abs(limit), abs(limit)))

    @staticmethod
    def wrap_angles(angles):
        return (angles + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def G(F, f, dx):
        if (abs(dx) > 1e-10):
            return -f * np.sign(dx)
        elif abs(F) <= f:
            return -F
        else:
            return -f * np.sign(F)

    @staticmethod
    def yc(D1, D2, B1):
        """Centroid of a trapezoid where left is posative and right is negative."""
        if (D1 == 0 and D2 == 0):
            return 0.0
        return (2 * D1 + D2) / (3 * (D1 + D2)) * B1 - B1 / 2
    
    # ---------------- Kinematics ----------------
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
        S[0:3, 1] = self.R_lg[:, 1] * (-self.x_ICR) # lateral/turning velocity
        S[3:6, 1] = self.J_lg[:, 2]                 # yaw contribution

        return S

    def get_x_icr(self, eps: float = 1e-3):
        if abs(self.daBg[2]) < eps:
            return 0.0
        return float(np.clip(-self.dxyz[1] / self.daBg[2], -self.l / 2, self.l / 2))
    
    # ---------------- Dynamics ----------------
    def blade_terrain_interaction(self):
        a_rel = self.bld_ang[0]
        hp    = abs(self.L * np.sin(self.bld_ang[1]))
        
        H1     = self.B1 * np.tan(abs(a_rel))
        H2     = hp / np.cos(abs(a_rel))
        H3_sub = - H2 + (np.sign(a_rel) * H1 / 2) - (H1 / 2)
        H4_sub = - H2 - (np.sign(a_rel) * H1 / 2) - (H1 / 2)
        H3     = self.H  + H3_sub
        H4     = self.H  + H4_sub

        a_val = np.tan(abs(a_rel)) ** 2
        c_val = (H3 + H4) / 2
        V     = 0.5 / np.tan(self.beta0) * (1 / 12 * a_val * self.B1 ** 3 + c_val ** 2 * self.B1)

        # TODO: fill assumes a spawn at the origin, but could be adapted to a more general case if needed
        fill_percent = np.linalg.norm(self.q[:3]) / self.fill_distance
        Gt           = V * self.gamma_g * fill_percent
        
        hyp      = self.B1 / np.cos(abs(a_rel))
        area_cut = 0.5 * self.B1 * H1 + hyp * hp
        F1       = area_cut * self.kb
        F2       = Gt * self.mu_ss
        self.Fb  = -F1 - F2 

        yc1     = self.yc(H3 / np.tan(self.beta0), H4 / np.tan(self.beta0), self.B1)
        yc2     = self.yc(-H3_sub, -H4_sub, self.B1)
        self.Mb = yc1 * F1 + yc2 * F2

    def track_terrain_interaction(self):
        self.vtL = self.saturation(self.dxyz[0] - self.b / 2 * self.daBg[2], self.longitudinal_velocity_limit)
        self.vtR = self.saturation(self.dxyz[0] + self.b / 2 * self.daBg[2], self.longitudinal_velocity_limit)

        FtL, FtR = self.F_track[0], self.F_track[1]
        
        RlL     = self.G(FtL, self.rl, self.vtL)
        RlR     = self.G(FtR, self.rl, self.vtR)
        self.Rl = np.array([RlL, RlR])
        
        self.Fy = -2 * np.sign(self.dxyz[1]) * self.fy * abs(self.x_ICR)

        M       = ((FtR + RlR) - (FtL + RlL)) * self.b / 2
        mr      = 2 * self.fy * (((self.l ** 2) / 4) - (self.x_ICR ** 2))
        self.Mr = self.G(M, mr, self.daBg[2])
        
    def Sd_matrix(self):
        """
        Time derivative of the S matrix.
        """
        a, B, g    = self.q[3:6]
        Ad, Bd, Gd = self.q_dot[3:6]

        sa, ca = np.sin(a), np.cos(a)
        sB, cB = np.sin(B), np.cos(B)
        sg, cg = np.sin(g), np.cos(g)
        tB     = np.tan(B)

        R = self.R_lg  # R_{i,j} = R_lg[i,j] per equation notation

        S_11 = -sB * cg * Bd - R[1, 0] * Gd
        S_21 = -sB * sg * Bd + R[0, 0] * Gd
        S_31 = -cB * Bd

        S_12 = -self.x_ICR * ( R[0, 2] * Ad + R[2, 1] * cg * Bd - R[1, 1] * Gd) - self.x_ICR_dot * R[0, 1]
        S_22 = -self.x_ICR * ( R[1, 2] * Ad + R[2, 1] * sg * Bd + R[0, 1] * Gd) - self.x_ICR_dot * R[1, 1]
        S_32 = -self.x_ICR * ( R[2, 2] * Ad - sa * sB * Bd)                     - self.x_ICR_dot * R[2, 1]

        S_42 = -sa * tB * Ad + ca / (cB**2) * Bd
        S_52 = -ca * Ad
        S_62 = -sa / cB * Ad + ca * tB / cB * Bd

        Sd = np.array([
            [S_11, S_12],
            [S_21, S_22],
            [S_31, S_32],
            [0.0,  S_42],
            [0.0,  S_52],
            [0.0,  S_62]
        ])

        return Sd

    def vehicle_dynamics(self):
        # update forces and moments for current time step
        self.track_terrain_interaction()
        self.blade_terrain_interaction()

        B_mat = np.zeros((6, 2))
        B_mat[0:3, 0] = self.R_lg[:, 0]
        B_mat[0:3, 1] = self.R_lg[:, 0]
        B_mat[3:6, 0] = -self.R_lg[:, 2] * self.b / 2
        B_mat[3:6, 1] =  self.R_lg[:, 2] * self.b / 2

        #UPDATE: when changing the rotation angle convention
        ab, Bb, gb = self.bld_ang
        
        Ct_vec = np.array([
            self.Rl.sum(), 
            self.Fy, 0,
            0, 
            0,
            self.Mr + (self.Rl[1] - self.Rl[0]) * self.b / 2
        ])
        R_blade = self.rotation_lg(ab, Bb, gb)
        
        R6 = np.zeros((6, 6))
        R6[0:3, 0:3] = R_blade
        R6[3:6, 3:6] = R_blade

        blade_vec = np.array([self.Fb, 0.0, 0.0, 0.0, 0.0, self.Mb])
        Cb_vec    = self.elim @ R6 @ blade_vec
        
        R6_lg = np.zeros((6, 6))
        R6_lg[0:3, 0:3] = self.R_lg
        R6_lg[3:6, 3:6] = self.R_lg
        C = R6_lg @ (Ct_vec)
        
        S  = self.S_matrix()
        Sd = self.Sd_matrix()

        Bt = S.T @ B_mat
        Ct = S.T @ C
        Pt = S.T @ self.P
        Mt = S.T @ self.M @ S
        Et = S.T @ self.M @ Sd

        v_dot = np.linalg.solve(Mt, Bt @ self.F_track + Ct - Et @ self.v - Pt)
        return v_dot
    
    # ---------------- Control ----------------
    def controller_errors(self):
        """
        Computes blade roll, pitch, yaw errors relative to desired surface and depth.
        """
        roll, pitch, yaw                          = self.bld_ang
        desired_roll, des_pitch_mult, desired_yaw = self.desired_abg

        # TODO: update blade angle limits from -1 to 1 to something more realistic, and update the test cases accordingly
        desired_pitch = des_pitch_mult * np.arcsin(np.clip(self.desired_depth / self.L, -1.0, 1.0))

        errors  = np.array([desired_roll  - roll,      desired_pitch - pitch, desired_yaw - yaw])
        plot_out = np.array(          [errors[0], np.sin(errors[1]) * self.L,         errors[2]])

        return errors, plot_out
    
    # ---------------- Main Loop ----------------
    def run(self):
        t = 0.0
        for _ in range(int(self.stop_time / self.dt)):
            errors, plot_err = self.controller_errors()

            self.bld_ang += self.gain * self.Kp * errors
            self.v_dot = self.vehicle_dynamics()
            self.v    += self.dt * self.v_dot
            self.v[0]  = max(min(self.v[0], self.longitudinal_velocity_limit), 0)
            self.v[1]  = self.saturation(self.v[1], self.angular_velocity_limit)
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

            self.dxyz[1] = self.saturation(self.dxyz[1], self.laterial_velocity_limit)
            self.x_ICR   = self.get_x_icr()

            self.log.append([t, *self.q, self.x_ICR, self.dxyz[1], self.daBg[2]])
            t += self.dt

    # ---------------- Visualization ----------------
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

    def plot_track_force_sweep(self):
        """Overlay XY trajectories: left track fixed at F_track_base, right track swept over 100 steps from 0 to 0.5 x F_track_base."""
        fractions = np.linspace(0, 0.8, 100)
        colors = plt.cm.viridis(np.linspace(0, 0.6, 100))

        fig, ax = plt.subplots(figsize=(10, 8))

        for i, fraction in enumerate(fractions):
            sim = BulldozerSimulation()
            sim.F_track[0] = sim.F_track_base
            sim.F_track[1] = fraction * sim.F_track_base
            sim.run()

            data = np.array(sim.log)
            ax.plot(data[:, 1], data[:, 2], color=colors[i], linewidth=0.8, alpha=0.7)

        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
        plt.colorbar(sm, ax=ax, label="right track fraction of F_base")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("XY trajectories — left track = F_base, right track = 0 to 1 x F_base (100 steps)")
        ax.set_aspect("equal")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig("track_force_sweep.png", dpi=150)
        plt.close(fig)
        print("Saved track_force_sweep.png")


def main():
    sim = BulldozerSimulation()
    sim.plot_track_force_sweep()
    sim.run_and_plot(stop_time=2.0, first_frame_only=False)

if __name__ == "__main__":
    main()
