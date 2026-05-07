"""
Python port of the 3D bulldozer blade control simulation.
Replaces the Simulink sim('simulation_3d') call with an explicit
forward-Euler integration loop.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
matplotlib.use("Agg")   # headless; remove if running interactively

class BulldozerSimulation:
    def __init__(self):
        # ───────────────── Parameters ─────────────────
        m                   = 10156.0 /10
        self.dt             = 1/100
        self.stop_distance  = 5
        self.stop_time      = 10

        grav    = 9.81
        h       = 2.762
        self.h  = h
        self.l  = 2.349
        self.b  = 1.75
        self.B1 = 2.921
        self.H  = 0.955
        self.L  = 1.2
        
        self.mu_l  = 0.1
        self.mu_t  = 0.9
        self.mu_ss = 0.5
        self.kb    = 0.734e6

        self.beta0 = np.radians(38.0)


        self.gain           = 1 / 40
        self.velocity_limit = 2.222
        self.turn_vel_limit = 2 * self.velocity_limit / self.b
        self.fill_distance  = 8.0
        self.gamma_g        = 1640 * grav
        self.elim           = np.diag([1, 1, 1, 0, 0, 1])

        # Dynamic motion parameters
        self.rl = self.mu_l * m * grav / 2
        self.fy = self.mu_t * m * grav / self.l

        Ix     = m * (self.b**2 +      h**2) / 12
        Iy     = m * (     h**2 + self.l**2) / 12
        Iz     = m * (self.b**2 + self.l**2) / 12
        self.M = np.diag([m, m, m, Ix, Iy, Iz])
        self.P = np.array([0, 0, m * grav, 0, 0, 0])

        # Controller (proportional placeholders)
        self.Kp = -3.0

        # ───────────────── Initial Conditions ─────────────────
        self.desired_depth = -0.4
        self.desired_abg   = np.array([-0.00, 0, 0.000])
        self.surface_abg   = np.array([ 0.00, 0.0,  0.00])
        self.F_track       = np.array([60000.0, 10000.0])
        self.x_ICR_dot  = 0.0
        self.bld_ang = np.zeros(3)
        self.dxyz    = np.zeros(3)
        self.daBg    = np.zeros(3)

        self.q = np.array([
            0.0, 0.0, 0.0,
            self.surface_abg[0],
            self.surface_abg[1],
            self.surface_abg[2]
        ])
        self.q_dot = np.zeros(6)

        self.v       = np.zeros(2)
        self.x_ICR   = 0.0
        self.R_lg    = self.rotation_lg(self.q[3], self.q[4], self.q[5])
        _, self.J_lg = self.rotation_derivatives(self.q[3], self.q[4])

        self.Fb       = 0.0
        self.Mb       = 0.0
        self.Rl       = np.zeros(2)
        self.Fy       = 0.0
        self.Mr       = 0.0
        self.vtL      = 0.0
        self.vtR      = 0.0

        # Logs
        self.log = []

    # ───────────────── Helpers ─────────────────
    @staticmethod
    def saturation(value, limit):
        return float(np.clip(value, -limit, limit))

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
        """Centroid of a trapezoid."""
        if (D1 == 0 and D2 == 0):
            return 0.0
        return (D1 + 2 * D2) / (3 * (D1 + D2)) * B1 - B1 / 2

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
            [1, sa * tB, ca * tB],
            [0,      ca,    - sa],
            [0, sa / cB,  ca / cB]
        ])

        J_lg = np.array([
            [1,   0,     -sB],
            [0,  ca, sa * cB],
            [0, -sa, ca * cB]
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
        x = self.safe_division_x_icr()

        S = np.zeros((6, 2))
        S[0:3, 0] = self.R_lg[:, 0]            # forward velocity
        S[0:3, 1] = self.R_lg[:, 1] * (-1.0/x) # lateral/turning velocity
        S[3:6, 1] = self.J_lg[:, 2]  # yaw contribution

        return S

    def get_x_icr(self, eps: float = 1e-3):
        if abs(self.daBg[2]) < eps:
            return 0.0
        return float(np.clip(self.dxyz[1] / self.daBg[2], -self.l / 2, self.l / 2))
    
    def safe_division_x_icr(self, eps: float = 1e-1):
        if abs(self.x_ICR) < eps:
            return np.finfo(float).max
        return self.x_ICR

    # ───────────────── Dynamics ─────────────────
    def blade_terrain_interaction(self):
        a_rel = self.surface_abg[0] - self.bld_ang[0]
        hp    = abs(self.L * np.sin(self.bld_ang[1]))

        H1 = self.B1 * np.tan(abs(a_rel))
        H2 = hp / np.cos(a_rel)
        H3 = self.H - H2 + np.sign(a_rel) * H1 / 2 - H1 / 2
        H4 = self.H - H2 - np.sign(a_rel) * H1 / 2 - H1 / 2

        a_val = np.tan(abs(a_rel)) ** 2
        c_val = (H3 + H4) / 2
        V = 0.5 / np.tan(self.beta0) * (
            1 / 12 * a_val ** 2 * self.B1 ** 3 + c_val ** 2 * self.B1
        )

        # TODO: fill assumes a spawn at the origin, but could be adapted to a more general case if needed
        fill_percent = np.linalg.norm(self.q[:3]) / self.fill_distance
        Gt = V * self.gamma_g * fill_percent

        hyp      = self.B1 / np.cos(abs(a_rel))
        area_cut = 0.5 * self.B1 * H1 + hyp * hp
        F1       = area_cut * self.kb
        F2       = Gt * self.mu_ss
        self.Fb  = -F1 - F2

        yc1     = self.yc(H3 / np.tan(self.beta0), H4 / np.tan(self.beta0), self.B1)
        yc2     = self.yc(H2, H1 + H2, self.B1)
        self.Mb = yc1 * F1 + yc2 * F2

    def track_terrain_interaction(self):
        vtL = self.saturation(self.dxyz[0] - self.b / 2 * self.daBg[2], self.velocity_limit)
        vtR = self.saturation(self.dxyz[0] + self.b / 2 * self.daBg[2], self.velocity_limit)
        self.vtL = vtL
        self.vtR = vtR

        FtL, FtR = self.F_track[0], self.F_track[1]
        
        RlL     = self.G(FtL, self.rl, vtL)
        RlR     = self.G(FtR, self.rl, vtR)
        self.Rl = np.array([RlL, RlR])
        
        self.Fy = -2 * np.sign(self.dxyz[1]) * self.fy * abs(self.x_ICR)

        M       = ((FtR + RlR) - (FtL + RlL)) * self.b / 2
        mr      = 2 * self.fy * ((self.l ** 2) / 4 - self.x_ICR ** 2)
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

        # Time-derivative of S (Sd)
        # TODO: derive a matrix form for this instead of hardcoding each element (using R_gl since many of the values are already stored there)
        S_11 =                                     -sB * cg * Bd                  - cB * sg * Gd
        S_21 = (ca * sB * cg + sa * sg) * Ad + sa * cB * cg * Bd - (sa * sB * sg + ca * cg) * Gd
        S_31 = (ca * sg - sa * sB * cg) * Ad + ca * cB * cg * Bd + (sa * cg - ca * sB * sg) * Gd

        S_12 = -self.x_ICR * (-sB * sg * Bd + cB * cg * Gd)                                                       - self.x_ICR_dot * (cB * sg)
        S_22 = -self.x_ICR * ((ca * sB * sg - sa * cg) * Ad + sa * cB * sg * Bd + (sa * sB * cg - ca * sg) * Gd)  - self.x_ICR_dot * (sa * sB * sg + ca * cg)
        S_32 = -self.x_ICR * (-(sa * sB * sg + ca * cg) * Ad + ca * cB * sg * Bd + (ca * sB * cg + sa * sg) * Gd) - self.x_ICR_dot * (ca * sB * sg - sa * cg)

        S_42 = -cB * Bd
        S_52 =  ca * cB * Ad - sa * sB * Bd
        S_62 = -sa * cB * Ad - ca * sB * Bd

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
        self.blade_terrain_interaction()
        self.track_terrain_interaction()

        B_mat = np.zeros((6, 2))
        B_mat[0:3, 0] = self.R_lg[:, 0]
        B_mat[0:3, 1] = self.R_lg[:, 0]
        B_mat[3:6, 0] = -self.R_lg[:, 2] * self.b / 2
        B_mat[3:6, 1] =  self.R_lg[:, 2] * self.b / 2

        #UPDATE: when changing the rotation angle convention
        ab, Bb, gb = self.bld_ang
        R_blade = self.rotation_lg(ab, Bb, gb)

        R6 = np.zeros((6, 6))
        R6[0:3, 0:3] = R_blade
        R6[3:6, 3:6] = R_blade
        Ct_vec = np.array([
            self.Rl.sum(), self.Fy, 0,
            0, 0,
            self.Mr + (self.Rl[1] - self.Rl[0]) * self.b / 2
        ])

        blade_vec = np.array([self.Fb, 0.0, 0.0, 0.0, 0.0, self.Mb])
        Cb_vec = self.elim @ R6 @ blade_vec

        R6_lg = np.zeros((6, 6))
        R6_lg[0:3, 0:3] = self.R_lg
        R6_lg[3:6, 3:6] = self.R_lg
        C = R6_lg @ (Ct_vec + Cb_vec)

        S  = self.S_matrix()
        Sd = self.Sd_matrix()

        Bt = S.T @ B_mat
        Ct = S.T @ C

        Pt = S.T @ self.P
        Mt = S.T @ self.M @ S
        Et = S.T @ self.M @ Sd
        v_dot = np.linalg.solve(Mt, Bt @ self.F_track + Ct - Et @ self.v - Pt)
        return v_dot
    # ───────────────── Controller ─────────────────
    def controller_errors(self):
        """
        Computes blade roll, pitch, yaw errors relative to desired surface and depth.

        Returns
        -------
        errors : np.ndarray, shape (3,)
            [roll_error, pitch_error, yaw_error]
        plot_out : np.ndarray, shape (3,)
            [roll_error, depth_error, yaw_error]
        """
        roll, pitch, yaw = self.bld_ang
        desired_roll, des_pitch_mult, desired_yaw = self.desired_abg

        desired_pitch = des_pitch_mult * np.arcsin(
            np.clip(self.desired_depth / self.L, -1.0, 1.0)
        )

        errors = np.array([
            roll  - desired_roll,
            pitch - desired_pitch,
            yaw   - desired_yaw
        ])

        # Matches errors_and_plots.m convention
        plot_out = np.array([
            errors[0],
            np.sin(errors[1]) * self.L,
            errors[2]
        ])

        return errors, plot_out

    # ───────────────── Main Integration Loop ─────────────────
    def run(self):
        t = 0.0
        for _ in range(int(self.stop_time / self.dt)):
            #TODO: replace this with a distance traveled stopping condition in addition to time and
            if abs(self.q[0]) + abs(self.q[1]) > self.stop_distance:
                break

            errors, plot_err = self.controller_errors()

            self.bld_ang += self.gain * self.Kp * errors

            v_dot = self.vehicle_dynamics()

            self.v += self.dt * v_dot
            self.v[0] = max(min(self.v[0], self.velocity_limit), 0)
            self.v[1] = np.clip(self.v[1], -self.turn_vel_limit, self.turn_vel_limit)
            self.q_dot = self.S_matrix() @ self.v

            # update global/local positions and orientations for next time step
            self.q   += self.dt * self.q_dot
            self.q[3:6]    = self.wrap_angles(self.q[3:6])
            # TODO: pass just q[3:6] directly to the rotation functions
            a, B, g         = self.q[3:6]
            self.R_lg       = self.rotation_lg(a, B, g)
            R_gl            = self.rotation_gl(a, B, g)
            J_gl, self.J_lg = self.rotation_derivatives(a, B)
            

            self.dxyz = R_gl @ self.q_dot[0:3]
            self.daBg = J_gl @ self.q_dot[3:6]
            
            prev           = self.x_ICR
            self.x_ICR     = self.get_x_icr()
            self.x_ICR_dot = (self.x_ICR - prev) / self.dt

            self.log.append([t, *self.q, *plot_err,
                             self.dxyz[0], self.dxyz[1], self.daBg[2], self.x_ICR,
                             self.vtL, self.vtR,
                             self.F_track[0], self.F_track[1],
                             self.Rl[0], self.Rl[1],
                             self.Fy, self.Mr,
                             self.v[0], self.v[1]])
            t += self.dt
            
    def post_process_and_plot(self):
        data = np.array(self.log)
        time = data[:, 0]

        roll_error  = data[:, 7]
        depth_error = data[:, 8]
        yaw_error   = data[:, 9]

        def rmse_me(x):
            rmse = np.sqrt(np.mean(x ** 2))
            me   = np.max(np.abs(x))
            return rmse, me

        rmse_r, me_r = rmse_me(roll_error)
        rmse_d, me_d = rmse_me(depth_error)
        rmse_y, me_y = rmse_me(yaw_error)

        print(f"RMSE  roll={rmse_r*1000:.4f} mrad  "
              f"depth={rmse_d*1000:.4f} mm  "
              f"yaw={rmse_y*1000:.4f} mrad")
        print(f"Max-E roll={me_r*1000:.4f}        "
              f"depth={me_d*1000:.4f}       "
              f"yaw={me_y*1000:.4f}")

        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], hspace=0.4, wspace=0.35)
        ax3d = fig.add_subplot(gs[:, 0], projection='3d')

        # Surface plane: normal is col 2 of R_lg = rotation_lg(surface_abg)
        a, B, g = self.surface_abg
        sa, ca = np.sin(a), np.cos(a)
        sB, cB = np.sin(B), np.cos(B)
        sg, cg = np.sin(g), np.cos(g)
        nx, ny, nz = ca*sB*cg + sa*sg, ca*sB*sg - sa*cg, ca*cB
        margin = 1.0
        cx = (data[:,1].max() + data[:,1].min()) / 2
        cy = (data[:,2].max() + data[:,2].min()) / 2
        cz = (data[:,3].max() + data[:,3].min()) / 2
        half_s = max(data[:,1].max() - data[:,1].min(),
                     data[:,2].max() - data[:,2].min(),
                     data[:,3].max() - data[:,3].min()) / 2 + margin
        xs = np.linspace(cx - half_s, cx + half_s, 30)
        ys = np.linspace(cy - half_s, cy + half_s, 30)
        Xs, Ys = np.meshgrid(xs, ys)
        Zs = -(nx * Xs + ny * Ys) / nz
        ax3d.plot_surface(Xs, Ys, Zs, alpha=0.3, color='tan')
    
        ax3d.plot(data[:,1], data[:,2], data[:,3],
                  color='blue', linewidth=2)
        print(data[:,1].min(), data[:,1].max())
        ax3d.set_xlim(cx - half_s, cx + half_s)
        ax3d.set_ylim(cy - half_s, cy + half_s)
        ax3d.set_zlim(cz - half_s, cz + half_s)
        ax3d.set_xlabel("X (m)")
        ax3d.set_ylabel("Y (m)")
        ax3d.set_zlabel("Z (m)")

        ax_roll = fig.add_subplot(gs[0, 1])
        ax_roll.plot(time, data[:,4])
        ax_roll.set_ylabel("Roll (rad)")

        ax_yaw = fig.add_subplot(gs[1, 1])
        ax_yaw.plot(time, data[:,6])
        ax_yaw.set_ylabel("Yaw (rad)")
        plt.savefig("simulation_results.png", dpi=150)

        
    def make_position_gif(self):
        data = np.array(self.log)[::5]

        fig = plt.figure(figsize=(7, 7))
        ax  = fig.add_subplot(111, projection='3d')

        # Surface plane (same normal derivation as post_process_and_plot)
        a, B, g = self.surface_abg
        sa, ca  = np.sin(a), np.cos(a)
        sB, cB  = np.sin(B), np.cos(B)
        sg, cg  = np.sin(g), np.cos(g)
        nx = ca*sB*cg + sa*sg
        ny = ca*sB*sg - sa*cg
        nz = ca*cB
        margin = 1.0
        cx = (data[:,1].max() + data[:,1].min()) / 2
        cy = (data[:,2].max() + data[:,2].min()) / 2
        cz = (data[:,3].max() + data[:,3].min()) / 2
        half = max(data[:,1].max() - data[:,1].min(),
                   data[:,2].max() - data[:,2].min(),
                   data[:,3].max() - data[:,3].min()) / 2 + margin
        xs = np.linspace(cx - half, cx + half, 30)
        ys = np.linspace(cy - half, cy + half, 30)
        Xs, Ys = np.meshgrid(xs, ys)
        Zs = -(nx * Xs + ny * Ys) / nz
        ax.plot_surface(Xs, Ys, Zs, alpha=0.3, color='tan', zorder=0)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_zlim(cz - half, cz + half)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        # Box corners in local frame, origin = bottom centre
        hl, hb = self.l / 2, self.b / 2
        c_local = np.array([
            [-hl, -hb,          0],  # 0 bottom rear-left
            [+hl, -hb,          0],  # 1 bottom front-left
            [+hl, +hb,          0],  # 2 bottom front-right
            [-hl, +hb,          0],  # 3 bottom rear-right
            [-hl, -hb, self.h], # 4 top rear-left
            [+hl, -hb, self.h], # 5 top front-left
            [+hl, +hb, self.h], # 6 top front-right
            [-hl, +hb, self.h], # 7 top rear-right
        ])
        box_edges = [(0,1),(1,2),(2,3),(3,0),
                     (4,5),(5,6),(6,7),(7,4),
                     (0,4),(1,5),(2,6),(3,7)]
        box_lines = [None] * 12

        trail, = ax.plot([], [], [], 'b-', linewidth=1.5)

        def update(i):
            trail.set_data(data[:i+1, 1], data[:i+1, 2])
            trail.set_3d_properties(data[:i+1, 3])

            for line in box_lines:
                if line is not None:
                    line.remove()

            R   = self.rotation_lg(data[i, 4], data[i, 5], data[i, 6])
            pos = data[i, 1:4]
            c_g = pos + (R @ c_local.T).T  # (8, 3) corners in global frame

            for j, (a, b) in enumerate(box_edges):
                p1, p2 = c_g[a], c_g[b]
                box_lines[j], = ax.plot(
                    [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color='red', linewidth=1.5
                )

            ax.set_title(f"t = {data[i, 0]:.2f} s")
            return trail,

        anim = animation.FuncAnimation(
            fig, update, frames=len(data), blit=False, interval=50
        )
        anim.save("position_3d.gif", writer=animation.PillowWriter(fps=20))
        plt.close(fig)
        print("Saved position_3d.gif")

def main():
    sim = BulldozerSimulation()
    sim.run()
    sim.post_process_and_plot()
    sim.make_position_gif()


if __name__ == "__main__":
    main()
