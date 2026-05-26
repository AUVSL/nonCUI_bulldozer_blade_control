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
        self.stop_time     = 0.7
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
        self.velocity_limit          = 2.222
        self.laterial_velocity_limit = 0.0     # this governs how much the dozer can "slide" laterally
        self.angular_velocity_limit  = 2 * self.velocity_limit / self.b
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
        self.dxyz                = np.zeros(3)
        self.daBg                = np.zeros(3)
        self.bld_ang             = np.zeros(3)  # [roll, pitch, yaw]
        self.q                   = np.array([0.0, 0.0, 0.0, self.surface_abg[0], self.surface_abg[1], self.surface_abg[2]])
        self.q_dot               = np.zeros(6)
        self.v                   = np.zeros(2)
        self.R_lg                = self.rotation_lg(self.q[3], self.q[4], self.q[5])
        _, self.J_lg             = self.rotation_derivatives(self.q[3], self.q[4])
        self.x_ICR               = 0.0
        self.x_ICR_dot           = 0.0
        self.Fb                  = 0.0
        self.Mb                  = 0.0
        self.Rl                  = np.zeros(2)
        self.Fy                  = 0.0
        self.Mr                  = 0.0
        self.vtL                 = 0.0
        self.vtR                 = 0.0
        self.v_dot               = np.zeros(2)
        self.log                 = []
        self.cross_track_err     = 0.0
        self.heading_err         = 0.0
        self.backward            = False
        self._active_controller  = None
        self.path_points         = self.figure8_path()
        self.stop_after_first_loop = True
        self._nearest_path_idx   = 0
        self._passed_halfway     = False

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
        self.vtL = self.saturation(self.dxyz[0] - self.b / 2 * self.daBg[2], self.velocity_limit)
        self.vtR = self.saturation(self.dxyz[0] + self.b / 2 * self.daBg[2], self.velocity_limit)

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
    
    def figure8_path(self, A=5.0, B=2.5, n_points=2000):
        """Dense (x, y) waypoints for a figure-8: x=A·sin(t), y=B·sin(2t)."""
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        return np.column_stack([A * np.sin(t), B * np.sin(2 * t)])

    def signed_cross_track_error(self, pos_xy):
        """Signed perpendicular distance from pos_xy to self.path_points.
        Positive when the vehicle is to the left of the path tangent direction."""
        diffs = self.path_points - pos_xy
        idx   = int(np.argmin(np.linalg.norm(diffs, axis=1)))
        next_idx = (idx + 1) % len(self.path_points)
        tangent  = self.path_points[next_idx] - self.path_points[idx]
        norm     = np.linalg.norm(tangent)
        if norm < 1e-10:
            return 0.0
        tangent /= norm
        left_normal = np.array([-tangent[1], tangent[0]])  # CCW 90° of tangent
        return float(np.dot(pos_xy - self.path_points[idx], left_normal))

    def pure_pursuit_heading_error(self):
        """
        Pure-pursuit: find the lookahead point on the path at distance
        self._lookahead_dist from the vehicle, return the signed angle
        from current heading to that point.
        """
        pos = self.q[:2]
        L   = self._lookahead_dist
        n   = len(self.path_points)

        dists      = np.linalg.norm(self.path_points - pos, axis=1)
        nearest    = int(np.argmin(dists))
        self._nearest_path_idx = nearest
        lookahead_pt = None

        # Walk forward along path segments looking for circle intersection
        for k in range(n):
            i   = (nearest + k) % n
            j   = (i + 1) % n
            p1  = self.path_points[i]
            p2  = self.path_points[j]
            d   = p2 - p1
            f   = p1 - pos
            a   = float(np.dot(d, d))
            b   = 2.0 * float(np.dot(f, d))
            c   = float(np.dot(f, f)) - L * L
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            t2 = (-b + np.sqrt(disc)) / (2 * a)   # forward intersection
            if 0.0 <= t2 <= 1.0:
                lookahead_pt = p1 + t2 * d
                break

        if lookahead_pt is None:                    # fallback: nearest point
            lookahead_pt = self.path_points[nearest]

        dx  = lookahead_pt[0] - pos[0]
        dy  = lookahead_pt[1] - pos[1]
        err = np.arctan2(dy, dx) - self.q[5]
        return float((err + np.pi) % (2 * np.pi) - np.pi)

    def _log_path_controller(self):
        """Pure-pursuit heading error → track fraction via log mapping:
        fraction = threshold - A * log(|heading_err| + 1)"""
        self.heading_err     = self.pure_pursuit_heading_error()
        self.cross_track_err = self.signed_cross_track_error(self.q[:2])
        ang      = min(abs(self.heading_err), self._log_ang_max)
        fraction = float(np.clip(self._log_threshold - self._log_A * np.log(ang + 1), 0.0, 1.0))
        if self.heading_err > 0:
            self.F_track[0] = fraction * self.F_track_base
            self.F_track[1] = self.F_track_base
        else:
            self.F_track[0] = self.F_track_base
            self.F_track[1] = fraction * self.F_track_base
    
    def build_log_controller(self, threshold=0.7993, n_samples=200):
        """Calibrate log-based angle→fraction mapping.

        Fits scalar A via least-squares so that:
            fraction = threshold - A * log(|yaw| + 1)
        Returns (A, threshold, ang_max).
        """
        fractions = np.linspace(0.0, threshold, n_samples)
        yaws = []
        for frac in fractions:
            sim = BulldozerSimulation()
            sim.F_track[0] = sim.F_track_base
            sim.F_track[1] = frac * sim.F_track_base
            sim.run()
            yaws.append(np.array(sim.log)[-1, 6])
        yaws  = np.array(yaws)
        ang   = np.abs(yaws)
        T     = threshold - fractions
        basis = np.log(ang + 1)
        A     = float(np.dot(basis, T) / np.dot(basis, basis))
        ang_max = ang.max()
        rmse  = np.sqrt(np.mean((T - A * basis) ** 2))
        print(f"Log controller: A={A:.6f}  ang_max={ang_max:.4f} rad  RMSE={rmse:.6f}")
        return A, threshold, ang_max
    
    def use_log_controller(self, A, threshold, ang_max, lookahead_dist=1.5):
        self._log_A             = float(A)
        self._log_threshold     = float(threshold)
        self._log_ang_max       = float(ang_max)
        self._lookahead_dist    = float(lookahead_dist)
        self._active_controller = self._log_path_controller

    # ---------------- Main Loop ----------------
    def run(self):
        t = 0.0
        for _ in range(int(self.stop_time / self.dt)):
            errors, plot_err = self.controller_errors()
            self.bld_ang += self.gain * self.Kp * errors

            if self._active_controller is not None:
                self._active_controller()
                if self.stop_after_first_loop:
                    n = len(self.path_points)
                    if self._nearest_path_idx > n // 2:
                        self._passed_halfway = True
                    if self._passed_halfway and self._nearest_path_idx < n // 10:
                        break

            self.v_dot = self.vehicle_dynamics()
            self.v    += self.dt * self.v_dot
            if self.backward:
                self.v[0] = min(max(self.v[0], -self.velocity_limit), 0)
            else:
                self.v[0] = max(min(self.v[0], self.velocity_limit), 0)
            self.v[1]  = self.saturation(self.v[1], self.angular_velocity_limit)
            self.q_dot = self.S_matrix() @ self.v

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

            self.log.append([t, *self.q, self.cross_track_err, self.heading_err,
                             self.Mb, self.Fb, self.Rl[0], self.Rl[1], self.Fy, self.Mr,
                             self.v[0], self.v[1]])
            t += self.dt

    # ---------------- Visualization ----------------
    def plot_track_force_sweep(self):
        """Overlay XY trajectories: left track fixed at F_track_base, right track swept over 100 steps from 0 to 1 x F_track_base."""
        fractions = np.linspace(0, 1, 100)
        colors = plt.cm.viridis(np.linspace(0, 1, 100))

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

    def threshold_run(self, fraction):
        sim = BulldozerSimulation()
        sim.F_track[0] = sim.F_track_base
        sim.F_track[1] = fraction * sim.F_track_base
        sim.run()
        data = np.array(sim.log)
        return np.max(np.abs(data[:, 2]))   # peak |y| displacement

    def find_straight_threshold(self, tol=1e-4, n_iter=1000):
        """Binary search for the largest fraction where peak |y| > tol."""
        lo, hi = 0.0, 1.0
        for _ in range(n_iter):
            mid = (lo + hi) / 2
            if self.threshold_run(mid) > tol:
                lo = mid
            else:
                hi = mid
        print(f"Converged to straight threshold: {(lo + hi) / 2}")
        return (lo + hi) / 2

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

    def plot_path_tracking(self):
        data  = np.array(self.log)
        time  = data[:, 0]
        x_traj, y_traj = data[:, 1], data[:, 2]
        cross_track     = data[:, 7]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(self.path_points[:, 0], self.path_points[:, 1],
                 'g--', linewidth=1.5, label='Figure-8 reference')
        ax1.plot(x_traj, y_traj, 'b-', linewidth=1.5, label='Vehicle')
        ax1.plot(x_traj[0], y_traj[0], 'ko', markersize=7, label='Start')
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_title("Top-down path tracking")
        ax1.legend()
        ax1.set_aspect('equal')
        ax1.grid(True)

        ax2.plot(time, cross_track, 'r-', linewidth=1.2)
        ax2.axhline(0, color='k', linestyle='--', linewidth=0.8)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Cross-track error (m)")
        ax2.set_title("Perpendicular path error")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("path_tracking.png", dpi=150)
        plt.close(fig)
        print("Saved path_tracking.png")

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
        margin = 2.0
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

        ax.scatter(data[0, 1], data[0, 2], data[0, 3], color='blue', s=60, zorder=5)

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

    def demo_log_controller(self,stop_time=30.0, lookahead_dist=0.7):
        """Run figure-8 with pure-pursuit + log-based torque mapping."""
        print("Building log controller (200 calibration sims)...")
        A, threshold, ang_max = self.build_log_controller()

        print(f"Running figure-8 for {stop_time}s  lookahead={lookahead_dist}m ...")
        sim = BulldozerSimulation()
        sim.stop_time = stop_time
        sim.use_log_controller(A, threshold, ang_max, lookahead_dist=lookahead_dist)
        sim.run()

        data        = np.array(sim.log)
        cross_track = data[:, 7]
        heading_err = data[:, 8]

        def _stats(arr):
            return np.sqrt(np.mean(arr**2)), np.mean(np.abs(arr)), np.max(np.abs(arr))

        ct_rmse, ct_mae, ct_max = _stats(cross_track)
        he_rmse, he_mae, he_max = _stats(np.degrees(heading_err))

        print(f"\n{'':>16}  {'RMSE':>10}  {'MAE':>10}  {'Max |err|':>10}")
        print("-" * 52)
        print(f"{'Cross-track (m)':>16}  {ct_rmse:>10.4f}  {ct_mae:>10.4f}  {ct_max:>10.4f}")
        print(f"{'Heading (deg)':>16}  {he_rmse:>10.4f}  {he_mae:>10.4f}  {he_max:>10.4f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(sim.path_points[:, 0], sim.path_points[:, 1], 'g--', linewidth=1.5, label='Reference')
        ax1.plot(data[:, 1], data[:, 2], 'b-', linewidth=1.5, label='Vehicle')
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_title("Figure-8: log controller")
        ax1.legend()
        ax1.set_aspect('equal')
        ax1.grid(True)
        ax2.plot(data[:, 0], cross_track, 'r-', linewidth=1.2)
        ax2.axhline(0, color='k', linestyle='--', linewidth=0.8)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Cross-track error (m)")
        ax2.set_title("Cross-track error")
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig("log_controller_demo.png", dpi=150)
        plt.close(fig)
        print("Saved log_controller_demo.png")

def main():
    sim = BulldozerSimulation()
    # sim.plot_track_force_sweep()
    # sim.find_straight_threshold()
    sim.demo_log_controller()
    # sim.run_and_plot(stop_time=2.0, first_frame_only=False)
    

if __name__ == "__main__":
    main()
