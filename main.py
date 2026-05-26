"""
Python port of the 3D bulldozer blade control simulation.
Replaces the Simulink sim('simulation_3d') call with an explicit
forward-Euler integration loop.
"""
from xml.parsers.expat import errors
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
matplotlib.use("Agg")   # headless; remove if running interactively

os.makedirs("figures", exist_ok=True)

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
        m                            = 10156.0 
        self.h                       = 2.762/2 # scaled down by Sam
        self.l                       = 2.349  
        self.b                       = 1.75
        self.velocity_limit          = 2.222
        self.laterial_velocity_limit = 0.0     # this governs how much the dozer can "slide" laterally
        self.angular_velocity_limit  = 2 * self.velocity_limit / self.b
        self.F_track_base            = 600000.0
        self.F_track                 = np.array([self.F_track_base, 0])

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
        self.dxyz               = np.zeros(3)
        self.daBg               = np.zeros(3)
        self.bld_ang            = np.zeros(3)  # [roll, pitch, yaw]
        self.q                  = np.array([0.0, 0.0, 0.0, self.surface_abg[0], self.surface_abg[1], self.surface_abg[2]])
        self.q_dot              = np.zeros(6)
        self.v                  = np.zeros(2)
        self.R_lg               = self.rotation_lg(self.q[3], self.q[4], self.q[5])
        _, self.J_lg            = self.rotation_derivatives(self.q[3], self.q[4])
        self.x_ICR              = 0.0
        self.x_ICR_dot          = 0.0
        self.Fb                 = 0.0
        self.Mb                 = 0.0
        self.Rl                 = np.zeros(2)
        self.Fy                 = 0.0
        self.Mr                 = 0.0
        self.vtL                = 0.0
        self.vtR                = 0.0
        self.v_dot              = np.zeros(2)
        self.log                = []
        self.cross_track_err    = 0.0
        self.heading_err        = 0.0
        self._4pl_coeffs        = np.array([0.1499756, 1.87300871, 3.47736035, 5.59836133])
        self._4pl_knots         = (1.0593712690788024, 1.1706408294732875, 1.2077716115263897)
        self._4pl_threshold     = 0.799
        self._4pl_ang_max       = 1.2256907107093555
        self._use_path_controller = False
        self._lookahead_dist    = 1.5
        self._nearest_path_idx  = 0
        self.path_points        = self.figure8_path(A=5.0, B=2.5)

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
        """Dense (x, y, z) waypoints for a figure-8 on the surface plane defined by surface_abg."""
        R_surf = self.rotation_lg(*self.surface_abg)
        e1, e2 = R_surf[:, 0], R_surf[:, 1]
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        xs, ys = A * np.sin(t), B * np.sin(2 * t)
        return np.outer(xs, e1) + np.outer(ys, e2)

    def signed_cross_track_error(self, pos_xyz):
        """Signed perpendicular distance from pos_xyz to self.path_points on the surface plane.
        Positive when the vehicle is to the left of the path tangent direction."""
        diffs = self.path_points - pos_xyz
        idx      = int(np.argmin(np.linalg.norm(diffs, axis=1)))
        next_idx = (idx + 1) % len(self.path_points)
        tangent  = self.path_points[next_idx] - self.path_points[idx]
        norm     = np.linalg.norm(tangent)
        if norm < 1e-10:
            return 0.0
        tangent /= norm
        n_surf      = self.rotation_lg(*self.surface_abg)[:, 2]
        left_normal = np.cross(n_surf, tangent)  # left of tangent within surface plane
        return float(np.dot(pos_xyz - self.path_points[idx], left_normal))

    def path_controller(self):
        """Proportional controller: differential track forces to reduce cross-track error.

        Sign convention: left_normal points left of the path direction.
        e > 0  → vehicle is left  → increase F_left  → turn right toward path.
        e < 0  → vehicle is right → increase F_right → turn left toward path.
        """
        self.cross_track_err = self.signed_cross_track_error(self.q[:3])
        delta         = self.Kp_path * self.cross_track_err
        F_max         = 2* self.F_track_base
        self.F_track[0] = float(np.clip(self.F_track_base + delta, 0.0, F_max))
        self.F_track[1] = float(np.clip(self.F_track_base - delta, 0.0, F_max))

    def _eval_4pl(self, ang):
        k1, k2, k3 = self._4pl_knots
        c = self._4pl_coeffs
        T = c[0]*ang + c[1]*max(ang-k1, 0) + c[2]*max(ang-k2, 0) + c[3]*max(ang-k3, 0)
        return self._4pl_threshold - T

    def pure_pursuit_heading_error(self):
        """
        Pure-pursuit: find the lookahead point on the path at distance
        self._lookahead_dist from the vehicle, return the signed angle
        from current heading to that point, measured within the surface plane.
        """
        pos = self.q[:3]
        L   = self._lookahead_dist
        n   = len(self.path_points)

        dists      = np.linalg.norm(self.path_points - pos, axis=1)
        nearest    = int(np.argmin(dists))
        self._nearest_path_idx = nearest
        lookahead_pt = None

        # Walk forward along path segments looking for sphere intersection
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

        R_surf  = self.rotation_lg(*self.surface_abg)
        n_surf  = R_surf[:, 2]
        e1_surf = R_surf[:, 0]
        e2_surf = R_surf[:, 1]
        d_vec   = lookahead_pt - pos
        d_proj  = d_vec - np.dot(d_vec, n_surf) * n_surf   # project onto surface plane
        angle   = np.arctan2(np.dot(d_proj, e2_surf), np.dot(d_proj, e1_surf))
        err     = angle - self.q[5]
        return float((err + np.pi) % (2 * np.pi) - np.pi)

    def angular_path_controller(self):
        """Assign track forces via 4PL lookup on pure-pursuit heading error."""
        self.heading_err     = self.pure_pursuit_heading_error()
        self.cross_track_err = self.signed_cross_track_error(self.q[:3])
        ang      = min(abs(self.heading_err), self._4pl_ang_max)
        fraction = float(np.clip(self._eval_4pl(ang), 0.0, 1.0))
        if self.heading_err > 0:          # need to turn left  → weaken left track
            self.F_track[0] = fraction * self.F_track_base
            self.F_track[1] = self.F_track_base
        else:                              # need to turn right → weaken right track
            self.F_track[0] = self.F_track_base
            self.F_track[1] = fraction * self.F_track_base

    # ---------------- Main Loop ----------------
    def run(self):
        t = 0.0
        n_pts   = len(self.path_points)
        max_idx = 0
        for _ in range(int(self.stop_time / self.dt)):
            if self._use_path_controller:
                self.angular_path_controller()
                max_idx = max(max_idx, self._nearest_path_idx)
                if max_idx > n_pts * 0.9 and self._nearest_path_idx < n_pts * 0.1:
                    break
            
            errors, _     = self.controller_errors()
            self.bld_ang += self.gain * self.Kp * errors

            self.v_dot = self.vehicle_dynamics()
            self.v    += self.dt * self.v_dot
            self.v[0]  = max(min(self.v[0], self.velocity_limit), 0)
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
    @staticmethod
    def _plot_forces(log_arr, fname="figures/forces.png"):
        t     = log_arr[:, 0]
        Mb    = log_arr[:, 9]
        Fb    = log_arr[:, 10]
        RlL   = log_arr[:, 11]
        RlR   = log_arr[:, 12]
        Fy    = log_arr[:, 13]
        Mr    = log_arr[:, 14]
        v_fwd = log_arr[:, 15]
        v_trn = log_arr[:, 16]

        fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
        fig.suptitle("Forces & Moments over Time")

        axes[0, 0].plot(t, Fb);  axes[0, 0].set_ylabel("Fb (N)");   axes[0, 0].set_title("Blade force")
        axes[0, 1].plot(t, Mb);  axes[0, 1].set_ylabel("Mb (N·m)"); axes[0, 1].set_title("Blade moment")
        axes[1, 0].plot(t, RlL, label="Left"); axes[1, 0].plot(t, RlR, label="Right")
        axes[1, 0].set_ylabel("Rl (N)"); axes[1, 0].set_title("Track rolling resistance"); axes[1, 0].legend()
        axes[1, 1].plot(t, Fy);  axes[1, 1].set_ylabel("Fy (N)");   axes[1, 1].set_title("Lateral track force")
        axes[2, 0].plot(t, Mr);  axes[2, 0].set_ylabel("Mr (N·m)"); axes[2, 0].set_title("Track turning moment")
        axes[2, 1].plot(t, v_fwd, label="forward"); axes[2, 1].plot(t, v_trn, label="turn")
        axes[2, 1].set_ylabel("v (m/s  or  rad/s)"); axes[2, 1].set_title("v"); axes[2, 1].legend()

        for ax in axes.flat:
            ax.set_xlabel("Time (s)")
            ax.grid(True, linewidth=0.4)

        fig.tight_layout()
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        print(f"Saved {fname}")

    def run_and_plot(self, stop_time=30.0, lookahead_dist=0.9,
                     use_path_controller=True, first_frame_only=False):
        """Run the simulation and render a multi-panel GIF.

        use_path_controller=True  — build Bezier lookup, run pure-pursuit figure-8, overlay path.
        use_path_controller=False — run with fixed track forces, no path overlay.
        """
        sim = BulldozerSimulation()
        sim.stop_time = stop_time

        if use_path_controller:
            print(f"Running figure-8 for {stop_time}s  lookahead={lookahead_dist}m ...")
            sim._lookahead_dist      = float(lookahead_dist)
            sim._use_path_controller = True
        sim.run()

        if use_path_controller:
            log_arr     = np.array(sim.log)
            cross_track = log_arr[:, 7]
            heading_err = log_arr[:, 8]

            def _stats(arr):
                return (np.sqrt(np.mean(arr**2)), np.mean(np.abs(arr)), np.max(np.abs(arr)))

            ct_rmse, ct_mae, ct_max = _stats(cross_track)
            he_rmse, he_mae, he_max = _stats(np.degrees(heading_err))
            print(f"\n{'':>16}  {'RMSE':>10}  {'MAE':>10}  {'Max |err|':>10}")
            print("-" * 52)
            print(f"{'Cross-track (m)':>16}  {ct_rmse:>10.4f}  {ct_mae:>10.4f}  {ct_max:>10.4f}")
            print(f"{'Heading (deg)':>16}  {he_rmse:>10.4f}  {he_mae:>10.4f}  {he_max:>10.4f}")

        print("Rendering GIF...")
        data     = np.array(sim.log)[::5]
        n_frames = 1 if first_frame_only else len(data)

        fig = plt.figure(figsize=(14, 7))
        gs  = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], hspace=0.35, wspace=0.3)
        ax      = fig.add_subplot(gs[:, 0], projection='3d')
        ax_top  = fig.add_subplot(gs[0, 1])
        ax_side = fig.add_subplot(gs[1, 1])

        # Surface plane normal
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

        if use_path_controller:
            ax.plot(sim.path_points[:, 0], sim.path_points[:, 1], sim.path_points[:, 2],
                    'g--', linewidth=1.2, alpha=0.7)
            ax_top.plot(sim.path_points[:, 0], sim.path_points[:, 1],
                        'g--', linewidth=1.2, alpha=0.7)
            ax_side.plot(sim.path_points[:, 0], sim.path_points[:, 2],
                         'g--', linewidth=1.2, alpha=0.7)

        ax_top.set_xlim(cx - half, cx + half)
        ax_top.set_ylim(cy - half, cy + half)
        ax_top.set_xlabel("X (m)")
        ax_top.set_ylabel("Y (m)")
        ax_top.set_title("Top View (X-Y)")
        ax_top.set_aspect('equal', adjustable='box')
        ax_top.grid(True, linewidth=0.4)

        ax_side.set_xlim(cx - half, cx + half)
        ax_side.set_ylim(cz - half, cz + half)
        ax_side.set_xlabel("X (m)")
        ax_side.set_ylabel("Z (m)")
        ax_side.set_title("Side View (X-Z)")
        ax_side.set_aspect('equal', adjustable='box')
        ax_side.grid(True, linewidth=0.4)

        x_line = np.array([cx - half, cx + half])
        ax_top.axhline(cy, color='tan', linewidth=2, alpha=0.7)
        ax_side.plot(x_line, -(nx * x_line + ny * cy) / nz, color='tan', linewidth=2, alpha=0.7)

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
        ax_top.scatter(data[0, 1], data[0, 2], color='blue', s=60, zorder=5)
        ax_side.scatter(data[0, 1], data[0, 3], color='blue', s=60, zorder=5)

        trail,      = ax.plot([], [], [], 'b-', linewidth=1.5)
        trail_top,  = ax_top.plot([], [], 'b-', linewidth=1.5)
        trail_side, = ax_side.plot([], [], 'b-', linewidth=1.5)

        def update(i):
            trail.set_data(data[:i+1, 1], data[:i+1, 2])
            trail.set_3d_properties(data[:i+1, 3])
            trail_top.set_data(data[:i+1, 1], data[:i+1, 2])
            trail_side.set_data(data[:i+1, 1], data[:i+1, 3])

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
                box_lines_top[j], = ax_top.plot(
                    [p1[0], p2[0]], [p1[1], p2[1]],
                    color='red', linewidth=1.5)
                box_lines_side[j], = ax_side.plot(
                    [p1[0], p2[0]], [p1[2], p2[2]],
                    color='red', linewidth=1.5)

            ax.set_title(f"t = {data[i, 0]:.2f} s")
            return trail,

        anim = animation.FuncAnimation(
            fig, update, frames=n_frames, blit=False, interval=50
        )
        fname = "figures/simulation.gif"
        anim.save(fname, writer=animation.PillowWriter(fps=20))
        plt.close(fig)
        print(f"Saved {fname}")
        self._plot_forces(np.array(sim.log))


def main():
    sim = BulldozerSimulation()
    sim.run_and_plot(lookahead_dist=0.8)


if __name__ == "__main__":
    main()
