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
        self.stop_time      = 30

        m                  = 10156.0 /4 # scaled down by Sam
        self.F_track_base  = 60000.0
        self.F_track       = np.array([self.F_track_base, 29894])
        h                  = 2.762 /3   # scaled down by Sam
        self.l             = 2.349 /1.5 # scaled down by Sam
        self.b             = 1.75  /1.5 # scaled down by Sam
        self.B1            = 2.921
        
        self.h  = h
        self.H  = 0.955
        self.L  = 1.2
        grav    = 9.81
        self.mu_l  = 0.1
        self.mu_t  = 0.9
        self.mu_ss = 0.5
        self.kb    = 0.734e6
        self.dt    = 1/100
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
        self.Kp      = -3.0
        self.Kp_path = 8000.0   # N/m  — track-force gain for cross-track error

        # ───────────────── Initial Conditions ─────────────────
        self.desired_depth = -0.4
        self.desired_abg   = np.array([ 0.0, 0, 0.000])
        self.surface_abg   = np.array([ 0.0, 0.0,  0.00])
        self.x_ICR_dot     = 0.0
        self.bld_ang       = np.zeros(3)
        self.dxyz          = np.zeros(3)
        self.daBg          = np.zeros(3)

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

        self.cross_track_err  = 0.0
        self.heading_err      = 0.0
        self.path_points      = self.figure8_path(A=5.0, B=2.5)

        self.Fb       = 0.0
        self.Mb       = 0.0
        self.Rl       = np.zeros(2)
        self.Fy       = 0.0
        self.Mr       = 0.0
        self.vtL      = 0.0
        self.vtR      = 0.0

        # Bezier-6-pinned angular controller (set via use_bezier_controller())
        self._bezier_coeffs   = None
        self._bezier_ang_max  = None
        self._lookahead_dist  = 1
        self._nearest_path_idx = 0

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
    
    def safe_division_x_icr(self, eps: float = 5e-2):
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

    # ───────────────── Figure-8 Path & Tracker ─────────────────
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

    def path_controller(self):
        """Proportional controller: differential track forces to reduce cross-track error.

        Sign convention: left_normal points left of the path direction.
        e > 0  → vehicle is left  → increase F_left  → turn right toward path.
        e < 0  → vehicle is right → increase F_right → turn left toward path.
        """
        self.cross_track_err = self.signed_cross_track_error(self.q[:2])
        delta         = self.Kp_path * self.cross_track_err
        F_max         = 2* self.F_track_base
        self.F_track[0] = float(np.clip(self.F_track_base + delta, 0.0, F_max))
        self.F_track[1] = float(np.clip(self.F_track_base - delta, 0.0, F_max))

    def use_bezier_controller(self, coeffs, ang_max, lookahead_dist=1):
        self._bezier_coeffs   = np.asarray(coeffs)
        self._bezier_ang_max  = float(ang_max)
        self._lookahead_dist  = float(lookahead_dist)

    def _eval_bezier6(self, t):
        from math import comb as _c
        return sum(_c(6, i) * t**i * (1-t)**(6-i) * self._bezier_coeffs[i]
                   for i in range(7))

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

    def angular_path_controller(self):
        """Assign track forces via Bezier-6-pinned lookup on pure-pursuit heading error."""
        self.heading_err     = self.pure_pursuit_heading_error()
        self.cross_track_err = self.signed_cross_track_error(self.q[:2])
        ang      = min(abs(self.heading_err), self._bezier_ang_max)
        t        = ang / self._bezier_ang_max
        fraction = float(np.clip(self._eval_bezier6(t), 0.0, 1.0))
        if self.heading_err > 0:          # need to turn left  → weaken left track
            self.F_track[0] = fraction * self.F_track_base
            self.F_track[1] = self.F_track_base
        else:                              # need to turn right → weaken right track
            self.F_track[0] = self.F_track_base
            self.F_track[1] = fraction * self.F_track_base

    # ───────────────── Main Integration Loop ─────────────────
    def run(self):
        t = 0.0
        n_pts   = len(self.path_points)
        max_idx = 0
        for _ in range(int(self.stop_time / self.dt)):
            if self._bezier_coeffs is not None:
                self.angular_path_controller()
                max_idx = max(max_idx, self._nearest_path_idx)
                if max_idx > n_pts * 0.9 and self._nearest_path_idx < n_pts * 0.1:
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

            self.log.append([t, *self.q, self.cross_track_err, self.heading_err])
            t += self.dt
            
def build_bezier6_pinned(threshold=0.498, n_samples=60):
    """Fit Bezier-6 pinned curve and return (coeffs, ang_max)."""
    from math import comb as _comb

    fractions = np.linspace(0.0, threshold, n_samples)
    yaws = []
    for frac in fractions:
        sim = BulldozerSimulation()
        sim.stop_time  = 0.5          # short calibration run, constant forces
        sim.F_track[0] = sim.F_track_base
        sim.F_track[1] = frac * sim.F_track_base
        sim.run()
        yaws.append(np.array(sim.log)[-1, 6])
    yaws    = np.array(yaws)
    ang     = np.abs(yaws)
    ang_max = ang.max()
    t       = ang / ang_max

    B = np.zeros((len(t), 7))
    for i in range(7):
        B[:, i] = _comb(6, i) * t**i * (1 - t)**(6 - i)

    rhs   = fractions - threshold * B[:, 0]
    c_int, _, _, _ = np.linalg.lstsq(B[:, 1:-1], rhs, rcond=None)
    coeffs = np.concatenate([[threshold], c_int, [0.0]])
    print(f"Bezier-6 pinned built  ang_max={ang_max:.4f} rad  "
          f"coeffs={np.array2string(coeffs, precision=4)}")
    return coeffs, ang_max


def demo_angular_controller(stop_time=30.0, lookahead_dist=1.0):
    """Run figure-8 with pure-pursuit + Bezier-6-pinned torque lookup."""
    print("Building Bezier-6 pinned lookup table (600 calibration sims)...")
    coeffs, ang_max = build_bezier6_pinned()

    print(f"Running figure-8 for {stop_time}s  lookahead={lookahead_dist}m ...")
    sim = BulldozerSimulation()
    sim.stop_time = stop_time
    sim.use_bezier_controller(coeffs, ang_max, lookahead_dist=lookahead_dist)
    sim.run()

    data         = np.array(sim.log)
    cross_track  = data[:, 7]
    heading_err  = data[:, 8]

    def _stats(arr):
        rmse = np.sqrt(np.mean(arr**2))
        mae  = np.mean(np.abs(arr))
        mx   = np.max(np.abs(arr))
        return rmse, mae, mx

    ct_rmse, ct_mae, ct_max   = _stats(cross_track)
    he_rmse, he_mae, he_max   = _stats(np.degrees(heading_err))

    print(f"\n{'':>16}  {'RMSE':>10}  {'MAE':>10}  {'Max |err|':>10}")
    print("-" * 52)
    print(f"{'Cross-track (m)':>16}  {ct_rmse:>10.4f}  {ct_mae:>10.4f}  {ct_max:>10.4f}")
    print(f"{'Heading (deg)':>16}  {he_rmse:>10.4f}  {he_mae:>10.4f}  {he_max:>10.4f}")

    print("Rendering GIF...")
    data_full = np.array(sim.log)
    data      = data_full[::5]          # 5x downsample → real-time at 20 fps

    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(111, projection='3d')

    # Ground plane (surface_abg = 0 so it's flat at z = 0)
    margin = 2.0
    cx   = (data[:, 1].max() + data[:, 1].min()) / 2
    cy   = (data[:, 2].max() + data[:, 2].min()) / 2
    cz   = sim.h / 2
    half = max(data[:, 1].max() - data[:, 1].min(),
               data[:, 2].max() - data[:, 2].min(),
               sim.h) / 2 + margin
    xs = np.linspace(cx - half, cx + half, 20)
    ys = np.linspace(cy - half, cy + half, 20)
    Xs, Ys = np.meshgrid(xs, ys)
    ax.plot_surface(Xs, Ys, np.zeros_like(Xs), alpha=0.25, color='tan', zorder=0)

    # Figure-8 reference path on the ground
    ax.plot(sim.path_points[:, 0], sim.path_points[:, 1],
            np.zeros(len(sim.path_points)),
            'g--', linewidth=1.2, alpha=0.7)

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Box corners in local frame (origin = bottom centre)
    hl, hb = sim.l / 2, sim.b / 2
    c_local = np.array([
        [-hl, -hb,      0],   # 0 bottom rear-left
        [+hl, -hb,      0],   # 1 bottom front-left
        [+hl, +hb,      0],   # 2 bottom front-right
        [-hl, +hb,      0],   # 3 bottom rear-right
        [-hl, -hb, sim.h],   # 4 top rear-left
        [+hl, -hb, sim.h],   # 5 top front-left
        [+hl, +hb, sim.h],   # 6 top front-right
        [-hl, +hb, sim.h],   # 7 top rear-right
    ])
    box_edges = [(0,1),(1,2),(2,3),(3,0),
                 (4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
    box_lines = [None] * 12

    trail, = ax.plot([], [], [], 'b-', linewidth=1.2)

    def update(i):
        trail.set_data(data[:i+1, 1], data[:i+1, 2])
        trail.set_3d_properties(data[:i+1, 3])

        for line in box_lines:
            if line is not None:
                line.remove()

        R   = sim.rotation_lg(data[i, 4], data[i, 5], data[i, 6])
        pos = data[i, 1:4]
        c_g = pos + (R @ c_local.T).T

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
    anim.save("angular_controller_demo.gif", writer=animation.PillowWriter(fps=20))
    plt.close(fig)
    print("Saved angular_controller_demo.gif")


def main():
    demo_angular_controller()


if __name__ == "__main__":
    main()
