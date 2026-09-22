from __future__ import annotations
from typing import Callable, Optional, Sequence
from numpy.typing import ArrayLike
from sim_types import TerrainGrid, ForceSample, PileState, BladeFrame, ContactMap, FloatArray, GridCell, Point3, Scalar, VectorLike

import numpy as np
import networkx as nx

from controllers import Control
from visualization import Visualization


class DozerSimulation():
    """Surface-aware bulldozer with an optional deforming blade."""

    def __init__(self, is_uphill: bool = True, is_surface_pitched: bool = False, is_surface_rolled: bool = False,
                 is_backwards: bool = False, enable_blade: bool = True, blade_roll_pitch_yaw: FloatArray = np.zeros(3),
                 controller_type: str = "pd", enable_blade_control: bool = True,
                 use_path_controller: bool = False, path_points: Optional[ArrayLike]=None, lookahead_dist: float=1.5) -> None:
        #TODO: add comments with parameter descriptions and units (maybe change names to be more descriptive?)
        # ----------------------------- Load passed parameters -----------------------------
        self.is_surface_pitched: bool   = is_surface_pitched
        self.is_surface_rolled: bool    = is_surface_rolled
        self.is_backwards: bool         = is_backwards
        self.enable_blade: bool         = enable_blade
        self.enable_blade_control: bool = enable_blade_control
        self.use_path_controller: bool = use_path_controller
        self.blade_roll_pitch_yaw: FloatArray = blade_roll_pitch_yaw

        # ------------------------- General simulation parameters --------------------------
        self.dt: float             = 1/100
        gravity             = 9.81
        self.total_distance: Scalar = 0.0
        self.fill_progress_distance: Scalar = 0.0
        self.spill_reference_seconds: float = 1.0  # positive duration over which spill_factor is retained

        # -------------------- Independent simulation limit parameters ---------------------
        # change if tasks require move total distance to be traveled, still want it tight for fast tuning
        self.stop_time: float              = 15
        # contact-angle root-find tolerance (rad of angle / m of hang). Must stay
        # tighter than the settle loop's 1e-6 convergence check, or the pose it
        # returns is noisier than that check and the settle loop never converges
        # -- which burns *more* passes, so loosening this is a net slowdown.
        self.contact_tol: float            = 1e-7
        self.fill_distance: float          = 8.0    # maximum distance for soil fill calculation TODO: should be computed relative to depth of cut
        self.velocity_limit: float         = 2.222
        self.lateral_velocity_limit: float = 0.0    # this governs how much the dozer can "slide" laterally
        self.blade_roll_pitch_yaw_limits: FloatArray      = np.array([0.0735, 0.430, 0.387]) # radians
        zero_to_max_angle_time = 1              # second
        self.blade_roll_pitch_yaw_rate_limits: FloatArray = self.blade_roll_pitch_yaw_limits / zero_to_max_angle_time # radians per second

        # --------------------------- Bulldozer body parameters ----------------------------
        mass              = 10156.0 # of the unloaded vehicle in kilograms
        self.b: float            = 1.75    # width between track centers (track guage) in meters
        self.l: float            = 2.349   # track length in meters
        self.h: float            = 2.762/2 # body height in meters, scaled down by Sam
        self.F_track_base: float = 600000.0
        self.track_width: float  = 0.7112
        self.track_height: float = 0.5

        # -------------------------- Bulldozer blade parameters ----------------------------
        self.B1: float = 2.921 # blade width in meters
        self.H: float  = 0.955 # blade height in meters
        self.L: float  = 1.2   # blade arm length in meters
        self.blade_arm_offset: FloatArray = np.array([0.5, 0, 0])

        # -------------------------------- Soil parameters ---------------------------------
        self.mu_l: float    = 0.1              # longitudinal friction coefficient
        self.mu_t: float    = 0.9              # lateral      friction coefficient
        self.mu_ss: float   = 0.5              # shear        friction coefficient
        self.kb: float      = 0.734e6          # blade soil interaction constant
        self.beta0: Scalar   = np.radians(38.0) # soil accumulation angle in radians
        self.gamma_g: float = 1640 * gravity   # soil weight per cubic meter

        # ------------------------------- Surface parameters -------------------------------
        self.division_factor: int = 4
        self.surface_abg: FloatArray     = np.array([ 0.0, 0.0, 0.0])
        self.u_split: float         = 20  # u-value where the grid switches to surface_abg2
        self.v_split: float         = 20  # u-value where the grid switches to surface_abg2

        # ----- Set up Center Of Mass (COM) position, COM velocity, simulation surface -----
        self.u_range: tuple[float, float] = (-1* self.b/2, 8 * self.b) if is_surface_pitched else (-self.b/2, 4*    self.b)
        self.v_range: tuple[float, float] = (-self.b/2,   self.b/2) if is_surface_pitched else (-1* self.b/2, 8 * self.b)
        reference_rotation = self._rotation_lg(*self.surface_abg)
        reference_path = (Control.figure8_path(surface_rotation=reference_rotation)
                          if path_points is None else np.asarray(path_points, dtype=float))
        # Set up the controller before the grid so reference bounds are available.
        self.controller: Control = Control(self.L, self.dt, zero_to_max_angle_time, controller_type)
        self.controller.configure_path(reference_path, reference_rotation, lookahead_dist)
        if use_path_controller:
            # Fit the route itself instead of retaining the straight-run bounds.
            # Allow the blade and tracks to turn anywhere on the route, plus 0.5 m.
            blade_radius = np.hypot(self.L + abs(self.blade_arm_offset[0]),
                                    self.B1 / 2 + abs(self.blade_arm_offset[1]))
            track_radius = np.hypot(self.l / 2, (self.b + self.track_width) / 2)
            margin = max(blade_radius, track_radius) + 0.5
            path_xy = self.controller.path_points[:, :2]
            lower = path_xy.min(axis=0) - margin
            upper = path_xy.max(axis=0) + margin
            self.u_range = (float(lower[0]), float(upper[0]))
            self.v_range = (float(lower[1]), float(upper[1]))
        self.us: FloatArray           = np.arange(self.u_range[0], self.u_range[1] + self.subdivision, self.subdivision)
        self.vs: FloatArray           = np.arange(self.v_range[0], self.v_range[1] + self.subdivision, self.subdivision)

        self.offset: FloatArray             = np.array([0, 0, self.b]) if is_uphill else np.array([0, 0, -self.b])
        self.transition_tiles: float   = 1/2 *self.b * self.division_factor # tiles over which the offset ramps down past u_split

        self.surf_grid: nx.Graph = self._surface_grid()

        # node (x, y, z) cached as plain Python floats, indexed [i][j], so the
        # hot height lookups skip the networkx attribute dicts entirely.
        self.grid_pts: TerrainGrid = [[(self.surf_grid.nodes[(i, j)]['x'],
                           self.surf_grid.nodes[(i, j)]['y'],
                           self.surf_grid.nodes[(i, j)]['z'])
                          for j in range(len(self.vs))] for i in range(len(self.us))]
        self.starting_grid_heights: FloatArray = self._grid_heights().copy()

        # --------------------- Dependent simulation limit parameters ----------------------
        self.max_world_cut_depth: float    = self.H
        stop_index                  = 0 if self.is_backwards else -1
        self.stop_distance: Scalar = abs(self.us[stop_index]) if is_surface_pitched else abs(self.vs[stop_index])
        if use_path_controller:
            self.stop_distance = np.inf  # Stop on time or completing the closed path.
        self.angular_velocity_limit: float = 2 * self.velocity_limit / self.b

        # ------------------------ Forces and moment Parameters ---------------------------
        self.F_track: FloatArray = np.array([self.F_track_base, self.F_track_base])  #[left, right]
        self.rl: float      = self.mu_l * mass * gravity / 2      # longitudinal track force limit
        self.fy: float      = self.mu_t * mass * gravity / self.l # lateral force limit
        Ix           = mass * (self.b**2 + self.h**2) / 12 # moment of inertia about x-axis
        Iy           = mass * (self.h**2 + self.l**2) / 12 # moment of inertia about y-axis
        Iz           = mass * (self.b**2 + self.l**2) / 12 # moment of inertia about z-axis
        self.M: FloatArray       = np.diag([mass, mass, mass, Ix, Iy, Iz])             # mass matrix
        self.P: FloatArray       = np.array([0.0, 0.0, mass * gravity, 0.0, 0.0, 0.0]) # gravitational force vector
        self.elim: FloatArray    = np.diag([1.0, 1.0, 1.0, 0.0, 0.0, 1.0])

        # ------- Velocity and rotation + corresponding derivative vector variables --------
        self.v: FloatArray     = np.zeros(2)  # initial local velocity vector
        self.v_dot: FloatArray = np.zeros(2)  # initial local acceleration vector
        self.q: FloatArray     = np.array([0.0, 0.0, 0.0, self.surface_abg[0], self.surface_abg[1],
                               0.0 if is_surface_pitched else np.pi / 2])

        self.q_dot: FloatArray = np.zeros(6)
        self.dxyz: FloatArray  = np.zeros(3)
        self.daBg: FloatArray  = np.zeros(3)
        self.Rl: FloatArray    = np.zeros(2)
        self.R_lg: FloatArray  = self._rotation_lg(*self.q[3:6])
        _, self.J_lg = self._rotation_derivatives()

        # ----------------- Other kinematics and dynamic scalar variables ------------------
        self.x_ICR: Scalar     = 0.0
        self.x_ICR_dot: Scalar = 0.0
        self.Fb: Scalar        = 0.0
        self.cut_limit_reached: bool = False
        self.Mb: Scalar        = 0.0
        self.H3: Scalar        = 0.0
        self.H4: Scalar        = 0.0
        self.H3_sub: Scalar    = 0.0
        self.H4_sub: Scalar    = 0.0
        self.Fy: Scalar        = 0.0
        self.Mr: Scalar        = 0.0
        self.vtL: Scalar       = 0.0
        self.vtR: Scalar       = 0.0

        self.requested_blade_rates: FloatArray = np.zeros(3)  # roll, pitch, yaw before limiting (rad/s)

        # ----------------------- Write 1st entry to simulation logs -----------------------
        self.log: list[list[Scalar]]          = []
        self.blade_log: list[BladeFrame]    = []
        self.pile_log: list[PileState]     = []  # H3, H4, blade-local roll and yaw; geometry is rendered later.
        self.neighbor_log: list[FloatArray] = []
        self.grid_log: list[FloatArray]     = []
        self.force_log: list[ForceSample]    = []
        self.blade_depth_rmse: Optional[float] = None

        self.is_initialized: bool = False
        neighbor_points     = self._body_update()


        self.log.append([0, *self.q])
        self._log_forces(0.0)
        self.grid_log.append(self._grid_heights())

        # seed the blade points the same way, so blade_log stays aligned frame
        # for frame with log/neighbor_log/grid_log
        blade_points, blade_neighbors = self._blade_update(deform=False)
        if self.enable_blade:
            neighbor_points.extend(point for neighbors in blade_neighbors for point in neighbors)
        self.neighbor_log.append(np.array(neighbor_points))
        self.blade_log.append(blade_points)
        self.pile_log.append((self.H3, self.H4, float(self.blade_roll_pitch_yaw[0]),
                              float(self.blade_roll_pitch_yaw[2])))

    # ---------------------------- SURFACE GENERATION AND INFO ----------------------------
    @property
    def subdivision(self) -> float:
        """Grid spacing, sized relative to the dozer width ``self.b``."""
        return self.b / self.division_factor

    def _surface_grid(self) -> nx.Graph:
        R_surf = self._rotation_lg(*self.surface_abg)
        e1, e2, e3 = R_surf[:, 0], R_surf[:, 1], R_surf[:, 2]
        grid = nx.grid_2d_graph(len(self.us), len(self.vs))
        u_start = self.us[self.us <= self.u_split][-1]
        v_start = self.vs[self.vs <= self.v_split][-1]
        ramp_width = max(1, round(self.transition_tiles)) * self.subdivision

        for i, u in enumerate(self.us):
            for j, v in enumerate(self.vs):
                # Full offset before each split, followed by a linear ramp.
                u_clip = np.clip((u - u_start) / ramp_width, 0.0, 1.0)
                v_clip = np.clip((v - v_start) / ramp_width, 0.0, 1.0)

                weight = u_clip if self.is_surface_pitched else v_clip
                if self.is_surface_rolled:
                    weight += v_clip if self.is_surface_pitched else u_clip

                x, y, z = u * e1 + v * e2 + weight * self.offset * e3

                node = grid.nodes[(i, j)]
                node["x"], node["y"], node["z"] = float(x), float(y), float(z)
        return grid

    def _grid_heights(self) -> FloatArray:
        """Snapshot of every node height, indexed [i][j], for the GIF's surface."""
        return np.array([[pt[2] for pt in col] for col in self.grid_pts])

    def _get_neighbor_points(self, point: VectorLike) -> list[Point3]:
        i, j = self._grid_cell(point)
        col_i, col_i1 = self.grid_pts[i], self.grid_pts[i + 1]
        return [col_i[j], col_i1[j], col_i[j + 1], col_i1[j + 1]]

    # ------------------------------------- ROTATION --------------------------------------
    def _rotation_lg(self, a: Scalar, B: Scalar, g: Scalar) -> FloatArray:
        """Rotation matrix: local to global frame."""
        return self._rotation_gl(a, B, g).T

    def _rotation_gl(self, a: Scalar, B: Scalar, g: Scalar) -> FloatArray:
        """Rotation matrix: global to local frame."""
        # TODO: change to accept input array
        sa, ca = np.sin(a), np.cos(a)
        sB, cB = np.sin(B), np.cos(B)
        sg, cg = np.sin(g), np.cos(g)

        return np.array([
            [cB * cg, sa * sB * cg - ca * sg, ca * sB * cg + sa * sg],
            [cB * sg, sa * sB * sg + ca * cg, ca * sB * sg - sa * cg],
            [-sB, sa * cB, ca * cB],
        ]).T

    # ------------------------------------ BODY UPDATE ------------------------------------
    def _body_update(self) -> list[Point3]:
        if self.is_initialized:
            if self.use_path_controller:
                self.F_track = self.controller.angular_path_controller(
                    self.q, self._rotation_lg(*self.q[3:6]), self.F_track_base)
            else:
                self.F_track = self.controller.dummy_track_controlller(self.q)
            self._update_q_dot()
            self.q += self.dt * self.q_dot
            self.q[3:6] = (self.q[3:6] + np.pi) % (2 * np.pi) - np.pi
        else:
            self.is_initialized = True

        #TODO: when optimizing run time for machine learning remove neighbor point tracking
        neighbor_points = self._get_neighbor_points(self.q)
        self._settle_tracks(self.q[3:6].copy())
        return neighbor_points

    # ----------------------------------- BODY DYNAMICS -----------------------------------
    def _update_q_dot(self) -> None:
        """Advance internal velocity state and return generalized velocity."""
        self.R_lg       = self._rotation_lg(*self.q[3:6])
        R_gl            = self.R_lg.T
        J_gl, self.J_lg = self._rotation_derivatives()

        self.dxyz = R_gl @ self.q_dot[:3]
        self.daBg = J_gl @ self.q_dot[3:6]

        self.dxyz[1] = self._saturation(self.dxyz[1], self.lateral_velocity_limit)
        self.x_ICR = self._get_x_icr()

        self._vehicle_dynamics()

        self.v += self.dt * self.v_dot

        if self.is_backwards:
            self.v[0] = np.clip(self.v[0], -self.velocity_limit, 0.0)
        else:
            self.v[0] = np.clip(self.v[0], 0.0, self.velocity_limit)
        self.v[1] = self._saturation(self.v[1], self.angular_velocity_limit)

        self.q_dot = self._S_matrix() @ self.v

    def _rotation_derivatives(self) -> tuple[FloatArray, FloatArray]:
        """Rotation derivative matrices"""
        a = self.q[3]
        B = self.q[4]

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

    def _saturation(self, value: Scalar, limit: Scalar) -> float:
        return float(np.clip(value, -abs(float(limit)), abs(float(limit))))

    def _get_x_icr(self, eps: float = 1e-3) -> float:
        if abs(self.daBg[2]) < eps:
            return 0.0
        return float(np.clip(-self.dxyz[1] / self.daBg[2], -self.l / 2, self.l / 2))

    def _vehicle_dynamics(self) -> None:
        self._track_terrain_interaction()
        self._blade_terrain_interaction()

        B_mat = np.zeros((6, 2))
        B_mat[:3, 0]  =  self.R_lg[:, 0]
        B_mat[:3, 1]  =  self.R_lg[:, 0]
        B_mat[3:6, 0] = -self.R_lg[:, 2] * self.b / 2
        B_mat[3:6, 1] =  self.R_lg[:, 2] * self.b / 2

        Ct_vec = np.array([
            self.Rl.sum(), self.Fy, 0.0, 0.0, 0.0,
            self.Mr + (self.Rl[1] - self.Rl[0]) * self.b / 2,
        ])

        R6 = np.zeros((6, 6))
        R_blade = self._rotation_lg(*self.blade_roll_pitch_yaw)
        R6[:3, :3]   = R_blade
        R6[3:6, 3:6] = R_blade

        blade_vec = np.array([self.Fb, 0.0, 0.0, 0.0, 0.0, self.Mb])
        Cb_vec    = self.elim @ R6 @ blade_vec

        R6_lg = np.zeros((6, 6))
        R6_lg[:3, :3]   = self.R_lg
        R6_lg[3:6, 3:6] = self.R_lg
        C = R6_lg @ (Ct_vec + Cb_vec)

        S  = self._S_matrix()
        Sd = self._Sd_matrix()

        Bt = S.T @ B_mat
        Ct = S.T @ C
        Pt = S.T @ self.P
        Mt = S.T @ self.M @ S
        Et = S.T @ self.M @ Sd

        self.v_dot = np.linalg.solve(Mt, Bt @ self.F_track + Ct - Et @ self.v - Pt)

    def _track_terrain_interaction(self) -> None:
        self.vtL = self._saturation(self.dxyz[0] - self.b / 2 * self.daBg[2],
                                   self.velocity_limit)
        self.vtR = self._saturation(self.dxyz[0] + self.b / 2 * self.daBg[2],
                                   self.velocity_limit)

        FtL, FtR = self.F_track

        RlL     = self._G(FtL, self.rl, self.vtL)
        RlR     = self._G(FtR, self.rl, self.vtR)
        self.Rl = np.array([RlL, RlR])

        self.Fy = -2 * np.sign(self.dxyz[1]) * self.fy * abs(self.x_ICR)

        moment          = ((FtR + RlR) - (FtL + RlL)) * self.b / 2
        moment_friction = 2 * self.fy * ((self.l**2) / 4 - (self.x_ICR**2))
        self.Mr         = self._G(moment, moment_friction, self.daBg[2])

    def _G(self, force: Scalar, friction: Scalar, velocity: Scalar) -> Scalar:
        if abs(float(velocity)) > 1e-10:
            return -friction * np.sign(velocity)
        if abs(float(force)) <= float(friction):
            return -force
        return -friction * np.sign(force)

    def _blade_terrain_interaction(self) -> None:
        # TODO: update this to be surface - body angle + blade angle, to account for non-flat surfaces
        a_rel = self.blade_roll_pitch_yaw[0]

        cut_height = abs(self.L * np.sin(self.blade_roll_pitch_yaw[1]))

        H1     = self.B1 * np.tan(abs(a_rel))
        H2     = cut_height / np.cos(abs(a_rel))
        H3_sub = -H2 + (np.sign(a_rel) * H1 / 2) - (H1 / 2)
        H4_sub = -H2 - (np.sign(a_rel) * H1 / 2) - (H1 / 2)
        H3     = self.H + H3_sub
        H4     = self.H + H4_sub

        a_val = np.tan(abs(a_rel))**2
        c_val = (H3 + H4) / 2

        fill_percent = min(1.0, self.fill_progress_distance / self.fill_distance)

        # TODO: update to account for later dump c_ycles
        self.H3_sub = -H3_sub
        self.H4_sub = -H4_sub

        # TODO: handle case where both H3_sub and H4_sub near zero
        if (H3_sub == 0) and (H4_sub == 0):
            volume = 0.0
            self.H3 = self.H4 = 0.0
        else:
            volume = 0.5 / np.tan(self.beta0) * (a_val * self.B1**3 / 12 + c_val**2 * self.B1)
            self.H3 = max(H3, 0.0) * fill_percent
            self.H4 = max(H4, 0.0) * fill_percent

        soil_weight       = volume * self.gamma_g * fill_percent
        hypotenuse        = self.B1 / np.cos(abs(a_rel))
        cut_area          = 0.5 * self.B1 * H1 + hypotenuse * cut_height
        soil_shear_force  = cut_area    * self.kb
        soil_normal_force = soil_weight * self.mu_ss

        self.Fb = -soil_shear_force - soil_normal_force

        shear_moment_arm  = self._yc(-H3_sub, -H4_sub, self.B1)
        normal_moment_arm = self._yc(     H3,      H4, self.B1)

        self.Mb  = (   soil_shear_force *  shear_moment_arm
                    + soil_normal_force * normal_moment_arm)

        if self.enable_blade and self.cut_limit_reached:
            # The velocity clamp prevents resistance from reversing the vehicle.
            self.Fb = 1e9 if self.is_backwards else -1e9
            self.Mb = 0.0

    def _yc(self, left_depth: Scalar, right_depth: Scalar, width: Scalar) -> Scalar:
        if left_depth == 0 and right_depth == 0:
            return 0.0
        return (
            (2 * left_depth + right_depth) / (3 * (left_depth + right_depth)) * width
            - width / 2
        )

    def _S_matrix(self) -> FloatArray:
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

    def _Sd_matrix(self) -> FloatArray:
        """
        Time derivative of the S matrix.
        """
        a, B, g    = self.q[3:6]
        Ad, Bd, Gd = self.q_dot[3:6]

        sa, ca = np.sin(a), np.cos(a)
        sB, cB = np.sin(B), np.cos(B)
        sg, cg = np.sin(g), np.cos(g)
        tB = np.tan(B)

        R = self.R_lg

        S_11 = -sB * cg * Bd - R[1, 0] * Gd
        S_21 = -sB * sg * Bd + R[0, 0] * Gd
        S_31 = -cB * Bd

        S_12 = (-self.x_ICR * (R[0, 2] * Ad + R[2, 1] * cg * Bd - R[1, 1] * Gd)
                - self.x_ICR_dot * R[0, 1])

        S_22 = (-self.x_ICR * (R[1, 2] * Ad + R[2, 1] * sg * Bd + R[0, 1] * Gd)
                - self.x_ICR_dot * R[1, 1])

        S_32 = (-self.x_ICR * (R[2, 2] * Ad - sa * sB * Bd)
                - self.x_ICR_dot * R[2, 1])

        S_42 = -sa * tB * Ad + ca / (cB**2) * Bd
        S_52 = -ca * Ad
        S_62 = -sa / cB * Ad + ca * tB / cB * Bd

        return np.array([
            [S_11, S_12],
            [S_21, S_22],
            [S_31, S_32],
            [ 0.0, S_42],
            [ 0.0, S_52],
            [ 0.0, S_62],
        ])

    # ------------------------------ TRACK SURFACE CONTACT --------------------------------
    def _settle_tracks(self, orient: FloatArray) -> None:
        """Set the supported pose from tracks and the enabled blade top edge."""
        self.q[3:6] = orient
        for _ in range(20):
            # Balance track/blade support across the body. Yaw is the heading
            # and cannot change during settling.
            new_orient = self.q[3:6].copy()
            new_orient[0] = self._body_contact_roll(new_orient)
            new_orient[1] = self._body_contact_pitch(new_orient)
            new_z = self._resting_height(self._rotation_lg(*new_orient))

            converged = (
                abs(new_z - self.q[2]) < 1e-6
                and np.all(np.abs(new_orient - self.q[3:6]) < 1e-6)
            )
            self.q[2], self.q[3:6] = new_z, new_orient
            if converged:
                break

        # Commit the stop check only for the settled orientation, never a
        # trial angle visited by the contact solver.
        self._support_heights(self._rotation_lg(*self.q[3:6]), check_cut_limit=True)

    def _body_contact_roll(self, orient: FloatArray) -> Scalar:
        """Roll that balances left/right track and blade support."""
        def imbalance(roll: Scalar) -> Scalar:
            z_left, z_right = self._track_resting_heights(
                self._rotation_lg(roll, orient[1], orient[2]))
            return z_left - z_right
        return self._contact_angle(imbalance, orient[0])

    def _track_resting_heights(self, R: FloatArray) -> tuple[float, float]:
        """
        Required support heights on each side of the body, as (left, right).
        Includes the blade when enabled; _body_contact_roll balances the pair.
        """
        support = self._support_heights(R)
        return float(support[1].max()), float(support[0].max())

    def _support_heights(self, R: FloatArray, check_cut_limit: bool=False) -> FloatArray:
        """
        Lowest body height each quarter of the contact patch calls for, as a
        [right, left] x [back, front] array. Tracks use current soil; the enabled
        blade top uses the original soil surface. The body rests at the highest
        of the four, so any quarter asking for less hangs clear of the
        ground unless the pose balances them: _body_contact_roll balances left
        against right, _contact_pitch front against back.

        The halves overlap at the track midpoint, so a body pivoting on a peak
        directly beneath its center reads as balanced from every direction --
        which it is, since no rotation can bring anything else down onto the
        ground without driving that peak through the belly.

        The surface is piecewise planar with kinks only on the grid lines, so a
        straight track's deepest penetration always occurs at an endpoint or a
        grid-line crossing; between those the clearance varies linearly. Sampling
        the ends plus every grid crossing therefore finds the true deepest point
        exactly (a uniform scan would step over the kinks).
        """
        fwd     = R[:, 0]
        lat     = R[:, 1] * (self.b / 2)      # body center -> track lateral offset (local +y)
        half_l  = self.l / 2
        support = np.full((2, 2), -np.inf)    # [right, left] x [back, front]
        for i, side in enumerate((-1.0, 1.0)):   # right (-lat) and left (+lat) tracks
            base  = self.q[:3] + side * lat
            ss    = {-half_l, 0, half_l}         # track ends

            # the track only spans the tiles between its two ends, so search that
            # neighborhood of grid lines instead of the whole grid
            back_cell  = self._grid_cell(base - half_l * fwd)
            front_cell = self._grid_cell(base + half_l * fwd)
            # seach x axis tiles via (self.us, 0) then y axis tiles via (self.vs, 1)
            for grid, axis in ((self.us, 0), (self.vs, 1)):
                # skip if the track runs parallel to this axis' lines (fwd[axis] ~ 0) since it never crosses
                # plus avoid divide by zero error later
                if abs(fwd[axis]) > 1e-12:
                    # grid lines bounding the tiles the ends fall in (+1 stop, +1 for the far tile's upper line)
                    lo, hi = sorted((back_cell[axis], front_cell[axis]))
                    #TODO: could super sample for smaller grid sizes so that surfaces difference above a certain size
                    # are treated as "real" surface differnces
                    near   = grid[lo:hi + 2]
                    # solve grid = base[axis] + s*fwd[axis] for every nearby grid value at once
                    for s in (near - base[axis]) / fwd[axis]:
                        if -half_l < s < half_l:   # keep only crossings within the track
                            ss.add(float(s))

            for s in ss:
                grid_crossing_point = base + s * fwd
                neighbor_points = self._get_neighbor_points(grid_crossing_point)
                height_to_surface = self._bilinear_height(grid_crossing_point, neighbor_points)
                delta = height_to_surface - grid_crossing_point[2]
                needed = delta + self.q[2]
                # the midpoint (s == 0) is the last point of both halves
                if s <= 0:
                    support[i, 0] = max(support[i, 0], needed)
                if s >= 0:
                    support[i, 1] = max(support[i, 1], needed)

        if self.enable_blade:
            if check_cut_limit:
                blade_R = R @ self._rotation_lg(
                    self.blade_roll_pitch_yaw[0], 0.0, self.blade_roll_pitch_yaw[2])
                deepest_depth = (blade_R[2, 2] * self.blade_cut_depth() + abs(blade_R[2, 1]) * self.B1 / 2)
                stop_height = (self._undeformed_height(self.q) - self.max_world_cut_depth + deepest_depth)
                self.cut_limit_reached = bool(support.max() <= stop_height + self.contact_tol)
            support = np.maximum(support, self._blade_support_heights(R))
        return support

    def _blade_support_heights(self, R: FloatArray) -> FloatArray:
        """Required body heights for the blade top above undeformed soil.

        Classify support by body-local left/right and front/back, like the
        tracks. Partition the full edge at grid and body-axis crossings. Within
        each tile the interpolated clearance is quadratic, so also check its
        interior maximum instead of relying only on the blade's endpoints.
        """
        edge        = self._blade_edge_local(top=True)
        local_delta = edge[1] - edge[0]
        start       = self.q[:3] + R @ edge[0]
        delta       = R @ local_delta
        breaks      = {0.0, 0.5, 1.0}
        # seach x axis tiles via (self.us, 0) then y axis tiles via (self.vs, 1)
        for grid, axis in ((self.us, 0), (self.vs, 1)):
            # skip if the track runs parallel to this axis' lines (fwd[axis] ~ 0)
            # to avoid divide by zero issue
            if abs(delta[axis]) > 1e-12:
                # log splits along the local normal blade length where x and y grid crossings are
                lower, upper = sorted((start[axis], start[axis] + delta[axis]))
                for coordinate in grid[(grid > lower) & (grid < upper)]:
                    breaks.add(float((coordinate - start[axis]) / delta[axis]))

        def needed(t: Scalar) -> Scalar:
            point = start + t * delta
            return self._undeformed_height(point) - (R @ (edge[0] + t * local_delta))[2]

        # Support holds the highest requirement found for each region.
        # Starting at negative infinity lets the first real value replace it;
        # regions with no blade coverage remain -np.inf.
        support = np.full((2, 2), -np.inf)
        sorted_breaks = sorted(breaks)
        # This visits consecutive intervals: [0, 0.3], [0.3, 0.5], and so on.
        for lo, hi in zip(sorted_breaks[:-1], sorted_breaks[1:]):
            left, middle, right = needed(lo), needed((lo + hi) / 2), needed(hi)
            maximum             = max(left, right)
            # A bilinear tile's maximum is at a corner, but the blade may miss that
            # corner and encounter its highest terrain point inside the interval.
            # For example, this unit tile has h(x, y) = x + y - 2*x*y:
            #
            #   y = 1    1 ------- 0
            #            |         |
            #   y = 0    0 ------- 1
            #           x = 0     x = 1
            #
            # Along x = y = t, h(t) = 2*t - 2*t**2: both endpoints have height 0,
            # but the midpoint has height 0.5. Other oblique paths can also have
            # interior peaks, so check the quadratic's maximum, not just endpoints.
            # Fit f(s) = A*s**2 + B*s + C, where s spans this interval:
            # t = lo + s*(hi - lo). The samples give:
            #   f(0)   = left   => C = left
            #   f(1)   = right  => B = right - left - A
            #   f(0.5) = middle => middle = (left + right)/2 - A/4
            # Hence A = 2*(left + right - 2*middle).
            #
            # If A < 0, the curve bends downward and may have an interior maximum.
            # Setting f'(s) = 2*A*s + B = 0 gives s = -B/(2*A).
            # If 0 < s < 1, evaluate needed(lo + s*(hi - lo)) and compare it
            # with the endpoint maximum. Otherwise, the endpoints suffice.
            # The tolerance avoids dividing by a nearly zero A.
            quadratic = 2 * (left + right - 2 * middle)
            linear    = right - left - quadratic
            if quadratic < -1e-12:
                fraction = -linear / (2 * quadratic)
                if 0.0 < fraction < 1.0:
                    maximum = max(maximum, needed(lo + fraction * (hi - lo)))

            # Splitting at center of tile ensures each local coordinate is non-zero since we include local x=0
            local  = edge[0] + (lo + hi) / 2 * local_delta

            sides = [i for i, sign in enumerate((-1, 1)) if sign * local[1] >= -1e-12]

            # The blade edge always stays in front of the body origin (local x > 0),
            # so update only the front support column (index 1).
            for i in sides:
                support[i, 1] = max(support[i, 1], maximum)
        return support

    def _blade_edge_local(self, top: bool=False) -> FloatArray:
        """Blade edge endpoints in the body frame, including the arm offset."""
        local_R = self._rotation_lg(
            self.blade_roll_pitch_yaw[0], 0.0, self.blade_roll_pitch_yaw[2]
        )
        z = (self.H if top else 0.0) - self.blade_cut_depth()
        return np.array([
            self.blade_arm_offset + local_R @ np.array([self.L, y, z])
            for y in (-self.B1 / 2, self.B1 / 2)
        ])

    def _blade_bottom_center(self) -> FloatArray:
        """World position of the cutting-edge center, including both transforms."""
        body_R = self._rotation_lg(*self.q[3:6])
        return self.q[:3] + body_R @ self._blade_edge_local().mean(axis=0)

    def _undeformed_height(self, point: VectorLike) -> Scalar:
        """Original soil height, unaffected by cuts made during the run."""
        i, j = self._grid_cell(point)
        corners = [
            (*self.grid_pts[ci][cj][:2], self.starting_grid_heights[ci, cj])
            for ci, cj in ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1))
        ]
        return self._bilinear_height(point, corners)

    def _bilinear_height(self, point: VectorLike, corners: Sequence[VectorLike]) -> Scalar:
        """Bilinearly interpolate height from a tile's four corner vertices."""
        (x1, y1, h1), (x2, y2, h2), (x3, y3, h3), (_, _, h4) = corners

        dx, dy = point[0] - x1, point[1] - y1
        esx, esy = x2 - x1, y2 - y1
        etx, ety = x3 - x1, y3 - y1
        s = (dx * esx + dy * esy) / (esx * esx + esy * esy)
        t = (dx * etx + dy * ety) / (etx * etx + ety * ety)
        s = 0.0 if s < 0.0 else 1.0 if s > 1.0 else s
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t

        return (
              h1 * (1 - s) * (1 - t)
            + h2 * s * (1 - t)
            + h3 * (1 - s) * t
            + h4 * s * t
        )

    def _contact_angle(self, imbalance: Callable[[Scalar], Scalar], fitted: Scalar) -> Scalar:
        """Solve for a balanced contact angle using Blade's root solver (the Illinois method of Regula Falsi)."""
        limit, span = np.pi / 2 - 1e-3, 0.25
        while True:
            lo, hi = max(fitted - span, -limit), min(fitted + span, limit)
            f_lo, f_hi = imbalance(lo), imbalance(hi)
            if f_lo * f_hi <= 0:
                break
            if lo <= -limit and hi >= limit:
                return fitted
            span *= 2

        angle = fitted
        for _ in range(60):
            if f_hi == f_lo:
                break
            angle = (lo * f_hi - hi * f_lo) / (f_hi - f_lo)
            value = imbalance(angle)
            if abs(float(value)) < self.contact_tol or hi - lo < self.contact_tol:
                break
            if value * f_lo > 0:
                lo, f_lo = angle, value
                f_hi *= 0.5
            else:
                hi, f_hi = angle, value
                f_lo *= 0.5
        return angle

    def _body_contact_pitch(self, orient: FloatArray) -> Scalar:
        """Pitch that balances front/back track and blade support."""
        def imbalance(pitch: Scalar) -> Scalar:
            z_front, z_back = self._half_resting_heights(
                self._rotation_lg(orient[0], pitch, orient[2]))
            return z_front - z_back
        return self._contact_angle(imbalance, orient[1])

    def _half_resting_heights(self, R: FloatArray) -> tuple[float, float]:
        """
        The same question asked of each half of the body, as (front, back).
        _body_contact_pitch balances the pair.
        """
        support = self._support_heights(R)
        return float(support[:, 1].max()), float(support[:, 0].max())

    def _resting_height(self, R: FloatArray) -> float:
        """
        Lowest body height q[2] (for orientation R and the current q[:2]) that
        keeps the tracks above current soil and the enabled blade top above
        undeformed soil. The body rests on its highest required contact.
        """
        return float(self._support_heights(R).max())

    # ---------------------------- BLADE SURFACE DEFORMATION ------------------------------
    def _blade_update(self, deform: bool=True) -> tuple[BladeFrame, list[list[Point3]]]:
        if deform and self.enable_blade_control:
            previous_angles   = np.array(self.blade_roll_pitch_yaw, dtype=float, copy=True)
            blade_center = self._blade_bottom_center()
            starting_height = self._undeformed_height(blade_center)
            controller_output = self.controller.proportional_blade_controller(
                starting_height, blade_center[2])
            requested_angles  = previous_angles + controller_output

            # dx = (x_t - x_{t-1}) / dt; limit each axis in radians per second.
            requested_rates = (requested_angles - previous_angles) / self.dt

            self.requested_blade_rates = requested_rates.copy()
            limited_rates = np.clip(
                requested_rates, -self.blade_roll_pitch_yaw_rate_limits, self.blade_roll_pitch_yaw_rate_limits)
            self.blade_roll_pitch_yaw = np.clip(
                previous_angles + limited_rates * self.dt, -self.blade_roll_pitch_yaw_limits, self.blade_roll_pitch_yaw_limits)
            self.controller.apply_actuator_feedback(
                controller_output, self.blade_roll_pitch_yaw - previous_angles)
        elif deform:
            self.requested_blade_rates[:] = 0.0

        if deform:
            # Settle the body before cutting, including when blade angles are fixed.
            self._settle_tracks(self.q[3:6].copy())

        body_R = self._rotation_lg(*self.q[3:6])
        p0, p1 = self.q[:3] + self._blade_edge_local() @ body_R.T

        # TODO: when go back to optimize the code consider using the vector between p0 and p1 so not O(n) points
        length = np.linalg.norm(p1 - p0)
        n_segments = max(1, int(np.ceil(length / self.subdivision)))

        xyz = [
        (1 - t) * p0 + t * p1
            for t in np.linspace(0.0, 1.0, n_segments + 1)
        ]

        contacts_by_tile: dict[GridCell, list[FloatArray]] = {}
        contact_points = []
        for point in xyz:
            contacts_by_tile.setdefault(self._grid_cell(point), []).append(point)
            contact_points.append(point)

        if deform:
            self._deform_blade_tiles(contacts_by_tile)

        blade_points    = [np.concatenate((point, self.q[3:6])) for point in xyz]
        blade_neighbors = [self._get_neighbor_points(point) for point in contact_points]
        return blade_points, blade_neighbors

    def _blade_rotation_lg(self, orientation: VectorLike) -> FloatArray:
        """Blade-local roll/yaw composed on top of the body orientation."""
        body_R = self._rotation_lg(*orientation)
        local_R = self._rotation_lg(
            self.blade_roll_pitch_yaw[0], 0.0, self.blade_roll_pitch_yaw[2]
        )
        return body_R @ local_R

    def blade_cut_depth(self) -> Scalar:
        """Vertical cut depth commanded by blade pitch without tilting the blade."""
        return self.L * np.sin(self.blade_roll_pitch_yaw[1])

    def _grid_cell(self, point: VectorLike) -> GridCell:
        """Index of the grid tile containing point's xy position, clipped."""
        i = int((point[0] - self.us[0]) // self.subdivision)
        j = int((point[1] - self.vs[0]) // self.subdivision)
        i_max, j_max = len(self.us) - 2, len(self.vs) - 2
        i = 0 if i < 0 else i_max if i > i_max else i
        j = 0 if j < 0 else j_max if j > j_max else j
        return i, j

    def _deform_blade_tiles(self, contacts_by_tile: ContactMap) -> None:
        """Cut only the two vertices ahead of the blade in each contacted tile."""
        vel = np.array(self.q_dot[:3])
        if self.is_backwards:
            vel *= -1

        # Determine the direct footprint first. A shared vertex is still cut
        # only once when the blade contacts adjacent tiles in the same frame.
        direct_nodes: dict[GridCell, Scalar] = {}
        for (i, j), contacts in contacts_by_tile.items():
            local_blade_z = min(point[2] for point in contacts)
            corners = ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1))
            for ci, cj in corners:
                cx, cy, _ = self.grid_pts[ci][cj]
                if any((cx - point[0]) * vel[0] + (cy - point[1]) * vel[1] > 0.0
                       for point in contacts):
                    # Shared vertices take the deepest actual blade plane from
                    # their contacted tiles, never a relative depth copied from
                    # terrain at a different lateral height.
                    direct_nodes[(ci, cj)] = min(
                        direct_nodes.get((ci, cj), np.inf), local_blade_z
                    )

        if not direct_nodes:
            return

        for (i, j), local_blade_z in direct_nodes.items():
            x, y, height = self.grid_pts[i][j]

            min_height = self.starting_grid_heights[i, j] - self.max_world_cut_depth
            blade_z = max(local_blade_z, min_height)
            if blade_z < height :
                self.grid_pts[i][j] = (x, y, blade_z)
                self.surf_grid.nodes[(i, j)]['z'] = blade_z

    # ------------------------------ MAIN/ ENTRY FUNCTIONS --------------------------------
    def run_and_plot(self, show_neighbors: bool = False, show_desired_depth: bool = False) -> None:
        """Run the simulation and visualize the active body/blade view."""
        self.run()
        visualizer = Visualization(self)
        visualizer.visualization(show_neighbors, show_desired_depth=show_desired_depth)
        visualizer.forces_visualization()

    def _update_fill_progress(self, distance_step: Scalar) -> None:
        # might not do all that much consider removal when get to fitting
        """Accumulate travel and apply soil retention once per simulation step."""
        spill_factor = (float(np.clip(1.0 - abs(self.v[1]) / self.angular_velocity_limit, 0.0, 1.0))
                        if self.angular_velocity_limit > 0.0 else 1.0)
        retention = spill_factor ** (self.dt / self.spill_reference_seconds)
        self.fill_progress_distance = min(
            self.fill_distance, self.fill_progress_distance + distance_step
        ) * retention

    def run(self) -> None:
        """Advance either the blade-enabled or tracked-body simulation."""
        t = 0.0
        # Include the initial pose and every completed step of this run.
        # Read the reference now in case it was changed after construction.
        depth_errors = []
        if self.enable_blade:
            center = self._blade_bottom_center()
            depth_errors.append(self._undeformed_height(center) - center[2] - self.controller.desired_depth)
        for _ in range(int(self.stop_time / self.dt)):
            t += self.dt

            if self.enable_blade:
                # Cut at the current pose, settle on the new terrain, then log
                # blade geometry without cutting the same frame twice.
                self._blade_update()
                neighbor_points = self._body_update()
                blade_points, _ = self._blade_update(deform=False)
            else:
                neighbor_points = self._body_update()

            distance_step = np.linalg.norm(self.q_dot[:3] * self.dt)
            self.total_distance += distance_step
            self._update_fill_progress(distance_step)

            # log data for visualization and analysis
            self.log.append([t, *self.q])
            self._log_forces(t)
            if self.enable_blade:
                depth_errors.append(self.force_log[-1]["blade_depth_error"])
            if self.enable_blade:
                self.neighbor_log.append(np.array(neighbor_points))
                self.blade_log.append(blade_points)
                self.pile_log.append((self.H3, self.H4, float(self.blade_roll_pitch_yaw[0]),
                                      float(self.blade_roll_pitch_yaw[2])))
                self.grid_log.append(self._grid_heights())
            else:
                self.neighbor_log.append(np.array(neighbor_points))

            # Keep the terminal pose in the history and RMSE as well.
            if (self.total_distance >= self.stop_distance
                    or (self.use_path_controller and self.controller.path_complete)):
                break

        self.blade_depth_rmse = float(np.sqrt(np.mean(np.square(depth_errors)))) if depth_errors else None
        if self.blade_depth_rmse is not None:
            print(f"Blade depth RMSE: {self.blade_depth_rmse:.6f} m ({len(depth_errors)} samples)")

    def _log_forces(self, t: Scalar) -> None:
        """Snapshot dynamics values alongside each logged pose."""
        blade_center = self._blade_bottom_center()
        starting_height = float(self._undeformed_height(blade_center))
        blade_depth = starting_height - float(blade_center[2])
        desired_depth = float(self.controller.desired_depth)
        self.force_log.append({
            "time": float(t),
            "heading_error": float(self.controller.heading_err),
            "cross_track_error": float(self.controller.cross_track_err),
            "Fb": float(self.Fb), "Mb": float(self.Mb),
            "Rl_left": float(self.Rl[0]), "Rl_right": float(self.Rl[1]),
            "Fy": float(self.Fy), "Mr": float(self.Mr),
            "v_forward": float(self.v[0]), "v_turn": float(self.v[1]),
            "roll_error": float(self.controller.blade_controller_errors(self.blade_roll_pitch_yaw)[0][0]),
            "blade_pitch": float(self.blade_roll_pitch_yaw[1]),
            "requested_roll_rate": float(self.requested_blade_rates[0]),
            "requested_pitch_rate": float(self.requested_blade_rates[1]),
            "requested_yaw_rate": float(self.requested_blade_rates[2]),
            "starting_surface_height": starting_height,
            "blade_depth": blade_depth,
            "desired_depth": desired_depth,
            "blade_depth_error": blade_depth - desired_depth,
            "blade_bottom_height": float(blade_center[2]),
            "drive_left": float(self.F_track[0]), "drive_right": float(self.F_track[1]),
        })

    _run = run  # Compatibility with the pre-unification entry point.

    # Used by 3D visualizer
    def _track_xyz(self, pose: VectorLike) -> FloatArray:
        """Return [right, left] x [front, center, back] track points."""
        pose       = np.asarray(pose)
        R          = self._rotation_lg(*pose[3:6])
        half_width = R @ np.array([0.0, self.b / 2, 0.0])
        half_track = R @ np.array([self.l / 2, 0.0, 0.0])
        centers    = np.array([pose[:3] - half_width, pose[:3] + half_width])
        return np.stack(
            [centers + half_track, centers, centers - half_track], axis=1
        )


if __name__ == "__main__":
    simulation = DozerSimulation(
        is_uphill            = True,
        is_surface_pitched   = False,
        is_surface_rolled    = False,
        is_backwards         = False,
        enable_blade         = True,
        blade_roll_pitch_yaw = np.array([0.0, 0.0, 0.0]),
        enable_blade_control = True,
        use_path_controller  = True
    )

    simulation.run_and_plot()
