import numpy as np
import networkx as nx

from visualization import Visualization


class Control():
    def __init__(self, L, dt, zero_to_max_angle_time):
        # controller reference
        self.L = L
        self.desired_depth = 0.3
        self.desired_roll_pitch_yaw = np.array([ 0, 1, 0])
        
        # Blade proportional and derivative controller gains
        self.Kp = np.ones(3) * dt * zero_to_max_angle_time
        self.Kp[1] = 1.28  # tuned pitch gain, assigned directly (no dt scaling)
        self.dt = dt
        self.Kd = np.ones(3) * 0.1 * dt * zero_to_max_angle_time
        self.Kd[1] = 0.32  # tuned pitch derivative gain; zero gives P-only
        self.previous_blade_error = None
        self.derivative_error = np.zeros(3)  # angular error change (rad/s)
        
    def blade_controller_errors(self, blade_roll_pitch_yaw):
        """
        Computes blade roll, pitch, yaw errors relative to desired surface and depth.
        """
        roll, pitch, yaw                                    = blade_roll_pitch_yaw
        desired_roll, desired_pitch_multiplier, desired_yaw = self.desired_roll_pitch_yaw

        # TODO: update blade angle limits from -1 to 1 to something more realistic, and update the test cases accordingly
        desired_pitch = desired_pitch_multiplier * np.arcsin(np.clip(self.desired_depth / self.L, -1.0, 1.0))

        errors   = np.array([desired_roll  - roll,       desired_pitch - pitch, desired_yaw - yaw])
        plot_out = np.array(            [errors[0], np.sin(errors[1]) * self.L,         errors[2]])

        return errors, plot_out
    
    def blade_deformation_errors(self, starting_surface_height, blade_bottom_height):
        """ Positive deformation is below the original surface; positive depth
        error requests more cut. The pitch gain maps metres to angle commands.
        """
        deformation   =  starting_surface_height - blade_bottom_height
        AAA           = self.desired_depth - deformation
        desired_pitch = np.arcsin(np.clip(AAA / self.L, -1.0, 1.0))
        errors        = np.array([0.0, desired_pitch, 0.0])
        return errors, errors.copy()

    def proportional_blade_controller(self, starting_surface_height, blade_bottom_height):
        """Return a PD angle increment using the existing angular depth error."""
        errors = self.blade_deformation_errors(starting_surface_height, blade_bottom_height)[0]
        # No previous sample exists on startup or after a reset.
        self.derivative_error = (np.zeros(3) if self.previous_blade_error is None
                                 else (errors - self.previous_blade_error) / self.dt)
        self.previous_blade_error = errors.copy()
        return self.Kp * errors + self.Kd * self.derivative_error

    def reset_derivative(self):
        self.previous_blade_error = None
        self.derivative_error[:] = 0.0

    def dummy_track_controlller(self, dozer_position_and_orientation):
        F_track_base = 600000.0
        return np.array([F_track_base, F_track_base])


class PIControl(Control):
    """Separate PI blade controller with conditional-integration anti-windup."""

    def __init__(self, L, dt, zero_to_max_angle_time):
        super().__init__(L, dt, zero_to_max_angle_time)
        self.Kp = np.array([0.01, 0.00025, 0.01])  # tuned PI pitch gain
        self.Ki = np.array([0.0, 0.0001, 0.0])  # tuned integral gain
        self.integral_error = np.zeros(3)  # integral of angular depth error (rad s)
        self._previous_integral = self.integral_error.copy()

    def proportional_blade_controller(self, starting_surface_height, blade_bottom_height):
        errors = self.blade_deformation_errors(starting_surface_height, blade_bottom_height)[0]
        self._previous_integral = self.integral_error.copy()
        self.integral_error += np.where(self.Ki != 0., errors * self.dt, 0.)
        return self.Kp * errors + self.Ki * self.integral_error

    def apply_actuator_feedback(self, requested_increment, applied_increment):
        """Freeze integration into a limit, but permit unwinding out of it."""
        blocked_increment = requested_increment - applied_increment
        integral_increment = self.Ki * (self.integral_error - self._previous_integral)
        blocked = (np.abs(blocked_increment) > 1e-12) & (blocked_increment * integral_increment > 0.)
        self.integral_error[blocked] = self._previous_integral[blocked]

    def reset_integrator(self):
        self.integral_error[:] = 0.
        self._previous_integral[:] = 0.


class DozerSimulation():
    """Surface-aware bulldozer with an optional deforming blade."""

    def __init__(self, is_uphill = True, is_surface_pitched: bool = False, is_surface_rolled: bool = False,
                 is_backwards: bool = False, enable_blade: bool = True, blade_roll_pitch_yaw = np.zeros(3),
                 controller_type: str = "pd"):
        #TODO: add comments with parameter descriptions and units (maybe change names to be more descriptive?)
        # ----------------------------- Load passed parameters ----------------------------- 
        self.is_surface_pitched   = is_surface_pitched
        self.is_surface_rolled    = is_surface_rolled
        self.is_backwards         = is_backwards
        self.enable_blade         = enable_blade
        self.blade_roll_pitch_yaw = blade_roll_pitch_yaw
                           
        # ------------------------- General simulation parameters -------------------------- 
        self.dt             = 1/100
        gravity             = 9.81 
        self.total_distance = 0.0

        # -------------------- Independent simulation limit parameters --------------------- 
        # change if tasks require move total distance to be traveled, still want it tight for fast tuning
        self.stop_time              = 5.0
        # contact-angle root-find tolerance (rad of angle / m of hang). Must stay
        # tighter than the settle loop's 1e-6 convergence check, or the pose it
        # returns is noisier than that check and the settle loop never converges
        # -- which burns *more* passes, so loosening this is a net slowdown.
        self.contact_tol            = 1e-7
        self.fill_distance          = 8.0    # maximum distance for soil fill calculation TODO: should be computed relative to depth of cut
        self.velocity_limit         = 2.222
        self.lateral_velocity_limit = 0.0    # this governs how much the dozer can "slide" laterally
        self.blade_roll_pitch_yaw_limits      = np.array([0.0735, 0.430, 0.387]) # radians
        zero_to_max_angle_time = 1              # second
        self.blade_roll_pitch_yaw_rate_limits = self.blade_roll_pitch_yaw_limits / zero_to_max_angle_time # radians per second
        
        # --------------------------- Bulldozer body parameters ---------------------------- 
        mass              = 10156.0 # of the unloaded vehicle in kilograms
        self.b            = 1.75    # width between track centers (track guage) in meters
        self.l            = 2.349   # track length in meters
        self.h            = 2.762/2 # body height in meters, scaled down by Sam
        self.F_track_base = 600000.0
        self.track_width  = 0.7112
        self.track_height = 0.5
        
        # -------------------------- Bulldozer blade parameters ---------------------------- 
        self.B1 = 2.921 # blade width in meters
        self.H  = 0.955 # blade height in meters
        self.L  = 1.2   # blade arm length in meters
        self.blade_arm_offset = np.array([0.5, 0, 0])
        
        # -------------------------------- Soil parameters --------------------------------- 
        self.mu_l    = 0.1              # longitudinal friction coefficient
        self.mu_t    = 0.9              # lateral      friction coefficient
        self.mu_ss   = 0.5              # shear        friction coefficient
        self.kb      = 0.734e6          # blade soil interaction constant
        self.beta0   = np.radians(38.0) # soil accumulation angle in radians
        self.gamma_g = 1640 * gravity   # soil weight per cubic meter
        
        # ------------------------------- Surface parameters ------------------------------- 
        self.division_factor = 8
        self.surface_abg     = np.array([ 0.0, 0.0, 0.0])
        self.u_split         = 20  # u-value where the grid switches to surface_abg2
        self.v_split         = 20  # u-value where the grid switches to surface_abg2

        # ----- Set up Center Of Mass (COM) position, COM velocity, simulation surface ----- 
        self.u_range = (-1* self.b/2, 8 * self.b) if is_surface_pitched else (-self.b/2,     self.b/2) 
        self.v_range = (-self.b/2,   self.b/2) if is_surface_pitched else (-1* self.b/2, 8 * self.b)
        if is_backwards:
            self.q_dot   *= -1
            self.u_split *= self.q_dot[0] / np.linalg.norm(self.q_dot)
            self.v_split *= self.q_dot[1] / np.linalg.norm(self.q_dot)
            self.u_range  = (-2.5 * self.b, self.b/2) if is_surface_pitched else (-self.b/2,     self.b/2) 
            self.v_range  = (-self.b/2,     self.b/2) if is_surface_pitched else (-2.5 * self.b, self.b/2)
        self.us           = np.arange(self.u_range[0], self.u_range[1] + self.subdivision, self.subdivision)
        self.vs           = np.arange(self.v_range[0], self.v_range[1] + self.subdivision, self.subdivision)
        
        self.offset             = np.array([0, 0, self.b]) if is_uphill else np.array([0, 0, -self.b])
        self.transition_tiles   = 1/2 *self.b * self.division_factor # tiles over which the offset ramps down past u_split

        self.surf_grid = self._surface_grid()

        # node (x, y, z) cached as plain Python floats, indexed [i][j], so the
        # hot height lookups skip the networkx attribute dicts entirely.
        self.grid_pts = [[(self.surf_grid.nodes[(i, j)]['x'],
                           self.surf_grid.nodes[(i, j)]['y'],
                           self.surf_grid.nodes[(i, j)]['z'])
                          for j in range(len(self.vs))] for i in range(len(self.us))]
        self.starting_grid_heights = self._grid_heights().copy()
    
        # --------------------- Dependent simulation limit parameters ---------------------- 
        self.max_world_cut_depth    = self.H
        stop_index                  = 0 if self.is_backwards else -1 
        self.stop_distance = abs(self.us[stop_index]) if is_surface_pitched else abs(self.vs[stop_index])
        self.angular_velocity_limit = 2 * self.velocity_limit / self.b

        # ------------------------ Forces and moment Parameters --------------------------- 
        self.F_track = np.array([self.F_track_base, self.F_track_base])  #[left, right]
        self.rl      = self.mu_l * mass * gravity / 2      # longitudinal track force limit
        self.fy      = self.mu_t * mass * gravity / self.l # lateral force limit
        Ix           = mass * (self.b**2 + self.h**2) / 12 # moment of inertia about x-axis
        Iy           = mass * (self.h**2 + self.l**2) / 12 # moment of inertia about y-axis
        Iz           = mass * (self.b**2 + self.l**2) / 12 # moment of inertia about z-axis
        self.M       = np.diag([mass, mass, mass, Ix, Iy, Iz])             # mass matrix
        self.P       = np.array([0.0, 0.0, mass * gravity, 0.0, 0.0, 0.0]) # gravitational force vector
        self.elim    = np.diag([1.0, 1.0, 1.0, 0.0, 0.0, 1.0])
        
        # ------- Velocity and rotation + corresponding derivative vector variables -------- 
        self.v     = np.zeros(2)  # initial local velocity vector
        self.v_dot = np.zeros(2)  # initial local acceleration vector
        self.q     = np.array([0.0, 0.0, 0.0, self.surface_abg[0], self.surface_abg[1], 
                               0.0 if is_surface_pitched else np.pi / 2])
        
        self.q_dot = np.zeros(6) 
        self.dxyz  = np.zeros(3)
        self.daBg  = np.zeros(3)
        self.Rl    = np.zeros(2)
        self.R_lg  = self._rotation_lg(*self.q[3:6])
        self.J_lg  = self._rotation_derivatives()
                
        # ----------------- Other kinematics and dynamic scalar variables ------------------ 
        self.x_ICR     = 0.0
        self.x_ICR_dot = 0.0
        self.Fb        = 0.0
        self.cut_limit_reached = False
        self.Mb        = 0.0
        self.H3        = 0.0
        self.H4        = 0.0
        self.H3_sub    = 0.0
        self.H4_sub    = 0.0
        self.Fy        = 0.0
        self.Mr        = 0.0
        self.vtL       = 0.0
        self.vtR       = 0.0
        
        # ----------------------------------- Controller ----------------------------------- 
        if controller_type not in ("pd", "pi"):
            raise ValueError("controller_type must be 'pd' or 'pi'")
        controller_class = PIControl if controller_type == "pi" else Control
        self.controller = controller_class(self.L, self.dt, zero_to_max_angle_time)
        self.requested_blade_rates = np.zeros(3)  # roll, pitch, yaw before limiting (rad/s)
        
        # ----------------------- Write 1st entry to simulation logs ----------------------- 
        self.log          = []
        self.blade_log    = []
        self.neighbor_log = []
        self.grid_log     = []
        self.force_log    = []
        self.blade_depth_rmse = None
        
        self.is_initialized = False
        neighbor_points     = self._body_update()

    
        self.log.append([0, *self.q])
        self._log_forces(0.0)
        self.grid_log.append(self._grid_heights())

        # seed the blade points the same way, so blade_log stays aligned frame
        # for frame with log/neighbor_log/grid_log
        blade_points, blade_neighbors = self._blade_update(deform=False)
        neighbor_points.extend(point for neighbors in blade_neighbors for point in neighbors)
        self.neighbor_log.append(np.array(neighbor_points))
        self.blade_log.append(blade_points)

    # ---------------------------- SURFACE GENERATION AND INFO ----------------------------     
    @property
    def subdivision(self):
        """Grid spacing, sized relative to the dozer width ``self.b``."""
        return self.b / self.division_factor

    def _surface_grid(self):
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

    def _grid_heights(self):
        """Snapshot of every node height, indexed [i][j], for the GIF's surface."""
        return np.array([[pt[2] for pt in col] for col in self.grid_pts])

    def _get_neighbor_points(self, point):
        i, j = self._grid_cell(point)
        col_i, col_i1 = self.grid_pts[i], self.grid_pts[i + 1]
        return [col_i[j], col_i1[j], col_i[j + 1], col_i1[j + 1]]

    # ------------------------------------- ROTATION --------------------------------------     
    def _rotation_lg(self, a, B, g):
        """Rotation matrix: local to global frame."""
        return self._rotation_gl(a, B, g).T

    def _rotation_gl(self, a, B, g):
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
    def _body_update(self):
        if self.is_initialized:
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
    def _update_q_dot(self):
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

    def _rotation_derivatives(self):
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

    def _saturation(self, value, limit):
        return float(np.clip(value, -abs(limit), abs(limit)))

    def _get_x_icr(self, eps: float = 1e-3):
        if abs(self.daBg[2]) < eps:
            return 0.0
        return float(np.clip(-self.dxyz[1] / self.daBg[2], -self.l / 2, self.l / 2))

    def _vehicle_dynamics(self):
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

    def _track_terrain_interaction(self):
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

    def _G(self, force, friction, velocity):
        if abs(velocity) > 1e-10:
            return -friction * np.sign(velocity)
        if abs(force) <= friction:
            return -force
        return -friction * np.sign(force)

    def _blade_terrain_interaction(self):
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
        
        fill_percent = min(1.0, self.total_distance / self.fill_distance)
        
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

    def _yc(self, left_depth, right_depth, width):
        if left_depth == 0 and right_depth == 0:
            return 0.0
        return (
            (2 * left_depth + right_depth) / (3 * (left_depth + right_depth)) * width
            - width / 2
        )

    def _S_matrix(self):
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

    def _Sd_matrix(self):
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
        
        S_12 = -self.x_ICR * (R[0, 2] * Ad + R[2, 1] * cg * Bd - R[1, 1] * Gd) 
        - self.x_ICR_dot * R[0, 1]
        
        S_22 = -self.x_ICR * (R[1, 2] * Ad + R[2, 1] * sg * Bd + R[0, 1] * Gd) 
        - self.x_ICR_dot * R[1, 1]
        
        S_32 = -self.x_ICR * (R[2, 2] * Ad - sa * sB * Bd) 
        - self.x_ICR_dot * R[2, 1]
        
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
    def _settle_tracks(self, orient):
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

    def _body_contact_roll(self, orient):
        """Roll that balances left/right track and blade support."""
        def imbalance(roll):
            z_left, z_right = self._track_resting_heights(
                self._rotation_lg(roll, orient[1], orient[2]))
            return z_left - z_right
        return self._contact_angle(imbalance, orient[0])

    def _track_resting_heights(self, R):
        """
        Required support heights on each side of the body, as (left, right).
        Includes the blade when enabled; _body_contact_roll balances the pair.
        """
        support = self._support_heights(R)
        return float(support[1].max()), float(support[0].max())

    def _support_heights(self, R, check_cut_limit=False):
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

    def _blade_support_heights(self, R):
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

        def needed(t):
            point = start + t * delta
            return self._undeformed_height(point) - (R @ (edge[0] + t * local_delta))[2]
        
        # Support holds the highest requirement found for each region. 
        # Starting at negative infinity lets the first real value replace it; 
        # regions with no blade coverage remain -np.inf.
        support = np.full((2, 2), -np.inf)
        breaks  = sorted(breaks)
        # This visits consecutive intervals: [0, 0.3], [0.3, 0.5], and so on.
        for lo, hi in zip(breaks[:-1], breaks[1:]):
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

    def _blade_edge_local(self, top=False):
        """Blade edge endpoints in the body frame, including the arm offset."""
        local_R = self._rotation_lg(
            self.blade_roll_pitch_yaw[0], 0.0, self.blade_roll_pitch_yaw[2]
        )
        z = (self.H if top else 0.0) - self.blade_cut_depth()
        return np.array([
            self.blade_arm_offset + local_R @ np.array([self.L, y, z])
            for y in (-self.B1 / 2, self.B1 / 2)
        ])

    def _blade_bottom_center(self):
        """World position of the cutting-edge center, including both transforms."""
        body_R = self._rotation_lg(*self.q[3:6])
        return self.q[:3] + body_R @ self._blade_edge_local().mean(axis=0)

    def _undeformed_height(self, point):
        """Original soil height, unaffected by cuts made during the run."""
        i, j = self._grid_cell(point)
        corners = [
            (*self.grid_pts[ci][cj][:2], self.starting_grid_heights[ci, cj])
            for ci, cj in ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1))
        ]
        return self._bilinear_height(point, corners)

    def _bilinear_height(self, point, corners):
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

    def _contact_angle(self, imbalance, fitted):
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
            if abs(value) < self.contact_tol or hi - lo < self.contact_tol:
                break
            if value * f_lo > 0:
                lo, f_lo = angle, value
                f_hi *= 0.5
            else:
                hi, f_hi = angle, value
                f_lo *= 0.5
        return angle

    def _body_contact_pitch(self, orient):
        """Pitch that balances front/back track and blade support."""
        def imbalance(pitch):
            z_front, z_back = self._half_resting_heights(
                self._rotation_lg(orient[0], pitch, orient[2]))
            return z_front - z_back
        return self._contact_angle(imbalance, orient[1])

    def _half_resting_heights(self, R):
        """
        The same question asked of each half of the body, as (front, back).
        _body_contact_pitch balances the pair.
        """
        support = self._support_heights(R)
        return float(support[:, 1].max()), float(support[:, 0].max())

    def _resting_height(self, R):
        """
        Lowest body height q[2] (for orientation R and the current q[:2]) that
        keeps the tracks above current soil and the enabled blade top above
        undeformed soil. The body rests on its highest required contact.
        """
        return float(self._support_heights(R).max())

    # ---------------------------- BLADE SURFACE DEFORMATION ------------------------------ 
    def _blade_update(self, deform=True):
        if deform:
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
            if isinstance(self.controller, PIControl):
                self.controller.apply_actuator_feedback(
                    controller_output, self.blade_roll_pitch_yaw - previous_angles)
            # Apply the new blade command to the supported body pose before
            # any soil is removed; the original terrain remains the constraint.
            self._settle_tracks(self.q[3:6].copy())
            #TODO: if performance is an issue I could just call settle track after the body acceleration update
    
        body_R = self._rotation_lg(*self.q[3:6])
        p0, p1 = self.q[:3] + self._blade_edge_local() @ body_R.T

        # TODO: when go back to optimize the code consider using the vector between p0 and p1 so not O(n) points
        length = np.linalg.norm(p1 - p0)
        n_segments = max(1, int(np.ceil(length / self.subdivision)))

        xyz = [
        (1 - t) * p0 + t * p1
            for t in np.linspace(0.0, 1.0, n_segments + 1)
        ]

        contacts_by_tile = {}
        contact_points = []
        for point in xyz:
            contacts_by_tile.setdefault(self._grid_cell(point), []).append(point)
            contact_points.append(point)

        if deform:
            self._deform_blade_tiles(contacts_by_tile)

        blade_points    = [np.concatenate((point, self.q[3:6])) for point in xyz]
        blade_neighbors = [self._get_neighbor_points(point) for point in contact_points]
        return blade_points, blade_neighbors

    def _blade_rotation_lg(self, orientation):
        """Blade-local roll/yaw composed on top of the body orientation."""
        body_R = self._rotation_lg(*orientation)
        local_R = self._rotation_lg(
            self.blade_roll_pitch_yaw[0], 0.0, self.blade_roll_pitch_yaw[2]
        )
        return body_R @ local_R

    def blade_cut_depth(self):
        """Vertical cut depth commanded by blade pitch without tilting the blade."""
        return self.L * np.sin(self.blade_roll_pitch_yaw[1])
   
    def _grid_cell(self, point):
        """Index of the grid tile containing point's xy position, clipped."""
        i = int((point[0] - self.us[0]) // self.subdivision)
        j = int((point[1] - self.vs[0]) // self.subdivision)
        i_max, j_max = len(self.us) - 2, len(self.vs) - 2
        i = 0 if i < 0 else i_max if i > i_max else i
        j = 0 if j < 0 else j_max if j > j_max else j
        return i, j

    def _deform_blade_tiles(self, contacts_by_tile):
        """Cut only the two vertices ahead of the blade in each contacted tile."""
        vel = np.array(self.q_dot[:3])
        if self.is_backwards:
            vel *= -1

        # Determine the direct footprint first. A shared vertex is still cut
        # only once when the blade contacts adjacent tiles in the same frame.
        direct_nodes = {}
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
    def run_and_plot(self, show_neighbors: bool = False):
        """Run the simulation and visualize the active body/blade view."""
        self.run()
        visualizer = Visualization(self)
        visualizer.visualization(show_neighbors)
        visualizer.forces_visualization()

    def run(self):
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

            self.total_distance += np.linalg.norm(self.q_dot[:3] * self.dt)

            # log data for visualization and analysis
            self.log.append([t, *self.q])
            self._log_forces(t)
            if self.enable_blade:
                depth_errors.append(self.force_log[-1]["blade_depth_error"])
            if self.enable_blade:
                self.neighbor_log.append(np.array(neighbor_points))
                self.blade_log.append(blade_points)
                self.grid_log.append(self._grid_heights())
            else:
                self.neighbor_log.append(np.array(neighbor_points))

            # Keep the terminal pose in the history and RMSE as well.
            if self.total_distance >= self.stop_distance:
                break

        self.blade_depth_rmse = float(np.sqrt(np.mean(np.square(depth_errors)))) if depth_errors else None
        if self.blade_depth_rmse is not None:
            print(f"Blade depth RMSE: {self.blade_depth_rmse:.6f} m ({len(depth_errors)} samples)")

    def _log_forces(self, t):
        """Snapshot dynamics values alongside each logged pose."""
        blade_center = self._blade_bottom_center()
        starting_height = float(self._undeformed_height(blade_center))
        blade_depth = starting_height - float(blade_center[2])
        desired_depth = float(self.controller.desired_depth)
        self.force_log.append({
            "time": float(t),
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

    # Used by 3D visualizer
    def _track_xyz(self, pose):
        """Return [right, left] x [front, center, back] track points."""
        pose       = np.asarray(pose)
        R          = self._rotation_lg(*pose[3:6])
        half_width = R @ np.array([0.0, self.b / 2, 0.0])
        half_track = R @ np.array([self.l / 2, 0.0, 0.0])
        centers    = np.array([pose[:3] - half_width, pose[:3] + half_width])
        return np.stack(
            [centers + half_track, centers, centers - half_track], axis=1
        )


# Compatibility with the pre-unification private entry point.
DozerSimulation._run = DozerSimulation.run

if __name__ == "__main__":
    simulation = DozerSimulation(
        is_uphill            = True,
        is_surface_pitched   = False,
        is_surface_rolled    = False,
        is_backwards         = False,
        enable_blade         = True,
        blade_roll_pitch_yaw = np.array([0.0, 0.0, 0.0])
    )

    simulation.run_and_plot()
