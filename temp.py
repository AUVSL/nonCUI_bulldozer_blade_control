class _DozerDynamics(_Rotation):
    """Two-input tracked-dozer dynamics used to update a simulation's ``q_dot``.

    The model owns force and velocity state but not pose or terrain state.  A
    :class:`DozerSimulation` supplies its current pose, blade angles, and time
    step to :meth:`step`; the returned generalized velocity can then be
    integrated before the tracks are settled against the terrain.
    """

    def __init__(self, dt):
        self.dt = dt
        #TODO: add comments with parameter descriptions and units (maybe change names to be more descriptive?)
        # simulation and control parameters
        gravity            = 9.81
        self.fill_distance = 8.0 # maximum distance for soil fill calculation

        # Dozer body parameters
        mass                         = 10156.0
        self.b                       = 1.75    # track width in meters
        self.l                       = 2.349   # track length in meters
        self.h                       = 2.762/2 # body height in meters, scaled down by Sam
        self.velocity_limit          = 2.222
        self.lateral_velocity_limit  = 0.0     # this governs how much the dozer can "slide" laterally
        self.angular_velocity_limit  = 2 * self.velocity_limit / self.b
        self.F_track_base            = 600000.0
        self.F_track = np.array([self.F_track_base, self.F_track_base])  #[left, right]
        
        # Bulldozer blade parameters
        self.B1 = 2.921 # blade width in meters
        self.H  = 0.955 # blade height in meters
        self.L  = 1.2 # blade arm length in meters

        # Soil parameters
        self.mu_l          = 0.1              # longitudinal friction coefficient
        self.mu_t          = 0.9              # lateral      friction coefficient
        self.mu_ss         = 0.5              # shear        friction coefficient
        self.kb            = 0.734e6          # blade soil interaction constant
        self.beta0         = np.radians(38.0) # soil accumulation angle in radians
        self.gamma_g       = 1640 * gravity   # soil weight per cubic meter

        # Dynamic motion parameters
        self.rl   = self.mu_l * mass * gravity / 2      # longitudinal track force limit
        self.fy   = self.mu_t * mass * gravity / self.l # lateral force limit
        Ix        = mass * (self.b**2 + self.h**2) / 12 # moment of inertia about x-axis
        Iy        = mass * (self.h**2 + self.l**2) / 12 # moment of inertia about y-axis
        Iz        = mass * (self.b**2 + self.l**2) / 12 # moment of inertia about z-axis
        self.M    = np.diag([mass, mass, mass, Ix, Iy, Iz])             # mass matrix
        self.P    = np.array([0.0, 0.0, mass * gravity, 0.0, 0.0, 0.0]) # gravitational force vector
        self.elim = np.diag([1.0, 1.0, 1.0, 0.0, 0.0, 1.0])

        # Initial conditions vectors
        self.v         = np.zeros(2)  # initial local velocity vector
        self.v_dot     = np.zeros(2)  # initial local acceleration vector
        self.q_dot     = np.zeros(6)  # initial global velocity vector
        self.q         = np.zeros(6)   # initial global position vector
        self.dxyz      = np.zeros(3)
        self.daBg      = np.zeros(3)
        self.Rl        = np.zeros(2)
        self.R_lg      = self.rotation_lg(self.q[3], self.q[4], self.q[5])
        self.J_lg      = self.rotation_derivatives(self.q[3], self.q[4])
        
        # Initial conditions scalars
        self.x_ICR     = 0.0
        self.x_ICR_dot = 0.0
        self.Fb        = 0.0
        self.Mb        = 0.0
        self.H3        = 0.0
        self.H4        = 0.0
        self.H3_sub    = 0.0
        self.H4_sub    = 0.0
        self.Fy        = 0.0
        self.Mr        = 0.0
        self.vtL       = 0.0
        self.vtR       = 0.0

    @staticmethod
    def saturation(value, limit):
        return float(np.clip(value, -abs(limit), abs(limit)))

    @staticmethod
    def G(force, friction, velocity):
        if abs(velocity) > 1e-10:
            return -friction * np.sign(velocity)
        if abs(force) <= friction:
            return -force
        return -friction * np.sign(force)

    @staticmethod
    def yc(left_depth, right_depth, width):
        if left_depth == 0 and right_depth == 0:
            return 0.0
        return (
            (2 * left_depth + right_depth) / (3 * (left_depth + right_depth)) * width
            - width / 2
        )

    def rotation_derivatives(self, a, B):
        """Rotation derivative matrices"""
        # TODO: change to accept input array
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

    def get_x_icr(self, eps: float = 1e-3):
        if abs(self.daBg[2]) < eps:
            return 0.0
        return float(np.clip(-self.dxyz[1] / self.daBg[2], -self.l / 2, self.l / 2))

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

    def Sd_matrix(self):
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

    def track_terrain_interaction(self):
        self.vtL = self.saturation(self.dxyz[0] - self.b / 2 * self.daBg[2], 
                                   self.velocity_limit)
        self.vtR = self.saturation(self.dxyz[0] + self.b / 2 * self.daBg[2], 
                                   self.velocity_limit)
        
        FtL, FtR = self.F_track
        
        RlL     = self.G(FtL, self.rl, self.vtL)
        RlR     = self.G(FtR, self.rl, self.vtR)
        self.Rl = np.array([RlL, RlR])
        
        self.Fy = -2 * np.sign(self.dxyz[1]) * self.fy * abs(self.x_ICR)
        
        moment          = ((FtR + RlR) - (FtL + RlL)) * self.b / 2
        moment_friction = 2 * self.fy * ((self.l**2) / 4 - (self.x_ICR**2))
        self.Mr         = self.G(moment, moment_friction, self.daBg[2])

    def blade_terrain_interaction(self, blade_roll, blade_pitch, total_distance):
        # TODO: update this to be surface - body angle + blade angle, to account for non-flat surfaces
        a_rel = blade_roll
        
        cut_height = abs(self.L * np.sin(blade_pitch))
        
        H1     = self.B1 * np.tan(abs(a_rel))
        H2     = cut_height / np.cos(abs(a_rel))
        H3_sub = -H2 + (np.sign(a_rel) * H1 / 2) - (H1 / 2)
        H4_sub = -H2 - (np.sign(a_rel) * H1 / 2) - (H1 / 2)
        H3     = self.H + H3_sub
        H4     = self.H + H4_sub
        
        a_val = np.tan(abs(a_rel))**2
        c_val = (H3 + H4) / 2
        
        fill_percent = min(1.0, total_distance / self.fill_distance)
        
        # TODO: update to account for later dump cycles
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
        
        shear_moment_arm  = self.yc(-H3_sub, -H4_sub, self.B1)
        normal_moment_arm = self.yc(     H3,      H4, self.B1) 
        
        self.Mb  = (   soil_shear_force *  shear_moment_arm 
                    + soil_normal_force * normal_moment_arm)

    def vehicle_dynamics(self, blade_orientation=None, blade_pitch=0.0, total_distance=0.0):
        self.track_terrain_interaction()
        self.blade_terrain_interaction(blade_orientation[0], blade_pitch, total_distance)
        
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
        R_blade = self.rotation_lg(*blade_orientation)
        R6[:3, :3]   = R_blade
        R6[3:6, 3:6] = R_blade
        
        blade_vec = np.array([self.Fb, 0.0, 0.0, 0.0, 0.0, self.Mb])
        Cb_vec    = self.elim @ R6 @ blade_vec

        R6_lg = np.zeros((6, 6))
        R6_lg[:3, :3]   = self.R_lg
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

    def update_q_dot(self, q, q_dot, blade_orientation,
                     total_distance=0.0):
        """Advance internal velocity state and return generalized velocity."""
        self.q         = np.asarray(q, dtype=float).copy()
        a, B, g        = self.q[3:6]
        self.R_lg      = self.rotation_lg(a, B, g)
        _, self.J_lg   = self.rotation_derivatives(a, B)
        
        R_gl = self.R_lg.T
        J_gl, self.J_lg = self.rotation_derivatives(a, B)
        self.q_dot = np.asarray(q_dot, dtype=float)
        
        self.dxyz = R_gl @ self.q_dot[:3]
        self.daBg = J_gl @ self.q_dot[3:6]
        
        self.dxyz[1] = self.saturation(self.dxyz[1], self.lateral_velocity_limit)
        self.x_ICR = self.get_x_icr()
        
        self.v_dot = self.vehicle_dynamics(blade_orientation, total_distance)
        
        self.v += self.dt * self.v_dot

        self.v[0] = np.clip(self.v[0], -self.velocity_limit, 0.0)
        self.v[1] = self.saturation(self.v[1], self.angular_velocity_limit)
        
        next_q_dot = self.S_matrix() @ self.v

        return next_q_dot.copy()