        self.x_ICR_dot  = 0.0
        self.x_ICR_pm1  = 0.0
        self.x_ICR_pm2  = 0.0
        self.v_dot_prev = None
        self.q_dot_prev = None
        self._step      = 0         

if self.v_dot_prev is None:
                self.v += self.dt * v_dot
            else:
                self.v += self.dt * (1.5 * v_dot - 0.5 * self.v_dot_prev)
            self.v_dot_prev = v_dot.copy()
            self.v[0] = max(min(self.v[0], self.velocity_limit), 0)
            self.v[1] = np.clip(self.v[1], -self.turn_vel_limit, self.turn_vel_limit)
            self.q_dot = self.S_matrix() @ self.v

            # update global/local positions and orientations for next time step
            if self.q_dot_prev is None:
                self.q += self.dt * self.q_dot
            else:
                self.q += self.dt * (1.5 * self.q_dot - 0.5 * self.q_dot_prev)
            self.q_dot_prev = self.q_dot.copy()
            self.q[3:6]    = self.wrap_angles(self.q[3:6])

            
            self.x_ICR_pm2 = self.x_ICR_pm1
            self.x_ICR_pm1 = self.x_ICR
            self.x_ICR     = self.get_x_icr()
            if self._step < 2:
                self.x_ICR_dot = (self.x_ICR - self.x_ICR_pm1) / self.dt
            else:
                self.x_ICR_dot = (3*self.x_ICR - 4*self.x_ICR_pm1 + self.x_ICR_pm2) / (2*self.dt)
            self._step += 1