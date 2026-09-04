import numpy as np
import networkx as nx

from visualization import Visualization

class _Rotation():
    """
    Rotation matrices for transforming between local and global frames.
    """
    
    def _rotation_lg(self, a, B, g):
        """Rotation matrix: local to global frame."""
        # TODO: change to accept input array
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


class _Surface(_Rotation):
    """Terrain geometry shared by both modes of :class:`Surface`.

    The defaults follow blade-enabled behavior. ``_BodyMode`` customizes the
    few places where tracked-body terrain/contact behavior differs.
    """

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

    def _grid_cell(self, point):
        """Index of the grid tile containing point's xy position, clipped."""
        i = int((point[0] - self.us[0]) // self.subdivision)
        j = int((point[1] - self.vs[0]) // self.subdivision)
        i_max, j_max = len(self.us) - 2, len(self.vs) - 2
        i = 0 if i < 0 else i_max if i > i_max else i
        j = 0 if j < 0 else j_max if j > j_max else j
        return i, j

    def _get_neighbor_points(self, point):
        i, j = self._grid_cell(point)
        col_i, col_i1 = self.grid_pts[i], self.grid_pts[i + 1]
        return [col_i[j], col_i1[j], col_i[j + 1], col_i1[j + 1]]

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


class _DozerTrackSimulation(_Surface):
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

    def _settle_tracks(self, orient):
        """Set q's supported pose from both complete track contact lines."""
        self.q[3:6] = orient
        for _ in range(20):
            # Roll and pitch settle onto the ground so no quarter of the body
            # hangs. Yaw is the heading and cannot change during settling.
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

    def _body_contact_roll(self, orient):
        """Roll (at orient's pitch and yaw) that leaves both tracks on the ground."""
        def imbalance(roll):
            z_left, z_right = self._track_resting_heights(
                self._rotation_lg(roll, orient[1], orient[2]))
            return z_left - z_right
        return self._contact_angle(imbalance, orient[0])

    def _track_resting_heights(self, R):
        """
        The lowest-height question asked of each track on its own, as
        (left, right). _body_contact_roll balances the pair.
        """
        support = self._support_heights(R)
        return float(support[1].max()), float(support[0].max())

    def _support_heights(self, R):
        """
        Lowest body height each quarter of the contact patch calls for, as a
        [right, left] x [back, front] array. The rigid body can only rest at the
        highest of the four, so any quarter asking for less hangs clear of the
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
        return support

    def _body_contact_pitch(self, orient):
        """Pitch (at orient's roll and yaw) that leaves both ends on the ground."""
        def imbalance(pitch):
            z_front, z_back = self._half_resting_heights(
                self._rotation_lg(orient[0], pitch, orient[2]))
            return z_front - z_back
        return self._contact_angle(imbalance, orient[1])

    def _half_resting_heights(self, R):
        """
        The same question asked of each half of the body, as (front, back).
        _contact_pitch balances the pair.
        """
        support = self._support_heights(R)
        return float(support[:, 1].max()), float(support[:, 0].max())

    def _resting_height(self, R):
        """
        Lowest body height q[2] (for orientation R and the current q[:2]) that
        keeps every point of both rigid tracks on or above the surface, so the
        body rests on its highest contact and no track segment is submerged.
        """
        return float(self._support_heights(R).max())


class DozerSimulation(_DozerTrackSimulation):
    """Surface-aware bulldozer with an optional deforming blade."""

    def __init__(self, is_uphill = True, is_surface_pitched: bool = False, is_surface_rolled: bool = False,
                 is_backwards: bool = False, enable_blade: bool = True, blade_local_yaw: float = 0.0,
                 blade_local_roll: float = 0.0, blade_pitch: float = None):
        
        # ------ Load passed parameters ------ 
        self.is_surface_pitched = is_surface_pitched
        self.is_surface_rolled  = is_surface_rolled
        self.is_backwards       = is_backwards
        self.enable_blade       = enable_blade
        self.blade_local_yaw    = blade_local_yaw
        self.blade_local_roll   = blade_local_roll
        self.blade_pitch        = (np.arcsin((self.H / 4) / self.L) if blade_pitch is None
                                   else blade_pitch)
        # ------ General simulation parameters ------ 
        self.dt                 = 1/100 
        self.total_distance     = 0.0
        
        # ------ Independent simulation limits ------ 
        # change if tasks require move total distance to be traveled, still want it tight for fast tuning
        self.stop_time   = 300.0
        # contact-angle root-find tolerance (rad of angle / m of hang). Must stay
        # tighter than the settle loop's 1e-6 convergence check, or the pose it
        # returns is noisier than that check and the settle loop never converges
        # -- which burns *more* passes, so loosening this is a net slowdown.
        self.contact_tol = 1e-7

        # ------ Bulldozer body parameters ------ 
        self.b    = 1.75
        self.l    = 2.349

        # ------ Bulldozer blade parameters ------ 
        self.B1   = 2.921
        self.H    = 0.955
        self.L    = 1.2 

        # ------ Surface parameters ------ 
        self.division_factor = 8
        self.surface_abg     = np.array([ 0.0, 0.0, 0.0])
        self.u_split         = 0  # u-value where the grid switches to surface_abg2
        self.v_split         = 0  # u-value where the grid switches to surface_abg2

        # ------ set up Center Of Mass (COM) position, COM velocity, simulation surface ------ 
        self.q       = np.array([0.0, 0.0, 0.0, self.surface_abg[0], self.surface_abg[1], 0.0 
                                 if is_surface_pitched else np.pi / 2])
        self.q_dot   = np.array([2.0 if is_surface_pitched else 0.0, 0.0 if is_surface_pitched else 2.0, 
                                 0.0, 0.0, 0.0, 0.0]) 
        self.u_range = (-2* self.b/2, 4 * self.b) if is_surface_pitched else (-self.b/2,     self.b/2) 
        self.v_range = (-self.b/2,   self.b/2) if is_surface_pitched else (-2* self.b/2, 4 * self.b)
        if is_backwards:
            self.q_dot   *= -1
            self.u_split *= self.q_dot[0] / np.linalg.norm(self.q_dot)
            self.v_split *= self.q_dot[1] / np.linalg.norm(self.q_dot)
            self.u_range       = (-2.5 * self.b, self.b/2) if is_surface_pitched else (-self.b/2,     self.b/2) 
            self.v_range       = (-self.b/2,     self.b/2) if is_surface_pitched else (-2.5 * self.b, self.b/2)
        self.us            = np.arange(self.u_range[0], self.u_range[1] + self.subdivision, self.subdivision)
        self.vs            = np.arange(self.v_range[0], self.v_range[1] + self.subdivision, self.subdivision)
        
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
    
        # ------ Dependent simulation limits ------ 
        self.max_world_cut_depth = self.H
        stop_index         = 0 if self.is_backwards else -1 
        self.stop_distance = abs(self.us[stop_index]) if is_surface_pitched else abs(self.vs[stop_index])
            
        # ------ Write 1st entry to simulation logs ------ 
        self.log                = []
        self.blade_log          = []
        self.neighbor_log       = []
        self.grid_log           = []
        
        self.is_initalization   = True
        neighbor_points = self._body_update()
    
        self.log.append([0, *self.q])
        self.grid_log.append(self._grid_heights())

        # seed the blade points the same way, so blade_log stays aligned frame
        # for frame with log/neighbor_log/grid_log
        blade_points, blade_neighbors = self._blade_update(deform=False)
        neighbor_points.extend(point for neighbors in blade_neighbors for point in neighbors)
        self.neighbor_log.append(np.array(neighbor_points))
        self.blade_log.append(blade_points)

    @property
    def blade_cut_depth(self):
        """Vertical cut depth commanded by blade pitch without tilting the blade."""
        return self.L * np.sin(self.blade_pitch)

    def _blade_rotation_lg(self, orient):
        """Blade-local roll/yaw composed on top of the body orientation."""
        body_R = self._rotation_lg(*orient)
        local_R = self._rotation_lg(
            self.blade_local_roll, 0.0, self.blade_local_yaw
        )
        return body_R @ local_R

    def run_and_plot(self, show_neighbors: bool = False):
        """Run the simulation and visualize the active body/blade view."""
        self.run()
        return Visualization(self).visualization(show_neighbors)

    def run(self):
        """Advance either the blade-enabled or tracked-body simulation."""
        t = 0.0
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

            # stop condition check
            if self.total_distance >= self.stop_distance:
                break

            # log data for visualization and analysis
            self.log.append([t, *self.q])
            if self.enable_blade:
                self.neighbor_log.append(np.array(neighbor_points))
                self.blade_log.append(blade_points)
                self.grid_log.append(self._grid_heights())
            else:
                self.neighbor_log.append(np.array(neighbor_points))
                
    def _body_update(self):
        if not self.is_initalization:
            self.q += self.dt * self.q_dot
            self.q[3:6] = (self.q[3:6] + np.pi) % (2 * np.pi) - np.pi
        else:
            self.is_initalization = False

        #TODO: when optimizing run time for machine learning remove neighbor point tracking
        neighbor_points = self._get_neighbor_points(self.q)
        self._settle_tracks(self.q[3:6].copy())
        return neighbor_points

    def _blade_update(self, deform=True):
        orient = self.q[3:6].copy()
        R      = self._blade_rotation_lg(orient)
        p0 = self.q[:3] + R @ np.array([self.L, -self.B1 / 2, -self.blade_cut_depth])
        p1 = self.q[:3] + R @ np.array([self.L,  self.B1 / 2, -self.blade_cut_depth])

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

        blade_points = [np.concatenate((point, orient)) for point in xyz]
        blade_neighbors = [self._get_neighbor_points(point) for point in contact_points]
        return blade_points, blade_neighbors

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

    def _grid_heights(self):
        """Snapshot of every node height, indexed [i][j], for the GIF's surface."""
        return np.array([[pt[2] for pt in col] for col in self.grid_pts])

# Compatibility with the pre-unification private entry point.
DozerSimulation._run = DozerSimulation.run

if __name__ == "__main__":
    simulation = DozerSimulation(
        is_uphill          = True,
        is_surface_pitched = False,
        is_surface_rolled  = True,
        is_backwards       = False,
        enable_blade       = True,
        blade_local_roll   = -0.3, 
        blade_local_yaw    =  0.0, 
        blade_pitch        =  0.0
    )

    simulation.run_and_plot()
