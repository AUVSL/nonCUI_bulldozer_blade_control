/*
 * C++ port of the 3D bulldozer blade control simulation.
 * Requires Eigen3. Build with CMakeLists.txt in this directory.
 * Plotting is replaced by CSV output (simulation_log.csv, forces_log.csv).
 */
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using Vec2  = Eigen::Vector2d;
using Vec3  = Eigen::Vector3d;
using Vec6  = Eigen::Matrix<double, 6, 1>;
using Mat3  = Eigen::Matrix3d;
using Mat6  = Eigen::Matrix<double, 6, 6>;
using Mat62 = Eigen::Matrix<double, 6, 2>;
using Mat22 = Eigen::Matrix2d;
using MatXd = Eigen::MatrixXd;
using VecXd = Eigen::VectorXd;

// Log row: [t, q(0..5), cross_track_err, heading_err, Mb, Fb, RlL, RlR, Fy, Mr, v_fwd, v_trn]
using LogEntry = std::array<double, 17>;

static int comb(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    long long r = 1;
    for (int i = 0; i < k; ++i)
        r = r * (n - i) / (i + 1);
    return static_cast<int>(r);
}

static double sign(double x) {
    if (x > 0.0) return  1.0;
    if (x < 0.0) return -1.0;
    return 0.0;
}

static double wrap_angle(double x) {
    double r = std::fmod(x + M_PI, 2.0 * M_PI);
    if (r < 0.0) r += 2.0 * M_PI;
    return r - M_PI;
}

static Vec3 wrap_angles(const Vec3& a) {
    return Vec3(wrap_angle(a(0)), wrap_angle(a(1)), wrap_angle(a(2)));
}

// ─────────────────────────────────────────────────────────────────────────────
class BulldozerSimulation {
public:
    // Parameters
    Vec3   desired_abg;
    double desired_depth;
    double stop_time;
    Vec3   surface_abg;
    double F_track_base;
    Vec2   F_track;
    double l, b, B1, h, H, L;
    double mu_l, mu_t, mu_ss, kb, dt, beta0;
    double gain, velocity_limit, turn_vel_limit, fill_distance, gamma_g;
    Mat6   elim;
    double cross_track_err, heading_err;
    std::vector<Vec3> path_points;

    double rl, fy;
    Mat6   M_mat;
    Vec6   P;
    double Kp, Kp_path;

    // State
    double x_ICR_dot;
    Vec3   dxyz, daBg, bld_ang;
    Vec6   q, q_dot;
    Vec2   v, v_dot;
    double x_ICR;
    Mat3   R_lg, J_lg;
    double Fb, Mb;
    Vec2   Rl;
    double Fy, Mr, vtL, vtR;

    // Bezier controller
    Eigen::Matrix<double, 7, 1> _bezier_coeffs;
    bool   _has_bezier;
    double _bezier_ang_max;
    double _lookahead_dist;
    int    _nearest_path_idx;

    std::vector<LogEntry> log;

    // ─── Constructor ───────────────────────────────────────────────────────
    BulldozerSimulation() {
        desired_abg   = Vec3(0.3, 0.0, 0.0);
        desired_depth = -0.05;
        stop_time     = 0.2;
        surface_abg   = Vec3::Zero();

        const double grav   = 9.81;
        const double m      = 10156.0 / 4.0;
        F_track_base        = 60000.0;
        F_track             = Vec2(F_track_base, F_track_base);
        const double h_val  = 2.762 / 2.0;
        h  = h_val;
        l  = 2.349 / 1.5;
        b  = 1.75  / 1.5;
        B1 = 2.921;
        H  = 0.955;
        L  = 1.2;

        mu_l  = 0.1;
        mu_t  = 0.9;
        mu_ss = 0.5;
        kb    = 0.734e6;
        dt    = 1.0 / 100.0;
        beta0 = M_PI / 180.0 * 38.0;

        gain           = 1.0 / 40.0;
        velocity_limit = 2.222;
        turn_vel_limit = 2.0 * velocity_limit / b;
        fill_distance  = 8.0;
        gamma_g        = 1640.0 * grav;

        elim = Mat6::Zero();
        elim(0,0) = 1; elim(1,1) = 1; elim(2,2) = 1; elim(5,5) = 1;

        cross_track_err = 0.0;
        heading_err     = 0.0;
        path_points     = figure8_path(5.0, 2.5, 2000);

        rl = mu_l * m * grav / 2.0;
        fy = mu_t * m * grav / l;

        double Ix = m * (b*b + h_val*h_val) / 12.0;
        double Iy = m * (h_val*h_val + l*l) / 12.0;
        double Iz = m * (b*b + l*l) / 12.0;
        M_mat = Mat6::Zero();
        M_mat.diagonal() << m, m, m, Ix, Iy, Iz;
        P = Vec6::Zero();
        P(2) = m * grav;

        Kp      = -3.0;
        Kp_path = 8000.0;

        // Initial conditions
        x_ICR_dot = 0.0;
        dxyz      = Vec3::Zero();
        daBg      = Vec3::Zero();
        bld_ang   = Vec3::Zero();
        q         = Vec6::Zero();
        q(3)      = surface_abg(0);
        q(4)      = surface_abg(1);
        q(5)      = surface_abg(2);
        q_dot     = Vec6::Zero();

        v     = Vec2::Zero();
        v_dot = Vec2::Zero();
        x_ICR = 0.0;

        R_lg = rotation_lg(q(3), q(4), q(5));
        Mat3 J_gl_tmp;
        rotation_derivatives(q(3), q(4), J_gl_tmp, J_lg);

        Fb = 0.0; Mb = 0.0;
        Rl = Vec2::Zero();
        Fy = 0.0; Mr = 0.0;
        vtL = 0.0; vtR = 0.0;

        _has_bezier       = false;
        _bezier_coeffs    = Eigen::Matrix<double, 7, 1>::Zero();
        _bezier_ang_max   = 1.0;
        _lookahead_dist   = 1.5;
        _nearest_path_idx = 0;
    }

    // ─── Helpers ───────────────────────────────────────────────────────────
    static double saturation(double value, double limit) {
        return std::max(-limit, std::min(limit, value));
    }

    static double G_func(double F, double f, double dx) {
        if (std::abs(dx) > 1e-10)
            return -f * sign(dx);
        if (std::abs(F) <= f)
            return -F;
        return -f * sign(F);
    }

    static double yc_centroid(double D1, double D2, double B1_) {
        if (D1 == 0.0 && D2 == 0.0) return 0.0;
        return (D1 + 2.0*D2) / (3.0*(D1 + D2)) * B1_ - B1_ / 2.0;
    }

    // ─── Kinematics ────────────────────────────────────────────────────────
    // Returns R_gl (global→local). Python transposes the literal so cols = rows here.
    Mat3 rotation_gl(double a, double beta, double g) const {
        double sa = std::sin(a),  ca = std::cos(a);
        double sB = std::sin(beta), cB = std::cos(beta);
        double sg = std::sin(g),  cg = std::cos(g);
        Mat3 R;
        R << cB*cg,   sa*sB*cg - ca*sg,   ca*sB*cg + sa*sg,
             cB*sg,   sa*sB*sg + ca*cg,   ca*sB*sg - sa*cg,
            -sB,      sa*cB,              ca*cB;
        // Python: np.array([...]).T  → transpose the literal
        return R.transpose();
    }

    // R_lg = R_gl^T  (local→global)
    Mat3 rotation_lg(double a, double beta, double g) const {
        return rotation_gl(a, beta, g).transpose();
    }

    void rotation_derivatives(double a, double beta,
                               Mat3& J_gl_out, Mat3& J_lg_out) const {
        double sa = std::sin(a),  ca = std::cos(a);
        double sB = std::sin(beta), cB = std::cos(beta), tB = std::tan(beta);
        J_gl_out << 1,    0,    -sB,
                    0,   ca, sa*cB,
                    0,  -sa, ca*cB;
        J_lg_out << 1, sa*tB,  ca*tB,
                    0,    ca,    -sa,
                    0, sa/cB, ca/cB;
    }

    Mat62 S_matrix() const {
        double x = safe_division_x_icr();
        Mat62 S  = Mat62::Zero();
        S.block<3,1>(0,0) = R_lg.col(0);
        S.block<3,1>(0,1) = R_lg.col(1) * (-1.0 / x);
        S.block<3,1>(3,1) = J_lg.col(2);
        return S;
    }

    double get_x_icr(double eps = 1e-3) const {
        if (std::abs(daBg(2)) < eps) return 0.0;
        return std::max(-l/2.0, std::min(l/2.0, dxyz(1) / daBg(2)));
    }

    double safe_division_x_icr(double eps = 5e-2) const {
        if (std::abs(x_ICR) < eps) return std::numeric_limits<double>::max();
        return x_ICR;
    }

    // ─── Terrain interactions ───────────────────────────────────────────────
    void blade_terrain_interaction() {
        double a_rel = surface_abg(0) - bld_ang(0);
        double hp    = std::abs(L * std::sin(bld_ang(1)));

        double H1 = B1 * std::tan(std::abs(a_rel));
        double H2 = hp / std::cos(a_rel);
        double H3 = H - H2 + sign(a_rel) * H1/2.0 - H1/2.0;
        double H4 = H - H2 - sign(a_rel) * H1/2.0 - H1/2.0;

        double tan_abs = std::tan(std::abs(a_rel));
        double a_val   = tan_abs * tan_abs;
        double c_val   = (H3 + H4) / 2.0;
        double V = 0.5/std::tan(beta0) * (a_val*a_val*B1*B1*B1/12.0 + c_val*c_val*B1);

        double fill_percent = q.head<3>().norm() / fill_distance;
        double Gt = V * gamma_g * fill_percent;

        double hyp      = B1 / std::cos(std::abs(a_rel));
        double area_cut = 0.5 * B1 * H1 + hyp * hp;
        double F1 = area_cut * kb;
        double F2 = Gt * mu_ss;
        Fb = -F1 - F2;

        double yc1 = yc_centroid(H3/std::tan(beta0), H4/std::tan(beta0), B1);
        double yc2 = yc_centroid(H2, H1 + H2, B1);
        Mb = yc1*F1 + yc2*F2;
    }

    void track_terrain_interaction() {
        vtL = saturation(dxyz(0) - b/2.0 * daBg(2), velocity_limit);
        vtR = saturation(dxyz(0) + b/2.0 * daBg(2), velocity_limit);

        double FtL = F_track(0), FtR = F_track(1);
        Rl(0) = G_func(FtL, rl, vtL);
        Rl(1) = G_func(FtR, rl, vtR);

        Fy = -2.0 * sign(dxyz(1)) * fy * std::abs(x_ICR);

        double M_val = ((FtR + Rl(1)) - (FtL + Rl(0))) * b / 2.0;
        double mr    = 2.0 * fy * (l*l/4.0 - x_ICR*x_ICR);
        Mr = G_func(M_val, mr, daBg(2));
    }

    // ─── S-dot matrix ──────────────────────────────────────────────────────
    Mat62 Sd_matrix() const {
        double a    = q(3), beta = q(4), g = q(5);
        double Ad   = q_dot(3), Bd = q_dot(4), Gd = q_dot(5);
        double sa   = std::sin(a),  ca = std::cos(a);
        double sB   = std::sin(beta), cB = std::cos(beta);
        double sg   = std::sin(g),  cg = std::cos(g);
        double tB   = std::tan(beta);

        const Mat3& R = R_lg;

        double S_11 = -sB*cg*Bd - R(1,0)*Gd;
        double S_21 = -sB*sg*Bd + R(0,0)*Gd;
        double S_31 = -cB*Bd;

        double S_12 = -x_ICR*(R(0,2)*Ad + R(2,1)*cg*Bd - R(1,1)*Gd) - x_ICR_dot*R(0,1);
        double S_22 = -x_ICR*(R(1,2)*Ad + R(2,1)*sg*Bd + R(0,1)*Gd) - x_ICR_dot*R(1,1);
        double S_32 = -x_ICR*(R(2,2)*Ad - sa*sB*Bd)                  - x_ICR_dot*R(2,1);

        double S_42 = -sa*tB*Ad + ca/(cB*cB)*Bd;
        double S_52 = -ca*Ad;
        double S_62 = -sa/cB*Ad + ca*tB/cB*Bd;

        Mat62 Sd = Mat62::Zero();
        Sd(0,0)=S_11; Sd(0,1)=S_12;
        Sd(1,0)=S_21; Sd(1,1)=S_22;
        Sd(2,0)=S_31; Sd(2,1)=S_32;
        Sd(3,1)=S_42;
        Sd(4,1)=S_52;
        Sd(5,1)=S_62;
        return Sd;
    }

    // ─── Vehicle dynamics ───────────────────────────────────────────────────
    Vec2 vehicle_dynamics() {
        track_terrain_interaction();
        blade_terrain_interaction();

        Mat62 B_mat = Mat62::Zero();
        B_mat.block<3,1>(0,0) = R_lg.col(0);
        B_mat.block<3,1>(0,1) = R_lg.col(0);
        B_mat.block<3,1>(3,0) = -R_lg.col(2) * b / 2.0;
        B_mat.block<3,1>(3,1) =  R_lg.col(2) * b / 2.0;

        double ab = bld_ang(0), Bb = bld_ang(1), gb = bld_ang(2);

        Vec6 Ct_vec = Vec6::Zero();
        Ct_vec(0) = Rl(0) + Rl(1);
        Ct_vec(1) = Fy;
        Ct_vec(5) = Mr + (Rl(1) - Rl(0)) * b / 2.0;

        Mat3 R_blade = rotation_lg(ab, Bb, gb);
        Mat6 R6      = Mat6::Zero();
        R6.block<3,3>(0,0) = R_blade;
        R6.block<3,3>(3,3) = R_blade;

        Vec6 blade_vec = Vec6::Zero();
        blade_vec(0) = Fb;
        blade_vec(5) = Mb;
        Vec6 Cb_vec = elim * R6 * blade_vec;

        Mat6 R6_lg = Mat6::Zero();
        R6_lg.block<3,3>(0,0) = R_lg;
        R6_lg.block<3,3>(3,3) = R_lg;
        Vec6 C = R6_lg * (Ct_vec + Cb_vec);

        Mat62 S  = S_matrix();
        Mat62 Sd = Sd_matrix();

        Mat22 Bt = S.transpose() * B_mat;
        Vec2  Ct = S.transpose() * C;
        Vec2  Pt = S.transpose() * P;
        Mat22 Mt = S.transpose() * M_mat * S;
        Mat22 Et = S.transpose() * M_mat * Sd;

        std::cout << "Ct: " << Ct.transpose()
                  << "  Bt@F_track: " << (Bt * F_track).transpose() << "\n";

        return Mt.ldlt().solve(Bt * F_track + Ct - Et * v - Pt);
    }

    // ─── Controller ────────────────────────────────────────────────────────
    std::pair<Vec3, Vec3> controller_errors() const {
        double roll = bld_ang(0), pitch = bld_ang(1), yaw = bld_ang(2);
        double desired_roll  = desired_abg(0);
        double des_pitch_mult= desired_abg(1);
        double desired_yaw   = desired_abg(2);
        double desired_pitch = des_pitch_mult * std::asin(
            std::max(-1.0, std::min(1.0, desired_depth / L)));

        Vec3 errors(roll - desired_roll, pitch - desired_pitch, yaw - desired_yaw);
        Vec3 plot_out(errors(0), std::sin(errors(1)) * L, errors(2));
        return {errors, plot_out};
    }

    // ─── Figure-8 path ─────────────────────────────────────────────────────
    std::vector<Vec3> figure8_path(double A, double B2, int n_points) const {
        Mat3 R_surf = rotation_lg(surface_abg(0), surface_abg(1), surface_abg(2));
        Vec3 e1 = R_surf.col(0);
        Vec3 e2 = R_surf.col(1);
        std::vector<Vec3> pts(n_points);
        for (int i = 0; i < n_points; ++i) {
            double t  = 2.0 * M_PI * i / n_points;
            double xs = A  * std::sin(t);
            double ys = B2 * std::sin(2.0 * t);
            pts[i] = xs * e1 + ys * e2;
        }
        return pts;
    }

    double signed_cross_track_error(const Vec3& pos) const {
        int n = (int)path_points.size();
        int idx = 0;
        double min_d = std::numeric_limits<double>::max();
        for (int i = 0; i < n; ++i) {
            double d = (path_points[i] - pos).norm();
            if (d < min_d) { min_d = d; idx = i; }
        }
        int  next_idx = (idx + 1) % n;
        Vec3 tangent  = path_points[next_idx] - path_points[idx];
        double nrm    = tangent.norm();
        if (nrm < 1e-10) return 0.0;
        tangent /= nrm;
        Vec3 n_surf     = rotation_lg(surface_abg(0), surface_abg(1), surface_abg(2)).col(2);
        Vec3 left_normal = n_surf.cross(tangent);
        return (pos - path_points[idx]).dot(left_normal);
    }

    void path_controller() {
        cross_track_err = signed_cross_track_error(q.head<3>());
        double delta = Kp_path * cross_track_err;
        double F_max = 2.0 * F_track_base;
        F_track(0) = std::max(0.0, std::min(F_max, F_track_base + delta));
        F_track(1) = std::max(0.0, std::min(F_max, F_track_base - delta));
    }

    void use_bezier_controller(const Eigen::Matrix<double,7,1>& coeffs,
                                double ang_max, double lookahead_dist = 1.0) {
        _bezier_coeffs  = coeffs;
        _has_bezier     = true;
        _bezier_ang_max = ang_max;
        _lookahead_dist = lookahead_dist;
    }

    double eval_bezier6(double t) const {
        double result = 0.0;
        for (int i = 0; i <= 6; ++i)
            result += comb(6,i) * std::pow(t,i) * std::pow(1.0-t, 6-i) * _bezier_coeffs(i);
        return result;
    }

    double pure_pursuit_heading_error() {
        Vec3 pos  = q.head<3>();
        double La = _lookahead_dist;
        int n     = (int)path_points.size();

        int nearest = 0;
        double min_d = std::numeric_limits<double>::max();
        for (int i = 0; i < n; ++i) {
            double d = (path_points[i] - pos).norm();
            if (d < min_d) { min_d = d; nearest = i; }
        }
        _nearest_path_idx = nearest;

        Vec3 lookahead_pt = path_points[nearest]; // fallback
        for (int k = 0; k < n; ++k) {
            int  i    = (nearest + k) % n;
            int  j    = (i + 1) % n;
            Vec3 p1   = path_points[i];
            Vec3 p2   = path_points[j];
            Vec3 dseg = p2 - p1;
            Vec3 fseg = p1 - pos;
            double aa  = dseg.dot(dseg);
            double bb  = 2.0 * fseg.dot(dseg);
            double cc  = fseg.dot(fseg) - La*La;
            double disc = bb*bb - 4.0*aa*cc;
            if (disc < 0.0) continue;
            double t2 = (-bb + std::sqrt(disc)) / (2.0 * aa);
            if (t2 >= 0.0 && t2 <= 1.0) {
                lookahead_pt = p1 + t2 * dseg;
                break;
            }
        }

        Mat3 R_surf  = rotation_lg(surface_abg(0), surface_abg(1), surface_abg(2));
        Vec3 n_surf  = R_surf.col(2);
        Vec3 e1_surf = R_surf.col(0);
        Vec3 e2_surf = R_surf.col(1);
        Vec3 d_vec   = lookahead_pt - pos;
        Vec3 d_proj  = d_vec - d_vec.dot(n_surf) * n_surf;
        double angle = std::atan2(d_proj.dot(e2_surf), d_proj.dot(e1_surf));
        double err   = angle - q(5);
        return wrap_angle(err);
    }

    void angular_path_controller() {
        heading_err     = pure_pursuit_heading_error();
        cross_track_err = signed_cross_track_error(q.head<3>());
        double ang      = std::min(std::abs(heading_err), _bezier_ang_max);
        double t        = ang / _bezier_ang_max;
        double fraction = std::max(0.0, std::min(1.0, eval_bezier6(t)));
        if (heading_err > 0.0) {
            F_track(0) = fraction * F_track_base;
            F_track(1) = F_track_base;
        } else {
            F_track(0) = F_track_base;
            F_track(1) = fraction * F_track_base;
        }
    }

    // ─── Main integration loop ──────────────────────────────────────────────
    void run() {
        double t    = 0.0;
        int n_pts   = (int)path_points.size();
        int max_idx = 0;
        int n_steps = (int)(stop_time / dt);

        for (int step = 0; step < n_steps; ++step) {
            if (_has_bezier) {
                angular_path_controller();
                max_idx = std::max(max_idx, _nearest_path_idx);
                if (max_idx > n_pts * 0.9 && _nearest_path_idx < n_pts * 0.1)
                    break;
            }

            auto [errors, plot_err] = controller_errors();
            bld_ang += gain * Kp * errors;

            v_dot = vehicle_dynamics();
            v    += dt * v_dot;
            v(0)  = std::max(0.0, std::min(v(0), velocity_limit));
            v(1)  = std::max(-turn_vel_limit, std::min(turn_vel_limit, v(1)));
            q_dot = S_matrix() * v;

            q += dt * q_dot;
            q.tail<3>() = wrap_angles(q.tail<3>());

            double a_cur = q(3), B_cur = q(4), g_cur = q(5);
            R_lg = rotation_lg(a_cur, B_cur, g_cur);
            Mat3 R_gl = rotation_gl(a_cur, B_cur, g_cur);
            Mat3 J_gl;
            rotation_derivatives(a_cur, B_cur, J_gl, J_lg);

            dxyz = R_gl * q_dot.head<3>();
            daBg = J_gl * q_dot.tail<3>();

            double prev_x_ICR = x_ICR;
            x_ICR     = get_x_icr();
            x_ICR_dot = (x_ICR - prev_x_ICR) / dt;

            LogEntry e;
            e[0]  = t;
            e[1]  = q(0); e[2] = q(1); e[3] = q(2);
            e[4]  = q(3); e[5] = q(4); e[6] = q(5);
            e[7]  = cross_track_err;
            e[8]  = heading_err;
            e[9]  = Mb;
            e[10] = Fb;
            e[11] = Rl(0);
            e[12] = Rl(1);
            e[13] = Fy;
            e[14] = Mr;
            e[15] = v(0);
            e[16] = v(1);
            log.push_back(e);

            t += dt;
        }
    }

    // ─── Bezier calibration ─────────────────────────────────────────────────
    std::pair<Eigen::Matrix<double,7,1>, double>
    build_bezier6_pinned(double threshold = 0.498, int n_samples = 60) {
        std::vector<double> fracs(n_samples);
        for (int i = 0; i < n_samples; ++i)
            fracs[i] = threshold * i / (n_samples - 1);

        std::vector<double> yaws(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            BulldozerSimulation sim;
            sim.stop_time  = 0.5;
            sim.F_track(0) = sim.F_track_base;
            sim.F_track(1) = fracs[i] * sim.F_track_base;
            sim.run();
            yaws[i] = sim.log.empty() ? 0.0 : sim.log.back()[6];
        }

        double ang_max = 0.0;
        std::vector<double> ang(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            ang[i]  = std::abs(yaws[i]);
            ang_max = std::max(ang_max, ang[i]);
        }

        std::vector<double> t_bez(n_samples);
        for (int i = 0; i < n_samples; ++i)
            t_bez[i] = ang[i] / ang_max;

        // Build Bernstein basis (n_samples × 7)
        MatXd B_basis(n_samples, 7);
        for (int i = 0; i <= 6; ++i)
            for (int r = 0; r < n_samples; ++r)
                B_basis(r,i) = comb(6,i) * std::pow(t_bez[r],i) * std::pow(1.0-t_bez[r], 6-i);

        // rhs = fracs - threshold * B[:,0]
        VecXd rhs(n_samples);
        for (int r = 0; r < n_samples; ++r)
            rhs(r) = fracs[r] - threshold * B_basis(r, 0);

        // Least-squares solve: B[:,1:6] * c_int = rhs
        MatXd B_inner = B_basis.block(0, 1, n_samples, 5);
        VecXd c_int   = B_inner.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);

        Eigen::Matrix<double,7,1> coeffs;
        coeffs(0) = threshold;
        for (int i = 0; i < 5; ++i) coeffs(i+1) = c_int(i);
        coeffs(6) = 0.0;

        std::cout << "Bezier-6 pinned built  ang_max=" << ang_max << " rad  coeffs=["
                  << coeffs.transpose() << "]\n";
        return {coeffs, ang_max};
    }

    // ─── Save log to CSV ────────────────────────────────────────────────────
    void save_log_csv(const std::string& fname = "simulation_log.csv") const {
        std::ofstream f(fname);
        f << "t,x,y,z,roll,pitch,yaw,cross_track_err,heading_err,"
             "Mb,Fb,RlL,RlR,Fy,Mr,v_fwd,v_trn\n";
        for (const auto& e : log) {
            for (int i = 0; i < 17; ++i) {
                f << e[i];
                if (i < 16) f << ',';
            }
            f << '\n';
        }
        std::cout << "Saved " << fname << '\n';
    }

    // run_and_plot equivalent: run + print stats, save CSVs
    void run_and_plot(double sim_stop_time = 30.0, double lookahead_dist = 1.2,
                      bool use_path_controller = true) {
        BulldozerSimulation sim;
        sim.stop_time = sim_stop_time;

        if (use_path_controller) {
            std::cout << "Building Bezier-6 pinned lookup table...\n";
            auto [coeffs, ang_max] = build_bezier6_pinned();
            std::cout << "Running figure-8 for " << sim_stop_time
                      << "s  lookahead=" << lookahead_dist << "m ...\n";
            sim.use_bezier_controller(coeffs, ang_max, lookahead_dist);
        }

        sim.run();

        if (use_path_controller && !sim.log.empty()) {
            auto rmse_mae_max = [&](int col) {
                double sse = 0, sae = 0, mx = 0;
                for (auto& e : sim.log) {
                    double v = e[col];
                    sse += v*v; sae += std::abs(v); mx = std::max(mx, std::abs(v));
                }
                int n = (int)sim.log.size();
                return std::make_tuple(std::sqrt(sse/n), sae/n, mx);
            };

            auto [ct_rmse, ct_mae, ct_max] = rmse_mae_max(7);
            auto [he_rmse, he_mae, he_max] = rmse_mae_max(8);
            // convert heading to degrees
            he_rmse *= 180.0/M_PI; he_mae *= 180.0/M_PI; he_max *= 180.0/M_PI;

            printf("\n%16s  %10s  %10s  %10s\n", "", "RMSE", "MAE", "Max |err|");
            printf("%s\n", std::string(52, '-').c_str());
            printf("%16s  %10.4f  %10.4f  %10.4f\n", "Cross-track (m)", ct_rmse, ct_mae, ct_max);
            printf("%16s  %10.4f  %10.4f  %10.4f\n", "Heading (deg)",  he_rmse, he_mae, he_max);
        }

        sim.save_log_csv("simulation_log.csv");
    }
};

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    BulldozerSimulation sim;
    sim.run_and_plot(/*stop_time=*/2.0, /*lookahead=*/1.2, /*use_path_controller=*/false);
    return 0;
}
