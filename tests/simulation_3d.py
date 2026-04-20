"""
Python port of the 3D bulldozer blade control simulation.
Replaces the Simulink sim('simulation_3d') call with an explicit
forward-Euler integration loop.

NOTE — information that could NOT be recovered from the .m files alone
(i.e. was encoded inside the .slx Simulink model):
  1. PID gains for roll and yaw blade controllers (only KpP for pitch was
     in parameters.m; Ki, Kd for all axes are unknown).
  2. Derivative filter coefficients / structure for the PID blocks.
  3. Whether blade angular velocity (bld_ang_vel) fed to hydraulics() comes
     from a PID output or a simpler proportional law — assumed proportional
     here using the errors and gain from parameters.m.
  4. The exact signals logged to sim_out / output.data (column ordering
     assumed from errors_and_plots.m).
  5. Any feedforward, anti-windup, or output-clamping logic inside the
     Simulink controller subsystem.
  6. Track torque scheduling — F_track is held constant here; the Simulink
     model may have varied it.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # headless; remove if running interactively

# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────
def saturation(value, limit):
    return float(np.clip(value, -limit, limit))

def G(F, f, dx):
    """Friction force model."""
    if dx != 0:
        return -f * np.sign(dx)
    elif ((dx == 0) and (abs(F) <= f)):
        return -F
    else:
        return -f * np.sign(F)

def yc(D1, D2, B1):
    """Centroid of a trapezoid."""
    if (D1 == 0 and D2 == 0):
        return 0.0
    return (D1 + 2 * D2) / (3 * (D1 + D2)) * B1 - B1 / 2

def Hx(yb, H3, H4, B1):
    return (-H3 + H4) / B1 * yb + (H3 + H4) / 2

def Dx(yb, beta_0, H3, H4, B1):
    return Hx(yb, H3, H4, B1) / np.tan(beta_0)

# ──────────────────────────────────────────────────────────────────────────────
# Physics functions
# ──────────────────────────────────────────────────────────────────────────────
def blade_terrain_interaction(hp, kb, gamma_g, mu_ss, a_s, beta0,
                               a_b, B1, H, fill_percent):
    a_rel = a_s - a_b
    H1 = B1 * np.tan(abs(a_rel))
    H2 = hp / np.cos(a_rel)                     # hp * sec(a_rel)
    H3 = H - H2 + np.sign(a_rel) * H1 / 2 - H1 / 2
    H4 = H - H2 - np.sign(a_rel) * H1 / 2 - H1 / 2

    a_val = np.tan(abs(a_rel)) ** 2
    c_val = (H3 + H4) / 2
    V = 0.5 / np.tan(beta0) * (1 / 12 * a_val ** 2 * B1 ** 3 + c_val ** 2 * B1)

    Gt = V * gamma_g * fill_percent

    hyp = B1 / np.cos(abs(a_rel))
    area_cut = 0.5 * B1 * H1 + hyp * hp
    F1 = area_cut * kb
    F2 = Gt * mu_ss
    FT = -F1 - F2

    yc1 = yc(Dx(-B1 / 2, beta0, H3, H4, B1),
              Dx(B1 / 2,  beta0, H3, H4, B1), B1)
    yc2 = yc(H2, H1 + H2, B1)
    Mb = yc1 * F1 + yc2 * F2
    return FT, Mb

def blade_and_track(F_track, q, q_dot, x_ICR, new_bld_ang, bt_params):
    a, B, g = q[3], q[4], q[5]
    X_dot, Y_dot, Z_dot = q_dot[0], q_dot[1], q_dot[2]
    a_dot, B_dot, g_dot = q_dot[3], q_dot[4], q_dot[5]

    (B1, H, L, b, l, r, m, grav, velocity_limit, fill_distance,
     mu_t, mu_l, mu_ss, kb, gamma_g, beta0, a_s, B_s, g_s) = bt_params

    a_b, B_b, g_b = new_bld_ang

    sa, ca = np.sin(a), np.cos(a)
    sB, cB = np.sin(B), np.cos(B)
    sg, cg = np.sin(g), np.cos(g)

    hp = abs(L * np.sin(B_b))

    R_gl = np.array([
        [cB * cg,          sa * sB * cg - ca * sg, ca * sB * cg + sa * sg],
        [cB * sg,          sa * sB * sg + ca * cg, ca * sB * sg - sa * cg],
        [-sB,              sa * cB,                ca * cB]
    ]).T

    dxyz = R_gl @ np.array([X_dot, Y_dot, Z_dot])
    daBg = R_gl @ np.array([a_dot, B_dot, g_dot])

    vtL = saturation(dxyz[0] - b / 2 * daBg[2], velocity_limit)
    vtR = saturation(dxyz[0] + b / 2 * daBg[2], velocity_limit)

    fill_percent = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2) / fill_distance

    Fb, Mb = blade_terrain_interaction(hp, kb, gamma_g, mu_ss,
                                        a_s, beta0, a_b, B1, H, fill_percent)

    FtL, FtR = F_track[0], F_track[1]
    rl = mu_l * m * grav / 2
    RlL = G(FtL, rl, vtL)
    RlR = G(FtR, rl, vtR)
    Rl = np.array([RlL, RlR])

    fy = mu_t * m * grav / l
    Fy = -2 * np.sign(dxyz[1]) * fy * abs(x_ICR)

    mr = 2 * fy * ((l ** 2) / 4 - x_ICR ** 2)
    M  = ((FtR + RlR) - (FtL + RlL)) * b / 2
    Mr = G(M, mr, daBg[2])

    return Rl, Fy, Mr, Fb, Mb

def get_x_icr(q, q_dot, l):
    a, B, g = q[3], q[4], q[5]
    X_dot, Y_dot, Z_dot = q_dot[0], q_dot[1], q_dot[2]
    a_dot, B_dot, g_dot = q_dot[3], q_dot[4], q_dot[5]

    sa, ca = np.sin(a), np.cos(a)
    sB, cB = np.sin(B), np.cos(B)
    sg, cg = np.sin(g), np.cos(g)

    R_gl = np.array([
        [cB * cg,          sa * sB * cg - ca * sg, ca * sB * cg + sa * sg],
        [cB * sg,          sa * sB * sg + ca * cg, ca * sB * sg - sa * cg],
        [-sB,              sa * cB,                ca * cB]
    ]).T

    dxyz = R_gl @ np.array([X_dot, Y_dot, Z_dot])
    daBg = R_gl @ np.array([a_dot, B_dot, g_dot])

    if abs(daBg[2]) < 0.001:
        return 0.0
    return float(np.clip(dxyz[1] / daBg[2], -l / 2, l / 2))

def v_to_q_dot(q, x_ICR, v):
    a, B, g = q[3], q[4], q[5]
    sa, ca = np.sin(a), np.cos(a)
    sB, cB = np.sin(B), np.cos(B)
    sg, cg = np.sin(g), np.cos(g)

    if abs(x_ICR) < 0.001:
        x_ICR = np.finfo(float).max

    R_lg_x = np.array([cB * cg,  cB * sg, -sB])
    R_lg_y = np.array([sa * sB * cg - ca * sg,
                        sa * sB * sg + ca * cg,
                        sa * cB])
    R_lg_z = np.array([ca * sB * cg + sa * sg,
                        ca * sB * sg - sa * cg,
                        ca * cB])

    S = np.zeros((6, 2))
    S[0:3, 0] = R_lg_x
    S[3:6, 0] = 0.0
    S[0:3, 1] = R_lg_y
    S[3:6, 1] = R_lg_z * (-1.0 / x_ICR)

    return S @ v

def vehicle_dynamics(F_track, Rl, Fy, Mr, Fb, Mb,
                     q, q_dot, x_ICR, x_ICR_dot, v, new_bld_ang, vd_params):
    a, B, g = q[3], q[4], q[5]
    Ad, Bd, Gd = q_dot[3], q_dot[4], q_dot[5]   # note: original MATLAB has a typo (Ad=Bd)

    m, h, b, l, r, grav = vd_params

    if x_ICR == 0:
        x_ICR = np.finfo(float).max

    sa, ca = np.sin(a), np.cos(a)
    sB, cB = np.sin(B), np.cos(B)
    sg, cg = np.sin(g), np.cos(g)

    ab, Bb, gb = new_bld_ang
    cab, sab = np.cos(ab), np.sin(ab)
    cBb, sBb = np.cos(Bb), np.sin(Bb)
    cgb, sgb = np.cos(gb), np.sin(gb)

    R_blade = np.array([
        [cBb * cgb, sab * sBb * cgb - cab * sgb, cab * sBb * cgb + sab * sgb],
        [cBb * sgb, sab * sBb * sgb + cab * cgb, cab * sBb * sgb - sab * cgb],
        [-sBb,      sab * cBb,                   cab * cBb]
    ])

    Ix = 1 / 12 * m * (b ** 2 + h ** 2)
    Iy = 1 / 12 * m * (h ** 2 + l ** 2)
    Iz = 1 / 12 * m * (b ** 2 + l ** 2)

    R_lg = np.array([
        [cB * cg,          sa * sB * cg - ca * sg, ca * sB * cg + sa * sg],
        [cB * sg,          sa * sB * sg + ca * cg, ca * sB * sg - sa * cg],
        [-sB,              sa * cB,                ca * cB]
    ])

    R_lg_x = R_lg[:, 0]
    R_lg_y = R_lg[:, 1]
    R_lg_z = R_lg[:, 2]

    S = np.zeros((6, 2))
    S[0:3, 0] = R_lg_x
    S[3:6, 0] = 0.0
    S[0:3, 1] = R_lg_y
    S[3:6, 1] = R_lg_z * (-1.0 / x_ICR)

    B_mat = np.zeros((6, 2))
    B_mat[0:3, 0] = R_lg_x
    B_mat[0:3, 1] = R_lg_x
    B_mat[5, 0] = -ca * cB * b / 2
    B_mat[5, 1] =  ca * cB * b / 2

    elim = np.diag([1, 1, 1, 0, 0, 1])

    R6 = np.zeros((6, 6))
    R6[0:3, 0:3] = R_blade
    R6[3:6, 3:6] = R_blade

    Ct_vec = np.array([Rl[0] + Rl[1], Fy, 0.0, 0.0, 0.0,
                        Mr + (Rl[1] - Rl[0]) * b / 2])
    blade_vec = np.array([Fb, 0.0, 0.0, 0.0, 0.0, Mb])
    Cb_vec = elim @ R6 @ blade_vec

    R6_lg = np.zeros((6, 6))
    R6_lg[0:3, 0:3] = R_lg
    R6_lg[3:6, 3:6] = R_lg
    C = R6_lg @ (Ct_vec + Cb_vec)

    M_mat = np.diag([m, m, m, Ix, Iy, Iz])
    P = np.array([0.0, 0.0, m * grav, 0.0, 0.0, 0.0])

    # Time-derivative of S (Sd)
    S_11 = -sB * cg * Bd - cB * sg * Gd
    S_21 = (ca * sB * cg + sa * sg) * Ad + sa * cB * cg * Bd - (sa * sB * sg + ca * cg) * Gd
    S_31 = (ca * sg - sa * sB * cg) * Ad + ca * cB * cg * Bd + (sa * cg - ca * sB * sg) * Gd
    S_12 = -sB * sg * Bd + cB * cg * Gd
    S_22 = (ca * sB * sg - sa * cg) * Ad + sa * cB * sg * Bd + (sa * sB * cg - ca * sg) * Gd
    S_32 = -(sa * sB * sg + ca * cg) * Ad + ca * cB * sg * Bd + (ca * sB * cg + sa * sg) * Gd
    S_42 =  cB * x_ICR ** (-1) * Bd - sB * x_ICR ** (-2) * x_ICR_dot
    S_52 = -ca * cB * x_ICR ** (-1) * Ad + sa * sB * x_ICR ** (-1) * Bd + sa * cB * x_ICR ** (-2) * x_ICR_dot
    S_62 =  sa * cB * x_ICR ** (-1) * Ad + ca * sB * x_ICR ** (-1) * Bd + ca * cB * x_ICR ** (-2) * x_ICR_dot

    Sd = np.array([
        [S_11, S_12],
        [S_21, S_22],
        [S_31, S_32],
        [0.0,  S_42],
        [0.0,  S_52],
        [0.0,  S_62]
    ])

    Bt = S.T @ B_mat
    Ct2 = S.T @ C
    Pt = S.T @ P
    Mt = S.T @ M_mat @ S
    Et = S.T @ M_mat @ Sd

    v_dot = np.linalg.solve(Mt, Bt @ F_track + Ct2 - Et @ v - Pt)
    return v_dot

def vel_limiter(v, v_limit):
    v_temp = np.sign(v) * np.minimum(np.abs(v), v_limit)
    v_temp[0] = max(v_temp[0], 0.0)   # no reversing
    return v_temp

def hydraulics(bld_ang, bld_ang_vel, gain):
    return bld_ang + gain * bld_ang_vel

def controller_errors(bld_ang, desired_depth, des_ang, L):
    roll, pitch, yaw = bld_ang
    desired_roll, des_pitch_mult, desired_yaw = des_ang
    desired_pitch = des_pitch_mult * np.arcsin(np.clip(desired_depth / L, -1, 1))
    errors = np.array([roll - desired_roll,
                        pitch - desired_pitch,
                        yaw - desired_yaw])
    plot_out = np.array([errors[0], np.sin(errors[1]) * L, errors[2]])
    return errors, plot_out

# ──────────────────────────────────────────────────────────────────────────────
# Parameters  (from parameters.m)
# ──────────────────────────────────────────────────────────────────────────────
h  = 2.762
l  = 2.349
w  = 0.7112
b  = 1.75
r  = 0.4
B1 = 2.921
H  = 0.955
L  = 1.2
m  = 10156.0

mu_l      = 0.1
mu_t      = 0.9
mu_ss     = 0.5
kb        = 0.734e6
beta0_deg = 38.0
c_soil    = 13000.0

grav             = 9.81
stop_distance    = 0.3
gain             = 1 / 40
velocity_limit   = 2.222
fill_distance    = 8.0
gamma_g          = 1640 * 9.81
dt               = 0.001
stop_time        = 2.0

KpP = -3.0   # pitch proportional gain (from parameters.m)
# ── UNKNOWN from .slx ──────────────────────────────────────────────────────
# Roll and yaw PID gains are not in any .m file.
# Using the same proportional-only gain as pitch as a placeholder.
KpR = KpP
KpY = KpP
# ───────────────────────────────────────────────────────────────────────────
turn_vel_limit = 2 * velocity_limit / b
beta0          = np.radians(beta0_deg)

# ──────────────────────────────────────────────────────────────────────────────
# Initial conditions  (from main.m)
# ──────────────────────────────────────────────────────────────────────────────
desired_depth = -0.03
desired_abg   = np.array([-0.005, 1.0, -0.005])
surface_abg   = np.array([ 0.005, 0.0,  0.005])

bld_ang = np.array([0.0, 0.0, 0.0])
F_track = np.array([60000.0, 60000.0])

q     = np.array([0.0, 0.0, 0.0,
                   surface_abg[0], surface_abg[1], surface_abg[2]])
q_dot = np.zeros(6)
x_ICR = 0.0
v     = np.zeros(2)

bt_params = (B1, H, L, b, l, r, m, grav, velocity_limit, fill_distance,
             mu_t, mu_l, mu_ss, kb, gamma_g, beta0,
             surface_abg[0], surface_abg[1], surface_abg[2])

vd_params = (m, h, b, l, r, grav)

v_limit = np.array([velocity_limit, turn_vel_limit])

# ──────────────────────────────────────────────────────────────────────────────
# Storage for logging  (mirrors errors_and_plots.m column ordering)
# col:  time, X, Y, Z, roll, pitch, yaw, roll_err, depth_err, yaw_err
# ──────────────────────────────────────────────────────────────────────────────
log = []

# ──────────────────────────────────────────────────────────────────────────────
# Main forward-Euler loop
# ──────────────────────────────────────────────────────────────────────────────
t = 0.0
n_steps = int(stop_time / dt)

for step in range(n_steps):

    # ── stop condition (mirrors simulation_stopping_and_state_loading.m) ──
    if abs(q[0]) + abs(q[1]) > stop_distance:
        print(f"Stop condition triggered at t={t:.4f} s")
        break

    # ── blade angle wrapping to [-pi, pi] ────────────────────────────────
    for i in range(3):
        if bld_ang[i] < -np.pi or bld_ang[i] > np.pi:
            bld_ang[i] = (bld_ang[i] + np.pi) % (2 * np.pi) - np.pi

    # ── controller: compute errors ────────────────────────────────────────
    errors, plot_errors = controller_errors(bld_ang, desired_depth, desired_abg, L)

    # ── controller: blade angular velocity command (proportional only) ───
    # NOTE: The actual PID structure lives in the .slx file and is unknown.
    bld_ang_vel = np.array([KpR * errors[0],
                             KpP * errors[1],
                             KpY * errors[2]])

    # ── hydraulics: update blade angles ──────────────────────────────────
    bld_ang = hydraulics(bld_ang, bld_ang_vel, gain)

    # ── instantaneous centre of rotation ─────────────────────────────────
    x_ICR_prev = x_ICR
    x_ICR = get_x_icr(q, q_dot, l)
    x_ICR_dot = (x_ICR - x_ICR_prev) / dt

    # ── blade & track forces ──────────────────────────────────────────────
    Rl, Fy, Mr, Fb, Mb = blade_and_track(
        F_track, q, q_dot, x_ICR, bld_ang, bt_params)

    # ── vehicle dynamics → v_dot ──────────────────────────────────────────
    v_dot = vehicle_dynamics(
        F_track, Rl, Fy, Mr, Fb, Mb,
        q, q_dot, x_ICR, x_ICR_dot, v, bld_ang, vd_params)

    # ── forward-Euler integration: v ─────────────────────────────────────
    v = v + dt * v_dot
    v = vel_limiter(v, v_limit)

    # ── kinematics: q_dot from v ──────────────────────────────────────────
    q_dot = v_to_q_dot(q, x_ICR, v)

    # ── forward-Euler integration: q ─────────────────────────────────────
    q = q + dt * q_dot

    # ── log ───────────────────────────────────────────────────────────────
    log.append([t, q[0], q[1], q[2],
                 q[3], q[4], q[5],
                 plot_errors[0], plot_errors[1], plot_errors[2]])

    t += dt

# ──────────────────────────────────────────────────────────────────────────────
# Post-processing  (mirrors errors_and_plots.m)
# ──────────────────────────────────────────────────────────────────────────────
data = np.array(log)
time        = data[:, 0]
body_x      = data[:, 1]
body_y      = data[:, 2]
body_roll   = data[:, 4]
body_yaw    = data[:, 6]
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

# ── plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(10, 10))
fig.tight_layout(pad=3.0)

axes[0, 0].plot(body_x, body_y, '-k', linewidth=2)
axes[0, 0].set_xlabel('Body X (m)'); axes[0, 0].set_ylabel('Body Y (m)')

axes[0, 1].plot(time, body_roll, '-k', linewidth=2)
axes[0, 1].set_xlabel('Time (s)'); axes[0, 1].set_ylabel('Body Roll (rad)')

axes[1, 0].plot(time, body_yaw, '-k', linewidth=2)
axes[1, 0].set_xlabel('Time (s)'); axes[1, 0].set_ylabel('Body Yaw (rad)')

axes[1, 1].plot(time, roll_error, '-k', linewidth=2)
axes[1, 1].set_xlabel('Time (s)'); axes[1, 1].set_ylabel('Roll Error (rad)')

axes[2, 0].plot(time, depth_error, '-k', linewidth=2)
axes[2, 0].set_xlabel('Time (s)'); axes[2, 0].set_ylabel('Depth Error (m)')

axes[2, 1].plot(time, yaw_error, '-k', linewidth=2)
axes[2, 1].set_xlabel('Time (s)'); axes[2, 1].set_ylabel('Yaw Error (rad)')

plt.savefig('simulation_results.png', dpi=150)
print("Plot saved to simulation_results.png")
