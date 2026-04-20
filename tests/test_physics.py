import numpy as np
from simulation_3d import (
    blade_terrain_interaction,
    get_x_icr,
    v_to_q_dot,
    vel_limiter,
    hydraulics,
    controller_errors
)

# ------------------ blade interaction ------------------
def test_blade_zero_fill():
    FT, Mb = blade_terrain_interaction(
        hp=1, kb=1, gamma_g=1, mu_ss=1,
        a_s=0, beta0=1, a_b=0,
        B1=1, H=1, fill_percent=0
    )
    assert FT <= 0

def test_blade_finite_outputs():
    FT, Mb = blade_terrain_interaction(
        hp=1, kb=1, gamma_g=1, mu_ss=1,
        a_s=0, beta0=1, a_b=0,
        B1=1, H=1, fill_percent=1
    )
    assert np.isfinite(FT)
    assert np.isfinite(Mb)

# ------------------ ICR ------------------
def test_icr_zero(zero_state, zero_velocity):
    assert get_x_icr(zero_state, zero_velocity, 2) == 0.0

def test_icr_clipping(sample_state):
    q_dot = np.array([0, 10, 0, 0, 0, 1])
    val = get_x_icr(sample_state, q_dot, 2)
    assert abs(val) <= 1

# ------------------ v_to_q_dot ------------------
def test_v_to_q_dot_zero(zero_state):
    v = np.zeros(2)
    out = v_to_q_dot(zero_state, 1, v)
    assert np.allclose(out, 0)

def test_v_to_q_dot_linear(zero_state):
    v = np.array([1.0, 2.0])
    out1 = v_to_q_dot(zero_state, 1, v)
    out2 = v_to_q_dot(zero_state, 1, 2*v)
    assert np.allclose(out2, 2*out1)

# ------------------ velocity limiter ------------------
def test_vel_limiter():
    v = np.array([2.0, -3.0])
    limit = np.array([1.0, 1.0])
    out = vel_limiter(v, limit)
    assert np.allclose(out, [1.0, -1.0])

def test_vel_limiter_no_reverse():
    v = np.array([-2.0, 1.0])
    limit = np.array([3.0, 3.0])
    out = vel_limiter(v, limit)
    assert out[0] == 0.0

# ------------------ hydraulics ------------------
def test_hydraulics():
    ang = np.array([1,2,3])
    vel = np.array([0.1,0.2,0.3])
    gain = 2
    assert np.allclose(hydraulics(ang, vel, gain), ang + 2*vel)

# ------------------ controller ------------------
def test_controller_zero():
    ang = np.zeros(3)
    des = np.zeros(3)
    err, _ = controller_errors(ang, 0, des, 1)
    assert np.allclose(err, 0)

def test_controller_finite():
    ang = np.array([0,0.1,0])
    des = np.array([0,1,0])
    err, plot = controller_errors(ang, 0.5, des, 1)
    assert np.all(np.isfinite(plot))