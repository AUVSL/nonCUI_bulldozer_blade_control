import numpy as np
import pytest

@pytest.fixture
def zero_state():
    return np.zeros(6)

@pytest.fixture
def zero_velocity():
    return np.zeros(6)

@pytest.fixture
def sample_state():
    return np.array([1.0, 1.0, 1.0, 0.1, 0.2, 0.3])

@pytest.fixture
def sample_velocity():
    return np.array([0.5, -0.2, 0.1, 0.01, 0.02, 0.03])

@pytest.fixture
def bt_params():
    # Reasonable physical values
    return (
        2.0,   # B1
        1.0,   # H
        3.0,   # L
        1.0,   # b
        2.0,   # l
        0.5,   # r
        1000.0,# m
        9.81,  # grav
        5.0,   # velocity_limit
        2.0,   # fill_distance
        0.8,   # mu_t
        0.9,   # mu_l
        0.5,   # mu_ss
        100.0, # kb
        1800.0,# gamma_g
        np.pi/6,# beta0
        0.0,   # a_s
        0.0,   # B_s
        0.0    # g_s
    )

@pytest.fixture
def vd_params():
    return (
        1000.0, # m
        1.5,    # h
        2.0,    # b
        3.0,    # l
        0.5,    # r
        9.81    # gravity
    )

@pytest.fixture
def track_forces():
    return np.array([5.0, 5.0])

@pytest.fixture
def blade_angles():
    return np.array([0.0, 0.1, 0.0])

@pytest.fixture
def control_input():
    return np.array([1.0, 0.2])