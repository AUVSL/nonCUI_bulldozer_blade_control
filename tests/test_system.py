import numpy as np
from simulation_3d import (
    get_x_icr,
    v_to_q_dot
)

def test_pipeline_no_nan(zero_state):
    q = zero_state
    q_dot = np.zeros(6)

    x_icr = get_x_icr(q, q_dot, 2)

    v = np.array([1.0, 0.1])
    q_dot = v_to_q_dot(q, x_icr, v)

    assert np.all(np.isfinite(q_dot))

def test_symmetry_tracks():
    # symmetric velocity input should not create lateral bias
    q = np.zeros(6)
    v = np.array([1.0, 0.0])
    out = v_to_q_dot(q, 1e6, v)
    assert np.isclose(out[1], 0.0)