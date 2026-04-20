import numpy as np
from simulation_3d import saturation, G, yc, Hx, Dx

# ------------------ saturation ------------------
def test_saturation_bounds():
    assert saturation(10, 5) == 5
    assert saturation(-10, 5) == -5
    assert saturation(3, 5) == 3

def test_saturation_symmetry():
    x, L = 4.2, 3.0
    assert saturation(x, L) == -saturation(-x, L)

# ------------------ G ------------------
def test_G_sliding():
    assert G(10, 5, 1) == -5
    assert G(10, 5, -1) == 5

def test_G_static():
    assert G(3, 5, 0) == -3

def test_G_breakaway():
    assert G(10, 5, 0) == -5
    assert G(-10, 5, 0) == 5

# ------------------ yc ------------------
def test_yc_zero():
    assert yc(0, 0, 10) == 0.0

def test_yc_symmetry():
    assert np.isclose(yc(5, 5, 10), 0.0)

def test_yc_scaling():
    y1 = yc(2, 4, 10)
    y2 = yc(2, 4, 20)
    assert np.isclose(y2, 2*y1)

# ------------------ Hx ------------------
def test_Hx_midpoint():
    assert np.isclose(Hx(0, 2, 6, 10), 4)

def test_Hx_endpoints():
    B1 = 10
    assert np.isclose(Hx(-B1/2, 2, 6, B1), 2)
    assert np.isclose(Hx(B1/2, 2, 6, B1), 6)

# ------------------ Dx ------------------
def test_Dx_basic():
    val = Dx(0, np.pi/4, 2, 6, 10)
    assert np.isclose(val, Hx(0, 2, 6, 10))

def test_Dx_monotonic_beta():
    yb = 1
    val1 = Dx(yb, np.pi/6, 2, 6, 10)
    val2 = Dx(yb, np.pi/3, 2, 6, 10)
    assert val1 > val2