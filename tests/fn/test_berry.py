"""Test berry."""
import numpy as np
from moirais.fn.berry import berry_phase


def test_berry_basic():
    r = berry_phase(n_points=200)
    assert r.value is not None
    assert r.name == "berry_phase"


def test_berry_analytic():
    r = berry_phase(n_points=500)
    analytic = -np.pi * (1 - np.cos(np.pi / 3))
    assert abs(r.value - analytic) < 0.1
