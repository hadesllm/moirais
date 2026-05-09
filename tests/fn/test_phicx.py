"""Tests for phi complexity."""
import numpy as np
from moirais.fn.phicx import phi_complexity, phicx


def test_diagonal():
    C = np.diag([1.0, 2.0, 3.0])
    r = phi_complexity(C)
    assert abs(r.estimate) < 1e-10


def test_correlated():
    C = np.array([[1.0, 0.8, 0.8], [0.8, 1.0, 0.8], [0.8, 0.8, 1.0]])
    r = phi_complexity(C)
    assert r.estimate >= 0


def test_alias():
    assert phicx is phi_complexity
