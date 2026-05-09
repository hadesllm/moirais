"""Tests for moirais.fn.clfrt — Classification rate."""
import numpy as np
from moirais.fn.clfrt import clfrt


def test_clfrt_perfect():
    V = np.array([[1, 0], [0, 1]])
    X = np.array([[0.0], [2.0]])
    A = np.array([[[0.0], [2.0]], [[2.0], [0.0]]])
    r = clfrt(V, X, A)
    assert r.value == 1.0


def test_clfrt_random():
    rng = np.random.default_rng(42)
    V = rng.integers(0, 2, (5, 3)).astype(float)
    X = rng.standard_normal((5, 1))
    A = rng.standard_normal((3, 2, 1))
    r = clfrt(V, X, A)
    assert 0.0 <= r.value <= 1.0


def test_clfrt_extra():
    V = np.array([[1.0]])
    X = np.array([[0.0]])
    A = np.array([[[0.0], [1.0]]])
    r = clfrt(V, X, A)
    assert r.extra["n_total"] == 1
