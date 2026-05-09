"""Tests for moirais.fn.nmnlt — NOMINATE scaling."""
import numpy as np
from moirais.fn.nmnlt import nmnlt


def test_nmnlt_basic():
    rng = np.random.default_rng(42)
    V = rng.integers(0, 2, (8, 5)).astype(float)
    r = nmnlt(V, n_dims=1, max_iter=20)
    assert r.value["ideal_points"].shape == (8, 1)


def test_nmnlt_2d():
    rng = np.random.default_rng(42)
    V = rng.integers(0, 2, (6, 4)).astype(float)
    r = nmnlt(V, n_dims=2, max_iter=10)
    assert r.value["ideal_points"].shape == (6, 2)


def test_nmnlt_loglik():
    V = np.array([[1, 1, 0], [0, 0, 1]], dtype=float)
    r = nmnlt(V, max_iter=5)
    assert r.extra["log_likelihood"] < 0
