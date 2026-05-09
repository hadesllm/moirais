"""Tests for simple kriging."""
import numpy as np
from moirais.fn.sgsk import sgsk


def test_sgsk_smoke():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 10, (20, 2))
    Z = np.sin(coords[:, 0]) + rng.normal(0, 0.1, 20)
    r = sgsk(Z, coords, np.array([5.0, 5.0]))
    assert r.name == "simple_kriging"
    assert "predictions" in r.extra
    assert "variances" in r.extra
    assert len(r.extra["predictions"]) == 1


def test_sgsk_variance_positive():
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    Z = np.array([1.0, 2.0, 3.0, 4.0])
    r = sgsk(Z, coords, np.array([0.5, 0.5]))
    assert r.extra["variances"][0] >= 0
